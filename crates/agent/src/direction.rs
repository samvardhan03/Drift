//! Direction and recommendation checks on a narration, complementing the
//! numeric grounding check (`crate::grounding`): a number can exist in the
//! trace and still be described backwards ("gold is a stabiliser" when the
//! trace shows it as the biggest drag), or a recommendation can name a
//! holding that the trace never singles out ("reduce Reliance" when the
//! deepest drawdowns are elsewhere). Both are keyword heuristics over
//! sentences, tuned to flag clear contradictions rather than every
//! ambiguous phrasing -- a false alarm costs a Gemini retry.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use compute::trace::EvidenceTrace;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Word stems (matched as the *start* of a word, so "gain" matches "gained"
/// but not "against").
const POSITIVE_WORDS: &[&str] = &[
    "stabilis", "stabiliz", "hedg", "protect", "gain", "positive", "helped", "offset", "buffer", "cushion",
    "outperform", "resilient", "defensive", "supported",
];
const POSITIVE_PHRASES: &[&str] = &["strong performance", "safe haven", "contributed positively"];
// "risk" is deliberately absent: it appears in nearly every sentence about
// a perfectly healthy holding ("contributes 12% of risk").
const NEGATIVE_WORDS: &[&str] = &[
    "drag", "loss", "lost", "hurt", "negative", "worst", "fell", "declin", "weak", "underperform", "drawdown",
    "damag",
];
/// Words that clearly *assert* a direction about the entity. A sentence is
/// only flagged when it uses one of these and none of the opposite tone's
/// broad words above -- "drawdown", "gains", "loss" etc. also appear in
/// perfectly correct sentences ("reduce it to limit drawdown", "locking in
/// gains", "offset the loss") and caused false alarms in live use.
const STRONG_POSITIVE: &[&str] = &["stabilis", "stabiliz", "hedg", "protect", "cushion", "buffer", "outperform", "resilient", "defensive"];
const STRONG_POSITIVE_PHRASES: &[&str] = &["safe haven", "contributed positively", "strong performance"];
const STRONG_NEGATIVE: &[&str] = &["drag", "hurt", "underperform", "damag", "worst", "weak", "laggard"];
/// A negation anywhere in the sentence ("failing to contribute positively")
/// makes the tone ambiguous, so the sentence is not judged.
const NEGATIONS: &[&str] = &["not", "no", "never", "fail", "fails", "failing", "failed", "without", "neither", "nor", "isn", "wasn", "doesn", "didn"];
const REDUCE_WORDS: &[&str] = &["reduc", "trim", "cut", "sell", "exit", "lighten", "offload"];
const REDUCE_PHRASES: &[&str] = &["scale back", "scaling back", "pare back"];

/// One contradiction between a narration and its trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectionalViolation {
    pub entity: String,
    /// `"positive"`/`"negative"` for a direction contradiction;
    /// `"not_a_top_contributor"` for an unsupported recommendation.
    pub trace_direction: String,
    pub sentence: String,
    /// Human-readable description, used both as the retry instruction and
    /// as the surfaced warning.
    pub message: String,
}

#[derive(Debug, Default)]
struct Evidence {
    /// bare lowercase ticker -> original ticker, for every holding seen.
    holdings: BTreeMap<String, String>,
    /// bare lowercase ticker or factor alias key -> directions seen.
    directions: BTreeMap<String, BTreeSet<i8>>,
    /// Factor aliases: alias word -> factor key (a key in `directions`).
    factor_aliases: Vec<(String, String)>,
    /// Holdings the trace singles out as top drags / deepest drawdowns /
    /// biggest vol contributors (bare lowercase tickers).
    recommendable: BTreeSet<String>,
    /// Holdings that are top-3 by absolute P&L or return, or top-2 by vol
    /// contribution: the only holdings a narration may bring up from an
    /// earlier turn (see `check_contamination`).
    top_contributors: BTreeSet<String>,
    /// Every holding of the portfolio the trace was run on.
    portfolio_tickers: BTreeSet<String>,
    /// The three biggest gainers by P&L / return, so the narration is also
    /// told which holdings *helped*.
    top_gainers: BTreeSet<String>,
    /// Every factor's sign (not just the distinctive ones that get
    /// sentence-checked), for the narration's direction constraints.
    factor_signs: BTreeMap<String, i8>,
    has_holding_data: bool,
}

fn bare(ticker: &str) -> String {
    let up = ticker.trim().to_uppercase();
    up.strip_suffix(".NS").or_else(|| up.strip_suffix(".BO")).unwrap_or(&up).to_lowercase()
}

fn sign(v: f64) -> Option<i8> {
    if v > 1e-9 {
        Some(1)
    } else if v < -1e-9 {
        Some(-1)
    } else {
        None
    }
}

/// Largest `n` values by magnitude, either sign.
fn top_abs(items: &[(String, f64)], n: usize) -> Vec<String> {
    let mut v: Vec<&(String, f64)> = items.iter().collect();
    v.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
    v.into_iter().take(n).map(|(t, _)| bare(t)).collect()
}

/// Largest `n` values, only those above zero.
fn top_positive(items: &[(String, f64)], n: usize) -> Vec<String> {
    let mut v: Vec<&(String, f64)> = items.iter().filter(|(_, x)| *x > 0.0).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v.into_iter().take(n).map(|(t, _)| bare(t)).collect()
}

/// Smallest `n` values by `key`, only those below zero.
fn worst_negative(items: &[(String, f64)], n: usize) -> Vec<String> {
    let mut v: Vec<&(String, f64)> = items.iter().filter(|(_, x)| *x < 0.0).collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    v.into_iter().take(n).map(|(t, _)| bare(t)).collect()
}

fn collect(traces: &[EvidenceTrace]) -> Evidence {
    let mut ev = Evidence::default();
    // Generic factors (MARKET, RATES_PROXY) appear in almost every sentence
    // and are described both ways legitimately, so only distinctive ones
    // get direction-checked.
    let factor_words: &[(&str, &[&str])] =
        &[("BRENT", &["brent", "crude"]), ("GOLD_USD", &["gold"]), ("USDINR", &["rupee", "usdinr"])];

    for trace in traces {
        if let Some(rows) = trace.inputs.get("portfolio").and_then(|p| p.get("holdings")).and_then(Value::as_array) {
            for h in rows {
                if let Some(t) = h.get("ticker").and_then(Value::as_str) {
                    ev.portfolio_tickers.insert(bare(t));
                    ev.holdings.entry(bare(t)).or_insert_with(|| t.to_string());
                }
            }
        }
        let Some(result) = trace.outputs.get("result") else { continue };
        let note = |ev: &mut Evidence, ticker: &str, v: f64| {
            let b = bare(ticker);
            ev.holdings.entry(b.clone()).or_insert_with(|| ticker.to_string());
            if let Some(s) = sign(v) {
                ev.directions.entry(b).or_default().insert(s);
            }
            ev.has_holding_data = true;
        };

        // FactorShock: per_holding[{ticker, pnl_inr}]
        if let Some(rows) = result.get("per_holding").and_then(Value::as_array) {
            let pnl: Vec<(String, f64)> = rows
                .iter()
                .filter_map(|r| Some((r.get("ticker")?.as_str()?.to_string(), r.get("pnl_inr")?.as_f64()?)))
                .collect();
            for (t, v) in &pnl {
                note(&mut ev, t, *v);
            }
            ev.recommendable.extend(worst_negative(&pnl, 3));
            ev.top_contributors.extend(top_abs(&pnl, 3));
            ev.top_gainers.extend(top_positive(&pnl, 3));
        }
        // PortfolioPerformance: holding_returns{ticker: {total_return_pct, max_drawdown_pct}}
        if let Some(map) = result.get("holding_returns").and_then(Value::as_object) {
            let mut returns = Vec::new();
            let mut drawdowns = Vec::new();
            for (t, perf) in map {
                if let Some(r) = perf.get("total_return_pct").and_then(Value::as_f64) {
                    note(&mut ev, t, r);
                    returns.push((t.clone(), r));
                }
                if let Some(d) = perf.get("max_drawdown_pct").and_then(Value::as_f64) {
                    drawdowns.push((t.clone(), d));
                }
            }
            ev.recommendable.extend(worst_negative(&returns, 3));
            ev.recommendable.extend(worst_negative(&drawdowns, 3));
            ev.top_contributors.extend(top_abs(&returns, 3));
            ev.top_gainers.extend(top_positive(&returns, 3));
            // Realised vol is this experiment's "vol contribution": the two
            // most volatile holdings count as legitimate reduce targets.
            let vols: Vec<(String, f64)> = map
                .iter()
                .filter_map(|(t, p)| Some((t.clone(), p.get("annualized_vol_pct")?.as_f64()?)))
                .collect();
            ev.recommendable.extend(top_abs(&vols, 2));
        }
        // RiskDecomposition: by_stock[{ticker, contribution}] -- a vol
        // contribution has no good/bad direction, but names the top-2
        // holdings a reduce recommendation may target.
        if let Some(rows) = result.get("by_stock").and_then(Value::as_array) {
            let mut vol: Vec<(String, f64)> = rows
                .iter()
                .filter_map(|r| Some((r.get("ticker")?.as_str()?.to_string(), r.get("contribution")?.as_f64()?)))
                .collect();
            for (t, _) in &vol {
                ev.holdings.entry(bare(t)).or_insert_with(|| t.clone());
                ev.has_holding_data = true;
            }
            vol.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ev.recommendable.extend(vol.iter().take(2).map(|(t, _)| bare(t)));
            ev.top_contributors.extend(vol.iter().take(2).map(|(t, _)| bare(t)));
        }
        // ReverseStress: holding_pnl{ticker: inr}
        if let Some(map) = result.get("holding_pnl").and_then(Value::as_object) {
            let pnl: Vec<(String, f64)> =
                map.iter().filter_map(|(t, v)| Some((t.clone(), v.as_f64()?))).collect();
            for (t, v) in &pnl {
                note(&mut ev, t, *v);
            }
            ev.recommendable.extend(worst_negative(&pnl, 3));
            ev.top_contributors.extend(top_abs(&pnl, 3));
            ev.top_gainers.extend(top_positive(&pnl, 3));
        }
        // FactorShock: factor_attribution_log_inr{factor: inr};
        // ReverseStress: factor_attribution{factor: inr}
        for key in ["factor_attribution_log_inr", "factor_attribution"] {
            let Some(map) = result.get(key).and_then(Value::as_object) else { continue };
            for (factor, v) in map {
                if let Some(sg) = v.as_f64().and_then(sign) {
                    ev.factor_signs.insert(factor.clone(), sg);
                }
                if let (Some(v), Some((_, words))) = (v.as_f64(), factor_words.iter().find(|(f, _)| f == factor)) {
                    if let Some(s) = sign(v) {
                        let key = format!("factor:{factor}");
                        ev.directions.entry(key.clone()).or_default().insert(s);
                        for w in *words {
                            ev.factor_aliases.push((w.to_string(), key.clone()));
                        }
                    }
                }
            }
        }
    }
    ev
}

/// "Mandatory direction constraints" for the narration context: one line per
/// holding the trace singles out (top movers, worst contributors) and per
/// factor, stating which way the trace says it moved the portfolio.
pub fn direction_lines(traces: &[EvidenceTrace]) -> Vec<String> {
    let ev = collect(traces);
    let mut out = Vec::new();
    for (base, original) in &ev.holdings {
        if !(ev.top_contributors.contains(base) || ev.recommendable.contains(base) || ev.top_gainers.contains(base)) {
            continue;
        }
        let Some(dirs) = ev.directions.get(base) else { continue };
        if dirs.len() != 1 {
            continue;
        }
        out.push(if dirs.contains(&-1) {
            format!("{original}: NEGATIVE contributor \u{2014} it HURT the portfolio (describe it as a drag or a loss; never a hedge, stabiliser, buffer or offset)")
        } else {
            format!("{original}: POSITIVE contributor \u{2014} it HELPED the portfolio (describe it as a gain; never a drag or a loss)")
        });
        if out.len() >= 12 {
            break;
        }
    }
    for (factor, sg) in &ev.factor_signs {
        out.push(if *sg < 0 {
            format!("{factor} factor: NEGATIVE attribution \u{2014} it HURT the portfolio")
        } else {
            format!("{factor} factor: POSITIVE attribution \u{2014} it HELPED the portfolio")
        });
    }
    out
}

/// The holdings a "reduce" recommendation may name (the trace's worst
/// contributors), or `None` when the trace has no per-holding data to judge
/// by (so no constraint should be stated).
pub fn reduce_candidates(traces: &[EvidenceTrace]) -> Option<Vec<String>> {
    let ev = collect(traces);
    if !ev.has_holding_data {
        return None;
    }
    Some(ev.recommendable.iter().map(|b| ev.holdings.get(b).cloned().unwrap_or_else(|| b.to_uppercase())).collect())
}

/// Splits on sentence-ending punctuation followed by whitespace (so
/// "RELIANCE.NS" and "2.5%" are not split).
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        let ends = matches!(c, '.' | '!' | '?') && chars.get(i + 1).is_none_or(|n| n.is_whitespace());
        if ends || c == '\n' {
            if !cur.trim().is_empty() {
                out.push(cur.trim().to_string());
            }
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn tokens(lower_sentence: &str) -> Vec<&str> {
    lower_sentence.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()).collect()
}

fn has_word(tokens: &[&str], lower_sentence: &str, stems: &[&str], phrases: &[&str]) -> bool {
    tokens.iter().any(|t| stems.iter().any(|s| t.starts_with(s))) || phrases.iter().any(|p| lower_sentence.contains(p))
}

fn mentions_base(sentence: &str, base: &str) -> bool {
    let pat = format!(r"(?i)(?:^|[^a-z0-9&-]){}(?:$|[^a-z0-9&-])", regex::escape(base));
    Regex::new(&pat).map(|re| re.is_match(sentence)).unwrap_or(false)
}

/// Direction contradictions: a holding or distinctive factor the trace
/// shows moving one way, described in a sentence using only words of the
/// opposite tone. A sentence with both tones ("gold offset the loss"), or
/// that mentions entities the trace shows moving opposite ways, is left
/// alone as ambiguous.
pub fn check_directions(narration: &str, traces: &[EvidenceTrace]) -> Vec<DirectionalViolation> {
    let ev = collect(traces);
    let mut out: Vec<DirectionalViolation> = Vec::new();
    for sentence in sentences(narration) {
        let lower = sentence.to_lowercase();
        let toks = tokens(&lower);
        let mut matched: BTreeSet<&str> = BTreeSet::new(); // direction keys
        let mut labels: HashMap<&str, String> = HashMap::new();
        for (base, original) in &ev.holdings {
            if !ev.directions.contains_key(base) {
                continue;
            }
            // Full symbol, or a distinctive stem of it ("gold" for GOLDCASE):
            // at least 4 letters and at least half the symbol.
            let by_stem = toks.iter().any(|t| t.len() >= 4 && t.len() * 2 >= base.len() && base.starts_with(t));
            if mentions_base(&sentence, base) || by_stem {
                matched.insert(base.as_str());
                labels.insert(base.as_str(), original.clone());
            }
        }
        for (word, key) in &ev.factor_aliases {
            if toks.contains(&word.as_str()) {
                matched.insert(key.as_str());
                labels.insert(key.as_str(), word.clone());
            }
        }
        if matched.is_empty() {
            continue;
        }
        let dirs: BTreeSet<i8> = matched.iter().flat_map(|k| ev.directions[*k].iter().copied()).collect();
        if dirs.len() != 1 {
            continue; // opposite movers in one sentence, or a holding both up and down across tools
        }
        let trace_dir = *dirs.iter().next().unwrap();
        if toks.iter().any(|t| NEGATIONS.contains(t)) {
            continue;
        }
        let pos = has_word(&toks, &lower, POSITIVE_WORDS, POSITIVE_PHRASES);
        let neg = has_word(&toks, &lower, NEGATIVE_WORDS, &[]);
        let strong_pos = has_word(&toks, &lower, STRONG_POSITIVE, STRONG_POSITIVE_PHRASES);
        let strong_neg = has_word(&toks, &lower, STRONG_NEGATIVE, &[]);
        let (bad, trace_word, narr_word) = match trace_dir {
            -1 if strong_pos && !neg => (true, "negative", "positively"),
            1 if strong_neg && !pos => (true, "positive", "negatively"),
            _ => (false, "", ""),
        };
        if bad {
            let entity = matched.iter().map(|k| labels[k].clone()).collect::<Vec<_>>().join(", ");
            out.push(DirectionalViolation {
                message: format!(
                    "{entity} is described {narr_word} but the trace shows it as a {trace_word} contributor. Sentence: '{sentence}'"
                ),
                entity,
                trace_direction: trace_word.to_string(),
                sentence,
            });
        }
    }
    out
}

/// Recommendations to cut a holding the trace does not single out (see
/// `NARRATE_SYSTEM_PROMPT`'s RECOMMENDATION RULE). Only runs when the
/// trace has per-holding data to judge against.
pub fn check_recommendations(narration: &str, traces: &[EvidenceTrace]) -> Vec<DirectionalViolation> {
    let ev = collect(traces);
    if !ev.has_holding_data {
        return Vec::new();
    }
    let mut out = Vec::new();
    for sentence in sentences(narration) {
        let lower = sentence.to_lowercase();
        let toks = tokens(&lower);
        if !has_word(&toks, &lower, REDUCE_WORDS, REDUCE_PHRASES) {
            continue;
        }
        for (base, original) in &ev.holdings {
            if mentions_base(&sentence, base) && !ev.recommendable.contains(base) {
                let allowed: Vec<&str> = ev.recommendable.iter().map(String::as_str).collect();
                out.push(DirectionalViolation {
                    message: format!(
                        "The narration recommends reducing {original}, which is not among the trace's worst contributors ({}). Sentence: '{sentence}'",
                        if allowed.is_empty() { "none".to_string() } else { allowed.join(", ") }
                    ),
                    entity: original.clone(),
                    trace_direction: "not_a_top_contributor".to_string(),
                    sentence: sentence.clone(),
                });
            }
        }
    }
    out
}

/// Whether `text` refers to the holding with bare symbol `base`: the whole
/// symbol as a word, or a distinctive stem of it (a word of at least 4
/// letters covering at least half the symbol -- "Zydus" for ZYDUSWELL).
fn mentions_entity(text: &str, base: &str) -> bool {
    if mentions_base(text, base) {
        return true;
    }
    let lower = text.to_lowercase();
    tokens(&lower).iter().any(|t| t.len() >= 4 && t.len() * 2 >= base.len() && base.starts_with(t))
}

/// Entity contamination: a holding discussed in an *earlier* turn that this
/// narration brings up although the current trace does not single it out
/// (not a top-3 mover, not a top-2 vol contributor, not the stock the user
/// asked about now). Backstop for `NARRATE_SYSTEM_PROMPT`'s Rule 0.
pub fn check_contamination(
    narration: &str,
    history: &[crate::conversation::ConversationTurn],
    traces: &[EvidenceTrace],
    focus_holding: Option<&str>,
) -> Vec<DirectionalViolation> {
    if history.is_empty() {
        return Vec::new();
    }
    let ev = collect(traces);
    let history_text: String = history.iter().map(|t| t.content.as_str()).collect::<Vec<_>>().join("\n");
    let focus = focus_holding.map(bare);
    let mut out = Vec::new();
    for base in &ev.portfolio_tickers {
        let allowed = ev.top_contributors.contains(base) || focus.as_deref() == Some(base.as_str());
        if allowed || !mentions_entity(&history_text, base) {
            continue;
        }
        if let Some(sentence) = sentences(narration).into_iter().find(|s| mentions_entity(s, base)) {
            let original = ev.holdings.get(base).cloned().unwrap_or_else(|| base.to_uppercase());
            out.push(DirectionalViolation {
                message: format!(
                    "{original} was discussed in an earlier turn but is not a top contributor in the current result, so it must not be mentioned. Remove every reference to it. Sentence: '{sentence}'"
                ),
                entity: original,
                trace_direction: "prior_turn_entity".to_string(),
                sentence,
            });
        }
    }
    out
}

/// Every narration check except the numeric one.
pub fn check_all(
    narration: &str,
    traces: &[EvidenceTrace],
    history: &[crate::conversation::ConversationTurn],
    focus_holding: Option<&str>,
) -> Vec<DirectionalViolation> {
    let mut v = check_directions(narration, traces);
    v.extend(check_recommendations(narration, traces));
    v.extend(check_contamination(narration, history, traces, focus_holding));
    v
}

/// The retry instruction for `violations`.
pub fn correction_instructions(violations: &[DirectionalViolation]) -> String {
    let lines: Vec<String> = violations.iter().map(|v| format!("- {}", v.message)).collect();
    format!(
        "CRITICAL CORRECTION REQUIRED:\nThe following claims contradict the model:\n{}\n\
         The model output is the ground truth. Correct all directional descriptions to match the trace \
         exactly, never mention holdings from earlier turns that are not in this result, and only recommend reducing holdings the trace shows as top losers, deepest drawdowns \
         or largest vol contributors.",
        lines.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn trace_with(result: Value) -> EvidenceTrace {
        // Only `outputs.result` matters to these checks.
        let mut t: EvidenceTrace = serde_json::from_value(json!({
            "experiment": "X", "inputs": {}, "data_window": {"frequency": "Daily", "window_periods": 1, "start": "2025-01-01", "end": "2025-01-02"},
            "data_quality": {"date_range_start": "2025-01-01", "date_range_end": "2025-01-02", "trading_days": 2, "per_series": []},
            "model_params": {"frequency": "Daily", "window_periods": 1, "factor_names": [], "shrinkage_intensity": 0.0, "annualization_factor": 252.0, "regime_state": null, "regime_fallback_warnings": [], "cap_source": null},
            "outputs": {}, "invariants": [], "engine_version": "t"
        }))
        .unwrap();
        t.outputs = json!({ "result": result });
        t
    }

    fn shock_trace() -> EvidenceTrace {
        trace_with(json!({
            "per_holding": [
                {"ticker": "GOLDCASE.NS", "pnl_inr": -9000.0},
                {"ticker": "KIOCL.NS", "pnl_inr": -5000.0},
                {"ticker": "BLS.NS", "pnl_inr": -4000.0},
                {"ticker": "DYCL.BO", "pnl_inr": -3000.0},
                {"ticker": "RELIANCE.NS", "pnl_inr": 2000.0},
            ],
            "factor_attribution_log_inr": {"GOLD_USD": -8000.0, "MARKET": -3000.0}
        }))
    }

    #[test]
    fn a_negative_contributor_described_as_a_stabiliser_is_flagged() {
        let v = check_directions("Gold stabilises the portfolio during the selloff.", &[shock_trace()]);
        assert_eq!(v.len(), 1, "{v:?}");
        assert_eq!(v[0].trace_direction, "negative");
        assert!(v[0].message.contains("described positively"));
    }

    #[test]
    fn consistent_direction_is_not_flagged() {
        let t = shock_trace();
        assert!(check_directions("GOLDCASE.NS was the largest drag on the portfolio.", std::slice::from_ref(&t)).is_empty());
        assert!(check_directions("Reliance held up and gained during the shock.", &[t]).is_empty());
    }

    #[test]
    fn a_positive_contributor_described_as_a_drag_is_flagged() {
        let v = check_directions("RELIANCE.NS was a drag and hurt returns.", &[shock_trace()]);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].trace_direction, "positive");
    }

    #[test]
    fn mixed_tone_and_mixed_entity_sentences_are_left_alone() {
        let t = shock_trace();
        // "offset the loss" has both tones.
        assert!(check_directions("Reliance offset part of the loss.", std::slice::from_ref(&t)).is_empty());
        // Reliance (up) and KIOCL (down) in one sentence.
        assert!(check_directions("KIOCL dragged while Reliance helped.", &[t]).is_empty());
    }

    #[test]
    fn live_false_alarms_are_not_flagged() {
        // Seen in production: correct sentences the first heuristic flagged.
        let perf = trace_with(json!({"holding_returns": {
            "RATNAVEER.NS": {"total_return_pct": 115.0, "max_drawdown_pct": -21.0, "annualized_vol_pct": 60.0},
            "KIOCL.NS": {"total_return_pct": -27.0, "max_drawdown_pct": -40.0, "annualized_vol_pct": 45.0},
            "ZYDUSWELL.NS": {"total_return_pct": -0.1, "max_drawdown_pct": -18.0, "annualized_vol_pct": 38.0},
        }}));
        let ts = std::slice::from_ref(&perf);
        // drawdown is not a negative description of a winner
        assert!(check_directions("I recommend you reduce RATNAVEER.NS to limit its drawdown potential.", ts).is_empty());
        // "gains" belongs to another holding
        assert!(check_directions("Alongside KIOCL.NS you should consider locking in some of your gains.", ts).is_empty());
        // negation
        assert!(check_directions("ZYDUSWELL.NS is failing to contribute positively to returns.", ts).is_empty());
        // the highest-vol holding is a legitimate reduce target
        assert!(check_recommendations("I recommend you reduce RATNAVEER.NS to lower volatility.", ts).is_empty());
    }

    #[test]
    fn word_stems_do_not_match_inside_other_words() {
        // "against" contains "gain"; "switching" contains "itc".
        let t = trace_with(json!({"per_holding": [{"ticker": "ITC.NS", "pnl_inr": -100.0}]}));
        assert!(check_directions("Switching against the trend.", &[t]).is_empty());
    }

    #[test]
    fn tickers_with_dots_do_not_split_sentences() {
        let s = sentences("GOLDCASE.NS fell 2.5% today. KIOCL.NS also dropped!");
        assert_eq!(s.len(), 2, "{s:?}");
    }

    #[test]
    fn recommending_a_holding_outside_the_worst_contributors_is_flagged() {
        let v = check_recommendations(
            "Consider reducing your Reliance exposure to cut vol.",
            &[shock_trace()],
        );
        assert_eq!(v.len(), 1, "{v:?}");
        assert_eq!(v[0].trace_direction, "not_a_top_contributor");
        // KIOCL is a top-3 loser (GOLDCASE, KIOCL, BLS): fine. DYCL is 4th: flagged.
        assert!(check_recommendations("Reduce KIOCL.NS first.", &[shock_trace()]).is_empty());
        assert_eq!(check_recommendations("Trim DYCL as well.", &[shock_trace()]).len(), 1);
    }

    #[test]
    fn recommendations_use_drawdowns_and_vol_contributions_too() {
        let perf = trace_with(json!({"holding_returns": {
            "KIOCL.NS": {"total_return_pct": -30.0, "max_drawdown_pct": -45.0},
            "BLS.NS": {"total_return_pct": -20.0, "max_drawdown_pct": -35.0},
            "DYCL.NS": {"total_return_pct": -10.0, "max_drawdown_pct": -25.0},
            "ELECON.NS": {"total_return_pct": -5.0, "max_drawdown_pct": -15.0},
            "TCS.NS": {"total_return_pct": 5.0, "max_drawdown_pct": -4.0},
        }}));
        assert!(check_recommendations("Cut KIOCL.NS.", std::slice::from_ref(&perf)).is_empty());
        assert_eq!(check_recommendations("Cut TCS.NS.", &[perf]).len(), 1);
        let vol = trace_with(json!({"by_stock": [
            {"ticker": "A.NS", "contribution": 0.5}, {"ticker": "B.NS", "contribution": 0.3}, {"ticker": "C.NS", "contribution": 0.1}
        ]}));
        assert!(check_recommendations("Reduce A.NS and B.NS.", std::slice::from_ref(&vol)).is_empty());
        assert_eq!(check_recommendations("Reduce C.NS.", &[vol]).len(), 1);
    }

    #[test]
    fn no_per_holding_data_means_no_recommendation_check() {
        let t = trace_with(json!({"portfolio_vol_annualized": 0.2}));
        assert!(check_recommendations("Reduce Reliance.", &[t]).is_empty());
    }
}
