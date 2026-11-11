//! Plain-language narration of a completed `EvidenceTrace` via Gemini.
//! Grounding (verifying every number Gemini states actually appears in the
//! trace) lives in `crate::grounding`; this module only makes the call.

use compute::trace::EvidenceTrace;
use thiserror::Error;

use crate::conversation::{turn_to_content, ConversationTurn};
use crate::gemini::{Content, GeminiClient, GeminiError, GeminiRequest, Part, MODEL_NARRATE};

/// The strict narration prompt: the model only translates trace values into
/// English. Per-experiment field hints, direction constraints, the
/// recommendation whitelist and the user's question travel in the *context*
/// message (see `build_context`), not here.
pub const NARRATE_SYSTEM_PROMPT: &str = r##"You are a narrator. Your only job is to 
translate the numbers and values produced 
by the Drift risk engine into plain English.

You have NO independent knowledge, reasoning, 
or opinions about markets, stocks, or finance.
Everything you say must come directly from 
the trace data provided.

STRICT RULES — violation of any rule is 
a critical failure:

RULE 1 — Numbers only from the trace.
Every number, percentage, and rupee amount 
you state must exist in the trace.
Never compute, estimate, round differently, 
or derive new numbers.
Use the _pct fields for percentages — they 
are already rounded correctly.

RULE 2 — Direction from the trace.
If the trace shows a holding or factor with 
negative P&L, attribution, or return:
  → it HURT the portfolio
  → use words: hurt, dragged, reduced, 
    negative contributor, loss
  → NEVER use: stabiliser, hedge, offset, 
    protection, buffer, safe haven, helped

If the trace shows positive P&L or return:
  → it HELPED the portfolio  
  → use words: contributed positively, gained,
    helped, positive contributor
  → NEVER use: drag, risk, hurt, loss

You do not decide direction. The trace decides.

RULE 3 — Recommendations only from trace data.
If you recommend reducing a position, the 
holding MUST appear in the trace as one of:
  - top-3 most negative P&L in holding_pnl
  - top-3 deepest drawdown in deepest_drawdowns
  - highest Euler vol contributor with negative 
    return in holding_returns

NEVER recommend action on a holding based on:
  - its name or company reputation
  - general market knowledge
  - prior conversation
  - any reasoning not in the current trace

RULE 4 — Answer the user's specific question.
The user's current question is provided at 
the top of the context.
If they asked about a specific stock, 
answer about that stock.
If they asked about beta, state the beta.
If they asked about a macro shock, describe 
the shock impact.
Do not give a generic portfolio summary when 
a specific question was asked.
If sector_question is true in the context:
Do NOT classify any holding into a sector.
State that sector data is not available and
list all holdings by the relevant metric 
(vol contribution or return) so the user 
can identify their sector holdings.

RULE 5 — Format.
All rupee amounts: Indian notation 
(₹X.XL, ₹X.XCr, ₹X,XXX).
All percentages: one decimal place (14.7%).
No raw decimals (never 0.14678).
No field names from the trace.
No log-space values.
3-5 sentences for simple results.
Up to 8 sentences for complex multi-tool results.

RULE 6 — Structure every response as:
Sentence 1: Direct answer to the user's question 
            with the key number from the trace.
Sentence 2-4: Supporting context from the trace 
              (factor contributions, regime, 
              top contributors — all from trace).
Final sentence: One concrete recommendation 
                based only on trace data.

RULE 6 ADDITION — Decomposition consistency:
In a single response, use either factor-level
percentages OR holding-level percentages,
never both. If you state that MARKET accounts
for 77.5% of volatility, do not also state
that GOLDCASE.NS accounts for 28.5% of
volatility in the same response — these are
two different mathematical views of the same
risk and presenting both confuses users.

The exception: recommendations may name a
specific holding even when the body uses
factor decomposition.

RULE 7 — Current question context.
The user's current question will be provided 
as the first item in the conversation.
Treat it as the primary instruction.
Do not let prior conversation turns override 
what the current question is asking."##;

#[derive(Debug, Error)]
pub enum NarrateError {
    #[error("gemini error: {0}")]
    Gemini(#[from] GeminiError),
    #[error("failed to serialize the evidence trace: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("gemini response had no candidates")]
    NoCandidates,
    #[error("gemini response had no text part")]
    NoText,
}

/// Marks where the prior conversation ends, so entities discussed earlier
/// are not carried into this narration (see `NARRATE_SYSTEM_PROMPT` Rule 0).
const HISTORY_SEPARATOR: &str = "--- PRIOR CONVERSATION ENDS HERE ---\nWhat follows is the CURRENT experiment result. Narrate ONLY this. Do not reference entities from the prior conversation unless they appear in this trace.";
const HISTORY_SEPARATOR_ACK: &str =
    "Understood. I will narrate only the current experiment result.";

/// Per-call narration inputs beyond the traces themselves.
#[derive(Debug, Clone, Default)]
pub struct NarrationOptions {
    /// The holding the user asked about specifically (already resolved
    /// against their portfolio, see `orchestrator::apply_focus_holding`).
    pub focus_holding: Option<String>,
    /// The user's current message, shown at the top of the context.
    pub user_question: Option<String>,
    /// When the user asked about "today"/"right now": the sentence the
    /// narration must open with (data is daily, not intraday).
    pub realtime_note: Option<String>,
    /// The user asked which stocks in a sector are riskiest/best/worst.
    pub sector_question: bool,
}

/// One-line hint on which fields carry the answer, per experiment type.
/// (Names are the real output fields; they sit in the context rather than
/// the system prompt so the prompt itself stays experiment-agnostic.)
fn field_hint(experiment: &str) -> Option<&'static str> {
    Some(match experiment {
        "FactorShock" => "Key fields to narrate: portfolio_pnl_inr, given_shocks_pct, implied_shocks_pct, per_holding (top 3 by absolute pnl_inr), shock_historical_context.context_label",
        "RiskDecomposition" => "Key fields to narrate for RiskDecomposition:\n\
CHOOSE ONE decomposition view only -- never mix them in the same response:\n\
- If the question is about factor exposure or macro risk (what factors, market beta): use the FACTOR decomposition: portfolio_vol_annualized_pct, by_factor (top 2 by fraction_of_vol_pct), specific_risk_fraction_of_vol_pct, portfolio_betas (the MARKET beta, to 2 decimals, when asked about beta). Do NOT mention individual holding names in the risk decomposition.\n\
- If the question is about which holdings are risky or concentrated: use the HOLDING decomposition: portfolio_vol_annualized_pct, by_stock (top 3 by fraction_of_vol_pct). Do NOT mention factor percentages alongside holding percentages.\n\
- For a general risk question ('what is my risk', 'where is my risk concentrated'): use the FACTOR decomposition first (2 sentences), then name the single largest HOLDING contributor (from by_stock) in the recommendation only, without its percentage. Never state both a factor % AND a holding % in the same paragraph.\n\
Also available in either view: model_params.regime_state.current_label, garch_forecast (20-day, vol_direction)",
        "CvarRebalance" => "Key fields to narrate: stats_before.historical_cvar, stats_after.historical_cvar, turnover, commission_cost_inr, policy_breaches_resolved (if present)",
        "ReverseStress" => "Key fields to narrate: severity_label, mahalanobis_severity, shock_vector (describe as a scenario), portfolio_pnl_inr, most_vulnerable_holdings (top 3)",
        "PolicyCheck" => "Key fields to narrate: policy_result.all_passed, policy_result.breach_count, and for breaches only each check's rule, actual and limit",
        "RiskDrift" => "Key fields to narrate: vol_before, vol_after, vol_change_pct, largest_contribution_increase, regime_before, regime_after, days_elapsed",
        "PortfolioPerformance" => "Key fields to narrate: total_return_pct, annualized_return_pct, annualized_vol_pct, max_drawdown_pct, the worst and best performer from holding_returns (ticker + total_return_pct), the current regime",
        _ => return None,
    })
}

/// Context for a sector question: there is no sector data, so the narration
/// must say so and show rankings from the trace instead of guessing which
/// holdings are PSU/banking/IT. Rankings are capped at 8 per list to keep
/// the answer readable for a large portfolio.
fn sector_context(traces: &[EvidenceTrace]) -> String {
    let mut out = String::from(
        "sector_question: true\n\
         I don't have sector classification data, so do NOT classify any holding as PSU, banking, IT, pharma or any other sector, and do not name any holding as belonging to one. \
         Say plainly that sector data is not available, show the rankings below so the user can pick out their own sector holdings, and end with one recommendation based only on the trace.\n",
    );
    for t in traces {
        let Some(map) = t
            .outputs
            .get("result")
            .and_then(|r| r.get("holding_returns"))
            .and_then(|h| h.as_object())
        else {
            continue;
        };
        let rows: Vec<(&String, f64, f64)> = map
            .iter()
            .filter_map(|(k, v)| {
                Some((
                    k,
                    v.get("annualized_vol_pct")?.as_f64()?,
                    v.get("total_return_pct")?.as_f64()?,
                ))
            })
            .collect();
        let mut by_vol = rows.clone();
        by_vol.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.push_str("Holdings by annualized volatility (highest first, top 8): ");
        out.push_str(
            &by_vol
                .iter()
                .take(8)
                .map(|(k, v, _)| format!("{k} {v:.1}%"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push('\n');
        let mut by_ret = rows;
        by_ret.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        out.push_str("Holdings by total return (lowest first, bottom 8): ");
        out.push_str(
            &by_ret
                .iter()
                .take(8)
                .map(|(k, _, r)| format!("{k} {r:.1}%"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push_str("\n\n");
    }
    out
}

/// The user-role message sent for narration: the question, then the
/// constraints the system prompt's rules rely on (direction per holding and
/// factor, the only holdings a reduce recommendation may name, field hints,
/// entity isolation), then the trace itself.
pub(crate) fn build_context(
    traces: &[EvidenceTrace],
    evidence_json: &str,
    history_is_empty: bool,
    options: &NarrationOptions,
) -> String {
    let mut out = String::new();
    if let Some(q) = &options.user_question {
        out.push_str(&format!("CURRENT USER QUESTION: {q}\n\n"));
    }
    if let Some(note) = &options.realtime_note {
        out.push_str(&format!(
            "BEGIN YOUR RESPONSE WITH EXACTLY THIS SENTENCE, THEN ANSWER: \"{note}\"\n\n"
        ));
    }
    if options.sector_question {
        out.push_str(&sector_context(traces));
    }
    if let Some(holding) = &options.focus_holding {
        out.push_str(&format!(
            "USER QUESTION IS SPECIFICALLY ABOUT: {holding}\n\
             Focus your ENTIRE response on this holding.\n\
             - How has it performed (total return %)?\n\
             - What % of portfolio risk does it contribute?\n\
             - How does it compare to your other holdings?\n\
             Do NOT give a generic portfolio summary.\n\
             Answer the question about this specific stock.\n\n"
        ));
    }
    if !history_is_empty {
        out.push_str(
            "ENTITY ISOLATION: mention only holdings and factors that appear in the trace below. \
             Do not carry any stock, conclusion or recommendation over from the earlier conversation.\n\n",
        );
    }

    let lines = crate::direction::direction_lines(traces);
    if !lines.is_empty() {
        out.push_str(
            "MANDATORY DIRECTION CONSTRAINTS (taken from the trace; never contradict them):\n",
        );
        for l in &lines {
            out.push_str(&format!("- {l}\n"));
        }
        out.push('\n');
    }
    if let Some(candidates) = crate::direction::reduce_candidates(traces) {
        if candidates.is_empty() {
            out.push_str("RECOMMENDATION CONSTRAINT: no holding in this trace qualifies for a reduce recommendation. Do not recommend reducing any specific holding.\n\n");
        } else {
            out.push_str(&format!(
                "RECOMMENDATION CONSTRAINT: if you recommend reducing a position, name only one of these holdings (they are the trace's worst contributors): {}. Never name any other holding.\n\n",
                candidates.join(", ")
            ));
        }
    }

    let mut hints: Vec<&str> = Vec::new();
    for t in traces {
        if let Some(h) = field_hint(&t.experiment) {
            if !hints.contains(&h) {
                hints.push(h);
            }
        }
    }
    for h in hints {
        out.push_str(h);
        out.push('\n');
    }
    out.push_str(&format!("\nTRACE:\n{evidence_json}"));
    out
}

/// The `contents` for a narration call: prior turns, then (if there were
/// any) a separator exchange, then the context message (see
/// `build_context`) as the final user turn.
fn build_contents(
    conversation_history: &[ConversationTurn],
    traces: &[EvidenceTrace],
    evidence_json: String,
    options: &NarrationOptions,
) -> Vec<Content> {
    let mut contents: Vec<Content> = conversation_history.iter().map(turn_to_content).collect();
    if !conversation_history.is_empty() {
        contents.push(Content {
            role: Some("user".to_string()),
            parts: vec![Part::text(HISTORY_SEPARATOR)],
        });
        contents.push(Content {
            role: Some("model".to_string()),
            parts: vec![Part::text(HISTORY_SEPARATOR_ACK)],
        });
    }
    let payload = build_context(
        traces,
        &evidence_json,
        conversation_history.is_empty(),
        options,
    );
    contents.push(Content {
        role: Some("user".to_string()),
        parts: vec![Part::text(payload)],
    });
    contents
}

async fn call_narrate<C: GeminiClient>(
    client: &C,
    traces: &[EvidenceTrace],
    evidence_json: String,
    extra_instructions: Option<&str>,
    conversation_history: &[ConversationTurn],
    options: &NarrationOptions,
) -> Result<String, NarrateError> {
    let mut system_prompt = NARRATE_SYSTEM_PROMPT.to_string();
    if let Some(extra) = extra_instructions {
        system_prompt.push_str("\n\n");
        system_prompt.push_str(extra);
    }
    let request = GeminiRequest {
        contents: build_contents(conversation_history, traces, evidence_json, options),
        system_instruction: Some(Content {
            role: None,
            parts: vec![Part::text(system_prompt)],
        }),
        tools: None,
    };
    let response = client.generate(MODEL_NARRATE, &request).await?;
    let part = response.first_part().ok_or(NarrateError::NoCandidates)?;
    part.text.clone().ok_or(NarrateError::NoText)
}

/// Narrates `trace`, optionally appending `extra_instructions` to the
/// system prompt (used by `grounding::grounded_narrate` to ask for a
/// grounding-corrected rewrite). `conversation_history` (if any) is
/// included as prior turns -- followed by a separator so earlier entities
/// don't bleed in -- before the trace-injection turn. The grounding check
/// itself still only validates this turn's narration against this call's
/// trace.
pub async fn narrate_with_instructions<C: GeminiClient>(
    client: &C,
    trace: &EvidenceTrace,
    extra_instructions: Option<&str>,
    conversation_history: &[ConversationTurn],
) -> Result<String, NarrateError> {
    let trace_json = serde_json::to_string(trace)?;
    call_narrate(
        client,
        std::slice::from_ref(trace),
        trace_json,
        extra_instructions,
        conversation_history,
        &NarrationOptions::default(),
    )
    .await
}

/// Narrates `trace` with the base system prompt only (no grounding retry,
/// no conversation history).
pub async fn narrate<C: GeminiClient>(
    client: &C,
    trace: &EvidenceTrace,
) -> Result<String, NarrateError> {
    narrate_with_instructions(client, trace, None, &[]).await
}

/// Multi-tool variant of `narrate_with_instructions`: injects every trace
/// in `traces` as a single JSON array user-turn message, rather than one
/// trace object.
pub async fn narrate_tools_with_instructions<C: GeminiClient>(
    client: &C,
    traces: &[EvidenceTrace],
    extra_instructions: Option<&str>,
    conversation_history: &[ConversationTurn],
) -> Result<String, NarrateError> {
    narrate_tools_with_options(
        client,
        traces,
        extra_instructions,
        conversation_history,
        &NarrationOptions::default(),
    )
    .await
}

/// `narrate_tools_with_instructions` plus `NarrationOptions` (the focus
/// holding for "how is X doing" questions).
pub async fn narrate_tools_with_options<C: GeminiClient>(
    client: &C,
    traces: &[EvidenceTrace],
    extra_instructions: Option<&str>,
    conversation_history: &[ConversationTurn],
    options: &NarrationOptions,
) -> Result<String, NarrateError> {
    let traces_json = serde_json::to_string(traces)?;
    call_narrate(
        client,
        traces,
        traces_json,
        extra_instructions,
        conversation_history,
        options,
    )
    .await
}

/// Multi-tool variant of `narrate` (base system prompt only, no grounding
/// retry, no conversation history).
pub async fn narrate_tools<C: GeminiClient>(
    client: &C,
    traces: &[EvidenceTrace],
) -> Result<String, NarrateError> {
    narrate_tools_with_instructions(client, traces, None, &[]).await
}
