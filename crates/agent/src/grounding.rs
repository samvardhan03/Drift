//! Grounding check: verifies every number stated in a narration actually
//! appears (after normalisation) somewhere in the evidence trace it
//! describes, retrying the narration if not.

use compute::trace::EvidenceTrace;
use regex::Regex;
use std::sync::OnceLock;

use crate::conversation::ConversationTurn;
use crate::gemini::GeminiClient;
use crate::narrate::{narrate_with_instructions, NarrateError};

/// Relative tolerance for matching a narration number against a trace
/// number, to allow for Gemini's own minor rounding in prose (spec value).
pub const RELATIVE_TOLERANCE: f64 = 0.02;
/// Max grounding-retry attempts after the initial narration (spec value).
pub const MAX_RETRIES: u32 = 2;

/// A number extracted from narration text that could not be matched
/// against any trace value.
#[derive(Debug, Clone, PartialEq)]
pub struct UnmatchedNumber {
    pub value: f64,
    pub raw_text: String,
    /// Byte offset of `raw_text` within the narration string.
    pub position: usize,
}

#[derive(Debug, Clone, Default)]
pub struct GroundingCheck {
    pub matched: Vec<f64>,
    pub unmatched: Vec<UnmatchedNumber>,
}

impl GroundingCheck {
    pub fn passed(&self) -> bool {
        self.unmatched.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct GroundedNarration {
    pub narration: String,
    /// Empty if the narration passed grounding (on the first attempt or
    /// after a retry); populated only if it still had unmatched numbers
    /// after `MAX_RETRIES` retries. The narration is returned either way —
    /// this field surfaces the warning rather than suppressing the answer.
    pub grounding_warnings: Vec<String>,
}

// --- Number extraction ---------------------------------------------------
//
// A single alternation, tried left-to-right at each position, so the most
// specific pattern (crore, then lakh, then percent, then a bare number)
// wins and consumes the whole span before the generic pattern gets a
// chance at the same digits:
//   crore:   (-?)(?:₹\s*)?([\d,]+(?:\.\d+)?)(?:\s*crore\b|Cr\b) -> value * 1e7
//   lakh:    (-?)(?:₹\s*)?([\d,]+(?:\.\d+)?)(?:\s*lakh\b|L\b)   -> value * 1e5
//   percent: (-?)([\d,]+(?:\.\d+)?)\s*%                          -> value / 100
//   plain:   (-?)(?:₹\s*)?([\d,]+(?:\.\d+)?)                      -> value as-is
// Both the abbreviated forms `NARRATE_SYSTEM_PROMPT` itself mandates
// (`₹X.XL`, `₹X.XCr`, no space before the suffix) and the spelled-out
// words (`lakh`, `crore`) are recognised -- an earlier version of this
// regex only matched `lakh`, so every narration correctly following our
// own formatting rule 0 (₹1L–₹1Cr -> `₹X.XL`) was spuriously flagged as
// ungrounded (caught live in this session's verification: "₹12.0L" went
// unmatched).
// The sign is captured *before* the optional ₹ symbol, not just before the
// digits: Indian financial prose (and Gemini's own narration) commonly
// writes a negative rupee amount as "\u{2212}\u{20b9}12,34,567" (minus,
// then rupee sign, then digits), not "\u{20b9}\u{2212}12,34,567" -- a
// sign-after-₹-only pattern would silently drop the minus and misread a
// loss as a gain (also caught live this session; regression-tested below).
// The minus sign accepts both ASCII '-' and Unicode minus '\u{2212}' ('−'),
// since Gemini (and Indian financial prose generally) sometimes uses the
// latter. Commas are stripped before parsing, which handles both Western
// (1,175,389) and Indian (11,75,389) grouping identically, since only the
// digit sequence (not the grouping) matters once separators are removed.
fn number_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?x)
            (?P<crore>
                (?P<crore_sign>[-−]?) (?:₹\s*)? (?P<crore_num>[\d,]+(?:\.\d+)?)
                (?: \s* crore\b | Cr\b )
            )
            |
            (?P<lakh>
                (?P<lakh_sign>[-−]?) (?:₹\s*)? (?P<lakh_num>[\d,]+(?:\.\d+)?)
                (?: \s* lakh\b | L\b )
            )
            |
            (?P<pct>
                (?P<pct_sign>[-−]?) (?P<pct_num>[\d,]+(?:\.\d+)?) \s* %
            )
            |
            (?P<plain>
                (?P<plain_sign>[-−]?) (?:₹\s*)? (?P<plain_num>[\d,]+(?:\.\d+)?)
            )
            ",
        )
        .expect("static regex is valid")
    })
}

/// Robust numeric parse: strips thousands separators and normalizes the
/// Unicode minus sign to ASCII before delegating to `f64::from_str`.
/// `sign` is the sign captured separately from `digits` (see
/// `number_regex`'s doc comment on why they're captured apart).
fn parse_digits(sign: &str, digits: &str) -> Option<f64> {
    let normalized: String = digits.chars().filter(|&c| c != ',').collect();
    let value: f64 = normalized.parse().ok()?;
    Some(if sign == "-" || sign == "\u{2212}" { -value } else { value })
}

/// One number extracted from narration text, with its normalized value,
/// original matched text, and byte position.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedNumber {
    pub value: f64,
    pub raw_text: String,
    pub position: usize,
}

/// Extracts and normalizes every numeric token from `text` (see the regex
/// doc comment above for exactly what's recognized).
pub fn extract_numbers(text: &str) -> Vec<ExtractedNumber> {
    let re = number_regex();
    let mut out = Vec::new();
    for caps in re.captures_iter(text) {
        let (value, whole) = if let Some(m) = caps.name("crore") {
            let sign = caps.name("crore_sign").unwrap().as_str();
            let num = caps.name("crore_num").unwrap().as_str();
            (parse_digits(sign, num).map(|v| v * 10_000_000.0), m)
        } else if let Some(m) = caps.name("lakh") {
            let sign = caps.name("lakh_sign").unwrap().as_str();
            let num = caps.name("lakh_num").unwrap().as_str();
            (parse_digits(sign, num).map(|v| v * 100_000.0), m)
        } else if let Some(m) = caps.name("pct") {
            let sign = caps.name("pct_sign").unwrap().as_str();
            let num = caps.name("pct_num").unwrap().as_str();
            (parse_digits(sign, num).map(|v| v / 100.0), m)
        } else if let Some(m) = caps.name("plain") {
            let sign = caps.name("plain_sign").unwrap().as_str();
            let num = caps.name("plain_num").unwrap().as_str();
            (parse_digits(sign, num), m)
        } else {
            continue;
        };
        if let Some(value) = value {
            out.push(ExtractedNumber {
                value,
                raw_text: whole.as_str().to_string(),
                position: whole.start(),
            });
        }
    }
    out
}

// --- Trace flattening ------------------------------------------------------

/// Collects every numeric JSON leaf in `value` (recursively; strings,
/// bools, and nulls are ignored).
pub fn numeric_leaves(value: &serde_json::Value) -> Vec<f64> {
    let mut out = Vec::new();
    collect_numeric_leaves(value, &mut out);
    out
}

fn collect_numeric_leaves(value: &serde_json::Value, out: &mut Vec<f64>) {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                out.push(f);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_numeric_leaves(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for v in map.values() {
                collect_numeric_leaves(v, out);
            }
        }
        _ => {}
    }
}

fn approx_eq(a: f64, b: f64, relative_tolerance: f64) -> bool {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1e-9);
    diff / scale <= relative_tolerance
}

/// Checks every number extracted from `narration` against `trace_numbers`
/// (from `numeric_leaves(&serde_json::to_value(trace)?)`), within
/// `RELATIVE_TOLERANCE`.
pub fn check_grounding(narration: &str, trace_numbers: &[f64]) -> GroundingCheck {
    let mut check = GroundingCheck::default();
    for extracted in extract_numbers(narration) {
        // Matches the extracted value against a trace number either
        // signed or negated: the conversational narration style (see
        // `NARRATE_SYSTEM_PROMPT`) routinely conveys a loss/decline's sign
        // through the surrounding words instead of a literal minus sign
        // ("lost 17.8%", "a drawdown of 20.4%"), which is exactly the
        // magnitude a negative trace value like `total_return_pct: -17.8`
        // grounds, just without the model repeating the sign character.
        // Caught live verifying this session's narration style change.
        let is_match = trace_numbers
            .iter()
            .any(|&t| approx_eq(extracted.value, t, RELATIVE_TOLERANCE) || approx_eq(-extracted.value, t, RELATIVE_TOLERANCE));
        if is_match {
            check.matched.push(extracted.value);
        } else {
            check.unmatched.push(UnmatchedNumber {
                value: extracted.value,
                raw_text: extracted.raw_text,
                position: extracted.position,
            });
        }
    }
    check
}

fn retry_instructions(unmatched: &[UnmatchedNumber]) -> String {
    let list = unmatched
        .iter()
        .map(|u| u.raw_text.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "The following numbers in your previous response could not be verified in the trace: \
         [{list}]. Rewrite your response using only numbers that appear verbatim in the trace \
         provided."
    )
}

/// Narrates `trace`, checking every stated number against the trace and
/// retrying (up to `MAX_RETRIES` times) with the failing numbers named in
/// an appended system instruction if the check fails. Always returns the
/// last narration produced; `grounding_warnings` is only non-empty if it
/// still had unmatched numbers after all retries.
pub async fn grounded_narrate<C: GeminiClient>(
    client: &C,
    trace: &EvidenceTrace,
    conversation_history: &[ConversationTurn],
) -> Result<GroundedNarration, NarrateError> {
    let trace_value = serde_json::to_value(trace)?;
    let trace_numbers = numeric_leaves(&trace_value);

    let mut narration =
        narrate_with_instructions(client, trace, None, conversation_history).await?;
    let mut check = check_grounding(&narration, &trace_numbers);
    let mut retries = 0;
    while !check.passed() && retries < MAX_RETRIES {
        let extra = retry_instructions(&check.unmatched);
        narration =
            narrate_with_instructions(client, trace, Some(&extra), conversation_history).await?;
        check = check_grounding(&narration, &trace_numbers);
        retries += 1;
    }

    let grounding_warnings = if check.passed() {
        Vec::new()
    } else {
        check
            .unmatched
            .iter()
            .map(|u| {
                format!(
                    "unverified number '{}' at byte position {} in the narration",
                    u.raw_text, u.position
                )
            })
            .collect()
    };

    Ok(GroundedNarration {
        narration,
        grounding_warnings,
    })
}

/// Multi-tool variant of `grounded_narrate`: narrates `traces` together
/// (via `narrate::narrate_tools_with_instructions`) and checks every
/// stated number against the *union* of all traces' numeric leaves, so a
/// number correctly attributed to (say) the second tool's result isn't
/// flagged just because it doesn't appear in the first tool's trace.
/// Returns the retry count alongside the result (unlike `grounded_narrate`)
/// since `AgentExecutionTrace::grounding_status.retry_count` needs it.
pub async fn grounded_narrate_many<C: GeminiClient>(
    client: &C,
    traces: &[EvidenceTrace],
    conversation_history: &[ConversationTurn],
) -> Result<(GroundedNarration, u32), NarrateError> {
    let trace_numbers: Vec<f64> = traces
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .flat_map(numeric_leaves)
        .collect();

    let mut narration =
        crate::narrate::narrate_tools_with_instructions(client, traces, None, conversation_history).await?;
    let mut check = check_grounding(&narration, &trace_numbers);
    let mut retries = 0;
    while !check.passed() && retries < MAX_RETRIES {
        let extra = retry_instructions(&check.unmatched);
        narration =
            crate::narrate::narrate_tools_with_instructions(client, traces, Some(&extra), conversation_history)
                .await?;
        check = check_grounding(&narration, &trace_numbers);
        retries += 1;
    }

    let grounding_warnings = if check.passed() {
        Vec::new()
    } else {
        check
            .unmatched
            .iter()
            .map(|u| {
                format!(
                    "unverified number '{}' at byte position {} in the narration",
                    u.raw_text, u.position
                )
            })
            .collect()
    };

    Ok((
        GroundedNarration {
            narration,
            grounding_warnings,
        },
        retries,
    ))
}
