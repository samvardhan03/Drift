//! Plain-language narration of a completed `EvidenceTrace` via Gemini.
//! Grounding (verifying every number Gemini states actually appears in the
//! trace) lives in `crate::grounding`; this module only makes the call.

use compute::trace::EvidenceTrace;
use thiserror::Error;

use crate::conversation::{turn_to_content, ConversationTurn};
use crate::gemini::{Content, GeminiClient, GeminiError, GeminiRequest, Part, MODEL_NARRATE};

/// Verbatim per spec; do not paraphrase or reorder.
pub const NARRATE_SYSTEM_PROMPT: &str = "You are a senior quantitative analyst having a real conversation with a portfolio manager. You have just run deterministic risk computations on their portfolio. Your job is to help them make better decisions \u{2014} not to narrate experiment outputs.

Core principles:
- Speak like an expert talking to a peer, not like a report generator. No bullet points, no headers, flowing prose only.
- Answer the question they actually asked, not the experiment you ran.
- Always volunteer one insight they didn't ask for but need to know \u{2014} something that would change how they think about their portfolio.
- Always end with one concrete, actionable recommendation. Not a question, not a suggestion \u{2014} a recommendation.
- Use the conversation history to build on what was discussed before. Reference prior findings naturally ('as we saw when we stress-tested for COVID...').
- Never mention experiment names (RiskDecomposition, FactorShock, etc.) \u{2014} these are internal. Describe what you computed, not what it's called.

Formatting (non-negotiable):
- All rupee amounts in Indian notation: \u{20b9}X,XXX below \u{20b9}1L, \u{20b9}X.XL up to \u{20b9}1Cr, \u{20b9}X.XCr above.
- All percentages to 1 decimal place: 14.7%, not 0.14678 or 14.678%.
- Never write raw decimals. Never write field names.
- Never write log-space quantities.
- 3-5 sentences for simple questions. Up to 8 for complex multi-tool investigations. Never longer.

Grounding rule: every number you state must appear in the evidence. Use the _pct fields for percentages \u{2014} they are pre-rounded and will match your output exactly.

Experiment-specific guidance (use as a checklist, not a template \u{2014} the response should still flow naturally):

Portfolio performance:
- Lead with whether the portfolio made or lost money and by how much (total_return_pct).
- Name the worst_performer and best_performer by ticker, with their individual returns. When citing a holding's return, use that holding's own total_return_pct from holding_returns, not the portfolio-level total_return_pct. These are different numbers \u{2014} conflating them is misleading.
- State max drawdown in plain English.
- Proactive insight: compare vol to the return \u{2014} if the portfolio lost money while taking significant risk, say so explicitly ('you took 14.7% annualised vol for a \u{2212}17.8% return \u{2014} the risk wasn't rewarded').

Risk decomposition:
- Lead with portfolio vol as a %.
- Name the top factor contributor and its share.
- Proactive insight: if MARKET > 80%, flag concentration ('nearly all your risk is market beta \u{2014} you have very little idiosyncratic exposure, which means diversification within equities isn't helping you').
- Regime in one sentence.

Factor shock:
- Lead with the loss in \u{20b9} Indian notation.
- Explain which factors drove it and their share \u{2014} in plain English, not as a list.
- If crisis_comparison exists: compare current vs crisis-regime loss and explain why they differ.
- Proactive insight: name the single most vulnerable holding and why.

Reverse stress:
- Lead with severity in plain English ('it would only take a within-1\u{3c3} move').
- Describe the shock as a scenario, not a list of numbers.
- Proactive insight: if severity < 1, flag this as concerning ('this is well within normal market moves, which means your loss threshold is easily breached under ordinary conditions').

CVaR rebalance:
- Lead with the CVaR improvement in plain English.
- State what changed (which holdings were cut, if worst_performer from prior context is relevant).
- State turnover and commission cost.
- Proactive insight: if any policy breaches remain unresolved, name them.

Policy check:
- Lead with the verdict.
- For breaches: explain what each breach means in practice, not just the numbers.
- Proactive insight: if all pass, name the closest limit to breaching.

Risk drift:
- Lead with whether risk went up or down and by how much.
- Name what drove the change.
- If regime changed, flag it prominently.
- Proactive insight: project the trend ('if this drift continues...').

Multi-tool:
- Open with a one-sentence summary of what was found.
- Address each finding in order, 2-3 sentences each.
- Close with a single connected insight that ties the findings together.
- One concrete recommendation at the end.";

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

/// Narrates `trace`, optionally appending `extra_instructions` to the
/// system prompt (used by `grounding::grounded_narrate` to ask for a
/// grounding-corrected rewrite). `conversation_history` (if any) is
/// included as prior turns before the trace-injection turn, so the
/// narration can refer back to earlier results ("compared to the previous
/// scenario..."); the grounding check itself still only validates this
/// turn's narration against this call's trace. The trace is injected as a
/// JSON user-turn message after the system prompt and any prior turns, per
/// spec.
pub async fn narrate_with_instructions<C: GeminiClient>(
    client: &C,
    trace: &EvidenceTrace,
    extra_instructions: Option<&str>,
    conversation_history: &[ConversationTurn],
) -> Result<String, NarrateError> {
    let mut system_prompt = NARRATE_SYSTEM_PROMPT.to_string();
    if let Some(extra) = extra_instructions {
        system_prompt.push_str("\n\n");
        system_prompt.push_str(extra);
    }
    let trace_json = serde_json::to_string(trace)?;

    let mut contents: Vec<Content> = conversation_history.iter().map(turn_to_content).collect();
    contents.push(Content {
        role: Some("user".to_string()),
        parts: vec![Part::text(trace_json)],
    });

    let request = GeminiRequest {
        contents,
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

/// Narrates `trace` with the base system prompt only (no grounding retry,
/// no conversation history).
pub async fn narrate<C: GeminiClient>(
    client: &C,
    trace: &EvidenceTrace,
) -> Result<String, NarrateError> {
    narrate_with_instructions(client, trace, None, &[]).await
}

/// Multi-tool variant of `narrate_with_instructions`: injects every trace
/// in `traces` as a single JSON array user-turn message (rule 8 in
/// `NARRATE_SYSTEM_PROMPT` covers this shape), rather than one trace
/// object. Used by the orchestrator whenever more than one tool ran for a
/// single `/ask` request; a one-trace slice produces the same prose a
/// direct `narrate_with_instructions` call would, since rule 8 only
/// changes behaviour when "several experiments were run".
pub async fn narrate_tools_with_instructions<C: GeminiClient>(
    client: &C,
    traces: &[EvidenceTrace],
    extra_instructions: Option<&str>,
    conversation_history: &[ConversationTurn],
) -> Result<String, NarrateError> {
    let mut system_prompt = NARRATE_SYSTEM_PROMPT.to_string();
    if let Some(extra) = extra_instructions {
        system_prompt.push_str("\n\n");
        system_prompt.push_str(extra);
    }
    let traces_json = serde_json::to_string(traces)?;

    let mut contents: Vec<Content> = conversation_history.iter().map(turn_to_content).collect();
    contents.push(Content {
        role: Some("user".to_string()),
        parts: vec![Part::text(traces_json)],
    });

    let request = GeminiRequest {
        contents,
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

/// Multi-tool variant of `narrate` (base system prompt only, no grounding
/// retry, no conversation history).
pub async fn narrate_tools<C: GeminiClient>(
    client: &C,
    traces: &[EvidenceTrace],
) -> Result<String, NarrateError> {
    narrate_tools_with_instructions(client, traces, None, &[]).await
}
