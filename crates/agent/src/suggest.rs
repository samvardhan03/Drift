//! Proactive follow-up suggestion: after a completed experiment, asks
//! Gemini for the single most important next *action* a risk manager
//! should take given what was found -- an imperative recommendation, not
//! a question. Plain text, no function calling, no grounding check (a
//! recommendation isn't a factual claim to verify against the trace).

use thiserror::Error;

use crate::gemini::{GeminiClient, GeminiError, GeminiRequest, MODEL_SUGGEST};

/// Verbatim per spec; do not paraphrase.
pub const SUGGEST_SYSTEM_PROMPT: &str = "You are a senior quant analyst. The user just received a risk analysis result. Suggest the single most important follow-up action \u{2014} not question \u{2014} they should take given what was found.

Rules:
- Frame it as an action, not a question: 'Run a stress test for the IL&FS scenario' not 'Would you like to see...'
- Make it specific to what was found, not generic.
- Under 15 words.
- If the result showed a breach or a risk, the suggestion should address it directly.
- Never suggest something already shown in this result.";

#[derive(Debug, Error)]
pub enum SuggestError {
    #[error("gemini error: {0}")]
    Gemini(#[from] GeminiError),
    #[error("gemini response had no candidates")]
    NoCandidates,
    #[error("gemini response had no text part")]
    NoText,
}

/// Asks Gemini for one follow-up action given `narration` (the completed
/// experiment's grounded narration). Returns the response text as-is,
/// including an empty string if that's what Gemini returned.
pub async fn suggest_follow_up<C: GeminiClient>(
    client: &C,
    narration: &str,
) -> Result<String, SuggestError> {
    let request = GeminiRequest::user_turn(SUGGEST_SYSTEM_PROMPT, narration);
    let response = client.generate(MODEL_SUGGEST, &request).await?;
    let part = response.first_part().ok_or(SuggestError::NoCandidates)?;
    part.text.clone().ok_or(SuggestError::NoText)
}
