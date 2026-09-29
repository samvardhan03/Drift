//! Natural-language -> `Experiment` extraction via a single Gemini
//! function-calling turn.

use compute::experiments::{
    CvarRebalanceInput, Experiment, FactorShockInput, PolicyCheckInput, Portfolio,
    PortfolioPerformanceInput, ReverseStressInput, RiskDecompositionInput, RiskDriftInput,
};
use thiserror::Error;

use crate::conversation::{turn_to_content, ConversationTurn};
use crate::gemini::{Content, GeminiClient, GeminiError, GeminiRequest, Part, Tool, MODEL_PARSE};
use crate::schema::{
    experiment_function_declarations, CVAR_REBALANCE_FUNCTION, FACTOR_SHOCK_FUNCTION,
    POLICY_CHECK_FUNCTION, PORTFOLIO_PERFORMANCE_FUNCTION, REVERSE_STRESS_FUNCTION,
    RISK_DECOMPOSITION_FUNCTION, RISK_DRIFT_FUNCTION,
};

/// Adapted from the checkpoint spec's original wording, which named a
/// single `run_experiment` function: that design doesn't work in practice
/// (see `schema`'s module doc — Gemini reliably drops the `"type"`
/// discriminator from a `oneOf`-typed function's args), so this names the
/// real functions instead. Everything else is unchanged.
pub const PARSE_SYSTEM_PROMPT: &str = "You are a parameter extraction engine. Your only job is \
to call the correct function -- run_factor_shock, run_risk_decomposition, run_cvar_rebalance, \
run_portfolio_performance, run_risk_drift, run_reverse_stress, or run_policy_check -- with the \
parameters extracted from the user's message. Do not add explanation. Do not ask clarifying \
questions. If the user's intent clearly maps to one of the seven experiment types, call the \
function. If it does not, return a text response with one sentence explaining what you cannot \
extract. For CvarRebalance: if the user does not mention a per-name cap, omit per_name_cap from \
the function call.";

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("gemini error: {0}")]
    Gemini(#[from] GeminiError),
    /// Gemini responded with text instead of calling a function: the
    /// user's message didn't clearly map to one of the three experiments.
    #[error("could not extract an experiment from the message: {0}")]
    Unrecognised(String),
    #[error("gemini returned no candidates")]
    NoCandidates,
    #[error("gemini returned neither a function call nor text")]
    EmptyResponse,
    #[error("gemini called an unexpected function: {0}")]
    UnexpectedFunction(String),
    #[error("failed to deserialize function-call args into an Experiment: {0}")]
    InvalidArgs(#[from] serde_json::Error),
}

/// Sends `user_message` to Gemini with the three per-experiment function
/// declarations (see `schema`'s module doc). `conversation_history` (if
/// any) is included as prior turns before the current user message, so the
/// model can resolve references like "now try with 25% turnover" without
/// the caller re-stating prior context; an empty history produces the same
/// request shape as before it existed. On a function-call response,
/// deserializes the args into the matching `Experiment` variant and
/// overwrites its `portfolio` field with the caller's `portfolio` (Gemini
/// is never given portfolio data — see `schema::CALLER_SUPPLIED_FIELDS` —
/// so this is always the source of truth, whether or not Gemini's args
/// happened to include one). On a text response, returns
/// `ParseError::Unrecognised`.
pub async fn parse_experiment<C: GeminiClient>(
    client: &C,
    user_message: &str,
    portfolio: Portfolio,
    conversation_history: &[ConversationTurn],
) -> Result<Experiment, ParseError> {
    let mut contents: Vec<Content> = conversation_history.iter().map(turn_to_content).collect();
    contents.push(Content {
        role: Some("user".to_string()),
        parts: vec![Part::text(user_message)],
    });

    let request = GeminiRequest {
        contents,
        system_instruction: Some(Content {
            role: None,
            parts: vec![Part::text(PARSE_SYSTEM_PROMPT)],
        }),
        tools: Some(vec![Tool {
            function_declarations: experiment_function_declarations(),
        }]),
    };

    let response = client.generate(MODEL_PARSE, &request).await?;
    let candidate = response.candidates.first().ok_or(ParseError::NoCandidates)?;

    for part in &candidate.content.parts {
        if let Some(call) = &part.function_call {
            let mut args = call.args.clone();
            if let serde_json::Value::Object(ref mut map) = args {
                map.insert("portfolio".to_string(), serde_json::to_value(&portfolio)?);
            }
            let experiment = match call.name.as_str() {
                FACTOR_SHOCK_FUNCTION => {
                    Experiment::FactorShock(serde_json::from_value::<FactorShockInput>(args)?)
                }
                RISK_DECOMPOSITION_FUNCTION => Experiment::RiskDecomposition(serde_json::from_value::<
                    RiskDecompositionInput,
                >(args)?),
                CVAR_REBALANCE_FUNCTION => {
                    Experiment::CvarRebalance(serde_json::from_value::<CvarRebalanceInput>(args)?)
                }
                PORTFOLIO_PERFORMANCE_FUNCTION => Experiment::PortfolioPerformance(
                    serde_json::from_value::<PortfolioPerformanceInput>(args)?,
                ),
                RISK_DRIFT_FUNCTION => {
                    Experiment::RiskDrift(serde_json::from_value::<RiskDriftInput>(args)?)
                }
                REVERSE_STRESS_FUNCTION => {
                    Experiment::ReverseStress(serde_json::from_value::<ReverseStressInput>(args)?)
                }
                POLICY_CHECK_FUNCTION => {
                    Experiment::PolicyCheck(serde_json::from_value::<PolicyCheckInput>(args)?)
                }
                other => return Err(ParseError::UnexpectedFunction(other.to_string())),
            };
            return Ok(experiment);
        }
        if let Some(text) = &part.text {
            return Err(ParseError::Unrecognised(text.clone()));
        }
    }

    Err(ParseError::EmptyResponse)
}
