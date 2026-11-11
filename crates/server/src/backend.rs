//! The `Backend` trait separates route handlers from the concrete
//! compute/Gemini calls, so routes can be tested against a mock backend
//! with no network access (Yahoo Finance or Gemini).

use agent::gemini::GeminiError;
use agent::pipeline::PipelineResult;
use agent::ConversationTurn;
use compute::experiments::{Experiment, Portfolio};
use compute::policy::RiskPolicy;
use compute::trace::EvidenceTrace;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum BackendError {
    /// The NL request didn't map to a supported experiment; the model's
    /// one-sentence explanation. Maps to 422.
    #[error("{0}")]
    Unrecognised(String),
    /// The planner declined the question as out of scope (an individual
    /// stock price scenario, valuation, prediction, news, or non-finance);
    /// the text is the redirect to show the user. `POST /ask` returns it as
    /// a 200 with `is_redirect: true` (see `routes::post_ask`), never as an
    /// error status.
    #[error("{0}")]
    Redirect(String),
    /// A ticker Yahoo Finance doesn't know even after resolution (or an
    /// ISIN given in place of a symbol); the message names it. Maps to 422.
    #[error("{0}")]
    UnresolvedTicker(String),
    /// Too little price history to analyse; the message explains. Maps to 422.
    #[error("{0}")]
    InsufficientData(String),
    /// Market data couldn't be fetched in time (Yahoo timeout/5xx). Maps to 503.
    #[error("{0}")]
    DataUnavailable(String),
    /// Any error from the compute layer (data fetch, model fit, LP solve,
    /// invalid input). Maps to 500.
    #[error("compute error: {0}")]
    Compute(String),
    /// Gemini returned an error (after its own internal retries) or a
    /// malformed response. Maps to 503.
    #[error("gemini unavailable: {0}")]
    GeminiUnavailable(String),
    /// Anything else unexpected. Maps to 500.
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<GeminiError> for BackendError {
    fn from(err: GeminiError) -> Self {
        match err {
            // A missing API key is a startup-time misconfiguration (main()
            // already refuses to start without one); if it's somehow hit
            // at request time, it's ours to fix, not a transient upstream
            // failure.
            GeminiError::MissingApiKey => BackendError::Internal(err.to_string()),
            _ => BackendError::GeminiUnavailable(err.to_string()),
        }
    }
}

impl From<agent::parse::ParseError> for BackendError {
    fn from(err: agent::parse::ParseError) -> Self {
        match err {
            agent::parse::ParseError::Unrecognised(text) => BackendError::Unrecognised(text),
            agent::parse::ParseError::Gemini(g) => BackendError::from(g),
            other => BackendError::Internal(other.to_string()),
        }
    }
}

impl From<agent::orchestrator::OrchestratorError> for BackendError {
    fn from(err: agent::orchestrator::OrchestratorError) -> Self {
        match err {
            agent::orchestrator::OrchestratorError::Gemini(g) => BackendError::from(g),
            agent::orchestrator::OrchestratorError::Unrecognised(text) => {
                BackendError::Unrecognised(text)
            }
            agent::orchestrator::OrchestratorError::Declined(text) => BackendError::Redirect(text),
            other => BackendError::Internal(other.to_string()),
        }
    }
}

impl From<agent::narrate::NarrateError> for BackendError {
    fn from(err: agent::narrate::NarrateError) -> Self {
        match err {
            agent::narrate::NarrateError::Gemini(g) => BackendError::from(g),
            other => BackendError::Internal(other.to_string()),
        }
    }
}

impl From<agent::suggest::SuggestError> for BackendError {
    fn from(err: agent::suggest::SuggestError) -> Self {
        match err {
            agent::suggest::SuggestError::Gemini(g) => BackendError::from(g),
            other => BackendError::Internal(other.to_string()),
        }
    }
}

impl From<agent::pipeline::PipelineError> for BackendError {
    fn from(err: agent::pipeline::PipelineError) -> Self {
        match err {
            agent::pipeline::PipelineError::Orchestrator(o) => BackendError::from(o),
            agent::pipeline::PipelineError::Compute(c) => BackendError::from(c),
            agent::pipeline::PipelineError::Narrate(n) => BackendError::from(n),
            agent::pipeline::PipelineError::Suggest(s) => BackendError::from(s),
            // A single failing tool keeps its own compute error's HTTP
            // classification (e.g. 422 for InsufficientData/UnresolvedTicker,
            // via the `From<ComputeError>` impl below) instead of collapsing
            // into a generic 500. Multiple failures, or a non-compute
            // failure (unknown tool name, bad params), still fall back to
            // the generic message -- there's no single error left to defer
            // to.
            agent::pipeline::PipelineError::AllToolsFailed(mut errors) => match errors.as_slice() {
                [agent::orchestrator::ToolError::Compute(_)] => match errors.pop().unwrap() {
                    agent::orchestrator::ToolError::Compute(e) => BackendError::from(e),
                    agent::orchestrator::ToolError::Other(_) => unreachable!(),
                },
                _ => BackendError::Compute(format!("every planned tool failed: {errors:?}")),
            },
        }
    }
}

impl From<compute::ComputeError> for BackendError {
    fn from(err: compute::ComputeError) -> Self {
        match err {
            // RiskDrift's "no baseline to diff against" case is a request
            // problem (the caller needs to run RiskDecomposition first, or
            // gave a bad snapshot id), not an internal failure -- map it
            // the same way `ParseError::Unrecognised` already is (422),
            // not the generic 500 every other compute error gets. Same
            // reasoning for ReverseStress's "threshold unreachable within
            // bounds" case.
            // The compute layer's own message names the specific
            // baseline/snapshot id problem, which isn't actionable for an
            // end user -- surface a fixed, user-facing instruction instead.
            compute::ComputeError::NoPriorSnapshot(_) => BackendError::Unrecognised(
                "No prior risk snapshot found for this portfolio. Run a risk analysis first \
                 before checking drift."
                    .to_string(),
            ),
            compute::ComputeError::ReverseStressInfeasible(message) => {
                BackendError::Unrecognised(message)
            }
            // A ticker that still 404s after `ticker_map::resolve_ticker`'s
            // resolution order is the caller's problem (bad/delisted
            // symbol in the uploaded portfolio), not this service's -- 422,
            // same reasoning as the two cases above. Its message is already
            // user-friendly (see `data::load_series_resolved`), so it's
            // passed through as-is.
            compute::ComputeError::UnresolvedTicker(message) => {
                BackendError::UnresolvedTicker(message)
            }
            compute::ComputeError::DataUnavailable(message) => {
                BackendError::DataUnavailable(message)
            }
            // Likewise a request problem (the portfolio/time window the
            // caller chose), not an internal failure -- the compute-layer
            // message is wrapped with user-facing framing and a concrete
            // suggestion.
            compute::ComputeError::InsufficientData(message) => {
                let message = message.trim_end_matches('.');
                BackendError::InsufficientData(format!(
                    "One or more holdings don't have enough price history for analysis. {message}. \
                     Try removing recently listed stocks from your portfolio."
                ))
            }
            other => BackendError::Compute(other.to_string()),
        }
    }
}

/// What the route handlers depend on. `RealBackend` wraps live
/// compute + Gemini calls; tests use a `MockBackend` (see
/// `tests/support`) with canned responses.
#[async_trait::async_trait]
pub trait Backend: Send + Sync {
    /// `portfolio` is passed alongside `experiment` (not just embedded in
    /// its input, as `FactorShockInput`/etc. all do) because `RiskDrift`'s
    /// own input carries no `portfolio` field of its own -- it diffs the
    /// current portfolio against a *stored* baseline, not two portfolios
    /// given inline. `policy` is the request's separate top-level `policy`
    /// field (see `routes::ExperimentRequest`/`AskRequest`), not anything
    /// embedded in `experiment` itself -- it drives the passive policy
    /// check every experiment type except `PolicyCheck`/`CvarRebalance`
    /// gets attached to its trace (see `compute::dispatch::run_experiment`).
    async fn run_experiment(
        &self,
        experiment: Experiment,
        portfolio: Portfolio,
        policy: Option<RiskPolicy>,
    ) -> Result<EvidenceTrace, BackendError>;
    async fn run_ask(
        &self,
        portfolio: Portfolio,
        message: String,
        conversation_history: Vec<ConversationTurn>,
        policy: Option<RiskPolicy>,
    ) -> Result<PipelineResult, BackendError>;
}

pub struct RealBackend {
    gemini: agent::gemini::HttpGeminiClient,
    store: std::sync::Arc<store::SnapshotStore>,
}

/// Backend used when GEMINI_API_KEY is not set. Compute experiments work
/// fully; /ask returns a clear 503 explaining how to enable AI features.
pub struct NoAiBackend {
    pub store: std::sync::Arc<store::SnapshotStore>,
}

impl RealBackend {
    pub fn new(
        gemini: agent::gemini::HttpGeminiClient,
        store: std::sync::Arc<store::SnapshotStore>,
    ) -> Self {
        RealBackend { gemini, store }
    }
}

#[async_trait::async_trait]
impl Backend for NoAiBackend {
    async fn run_experiment(
        &self,
        experiment: Experiment,
        portfolio: Portfolio,
        policy: Option<RiskPolicy>,
    ) -> Result<EvidenceTrace, BackendError> {
        let holdings: Vec<(String, f64)> = portfolio
            .holdings
            .iter()
            .map(|h| (h.ticker.clone(), h.weight))
            .collect();
        let ctx = compute::context::ExperimentContext {
            store: self.store.clone(),
            portfolio_hash: compute::portfolio::portfolio_hash(&holdings),
            policy,
        };
        let trace = tokio::task::spawn_blocking(move || {
            agent::pipeline::compute_trace(&experiment, &portfolio, &ctx)
        })
        .await
        .map_err(|e| BackendError::Internal(format!("compute task panicked: {e}")))??;
        Ok(trace)
    }

    async fn run_ask(
        &self,
        _portfolio: Portfolio,
        _message: String,
        _conversation_history: Vec<ConversationTurn>,
        _policy: Option<RiskPolicy>,
    ) -> Result<PipelineResult, BackendError> {
        Err(BackendError::GeminiUnavailable(
            "AI features require GEMINI_API_KEY — set the environment variable and restart"
                .to_string(),
        ))
    }
}

#[async_trait::async_trait]
impl Backend for RealBackend {
    async fn run_experiment(
        &self,
        experiment: Experiment,
        portfolio: Portfolio,
        policy: Option<RiskPolicy>,
    ) -> Result<EvidenceTrace, BackendError> {
        let holdings: Vec<(String, f64)> = portfolio
            .holdings
            .iter()
            .map(|h| (h.ticker.clone(), h.weight))
            .collect();
        let ctx = compute::context::ExperimentContext {
            store: self.store.clone(),
            portfolio_hash: compute::portfolio::portfolio_hash(&holdings),
            policy,
        };
        let trace = tokio::task::spawn_blocking(move || {
            agent::pipeline::compute_trace(&experiment, &portfolio, &ctx)
        })
        .await
        .map_err(|e| BackendError::Internal(format!("compute task panicked: {e}")))??;
        Ok(trace)
    }

    async fn run_ask(
        &self,
        portfolio: Portfolio,
        message: String,
        conversation_history: Vec<ConversationTurn>,
        policy: Option<RiskPolicy>,
    ) -> Result<PipelineResult, BackendError> {
        let holdings: Vec<(String, f64)> = portfolio
            .holdings
            .iter()
            .map(|h| (h.ticker.clone(), h.weight))
            .collect();
        let ctx = compute::context::ExperimentContext {
            store: self.store.clone(),
            portfolio_hash: compute::portfolio::portfolio_hash(&holdings),
            policy,
        };
        let result = agent::pipeline::run(
            &self.gemini,
            &message,
            portfolio,
            &conversation_history,
            &ctx,
        )
        .await?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_prior_snapshot_maps_to_a_fixed_user_facing_422_message() {
        let err: BackendError = compute::ComputeError::NoPriorSnapshot(
            "no stored snapshot found for id \"abc\"".to_string(),
        )
        .into();
        match err {
            BackendError::Unrecognised(msg) => {
                assert_eq!(
                    msg,
                    "No prior risk snapshot found for this portfolio. Run a risk analysis first \
                     before checking drift."
                );
            }
            other => panic!("expected BackendError::Unrecognised, got {other:?}"),
        }
    }

    #[test]
    fn unresolved_ticker_message_is_passed_through_as_is() {
        let original = "Could not fetch market data for ticker 'FOO.NS' (tried 'FOO.NS'). \
                         This symbol may be delisted, suspended, or use a different name on \
                         Yahoo Finance. Please remove it from your portfolio or replace it with \
                         the correct NSE symbol."
            .to_string();
        let err: BackendError = compute::ComputeError::UnresolvedTicker(original.clone()).into();
        match err {
            BackendError::UnresolvedTicker(msg) => assert_eq!(msg, original),
            other => panic!("expected BackendError::UnresolvedTicker, got {other:?}"),
        }
    }

    #[test]
    fn insufficient_data_is_wrapped_with_user_facing_framing_and_a_suggestion() {
        let err: BackendError = compute::ComputeError::InsufficientData(
            "ticker FOO.NS has only 20 return observations".to_string(),
        )
        .into();
        match err {
            BackendError::InsufficientData(msg) => {
                assert!(msg.starts_with(
                    "One or more holdings don't have enough price history for analysis."
                ));
                assert!(msg.contains("ticker FOO.NS has only 20 return observations"));
                assert!(msg.contains("Try removing recently listed stocks from your portfolio."));
            }
            other => panic!("expected BackendError::InsufficientData, got {other:?}"),
        }
    }

    #[test]
    fn data_unavailable_maps_to_its_own_variant() {
        let err: BackendError = compute::ComputeError::DataUnavailable("x".to_string()).into();
        assert!(matches!(err, BackendError::DataUnavailable(m) if m == "x"));
    }

    #[test]
    fn exhausted_gemini_retries_map_to_gemini_unavailable() {
        let err: BackendError = GeminiError::Unavailable {
            attempts: 6,
            last_error: "status 503".to_string(),
        }
        .into();
        assert!(matches!(err, BackendError::GeminiUnavailable(_)));
    }

    #[test]
    fn every_other_compute_error_still_maps_to_the_generic_500_compute_variant() {
        let err: BackendError =
            compute::ComputeError::Data("some internal detail".to_string()).into();
        match err {
            BackendError::Compute(_) => {}
            other => panic!("expected BackendError::Compute, got {other:?}"),
        }
    }
}
