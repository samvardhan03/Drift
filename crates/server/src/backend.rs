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
            agent::pipeline::PipelineError::Compute(c) => BackendError::Compute(c.to_string()),
            agent::pipeline::PipelineError::Narrate(n) => BackendError::from(n),
            agent::pipeline::PipelineError::Suggest(s) => BackendError::from(s),
            agent::pipeline::PipelineError::AllToolsFailed(errors) => {
                BackendError::Compute(format!("every planned tool failed: {errors:?}"))
            }
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
            compute::ComputeError::NoPriorSnapshot(message) => BackendError::Unrecognised(message),
            compute::ComputeError::ReverseStressInfeasible(message) => BackendError::Unrecognised(message),
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

impl RealBackend {
    pub fn new(gemini: agent::gemini::HttpGeminiClient, store: std::sync::Arc<store::SnapshotStore>) -> Self {
        RealBackend { gemini, store }
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
        let holdings: Vec<(String, f64)> =
            portfolio.holdings.iter().map(|h| (h.ticker.clone(), h.weight)).collect();
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
        let holdings: Vec<(String, f64)> =
            portfolio.holdings.iter().map(|h| (h.ticker.clone(), h.weight)).collect();
        let ctx = compute::context::ExperimentContext {
            store: self.store.clone(),
            portfolio_hash: compute::portfolio::portfolio_hash(&holdings),
            policy,
        };
        let result =
            agent::pipeline::run(&self.gemini, &message, portfolio, &conversation_history, &ctx).await?;
        Ok(result)
    }
}
