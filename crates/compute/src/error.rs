use thiserror::Error;

#[derive(Debug, Error)]
pub enum ComputeError {
    #[error("http request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("data error: {0}")]
    Data(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("snapshot store error: {0}")]
    Store(#[from] store::StoreError),

    /// `RiskDrift` couldn't resolve a baseline: no prior snapshot exists for
    /// this portfolio (or a specific `baseline_snapshot_id` was given but
    /// not found / belongs to a different portfolio). Distinct from
    /// `InvalidInput` so callers (e.g. `server::backend`) can map it to its
    /// own HTTP status instead of a generic 500.
    #[error("{0}")]
    NoPriorSnapshot(String),

    /// `ReverseStress`'s requested `loss_threshold_inr` cannot be breached
    /// within `factor_bounds` (the max-loss corner of the box doesn't reach
    /// it, per the linearised pre-solve feasibility check). Distinct from
    /// `InvalidInput` for the same reason as `NoPriorSnapshot` -- a request
    /// problem, not an internal failure.
    #[error("{0}")]
    ReverseStressInfeasible(String),

    /// A ticker still 404'd from Yahoo Finance after `ticker_map::resolve_ticker`
    /// ran its full resolution order -- a request problem (the symbol is
    /// delisted, suspended, or genuinely unknown), not an internal failure.
    /// Distinct from `Data` for the same reason as `NoPriorSnapshot`: callers
    /// map it to its own HTTP status (422) instead of a generic 500.
    #[error("{0}")]
    UnresolvedTicker(String),

    /// Not enough return observations to fit the factor model reliably --
    /// either a single ticker has fewer than `model::MIN_TICKER_OBSERVATIONS`
    /// return observations (too recently listed), or the shared factor-model
    /// window itself (bounded by how much factor history exists) has fallen
    /// below `model::MIN_MODEL_WINDOW`. A request problem (the caller's
    /// portfolio/time window), not an internal failure -- maps to 422.
    #[error("{0}")]
    InsufficientData(String),

    /// Yahoo Finance couldn't be reached in time (timeout, connection
    /// failure, or a 429/5xx response) -- transient, and this service's
    /// upstream problem rather than the caller's. Maps to 503.
    #[error("{0}")]
    DataUnavailable(String),
}

pub type Result<T> = std::result::Result<T, ComputeError>;
