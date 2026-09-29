//! Shared context threaded into `dispatch::run_experiment` for experiment
//! types that need more than their own input to run. `RiskDrift` uses
//! `store` (read access, to resolve its baseline); `PolicyCheck` and
//! `CvarRebalance` use `policy`; every other experiment type receives the
//! whole thing but ignores it.

use std::sync::Arc;

use store::SnapshotStore;

use crate::policy::RiskPolicy;

#[derive(Clone)]
pub struct ExperimentContext {
    pub store: Arc<SnapshotStore>,
    /// The current request's portfolio, hashed the same way
    /// `store::RiskSnapshot::portfolio_hash` is (see
    /// `crate::portfolio::portfolio_hash`) -- computed once by the caller
    /// (`server::backend`) and reused here rather than re-hashed per
    /// experiment.
    pub portfolio_hash: String,
    /// An optional policy attached to the request (`POST /experiment`'s or
    /// `POST /ask`'s top-level `policy` field), separate from the
    /// experiment itself so *any* experiment can be accompanied by a
    /// passive policy check (see `dispatch::run_experiment`'s doc).
    pub policy: Option<RiskPolicy>,
}
