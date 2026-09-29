//! Evidence Trace: the single structured record every experiment returns.
//! Rule (per spec): every number the LLM layer could later cite must exist
//! as a named field somewhere in this struct, not just be printed.

use chrono::NaiveDate;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::DataQuality;
use crate::model::Frequency;
use crate::policy::PolicyResult;
use crate::regime::RegimeState;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DataWindow {
    pub frequency: Frequency,
    pub window_periods: usize,
    pub start: NaiveDate,
    pub end: NaiveDate,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModelParams {
    pub frequency: Frequency,
    pub window_periods: usize,
    pub factor_names: Vec<String>,
    pub shrinkage_intensity: f64,
    pub annualization_factor: f64,
    /// Always populated (regime-conditioning is unconditional -- see
    /// `model::fit_factor_model`); `None` only for experiment types that
    /// fit no factor model at all and whose own standalone regime fit also
    /// failed (see `performance::run_portfolio_performance`).
    pub regime_state: Option<RegimeState>,
    /// Non-empty only when `regime_state` is `Some` and at least one
    /// regime had fewer than `model::MIN_REGIME_OBSERVATIONS` observations
    /// in the fitted window (see `model::fit_factor_model`).
    pub regime_fallback_warnings: Vec<String>,
    /// `Some` only for `CvarRebalance`: `"user-specified"` if the caller
    /// gave `per_name_cap`, `"server-default-0.20"` if it was defaulted
    /// (see `cvar::CvarRebalanceInput::per_name_cap`). `None` for
    /// `FactorShock`/`RiskDecomposition`, which have no such field.
    pub cap_source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct InvariantCheck {
    pub name: String,
    pub passed: bool,
    pub tolerance: f64,
    pub detail: String,
}

impl InvariantCheck {
    pub fn approx_eq(name: &str, lhs: f64, rhs: f64, tolerance: f64) -> Self {
        let diff = (lhs - rhs).abs();
        InvariantCheck {
            name: name.to_string(),
            passed: diff <= tolerance,
            tolerance,
            detail: format!("lhs={lhs:.12} rhs={rhs:.12} abs_diff={diff:.3e}"),
        }
    }
}

/// Key params from a `RiskDrift` baseline snapshot's own trace, carried
/// alongside the current fit's `model_params` so a `RiskDrift` trace is
/// self-contained (a reader doesn't have to separately fetch the baseline
/// snapshot to know what it was fit against).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BaselineModelParams {
    pub window_periods: usize,
    pub frequency: Frequency,
    pub shrinkage_intensity: f64,
    pub regime_label: Option<String>,
}

/// Provenance for a `FactorShock` run driven by a named historical
/// scenario preset (the "historical_stress" orchestrator tool), rather
/// than caller-specified shocks. Set by `agent::orchestrator` after the
/// fact -- `compute::experiments::run_factor_shock` itself has no concept
/// of "this shock came from a named scenario," it just receives raw
/// `shocks_pct`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ScenarioProvenance {
    pub scenario_id: String,
    pub scenario_name: String,
    // Owned `String`s (not `&'static str`, unlike `scenarios::HistoricalScenario`
    // itself): this is a snapshot copied onto a trace, which needs to
    // `#[derive(Deserialize)]` (round-tripping through `SnapshotStore`) --
    // a `&'static str` field can't derive `Deserialize` (see
    // `regime::RegimeState`'s hand-written impl for the same reason; this
    // struct just sidesteps it by not needing `'static` at all).
    pub date_range: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EvidenceTrace {
    /// UUID v4, generated fresh for every trace at construction time.
    /// `parent_trace_ids` (below) and `store::RiskSnapshot`/`agent::
    /// execution_trace::ToolResult::trace_id` all refer to this.
    #[serde(default)]
    pub id: String,
    pub experiment: String,
    pub inputs: serde_json::Value,
    pub data_window: DataWindow,
    pub data_quality: DataQuality,
    pub model_params: ModelParams,
    pub outputs: serde_json::Value,
    pub invariants: Vec<InvariantCheck>,
    pub engine_version: String,
    /// The build's git commit (`VERGEN_GIT_SHA`, embedded by `build.rs` via
    /// the `vergen` crate), or `"unknown"` if unavailable (e.g. a source
    /// tarball with no `.git` directory) -- see `engine_commit()`.
    #[serde(default)]
    pub engine_commit: String,
    /// The most recent date in this trace's aligned return series
    /// (`data_window.end`), as an ISO-8601 UTC timestamp (synthetic
    /// midnight -- the underlying data is date-granularity, not
    /// intraday).
    #[serde(default)]
    pub data_as_of: String,
    /// `Some` only for a `FactorShock` trace produced by the
    /// "historical_stress" orchestrator tool (a named scenario preset,
    /// not caller-specified shocks).
    #[serde(default)]
    pub scenario_provenance: Option<ScenarioProvenance>,
    /// For a multi-tool orchestrator run: the `id`s of traces that
    /// preceded this one in the same investigation (e.g. `current_risk`'s
    /// trace id, on `risk_drift`'s trace, when the plan chained the two).
    /// Empty for a standalone `/experiment` call or a single-tool `/ask`.
    #[serde(default)]
    pub parent_trace_ids: Vec<String>,
    /// `Some` only for `RiskDrift` (see `BaselineModelParams`); `None` for
    /// every other experiment type.
    #[serde(default)]
    pub baseline_model_params: Option<BaselineModelParams>,
    /// A passive policy check run alongside *any* experiment when the
    /// request carries a top-level `policy` field (see
    /// `dispatch::run_experiment`'s doc) -- `None` unless one was
    /// attached. `PolicyCheck`'s own result lives in `outputs.result`
    /// instead (it's the experiment, not a side effect of one), so this is
    /// always `None` for a `PolicyCheck` trace.
    #[serde(default)]
    pub policy_result: Option<PolicyResult>,
}

pub fn engine_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// The build's git commit -- `option_env!` (not `env!`) so a build
/// environment where `vergen`'s `.emit()` itself failed (see `build.rs`,
/// which discards that error rather than failing the build) still
/// compiles, falling back to `"unknown"` at runtime instead.
pub fn engine_commit() -> String {
    // vergen's actual failure mode (no `.git` directory available at
    // build time -- e.g. Cloud Build's source upload excludes it, or a
    // source tarball with none) isn't an *absent* env var, as this
    // function originally assumed: by default (without vergen's own
    // `.fail_on_error()`), it still sets `VERGEN_GIT_SHA`, just to its own
    // literal placeholder string, so `option_env!` sees a *present* var
    // that happens to be that sentinel -- caught live testing a real
    // Cloud Run deploy, where `.git` genuinely isn't in the build context.
    match option_env!("VERGEN_GIT_SHA") {
        Some(sha) if sha != "VERGEN_IDEMPOTENT_OUTPUT" => sha.to_string(),
        _ => "unknown".to_string(),
    }
}

/// A fresh UUID v4 for a new `EvidenceTrace::id`.
pub fn new_trace_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// `data_window.end` (the most recent date in the aligned return series)
/// as a synthetic-midnight ISO-8601 UTC timestamp -- see
/// `EvidenceTrace::data_as_of`'s doc.
pub fn data_as_of(data_window: &DataWindow) -> String {
    format!("{}T00:00:00Z", data_window.end)
}
