//! RiskDrift: compares current portfolio risk against a prior
//! `RiskDecomposition` snapshot, explaining what changed (volatility,
//! factor contributions, portfolio betas, factor correlations, market
//! regime, specific-risk share) between the two points in time.
//!
//! Needs `RiskDecompositionOutput::portfolio_betas`/`factor_correlation`
//! (added this session) to diff against a stored baseline -- neither was
//! previously persisted anywhere in the trace, so a baseline snapshot
//! created *before* this change can't serve as a `RiskDrift` baseline.
//! `run_risk_drift` detects that (empty `portfolio_betas`) and returns a
//! clear error rather than silently producing zeroed deltas.

use std::collections::BTreeMap;
use std::path::Path;

use chrono::NaiveDate;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::context::ExperimentContext;
use crate::error::{ComputeError, Result};
use crate::experiments::{CorrelationMatrix, Portfolio, RiskDecompositionInput};
use crate::model::{fit_factor_model, Frequency, ModelConfig};
use crate::trace::{BaselineModelParams, DataWindow, EvidenceTrace};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RiskDriftInput {
    /// Snapshot id to diff the current portfolio against. Omit (or pass
    /// `null`) to use the most recent snapshot stored for this portfolio.
    #[serde(default)]
    pub baseline_snapshot_id: Option<String>,
    /// Model window in periods; defaults to `frequency.default_window()`.
    #[serde(default)]
    pub window: Option<usize>,
    /// `"daily"` or `"weekly"`, case-insensitive; defaults to `"daily"`.
    #[serde(default)]
    pub frequency: Option<String>,
}

impl RiskDriftInput {
    fn resolved_frequency(&self) -> Result<Frequency> {
        Frequency::from_optional_str(self.frequency.as_deref())
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct RiskDriftOutputs {
    pub baseline_snapshot_id: String,
    pub baseline_created_at: String,
    pub current_as_of: NaiveDate,
    /// Calendar days between the baseline's and the current fit's data
    /// window end dates (i.e. how much the underlying market data advanced,
    /// not wall-clock time since the baseline was computed).
    pub days_elapsed: i64,

    pub vol_before: f64,
    pub vol_after: f64,
    pub vol_change_abs: f64,
    pub vol_change_pct: f64,

    pub factor_contributions_before: BTreeMap<String, f64>,
    pub factor_contributions_after: BTreeMap<String, f64>,
    /// `after - before`; positive = that factor got riskier.
    pub factor_contribution_delta: BTreeMap<String, f64>,
    pub largest_contribution_increase: String,
    pub largest_contribution_decrease: String,

    pub portfolio_betas_before: BTreeMap<String, f64>,
    pub portfolio_betas_after: BTreeMap<String, f64>,
    pub portfolio_beta_delta: BTreeMap<String, f64>,

    /// Upper triangle only, keyed `"FACTOR_A:FACTOR_B"`.
    pub correlation_before: BTreeMap<String, f64>,
    pub correlation_after: BTreeMap<String, f64>,
    pub correlation_delta: BTreeMap<String, f64>,
    pub largest_correlation_change: String,

    pub regime_before: String,
    pub regime_after: String,
    pub regime_changed: bool,
    pub smoothed_probs_before: [f64; 3],
    pub smoothed_probs_after: [f64; 3],

    pub specific_risk_share_before: f64,
    pub specific_risk_share_after: f64,
    pub specific_risk_share_delta: f64,

    pub vol_increased: bool,
    /// True for Bull->Bear, Bear->Crisis, or Bull->Crisis.
    pub regime_worsened: bool,
    pub top_growing_risk_factor: String,
    /// True if the largest single-factor contribution share
    /// (`fraction_of_vol`) increased by more than 5 percentage points.
    pub risk_became_more_concentrated: bool,
}

/// Fetches data, fits the current factor model, runs `RiskDecomposition`
/// for "after", resolves the baseline snapshot via `ctx.store`, and
/// delegates the actual diff to `compute_risk_drift` (the hermetic,
/// directly-testable part -- see its doc).
pub fn run_risk_drift(
    cache_dir: &Path,
    portfolio: &Portfolio,
    input: &RiskDriftInput,
    ctx: &ExperimentContext,
) -> Result<(RiskDriftOutputs, EvidenceTrace)> {
    let frequency = input.resolved_frequency()?;
    let window = input.window.unwrap_or_else(|| frequency.default_window());

    let baseline_snapshot = resolve_baseline(input, ctx)?;

    let tickers = portfolio.tickers();
    let data = crate::data::load_market_data(cache_dir, &tickers, false, frequency)?;
    let model = fit_factor_model(&data, &tickers, ModelConfig::new(window, frequency))?;
    let data_window = DataWindow {
        frequency,
        window_periods: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    };
    let current_input = RiskDecompositionInput {
        portfolio: portfolio.clone(),
        frequency,
        window: Some(window),
    };
    let (current_output, current_trace) = crate::experiments::run_risk_decomposition(
        &data.quality,
        data_window,
        &model,
        &current_input,
    )?;

    compute_risk_drift(&baseline_snapshot, current_output, current_trace, input)
}

/// The actual diff, given the baseline snapshot and an already-computed
/// "current" `RiskDecomposition` result -- no data fetch, no store access,
/// so this is what the compute-layer tests exercise directly (hermetic,
/// synthetic-data, matching the `run_factor_shock`/`run_risk_decomposition`
/// pattern of taking pre-fitted data/model rather than fetching their own).
pub fn compute_risk_drift(
    baseline_snapshot: &store::RiskSnapshot,
    current_output: crate::experiments::RiskDecompositionOutput,
    current_trace: EvidenceTrace,
    input: &RiskDriftInput,
) -> Result<(RiskDriftOutputs, EvidenceTrace)> {
    let baseline_trace: EvidenceTrace = serde_json::from_str(&baseline_snapshot.trace_json)?;
    if baseline_trace.experiment != "RiskDecomposition" {
        return Err(ComputeError::InvalidInput(format!(
            "baseline snapshot must be a RiskDecomposition result, got {}",
            baseline_trace.experiment
        )));
    }

    // --- "before" extraction from the baseline trace's stored JSON ---
    let baseline_result = baseline_trace
        .outputs
        .get("result")
        .ok_or_else(|| ComputeError::InvalidInput("baseline trace has no outputs.result".to_string()))?;
    let vol_before = required_f64(baseline_result, "portfolio_vol_annualized")?;
    let factor_contributions_before = factor_field_map(baseline_result, "contribution")?;
    let fraction_of_vol_before = factor_field_map(baseline_result, "fraction_of_vol")?;
    let portfolio_betas_before = betas_map(baseline_result.get("portfolio_betas"));
    let correlation_before = correlation_map_from_json(baseline_result.get("factor_correlation"));
    let specific_risk_share_before = required_f64(baseline_result, "specific_risk_fraction_of_vol")?;

    if portfolio_betas_before.is_empty() || correlation_before.is_empty() {
        return Err(ComputeError::InvalidInput(format!(
            "baseline snapshot (engine_version {}) predates portfolio_betas/factor_correlation \
             support; rerun RiskDecomposition to create a compatible baseline",
            baseline_snapshot.engine_version
        )));
    }

    let baseline_regime = baseline_trace
        .model_params
        .regime_state
        .clone()
        .ok_or_else(|| ComputeError::InvalidInput("baseline trace has no regime_state".to_string()))?;
    let current_regime = current_trace
        .model_params
        .regime_state
        .clone()
        .ok_or_else(|| ComputeError::InvalidInput("current trace has no regime_state".to_string()))?;

    // --- "after" values, straight from the typed current output ---
    let vol_after = current_output.portfolio_vol_annualized;
    let factor_contributions_after: BTreeMap<String, f64> =
        current_output.by_factor.iter().map(|f| (f.factor.clone(), f.contribution)).collect();
    let fraction_of_vol_after: BTreeMap<String, f64> =
        current_output.by_factor.iter().map(|f| (f.factor.clone(), f.fraction_of_vol)).collect();
    let portfolio_betas_after = current_output.portfolio_betas.clone();
    let correlation_after = correlation_map_from_matrix(&current_output.factor_correlation);
    let specific_risk_share_after = current_output.specific_risk_fraction_of_vol;
    let regime_before = baseline_regime.current_label.to_string();
    let regime_after = current_regime.current_label.to_string();

    // --- Deltas ---
    let vol_change_abs = vol_after - vol_before;
    let vol_change_pct = if vol_before != 0.0 { (vol_after / vol_before - 1.0) * 100.0 } else { 0.0 };

    let factor_contribution_delta = delta_map(&factor_contributions_before, &factor_contributions_after);
    let (largest_contribution_increase, largest_contribution_decrease) =
        largest_and_smallest(&factor_contribution_delta)?;

    let portfolio_beta_delta = delta_map(&portfolio_betas_before, &portfolio_betas_after);
    let correlation_delta = delta_map(&correlation_before, &correlation_after);
    let largest_correlation_change = largest_by_abs(&correlation_delta).unwrap_or_default();

    let specific_risk_share_delta = specific_risk_share_after - specific_risk_share_before;

    let max_share_before = fraction_of_vol_before.values().copied().fold(f64::MIN, f64::max);
    let max_share_after = fraction_of_vol_after.values().copied().fold(f64::MIN, f64::max);
    let risk_became_more_concentrated = (max_share_after - max_share_before) > 0.05;

    let regime_changed = regime_before != regime_after;
    let regime_worsened = match (regime_severity(&regime_before), regime_severity(&regime_after)) {
        (Some(b), Some(a)) => a > b,
        _ => false,
    };

    let days_elapsed = (current_trace.data_window.end - baseline_trace.data_window.end).num_days();

    let output = RiskDriftOutputs {
        baseline_snapshot_id: baseline_snapshot.id.clone(),
        baseline_created_at: baseline_snapshot.created_at.clone(),
        current_as_of: current_trace.data_window.end,
        days_elapsed,
        vol_before,
        vol_after,
        vol_change_abs,
        vol_change_pct,
        factor_contributions_before,
        factor_contributions_after,
        factor_contribution_delta,
        largest_contribution_increase: largest_contribution_increase.clone(),
        largest_contribution_decrease,
        portfolio_betas_before,
        portfolio_betas_after,
        portfolio_beta_delta,
        correlation_before,
        correlation_after,
        correlation_delta,
        largest_correlation_change,
        regime_before: regime_before.clone(),
        regime_after: regime_after.clone(),
        regime_changed,
        smoothed_probs_before: baseline_regime.smoothed_probs,
        smoothed_probs_after: current_regime.smoothed_probs,
        specific_risk_share_before,
        specific_risk_share_after,
        specific_risk_share_delta,
        vol_increased: vol_after > vol_before,
        regime_worsened,
        top_growing_risk_factor: largest_contribution_increase,
        risk_became_more_concentrated,
    };

    let euler_sum: f64 = output.factor_contribution_delta.values().sum();
    let invariants = vec![
        crate::trace::InvariantCheck::approx_eq(
            "sum(factor_contribution_delta) ~= vol_change_abs (approximate: Euler additivity \
             across two different fits is not exact)",
            euler_sum,
            vol_change_abs,
            1e-6,
        ),
        crate::trace::InvariantCheck {
            name: "regime_changed == (regime_before != regime_after)".to_string(),
            passed: regime_changed == (output.regime_before != output.regime_after),
            tolerance: 0.0,
            detail: format!("regime_before={regime_before} regime_after={regime_after} regime_changed={regime_changed}"),
        },
    ];

    let baseline_model_params = Some(BaselineModelParams {
        window_periods: baseline_trace.model_params.window_periods,
        frequency: baseline_trace.model_params.frequency,
        shrinkage_intensity: baseline_trace.model_params.shrinkage_intensity,
        regime_label: baseline_trace.model_params.regime_state.as_ref().map(|r| r.current_label.to_string()),
    });

    let trace = EvidenceTrace {
        id: crate::trace::new_trace_id(),
        experiment: "RiskDrift".to_string(),
        inputs: serde_json::to_value(input)?,
        data_as_of: current_trace.data_as_of.clone(),
        data_window: current_trace.data_window,
        data_quality: current_trace.data_quality,
        model_params: current_trace.model_params,
        outputs: serde_json::json!({ "result": output }),
        invariants,
        engine_version: crate::trace::engine_version(),
        engine_commit: crate::trace::engine_commit(),
        scenario_provenance: None,
        // RiskDrift always directly depends on its baseline snapshot's own
        // trace -- record that dependency, matching the field's own doc
        // ("IDs of traces that preceded this one in the same
        // investigation").
        parent_trace_ids: vec![baseline_snapshot.id.clone()],
        baseline_model_params,
        policy_result: None,
    };

    Ok((output, trace))
}

/// Resolves the baseline snapshot per `RiskDriftInput::baseline_snapshot_id`.
///
/// **Judgment call**: when `baseline_snapshot_id` is `None`, the most
/// recent stored snapshot for this portfolio (`latest_for_portfolio(hash,
/// 1)[0]`) is used directly as the baseline. At the point this resolves,
/// `RiskDrift`'s own result has not yet been stored (that happens
/// afterward, via the server's generic snapshot-insertion path -- see the
/// README), so nothing in the store at this moment is "the current
/// request" to skip past. Using the *second* most recent entry instead
/// would require two prior snapshots to already exist before `RiskDrift`
/// could run even once, which would break the documented live-run flow (a
/// single prior `RiskDecomposition` snapshot must be usable immediately as
/// a baseline for the very next `RiskDrift` call).
fn resolve_baseline(
    input: &RiskDriftInput,
    ctx: &ExperimentContext,
) -> Result<store::RiskSnapshot> {
    match &input.baseline_snapshot_id {
        Some(id) => {
            let snapshot = ctx
                .store
                .get(id)?
                .ok_or_else(|| ComputeError::NoPriorSnapshot(format!("no stored snapshot found for id {id:?}")))?;
            if snapshot.portfolio_hash != ctx.portfolio_hash {
                return Err(ComputeError::InvalidInput(format!(
                    "snapshot {id} belongs to a different portfolio than the current request"
                )));
            }
            Ok(snapshot)
        }
        None => {
            let recent = ctx.store.latest_for_portfolio(&ctx.portfolio_hash, 1)?;
            recent.into_iter().next().ok_or_else(|| {
                ComputeError::NoPriorSnapshot(
                    "No prior snapshot found for this portfolio. Run a RiskDecomposition first \
                     to establish a baseline."
                        .to_string(),
                )
            })
        }
    }
}

fn required_f64(value: &Value, field: &str) -> Result<f64> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| ComputeError::InvalidInput(format!("baseline trace is missing {field:?}")))
}

/// Extracts `{factor: entry[field]}` from a `RiskDecompositionOutput`-shaped
/// `result` JSON value's `by_factor` array.
fn factor_field_map(result: &Value, field: &str) -> Result<BTreeMap<String, f64>> {
    let arr = result
        .get("by_factor")
        .and_then(Value::as_array)
        .ok_or_else(|| ComputeError::InvalidInput("baseline trace is missing by_factor".to_string()))?;
    arr.iter()
        .map(|entry| {
            let factor = entry
                .get("factor")
                .and_then(Value::as_str)
                .ok_or_else(|| ComputeError::InvalidInput("by_factor entry missing \"factor\"".to_string()))?;
            let value = entry
                .get(field)
                .and_then(Value::as_f64)
                .ok_or_else(|| ComputeError::InvalidInput(format!("by_factor entry missing {field:?}")))?;
            Ok((factor.to_string(), value))
        })
        .collect()
}

/// Absent (predates this field) -> empty map, treated as "incompatible
/// baseline" by the caller.
fn betas_map(value: Option<&Value>) -> BTreeMap<String, f64> {
    value
        .and_then(Value::as_object)
        .map(|obj| obj.iter().filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f))).collect())
        .unwrap_or_default()
}

fn correlation_map_from_json(value: Option<&Value>) -> BTreeMap<String, f64> {
    let Some(value) = value else { return BTreeMap::new() };
    let names: Vec<String> = value
        .get("factor_names")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let Some(rows) = value.get("rows").and_then(Value::as_array) else {
        return BTreeMap::new();
    };
    let mut map = BTreeMap::new();
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            if let Some(v) = rows.get(i).and_then(Value::as_array).and_then(|r| r.get(j)).and_then(Value::as_f64) {
                map.insert(format!("{}:{}", names[i], names[j]), v);
            }
        }
    }
    map
}

fn correlation_map_from_matrix(m: &CorrelationMatrix) -> BTreeMap<String, f64> {
    let mut map = BTreeMap::new();
    let n = m.factor_names.len();
    for i in 0..n {
        for j in (i + 1)..n {
            map.insert(format!("{}:{}", m.factor_names[i], m.factor_names[j]), m.rows[i][j]);
        }
    }
    map
}

fn delta_map(before: &BTreeMap<String, f64>, after: &BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    let mut keys: std::collections::BTreeSet<&String> = before.keys().collect();
    keys.extend(after.keys());
    keys.into_iter()
        .map(|k| {
            let b = before.get(k).copied().unwrap_or(0.0);
            let a = after.get(k).copied().unwrap_or(0.0);
            (k.clone(), a - b)
        })
        .collect()
}

fn largest_and_smallest(delta: &BTreeMap<String, f64>) -> Result<(String, String)> {
    let largest = delta.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(k, _)| k.clone());
    let smallest = delta.iter().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(k, _)| k.clone());
    match (largest, smallest) {
        (Some(l), Some(s)) => Ok((l, s)),
        _ => Err(ComputeError::InvalidInput("no factor contributions to compare".to_string())),
    }
}

fn largest_by_abs(delta: &BTreeMap<String, f64>) -> Option<String> {
    delta.iter().max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap()).map(|(k, _)| k.clone())
}

fn regime_severity(label: &str) -> Option<usize> {
    crate::regime::REGIME_LABELS.iter().position(|&l| l == label)
}
