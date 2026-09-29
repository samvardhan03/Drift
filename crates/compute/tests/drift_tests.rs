//! Hermetic tests for `compute::drift::compute_risk_drift` (the pure diff
//! logic, factored out of `run_risk_drift` specifically so it doesn't need
//! network access or a populated store -- see its doc comment).

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::NaiveDate;
use compute::context::ExperimentContext;
use compute::data::DataQuality;
use compute::drift::{compute_risk_drift, run_risk_drift, RiskDriftInput};
use compute::experiments::{CorrelationMatrix, FactorContribution, Holding, Portfolio, RiskDecompositionOutput};
use compute::model::Frequency;
use compute::regime::RegimeState;
use compute::trace::{DataWindow, EvidenceTrace, ModelParams};
use compute::ComputeError;
use store::{RiskSnapshot, SnapshotStore};

fn sample_regime_state(idx: u8, label: &'static str, probs: [f64; 3]) -> RegimeState {
    RegimeState {
        current_regime: idx,
        current_label: label,
        smoothed_probs: probs,
        viterbi_sequence: vec![idx; 10],
        obs_count_per_regime: [10, 0, 0],
        log_likelihood: -1.0,
        n_iter: 5,
        smoothing_note: "full-history smoothed, not suitable for live trading signals",
    }
}

fn sample_data_window() -> DataWindow {
    DataWindow {
        frequency: Frequency::Daily,
        window_periods: 252,
        start: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        end: NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
    }
}

fn sample_data_quality() -> DataQuality {
    DataQuality {
        date_range_start: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
        date_range_end: NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
        trading_days: 1500,
        per_series: vec![],
    }
}

/// A `RiskDecompositionOutput` + its `EvidenceTrace`, with two factors
/// (enough for a non-empty correlation map) and given portfolio_betas so it
/// qualifies as a "compatible" baseline.
fn risk_decomposition_result(
    vol: f64,
    by_factor: &[(&str, f64, f64)],
    specific_risk_fraction: f64,
    regime: RegimeState,
    end_date: NaiveDate,
) -> (RiskDecompositionOutput, EvidenceTrace) {
    let betas: BTreeMap<String, f64> = by_factor.iter().map(|(f, c, _)| (f.to_string(), *c * 2.0)).collect();
    let factor_names: Vec<String> = by_factor.iter().map(|(f, _, _)| f.to_string()).collect();
    let output = RiskDecompositionOutput {
        portfolio_vol_annualized: vol,
        portfolio_vol_annualized_pct: vol * 100.0,
        by_stock: vec![],
        by_factor: by_factor
            .iter()
            .map(|(f, c, frac)| FactorContribution {
                factor: f.to_string(),
                contribution: *c,
                fraction_of_vol: *frac,
                fraction_of_vol_pct: *frac * 100.0,
            })
            .collect(),
        specific_risk_contribution: specific_risk_fraction * vol,
        specific_risk_fraction_of_vol: specific_risk_fraction,
        specific_risk_fraction_of_vol_pct: specific_risk_fraction * 100.0,
        portfolio_betas: betas,
        factor_correlation: CorrelationMatrix { factor_names, rows: vec![vec![1.0, 0.1], vec![0.1, 1.0]] },
    };
    let mut data_window = sample_data_window();
    data_window.end = end_date;
    let trace = EvidenceTrace {
        id: compute::trace::new_trace_id(),
        experiment: "RiskDecomposition".to_string(),
        inputs: serde_json::json!({}),
        data_as_of: compute::trace::data_as_of(&data_window),
        data_window,
        data_quality: sample_data_quality(),
        model_params: ModelParams {
            frequency: Frequency::Daily,
            window_periods: 252,
            factor_names: by_factor.iter().map(|(f, _, _)| f.to_string()).collect(),
            shrinkage_intensity: 0.03,
            annualization_factor: 252.0,
            regime_state: Some(regime),
            regime_fallback_warnings: vec![],
            cap_source: None,
        },
        outputs: serde_json::json!({ "result": output }),
        invariants: vec![],
        engine_version: "0.1.0".to_string(),
        engine_commit: compute::trace::engine_commit(),
        scenario_provenance: None,
        parent_trace_ids: Vec::new(),
        baseline_model_params: None,
        policy_result: None,
    };
    (output, trace)
}

fn baseline_snapshot_from(trace: &EvidenceTrace, vol: f64) -> RiskSnapshot {
    RiskSnapshot {
        id: "baseline-id".to_string(),
        created_at: "2026-01-01T00:00:00Z".to_string(),
        portfolio_hash: "hash".to_string(),
        experiment_type: "RiskDecomposition".to_string(),
        engine_version: "0.1.0".to_string(),
        regime_label: trace.model_params.regime_state.as_ref().map(|r| r.current_label.to_string()),
        smoothed_probs: trace.model_params.regime_state.as_ref().map(|r| r.smoothed_probs),
        portfolio_vol_annualized: Some(vol),
        cvar_historical: None,
        trace_json: serde_json::to_string(trace).unwrap(),
        narration: None,
        suggestion: None,
        grounding_warnings: None,
    }
}

fn default_input() -> RiskDriftInput {
    RiskDriftInput { baseline_snapshot_id: None, window: None, frequency: None }
}

#[test]
fn vol_change_abs_and_pct_are_computed_correctly() {
    let (_, baseline_trace) = risk_decomposition_result(
        0.10,
        &[("MARKET", 0.05, 0.5), ("USDINR", 0.03, 0.3)],
        0.2,
        sample_regime_state(0, "Bull", [0.9, 0.1, 0.0]),
        NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
    );
    let baseline = baseline_snapshot_from(&baseline_trace, 0.10);

    let (current_output, current_trace) = risk_decomposition_result(
        0.15,
        &[("MARKET", 0.08, 0.53), ("USDINR", 0.04, 0.27)],
        0.2,
        sample_regime_state(0, "Bull", [0.9, 0.1, 0.0]),
        NaiveDate::from_ymd_opt(2026, 1, 31).unwrap(),
    );

    let (output, _trace) = compute_risk_drift(&baseline, current_output, current_trace, &default_input()).unwrap();

    assert!((output.vol_before - 0.10).abs() < 1e-12);
    assert!((output.vol_after - 0.15).abs() < 1e-12);
    assert!((output.vol_change_abs - 0.05).abs() < 1e-9, "got {}", output.vol_change_abs);
    assert!((output.vol_change_pct - 50.0).abs() < 1e-6, "got {}", output.vol_change_pct);
    assert!(output.vol_increased);
    assert_eq!(output.days_elapsed, 30);
}

#[test]
fn factor_contribution_delta_residual_is_recorded_not_errored_when_it_does_not_sum_exactly() {
    // vol_change_abs = 0.05, but factor deltas sum to only 0.02 -- an
    // intentionally inexact case (Euler additivity across two different
    // fits is approximate, per the drift module doc): must not error, and
    // the recorded invariant must be marked failed rather than silently
    // dropped.
    let (_, baseline_trace) = risk_decomposition_result(
        0.10,
        &[("MARKET", 0.05, 0.5), ("USDINR", 0.03, 0.3)],
        0.2,
        sample_regime_state(0, "Bull", [0.9, 0.1, 0.0]),
        NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
    );
    let baseline = baseline_snapshot_from(&baseline_trace, 0.10);

    let (current_output, current_trace) = risk_decomposition_result(
        0.15,
        &[("MARKET", 0.06, 0.4), ("USDINR", 0.04, 0.27)],
        0.2,
        sample_regime_state(0, "Bull", [0.9, 0.1, 0.0]),
        NaiveDate::from_ymd_opt(2026, 1, 31).unwrap(),
    );

    let (output, trace) = compute_risk_drift(&baseline, current_output, current_trace, &default_input()).unwrap();

    let euler_sum: f64 = output.factor_contribution_delta.values().sum();
    assert!((euler_sum - 0.02).abs() < 1e-9);
    assert!((output.vol_change_abs - 0.05).abs() < 1e-9);

    let residual_invariant = trace
        .invariants
        .iter()
        .find(|i| i.name.contains("vol_change_abs"))
        .expect("residual invariant should always be recorded");
    assert!(!residual_invariant.passed, "residual should be recorded as failed, not silently dropped");
}

#[test]
fn regime_changed_is_true_when_regimes_differ_false_when_same() {
    let (_, baseline_trace) = risk_decomposition_result(
        0.10,
        &[("MARKET", 0.05, 0.5), ("USDINR", 0.03, 0.3)],
        0.2,
        sample_regime_state(0, "Bull", [0.9, 0.1, 0.0]),
        NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
    );
    let baseline = baseline_snapshot_from(&baseline_trace, 0.10);

    let (current_output, current_trace) = risk_decomposition_result(
        0.15,
        &[("MARKET", 0.08, 0.53), ("USDINR", 0.04, 0.27)],
        0.2,
        sample_regime_state(1, "Bear", [0.1, 0.8, 0.1]),
        NaiveDate::from_ymd_opt(2026, 1, 31).unwrap(),
    );
    let (output, _) = compute_risk_drift(&baseline, current_output, current_trace, &default_input()).unwrap();
    assert!(output.regime_changed);
    assert_eq!(output.regime_before, "Bull");
    assert_eq!(output.regime_after, "Bear");
    assert!(output.regime_worsened, "Bull -> Bear should count as worsened");

    let (current_output_same, current_trace_same) = risk_decomposition_result(
        0.11,
        &[("MARKET", 0.055, 0.5), ("USDINR", 0.033, 0.3)],
        0.2,
        sample_regime_state(0, "Bull", [0.85, 0.1, 0.05]),
        NaiveDate::from_ymd_opt(2026, 1, 31).unwrap(),
    );
    let (output_same, _) =
        compute_risk_drift(&baseline, current_output_same, current_trace_same, &default_input()).unwrap();
    assert!(!output_same.regime_changed);
    assert!(!output_same.regime_worsened);
}

#[test]
fn risk_became_more_concentrated_uses_a_5pp_threshold() {
    let (_, baseline_trace) = risk_decomposition_result(
        0.10,
        &[("MARKET", 0.05, 0.50), ("USDINR", 0.03, 0.30)],
        0.2,
        sample_regime_state(0, "Bull", [0.9, 0.1, 0.0]),
        NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
    );
    let baseline = baseline_snapshot_from(&baseline_trace, 0.10);

    // 50% -> 60%: a 10pp increase, above the 5pp threshold.
    let (current_more_concentrated, current_trace_more) = risk_decomposition_result(
        0.12,
        &[("MARKET", 0.072, 0.60), ("USDINR", 0.03, 0.25)],
        0.2,
        sample_regime_state(0, "Bull", [0.9, 0.1, 0.0]),
        NaiveDate::from_ymd_opt(2026, 1, 31).unwrap(),
    );
    let (output_more, _) =
        compute_risk_drift(&baseline, current_more_concentrated, current_trace_more, &default_input()).unwrap();
    assert!(output_more.risk_became_more_concentrated);

    // 50% -> 52%: only a 2pp increase, below the threshold.
    let (current_similar, current_trace_similar) = risk_decomposition_result(
        0.10,
        &[("MARKET", 0.052, 0.52), ("USDINR", 0.03, 0.30)],
        0.2,
        sample_regime_state(0, "Bull", [0.9, 0.1, 0.0]),
        NaiveDate::from_ymd_opt(2026, 1, 31).unwrap(),
    );
    let (output_similar, _) =
        compute_risk_drift(&baseline, current_similar, current_trace_similar, &default_input()).unwrap();
    assert!(!output_similar.risk_became_more_concentrated);
}

#[test]
fn no_baseline_snapshot_id_with_no_prior_snapshot_returns_no_prior_snapshot_error() {
    let store = Arc::new(SnapshotStore::open(":memory:").unwrap());
    let ctx = ExperimentContext { store, portfolio_hash: "empty-hash".to_string(), policy: None };
    let portfolio =
        Portfolio { holdings: vec![Holding { ticker: "AAA".to_string(), weight: 1.0 }], total_value_inr: 1.0 };

    let result = run_risk_drift(
        std::path::Path::new("data/cache"),
        &portfolio,
        &default_input(),
        &ctx,
    );

    match result {
        Err(ComputeError::NoPriorSnapshot(msg)) => {
            assert!(msg.contains("No prior snapshot found"), "unexpected message: {msg}");
        }
        other => panic!("expected ComputeError::NoPriorSnapshot, got {other:?}"),
    }
}
