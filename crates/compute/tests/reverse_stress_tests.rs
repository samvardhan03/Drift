mod common;

use std::collections::BTreeMap;

use compute::experiments::{Holding, Portfolio};
use compute::model::{fit_factor_model, Frequency, ModelConfig};
use compute::reverse_stress::{kkt_shock, mahalanobis_severity, run_reverse_stress, severity_label, ReverseStressInput};
use compute::trace::DataWindow;
use compute::ComputeError;
use nalgebra::{DMatrix, DVector};

#[test]
fn kkt_shock_matches_closed_form_for_a_single_active_factor() {
    // Diagonal F, with only factor index 2 having nonzero linear
    // sensitivity (p). Since p_j = 0 for j != 2 and F is diagonal, F*p is
    // zero everywhere except index 2 -- the whole problem decouples to a
    // single-factor closed form: s* = -L / p_k, mahalanobis = |s*| / sqrt(F_kk).
    let k = 5;
    let f = DMatrix::from_diagonal(&DVector::from_vec(vec![0.02, 0.01, 0.05, 0.03, 0.04]));
    let mut p = DVector::zeros(k);
    let active = 2;
    p[active] = 2_000_000.0; // arbitrary nonzero sensitivity
    let loss_threshold = 300_000.0;

    let s = kkt_shock(&f, &p, loss_threshold);

    let expected_s_active = -loss_threshold / p[active];
    assert!((s[active] - expected_s_active).abs() < 1e-6, "got {}", s[active]);
    for i in 0..k {
        if i != active {
            assert!(s[i].abs() < 1e-9, "expected factor {i} to be ~0, got {}", s[i]);
        }
    }

    let severity = mahalanobis_severity(&s, &f).unwrap();
    let expected_severity = expected_s_active.abs() / f[(active, active)].sqrt();
    assert!((severity - expected_severity).abs() < 1e-6, "got {severity} expected {expected_severity}");
}

#[test]
fn severity_label_buckets_correctly() {
    assert_eq!(severity_label(0.5), "within 1\u{3c3}");
    assert_eq!(severity_label(0.999), "within 1\u{3c3}");
    assert_eq!(severity_label(1.0), "1\u{2013}2\u{3c3}");
    assert_eq!(severity_label(1.5), "1\u{2013}2\u{3c3}");
    assert_eq!(severity_label(2.0), "2\u{2013}3\u{3c3}");
    assert_eq!(severity_label(2.9), "2\u{2013}3\u{3c3}");
    assert_eq!(severity_label(3.0), ">3\u{3c3}");
    assert_eq!(severity_label(10.0), ">3\u{3c3}");
}

fn two_stock_portfolio() -> Portfolio {
    Portfolio {
        holdings: vec![
            Holding { ticker: "AAA".to_string(), weight: 0.6 },
            Holding { ticker: "BBB".to_string(), weight: 0.4 },
        ],
        total_value_inr: 1_000_000.0,
    }
}

fn two_stock_model_and_data() -> (compute::data::MarketData, compute::model::FactorModel) {
    let stocks = [
        ("AAA", 0.0001, [0.9, 0.1, 0.0, 0.2, 0.3]),
        ("BBB", -0.0002, [0.4, -0.3, 0.5, 0.0, -0.1]),
    ];
    let data = common::synthetic_multi_stock(&stocks, 300, 0.0008, 123);
    let tickers = vec!["AAA".to_string(), "BBB".to_string()];
    let model = fit_factor_model(&data, &tickers, ModelConfig::new(252, Frequency::Daily)).unwrap();
    (data, model)
}

fn data_window(data: &compute::data::MarketData, window: usize) -> DataWindow {
    DataWindow {
        frequency: Frequency::Daily,
        window_periods: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    }
}

#[test]
fn infeasible_threshold_returns_the_correct_max_feasible_loss() {
    let (data, model) = two_stock_model_and_data();
    let portfolio = two_stock_portfolio();

    // Extremely tight bounds (±0.01% on every factor) make any breach of a
    // large threshold impossible.
    let mut tight_bounds = BTreeMap::new();
    for name in ["MARKET", "USDINR", "BRENT", "GOLD_USD", "RATES_PROXY"] {
        tight_bounds.insert(name.to_string(), (-0.01, 0.01));
    }
    let input = ReverseStressInput {
        loss_threshold_inr: 10_000_000.0, // far beyond what ±0.01% shocks could ever cause
        factor_bounds: Some(tight_bounds),
        window: Some(252),
        frequency: None,
    };

    let result = run_reverse_stress(&data.quality, data_window(&data, 252), &model, &portfolio, &input);
    match result {
        Err(ComputeError::ReverseStressInfeasible(msg)) => {
            assert!(msg.contains("cannot be breached"), "unexpected message: {msg}");
            assert!(msg.contains("Maximum feasible loss"), "unexpected message: {msg}");
        }
        other => panic!("expected ComputeError::ReverseStressInfeasible, got {other:?}"),
    }
}

#[test]
fn solution_respects_factor_bounds() {
    let (data, model) = two_stock_model_and_data();
    let portfolio = two_stock_portfolio();

    let mut bounds = BTreeMap::new();
    bounds.insert("MARKET".to_string(), (-15.0, 5.0));
    bounds.insert("USDINR".to_string(), (-3.0, 8.0));
    bounds.insert("BRENT".to_string(), (-20.0, 20.0));
    bounds.insert("GOLD_USD".to_string(), (-10.0, 10.0));
    bounds.insert("RATES_PROXY".to_string(), (-5.0, 5.0));

    let input = ReverseStressInput {
        loss_threshold_inr: 50_000.0,
        factor_bounds: Some(bounds.clone()),
        window: Some(252),
        frequency: None,
    };

    let (output, trace) =
        run_reverse_stress(&data.quality, data_window(&data, 252), &model, &portfolio, &input).unwrap();

    for (factor, (lb, ub)) in &bounds {
        let shock = output.shock_vector[factor];
        assert!(
            shock >= lb - 1e-6 && shock <= ub + 1e-6,
            "{factor} shock {shock} out of bounds [{lb}, {ub}]"
        );
    }

    let bounds_invariant = trace
        .invariants
        .iter()
        .find(|i| i.name.contains("factor_bounds_used"))
        .expect("bounds invariant should be recorded");
    assert!(bounds_invariant.passed);
}

#[test]
fn gradient_descent_converges_and_breaches_the_threshold_on_a_synthetic_portfolio() {
    let stocks: Vec<(&str, f64, [f64; 5])> = vec![
        ("S1", 0.0001, [0.9, 0.1, 0.0, 0.2, 0.3]),
        ("S2", -0.0002, [0.4, -0.3, 0.5, 0.0, -0.1]),
        ("S3", 0.0003, [1.1, 0.2, -0.1, 0.1, 0.4]),
        ("S4", 0.0000, [0.6, 0.0, 0.3, -0.2, 0.0]),
        ("S5", -0.0001, [0.8, -0.1, 0.2, 0.3, 0.2]),
        ("S6", 0.0002, [1.3, 0.15, 0.0, 0.1, 0.5]),
        ("S7", 0.0001, [0.5, 0.05, 0.1, 0.0, 0.1]),
        ("S8", -0.0003, [0.7, -0.2, 0.4, 0.2, -0.2]),
        ("S9", 0.0002, [1.0, 0.1, 0.1, 0.15, 0.3]),
        ("S10", 0.0001, [0.9, 0.0, 0.2, 0.1, 0.2]),
    ];
    let data = common::synthetic_multi_stock(&stocks, 300, 0.0008, 7);
    let tickers: Vec<String> = stocks.iter().map(|(t, _, _)| t.to_string()).collect();
    let model = fit_factor_model(&data, &tickers, ModelConfig::new(252, Frequency::Daily)).unwrap();

    let portfolio = Portfolio {
        holdings: tickers.iter().map(|t| Holding { ticker: t.clone(), weight: 0.1 }).collect(),
        total_value_inr: 10_000_000.0,
    };

    let input = ReverseStressInput {
        loss_threshold_inr: 500_000.0,
        factor_bounds: None,
        window: Some(252),
        frequency: None,
    };

    let (output, trace) =
        run_reverse_stress(&data.quality, data_window(&data, 252), &model, &portfolio, &input).unwrap();

    assert!(
        output.portfolio_pnl_inr <= -input.loss_threshold_inr * 0.999,
        "portfolio_pnl_inr {} should breach -{}",
        output.portfolio_pnl_inr,
        input.loss_threshold_inr
    );
    assert!(output.mahalanobis_severity > 0.0);
    assert!(["optimal", "converged", "max_iter"].contains(&output.solver_status.as_str()));
    assert_eq!(output.most_vulnerable_holdings.len(), 3);

    let breach_invariant = trace
        .invariants
        .iter()
        .find(|i| i.name.contains("loss_threshold_inr * 0.999"))
        .expect("breach invariant should be recorded");
    assert!(breach_invariant.passed);
}
