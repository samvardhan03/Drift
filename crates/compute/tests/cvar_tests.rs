mod common;

use compute::cvar::{run_cvar_rebalance, CvarRebalanceInput};
use compute::experiments::{Holding, Portfolio};

/// CRASH has 95 calm scenarios (+/-1% log return) and 5 severe crash
/// scenarios (-40% log return); SAFE is calm throughout. With a 95%
/// confidence level, k = round(100 * 0.05) = 5 exactly matches the crash
/// count, so any weight on CRASH pulls the entire CVaR tail down with it.
/// A CVaR-minimizing, adequately-capitalized rebalance should cut CRASH to
/// (near) zero.
fn crash_vs_safe_returns() -> (Vec<f64>, Vec<f64>) {
    let n = 100;
    let mut crash = Vec::with_capacity(n);
    let mut safe = Vec::with_capacity(n);
    for i in 0..n {
        // Deterministic small alternating noise, no RNG needed.
        let noise = if i % 2 == 0 { 0.01 } else { -0.01 };
        safe.push(noise * 0.5);
        if i < 5 {
            crash.push(-0.40);
        } else {
            crash.push(noise);
        }
    }
    (crash, safe)
}

#[test]
fn heavy_tail_asset_is_cut_to_near_zero_when_turnover_allows() {
    let (crash, safe) = crash_vs_safe_returns();
    let data = common::market_data_from_log_returns(&[("CRASH", crash), ("SAFE", safe)]);

    let input = CvarRebalanceInput {
        portfolio: Portfolio {
            holdings: vec![
                Holding {
                    ticker: "CRASH".to_string(),
                    weight: 0.5,
                },
                Holding {
                    ticker: "SAFE".to_string(),
                    weight: 0.5,
                },
            ],
            total_value_inr: 1_000_000.0,
        },
        confidence_level: 0.95,
        // 1.0 (not < 1.0): a tighter per-name cap on both names would force
        // some residual weight onto CRASH just to make weights sum to 1,
        // which is a real cap-driven effect, not what this test is after.
        per_name_cap: Some(1.0),
        turnover_limit: 1.5,
        commission_bps: 10.0,
        frequency: compute::model::Frequency::Daily,
        window: None,
        policy: None,
    };

    let (output, trace) = run_cvar_rebalance(&data.quality, &data, &input).unwrap();
    assert_eq!(output.status, "optimal", "diagnostics: {:?}", output.diagnostics);

    for inv in &trace.invariants {
        assert!(inv.passed, "invariant failed: {} ({})", inv.name, inv.detail);
    }

    let weights_after = output.weights_after.expect("optimal solve has weights_after");
    assert!(
        weights_after["CRASH"] < 0.05,
        "CVaR minimization should cut the heavy-tail asset close to zero, got {}",
        weights_after["CRASH"]
    );
    assert!(
        (weights_after["SAFE"] - (1.0 - weights_after["CRASH"])).abs() < 1e-6,
        "weights should still sum to 1"
    );

    let stats_after = output.stats_after.unwrap();
    let stats_before = output.stats_before;
    assert!(
        stats_after.historical_cvar < stats_before.historical_cvar,
        "rebalanced CVaR ({}) should be lower than the starting CVaR ({})",
        stats_after.historical_cvar,
        stats_before.historical_cvar
    );
}

#[test]
fn zero_turnover_limit_returns_starting_weights_unchanged() {
    let (crash, safe) = crash_vs_safe_returns();
    let data = common::market_data_from_log_returns(&[("CRASH", crash), ("SAFE", safe)]);

    let input = CvarRebalanceInput {
        portfolio: Portfolio {
            holdings: vec![
                Holding {
                    ticker: "CRASH".to_string(),
                    weight: 0.5,
                },
                Holding {
                    ticker: "SAFE".to_string(),
                    weight: 0.5,
                },
            ],
            total_value_inr: 1_000_000.0,
        },
        confidence_level: 0.95,
        per_name_cap: Some(0.9), // w0 already satisfies the cap, so tau=0 is feasible
        turnover_limit: 0.0,
        commission_bps: 10.0,
        frequency: compute::model::Frequency::Daily,
        window: None,
        policy: None,
    };

    let (output, _trace) = run_cvar_rebalance(&data.quality, &data, &input).unwrap();
    assert_eq!(output.status, "optimal", "diagnostics: {:?}", output.diagnostics);

    let weights_after = output.weights_after.unwrap();
    assert!((weights_after["CRASH"] - 0.5).abs() < 1e-6);
    assert!((weights_after["SAFE"] - 0.5).abs() < 1e-6);
    assert!((output.turnover.unwrap()).abs() < 1e-6);
}

/// 5 names at a 0.20 default cap allows exactly full investment
/// (0.20 * 5 = 1), so the LP stays feasible -- lets this test check both
/// the defaulted cap's effect on `weights_after` and `cap_source`, unlike
/// the 2-name portfolio used elsewhere in this file (where a 0.20 cap is
/// always infeasible, per `cap_too_tight_for_full_investment_...` below).
#[test]
fn per_name_cap_omitted_defaults_to_0_20_and_is_recorded_as_server_default() {
    let (crash, safe) = crash_vs_safe_returns();
    let data = common::market_data_from_log_returns(&[
        ("CRASH", crash),
        ("SAFE", safe.clone()),
        ("SAFE2", safe.clone()),
        ("SAFE3", safe.clone()),
        ("SAFE4", safe),
    ]);

    let input = CvarRebalanceInput {
        portfolio: Portfolio {
            holdings: vec![
                Holding { ticker: "CRASH".to_string(), weight: 0.6 },
                Holding { ticker: "SAFE".to_string(), weight: 0.1 },
                Holding { ticker: "SAFE2".to_string(), weight: 0.1 },
                Holding { ticker: "SAFE3".to_string(), weight: 0.1 },
                Holding { ticker: "SAFE4".to_string(), weight: 0.1 },
            ],
            total_value_inr: 1_000_000.0,
        },
        confidence_level: 0.95,
        per_name_cap: None,
        turnover_limit: 1.5,
        commission_bps: 10.0,
        frequency: compute::model::Frequency::Daily,
        window: None,
        policy: None,
    };

    let (output, trace) = run_cvar_rebalance(&data.quality, &data, &input).unwrap();
    assert_eq!(output.status, "optimal", "diagnostics: {:?}", output.diagnostics);
    assert_eq!(trace.model_params.cap_source.as_deref(), Some("server-default-0.20"));

    let weights_after = output.weights_after.unwrap();
    for w in weights_after.values() {
        assert!(*w <= 0.20 + 1e-6, "weight {w} exceeds the defaulted 0.20 cap");
    }
}

#[test]
fn per_name_cap_given_is_recorded_as_user_specified() {
    let (crash, safe) = crash_vs_safe_returns();
    let data = common::market_data_from_log_returns(&[("CRASH", crash), ("SAFE", safe)]);

    let input = CvarRebalanceInput {
        portfolio: Portfolio {
            holdings: vec![
                Holding {
                    ticker: "CRASH".to_string(),
                    weight: 0.5,
                },
                Holding {
                    ticker: "SAFE".to_string(),
                    weight: 0.5,
                },
            ],
            total_value_inr: 1_000_000.0,
        },
        confidence_level: 0.95,
        per_name_cap: Some(0.9),
        turnover_limit: 0.0,
        commission_bps: 10.0,
        frequency: compute::model::Frequency::Daily,
        window: None,
        policy: None,
    };

    let (_output, trace) = run_cvar_rebalance(&data.quality, &data, &input).unwrap();
    assert_eq!(trace.model_params.cap_source.as_deref(), Some("user-specified"));
}

#[test]
fn cap_too_tight_for_full_investment_is_reported_as_infeasible() {
    let (crash, safe) = crash_vs_safe_returns();
    let data = common::market_data_from_log_returns(&[("CRASH", crash), ("SAFE", safe)]);

    let input = CvarRebalanceInput {
        portfolio: Portfolio {
            holdings: vec![
                Holding {
                    ticker: "CRASH".to_string(),
                    weight: 0.5,
                },
                Holding {
                    ticker: "SAFE".to_string(),
                    weight: 0.5,
                },
            ],
            total_value_inr: 1_000_000.0,
        },
        confidence_level: 0.95,
        // 2 names * 0.2 cap = 0.4 < 1: can never reach full investment.
        per_name_cap: Some(0.2),
        turnover_limit: 1.5,
        commission_bps: 10.0,
        frequency: compute::model::Frequency::Daily,
        window: None,
        policy: None,
    };

    let (output, trace) = run_cvar_rebalance(&data.quality, &data, &input).unwrap();
    assert_eq!(output.status, "infeasible");
    assert!(output.diagnostics.is_some());
    assert!(output.weights_after.is_none());
    assert!(trace.invariants.iter().any(|i| !i.passed));
}
