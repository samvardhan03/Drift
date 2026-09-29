mod common;

use compute::experiments::{Holding, Portfolio};
use compute::model::Frequency;
use compute::performance::{run_portfolio_performance, PortfolioPerformanceInput};
use compute::trace::DataWindow;

fn window_and_trace_helper(
    data: &compute::data::MarketData,
    input: &PortfolioPerformanceInput,
) -> (
    compute::performance::PortfolioPerformanceOutput,
    compute::trace::EvidenceTrace,
) {
    let window = input.resolved_window();
    let data_window = DataWindow {
        frequency: input.frequency,
        window_periods: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    };
    run_portfolio_performance(&data.quality, data_window, data, input).unwrap()
}

/// A single stock with a constant +1% daily log return should compound to
/// a known, exactly-checkable total/annualized return, zero volatility,
/// and zero drawdown -- the simplest possible correctness check.
#[test]
fn constant_positive_return_compounds_correctly_with_zero_vol_and_drawdown() {
    let n = 100;
    let log_r = 0.01_f64;
    let data = common::market_data_from_log_returns(&[("STEADY", vec![log_r; n])]);

    let input = PortfolioPerformanceInput {
        portfolio: Portfolio {
            holdings: vec![Holding { ticker: "STEADY".to_string(), weight: 1.0 }],
            total_value_inr: 1_000_000.0,
        },
        frequency: Frequency::Daily,
        window: Some(n),
    };

    let (output, trace) = window_and_trace_helper(&data, &input);

    let simple_r = log_r.exp() - 1.0;
    let expected_total_return = (1.0 + simple_r).powi(n as i32) - 1.0;
    assert!(
        (output.total_return - expected_total_return).abs() < 1e-9,
        "total_return {} != expected {}",
        output.total_return,
        expected_total_return
    );
    assert!(output.annualized_vol_realized.abs() < 1e-9, "expected ~zero vol, got {}", output.annualized_vol_realized);
    assert!(output.max_drawdown.abs() < 1e-9, "expected zero drawdown for a monotonically rising series");
    assert!(output.end_value_inr > output.start_value_inr);

    for inv in &trace.invariants {
        assert!(inv.passed, "invariant failed: {} ({})", inv.name, inv.detail);
    }
}

/// A portfolio that falls 50% then fully recovers should show a real
/// max_drawdown of about -50% even though total_return ends near zero.
#[test]
fn drawdown_is_captured_even_when_total_return_recovers_to_near_zero() {
    let mut returns = vec![(0.5_f64).ln()]; // -50% in one day (as a log return)
    returns.extend(vec![(2.0_f64).ln()]); // +100% the next day: back to ~1.0x
    returns.extend(vec![0.0; 8]); // pad to a reasonable window

    let data = common::market_data_from_log_returns(&[("VSHAPE", returns)]);

    let input = PortfolioPerformanceInput {
        portfolio: Portfolio {
            holdings: vec![Holding { ticker: "VSHAPE".to_string(), weight: 1.0 }],
            total_value_inr: 1_000_000.0,
        },
        frequency: Frequency::Daily,
        window: Some(10),
    };

    let (output, _trace) = window_and_trace_helper(&data, &input);

    assert!(
        output.max_drawdown < -0.45,
        "expected a deep drawdown from the -50% day, got {}",
        output.max_drawdown
    );
    assert!(
        output.total_return.abs() < 1e-6,
        "expected the portfolio to have round-tripped back to ~0 total return, got {}",
        output.total_return
    );
}

/// Two-stock portfolio: the invariant (end_value == start_value * (1 +
/// total_return)) should hold regardless of the per-holding weight split.
#[test]
fn two_holding_portfolio_satisfies_its_own_value_invariant() {
    let n = 50;
    let mut rng = common::Rng::new(11);
    let a: Vec<f64> = (0..n).map(|_| rng.next_signed() * 0.01).collect();
    let b: Vec<f64> = (0..n).map(|_| rng.next_signed() * 0.015).collect();
    let data = common::market_data_from_log_returns(&[("A", a), ("B", b)]);

    let input = PortfolioPerformanceInput {
        portfolio: Portfolio {
            holdings: vec![
                Holding { ticker: "A".to_string(), weight: 0.3 },
                Holding { ticker: "B".to_string(), weight: 0.7 },
            ],
            total_value_inr: 2_500_000.0,
        },
        frequency: Frequency::Daily,
        window: Some(n),
    };

    let (output, trace) = window_and_trace_helper(&data, &input);

    assert!(
        (output.end_value_inr - output.start_value_inr * (1.0 + output.total_return)).abs() < 1.0,
        "end_value_inr should match start_value_inr * (1 + total_return)"
    );
    for inv in &trace.invariants {
        assert!(inv.passed, "invariant failed: {} ({})", inv.name, inv.detail);
    }
    assert_eq!(trace.experiment, "PortfolioPerformance");
    assert!(trace.model_params.regime_state.is_none(), "this experiment never fits a factor model");
}

/// One holding steadily up, one steadily down: `holding_returns` should
/// carry each holding's own performance (not the portfolio-level figure),
/// and `best_performer`/`worst_performer` should identify them correctly.
#[test]
fn best_and_worst_performer_are_identified_from_per_holding_returns() {
    let n = 60;
    let data = common::market_data_from_log_returns(&[
        ("WINNER", vec![0.01_f64; n]),
        ("LOSER", vec![-0.01_f64; n]),
    ]);

    let input = PortfolioPerformanceInput {
        portfolio: Portfolio {
            holdings: vec![
                Holding { ticker: "WINNER".to_string(), weight: 0.4 },
                Holding { ticker: "LOSER".to_string(), weight: 0.6 },
            ],
            total_value_inr: 1_000_000.0,
        },
        frequency: Frequency::Daily,
        window: Some(n),
    };

    let (output, _trace) = window_and_trace_helper(&data, &input);

    assert_eq!(output.best_performer, "WINNER");
    assert_eq!(output.worst_performer, "LOSER");
    assert_eq!(output.holding_returns.len(), 2);

    let winner = &output.holding_returns["WINNER"];
    let loser = &output.holding_returns["LOSER"];
    assert!(winner.total_return_pct > 0.0, "WINNER should have a positive return, got {}", winner.total_return_pct);
    assert!(loser.total_return_pct < 0.0, "LOSER should have a negative return, got {}", loser.total_return_pct);
    assert!(winner.annualized_vol_pct.abs() < 1e-6, "constant daily return implies zero realized vol");

    let expected_winner_contribution = (0.4 * winner.total_return_pct * 100.0).round() / 100.0;
    assert!(
        (winner.contribution_to_portfolio_return_pct - expected_winner_contribution).abs() < 1e-9,
        "contribution_to_portfolio_return_pct {} != expected {}",
        winner.contribution_to_portfolio_return_pct,
        expected_winner_contribution
    );
}
