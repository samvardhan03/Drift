mod common;

use std::collections::BTreeMap;

use compute::experiments::{
    log_to_simple, run_factor_shock, run_risk_decomposition, simple_to_log, FactorShockInput,
    Holding, Portfolio, RiskDecompositionInput,
};
use compute::model::{fit_factor_model, Frequency, ModelConfig};
use compute::trace::DataWindow;

fn two_stock_model() -> (compute::data::MarketData, compute::model::FactorModel) {
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
        frequency: compute::model::Frequency::Daily,
        window_periods: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    }
}

fn two_holding_portfolio() -> Portfolio {
    Portfolio {
        holdings: vec![
            Holding {
                ticker: "AAA".to_string(),
                weight: 0.6,
            },
            Holding {
                ticker: "BBB".to_string(),
                weight: 0.4,
            },
        ],
        total_value_inr: 1_000_000.0,
    }
}

/// Deterministic small PRNG, matching the pattern used elsewhere in this
/// crate's tests (no `rand` dependency).
struct Rng(u64);
impl Rng {
    fn next_signed(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        unit * 2.0 - 1.0
    }
}

/// Three 100-obs volatility segments in `order` (e.g. `[0.003, 0.012,
/// 0.035]` for Bull-then-Bear-then-Crisis), for injecting a genuine
/// regime signal into a synthetic MARKET factor series.
fn regime_structured_returns(order: [f64; 3], seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(300);
    for scale in order {
        for _ in 0..100 {
            out.push(rng.next_signed() * scale);
        }
    }
    out
}

/// Builds the same two-stock model as `two_stock_model`, but with the
/// MARKET factor series replaced by a regime-structured one (see
/// `regime_structured_returns`). `volatility_order` controls which regime
/// the window ends in (and therefore which regime is "current"):
/// `[0.003, 0.012, 0.035]` ends in Crisis, `[0.035, 0.012, 0.003]` ends in
/// Bull.
fn two_stock_model_with_regime(
    volatility_order: [f64; 3],
    seed: u64,
) -> (compute::data::MarketData, compute::model::FactorModel) {
    let stocks = [
        ("AAA", 0.0001, [0.9, 0.1, 0.0, 0.2, 0.3]),
        ("BBB", -0.0002, [0.4, -0.3, 0.5, 0.0, -0.1]),
    ];
    let mut data = common::synthetic_multi_stock(&stocks, 300, 0.0008, 123);
    data.factor_returns.insert(
        "MARKET".to_string(),
        regime_structured_returns(volatility_order, seed),
    );
    let tickers = vec!["AAA".to_string(), "BBB".to_string()];
    let model = fit_factor_model(&data, &tickers, ModelConfig::new(252, Frequency::Daily)).unwrap();
    (data, model)
}

#[test]
fn euler_contributions_sum_to_portfolio_vol_stock_and_factor_views() {
    let (data, model) = two_stock_model();
    let input = RiskDecompositionInput {
        portfolio: two_holding_portfolio(),
        frequency: compute::model::Frequency::Daily,
        window: Some(252),
    };

    let (output, trace) =
        run_risk_decomposition(&data.quality, data_window(&data, 252), &model, &input).unwrap();

    for inv in &trace.invariants {
        assert!(inv.passed, "invariant failed: {} ({})", inv.name, inv.detail);
    }

    let stock_sum: f64 = output.by_stock.iter().map(|s| s.contribution).sum();
    assert!((stock_sum - output.portfolio_vol_annualized).abs() < 1e-9);

    let factor_plus_specific: f64 = output.by_factor.iter().map(|f| f.contribution).sum::<f64>()
        + output.specific_risk_contribution;
    assert!((factor_plus_specific - output.portfolio_vol_annualized).abs() < 1e-9);
}

/// Under `linear_approximation: true`, the original checkpoint's exact
/// linearity holds: P&L = value * (beta . shock), so doubling the shock
/// doubles P&L exactly.
#[test]
fn linear_approximation_pnl_is_linear_in_shock_size() {
    let (data, model) = two_stock_model();
    let portfolio = two_holding_portfolio();

    let mut shocks = BTreeMap::new();
    shocks.insert("MARKET".to_string(), -12.0);

    let input1 = FactorShockInput {
        portfolio: portfolio.clone(),
        shocks_pct: shocks.clone(),
        propagate: false,
        linear_approximation: true,
        frequency: compute::model::Frequency::Daily,
        window: Some(252),
    };
    let mut shocks2 = shocks.clone();
    *shocks2.get_mut("MARKET").unwrap() *= 2.0;
    let input2 = FactorShockInput {
        portfolio,
        shocks_pct: shocks2,
        propagate: false,
        linear_approximation: true,
        frequency: compute::model::Frequency::Daily,
        window: Some(252),
    };

    let (out1, trace1) =
        run_factor_shock(&data.quality, data_window(&data, 252), &model, &input1).unwrap();
    let (out2, _) =
        run_factor_shock(&data.quality, data_window(&data, 252), &model, &input2).unwrap();

    for inv in &trace1.invariants {
        assert!(inv.passed, "invariant failed: {} ({})", inv.name, inv.detail);
    }

    assert!(
        (out2.portfolio_pnl_inr - 2.0 * out1.portfolio_pnl_inr).abs()
            < 1e-6 * out1.portfolio_pnl_inr.abs().max(1.0),
        "doubling the shock should double P&L under linear_approximation: {} vs 2x{}",
        out2.portfolio_pnl_inr,
        out1.portfolio_pnl_inr
    );
}

/// Default (non-`linear_approximation`) mode: betas are applied to LOG
/// shocks, so it's the *log-space holding return* that is exactly linear
/// in the log shock, not the simple-return P&L. Pick a second shock whose
/// log is exactly double the first's log (`s2 = (1+s1)^2 - 1`) and check
/// `log_return` doubles exactly.
#[test]
fn log_space_holding_return_is_linear_in_log_shock() {
    let (data, model) = two_stock_model();
    let portfolio = two_holding_portfolio();

    let s1 = -0.12; // -12% simple
    let log1 = simple_to_log(s1);
    let s2 = (1.0 + s1) * (1.0 + s1) - 1.0; // log2 = 2 * log1 exactly
    let log2 = simple_to_log(s2);
    assert!((log2 - 2.0 * log1).abs() < 1e-12);

    let mut shocks1 = BTreeMap::new();
    shocks1.insert("MARKET".to_string(), s1 * 100.0);
    let mut shocks2 = BTreeMap::new();
    shocks2.insert("MARKET".to_string(), s2 * 100.0);

    let input1 = FactorShockInput {
        portfolio: portfolio.clone(),
        shocks_pct: shocks1,
        propagate: false,
        linear_approximation: false,
        frequency: compute::model::Frequency::Daily,
        window: Some(252),
    };
    let input2 = FactorShockInput {
        portfolio,
        shocks_pct: shocks2,
        propagate: false,
        linear_approximation: false,
        frequency: compute::model::Frequency::Daily,
        window: Some(252),
    };

    let (out1, _) =
        run_factor_shock(&data.quality, data_window(&data, 252), &model, &input1).unwrap();
    let (out2, _) =
        run_factor_shock(&data.quality, data_window(&data, 252), &model, &input2).unwrap();

    for (h1, h2) in out1.per_holding.iter().zip(out2.per_holding.iter()) {
        assert!(
            (h2.log_return - 2.0 * h1.log_return).abs() < 1e-9,
            "log_return should scale exactly with the log shock: {} vs 2x{}",
            h2.log_return,
            h1.log_return
        );
    }
    assert!(
        (out2.portfolio_log_pnl_inr - 2.0 * out1.portfolio_log_pnl_inr).abs()
            < 1e-6 * out1.portfolio_log_pnl_inr.abs().max(1.0)
    );
}

#[test]
fn simple_log_round_trip_matches_within_tolerance() {
    for pct in [-50.0, -12.0, -1.0, -0.001, 0.0, 0.001, 1.0, 12.0, 20.0, 200.0] {
        let simple = pct / 100.0;
        let log = simple_to_log(simple);
        let back = log_to_simple(log);
        assert!(
            (back - simple).abs() < 1e-12,
            "round trip mismatch for {pct}%: simple={simple} -> log={log} -> back={back}"
        );
    }
}

#[test]
fn conditional_propagation_is_noop_when_all_factors_given() {
    let (data, model) = two_stock_model();
    let portfolio = two_holding_portfolio();

    let mut shocks = BTreeMap::new();
    shocks.insert("MARKET".to_string(), -12.0);
    shocks.insert("USDINR".to_string(), 2.0);
    shocks.insert("BRENT".to_string(), 20.0);
    shocks.insert("GOLD_USD".to_string(), 5.0);
    shocks.insert("RATES_PROXY".to_string(), 1.0);

    let input = FactorShockInput {
        portfolio,
        shocks_pct: shocks,
        propagate: true,
        linear_approximation: false,
        frequency: compute::model::Frequency::Daily,
        window: Some(252),
    };

    let (output, _) =
        run_factor_shock(&data.quality, data_window(&data, 252), &model, &input).unwrap();

    assert!(
        output.implied_shocks.is_empty(),
        "no factors should need propagation when all five are specified"
    );
}

/// Regime-conditioning is now unconditional (no more `regime_covariance`
/// flag): fitting against a window with a genuine regime-structured MARKET
/// series must always populate `regime_state` and use the *current*
/// regime's factor covariance, which differs from the naive full-window
/// covariance whenever the data has real regime structure.
#[test]
fn risk_decomposition_always_uses_regime_conditional_vol() {
    let portfolio = two_holding_portfolio();

    let stocks = [
        ("AAA", 0.0001, [0.9, 0.1, 0.0, 0.2, 0.3]),
        ("BBB", -0.0002, [0.4, -0.3, 0.5, 0.0, -0.1]),
    ];
    let mut data = common::synthetic_multi_stock(&stocks, 300, 0.0008, 123);
    data.factor_returns.insert(
        "MARKET".to_string(),
        regime_structured_returns([0.003, 0.012, 0.035], 2),
    );
    let tickers = vec!["AAA".to_string(), "BBB".to_string()];
    let model = fit_factor_model(&data, &tickers, ModelConfig::new(252, Frequency::Daily)).unwrap();
    assert!(model.regime_state.is_some(), "regime_state must always be populated");

    // Direct signal that regime-conditioning is actually doing something on
    // this regime-structured data (not a no-op): Bull and Crisis regimes'
    // own factor covariances differ from each other.
    let regimes = model.regime_factor_covariance_daily.as_ref().unwrap();
    assert!(
        regimes[0] != regimes[2],
        "Bull and Crisis regime covariances should differ on regime-structured data"
    );

    let input = RiskDecompositionInput {
        portfolio,
        frequency: Frequency::Daily,
        window: Some(252),
    };
    let (output, trace) =
        run_risk_decomposition(&data.quality, data_window(&data, 252), &model, &input).unwrap();

    for inv in &trace.invariants {
        assert!(inv.passed, "invariant failed: {} ({})", inv.name, inv.detail);
    }
    assert!(output.portfolio_vol_annualized > 0.0);
    assert!(trace.model_params.regime_state.is_some());
}

#[test]
fn factor_shock_crisis_comparison_present_when_current_regime_is_not_crisis() {
    // Ends in Bull (lowest vol last) -> current regime should be Bull, not Crisis.
    let (data, model) = two_stock_model_with_regime([0.035, 0.012, 0.003], 5);
    let state = model.regime_state.as_ref().unwrap();
    assert_ne!(state.current_label, "Crisis", "test setup: expected a non-Crisis current regime");

    let portfolio = two_holding_portfolio();
    let mut shocks = BTreeMap::new();
    shocks.insert("MARKET".to_string(), -12.0);
    shocks.insert("BRENT".to_string(), 20.0);
    let input = FactorShockInput {
        portfolio,
        shocks_pct: shocks,
        propagate: true,
        linear_approximation: false,
        frequency: Frequency::Daily,
        window: Some(252),
    };

    let (output, _) =
        run_factor_shock(&data.quality, data_window(&data, 252), &model, &input).unwrap();

    assert!(
        output.crisis_comparison.is_some(),
        "crisis_comparison should be present when the current regime isn't Crisis"
    );
    assert!(output.crisis_comparison_note.is_some());
}

#[test]
fn factor_shock_crisis_comparison_absent_when_current_regime_is_crisis() {
    // Ends in Crisis (highest vol last) -> current regime should be Crisis.
    let (data, model) = two_stock_model_with_regime([0.003, 0.012, 0.035], 2);
    let state = model.regime_state.as_ref().unwrap();
    assert_eq!(state.current_label, "Crisis", "test setup: expected a Crisis current regime");

    let portfolio = two_holding_portfolio();
    let mut shocks = BTreeMap::new();
    shocks.insert("MARKET".to_string(), -12.0);
    shocks.insert("BRENT".to_string(), 20.0);
    let input = FactorShockInput {
        portfolio,
        shocks_pct: shocks,
        propagate: true,
        linear_approximation: false,
        frequency: Frequency::Daily,
        window: Some(252),
    };

    let (output, _) =
        run_factor_shock(&data.quality, data_window(&data, 252), &model, &input).unwrap();

    assert!(
        output.crisis_comparison.is_none(),
        "crisis_comparison should be absent when the current regime already is Crisis"
    );
    assert!(output.crisis_comparison_note.is_none());
}
