//! Confirms regime is genuinely always-on end to end, against real Yahoo
//! Finance data, for every experiment type. Requires network access, so
//! `#[ignore]`d by default -- run explicitly with
//! `cargo test -p compute --test always_on_regime_tests -- --ignored`.

use std::collections::BTreeMap;
use std::path::Path;

use compute::cvar::CvarRebalanceInput;
use compute::experiments::{run_factor_shock, run_risk_decomposition, FactorShockInput, Holding, Portfolio, RiskDecompositionInput};
use compute::model::{fit_factor_model, Frequency, ModelConfig};
use compute::performance::{run_portfolio_performance, PortfolioPerformanceInput};
use compute::trace::DataWindow;

fn portfolio() -> Portfolio {
    Portfolio {
        holdings: vec![
            Holding { ticker: "RELIANCE.NS".to_string(), weight: 0.6 },
            Holding { ticker: "TCS.NS".to_string(), weight: 0.4 },
        ],
        total_value_inr: 1_000_000.0,
    }
}

#[test]
#[ignore]
fn all_experiment_types_have_non_null_regime_state_on_real_data() {
    let cache_dir = Path::new("data/cache");
    let tickers = portfolio().tickers();
    let data = compute::data::load_market_data(cache_dir, &tickers, false, Frequency::Daily)
        .expect("live Yahoo fetch failed");
    let window = 252;
    let data_window = DataWindow {
        frequency: Frequency::Daily,
        window_periods: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    };

    let model = fit_factor_model(&data, &tickers, ModelConfig::new(window, Frequency::Daily))
        .expect("factor model fit failed");

    // FactorShock
    let mut shocks_pct = BTreeMap::new();
    shocks_pct.insert("MARKET".to_string(), -12.0);
    let factor_shock_input = FactorShockInput {
        portfolio: portfolio(),
        shocks_pct,
        propagate: true,
        linear_approximation: false,
        frequency: Frequency::Daily,
        window: Some(window),
    };
    let (_, trace) = run_factor_shock(&data.quality, data_window.clone(), &model, &factor_shock_input)
        .expect("run_factor_shock failed");
    assert!(trace.model_params.regime_state.is_some(), "FactorShock trace must have a non-null regime_state");

    // RiskDecomposition
    let risk_decomp_input = RiskDecompositionInput {
        portfolio: portfolio(),
        frequency: Frequency::Daily,
        window: Some(window),
    };
    let (_, trace) = run_risk_decomposition(&data.quality, data_window.clone(), &model, &risk_decomp_input)
        .expect("run_risk_decomposition failed");
    assert!(trace.model_params.regime_state.is_some(), "RiskDecomposition trace must have a non-null regime_state");

    // CvarRebalance
    let cvar_input = CvarRebalanceInput {
        portfolio: portfolio(),
        confidence_level: 0.95,
        per_name_cap: None,
        turnover_limit: 0.3,
        commission_bps: 10.0,
        frequency: Frequency::Daily,
        window: None,
        policy: None,
    };
    let (_, trace) = compute::cvar::run_cvar_rebalance(&data.quality, &data, &cvar_input)
        .expect("run_cvar_rebalance failed");
    assert!(trace.model_params.regime_state.is_some(), "CvarRebalance trace must have a non-null regime_state");

    // PortfolioPerformance
    let perf_input = PortfolioPerformanceInput {
        portfolio: portfolio(),
        frequency: Frequency::Daily,
        window: Some(window),
    };
    let (output, trace) = run_portfolio_performance(&data.quality, data_window, &data, &perf_input)
        .expect("run_portfolio_performance failed");
    assert!(trace.model_params.regime_state.is_some(), "PortfolioPerformance trace must have a non-null regime_state");
    assert!(output.regime_label.is_some(), "PortfolioPerformance output.regime_label must be Some on real data");
}
