mod common;

use compute::experiments::{Holding, Portfolio};
use compute::model::{fit_factor_model, Frequency, ModelConfig};
use compute::policy::{evaluate_policy, max_factor_contribution_share, max_position_weight, portfolio_vol_annualized, RiskPolicy};

fn two_stock_portfolio() -> Portfolio {
    Portfolio {
        holdings: vec![
            Holding { ticker: "AAA".to_string(), weight: 0.75 },
            Holding { ticker: "BBB".to_string(), weight: 0.25 },
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

#[test]
fn max_vol_annualized_breach_reports_the_correct_magnitude() {
    let (data, model) = two_stock_model_and_data();
    let portfolio = two_stock_portfolio();
    let actual_vol = portfolio_vol_annualized(&portfolio, &model);

    let policy = RiskPolicy { max_vol_annualized: Some(actual_vol - 0.02), ..Default::default() };
    let result = evaluate_policy(&policy, &portfolio, &model, &data).unwrap();

    assert!(!result.all_passed);
    assert_eq!(result.breach_count, 1);
    let check = &result.checks[0];
    assert_eq!(check.rule, "max_vol_annualized");
    assert!(!check.passed);
    assert!((check.breach_magnitude - 0.02).abs() < 1e-9, "got {}", check.breach_magnitude);
    assert!(check.evidence.contains("volatility"));
}

#[test]
fn max_position_weight_breach_reports_the_correct_magnitude() {
    let (data, model) = two_stock_model_and_data();
    // The portfolio's max weight is 0.75 (AAA); a limit of 0.20 forces a
    // breach of exactly 0.55.
    let portfolio = two_stock_portfolio();
    let actual_max_weight = max_position_weight(&portfolio);
    assert!((actual_max_weight - 0.75).abs() < 1e-12);

    let policy = RiskPolicy { max_position_weight: Some(0.20), ..Default::default() };
    let result = evaluate_policy(&policy, &portfolio, &model, &data).unwrap();

    assert!(!result.all_passed);
    let check = result.checks.iter().find(|c| c.rule == "max_position_weight").unwrap();
    assert!(!check.passed);
    assert!((check.breach_magnitude - 0.55).abs() < 1e-9, "got {}", check.breach_magnitude);
}

#[test]
fn max_factor_contribution_share_breach_is_detected() {
    let (data, model) = two_stock_model_and_data();
    let portfolio = two_stock_portfolio();
    let (dominant_factor, actual_share) = max_factor_contribution_share(&portfolio, &model);

    // Pick a limit clearly below whatever the dominant factor's actual
    // share is, so this is a breach by construction regardless of the
    // exact synthetic-data-derived value.
    let policy =
        RiskPolicy { max_factor_contribution_share: Some(actual_share - 0.05), ..Default::default() };
    let result = evaluate_policy(&policy, &portfolio, &model, &data).unwrap();

    assert!(!result.all_passed);
    let check = result.checks.iter().find(|c| c.rule == "max_factor_contribution_share").unwrap();
    assert!(!check.passed);
    assert!((check.breach_magnitude - 0.05).abs() < 1e-9, "got {}", check.breach_magnitude);
    assert!(check.evidence.contains(&dominant_factor), "evidence should name the dominant factor: {}", check.evidence);
}

#[test]
fn all_rules_pass_when_limits_are_loose() {
    let (data, model) = two_stock_model_and_data();
    let portfolio = two_stock_portfolio();

    let policy = RiskPolicy {
        max_vol_annualized: Some(10.0),
        max_cvar_95: Some(10.0),
        max_factor_contribution_share: Some(1.0),
        max_position_weight: Some(1.0),
        max_turnover: None,
        max_loss_under_scenarios: None,
    };
    let result = evaluate_policy(&policy, &portfolio, &model, &data).unwrap();

    assert!(result.all_passed);
    assert_eq!(result.breach_count, 0);
    assert!(result.most_severe_breach.is_none());
    assert_eq!(result.checks.len(), 4);
}

#[test]
fn most_severe_breach_is_the_one_with_the_largest_breach_magnitude() {
    let (data, model) = two_stock_model_and_data();
    let portfolio = two_stock_portfolio();
    let actual_vol = portfolio_vol_annualized(&portfolio, &model);

    // Vol breach magnitude ~0.01; position-weight breach magnitude 0.55 --
    // the latter must win as most_severe_breach.
    let policy = RiskPolicy {
        max_vol_annualized: Some(actual_vol - 0.01),
        max_position_weight: Some(0.20),
        ..Default::default()
    };
    let result = evaluate_policy(&policy, &portfolio, &model, &data).unwrap();

    assert_eq!(result.breach_count, 2);
    let most_severe = result.most_severe_breach.expect("expected a most severe breach");
    assert_eq!(most_severe.rule, "max_position_weight");
}

#[test]
fn cvar_rebalance_with_policy_reports_pre_rebalance_breach() {
    use compute::cvar::{run_cvar_rebalance, CvarRebalanceInput};

    let (data, _model) = two_stock_model_and_data();
    let portfolio = two_stock_portfolio();

    // Fit the same regime model CvarRebalance's own informational path
    // would: since `window: None` below, CvarRebalance resolves its
    // scenario (and therefore regime-fit) window to *all* available
    // observations, not `frequency.default_window()` (see the README's
    // "Regime HMM window for CvarRebalance" judgment call) -- match that
    // exactly so the policy limit is guaranteed to be a breach.
    let total_obs = data.stock_returns[&portfolio.holdings[0].ticker].len();
    let regime_model =
        fit_factor_model(&data, &portfolio.tickers(), ModelConfig::new(total_obs, Frequency::Daily)).unwrap();
    let actual_vol_before = portfolio_vol_annualized(&portfolio, &regime_model);

    let input = CvarRebalanceInput {
        portfolio: portfolio.clone(),
        confidence_level: 0.95,
        per_name_cap: Some(0.9),
        turnover_limit: 0.5,
        commission_bps: 10.0,
        frequency: Frequency::Daily,
        window: None,
        policy: Some(RiskPolicy { max_vol_annualized: Some(actual_vol_before - 0.01), ..Default::default() }),
    };

    let (output, _trace) = run_cvar_rebalance(&data.quality, &data, &input).unwrap();

    let policy_result_before = output.policy_result_before.expect("policy_result_before should be Some");
    assert!(!policy_result_before.all_passed);
    assert_eq!(policy_result_before.breach_count, 1);
}

/// Regression test: `run_cvar_rebalance`'s internal `build_portfolio`
/// closure once iterated `weights_before`/`weights_after` (both
/// `BTreeMap<String, f64>`, hence alphabetically ordered) directly to
/// build the `Portfolio` handed to `evaluate_policy`, instead of following
/// `model.tickers`'s order -- harmless whenever the tickers *happened* to
/// already be alphabetical (as in every other test in this file), but a
/// hard `InvalidInput` ("portfolio holdings and fitted model tickers must
/// match 1:1, in order") the moment they weren't, caught live with a
/// real 10-ticker Nifty portfolio (RELIANCE.NS first, not alphabetically
/// first). Uses tickers in deliberately reverse-alphabetical order so this
/// would fail again if the bug ever came back.
#[test]
fn cvar_rebalance_with_policy_does_not_care_about_ticker_alphabetical_order() {
    use compute::cvar::{run_cvar_rebalance, CvarRebalanceInput};

    let stocks = [
        ("ZEBRA", 0.0001, [0.9, 0.1, 0.0, 0.2, 0.3]),
        ("MANGO", -0.0002, [0.4, -0.3, 0.5, 0.0, -0.1]),
        ("APPLE", 0.0002, [0.7, 0.0, 0.2, 0.1, 0.1]),
    ];
    let data = common::synthetic_multi_stock(&stocks, 300, 0.0008, 42);
    let portfolio = Portfolio {
        holdings: vec![
            Holding { ticker: "ZEBRA".to_string(), weight: 0.5 },
            Holding { ticker: "MANGO".to_string(), weight: 0.3 },
            Holding { ticker: "APPLE".to_string(), weight: 0.2 },
        ],
        total_value_inr: 1_000_000.0,
    };

    let input = CvarRebalanceInput {
        portfolio: portfolio.clone(),
        confidence_level: 0.95,
        per_name_cap: Some(0.9),
        turnover_limit: 0.5,
        commission_bps: 10.0,
        frequency: Frequency::Daily,
        window: None,
        policy: Some(RiskPolicy { max_vol_annualized: Some(0.0001), ..Default::default() }),
    };

    let (output, _trace) = run_cvar_rebalance(&data.quality, &data, &input).unwrap();
    assert_eq!(output.status, "optimal");
    let policy_result_before = output.policy_result_before.expect("policy_result_before should be Some");
    assert!(!policy_result_before.all_passed, "an near-zero vol limit should always breach");
}
