mod support;

use agent::conversation::ConversationTurn;
use agent::parse::{parse_experiment, ParseError};
use agent::schema::{
    CVAR_REBALANCE_FUNCTION, FACTOR_SHOCK_FUNCTION, POLICY_CHECK_FUNCTION,
    PORTFOLIO_PERFORMANCE_FUNCTION, REVERSE_STRESS_FUNCTION, RISK_DECOMPOSITION_FUNCTION,
    RISK_DRIFT_FUNCTION,
};
use compute::experiments::{Experiment, Holding, Portfolio};
use support::{function_call_response, text_response, MockGeminiClient};

fn two_stock_portfolio() -> Portfolio {
    Portfolio {
        holdings: vec![
            Holding {
                ticker: "RELIANCE.NS".to_string(),
                weight: 0.6,
            },
            Holding {
                ticker: "TCS.NS".to_string(),
                weight: 0.4,
            },
        ],
        total_value_inr: 1_000_000.0,
    }
}

#[tokio::test]
async fn parses_factor_shock_function_call() {
    let args = serde_json::json!({
        "shocks_pct": { "MARKET": -12.0, "BRENT": 20.0 },
        "propagate": true,
    });
    let client = MockGeminiClient::new(vec![function_call_response(FACTOR_SHOCK_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(&client, "what if the market drops 12%", portfolio.clone(), &[])
        .await
        .unwrap();

    match experiment {
        Experiment::FactorShock(input) => {
            assert_eq!(input.shocks_pct.get("MARKET"), Some(&-12.0));
            assert_eq!(input.shocks_pct.get("BRENT"), Some(&20.0));
            assert!(input.propagate);
            assert_eq!(input.portfolio.tickers(), portfolio.tickers());
        }
        other => panic!("expected FactorShock, got {other:?}"),
    }
}

#[tokio::test]
async fn parses_risk_decomposition_function_call() {
    let args = serde_json::json!({});
    let client =
        MockGeminiClient::new(vec![function_call_response(RISK_DECOMPOSITION_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(&client, "what's my portfolio risk?", portfolio.clone(), &[])
        .await
        .unwrap();

    match experiment {
        Experiment::RiskDecomposition(input) => {
            assert_eq!(input.portfolio.tickers(), portfolio.tickers());
        }
        other => panic!("expected RiskDecomposition, got {other:?}"),
    }
}

#[tokio::test]
async fn parses_portfolio_performance_function_call() {
    let args = serde_json::json!({});
    let client =
        MockGeminiClient::new(vec![function_call_response(PORTFOLIO_PERFORMANCE_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(
        &client,
        "how has my portfolio been performing recently?",
        portfolio.clone(),
        &[],
    )
    .await
    .unwrap();

    match experiment {
        Experiment::PortfolioPerformance(input) => {
            assert_eq!(input.portfolio.tickers(), portfolio.tickers());
        }
        other => panic!("expected PortfolioPerformance, got {other:?}"),
    }
}

#[tokio::test]
async fn parses_risk_drift_function_call_with_null_baseline_snapshot_id() {
    let args = serde_json::json!({ "baseline_snapshot_id": null });
    let client = MockGeminiClient::new(vec![function_call_response(RISK_DRIFT_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(&client, "what changed in my risk?", portfolio.clone(), &[])
        .await
        .unwrap();

    match experiment {
        Experiment::RiskDrift(input) => {
            assert_eq!(input.baseline_snapshot_id, None);
        }
        other => panic!("expected RiskDrift, got {other:?}"),
    }
}

#[tokio::test]
async fn parses_risk_drift_function_call_with_a_specific_baseline_snapshot_id() {
    let args = serde_json::json!({ "baseline_snapshot_id": "abc-123" });
    let client = MockGeminiClient::new(vec![function_call_response(RISK_DRIFT_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(&client, "compare my risk to snapshot abc-123", portfolio.clone(), &[])
        .await
        .unwrap();

    match experiment {
        Experiment::RiskDrift(input) => {
            assert_eq!(input.baseline_snapshot_id.as_deref(), Some("abc-123"));
        }
        other => panic!("expected RiskDrift, got {other:?}"),
    }
}

#[tokio::test]
async fn parses_reverse_stress_function_call_with_null_factor_bounds() {
    let args = serde_json::json!({ "loss_threshold_inr": 500_000.0, "factor_bounds": null });
    let client = MockGeminiClient::new(vec![function_call_response(REVERSE_STRESS_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(&client, "what shock would wipe out 5 lakh rupees?", portfolio.clone(), &[])
        .await
        .unwrap();

    match experiment {
        Experiment::ReverseStress(input) => {
            assert_eq!(input.loss_threshold_inr, 500_000.0);
            assert!(input.factor_bounds.is_none());
        }
        other => panic!("expected ReverseStress, got {other:?}"),
    }
}

#[tokio::test]
async fn parses_reverse_stress_function_call_with_explicit_factor_bounds() {
    let args = serde_json::json!({
        "loss_threshold_inr": 500_000.0,
        "factor_bounds": { "MARKET": [-50.0, 0.0], "BRENT": [-30.0, 80.0] },
    });
    let client = MockGeminiClient::new(vec![function_call_response(REVERSE_STRESS_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(
        &client,
        "what shock breaks my portfolio if Nifty can only fall 50% and Brent can only move 30%?",
        portfolio.clone(),
        &[],
    )
    .await
    .unwrap();

    match experiment {
        Experiment::ReverseStress(input) => {
            let bounds = input.factor_bounds.expect("factor_bounds should be present");
            assert_eq!(bounds.get("MARKET"), Some(&(-50.0, 0.0)));
            assert_eq!(bounds.get("BRENT"), Some(&(-30.0, 80.0)));
        }
        other => panic!("expected ReverseStress, got {other:?}"),
    }
}

#[tokio::test]
async fn parses_cvar_rebalance_function_call() {
    let args = serde_json::json!({
        "per_name_cap": 0.2,
        "turnover_limit": 0.3,
        "confidence_level": 0.95,
    });
    let client = MockGeminiClient::new(vec![function_call_response(CVAR_REBALANCE_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(&client, "rebalance to cut tail risk", portfolio.clone(), &[])
        .await
        .unwrap();

    match experiment {
        Experiment::CvarRebalance(input) => {
            assert_eq!(input.per_name_cap, Some(0.2));
            assert_eq!(input.turnover_limit, 0.3);
            assert_eq!(input.portfolio.tickers(), portfolio.tickers());
        }
        other => panic!("expected CvarRebalance, got {other:?}"),
    }
}

#[tokio::test]
async fn parses_policy_check_function_call_with_a_partial_policy() {
    let args = serde_json::json!({
        "policy": { "max_vol_annualized": 0.18 },
    });
    let client = MockGeminiClient::new(vec![function_call_response(POLICY_CHECK_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment =
        parse_experiment(&client, "is my portfolio within risk limits?", portfolio.clone(), &[])
            .await
            .unwrap();

    match experiment {
        Experiment::PolicyCheck(input) => {
            assert_eq!(input.policy.max_vol_annualized, Some(0.18));
            assert_eq!(input.policy.max_cvar_95, None);
            assert_eq!(input.policy.max_factor_contribution_share, None);
            assert_eq!(input.policy.max_position_weight, None);
            assert_eq!(input.policy.max_turnover, None);
            assert!(input.policy.max_loss_under_scenarios.is_none());
        }
        other => panic!("expected PolicyCheck, got {other:?}"),
    }
}

#[tokio::test]
async fn parses_cvar_rebalance_function_call_with_a_policy_field_attached() {
    let args = serde_json::json!({
        "turnover_limit": 0.3,
        "policy": { "max_vol_annualized": 0.15, "max_position_weight": 0.20 },
    });
    let client = MockGeminiClient::new(vec![function_call_response(CVAR_REBALANCE_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(
        &client,
        "rebalance to cut tail risk but keep vol under 15% and no name over 20%",
        portfolio.clone(),
        &[],
    )
    .await
    .unwrap();

    match experiment {
        Experiment::CvarRebalance(input) => {
            let policy = input.policy.expect("policy should be present");
            assert_eq!(policy.max_vol_annualized, Some(0.15));
            assert_eq!(policy.max_position_weight, Some(0.20));
        }
        other => panic!("expected CvarRebalance, got {other:?}"),
    }
}

/// The caller's portfolio always wins over whatever (if anything) Gemini
/// put in the function-call args' portfolio field, since Gemini is never
/// given real holdings/weights to extract from.
#[tokio::test]
async fn caller_portfolio_overrides_any_portfolio_in_the_function_call_args() {
    let args = serde_json::json!({
        "portfolio": { "holdings": [{"ticker": "MADE_UP.NS", "weight": 1.0}], "total_value_inr": 1.0 },
    });
    let client =
        MockGeminiClient::new(vec![function_call_response(RISK_DECOMPOSITION_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(&client, "risk please", portfolio.clone(), &[])
        .await
        .unwrap();

    match experiment {
        Experiment::RiskDecomposition(input) => {
            assert_eq!(input.portfolio.tickers(), portfolio.tickers());
        }
        other => panic!("expected RiskDecomposition, got {other:?}"),
    }
}

/// `conversation_history` must appear as prior turns, in order, before the
/// current user message -- and roles must map onto Gemini's own
/// `"user"`/`"model"` vocabulary (our `"assistant"` -> Gemini's `"model"`).
#[tokio::test]
async fn conversation_history_is_passed_as_prior_turns_in_order() {
    let args = serde_json::json!({
        "per_name_cap": 0.2,
        "turnover_limit": 0.25,
        "confidence_level": 0.95,
    });
    let client = MockGeminiClient::new(vec![function_call_response(CVAR_REBALANCE_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let history = vec![
        ConversationTurn::user("what's my portfolio risk?"),
        ConversationTurn::assistant("Vol is 15.5% annualised."),
    ];
    parse_experiment(&client, "now try with 25% turnover", portfolio, &history)
        .await
        .unwrap();

    let request = client.last_request();
    assert_eq!(request.contents.len(), 3, "2 prior turns + current message");
    assert_eq!(request.contents[0].role.as_deref(), Some("user"));
    assert_eq!(request.contents[0].parts[0].text.as_deref(), Some("what's my portfolio risk?"));
    assert_eq!(request.contents[1].role.as_deref(), Some("model"));
    assert_eq!(request.contents[1].parts[0].text.as_deref(), Some("Vol is 15.5% annualised."));
    assert_eq!(request.contents[2].role.as_deref(), Some("user"));
    assert_eq!(request.contents[2].parts[0].text.as_deref(), Some("now try with 25% turnover"));
}

/// An empty `conversation_history` must produce the exact same request
/// shape as before the parameter existed: a single user-turn `contents`
/// entry.
#[tokio::test]
async fn empty_conversation_history_matches_pre_existing_request_shape() {
    let args = serde_json::json!({});
    let client = MockGeminiClient::new(vec![function_call_response(RISK_DECOMPOSITION_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    parse_experiment(&client, "what's my portfolio risk?", portfolio, &[])
        .await
        .unwrap();

    let request = client.last_request();
    assert_eq!(request.contents.len(), 1);
    assert_eq!(request.contents[0].role.as_deref(), Some("user"));
    assert_eq!(
        request.contents[0].parts[0].text.as_deref(),
        Some("what's my portfolio risk?")
    );
}

#[tokio::test]
async fn text_response_is_unrecognised() {
    let client = MockGeminiClient::new(vec![text_response(
        "I cannot map this message to a supported experiment.",
    )]);

    let err = parse_experiment(&client, "what's the weather today?", two_stock_portfolio(), &[])
        .await
        .unwrap_err();

    match err {
        ParseError::Unrecognised(text) => {
            assert_eq!(text, "I cannot map this message to a supported experiment.");
        }
        other => panic!("expected Unrecognised, got {other:?}"),
    }
}
