//! Tests for `agent::orchestrator::plan_tools` -- the planning call's
//! response parsing and fallback behaviour. Doesn't exercise
//! `run_tool_plans` (that needs a real market-data fetch + snapshot store
//! to chain `current_risk` -> `risk_drift`; covered live in this session's
//! report instead, matching this codebase's existing convention of
//! `#[ignore]`d live-data tests for anything past the network/data-fetch
//! boundary).

mod support;

use agent::orchestrator::plan_tools;
use support::{text_response, MockGeminiClient};

#[tokio::test]
async fn single_tool_plan_is_parsed_as_is() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "current_risk", "params": {}, "reason": "user asked about current risk"}]"#,
    )]);

    let (plans, raw) = plan_tools(&client, "what's my portfolio risk?", &[]).await.unwrap();

    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].tool, "current_risk");
    assert_eq!(plans[0].reason, "user asked about current risk");
    assert!(raw.contains("current_risk"));
}

#[tokio::test]
async fn two_tool_plan_for_a_shock_and_rebalance_request_is_parsed_in_order() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[
            {"tool": "factor_shock", "params": {"shocks_pct": {"MARKET": -15.0}}, "reason": "hypothetical crash"},
            {"tool": "cvar_rebalance", "params": {"turnover_limit": 0.2}, "reason": "reduce tail risk after the shock"}
        ]"#,
    )]);

    let (plans, _raw) = plan_tools(&client, "what if the market drops 15%, and how would I rebalance to cut tail risk with 20% turnover?", &[])
        .await
        .unwrap();

    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].tool, "factor_shock");
    assert_eq!(plans[1].tool, "cvar_rebalance");
    assert_eq!(plans[1].params["turnover_limit"], 0.2);
}

#[tokio::test]
async fn current_risk_then_risk_drift_preserves_plan_order() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[
            {"tool": "current_risk", "params": {}, "reason": "establish current risk"},
            {"tool": "risk_drift", "params": {}, "reason": "compare against the last snapshot"}
        ]"#,
    )]);

    let (plans, _raw) = plan_tools(&client, "what is my risk and how has it changed since my last check?", &[])
        .await
        .unwrap();

    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].tool, "current_risk");
    assert_eq!(plans[1].tool, "risk_drift");
}

#[tokio::test]
async fn a_markdown_fenced_json_array_is_still_parsed() {
    let client = MockGeminiClient::new(vec![text_response(
        "```json\n[{\"tool\": \"portfolio_performance\", \"params\": {}, \"reason\": \"asked about returns\"}]\n```",
    )]);

    let (plans, _raw) = plan_tools(&client, "how has my portfolio done this year?", &[]).await.unwrap();

    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].tool, "portfolio_performance");
}

#[tokio::test]
async fn invalid_json_falls_back_to_a_single_current_risk_plan_and_keeps_the_raw_response() {
    let client = MockGeminiClient::new(vec![text_response("not valid json at all")]);

    let (plans, raw) = plan_tools(&client, "some unparseable request", &[]).await.unwrap();

    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].tool, "current_risk");
    assert_eq!(raw, "not valid json at all");
}

#[tokio::test]
async fn an_empty_array_response_also_falls_back_to_current_risk() {
    let client = MockGeminiClient::new(vec![text_response("[]")]);

    let (plans, _raw) = plan_tools(&client, "some request", &[]).await.unwrap();

    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].tool, "current_risk");
}

#[tokio::test]
async fn an_off_topic_message_declines_instead_of_running_a_fallback_experiment() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "decline", "params": {}, "reason": "I can only help with questions about your portfolio's risk and performance."}]"#,
    )]);

    let err = plan_tools(&client, "What is the weather in Mumbai?", &[]).await.unwrap_err();

    match err {
        agent::orchestrator::OrchestratorError::Unrecognised(text) => {
            assert_eq!(text, "I can only help with questions about your portfolio's risk and performance.");
        }
        other => panic!("expected Unrecognised, got {other:?}"),
    }
}
