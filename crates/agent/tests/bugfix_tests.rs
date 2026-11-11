//! Tests for the routing / narration / grounding bug-fix batch: stock-vs-
//! macro routing, entity isolation, focus holding, directional and
//! recommendation grounding. The Gemini mock returns canned text, so
//! tests of *model* behaviour pin the prompt wording and the deterministic
//! checks around it; the live behaviour is covered by the deployed checks.

mod support;

use agent::gemini::GeminiRequest;
use agent::grounding::grounded_narrate_many_with;
use agent::narrate::{narrate_tools_with_options, NarrationOptions, NARRATE_SYSTEM_PROMPT};
use agent::orchestrator::{apply_focus_holding, plan_tools, resolve_focus_holding, OrchestratorError, PLANNING_SYSTEM_PROMPT as P};
use agent::ConversationTurn;
use compute::experiments::{Holding, Portfolio};
use compute::trace::EvidenceTrace;
use support::{text_response, MockGeminiClient};

fn portfolio() -> Portfolio {
    let h = |t: &str, w: f64| Holding { ticker: t.to_string(), weight: w };
    Portfolio {
        holdings: vec![
            h("RELIANCE.NS", 0.3),
            h("RATNAVEER.NS", 0.1),
            h("NSIL.BO", 0.1),
            h("ZYDUSWELL.NS", 0.1),
            h("GOLDCASE.NS", 0.2),
            h("KIOCL.NS", 0.1),
            h("BLS.NS", 0.05),
            h("DYCL.BO", 0.05),
        ],
        total_value_inr: 1_000_000.0,
    }
}

fn trace(experiment: &str, result: serde_json::Value) -> EvidenceTrace {
    let mut t: EvidenceTrace = serde_json::from_value(serde_json::json!({
        "experiment": experiment, "inputs": {"portfolio": portfolio()},
        "data_window": {"frequency": "Daily", "window_periods": 1, "start": "2025-01-01", "end": "2025-01-02"},
        "data_quality": {"date_range_start": "2025-01-01", "date_range_end": "2025-01-02", "trading_days": 2, "per_series": []},
        "model_params": {"frequency": "Daily", "window_periods": 1, "factor_names": [], "shrinkage_intensity": 0.0, "annualization_factor": 252.0, "regime_state": null, "regime_fallback_warnings": [], "cap_source": null},
        "outputs": {}, "invariants": [], "engine_version": "t"
    }))
    .unwrap();
    t.outputs = serde_json::json!({ "result": result });
    t
}

/// A shock trace where GOLDCASE, KIOCL and BLS lose the most and RELIANCE gains.
fn shock_trace() -> EvidenceTrace {
    trace(
        "FactorShock",
        serde_json::json!({"per_holding": [
            {"ticker": "GOLDCASE.NS", "pnl_inr": -9000.0},
            {"ticker": "KIOCL.NS", "pnl_inr": -5000.0},
            {"ticker": "BLS.NS", "pnl_inr": -4000.0},
            {"ticker": "DYCL.BO", "pnl_inr": -300.0},
            {"ticker": "RELIANCE.NS", "pnl_inr": 2000.0},
            {"ticker": "RATNAVEER.NS", "pnl_inr": 100.0},
        ]}),
    )
}

fn system_prompt(req: &GeminiRequest) -> String {
    req.system_instruction.as_ref().unwrap().parts[0].text.clone().unwrap()
}

fn content_text(req: &GeminiRequest, i: usize) -> String {
    req.contents[i].parts[0].text.clone().unwrap()
}

// ---- Bug 1 / 6: routing and scope --------------------------------------

#[tokio::test]
async fn a_declined_stock_price_question_surfaces_as_declined_with_the_redirect_text() {
    let redirect = "I cannot model individual stock price movements -- the system analyses macro factor risks. Which would be most useful?";
    for msg in ["What if Ratnaveer increases 10%?", "What if NSIL goes up 5%?"] {
        let client = MockGeminiClient::new(vec![text_response(format!(
            r#"[{{"tool": "decline", "params": {{}}, "reason": "{redirect}"}}]"#
        ))]);
        match plan_tools(&client, msg, &[]).await.unwrap_err() {
            OrchestratorError::Declined(text) => assert_eq!(text, redirect),
            other => panic!("expected Declined, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn macro_questions_plan_a_tool_not_a_decline() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "factor_shock", "params": {"shocks_pct": {"MARKET": -10.0}}, "reason": "Nifty falls"}]"#,
    )]);
    let (plans, _) = plan_tools(&client, "What if Nifty falls 10%?", &[]).await.unwrap();
    assert_eq!(plans[0].tool, "factor_shock");
    assert_eq!(plans[0].params["shocks_pct"]["MARKET"], -10.0);
}

#[test]
fn the_planning_prompt_pins_rule_0_and_every_scope_boundary() {
    assert!(P.starts_with("You are a planning engine") || P.contains("ROUTING RULE 0"));
    // Rule 0 comes before the general tool rules.
    assert!(P.find("ROUTING RULE 0").unwrap() < P.find("Rules for tool selection:").unwrap());
    for example in ["Ratnaveer", "NSIL", "NEVER route to factor_shock or historical_stress"] {
        assert!(P.contains(example), "{example}");
    }
    assert!(P.contains("I cannot model individual stock price movements"));
    assert!(P.contains("Which would be most useful?"));
    // Historical scenarios only when the *current* message names one.
    assert!(P.contains("NEVER use historical_stress unless the CURRENT user message explicitly names"));
    assert!(P.contains("crude oil after an IL&FS question is a factor_shock"));
    // Out-of-scope messages.
    assert!(P.contains("Screener.in or Tickertape"));
    assert!(P.contains("I cannot forecast stock prices"));
    assert!(P.contains("I don't have real-time news"));
    assert!(P.contains("I can only help with portfolio risk analysis."));
    // Macro stays in scope.
    assert!(P.contains("'market crash', 'Nifty falls', 'bear market' -> factor_shock MARKET: -20.0"));
    assert!(P.contains("'Nifty falls 15%' -> MARKET: -15.0"));
}

#[tokio::test]
async fn an_explicit_historical_event_still_plans_historical_stress() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "historical_stress", "params": {"scenario_id": "ilfs_contagion"}, "reason": "IL&FS named"}]"#,
    )]);
    let (plans, _) = plan_tools(&client, "What is the IL&FS impact?", &[]).await.unwrap();
    assert_eq!(plans[0].params["scenario_id"], "ilfs_contagion");
}

#[tokio::test]
async fn the_planner_sees_prior_turns_but_a_crude_followup_plans_a_factor_shock() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "factor_shock", "params": {"shocks_pct": {"BRENT": 20.0}}, "reason": "crude spike"}]"#,
    )]);
    let history = vec![
        ConversationTurn::user("What would the IL&FS scenario do to my portfolio?"),
        ConversationTurn::assistant("The IL&FS scenario would cost you about 12%."),
    ];
    let (plans, _) = plan_tools(&client, "What if crude spikes?", &history).await.unwrap();
    assert_eq!(plans[0].tool, "factor_shock");
    assert_eq!(plans[0].params["shocks_pct"]["BRENT"], 20.0);
    // History is passed to the planner, followed by the current message.
    let req = client.last_request();
    assert_eq!(req.contents.len(), 3);
    assert_eq!(content_text(&req, 2), "What if crude spikes?");
}

// ---- Bug 3: focus holding ----------------------------------------------

#[test]
fn focus_holding_resolves_to_a_real_holding_even_from_a_slightly_wrong_symbol() {
    let p = portfolio();
    assert_eq!(resolve_focus_holding("ZYDUSWELL.NS", &p).as_deref(), Some("ZYDUSWELL.NS"));
    assert_eq!(resolve_focus_holding("zyduswell", &p).as_deref(), Some("ZYDUSWELL.NS"));
    // The planner's guess had a stray P; the holding is RATNAVEER.NS.
    assert_eq!(resolve_focus_holding("RATNAVEERP.NS", &p).as_deref(), Some("RATNAVEER.NS"));
    // A holding listed only on BSE still resolves from an .NS guess.
    assert_eq!(resolve_focus_holding("NSIL.NS", &p).as_deref(), Some("NSIL.BO"));
    assert_eq!(resolve_focus_holding("TCS.NS", &p), None);
}

#[tokio::test]
async fn how_is_zydus_doing_plans_portfolio_performance_with_the_focus_holding_pinned() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "portfolio_performance", "params": {"focus_holding": "ZYDUSWELL.NS"}, "reason": "specific stock"}]"#,
    )]);
    let (mut plans, _) = plan_tools(&client, "How is Zydus Wellness doing in my portfolio?", &[]).await.unwrap();
    assert_eq!(plans[0].tool, "portfolio_performance");
    assert_eq!(apply_focus_holding(&mut plans, &portfolio()).as_deref(), Some("ZYDUSWELL.NS"));
    assert_eq!(plans[0].params["focus_holding"], "ZYDUSWELL.NS");

    // An unknown stock is dropped rather than narrated about.
    let mut plans = vec![agent::ToolPlan {
        tool: "portfolio_performance".into(),
        params: serde_json::json!({"focus_holding": "TCS.NS"}),
        reason: String::new(),
    }];
    assert_eq!(apply_focus_holding(&mut plans, &portfolio()), None);
    assert!(plans[0].params.get("focus_holding").is_none());
}

#[tokio::test]
async fn the_focus_holding_is_prepended_to_the_trace_sent_for_narration() {
    let client = MockGeminiClient::new(vec![text_response("Zyduswell is up.")]);
    let t = trace("PortfolioPerformance", serde_json::json!({"holding_returns": {}}));
    let opts = NarrationOptions { focus_holding: Some("ZYDUSWELL.NS".to_string()), ..Default::default() };
    narrate_tools_with_options(&client, &[t], None, &[], &opts).await.unwrap();
    let req = client.last_request();
    let payload = content_text(&req, req.contents.len() - 1);
    assert!(payload.starts_with("USER QUESTION IS SPECIFICALLY ABOUT: ZYDUSWELL.NS\n"));
    assert!(payload.contains("Do NOT give a generic portfolio summary."));
    assert!(payload.contains("TRACE:\n["));

    // Without a focus the payload is just the trace JSON.
    let client = MockGeminiClient::new(vec![text_response("ok")]);
    narrate_tools_with_options(&client, &[shock_trace()], None, &[], &NarrationOptions::default()).await.unwrap();
    let req = client.last_request();
    let payload = content_text(&req, req.contents.len() - 1);
    assert!(!payload.contains("SPECIFICALLY ABOUT"));
    assert!(payload.contains("TRACE:\n["));
}

// ---- Bug 2: entity isolation -------------------------------------------

#[tokio::test]
async fn prior_turns_are_followed_by_a_separator_exchange_before_the_current_trace() {
    let client = MockGeminiClient::new(vec![text_response("ok")]);
    let history = vec![
        ConversationTurn::user("How is Ratnaveer doing?"),
        ConversationTurn::assistant("Ratnaveer is down."),
    ];
    narrate_tools_with_options(&client, &[shock_trace()], None, &history, &NarrationOptions::default())
        .await
        .unwrap();
    let req = client.last_request();
    let roles: Vec<&str> = req.contents.iter().map(|c| c.role.as_deref().unwrap()).collect();
    assert_eq!(roles, vec!["user", "model", "user", "model", "user"]);
    assert!(content_text(&req, 2).starts_with("--- PRIOR CONVERSATION ENDS HERE ---"));
    assert!(content_text(&req, 2).contains("Narrate ONLY this."));
    assert_eq!(content_text(&req, 3), "Understood. I will narrate only the current experiment result.");
    assert!(content_text(&req, 4).contains("TRACE:\n["), "the trace comes last");
    assert!(content_text(&req, 4).starts_with("ENTITY ISOLATION"));
}

#[tokio::test]
async fn no_history_means_no_separator() {
    let client = MockGeminiClient::new(vec![text_response("ok")]);
    narrate_tools_with_options(&client, &[shock_trace()], None, &[], &NarrationOptions::default()).await.unwrap();
    assert_eq!(client.last_request().contents.len(), 1);
}

#[test]
fn the_narration_prompt_is_the_strict_seven_rule_version() {
    let p = NARRATE_SYSTEM_PROMPT;
    assert!(p.starts_with("You are a narrator."));
    assert!(p.contains("You have NO independent knowledge"));
    for rule in 1..=7 {
        assert!(p.contains(&format!("RULE {rule} \u{2014}")), "rule {rule}");
    }
    assert!(p.contains("You do not decide direction. The trace decides."));
    assert!(p.contains("NEVER recommend action on a holding based on"));
    assert!(p.contains("The user's current question is provided at"));
    // The old experiment-specific rules moved out of the prompt.
    for gone in ["Factor shock:", "Risk decomposition:", "Reverse stress:", "Multi-tool:", "volunteer one insight"] {
        assert!(!p.contains(gone), "{gone}");
    }
}

#[tokio::test]
async fn the_context_leads_with_the_question_then_constraints_hints_and_the_trace() {
    let client = MockGeminiClient::new(vec![text_response("ok")]);
    let opts = NarrationOptions { user_question: Some("What happened to GOLDCASE?".to_string()), ..Default::default() };
    narrate_tools_with_options(&client, &[shock_trace()], None, &[], &opts).await.unwrap();
    let req = client.last_request();
    let ctx = content_text(&req, req.contents.len() - 1);
    assert!(ctx.starts_with("CURRENT USER QUESTION: What happened to GOLDCASE?\n\n"));
    // Direction constraints come from the trace's signs.
    assert!(ctx.contains("MANDATORY DIRECTION CONSTRAINTS"));
    assert!(ctx.contains("GOLDCASE.NS: NEGATIVE contributor"));
    assert!(ctx.contains("RELIANCE.NS: POSITIVE contributor"));
    // Reduce recommendations are limited to the worst contributors (KIOCL/BLS/GOLDCASE), never RELIANCE.
    let rec = ctx.lines().find(|l| l.starts_with("RECOMMENDATION CONSTRAINT")).unwrap();
    assert!(rec.contains("GOLDCASE.NS") && rec.contains("KIOCL.NS") && rec.contains("BLS.NS"));
    assert!(!rec.contains("RELIANCE"));
    assert!(ctx.contains("Key fields to narrate: portfolio_pnl_inr"));
    assert!(ctx.find("MANDATORY DIRECTION").unwrap() < ctx.find("TRACE:").unwrap());
    // None of this is in the system prompt.
    assert!(!system_prompt(&req).contains("MANDATORY DIRECTION"));
}

#[tokio::test]
async fn a_trace_with_no_losers_forbids_any_reduce_recommendation() {
    let client = MockGeminiClient::new(vec![text_response("ok")]);
    let t = trace("FactorShock", serde_json::json!({"per_holding": [{"ticker": "RELIANCE.NS", "pnl_inr": 100.0}]}));
    narrate_tools_with_options(&client, &[t], None, &[], &NarrationOptions::default()).await.unwrap();
    let req = client.last_request();
    assert!(content_text(&req, req.contents.len() - 1).contains("no holding in this trace qualifies for a reduce recommendation"));
}

#[tokio::test]
async fn reverse_stress_holding_pnl_feeds_the_direction_constraints() {
    let client = MockGeminiClient::new(vec![text_response("ok")]);
    let t = trace(
        "ReverseStress",
        serde_json::json!({"holding_pnl": {"GOLDCASE.NS": -9000.0, "RELIANCE.NS": 500.0},
                           "factor_attribution": {"MARKET": -8000.0, "GOLD_USD": 200.0}}),
    );
    narrate_tools_with_options(&client, &[t], None, &[], &NarrationOptions::default()).await.unwrap();
    let req = client.last_request();
    let ctx = content_text(&req, req.contents.len() - 1);
    assert!(ctx.contains("GOLDCASE.NS: NEGATIVE contributor"));
    assert!(ctx.contains("MARKET factor: NEGATIVE attribution"));
    assert!(ctx.contains("GOLD_USD factor: POSITIVE attribution"));
}

#[tokio::test]
async fn a_realtime_question_gets_the_data_caveat_in_the_context() {
    let client = MockGeminiClient::new(vec![text_response("ok")]);
    let note = "Note: I use historical daily data up to 2026-09-24. I don't have today's intraday prices.";
    let opts = NarrationOptions { realtime_note: Some(note.to_string()), ..Default::default() };
    narrate_tools_with_options(&client, &[shock_trace()], None, &[], &opts).await.unwrap();
    let req = client.last_request();
    assert!(content_text(&req, req.contents.len() - 1).contains(&format!("EXACTLY THIS SENTENCE, THEN ANSWER: \"{note}\"")));
}

#[test]
fn only_intraday_phrasing_gets_the_realtime_caveat() {
    use agent::orchestrator::apply_realtime_caveat;
    let plan = || vec![agent::ToolPlan { tool: "portfolio_performance".into(), params: serde_json::json!({}), reason: String::new() }];
    for msg in [
        "Why did my portfolio fall today?",
        "what is my portfolio doing this morning",
        "any intraday moves?",
        "Is my portfolio up today",
        "why is my portfolio down right now",
        "has it crashed right now?",
    ] {
        let mut p = plan();
        assert!(apply_realtime_caveat(&mut p, msg), "{msg}");
        assert_eq!(p[0].params["realtime_caveat"], true);
    }
    for msg in [
        "What is my biggest risk right now?",
        "What is my risk right now?",
        "how am I positioned right now",
        "What happened this week",
        "is it currently falling?",
        "How is my portfolio performing?",
    ] {
        let mut p = plan();
        assert!(!apply_realtime_caveat(&mut p, msg), "{msg}");
        assert!(p[0].params.get("realtime_caveat").is_none(), "{msg}");
    }
}

#[tokio::test]
async fn the_planner_is_shown_the_holdings_and_told_how_to_handle_a_missing_stock() {
    use agent::orchestrator::plan_tools_for;
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "decline", "params": {}, "reason": "IRFC is not in your current portfolio."}]"#,
    )]);
    let p = portfolio();
    let err = plan_tools_for(&client, "How is IRFC doing in my portfolio?", &[], Some(&p)).await.unwrap_err();
    match err {
        OrchestratorError::Declined(t) => assert_eq!(t, "IRFC is not in your current portfolio."),
        other => panic!("{other:?}"),
    }
    let sys = system_prompt(&client.last_request());
    assert!(sys.starts_with("PORTFOLIO HOLDINGS: RELIANCE.NS (30.0%), RATNAVEER.NS (10.0%)"));
    assert!(sys.contains("'[STOCK] is not in your current portfolio.'"));
    assert!(sys.contains("ROUTING RULE 0"), "the normal prompt still follows the holdings block");
}

#[tokio::test]
async fn a_narration_that_drags_in_a_prior_turns_stock_is_retried_and_cleaned() {
    let history = vec![
        ConversationTurn::user("How is Ratnaveer doing?"),
        ConversationTurn::assistant("Ratnaveer has fallen sharply."),
    ];
    let client = MockGeminiClient::new(vec![
        text_response("A crude spike hurts KIOCL most, while you are fixated on Ratnaveer."),
        text_response("A crude spike hurts KIOCL most."),
    ]);
    let (n, retries) =
        grounded_narrate_many_with(&client, &[shock_trace()], &history, &NarrationOptions::default()).await.unwrap();
    assert_eq!(retries, 1);
    assert!(!n.narration.to_lowercase().contains("ratnaveer"));
    assert!(n.grounding_warnings.is_empty());
    assert!(system_prompt(&client.last_request()).contains("CRITICAL CORRECTION REQUIRED"));
    assert!(system_prompt(&client.last_request()).contains("RATNAVEER.NS was discussed in an earlier turn"));
}

#[tokio::test]
async fn a_prior_turn_stock_that_is_a_current_top_mover_may_be_mentioned() {
    let history = vec![ConversationTurn::user("How is KIOCL doing?"), ConversationTurn::assistant("KIOCL is weak.")];
    let client = MockGeminiClient::new(vec![text_response("KIOCL is again one of the biggest losers here.")]);
    let (n, retries) =
        grounded_narrate_many_with(&client, &[shock_trace()], &history, &NarrationOptions::default()).await.unwrap();
    assert_eq!(retries, 0);
    assert!(n.directional_checks.is_empty());
}

#[tokio::test]
async fn the_stock_the_user_asks_about_now_is_always_allowed() {
    let history = vec![ConversationTurn::user("How is Ratnaveer doing?"), ConversationTurn::assistant("Ratnaveer is down.")];
    let client = MockGeminiClient::new(vec![text_response("Ratnaveer is barely moving.")]);
    let opts = NarrationOptions { focus_holding: Some("RATNAVEER.NS".to_string()), ..Default::default() };
    let (_, retries) = grounded_narrate_many_with(&client, &[shock_trace()], &history, &opts).await.unwrap();
    assert_eq!(retries, 0);
}

// ---- Bug 4: directional grounding --------------------------------------

#[tokio::test]
async fn a_wrong_direction_triggers_a_retry_and_the_corrected_narration_is_accepted() {
    let client = MockGeminiClient::new(vec![
        text_response("Gold is the primary stabiliser in this selloff."),
        text_response("GOLDCASE.NS was the largest drag in this selloff."),
    ]);
    let (n, retries) =
        grounded_narrate_many_with(&client, &[shock_trace()], &[], &NarrationOptions::default()).await.unwrap();
    assert_eq!(retries, 1);
    assert_eq!(n.narration, "GOLDCASE.NS was the largest drag in this selloff.");
    assert!(n.grounding_warnings.is_empty());
    assert!(n.directional_checks.is_empty() && n.directional_warnings.is_empty());
    let retry_prompt = system_prompt(&client.last_request());
    assert!(retry_prompt.contains("CRITICAL CORRECTION REQUIRED"));
    assert!(retry_prompt.contains("described positively but the trace shows it as a negative contributor"));
}

#[tokio::test]
async fn an_uncorrected_wrong_direction_is_reported_in_grounding_warnings() {
    let client = MockGeminiClient::new(vec![
        text_response("Gold is the primary stabiliser."),
        text_response("Gold remains a stabiliser here."),
        text_response("Gold is still the stabiliser."),
    ]);
    let (n, retries) =
        grounded_narrate_many_with(&client, &[shock_trace()], &[], &NarrationOptions::default()).await.unwrap();
    assert_eq!(retries, 2);
    assert_eq!(n.directional_checks.len(), 1);
    assert_eq!(n.directional_warnings.len(), 1);
    assert!(n.grounding_warnings.iter().any(|w| w.contains("described positively")));
}

#[tokio::test]
async fn consistent_direction_does_not_retry() {
    let client = MockGeminiClient::new(vec![text_response("Reliance gained while GOLDCASE.NS dragged the portfolio down.")]);
    let (_, retries) =
        grounded_narrate_many_with(&client, &[shock_trace()], &[], &NarrationOptions::default()).await.unwrap();
    assert_eq!(retries, 0);
    assert_eq!(client.call_count(), 1);
}

// ---- Bug 5: recommendations --------------------------------------------

#[tokio::test]
async fn a_recommendation_for_a_holding_the_trace_does_not_single_out_is_corrected() {
    let perf = trace(
        "PortfolioPerformance",
        serde_json::json!({"holding_returns": {
            "KIOCL.NS": {"total_return_pct": -40.0, "max_drawdown_pct": -55.0},
            "DYCL.BO": {"total_return_pct": -30.0, "max_drawdown_pct": -45.0},
            "BLS.NS": {"total_return_pct": -20.0, "max_drawdown_pct": -35.0},
            "ELECON.BO": {"total_return_pct": -10.0, "max_drawdown_pct": -25.0},
            "RELIANCE.NS": {"total_return_pct": 8.0, "max_drawdown_pct": -5.0},
        }}),
    );
    let client = MockGeminiClient::new(vec![
        text_response("Consider reducing Reliance because of its high beta."),
        text_response("Consider reducing KIOCL.NS, which has the deepest drawdown."),
    ]);
    let (n, retries) = grounded_narrate_many_with(&client, &[perf], &[], &NarrationOptions::default()).await.unwrap();
    assert_eq!(retries, 1);
    assert!(n.narration.contains("KIOCL.NS") && !n.narration.contains("Reliance"));
    assert!(system_prompt(&client.last_request()).contains("not among the trace's worst contributors"));
}

// ---- AskResponse / pipeline: a decline runs nothing ---------------------

#[tokio::test]
async fn the_pipeline_surfaces_a_decline_before_running_any_experiment() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "decline", "params": {}, "reason": "I cannot forecast stock prices."}]"#,
    )]);
    let ctx = compute::context::ExperimentContext {
        store: std::sync::Arc::new(store::SnapshotStore::open(":memory:").unwrap()),
        portfolio_hash: String::new(),
        policy: None,
    };
    let err = match agent::pipeline::run(&client, "Will Reliance reach 3000?", portfolio(), &[], &ctx).await {
        Err(e) => e,
        Ok(_) => panic!("a declined question must not produce a result"),
    };
    match err {
        agent::PipelineError::Orchestrator(OrchestratorError::Declined(t)) => assert_eq!(t, "I cannot forecast stock prices."),
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(client.call_count(), 1, "only the planning call ran");
}

/// Prints the exact request one narration call sends to Gemini (system
/// prompt + context message). Run with:
/// `cargo test -p agent --test bugfix_tests print_example_context -- --ignored --nocapture`
#[tokio::test]
#[ignore]
async fn print_example_context() {
    let client = MockGeminiClient::new(vec![text_response("ok")]);
    let history = vec![
        ConversationTurn::user("How is Ratnaveer doing?"),
        ConversationTurn::assistant("RATNAVEER.NS is up strongly."),
    ];
    let opts = NarrationOptions {
        user_question: Some("What happened to GOLDCASE in the IL&FS scenario?".to_string()),
        ..Default::default()
    };
    let mut t = shock_trace();
    t.outputs = serde_json::json!({"result": {"portfolio_pnl_inr": -9500.0, "per_holding": [
        {"ticker": "GOLDCASE.NS", "pnl_inr": -9000.0}, {"ticker": "KIOCL.NS", "pnl_inr": -5000.0},
        {"ticker": "BLS.NS", "pnl_inr": -4000.0}, {"ticker": "RELIANCE.NS", "pnl_inr": 2000.0}],
        "factor_attribution_log_inr": {"GOLD_USD": -8000.0, "MARKET": -3000.0}}});
    narrate_tools_with_options(&client, &[t], None, &history, &opts).await.unwrap();
    let req = client.last_request();
    println!("===== SYSTEM PROMPT =====\n{}\n", system_prompt(&req));
    for (i, c) in req.contents.iter().enumerate() {
        println!("===== CONTENT {i} (role: {}) =====\n{}\n", c.role.as_deref().unwrap_or("?"), content_text(&req, i));
    }
}


// ---- final fixes ---------------------------------------------------------

fn ctx() -> compute::context::ExperimentContext {
    compute::context::ExperimentContext {
        store: std::sync::Arc::new(store::SnapshotStore::open(":memory:").unwrap()),
        portfolio_hash: String::new(),
        policy: None,
    }
}

#[test]
fn a_planner_param_that_does_not_deserialize_falls_back_to_defaults_instead_of_failing() {
    use agent::orchestrator::{build_experiment_or_defaults, RiskTool};
    use compute::experiments::Experiment;
    // The live bug: a date range where an integer window belongs.
    let bad = serde_json::json!({"window": "2020-03-01:2020-04-01"});
    let e = build_experiment_or_defaults(RiskTool::PortfolioPerformance, "portfolio_performance", &bad, &portfolio(), &ctx(), None)
        .expect("must fall back, not fail");
    match e {
        Experiment::PortfolioPerformance(i) => assert_eq!(i.window, None, "defaults, not the bad value"),
        other => panic!("{other:?}"),
    }
    for (tool, name) in [(RiskTool::CurrentRisk, "current_risk"), (RiskTool::CvarRebalance, "cvar_rebalance")] {
        let bad = serde_json::json!({"window": "bad-string", "turnover_limit": 0.0});
        // current_risk has no required params; cvar still needs turnover_limit, which is
        // dropped with the rest -- so only the first must succeed.
        let r = build_experiment_or_defaults(tool, name, &bad, &portfolio(), &ctx(), None);
        if name == "current_risk" {
            assert!(r.is_ok());
        }
    }
}

#[test]
fn a_historical_stress_fallback_keeps_its_scenario_and_a_tool_with_required_params_still_errors() {
    use agent::orchestrator::{build_experiment_or_defaults, RiskTool};
    use compute::experiments::Experiment;
    let e = build_experiment_or_defaults(
        RiskTool::HistoricalStress,
        "historical_stress",
        &serde_json::json!({"scenario_id": "covid_crash", "window": "bad"}),
        &portfolio(),
        &ctx(),
        None,
    )
    .unwrap();
    match e {
        Experiment::FactorShock(i) => assert!(!i.shocks_pct.is_empty(), "covid shocks kept"),
        other => panic!("{other:?}"),
    }
    // factor_shock has no defaults for its shocks: the original error is returned.
    let r = build_experiment_or_defaults(
        RiskTool::FactorShock,
        "factor_shock",
        &serde_json::json!({"shocks_pct": "not-a-map"}),
        &portfolio(),
        &ctx(),
        None,
    );
    assert!(r.is_err());
}

#[test]
fn the_planning_prompt_has_the_window_cvar_and_sector_rules() {
    assert!(P.contains("The 'window' parameter must always be an integer"));
    assert!(P.contains("Invalid: '2020-03-01:2020-04-01'"));
    assert!(P.contains("select \
") || P.contains("cvar_rebalance with {'turnover_limit': 0.0, 'per_name_cap': 1.0, 'confidence_level': 0.95}"));
    assert!(P.contains("Do NOT select current_risk for CVaR questions."));
    assert!(P.contains("{'sector_question': true}"));
    assert!(P.contains("never guess which holdings belong to a sector"));
}

#[tokio::test]
async fn a_cvar_question_plans_a_read_only_cvar_rebalance() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "cvar_rebalance", "params": {"turnover_limit": 0.0, "per_name_cap": 1.0, "confidence_level": 0.95}, "reason": "report CVaR"}]"#,
    )]);
    let (plans, _) = plan_tools(&client, "What is my CVaR at 95% confidence?", &[]).await.unwrap();
    assert_eq!(plans[0].tool, "cvar_rebalance");
    assert_eq!(plans[0].params["turnover_limit"], 0.0);
    // ...and those params build a real CvarRebalance experiment.
    let e = agent::orchestrator::build_experiment_or_defaults(
        agent::orchestrator::RiskTool::CvarRebalance,
        "cvar_rebalance",
        &plans[0].params,
        &portfolio(),
        &ctx(),
        None,
    )
    .unwrap();
    match e {
        compute::experiments::Experiment::CvarRebalance(i) => {
            assert_eq!(i.turnover_limit, 0.0);
            assert_eq!(i.per_name_cap, Some(1.0));
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_sector_question_context_forbids_classifying_holdings_and_supplies_rankings() {
    let perf = trace(
        "PortfolioPerformance",
        serde_json::json!({"holding_returns": {
            "DATAPATTNS.NS": {"total_return_pct": 10.0, "max_drawdown_pct": -20.0, "annualized_vol_pct": 55.0},
            "BEL.NS": {"total_return_pct": 30.0, "max_drawdown_pct": -10.0, "annualized_vol_pct": 30.0},
            "KIOCL.NS": {"total_return_pct": -27.0, "max_drawdown_pct": -40.0, "annualized_vol_pct": 45.0},
        }}),
    );
    let client = MockGeminiClient::new(vec![text_response("ok")]);
    let opts = NarrationOptions {
        user_question: Some("Which of my PSU stocks is riskiest?".to_string()),
        sector_question: true,
        ..Default::default()
    };
    narrate_tools_with_options(&client, &[perf], None, &[], &opts).await.unwrap();
    let req = client.last_request();
    let ctx = content_text(&req, req.contents.len() - 1);
    assert!(ctx.contains("sector_question: true"));
    assert!(ctx.contains("I don't have sector classification data"));
    assert!(ctx.contains("do NOT classify any holding as PSU"));
    assert!(ctx.contains("Holdings by annualized volatility (highest first, top 8): DATAPATTNS.NS 55.0%, KIOCL.NS 45.0%, BEL.NS 30.0%"));
    assert!(ctx.contains("Holdings by total return (lowest first, bottom 8): KIOCL.NS -27.0%"));
    // The system prompt carries the rule too (Rule 4).
    assert!(system_prompt(&req).contains("If sector_question is true in the context:"));
    assert!(system_prompt(&req).contains("Do NOT classify any holding into a sector."));
}

#[test]
fn the_sector_flag_is_read_from_the_plan_params() {
    use agent::orchestrator::is_sector_question;
    let plan = |p: serde_json::Value| vec![agent::ToolPlan { tool: "portfolio_performance".into(), params: p, reason: String::new() }];
    assert!(is_sector_question(&plan(serde_json::json!({"sector_question": true}))));
    assert!(!is_sector_question(&plan(serde_json::json!({}))));
    assert!(!is_sector_question(&plan(serde_json::json!({"sector_question": false}))));
}
