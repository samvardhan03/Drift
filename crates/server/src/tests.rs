use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use chrono::NaiveDate;
use http_body_util::BodyExt;
use tower::ServiceExt;

use compute::data::DataQuality;
use compute::experiments::{Experiment, Holding, Portfolio, RiskDecompositionInput};
use compute::model::Frequency;
use compute::regime::RegimeState;
use compute::trace::{DataWindow, EvidenceTrace, ModelParams};

use crate::backend::{Backend, BackendError};
use crate::build_router;
use crate::routes::AppState;

fn sample_portfolio() -> Portfolio {
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

/// A fixed `RegimeState`, standing in for a real HMM fit -- regime is
/// always-on now (see `compute`'s always-on regime), so every mock trace
/// used by these tests carries one, matching what a real experiment
/// response always has.
fn sample_regime_state() -> RegimeState {
    RegimeState {
        current_regime: 0,
        current_label: "Bull",
        smoothed_probs: [0.7, 0.2, 0.1],
        viterbi_sequence: vec![0; 252],
        obs_count_per_regime: [200, 40, 12],
        log_likelihood: -123.45,
        n_iter: 12,
        smoothing_note: "full-history smoothed, not suitable for live trading signals",
    }
}

fn sample_trace() -> EvidenceTrace {
    EvidenceTrace {
        id: compute::trace::new_trace_id(),
        experiment: "RiskDecomposition".to_string(),
        inputs: serde_json::json!({}),
        data_as_of: "2026-09-24T00:00:00Z".to_string(),
        data_window: DataWindow {
            frequency: Frequency::Daily,
            window_periods: 252,
            start: NaiveDate::from_ymd_opt(2025, 9, 16).unwrap(),
            end: NaiveDate::from_ymd_opt(2026, 9, 24).unwrap(),
        },
        data_quality: DataQuality {
            date_range_start: NaiveDate::from_ymd_opt(2021, 9, 27).unwrap(),
            date_range_end: NaiveDate::from_ymd_opt(2026, 9, 24).unwrap(),
            trading_days: 1234,
            per_series: vec![],
        },
        model_params: ModelParams {
            frequency: Frequency::Daily,
            window_periods: 252,
            factor_names: vec!["MARKET".to_string()],
            shrinkage_intensity: 0.0374,
            annualization_factor: 252.0,
            regime_state: Some(sample_regime_state()),
            regime_fallback_warnings: vec![],
            cap_source: None,
        },
        outputs: serde_json::json!({ "result": { "portfolio_vol_annualized": 0.1552 } }),
        invariants: vec![],
        engine_version: "0.1.0".to_string(),
        engine_commit: compute::trace::engine_commit(),
        scenario_provenance: None,
        parent_trace_ids: Vec::new(),
        baseline_model_params: None,
        policy_result: None,
    }
}

fn sample_execution_trace() -> agent::AgentExecutionTrace {
    agent::AgentExecutionTrace {
        id: "exec-trace-id".to_string(),
        created_at: "2026-01-01T00:00:00Z".to_string(),
        user_message: "what's my portfolio risk?".to_string(),
        planning_response_raw: r#"[{"tool":"current_risk","params":{},"reason":"test"}]"#
            .to_string(),
        tool_plans: vec![agent::ToolPlan {
            tool: "current_risk".to_string(),
            params: serde_json::json!({}),
            reason: "test".to_string(),
        }],
        tool_results: vec![agent::ToolResult {
            tool: "current_risk".to_string(),
            trace_id: "trace-id".to_string(),
            latency_ms: 5,
            success: true,
            error: None,
        }],
        narration: "Vol is 15.5% annualised.".to_string(),
        grounding_status: agent::GroundingStatus {
            passed: true,
            warnings: vec![],
            retry_count: 0,
        },
        suggestion: "What if I reduce my turnover to 20%?".to_string(),
        total_latency_ms: 10,
        gemini_calls: 3,
    }
}

/// A `Backend` that returns fixed responses (or a fixed error) — no
/// network access to Yahoo Finance or Gemini.
struct MockBackend {
    experiment_result: Option<EvidenceTrace>,
    /// Takes priority over `experiment_result` when set -- lets a test
    /// exercise the `BackendError` -> HTTP status/code mapping (e.g.
    /// `RiskDrift`'s "no prior snapshot" 422) without needing a real
    /// backend to actually produce that error.
    experiment_error: Option<BackendError>,
    ask_result: Option<agent::pipeline::PipelineResult>,
    received_conversation_history: Mutex<Option<Vec<agent::ConversationTurn>>>,
    received_policy: Mutex<Option<Option<compute::policy::RiskPolicy>>>,
}

#[async_trait::async_trait]
impl Backend for MockBackend {
    async fn run_experiment(
        &self,
        _experiment: Experiment,
        _portfolio: Portfolio,
        policy: Option<compute::policy::RiskPolicy>,
    ) -> Result<EvidenceTrace, BackendError> {
        *self.received_policy.lock().unwrap() = Some(policy);
        if let Some(err) = &self.experiment_error {
            return Err(err.clone());
        }
        self.experiment_result.clone().ok_or_else(|| {
            BackendError::Internal("no mock experiment result configured".to_string())
        })
    }

    async fn run_ask(
        &self,
        _portfolio: Portfolio,
        _message: String,
        conversation_history: Vec<agent::ConversationTurn>,
        policy: Option<compute::policy::RiskPolicy>,
    ) -> Result<agent::pipeline::PipelineResult, BackendError> {
        *self.received_conversation_history.lock().unwrap() = Some(conversation_history);
        *self.received_policy.lock().unwrap() = Some(policy);
        self.ask_result
            .as_ref()
            .map(|r| agent::pipeline::PipelineResult {
                experiment: r.experiment.clone(),
                trace: r.trace.clone(),
                traces: r.traces.clone(),
                tool_plans: r.tool_plans.clone(),
                narration: agent::grounding::GroundedNarration {
                    narration: r.narration.narration.clone(),
                    grounding_warnings: r.narration.grounding_warnings.clone(),
                },
                assistant_turn: r.assistant_turn.clone(),
                suggestion: r.suggestion.clone(),
                execution_trace: r.execution_trace.clone(),
            })
            .ok_or_else(|| BackendError::Internal("no mock ask result configured".to_string()))
    }
}

fn app_with_backend(backend: MockBackend) -> axum::Router {
    app_with_backend_and_store(backend).0
}

/// Like `app_with_backend`, but also hands back the `SnapshotStore` handle
/// so a test can inspect what got persisted directly (rather than only
/// through the HTTP responses), e.g. after `POST /experiment`, which never
/// echoes a `result_id` back to the caller.
fn app_with_backend_and_store(backend: MockBackend) -> (axum::Router, Arc<store::SnapshotStore>) {
    let store = Arc::new(store::SnapshotStore::open(":memory:").unwrap());
    let app = build_router(AppState {
        backend: Arc::new(backend),
        store: store.clone(),
    });
    (app, store)
}

/// Whether `needle` appears in a rendered PDF's actual text content.
/// `printpdf`'s `PdfSaveOptions::default()` (used by `pdf::render_report`)
/// FlateDecode-compresses content streams, so the raw bytes don't contain
/// `ShowText` string literals verbatim -- `lopdf::Document::extract_text`
/// decompresses and decodes them properly.
fn pdf_contains_text(bytes: &[u8], needle: &str) -> bool {
    let doc = lopdf::Document::load_mem(bytes).expect("rendered report should be a valid PDF");
    let page_numbers: Vec<u32> = doc.get_pages().keys().copied().collect();
    let text = doc
        .extract_text(&page_numbers)
        .expect("failed to extract text from rendered PDF");
    text.contains(needle)
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn health_returns_200_and_expected_json() {
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn cors_preflight_on_health_returns_200_with_allow_origin_header() {
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("OPTIONS")
                .uri("/health")
                .header("origin", "https://example.com")
                .header("access-control-request-method", "GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .contains_key("access-control-allow-origin"),
        "expected an Access-Control-Allow-Origin header on the preflight response, got {:?}",
        response.headers()
    );
}

#[tokio::test]
async fn experiment_with_valid_request_returns_a_trace() {
    let app = app_with_backend(MockBackend {
        experiment_result: Some(sample_trace()),
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": { "type": "RiskDecomposition" },
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["experiment"], "RiskDecomposition");
    assert_eq!(
        body["outputs"]["result"]["portfolio_vol_annualized"],
        0.1552
    );
}

#[tokio::test]
async fn experiment_with_weights_not_summing_to_one_returns_400() {
    let app = app_with_backend(MockBackend {
        experiment_result: Some(sample_trace()),
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let bad_portfolio = serde_json::json!({
        "holdings": [
            {"ticker": "RELIANCE.NS", "weight": 0.6},
            {"ticker": "TCS.NS", "weight": 0.6},
        ],
        "total_value_inr": 1000000.0,
    });
    let req_body = serde_json::json!({
        "portfolio": bad_portfolio,
        "experiment": { "type": "RiskDecomposition" },
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "invalid_portfolio");
    assert!(body["error"].as_str().unwrap().contains("sum to 1.0"));
}

#[tokio::test]
async fn ask_with_mocked_pipeline_returns_grounding_warnings() {
    let pipeline_result = agent::pipeline::PipelineResult {
        experiment: Experiment::RiskDecomposition(RiskDecompositionInput {
            portfolio: sample_portfolio(),
            frequency: Frequency::Daily,
            window: None,
        }),
        trace: sample_trace(),
        traces: vec![sample_trace()],
        tool_plans: vec![],
        narration: agent::grounding::GroundedNarration {
            narration: "Vol is 99% (unverified).".to_string(),
            grounding_warnings: vec![
                "unverified number '99%' at byte position 7 in the narration".to_string(),
            ],
        },
        assistant_turn: agent::ConversationTurn::assistant("Vol is 99% (unverified)."),
        suggestion: "What if I reduce my turnover to 20%?".to_string(),
        execution_trace: sample_execution_trace(),
    };
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: Some(pipeline_result),
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "message": "what's my portfolio risk?",
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ask")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["narration"], "Vol is 99% (unverified).");
    assert_eq!(body["grounding_warnings"].as_array().unwrap().len(), 1);
    assert_eq!(body["experiment"]["type"], "RiskDecomposition");
    assert_eq!(body["assistant_turn"]["role"], "assistant");
    assert_eq!(
        body["assistant_turn"]["content"],
        "Vol is 99% (unverified)."
    );
    assert_eq!(body["suggestion"], "What if I reduce my turnover to 20%?");
    assert!(
        body["result_id"].is_string(),
        "expected a result_id field, got {body:?}"
    );
}

#[tokio::test]
async fn ask_with_non_empty_conversation_history_forwards_it_to_the_backend() {
    let pipeline_result = agent::pipeline::PipelineResult {
        experiment: Experiment::RiskDecomposition(RiskDecompositionInput {
            portfolio: sample_portfolio(),
            frequency: Frequency::Daily,
            window: None,
        }),
        trace: sample_trace(),
        traces: vec![sample_trace()],
        tool_plans: vec![],
        narration: agent::grounding::GroundedNarration {
            narration: "Vol is 15.5% annualised.".to_string(),
            grounding_warnings: vec![],
        },
        assistant_turn: agent::ConversationTurn::assistant("Vol is 15.5% annualised."),
        suggestion: "Now reduce my tail risk with 20% turnover?".to_string(),
        execution_trace: sample_execution_trace(),
    };
    let backend = Arc::new(MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: Some(pipeline_result),
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });
    let app = build_router(AppState {
        backend: backend.clone(),
        store: Arc::new(store::SnapshotStore::open(":memory:").unwrap()),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "message": "now reduce my tail risk, I can tolerate 20% turnover",
        "conversation_history": [
            {"role": "user", "content": "where is my risk concentrated?"},
            {"role": "assistant", "content": "Vol is 15.5% annualised."},
        ],
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ask")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let received = backend.received_conversation_history.lock().unwrap();
    let history = received.as_ref().expect("run_ask should have been called");
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].role, "user");
    assert_eq!(history[1].role, "assistant");
    assert_eq!(history[1].content, "Vol is 15.5% annualised.");
}

#[tokio::test]
async fn report_route_returns_pdf_for_a_result_stored_by_a_prior_ask() {
    let pipeline_result = agent::pipeline::PipelineResult {
        experiment: Experiment::RiskDecomposition(RiskDecompositionInput {
            portfolio: sample_portfolio(),
            frequency: Frequency::Daily,
            window: None,
        }),
        trace: sample_trace(),
        traces: vec![sample_trace()],
        tool_plans: vec![],
        narration: agent::grounding::GroundedNarration {
            narration: "Vol is 15.5% annualised.".to_string(),
            grounding_warnings: vec![],
        },
        assistant_turn: agent::ConversationTurn::assistant("Vol is 15.5% annualised."),
        suggestion: "Now reduce my tail risk?".to_string(),
        execution_trace: sample_execution_trace(),
    };
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: Some(pipeline_result),
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let ask_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "message": "where is my risk concentrated?",
    });
    let ask_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ask")
                .header("content-type", "application/json")
                .body(Body::from(ask_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ask_response.status(), StatusCode::OK);
    let ask_json = body_json(ask_response).await;
    let result_id = ask_json["result_id"]
        .as_str()
        .expect("result_id should be a string")
        .to_string();

    let report_response = app
        .oneshot(
            Request::builder()
                .uri(format!("/report/{result_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(report_response.status(), StatusCode::OK);
    let content_type = report_response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(content_type, "application/pdf");
    let bytes = report_response
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    assert!(bytes.starts_with(b"%PDF"), "expected a PDF file signature");
    assert!(
        pdf_contains_text(&bytes, "Vol is 15.5% annualised."),
        "expected the stored /ask narration text to appear in the report PDF"
    );
}

#[tokio::test]
async fn report_route_returns_404_for_an_unknown_id() {
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let unknown_id = uuid::Uuid::new_v4();
    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/report/{unknown_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn execution_trace_route_returns_404_for_an_unknown_id() {
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/execution-trace/nonexistent-id")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn execution_trace_route_returns_the_trace_stored_by_a_prior_ask() {
    let pipeline_result = agent::pipeline::PipelineResult {
        experiment: Experiment::RiskDecomposition(RiskDecompositionInput {
            portfolio: sample_portfolio(),
            frequency: Frequency::Daily,
            window: None,
        }),
        trace: sample_trace(),
        traces: vec![sample_trace()],
        tool_plans: vec![],
        narration: agent::grounding::GroundedNarration {
            narration: "Vol is 15.5% annualised.".to_string(),
            grounding_warnings: vec![],
        },
        assistant_turn: agent::ConversationTurn::assistant("Vol is 15.5% annualised."),
        suggestion: "What if I reduce my turnover to 20%?".to_string(),
        execution_trace: sample_execution_trace(),
    };
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: Some(pipeline_result),
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "message": "what's my portfolio risk?",
    });
    let ask_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ask")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ask_response.status(), StatusCode::OK);
    let ask_body = body_json(ask_response).await;
    let execution_trace_id = ask_body["agent_execution_trace"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(ask_body["agent_execution_trace"]["gemini_calls"], 3);

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/execution-trace/{execution_trace_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["id"], execution_trace_id);
    assert_eq!(body["narration"], "Vol is 15.5% annualised.");
    assert_eq!(body["gemini_calls"], 3);
}

#[tokio::test]
async fn scenarios_route_returns_three_scenarios_with_expected_fields() {
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/scenarios")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    let scenarios = body.as_array().expect("expected a JSON array");
    assert_eq!(scenarios.len(), 3);
    for scenario in scenarios {
        assert!(scenario["id"].is_string());
        assert!(scenario["name"].is_string());
        assert!(scenario["shocks_pct"].is_object());
    }
}

#[tokio::test]
async fn static_route_returns_200_and_html_content_type() {
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(content_type.starts_with("text/html"));
}

#[tokio::test]
async fn experiment_still_works_end_to_end_with_snapshot_store() {
    let (app, store) = app_with_backend_and_store(MockBackend {
        experiment_result: Some(sample_trace()),
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": { "type": "RiskDecomposition" },
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let recent = store.list_recent(10).unwrap();
    assert_eq!(
        recent.len(),
        1,
        "POST /experiment should have inserted exactly one snapshot"
    );
    let snapshot = &recent[0];
    assert_eq!(snapshot.experiment_type, "RiskDecomposition");
    assert_eq!(snapshot.portfolio_vol_annualized, Some(0.1552));
    assert_eq!(snapshot.regime_label.as_deref(), Some("Bull"));
    assert!(snapshot.smoothed_probs.is_some());

    let fetched = store.get(&snapshot.id).unwrap();
    assert!(
        fetched.is_some(),
        "the just-inserted snapshot should be retrievable by id"
    );
}

#[tokio::test]
async fn report_route_retrieves_an_experiment_originated_snapshot() {
    let (app, store) = app_with_backend_and_store(MockBackend {
        experiment_result: Some(sample_trace()),
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": { "type": "RiskDecomposition" },
    });
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    let id = store.list_recent(1).unwrap()[0].id.clone();
    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/report/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert!(bytes.starts_with(b"%PDF"), "expected a PDF file signature");
    assert!(
        pdf_contains_text(&bytes, "No narrative"),
        "an /experiment-originated report has no narration, so should render the placeholder text"
    );
}

#[tokio::test]
async fn regime_state_is_non_null_in_every_experiment_response() {
    let app = app_with_backend(MockBackend {
        experiment_result: Some(sample_trace()),
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": { "type": "RiskDecomposition" },
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert!(
        !body["model_params"]["regime_state"].is_null(),
        "regime_state should always be present, got {body:?}"
    );
    assert_eq!(
        body["model_params"]["regime_state"]["current_label"],
        "Bull"
    );
}

/// Builds a `multipart/form-data` body with a single "file" field, the way
/// a browser's `<input type="file">` upload would.
fn multipart_body(filename: &str, content_type: &str, bytes: &[u8]) -> (String, Vec<u8>) {
    let boundary = "----driftriskcopilotboundary";
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn upload(
    app: axum::Router,
    filename: &str,
    content_type: &str,
    bytes: &[u8],
) -> axum::response::Response {
    let (content_type_header, body) = multipart_body(filename, content_type, bytes);
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/portfolio/upload")
            .header("content-type", content_type_header)
            .body(Body::from(body))
            .unwrap(),
    )
    .await
    .unwrap()
}

fn empty_backend_app() -> axum::Router {
    app_with_backend(MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    })
}

#[tokio::test]
async fn upload_weight_based_csv_returns_200_and_correct_portfolio() {
    let csv = "ticker,weight\nRELIANCE.NS,0.6\nTCS.NS,0.4\n";
    let response = upload(
        empty_backend_app(),
        "portfolio.csv",
        "text/csv",
        csv.as_bytes(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["layout_detected"], "weight");
    assert_eq!(body["row_count"], 2);
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    assert_eq!(holdings.len(), 2);
    assert_eq!(holdings[0]["ticker"], "RELIANCE.NS");
    assert_eq!(holdings[0]["weight"], 0.6);
    assert_eq!(holdings[1]["ticker"], "TCS.NS");
    assert_eq!(holdings[1]["weight"], 0.4);
}

#[tokio::test]
async fn upload_value_based_csv_returns_200_and_weights_sum_to_one() {
    let csv = "ticker,shares,avg_price_inr\nRELIANCE.NS,10,2850.00\nHDFCBANK.NS,25,1640.00\n";
    let response = upload(
        empty_backend_app(),
        "portfolio.csv",
        "text/csv",
        csv.as_bytes(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["layout_detected"], "value");
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    let sum: f64 = holdings.iter().map(|h| h["weight"].as_f64().unwrap()).sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "weights should sum to 1.0 within 1e-6, got {sum}"
    );
}

#[tokio::test]
async fn upload_xlsx_returns_200() {
    let bytes = include_bytes!("../tests/fixtures/sample_portfolio.xlsx");
    let response = upload(
        empty_backend_app(),
        "portfolio.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        bytes,
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["layout_detected"], "weight");
    assert_eq!(body["row_count"], 2);
}

#[tokio::test]
async fn upload_unsupported_file_type_returns_400() {
    let response = upload(
        empty_backend_app(),
        "portfolio.txt",
        "text/plain",
        b"not a real portfolio file",
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "unsupported_file_type");
}

#[tokio::test]
async fn upload_weights_not_summing_to_one_returns_400() {
    let csv = "ticker,weight\nRELIANCE.NS,0.6\nTCS.NS,0.6\n";
    let response = upload(
        empty_backend_app(),
        "portfolio.csv",
        "text/csv",
        csv.as_bytes(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "invalid_portfolio");
}

#[tokio::test]
async fn upload_single_holding_returns_400() {
    let csv = "ticker,weight\nRELIANCE.NS,1.0\n";
    let response = upload(
        empty_backend_app(),
        "portfolio.csv",
        "text/csv",
        csv.as_bytes(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "invalid_portfolio");
}

fn sample_risk_drift_trace(
    vol_before: f64,
    vol_after: f64,
    regime_before: &str,
    regime_after: &str,
) -> EvidenceTrace {
    let mut trace = sample_trace();
    trace.experiment = "RiskDrift".to_string();
    trace.outputs = serde_json::json!({
        "result": {
            "vol_before": vol_before,
            "vol_after": vol_after,
            "vol_change_abs": vol_after - vol_before,
            "vol_change_pct": (vol_after / vol_before - 1.0) * 100.0,
            "regime_before": regime_before,
            "regime_after": regime_after,
            "days_elapsed": 14,
        }
    });
    trace
}

fn risk_drift_snapshot(
    portfolio_hash: &str,
    vol_before: f64,
    vol_after: f64,
) -> store::RiskSnapshot {
    let trace = sample_risk_drift_trace(vol_before, vol_after, "Bull", "Bear");
    store::RiskSnapshot {
        id: String::new(),
        created_at: String::new(),
        portfolio_hash: portfolio_hash.to_string(),
        experiment_type: "RiskDrift".to_string(),
        engine_version: "0.1.0".to_string(),
        regime_label: Some("Bear".to_string()),
        smoothed_probs: Some([0.2, 0.7, 0.1]),
        portfolio_vol_annualized: None,
        cvar_historical: None,
        trace_json: serde_json::to_string(&trace).unwrap(),
        narration: None,
        suggestion: None,
        grounding_warnings: None,
    }
}

#[tokio::test]
async fn experiment_risk_drift_with_a_pre_inserted_baseline_returns_200_with_non_null_vol_change() {
    // The baseline snapshot's presence in the store isn't exercised by
    // MockBackend (it never calls compute::drift itself -- that logic is
    // covered hermetically by compute's own drift_tests.rs); this test
    // covers the HTTP plumbing: a RiskDrift-shaped trace flows through
    // POST /experiment untouched and is stored/returned correctly.
    let (app, store) = app_with_backend_and_store(MockBackend {
        experiment_result: Some(sample_risk_drift_trace(0.12, 0.18, "Bull", "Bear")),
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });
    store
        .insert(&risk_drift_snapshot("baseline-hash", 0.10, 0.12))
        .unwrap();

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": { "type": "RiskDrift", "baseline_snapshot_id": null },
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["experiment"], "RiskDrift");
    assert!(!body["outputs"]["result"]["vol_change_abs"].is_null());
    assert_eq!(body["outputs"]["result"]["vol_before"], 0.12);
    assert_eq!(body["outputs"]["result"]["vol_after"], 0.18);
}

#[tokio::test]
async fn experiment_risk_drift_with_no_prior_snapshot_returns_422() {
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        experiment_error: Some(crate::backend::BackendError::Unrecognised(
            "No prior snapshot found for this portfolio. Run a RiskDecomposition first to \
             establish a baseline."
                .to_string(),
        )),
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": { "type": "RiskDrift" },
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = body_json(response).await;
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("No prior snapshot found"));
}

#[tokio::test]
async fn drift_route_returns_200_and_empty_snapshots_when_none_exist() {
    let app = empty_backend_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/drift?portfolio=unknown-hash")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["snapshots"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn drift_route_returns_summaries_not_full_traces_after_inserting_two_snapshots() {
    let (app, store) = app_with_backend_and_store(MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });
    store
        .insert(&risk_drift_snapshot("drift-hash", 0.10, 0.12))
        .unwrap();
    store
        .insert(&risk_drift_snapshot("drift-hash", 0.12, 0.20))
        .unwrap();
    // A non-RiskDrift snapshot for the same portfolio must be excluded.
    let mut other = risk_drift_snapshot("drift-hash", 0.0, 0.0);
    other.experiment_type = "RiskDecomposition".to_string();
    store.insert(&other).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/drift?portfolio=drift-hash")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    let snapshots = body["snapshots"].as_array().unwrap();
    assert_eq!(
        snapshots.len(),
        2,
        "expected exactly the 2 RiskDrift snapshots, got {snapshots:?}"
    );
    for snapshot in snapshots {
        assert!(snapshot["id"].is_string());
        assert!(snapshot["created_at"].is_string());
        assert!(snapshot["vol_before"].is_number());
        assert!(snapshot["vol_after"].is_number());
        assert!(snapshot["vol_change_pct"].is_number());
        assert_eq!(snapshot["regime_before"], "Bull");
        assert_eq!(snapshot["regime_after"], "Bear");
        assert!(snapshot["days_elapsed"].is_number());
        // Summary only -- no full trace/outputs object present.
        assert!(snapshot.get("trace_json").is_none());
        assert!(snapshot.get("outputs").is_none());
    }
}

fn sample_reverse_stress_trace(portfolio_pnl_inr: f64, loss_threshold_inr: f64) -> EvidenceTrace {
    let mut trace = sample_trace();
    trace.experiment = "ReverseStress".to_string();
    trace.outputs = serde_json::json!({
        "result": {
            "shock_vector": { "MARKET": -18.0, "USDINR": 12.0, "BRENT": 0.0, "GOLD_USD": 0.0, "RATES_PROXY": 0.0 },
            "shock_vector_log": { "MARKET": -0.198, "USDINR": 0.113, "BRENT": 0.0, "GOLD_USD": 0.0, "RATES_PROXY": 0.0 },
            "mahalanobis_severity": 1.8,
            "severity_label": "1\u{2013}2\u{3c3}",
            "portfolio_pnl_inr": portfolio_pnl_inr,
            "loss_threshold_inr": loss_threshold_inr,
            "holding_pnl": { "RELIANCE.NS": portfolio_pnl_inr * 0.6, "TCS.NS": portfolio_pnl_inr * 0.4 },
            "factor_attribution": { "MARKET": portfolio_pnl_inr * 0.7, "USDINR": portfolio_pnl_inr * 0.3 },
            "most_vulnerable_holdings": ["RELIANCE.NS", "TCS.NS"],
            "solver_status": "converged",
            "n_iterations": 12,
            "gradient_norm_final": 1e-7,
            "linearisation_error_inr": 500.0,
            "factor_bounds_used": { "MARKET": [-40.0, 0.0], "USDINR": [-5.0, 20.0], "BRENT": [-60.0, 100.0], "GOLD_USD": [-20.0, 40.0], "RATES_PROXY": [-10.0, 10.0] },
        }
    });
    trace
}

#[tokio::test]
async fn experiment_reverse_stress_with_valid_input_returns_200_with_negative_pnl() {
    let app = app_with_backend(MockBackend {
        experiment_result: Some(sample_reverse_stress_trace(-505_000.0, 500_000.0)),
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": { "type": "ReverseStress", "loss_threshold_inr": 500_000.0 },
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["experiment"], "ReverseStress");
    let pnl = body["outputs"]["result"]["portfolio_pnl_inr"]
        .as_f64()
        .unwrap();
    assert!(
        pnl.is_finite() && pnl < 0.0,
        "expected a present, negative portfolio_pnl_inr, got {pnl}"
    );
}

#[tokio::test]
async fn experiment_reverse_stress_with_infeasible_threshold_returns_422() {
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        experiment_error: Some(crate::backend::BackendError::Unrecognised(
            "The loss threshold \u{20b9}1,00,00,000 cannot be breached within the specified \
             factor bounds. Maximum feasible loss is \u{20b9}12,00,000."
                .to_string(),
        )),
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": { "type": "ReverseStress", "loss_threshold_inr": 10_000_000.0 },
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = body_json(response).await;
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("cannot be breached"));
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("Maximum feasible loss"));
}

fn sample_policy_result(all_passed: bool) -> serde_json::Value {
    if all_passed {
        serde_json::json!({
            "checks": [
                {"rule": "max_vol_annualized", "limit": 1.0, "actual": 0.1552, "passed": true, "breach_magnitude": 0.0, "evidence": "Portfolio annualized volatility is 15.5% (limit: 100.0%)"},
            ],
            "all_passed": true,
            "breach_count": 0,
            "most_severe_breach": null,
        })
    } else {
        serde_json::json!({
            "checks": [
                {"rule": "max_vol_annualized", "limit": 0.01, "actual": 0.1552, "passed": false, "breach_magnitude": 0.1452, "evidence": "Portfolio annualized volatility is 15.5% (limit: 1.0%)"},
            ],
            "all_passed": false,
            "breach_count": 1,
            "most_severe_breach": {"rule": "max_vol_annualized", "limit": 0.01, "actual": 0.1552, "passed": false, "breach_magnitude": 0.1452, "evidence": "Portfolio annualized volatility is 15.5% (limit: 1.0%)"},
        })
    }
}

fn sample_policy_check_trace(all_passed: bool) -> EvidenceTrace {
    let mut trace = sample_trace();
    trace.experiment = "PolicyCheck".to_string();
    trace.outputs = serde_json::json!({
        "result": {
            "policy_result": sample_policy_result(all_passed),
            "regime_label": "Bull",
            "portfolio_vol": 0.1552,
            "portfolio_cvar_95": 0.05,
            "max_position_weight": 0.6,
            "max_factor_share": 0.4,
            "scenario_losses": {},
        }
    });
    trace
}

#[tokio::test]
async fn experiment_policy_check_with_a_tight_limit_returns_200_with_all_passed_false() {
    let app = app_with_backend(MockBackend {
        experiment_result: Some(sample_policy_check_trace(false)),
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": { "type": "PolicyCheck", "policy": { "max_vol_annualized": 0.01 } },
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(
        body["outputs"]["result"]["policy_result"]["all_passed"],
        false
    );
}

#[tokio::test]
async fn experiment_policy_check_with_a_loose_limit_returns_200_with_all_passed_true() {
    let app = app_with_backend(MockBackend {
        experiment_result: Some(sample_policy_check_trace(true)),
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": { "type": "PolicyCheck", "policy": { "max_vol_annualized": 1.0 } },
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(
        body["outputs"]["result"]["policy_result"]["all_passed"],
        true
    );
    assert_eq!(
        body["outputs"]["result"]["policy_result"]["breach_count"],
        0
    );
}

#[tokio::test]
async fn experiment_with_an_attached_policy_runs_the_passive_check_and_returns_policy_result() {
    let (app, _store) = app_with_backend_and_store(MockBackend {
        // The mock stands in for what a real backend would do: attach
        // policy_result to whatever experiment actually ran (here
        // RiskDecomposition, not PolicyCheck itself).
        experiment_result: Some({
            let mut trace = sample_trace();
            trace.policy_result =
                Some(serde_json::from_value(sample_policy_result(false)).unwrap());
            trace
        }),
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": { "type": "RiskDecomposition" },
        "policy": { "max_vol_annualized": 0.01 },
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["experiment"], "RiskDecomposition");
    assert_eq!(body["policy_result"]["all_passed"], false);
    assert_eq!(body["policy_result"]["breach_count"], 1);
}
