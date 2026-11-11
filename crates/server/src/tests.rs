use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
use crate::routes::{AppState, UpstoxConfig};
use crate::upstox::{UpstoxClient, UpstoxError, UpstoxHolding};

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
        transition_matrix: [[0.0; 3]; 3],
        regime_forecast: vec![],
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
            short_history_tickers: vec![],
        },
        outputs: serde_json::json!({ "result": { "portfolio_vol_annualized": 0.1552 } }),
        invariants: vec![],
        engine_version: "0.1.0".to_string(),
        engine_commit: compute::trace::engine_commit(),
        scenario_provenance: None,
        parent_trace_ids: Vec::new(),
        baseline_model_params: None,
        policy_result: None,
        portfolio_history: None,
    }
}

fn sample_execution_trace() -> agent::AgentExecutionTrace {
    agent::AgentExecutionTrace {
        id: "exec-trace-id".to_string(),
        created_at: "2026-01-01T00:00:00Z".to_string(),
        user_message: "what's my portfolio risk?".to_string(),
        planning_response_raw: r#"[{"tool":"current_risk","params":{},"reason":"test"}]"#.to_string(),
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
        grounding_status: agent::GroundingStatus { passed: true, warnings: vec![], retry_count: 0, ..Default::default() },
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
        self.experiment_result
            .clone()
            .ok_or_else(|| BackendError::Internal("no mock experiment result configured".to_string()))
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
            ..Default::default()
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
        upstox_config: unconfigured_upstox_config(),
        upstox_client: Arc::new(MockUpstoxClient::unused()),
        upstox_state_map: Arc::new(Mutex::new(HashMap::new())),
        semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
        capacity: 8,
        ask_queue_timeout: std::time::Duration::from_millis(200),
        gemini_configured: true,
        isin_client: reqwest::Client::new(),
        isin_config: compute::isin::ResolverConfig { search_url: "http://127.0.0.1:1/unreachable".to_string(), ..fast_isin_config() },
    });
    (app, store)
}

/// A `Result<T, String>`-backed `UpstoxClient`: the `Err` variant carries
/// the raw message `UpstoxError::TokenExchange`/`HoldingsFetch` would wrap
/// (matching their "surface verbatim" contract), so a test configures
/// exactly what the real API would have returned.
struct MockUpstoxClient {
    token_result: Result<String, String>,
    holdings_result: Result<Vec<UpstoxHolding>, String>,
}

impl MockUpstoxClient {
    /// For an `AppState` built by a test that never exercises the Upstox
    /// routes at all -- any call into this is itself a test bug.
    fn unused() -> Self {
        MockUpstoxClient {
            token_result: Err("MockUpstoxClient::unused() was called".to_string()),
            holdings_result: Err("MockUpstoxClient::unused() was called".to_string()),
        }
    }
}

#[async_trait::async_trait]
impl UpstoxClient for MockUpstoxClient {
    async fn exchange_code_for_token(
        &self,
        _code: &str,
        _client_id: &str,
        _client_secret: &str,
        _redirect_uri: &str,
    ) -> Result<String, UpstoxError> {
        self.token_result.clone().map_err(UpstoxError::TokenExchange)
    }

    async fn fetch_holdings(&self, _access_token: &str) -> Result<Vec<UpstoxHolding>, UpstoxError> {
        self.holdings_result.clone().map_err(UpstoxError::HoldingsFetch)
    }
}

fn unconfigured_upstox_config() -> UpstoxConfig {
    UpstoxConfig { api_key: None, api_secret: None, redirect_uri: "https://example.com/auth/upstox/callback".to_string() }
}

fn configured_upstox_config() -> UpstoxConfig {
    UpstoxConfig {
        api_key: Some("test-client-id".to_string()),
        api_secret: Some("test-client-secret".to_string()),
        redirect_uri: "https://example.com/auth/upstox/callback".to_string(),
    }
}

/// Builds an app wired for the Upstox routes specifically: `configured`
/// controls whether `UpstoxConfig` has credentials set, and the returned
/// state map lets a test insert/inspect OAuth states directly (e.g. to
/// simulate an expired one) rather than only through HTTP responses.
fn app_with_upstox(
    upstox_client: Arc<dyn UpstoxClient>,
    configured: bool,
) -> (axum::Router, Arc<Mutex<HashMap<String, Instant>>>) {
    let backend = MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    };
    let upstox_config = if configured { configured_upstox_config() } else { unconfigured_upstox_config() };
    let upstox_state_map = Arc::new(Mutex::new(HashMap::new()));
    let app = build_router(AppState {
        backend: Arc::new(backend),
        store: Arc::new(store::SnapshotStore::open(":memory:").unwrap()),
        upstox_config,
        upstox_client,
        upstox_state_map: upstox_state_map.clone(),
        semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
        capacity: 8,
        ask_queue_timeout: std::time::Duration::from_millis(200),
        gemini_configured: true,
        isin_client: reqwest::Client::new(),
        isin_config: compute::isin::ResolverConfig { search_url: "http://127.0.0.1:1/unreachable".to_string(), ..fast_isin_config() },
    });
    (app, upstox_state_map)
}

/// Whether `needle` appears in a rendered PDF's actual text content.
/// `printpdf`'s `PdfSaveOptions::default()` (used by `pdf::render_report`)
/// FlateDecode-compresses content streams, so the raw bytes don't contain
/// `ShowText` string literals verbatim -- `lopdf::Document::extract_text`
/// decompresses and decodes them properly.
fn pdf_contains_text(bytes: &[u8], needle: &str) -> bool {
    let doc = lopdf::Document::load_mem(bytes).expect("rendered report should be a valid PDF");
    let page_numbers: Vec<u32> = doc.get_pages().keys().copied().collect();
    let text = doc.extract_text(&page_numbers).expect("failed to extract text from rendered PDF");
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
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
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
        response.headers().contains_key("access-control-allow-origin"),
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
    assert_eq!(body["outputs"]["result"]["portfolio_vol_annualized"], 0.1552);
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
            ..Default::default()
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
    assert_eq!(body["assistant_turn"]["content"], "Vol is 99% (unverified).");
    assert_eq!(body["suggestion"], "What if I reduce my turnover to 20%?");
    assert!(body["result_id"].is_string(), "expected a result_id field, got {body:?}");
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
            ..Default::default()
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
        upstox_config: unconfigured_upstox_config(),
        upstox_client: Arc::new(MockUpstoxClient::unused()),
        upstox_state_map: Arc::new(Mutex::new(HashMap::new())),
        semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
        capacity: 8,
        ask_queue_timeout: std::time::Duration::from_millis(200),
        gemini_configured: true,
        isin_client: reqwest::Client::new(),
        isin_config: compute::isin::ResolverConfig { search_url: "http://127.0.0.1:1/unreachable".to_string(), ..fast_isin_config() },
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
            ..Default::default()
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
    let result_id = ask_json["result_id"].as_str().expect("result_id should be a string").to_string();

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
    let bytes = report_response.into_body().collect().await.unwrap().to_bytes();
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
            ..Default::default()
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
    let execution_trace_id = ask_body["agent_execution_trace"]["id"].as_str().unwrap().to_string();
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
        .oneshot(Request::builder().uri("/scenarios").body(Body::empty()).unwrap())
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
    assert_eq!(recent.len(), 1, "POST /experiment should have inserted exactly one snapshot");
    let snapshot = &recent[0];
    assert_eq!(snapshot.experiment_type, "RiskDecomposition");
    assert_eq!(snapshot.portfolio_vol_annualized, Some(0.1552));
    assert_eq!(snapshot.regime_label.as_deref(), Some("Bull"));
    assert!(snapshot.smoothed_probs.is_some());

    let fetched = store.get(&snapshot.id).unwrap();
    assert!(fetched.is_some(), "the just-inserted snapshot should be retrievable by id");
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
        .oneshot(Request::builder().uri(format!("/report/{id}")).body(Body::empty()).unwrap())
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
    assert!(!body["model_params"]["regime_state"].is_null(), "regime_state should always be present, got {body:?}");
    assert_eq!(body["model_params"]["regime_state"]["current_label"], "Bull");
}

/// Builds a `multipart/form-data` body with a single "file" field, the way
/// a browser's `<input type="file">` upload would.
fn multipart_body(filename: &str, content_type: &str, bytes: &[u8]) -> (String, Vec<u8>) {
    let boundary = "----driftriskcopilotboundary";
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n").as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn upload(app: axum::Router, filename: &str, content_type: &str, bytes: &[u8]) -> axum::response::Response {
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
    let response = upload(empty_backend_app(), "portfolio.csv", "text/csv", csv.as_bytes()).await;

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
    let response = upload(empty_backend_app(), "portfolio.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["layout_detected"], "value");
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    let sum: f64 = holdings.iter().map(|h| h["weight"].as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-6, "weights should sum to 1.0 within 1e-6, got {sum}");
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
    let response = upload(empty_backend_app(), "portfolio.txt", "text/plain", b"not a real portfolio file").await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "unsupported_file_type");
}

#[tokio::test]
async fn upload_weights_not_summing_to_one_returns_400() {
    let csv = "ticker,weight\nRELIANCE.NS,0.6\nTCS.NS,0.6\n";
    let response = upload(empty_backend_app(), "portfolio.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "invalid_portfolio");
}

#[tokio::test]
async fn upload_single_holding_returns_422() {
    // A single surviving holding is now caught by the dedicated
    // insufficient-holdings check (422) before it ever reaches
    // `validate_portfolio`'s own, more general 400 check -- see the "File
    // with only 1 holding after filtering" requirement this session adds.
    let csv = "ticker,weight\nRELIANCE.NS,1.0\n";
    let response = upload(empty_backend_app(), "portfolio.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = body_json(response).await;
    assert_eq!(body["code"], "insufficient_holdings");
}

// ---------------------------------------------------------------------
// Broker-format upload support (9 brokers)
// ---------------------------------------------------------------------

/// Common assertions every per-broker test makes: 200, correct tickers
/// (normalised to `.NS`), the value derived from quantity * price, and
/// weights summing to 1.0 within 1e-6.
fn assert_two_holding_portfolio(
    body: &serde_json::Value,
    ticker_a: &str,
    value_a: f64,
    ticker_b: &str,
    value_b: f64,
) {
    assert_eq!(body["layout_detected"], "value");
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    assert_eq!(holdings.len(), 2);
    let total = value_a + value_b;
    assert!((body["portfolio"]["total_value_inr"].as_f64().unwrap() - total).abs() < 1e-6);

    let find = |ticker: &str| holdings.iter().find(|h| h["ticker"] == ticker).cloned();
    let a = find(ticker_a).unwrap_or_else(|| panic!("expected a holding for {ticker_a}, got {holdings:?}"));
    let b = find(ticker_b).unwrap_or_else(|| panic!("expected a holding for {ticker_b}, got {holdings:?}"));
    assert!((a["weight"].as_f64().unwrap() - value_a / total).abs() < 1e-5);
    assert!((b["weight"].as_f64().unwrap() - value_b / total).abs() < 1e-5);

    let sum: f64 = holdings.iter().map(|h| h["weight"].as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-6, "weights should sum to 1.0 within 1e-6, got {sum}");
}

#[tokio::test]
async fn zerodha_format_parses_correctly() {
    // Fixture reproduces Zerodha's real export shape: 3 metadata rows,
    // then "Instrument"/"Qty."/"Avg. cost"/"LTP", then a trailing "Total"
    // row (see crates/server/tests/fixtures/zerodha_holdings.xlsx).
    let bytes = include_bytes!("../tests/fixtures/zerodha_holdings.xlsx");
    let response = upload(
        empty_backend_app(),
        "holdings.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        bytes,
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    // 3 metadata rows + header + 2 holdings + a "Total" row -- the "Total"
    // row must be skipped, not counted as a third holding.
    assert_eq!(body["row_count"], 2);
    assert_two_holding_portfolio(&body, "RELIANCE.NS", 10.0 * 2850.00, "HDFCBANK.NS", 25.0 * 1640.00);
}

#[tokio::test]
async fn upstox_format_parses_correctly() {
    // Upstox's own API field names (trading_symbol/quantity/average_price)
    // and its CSV export's NSE: prefix on tickers, both in one test.
    let csv = "trading_symbol,quantity,average_price\nNSE:RELIANCE,10,2850.00\nNSE:HDFCBANK,25,1640.00\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_two_holding_portfolio(&body, "RELIANCE.NS", 10.0 * 2850.00, "HDFCBANK.NS", 25.0 * 1640.00);
}

#[tokio::test]
async fn groww_format_parses_correctly() {
    let csv = "Symbol,Units,Average Buy Price\nRELIANCE,10,2850.00\nHDFCBANK,25,1640.00\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_two_holding_portfolio(&body, "RELIANCE.NS", 10.0 * 2850.00, "HDFCBANK.NS", 25.0 * 1640.00);
}

#[tokio::test]
async fn angel_one_format_parses_correctly() {
    let csv = "Symbol,Net Quantity,Avg. Buy Price\nRELIANCE,10,2850.00\nHDFCBANK,25,1640.00\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_two_holding_portfolio(&body, "RELIANCE.NS", 10.0 * 2850.00, "HDFCBANK.NS", 25.0 * 1640.00);
}

#[tokio::test]
async fn hdfc_securities_format_parses_correctly() {
    // "multiple title rows above header" -- metadata rows ahead of the
    // real header, same scan logic as Zerodha's XLSX, now exercised for a
    // CSV export too.
    let csv = "HDFC Securities - Portfolio Holdings Report\nAs on: 02-Oct-2026\nSymbol,Quantity,Avg Rate\nRELIANCE,10,2850.00\nHDFCBANK,25,1640.00\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_two_holding_portfolio(&body, "RELIANCE.NS", 10.0 * 2850.00, "HDFCBANK.NS", 25.0 * 1640.00);
}

#[tokio::test]
async fn icici_direct_format_parses_correctly() {
    // "report header rows above data" -- same metadata-row scan.
    let csv = "ICICIdirect.com Portfolio Report\nGenerated: 02-Oct-2026\nStock Name,Quantity,Average Rate\nRELIANCE,10,2850.00\nHDFCBANK,25,1640.00\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_two_holding_portfolio(&body, "RELIANCE.NS", 10.0 * 2850.00, "HDFCBANK.NS", 25.0 * 1640.00);
}

#[tokio::test]
async fn five_paisa_format_parses_correctly() {
    let csv = "Symbol,Qty,Avg Price\nRELIANCE,10,2850.00\nHDFCBANK,25,1640.00\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_two_holding_portfolio(&body, "RELIANCE.NS", 10.0 * 2850.00, "HDFCBANK.NS", 25.0 * 1640.00);
}

#[tokio::test]
async fn motilal_oswal_format_parses_correctly() {
    let csv = "Scrip Name,Qty,Buy Avg Price\nRELIANCE,10,2850.00\nHDFCBANK,25,1640.00\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_two_holding_portfolio(&body, "RELIANCE.NS", 10.0 * 2850.00, "HDFCBANK.NS", 25.0 * 1640.00);
}

#[tokio::test]
async fn kotak_securities_format_parses_correctly() {
    let csv = "Scrip,Quantity,Average Price\nRELIANCE,10,2850.00\nHDFCBANK,25,1640.00\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_two_holding_portfolio(&body, "RELIANCE.NS", 10.0 * 2850.00, "HDFCBANK.NS", 25.0 * 1640.00);
}

// --- Ticker normalisation / row filtering ---------------------------------

#[tokio::test]
async fn exchange_prefixed_ticker_is_normalised() {
    let csv = "ticker,weight\nNSE:RELIANCE,0.6\nBSE/TCS,0.4\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    assert!(holdings.iter().any(|h| h["ticker"] == "RELIANCE.NS"));
    assert!(holdings.iter().any(|h| h["ticker"] == "TCS.NS"));
    let notes = body["tickers_normalised"].as_array().unwrap();
    assert!(notes.iter().any(|n| n == "NSE:RELIANCE -> RELIANCE.NS"));
    assert!(notes.iter().any(|n| n == "BSE/TCS -> TCS.NS"));
}

#[tokio::test]
async fn eq_suffixed_ticker_is_normalised() {
    let csv = "ticker,weight\nRELIANCE-EQ,0.6\nTCS.NS,0.4\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    assert!(holdings.iter().any(|h| h["ticker"] == "RELIANCE.NS"));
    let notes = body["tickers_normalised"].as_array().unwrap();
    assert!(notes.iter().any(|n| n == "RELIANCE-EQ -> RELIANCE.NS"));
}

#[tokio::test]
async fn a_known_isin_is_converted_to_its_nse_symbol_not_kept_as_is() {
    let csv = "ticker,weight\nINE002A01018,0.6\nTCS.NS,0.4\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    assert!(holdings.iter().any(|h| h["ticker"] == "RELIANCE.NS"));
    assert!(!holdings.iter().any(|h| h["ticker"] == "INE002A01018"));
    assert_eq!(body["isins_resolved"]["INE002A01018"], "RELIANCE.NS");
}

#[tokio::test]
async fn a_total_row_is_skipped() {
    let csv = "ticker,weight\nRELIANCE.NS,0.6\nTCS.NS,0.4\nTotal,1.0\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["row_count"], 2);
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    assert!(!holdings.iter().any(|h| h["ticker"] == "Total" || h["ticker"] == "Total.NS"));
}

#[tokio::test]
async fn a_row_with_an_empty_ticker_is_skipped() {
    let csv = "ticker,weight\nRELIANCE.NS,0.7\n,0.1\nTCS.NS,0.3\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["row_count"], 2);
}

#[tokio::test]
async fn zero_weight_holdings_are_dropped_and_counted() {
    // A zero-quantity row survives row-filtering (it's not an empty/
    // "Total"/placeholder ticker) but produces a 0.0 weight -- it must not
    // appear in the returned holdings, and must be counted separately
    // from `row_count` (which reflects the real, non-zero holdings only).
    let csv = "ticker,shares,avg_price_inr\nRELIANCE.NS,10,2850.00\nHDFCBANK.NS,25,1640.00\nZEROCO.NS,0,500.00\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["skipped_zero_weight"], 1);
    assert_eq!(body["row_count"], 2);
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    assert_eq!(holdings.len(), 2);
    assert!(!holdings.iter().any(|h| h["ticker"] == "ZEROCO.NS"));
}

#[tokio::test]
async fn total_value_inr_is_rounded_to_2dp_in_the_response() {
    // 10 * 2850.333 + 25 * 1640.111 = 28503.33 + 41002.775 = 69506.105,
    // which must come back rounded to 69506.11 (or 69506.10 -- either is
    // a valid 2dp rounding of the exact .105 midpoint), never the raw
    // 3-decimal-place sum.
    let csv = "ticker,shares,avg_price_inr\nRELIANCE.NS,10,2850.333\nHDFCBANK.NS,25,1640.111\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    let total = body["portfolio"]["total_value_inr"].as_f64().unwrap();
    let rounded_to_2dp = (total * 100.0).round() / 100.0;
    assert_eq!(total, rounded_to_2dp, "total_value_inr {total} is not already rounded to 2dp");
    assert!((total - 69506.11).abs() < 0.01 || (total - 69506.10).abs() < 0.01, "got {total}");
}

#[tokio::test]
async fn fewer_than_two_holdings_after_filtering_returns_422() {
    // Only one real holding survives filtering (the other two rows are a
    // blank ticker and a "Total" row) -- /portfolio/upload must reject
    // this itself (422) rather than handing a single-holding portfolio to
    // `validate_portfolio` (which would 400 instead).
    let csv = "ticker,weight\nRELIANCE.NS,1.0\n,0.0\nTotal,1.0\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn broker_format_weights_sum_to_one_within_tolerance() {
    let csv = "Symbol,Net Quantity,Avg. Buy Price\nRELIANCE,7,111.11\nHDFCBANK,13,222.22\nINFY,3,333.33\nTCS,9,77.77\nITC,1,999.99\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    let sum: f64 = holdings.iter().map(|h| h["weight"].as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-6, "weights should sum to 1.0 within 1e-6, got {sum}");
}

#[tokio::test]
async fn unrecognised_columns_returns_the_improved_error_message() {
    let csv = "foo,bar\nx,y\n";
    let response = upload(empty_backend_app(), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "unsupported_format");
    let message = body["error"].as_str().unwrap();
    assert!(message.contains("Could not parse portfolio file"));
    assert!(message.contains("Columns found: [foo, bar]"));
    assert!(message.contains("Could not identify: ticker column"));
    assert!(message.contains("Zerodha"));
    assert!(message.contains("Upstox"));
    assert!(message.contains("Groww"));
    assert!(message.contains("Angel One"));
    assert!(message.contains("HDFC Securities"));
    assert!(message.contains("ICICI Direct"));
    assert!(message.contains("5Paisa"));
    assert!(message.contains("Motilal Oswal"));
    assert!(message.contains("Kotak Securities"));
    assert_eq!(body["columns_found"], serde_json::json!(["foo", "bar"]));
    assert_eq!(body["missing"], "ticker");
}

fn sample_risk_drift_trace(vol_before: f64, vol_after: f64, regime_before: &str, regime_after: &str) -> EvidenceTrace {
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

fn risk_drift_snapshot(portfolio_hash: &str, vol_before: f64, vol_after: f64) -> store::RiskSnapshot {
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
    store.insert(&risk_drift_snapshot("baseline-hash", 0.10, 0.12)).unwrap();

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
    assert!(body["error"].as_str().unwrap().contains("No prior snapshot found"));
}

#[tokio::test]
async fn drift_route_returns_200_and_empty_snapshots_when_none_exist() {
    let app = empty_backend_app();

    let response = app
        .oneshot(Request::builder().uri("/drift?portfolio=unknown-hash").body(Body::empty()).unwrap())
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
    store.insert(&risk_drift_snapshot("drift-hash", 0.10, 0.12)).unwrap();
    store.insert(&risk_drift_snapshot("drift-hash", 0.12, 0.20)).unwrap();
    // A non-RiskDrift snapshot for the same portfolio must be excluded.
    let mut other = risk_drift_snapshot("drift-hash", 0.0, 0.0);
    other.experiment_type = "RiskDecomposition".to_string();
    store.insert(&other).unwrap();

    let response = app
        .oneshot(Request::builder().uri("/drift?portfolio=drift-hash").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    let snapshots = body["snapshots"].as_array().unwrap();
    assert_eq!(snapshots.len(), 2, "expected exactly the 2 RiskDrift snapshots, got {snapshots:?}");
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
    let pnl = body["outputs"]["result"]["portfolio_pnl_inr"].as_f64().unwrap();
    assert!(pnl.is_finite() && pnl < 0.0, "expected a present, negative portfolio_pnl_inr, got {pnl}");
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
    assert!(body["error"].as_str().unwrap().contains("cannot be breached"));
    assert!(body["error"].as_str().unwrap().contains("Maximum feasible loss"));
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
    assert_eq!(body["outputs"]["result"]["policy_result"]["all_passed"], false);
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
    assert_eq!(body["outputs"]["result"]["policy_result"]["all_passed"], true);
    assert_eq!(body["outputs"]["result"]["policy_result"]["breach_count"], 0);
}

#[tokio::test]
async fn experiment_with_an_attached_policy_runs_the_passive_check_and_returns_policy_result() {
    let (app, _store) = app_with_backend_and_store(MockBackend {
        // The mock stands in for what a real backend would do: attach
        // policy_result to whatever experiment actually ran (here
        // RiskDecomposition, not PolicyCheck itself).
        experiment_result: Some({
            let mut trace = sample_trace();
            trace.policy_result = Some(serde_json::from_value(sample_policy_result(false)).unwrap());
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

// ---------------------------------------------------------------------
// Upstox OAuth
// ---------------------------------------------------------------------

fn sample_nse_holding(symbol: &str, quantity: f64, average_price: f64) -> UpstoxHolding {
    UpstoxHolding {
        trading_symbol: symbol.to_string(),
        exchange: "NSE_EQ".to_string(),
        quantity,
        average_price,
    }
}

#[tokio::test]
async fn upstox_status_returns_configured_true_when_env_is_set() {
    let (app, _) = app_with_upstox(Arc::new(MockUpstoxClient::unused()), true);

    let response =
        app.oneshot(Request::builder().uri("/auth/upstox/status").body(Body::empty()).unwrap()).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["configured"], true);
}

#[tokio::test]
async fn upstox_status_returns_configured_false_when_env_is_absent() {
    let (app, _) = app_with_upstox(Arc::new(MockUpstoxClient::unused()), false);

    let response =
        app.oneshot(Request::builder().uri("/auth/upstox/status").body(Body::empty()).unwrap()).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["configured"], false);
}

#[tokio::test]
async fn upstox_login_returns_302_with_location_containing_upstox_domain_and_state() {
    let (app, _) = app_with_upstox(Arc::new(MockUpstoxClient::unused()), true);

    let response =
        app.oneshot(Request::builder().uri("/auth/upstox/login").body(Body::empty()).unwrap()).await.unwrap();

    assert_eq!(response.status(), StatusCode::FOUND);
    let location = response.headers().get("location").unwrap().to_str().unwrap();
    // See the judgment-call note on `routes::get_upstox_login`: the spec's
    // own URL construction (api.upstox.com) and its test description
    // (accounts.upstox.com) disagree -- asserting the former, which is
    // what the handler actually redirects to.
    assert!(location.contains("api.upstox.com"), "location was {location:?}");
    assert!(location.contains("state="), "location was {location:?}");
}

#[tokio::test]
async fn upstox_login_returns_503_when_not_configured() {
    let (app, _) = app_with_upstox(Arc::new(MockUpstoxClient::unused()), false);

    let response =
        app.oneshot(Request::builder().uri("/auth/upstox/login").body(Body::empty()).unwrap()).await.unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = body_json(response).await;
    assert_eq!(body["code"], "not_configured");
}

#[tokio::test]
async fn upstox_callback_with_invalid_state_returns_400() {
    let (app, _) = app_with_upstox(Arc::new(MockUpstoxClient::unused()), true);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/auth/upstox/callback?code=some-code&state=not-a-real-state")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert!(body["error"].as_str().unwrap().contains("Invalid or expired OAuth state"));
}

#[tokio::test]
async fn upstox_callback_with_expired_state_returns_400() {
    let (app, state_map) = app_with_upstox(Arc::new(MockUpstoxClient::unused()), true);
    let expired_state = "expired-state-id".to_string();
    state_map.lock().unwrap().insert(expired_state.clone(), Instant::now() - Duration::from_secs(601));

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/auth/upstox/callback?code=some-code&state={expired_state}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn upstox_callback_with_valid_state_and_mocked_holdings_returns_200_with_portfolio() {
    let upstox_client = MockUpstoxClient {
        token_result: Ok("mock-access-token".to_string()),
        holdings_result: Ok(vec![
            sample_nse_holding("RELIANCE", 10.0, 2500.0),
            sample_nse_holding("TCS", 5.0, 3800.0),
        ]),
    };
    let (app, state_map) = app_with_upstox(Arc::new(upstox_client), true);
    let state = "valid-state-id".to_string();
    state_map.lock().unwrap().insert(state.clone(), Instant::now());

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/auth/upstox/callback?code=some-code&state={state}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["holdings_count"], 2);
    assert!(!body["data_as_of"].as_str().unwrap().is_empty());
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    assert_eq!(holdings.len(), 2);
    assert!(holdings.iter().any(|h| h["ticker"] == "RELIANCE.NS"));
    assert!(holdings.iter().any(|h| h["ticker"] == "TCS.NS"));
}

#[tokio::test]
async fn upstox_callback_with_fewer_than_two_nse_eq_holdings_returns_422() {
    let upstox_client = MockUpstoxClient {
        token_result: Ok("mock-access-token".to_string()),
        holdings_result: Ok(vec![sample_nse_holding("RELIANCE", 10.0, 2500.0)]),
    };
    let (app, state_map) = app_with_upstox(Arc::new(upstox_client), true);
    let state = "valid-state-id".to_string();
    state_map.lock().unwrap().insert(state.clone(), Instant::now());

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/auth/upstox/callback?code=some-code&state={state}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = body_json(response).await;
    assert_eq!(body["code"], "upstox_error");
}

#[tokio::test]
async fn upstox_callback_surfaces_token_exchange_failure_as_502() {
    let upstox_client = MockUpstoxClient {
        token_result: Err("invalid_grant: authorization code has expired".to_string()),
        holdings_result: Err("unused".to_string()),
    };
    let (app, state_map) = app_with_upstox(Arc::new(upstox_client), true);
    let state = "valid-state-id".to_string();
    state_map.lock().unwrap().insert(state.clone(), Instant::now());

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/auth/upstox/callback?code=some-code&state={state}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = body_json(response).await;
    assert_eq!(body["error"], "invalid_grant: authorization code has expired");
}

// ---------------------------------------------------------------------
// Production-hardening tests: concurrency limiter, error mapping/shape,
// health, ISIN upload rejection.
// ---------------------------------------------------------------------

fn empty_mock() -> MockBackend {
    MockBackend {
        experiment_result: None,
        experiment_error: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
        received_policy: Mutex::new(None),
    }
}

fn experiment_request(app: axum::Router) -> impl std::future::Future<Output = axum::response::Response> {
    let body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": {"type": "RiskDecomposition"},
    });
    async move {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
    }
}

fn is_json(response: &axum::response::Response) -> bool {
    response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("application/json"))
}

async fn assert_error_body(response: axum::response::Response, status: StatusCode, code: &str) -> serde_json::Value {
    assert_eq!(response.status(), status);
    assert!(is_json(&response), "error response must be JSON");
    let body = body_json(response).await;
    assert_eq!(body["code"], code, "{body}");
    assert!(body["error"].as_str().is_some_and(|m| !m.is_empty()), "{body}");
    body
}

/// Answers `/ask` only once `gate` has a permit, counting how many
/// requests are inside the backend at once.
struct GatedBackend {
    inner: MockBackend,
    gate: Arc<tokio::sync::Semaphore>,
    entered: Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait::async_trait]
impl Backend for GatedBackend {
    async fn run_experiment(
        &self,
        e: Experiment,
        p: Portfolio,
        policy: Option<compute::policy::RiskPolicy>,
    ) -> Result<EvidenceTrace, BackendError> {
        self.inner.run_experiment(e, p, policy).await
    }

    async fn run_ask(
        &self,
        p: Portfolio,
        m: String,
        h: Vec<agent::ConversationTurn>,
        policy: Option<compute::policy::RiskPolicy>,
    ) -> Result<agent::pipeline::PipelineResult, BackendError> {
        self.entered.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.gate.acquire().await.unwrap().forget();
        let mut result = self.inner.run_ask(p, m, h, policy).await?;
        // The store keys snapshots by trace id, so concurrent asks sharing
        // one canned result need distinct ids to all persist.
        result.trace.id = uuid::Uuid::new_v4().to_string();
        result.execution_trace.id = uuid::Uuid::new_v4().to_string();
        Ok(result)
    }
}

fn sample_pipeline_result() -> agent::pipeline::PipelineResult {
    agent::pipeline::PipelineResult {
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
            ..Default::default()
        },
        assistant_turn: agent::ConversationTurn::assistant("Vol is 15.5% annualised."),
        suggestion: "Reduce turnover?".to_string(),
        execution_trace: sample_execution_trace(),
    }
}

#[tokio::test]
async fn nine_concurrent_asks_eight_proceed_and_one_gets_503_service_busy() {
    use std::sync::atomic::Ordering;
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let entered = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let backend = GatedBackend {
        inner: MockBackend { ask_result: Some(sample_pipeline_result()), ..empty_mock() },
        gate: gate.clone(),
        entered: entered.clone(),
    };
    let app = build_router(AppState {
        backend: Arc::new(backend),
        store: Arc::new(store::SnapshotStore::open(":memory:").unwrap()),
        upstox_config: unconfigured_upstox_config(),
        upstox_client: Arc::new(MockUpstoxClient::unused()),
        upstox_state_map: Arc::new(Mutex::new(HashMap::new())),
        semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
        capacity: 8,
        ask_queue_timeout: std::time::Duration::from_millis(300),
        gemini_configured: true,
        isin_client: reqwest::Client::new(),
        isin_config: compute::isin::ResolverConfig { search_url: "http://127.0.0.1:1/unreachable".to_string(), ..fast_isin_config() },
    });
    let ask = |app: axum::Router| {
        let body = serde_json::json!({"portfolio": sample_portfolio(), "message": "risk?"});
        async move {
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ask")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
        }
    };

    let mut handles: Vec<_> = (0..8).map(|_| tokio::spawn(ask(app.clone()))).collect();
    while entered.load(Ordering::SeqCst) < 8 {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    // All 8 permits are held, so the 9th waits out the queue timeout.
    let health = app
        .clone()
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(body_json(health).await["active_requests"], 8);
    let ninth = ask(app.clone()).await;
    let retry_after = ninth.headers().get("retry-after").cloned();
    let body = assert_error_body(ninth, StatusCode::SERVICE_UNAVAILABLE, "service_busy").await;
    assert_eq!(body["retry_after_seconds"], 30);
    assert_eq!(body["error"], crate::error::MSG_SERVICE_BUSY);
    assert_eq!(retry_after.unwrap(), "30");
    assert_eq!(entered.load(Ordering::SeqCst), 8, "the 9th must never reach the backend");

    gate.add_permits(8);
    for h in handles.drain(..) {
        assert_eq!(h.await.unwrap().status(), StatusCode::OK);
    }
    // Permits are released: a new ask goes through, and health is idle again.
    gate.add_permits(1);
    assert_eq!(ask(app.clone()).await.status(), StatusCode::OK);
    let health = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(body_json(health).await["active_requests"], 0);
}

#[tokio::test]
async fn health_reports_capacity_activity_and_integration_status() {
    let response = app_with_backend(empty_mock())
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body = body_json(response).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["gemini"], "configured");
    assert_eq!(body["upstox"], "not_configured");
    assert_eq!(body["active_requests"], 0);
    assert_eq!(body["capacity"], 8);
}

#[tokio::test]
async fn data_unavailable_maps_to_503_with_the_user_message() {
    let app = app_with_backend(MockBackend {
        experiment_error: Some(BackendError::DataUnavailable(
            "Market data temporarily unavailable. Please try again in a moment.".to_string(),
        )),
        ..empty_mock()
    });
    let body = assert_error_body(experiment_request(app).await, StatusCode::SERVICE_UNAVAILABLE, "data_unavailable").await;
    assert_eq!(body["error"], "Market data is temporarily unavailable. Please try again in a moment.");
}

#[tokio::test]
async fn gemini_unavailable_maps_to_503_ai_unavailable_without_leaking_detail() {
    let app = app_with_backend(MockBackend {
        experiment_error: Some(BackendError::GeminiUnavailable("status 503 body {secret}".to_string())),
        ..empty_mock()
    });
    let body = assert_error_body(experiment_request(app).await, StatusCode::SERVICE_UNAVAILABLE, "ai_unavailable").await;
    assert_eq!(body["error"], "The AI service is temporarily busy. Please try again in a moment.");
}

#[tokio::test]
async fn internal_and_compute_errors_return_a_generic_500_message() {
    for err in [
        BackendError::Internal("thread 'x' panicked at src/foo.rs:1".to_string()),
        BackendError::Compute("nalgebra: singular matrix".to_string()),
    ] {
        let app = app_with_backend(MockBackend { experiment_error: Some(err), ..empty_mock() });
        let response = experiment_request(app).await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = body_json(response).await;
        assert_eq!(body["error"], "Something went wrong on our end. Please try again.");
    }
}

#[tokio::test]
async fn an_isin_holding_surfaces_as_422_unresolved_ticker_with_the_isin_message() {
    let err: BackendError = compute::ticker_map::isin_error("INE672A01026").into();
    let app = app_with_backend(MockBackend { experiment_error: Some(err), ..empty_mock() });
    let body = assert_error_body(experiment_request(app).await, StatusCode::UNPROCESSABLE_ENTITY, "unresolved_ticker").await;
    let msg = body["error"].as_str().unwrap();
    assert!(msg.contains("ISIN codes (INE672A01026)") && msg.contains("RELIANCE.NS"), "{msg}");
}

#[tokio::test]
async fn insufficient_data_maps_to_422_insufficient_data() {
    let err: BackendError = compute::ComputeError::InsufficientData("FOO.NS has only 20 observations".to_string()).into();
    let app = app_with_backend(MockBackend { experiment_error: Some(err), ..empty_mock() });
    assert_error_body(experiment_request(app).await, StatusCode::UNPROCESSABLE_ENTITY, "insufficient_data").await;
}

/// A stub Yahoo search endpoint: `answer(isin)` gives `(status, quotes)`,
/// each quote `(symbol, exchange, quoteType)`.
fn yahoo_stub(
    answer: impl Fn(&str) -> (u16, Vec<(&'static str, &'static str, &'static str)>) + Send + 'static,
) -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1/finance/search", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut conn) = conn else { return };
            let mut buf = [0u8; 4096];
            let n = conn.read(&mut buf).unwrap_or(0);
            let head = String::from_utf8_lossy(&buf[..n]).to_string();
            let isin = head.split("q=").nth(1).and_then(|s| s.split(['&', ' ']).next()).unwrap_or("").to_string();
            let (status, quotes) = answer(&isin);
            let items: Vec<String> = quotes
                .iter()
                .map(|(s, e, t)| format!(r#"{{"symbol":"{s}","exchange":"{e}","quoteType":"{t}"}}"#))
                .collect();
            let body = format!(r#"{{"quotes":[{}]}}"#, items.join(","));
            let _ = write!(
                conn,
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    url
}

fn fast_isin_config() -> compute::isin::ResolverConfig {
    use std::time::Duration;
    compute::isin::ResolverConfig {
        first_timeout: Duration::from_millis(500),
        retry_timeout: Duration::from_millis(800),
        rate_limit_wait: Duration::from_millis(50),
        batch_gap: Duration::from_millis(10),
        ..Default::default()
    }
}

fn app_with_isin_url(url: String) -> axum::Router {
    build_router(AppState {
        backend: Arc::new(empty_mock()),
        store: Arc::new(store::SnapshotStore::open(":memory:").unwrap()),
        upstox_config: unconfigured_upstox_config(),
        upstox_client: Arc::new(MockUpstoxClient::unused()),
        upstox_state_map: Arc::new(Mutex::new(HashMap::new())),
        semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
        capacity: 8,
        ask_queue_timeout: std::time::Duration::from_millis(200),
        gemini_configured: true,
        isin_client: reqwest::Client::new(),
        isin_config: compute::isin::ResolverConfig { search_url: url, ..fast_isin_config() },
    })
}

const NO_QUOTES: (u16, Vec<(&str, &str, &str)>) = (200, Vec::new());

#[tokio::test]
async fn kotak_style_isin_upload_resolves_skips_and_renormalizes() {
    let url = yahoo_stub(|isin| match isin {
        "INE263A01024" => (200, vec![("BEL.NS", "NSI", "EQUITY")]),
        "INF000000012" => (200, vec![("0P0001.BO", "BSE", "MUTUALFUND")]),
        _ => NO_QUOTES, // INE999X99999: unknown
    });
    let csv = "Stock Name,ISIN,Quantity,Average buy price\n\
               RELIANCE INDUSTRIES,INE002A01018,10,1000\n\
               HDFC BANK,INE040A01034,10,1000\n\
               BHARAT ELECTRONICS LTD,INE263A01024,10,1000\n\
               SOME FUND,INF000000012,10,1000\n\
               GONE CO,INE999X99999,10,1000\n";
    let response = upload(app_with_isin_url(url), "holdings.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["layout_detected"], "value");
    assert_eq!(
        body["isins_resolved"],
        serde_json::json!({"INE002A01018": "RELIANCE.NS", "INE040A01034": "HDFCBANK.NS", "INE263A01024": "BEL.NS"})
    );
    let skipped = body["isins_skipped"].as_array().unwrap();
    assert_eq!(skipped.len(), 2);
    assert_eq!(skipped[0]["isin"], "INF000000012");
    assert_eq!(skipped[0]["category"], "non_equity");
    assert!(skipped[0]["reason"].as_str().unwrap().starts_with("Mutual fund units cannot be analysed"));
    assert_eq!(skipped[1]["isin"], "INE999X99999");
    assert_eq!(skipped[1]["category"], "not_found");

    assert_eq!(body["weights_renormalized"], true);
    assert_eq!(body["original_holding_count"], 5);
    assert_eq!(body["retained_holding_count"], 3);
    assert_eq!(body["row_count"], 5);
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    let tickers: Vec<&str> = holdings.iter().map(|h| h["ticker"].as_str().unwrap()).collect();
    assert_eq!(tickers, vec!["RELIANCE.NS", "HDFCBANK.NS", "BEL.NS"]);
    // Equal values, so equal weights; the total is the retained value only.
    assert_eq!(body["portfolio"]["total_value_inr"], 30000.0);
    let weight_sum: f64 = holdings.iter().map(|h| h["weight"].as_f64().unwrap()).sum();
    assert!((weight_sum - 1.0).abs() < 1e-5, "{weight_sum}");
}

#[tokio::test]
async fn an_all_isin_file_resolves_without_skips() {
    let url = yahoo_stub(|_| NO_QUOTES); // both ISINs are cached: must not matter
    let csv = "ISIN,Quantity,Average buy price\nINE002A01018,10,1000\nINE040A01034,30,1000\n";
    let response = upload(app_with_isin_url(url), "k.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["isins_resolved"].as_object().unwrap().len(), 2);
    assert_eq!(body["isins_skipped"], serde_json::json!([]));
    assert_eq!(body["weights_renormalized"], false);
    assert_eq!(body["original_holding_count"], 2);
    assert_eq!(body["retained_holding_count"], 2);
    assert_eq!(body["portfolio"]["total_value_inr"], 40000.0);
    assert_eq!(body["portfolio"]["holdings"][1]["weight"], 0.75);
}

#[tokio::test]
async fn upload_mixing_isins_and_nse_symbols_resolves_both() {
    let url = yahoo_stub(|_| NO_QUOTES);
    let csv = "ticker,weight\nINE002A01018,0.5\nTCS,0.3\nINFY.NS,0.2\n";
    let response = upload(app_with_isin_url(url), "p.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    let tickers: Vec<&str> = body["portfolio"]["holdings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["ticker"].as_str().unwrap())
        .collect();
    assert_eq!(tickers, vec!["RELIANCE.NS", "TCS.NS", "INFY.NS"]);
    assert_eq!(body["isins_resolved"], serde_json::json!({"INE002A01018": "RELIANCE.NS"}));
    assert_eq!(body["isins_skipped"], serde_json::json!([]));
}

#[tokio::test]
async fn a_symbol_file_has_empty_isin_fields_and_never_calls_yahoo() {
    // The URL is unreachable: any lookup attempt would fail the upload.
    let csv = "ticker,weight\nRELIANCE.NS,0.6\nTCS.NS,0.4\n";
    let response = upload(empty_backend_app(), "p.csv", "text/csv", csv.as_bytes()).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["isins_resolved"], serde_json::json!({}));
    assert_eq!(body["isins_skipped"], serde_json::json!([]));
    assert_eq!(body["weights_renormalized"], false);
    assert_eq!(body["retained_holding_count"], 2);
}

#[tokio::test]
async fn only_one_resolving_isin_is_422_insufficient_holdings_with_reasons() {
    let url = yahoo_stub(|isin| match isin {
        "INE342T07379" => NO_QUOTES, // debenture by ISIN structure
        _ => NO_QUOTES,
    });
    let csv = "ticker,weight\nINE002A01018,0.5\nINE342T07379,0.3\nINE999X99999,0.2\n";
    let response = upload(app_with_isin_url(url), "p.csv", "text/csv", csv.as_bytes()).await;

    let body = assert_error_body(response, StatusCode::UNPROCESSABLE_ENTITY, "insufficient_holdings").await;
    assert!(body["error"].as_str().unwrap().contains("Bonds and debentures"), "{body}");
    assert_eq!(body["isins_skipped"].as_array().unwrap().len(), 2);
    assert_eq!(body["isins_skipped"][0]["category"], "non_equity");
}

#[tokio::test]
async fn an_upload_where_every_isin_fails_is_a_descriptive_422() {
    let url = yahoo_stub(|_| NO_QUOTES);
    let csv = "ticker,weight\nINE999X99991,0.5\nINE999X99992,0.5\n";
    let response = upload(app_with_isin_url(url), "p.csv", "text/csv", csv.as_bytes()).await;
    let body = assert_error_body(response, StatusCode::UNPROCESSABLE_ENTITY, "insufficient_holdings").await;
    assert!(body["error"].as_str().unwrap().contains("Could not identify this security"), "{body}");
}

#[tokio::test]
async fn yahoo_being_down_is_503_data_unavailable_not_a_blame_the_file_error() {
    let url = yahoo_stub(|_| (503, Vec::new()));
    let csv = "ticker,weight\nINE999X99991,0.5\nINE999X99992,0.5\n";
    let response = upload(app_with_isin_url(url), "p.csv", "text/csv", csv.as_bytes()).await;
    assert_error_body(response, StatusCode::SERVICE_UNAVAILABLE, "data_unavailable").await;
}

#[tokio::test]
async fn framework_level_errors_still_return_json_bodies() {
    let app = app_with_backend(empty_mock());
    // 405: wrong method on a real route (axum's own response has no body).
    let r = app
        .clone()
        .oneshot(Request::builder().method("GET").uri("/ask").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_error_body(r, StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed").await;
    // Malformed JSON.
    let r = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ask")
                .header("content-type", "application/json")
                .body(Body::from("{not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(is_json(&r));
    assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    // Missing content type.
    let r = app
        .clone()
        .oneshot(Request::builder().method("POST").uri("/ask").body(Body::from("{}")).unwrap())
        .await
        .unwrap();
    assert!(is_json(&r));
    // Query-string extractor rejection (a non-JSON plain-text axum body).
    let r = app
        .clone()
        .oneshot(Request::builder().uri("/drift").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert!(r.status().is_client_error());
    assert!(is_json(&r));
    // Unknown report id.
    let r = app
        .oneshot(Request::builder().uri("/report/nope").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_error_body(r, StatusCode::NOT_FOUND, "result_not_found").await;
}


// ---------------------------------------------------------------------
// is_redirect: out-of-scope questions answer 200, not 422
// ---------------------------------------------------------------------

/// A backend whose `/ask` is always declined by the planner.
struct DecliningBackend(String);

#[async_trait::async_trait]
impl Backend for DecliningBackend {
    async fn run_experiment(
        &self,
        _e: Experiment,
        _p: Portfolio,
        _policy: Option<compute::policy::RiskPolicy>,
    ) -> Result<EvidenceTrace, BackendError> {
        Err(BackendError::Internal("unused".to_string()))
    }

    async fn run_ask(
        &self,
        _p: Portfolio,
        _m: String,
        _h: Vec<agent::ConversationTurn>,
        _policy: Option<compute::policy::RiskPolicy>,
    ) -> Result<agent::pipeline::PipelineResult, BackendError> {
        Err(BackendError::Redirect(self.0.clone()))
    }
}

fn ask_request(app: axum::Router, message: &str) -> impl std::future::Future<Output = axum::response::Response> {
    let body = serde_json::json!({"portfolio": sample_portfolio(), "message": message});
    async move {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/ask")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
    }
}

#[tokio::test]
async fn a_declined_question_is_http_200_with_is_redirect_true_and_nulls_for_the_experiment_fields() {
    let redirect = "I cannot model individual stock price movements. Which would be most useful?";
    let app = build_router(AppState {
        backend: Arc::new(DecliningBackend(redirect.to_string())),
        store: Arc::new(store::SnapshotStore::open(":memory:").unwrap()),
        upstox_config: unconfigured_upstox_config(),
        upstox_client: Arc::new(MockUpstoxClient::unused()),
        upstox_state_map: Arc::new(Mutex::new(HashMap::new())),
        semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
        capacity: 8,
        ask_queue_timeout: std::time::Duration::from_millis(200),
        gemini_configured: true,
        isin_client: reqwest::Client::new(),
        isin_config: compute::isin::ResolverConfig::default(),
    });
    let response = ask_request(app, "What if Ratnaveer increases 10%?").await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["is_redirect"], true);
    assert_eq!(body["narration"], redirect);
    assert_eq!(body["assistant_turn"]["role"], "assistant");
    assert_eq!(body["assistant_turn"]["content"], redirect);
    assert!(body["suggestion"].as_str().is_some_and(|s| !s.is_empty()), "the chip is still offered");
    assert_eq!(body["grounding_warnings"], serde_json::json!([]));
    for null_field in ["experiment", "trace", "result_id", "agent_execution_trace", "visualization"] {
        assert!(body[null_field].is_null(), "{null_field} should be null: {body}");
    }
}

#[tokio::test]
async fn a_normal_ask_has_is_redirect_false() {
    let app = app_with_backend(MockBackend { ask_result: Some(sample_pipeline_result()), ..empty_mock() });
    let response = ask_request(app, "risk?").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["is_redirect"], false);
    assert!(body["trace"].is_object() && body["result_id"].is_string());
}

#[tokio::test]
async fn empty_or_whitespace_message_is_400_empty_message_before_the_backend_runs() {
    for msg in ["", "   ", "\n\t"] {
        let backend = MockBackend { ask_result: Some(sample_pipeline_result()), ..empty_mock() };
        let response = ask_request(app_with_backend(backend), msg).await;
        let body = assert_error_body(response, StatusCode::BAD_REQUEST, "empty_message").await;
        assert_eq!(body["error"], "Message cannot be empty.");
    }
}

#[tokio::test]
async fn the_backend_is_never_called_for_an_empty_message() {
    let backend = Arc::new(MockBackend { ask_result: Some(sample_pipeline_result()), ..empty_mock() });
    let app = build_router(AppState {
        backend: backend.clone(),
        store: Arc::new(store::SnapshotStore::open(":memory:").unwrap()),
        upstox_config: unconfigured_upstox_config(),
        upstox_client: Arc::new(MockUpstoxClient::unused()),
        upstox_state_map: Arc::new(Mutex::new(HashMap::new())),
        semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
        capacity: 8,
        ask_queue_timeout: std::time::Duration::from_millis(200),
        gemini_configured: true,
        isin_client: reqwest::Client::new(),
        isin_config: compute::isin::ResolverConfig::default(),
    });
    let _ = ask_request(app, "").await;
    assert!(backend.received_conversation_history.lock().unwrap().is_none(), "run_ask must not have been called");
}
