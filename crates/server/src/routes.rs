use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::Json;
use compute::experiments::{Experiment, Portfolio};
use compute::policy::RiskPolicy;
use compute::trace::EvidenceTrace;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use store::{RiskSnapshot, SnapshotStore};

use crate::backend::Backend;
use crate::error::{ApiError, AppJson};
use crate::upstox::UpstoxClient;
use crate::validate::validate_portfolio;

const INDEX_HTML: &str = include_str!("../static/index.html");

/// Upstox OAuth config, read from env once at startup (see `main::main`).
/// `redirect_uri` always has a value (falls back to the production
/// default); `api_key`/`api_secret` are `None` in an unconfigured
/// environment (local dev with no credentials set).
#[derive(Clone)]
pub struct UpstoxConfig {
    pub api_key: Option<String>,
    pub api_secret: Option<String>,
    pub redirect_uri: String,
}

impl UpstoxConfig {
    pub fn is_configured(&self) -> bool {
        self.api_key.is_some() && self.api_secret.is_some()
    }
}

#[derive(Clone)]
pub struct AppState {
    pub backend: Arc<dyn Backend>,
    pub store: Arc<SnapshotStore>,
    pub upstox_config: UpstoxConfig,
    pub upstox_client: Arc<dyn UpstoxClient>,
    /// state UUID -> when it was issued; see `upstox::insert_state`/
    /// `upstox::validate_and_consume_state`.
    pub upstox_state_map: Arc<Mutex<HashMap<String, Instant>>>,
    /// Bounds concurrent `POST /ask` requests (each fans out to several
    /// Gemini calls plus a CPU-heavy compute step) -- see `post_ask`.
    pub semaphore: Arc<tokio::sync::Semaphore>,
    /// The semaphore's total permits, reported by `GET /health`.
    pub capacity: usize,
    /// How long a `/ask` waits for a permit before answering 503
    /// `service_busy`.
    pub ask_queue_timeout: Duration,
    pub gemini_configured: bool,
    /// Shared HTTP client for ISIN lookups on upload. It must send a
    /// browser-like User-Agent (see `main`).
    pub isin_client: reqwest::Client,
    pub isin_config: compute::isin::ResolverConfig,
}

/// Concurrent `/ask` requests allowed at once.
pub const MAX_CONCURRENT_ASKS: usize = 8;
/// How long a queued `/ask` waits for a free permit.
pub const ASK_QUEUE_TIMEOUT: Duration = Duration::from_secs(30);

/// Builds the `RiskSnapshot` `SnapshotStore::insert` persists for a
/// completed experiment: `trace_json` is the full trace (so `GET
/// /report/{id}` can render a PDF from it later), and the other fields are
/// pulled out of it for fast querying without re-parsing `trace_json`.
/// `regime_label`/`smoothed_probs` come from `model_params.regime_state`,
/// which every experiment type now always populates (see `compute`'s
/// always-on regime).
fn snapshot_from_trace(trace: &EvidenceTrace, portfolio: &Portfolio) -> Result<RiskSnapshot, ApiError> {
    let holdings: Vec<(String, f64)> =
        portfolio.holdings.iter().map(|h| (h.ticker.clone(), h.weight)).collect();
    let portfolio_hash = compute::portfolio::portfolio_hash(&holdings);

    let result = trace.outputs.get("result");
    let number_at = |path: &[&str]| -> Option<f64> {
        let mut v = result?;
        for key in path {
            v = v.get(key)?;
        }
        v.as_f64()
    };
    let portfolio_vol_annualized = number_at(&["portfolio_vol_annualized"]);
    let cvar_historical = number_at(&["stats_after", "historical_cvar"])
        .or_else(|| number_at(&["stats_before", "historical_cvar"]));

    let trace_json = serde_json::to_string(trace)
        .map_err(|e| ApiError::bad_request("internal_error", format!("failed to serialize trace: {e}")))?;

    Ok(RiskSnapshot {
        id: String::new(), // SnapshotStore::insert assigns the real id
        created_at: String::new(),
        portfolio_hash,
        experiment_type: trace.experiment.clone(),
        engine_version: trace.engine_version.clone(),
        regime_label: trace.model_params.regime_state.as_ref().map(|r| r.current_label.to_string()),
        smoothed_probs: trace.model_params.regime_state.as_ref().map(|r| r.smoothed_probs),
        portfolio_vol_annualized,
        cvar_historical,
        trace_json,
        // Populated by `post_ask` after this snapshot is built (this path
        // never calls Gemini, so `post_experiment` leaves them `None`).
        narration: None,
        suggestion: None,
        grounding_warnings: None,
    })
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    /// "configured" | "not_configured"
    pub gemini: &'static str,
    /// "configured" | "not_configured"
    pub upstox: &'static str,
    /// `/ask` requests currently holding a concurrency permit.
    pub active_requests: usize,
    pub capacity: usize,
}

fn configured(yes: bool) -> &'static str {
    if yes {
        "configured"
    } else {
        "not_configured"
    }
}

pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        gemini: configured(state.gemini_configured),
        upstox: configured(state.upstox_config.is_configured()),
        active_requests: state.capacity.saturating_sub(state.semaphore.available_permits()),
        capacity: state.capacity,
    })
}

#[derive(Deserialize)]
pub struct ExperimentRequest {
    pub portfolio: Portfolio,
    /// The tagged `Experiment` variant's own fields *except* `portfolio`
    /// (e.g. `{"type": "FactorShock", "shocks_pct": {...}}`); `portfolio`
    /// above is spliced in server-side before deserializing into
    /// `Experiment`, so callers never have to repeat it.
    pub experiment: serde_json::Value,
    /// Separate from `experiment` itself, so *any* experiment can be
    /// accompanied by a passive policy check (see
    /// `compute::dispatch::run_experiment`'s doc) without the caller
    /// needing to use `PolicyCheck` directly.
    #[serde(default)]
    pub policy: Option<RiskPolicy>,
}

pub async fn post_experiment(
    State(state): State<AppState>,
    AppJson(req): AppJson<ExperimentRequest>,
) -> Result<Json<EvidenceTrace>, ApiError> {
    validate_portfolio(&req.portfolio)?;

    let mut experiment_json = req.experiment;
    if let serde_json::Value::Object(ref mut map) = experiment_json {
        map.insert(
            "portfolio".to_string(),
            serde_json::to_value(&req.portfolio).expect("Portfolio always serializes"),
        );
    } else {
        return Err(ApiError::bad_request(
            "invalid_experiment",
            "experiment must be a JSON object with a \"type\" field",
        ));
    }
    let experiment: Experiment = serde_json::from_value(experiment_json).map_err(|e| {
        ApiError::bad_request("invalid_experiment", format!("invalid experiment: {e}"))
    })?;

    let trace = state.backend.run_experiment(experiment, req.portfolio.clone(), req.policy).await?;

    let snapshot = snapshot_from_trace(&trace, &req.portfolio)?;
    state.store.insert(&snapshot).map_err(|e| {
        ApiError::bad_request("internal_error", format!("failed to persist risk snapshot: {e}"))
    })?;

    Ok(Json(trace))
}

#[derive(Deserialize)]
pub struct AskRequest {
    pub portfolio: Portfolio,
    pub message: String,
    /// Prior turns of this conversation, oldest first; empty by default.
    /// See `agent::conversation::ConversationTurn`.
    #[serde(default)]
    pub conversation_history: Vec<agent::ConversationTurn>,
    /// See `ExperimentRequest::policy`'s doc -- same passive-check
    /// mechanism, attached here regardless of which experiment the NL
    /// message ends up parsing to.
    #[serde(default)]
    pub policy: Option<RiskPolicy>,
}

#[derive(Serialize)]
pub struct AskResponse {
    /// `None` only when `is_redirect` is true (no experiment ran).
    pub experiment: Option<Experiment>,
    /// `None` only when `is_redirect` is true.
    pub trace: Option<EvidenceTrace>,
    pub narration: String,
    pub grounding_warnings: Vec<String>,
    /// This turn's narration as an `assistant` turn, ready for the caller
    /// to append to `conversation_history` for the next request.
    pub assistant_turn: agent::ConversationTurn,
    /// One follow-up question a risk manager would naturally ask next.
    pub suggestion: String,
    /// Stored under this id in the persistent `SnapshotStore`;
    /// `GET /report/{result_id}` renders it as a PDF.
    /// `None` only when `is_redirect` is true (nothing was stored).
    pub result_id: Option<String>,
    /// A record of the orchestration itself (the planning call, each
    /// tool's execution, the combined narration/grounding outcome) --
    /// distinct from `trace`, which records the *experiment's* evidence.
    /// Also independently retrievable via
    /// `GET /execution-trace/{agent_execution_trace.id}`.
    /// `None` only when `is_redirect` is true.
    pub agent_execution_trace: Option<agent::AgentExecutionTrace>,
    /// Chart-ready data derived from `trace`, for the frontend to render
    /// without re-deriving it from the raw trace itself. `None` if
    /// `trace.experiment` isn't one of the seven recognised types (see
    /// `visualization::build_visualization`).
    pub visualization: Option<crate::visualization::VisualizationData>,
    /// True when the question was out of scope (an individual stock's
    /// price scenario, valuation, a price forecast, news, or not finance)
    /// and `narration` is a redirect explaining what the system can do
    /// instead. No experiment ran: `experiment`, `trace`, `result_id`,
    /// `agent_execution_trace` and `visualization` are all null. The HTTP
    /// status is 200 -- render `narration` as a normal (soft) assistant
    /// message, not an error.
    pub is_redirect: bool,
}

/// The follow-up chip shown under a redirect (there's no experiment to
/// derive one from).
const REDIRECT_SUGGESTION: &str = "Which macro factors drive my portfolio's risk most?";
const DEFAULT_REDIRECT: &str = "I can only help with portfolio risk analysis.";

impl AskResponse {
    fn redirect(message: String) -> Self {
        let message = if message.trim().is_empty() { DEFAULT_REDIRECT.to_string() } else { message };
        AskResponse {
            experiment: None,
            trace: None,
            assistant_turn: agent::ConversationTurn::assistant(message.clone()),
            narration: message,
            grounding_warnings: Vec::new(),
            suggestion: REDIRECT_SUGGESTION.to_string(),
            result_id: None,
            agent_execution_trace: None,
            visualization: None,
            is_redirect: true,
        }
    }
}

pub async fn post_ask(
    State(state): State<AppState>,
    AppJson(req): AppJson<AskRequest>,
) -> Result<Json<AskResponse>, ApiError> {
    // Before any Gemini call: an empty message would only come back from
    // Gemini as a confusing "AI unavailable".
    if req.message.trim().is_empty() {
        return Err(ApiError::bad_request("empty_message", "Message cannot be empty."));
    }
    validate_portfolio(&req.portfolio)?;

    // Held until the handler returns. Cheap validation above runs first so
    // a malformed request never occupies a permit.
    let _permit = match tokio::time::timeout(state.ask_queue_timeout, state.semaphore.acquire()).await {
        Ok(Ok(permit)) => permit,
        // Timed out waiting, or the semaphore was closed (never, today).
        Ok(Err(_)) | Err(_) => return Err(ApiError::service_busy()),
    };

    let portfolio = req.portfolio.clone();
    let result = match state
        .backend
        .run_ask(req.portfolio, req.message, req.conversation_history, req.policy)
        .await
    {
        Ok(result) => result,
        Err(crate::backend::BackendError::Redirect(message)) => return Ok(Json(AskResponse::redirect(message))),
        Err(other) => return Err(other.into()),
    };

    let mut snapshot = snapshot_from_trace(&result.trace, &portfolio)?;
    snapshot.narration = Some(result.narration.narration.clone());
    snapshot.suggestion = Some(result.suggestion.clone());
    snapshot.grounding_warnings = if result.narration.grounding_warnings.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&result.narration.grounding_warnings).map_err(|e| {
            ApiError::bad_request("internal_error", format!("failed to serialize grounding warnings: {e}"))
        })?)
    };
    let result_id = state.store.insert(&snapshot).map_err(|e| {
        ApiError::bad_request("internal_error", format!("failed to persist risk snapshot: {e}"))
    })?;

    let execution_trace_json = serde_json::to_string(&result.execution_trace).map_err(|e| {
        ApiError::bad_request("internal_error", format!("failed to serialize execution trace: {e}"))
    })?;
    state
        .store
        .insert_execution_trace(&result.execution_trace.id, &execution_trace_json)
        .map_err(|e| {
            ApiError::bad_request("internal_error", format!("failed to persist execution trace: {e}"))
        })?;

    let visualization = crate::visualization::build_visualization(&result.trace);

    Ok(Json(AskResponse {
        experiment: Some(result.experiment),
        trace: Some(result.trace),
        narration: result.narration.narration,
        grounding_warnings: result.narration.grounding_warnings,
        assistant_turn: result.assistant_turn,
        suggestion: result.suggestion,
        result_id: Some(result_id),
        agent_execution_trace: Some(result.execution_trace),
        visualization,
        is_redirect: false,
    }))
}

/// `GET /execution-trace/{id}`: the full `AgentExecutionTrace` for a prior
/// `/ask` call. 404 if `id` is unknown.
pub async fn get_execution_trace(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<agent::AgentExecutionTrace>, ApiError> {
    let json = state
        .store
        .get_execution_trace(&id)
        .map_err(|e| ApiError::bad_request("internal_error", format!("failed to read snapshot store: {e}")))?
        .ok_or_else(|| ApiError::not_found("execution_trace_not_found", format!("no stored execution trace for id {id}")))?;
    let trace: agent::AgentExecutionTrace = serde_json::from_str(&json).map_err(|e| {
        ApiError::bad_request("internal_error", format!("stored execution_trace_json is invalid: {e}"))
    })?;
    Ok(Json(trace))
}

/// Serves the embedded single-page UI for every unmatched path, so the
/// binary needs no separate static-file directory at runtime.
pub async fn static_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

/// `GET /scenarios`: the fixed set of historical scenario presets. No
/// authentication, no portfolio needed.
pub async fn get_scenarios() -> Json<serde_json::Value> {
    Json(serde_json::to_value(compute::scenarios::all_scenarios()).expect("scenarios always serialize"))
}

/// `GET /report/{result_id}`: a one-page PDF report for a previously
/// stored `/experiment` or `/ask` result. 404 if `result_id` is unknown
/// (never stored, or the id is simply wrong).
pub async fn get_report(State(state): State<AppState>, Path(result_id): Path<String>) -> Response {
    let snapshot = match state.store.get(&result_id) {
        Ok(Some(snapshot)) => snapshot,
        Ok(None) => {
            return ApiError::not_found("result_not_found", format!("no stored result for id {result_id}"))
                .into_response();
        }
        Err(e) => {
            return ApiError::bad_request("internal_error", format!("failed to read snapshot store: {e}"))
                .into_response();
        }
    };
    let trace: EvidenceTrace = match serde_json::from_str(&snapshot.trace_json) {
        Ok(trace) => trace,
        Err(e) => {
            return ApiError::bad_request("internal_error", format!("stored trace_json is invalid: {e}"))
                .into_response();
        }
    };
    let grounding_warnings: Vec<String> = match &snapshot.grounding_warnings {
        Some(json) => match serde_json::from_str(json) {
            Ok(warnings) => warnings,
            Err(e) => {
                return ApiError::bad_request("internal_error", format!("stored grounding_warnings is invalid: {e}"))
                    .into_response();
            }
        },
        None => Vec::new(),
    };
    let bytes = crate::pdf::render_report(&trace, snapshot.narration.as_deref(), &grounding_warnings);
    (StatusCode::OK, [(header::CONTENT_TYPE, "application/pdf")], bytes).into_response()
}

#[derive(Deserialize)]
pub struct DriftQuery {
    /// A `compute::portfolio::portfolio_hash` value.
    pub portfolio: String,
    #[serde(default = "default_drift_limit")]
    pub limit: usize,
}

fn default_drift_limit() -> usize {
    10
}

#[derive(Serialize)]
pub struct DriftSummary {
    pub id: String,
    pub created_at: String,
    pub vol_before: f64,
    pub vol_after: f64,
    pub vol_change_pct: f64,
    pub regime_before: String,
    pub regime_after: String,
    pub days_elapsed: i64,
}

#[derive(Serialize)]
pub struct DriftListResponse {
    pub snapshots: Vec<DriftSummary>,
}

/// The maximum number of a portfolio's recent snapshots (of any experiment
/// type) `get_drift` scans looking for `RiskDrift` ones. `SnapshotStore`
/// has no "list recent of this experiment type" query (out of this
/// session's scope -- see the README), so this filters client-side over a
/// generously-sized recent window instead; a portfolio with more than this
/// many *non*-`RiskDrift` snapshots since its oldest relevant `RiskDrift`
/// one could miss older entries. Fine for a hackathon-scale demo.
const DRIFT_SCAN_WINDOW: usize = 1000;

/// `GET /drift?portfolio=<hash>&limit=<n>`: the `n` most recent `RiskDrift`
/// snapshots for a portfolio, newest first, summary fields only (not the
/// full trace -- see `DriftSummary`). `{"snapshots": []}` (200) if none
/// exist for that portfolio hash.
pub async fn get_drift(
    State(state): State<AppState>,
    Query(query): Query<DriftQuery>,
) -> Result<Json<DriftListResponse>, ApiError> {
    let candidates = state.store.latest_for_portfolio(&query.portfolio, DRIFT_SCAN_WINDOW).map_err(|e| {
        ApiError::bad_request("internal_error", format!("failed to read snapshot store: {e}"))
    })?;

    let mut snapshots = Vec::new();
    for snapshot in candidates {
        if snapshot.experiment_type != "RiskDrift" {
            continue;
        }
        let trace: EvidenceTrace = serde_json::from_str(&snapshot.trace_json).map_err(|e| {
            ApiError::bad_request("internal_error", format!("stored trace_json is invalid: {e}"))
        })?;
        let result = trace.outputs.get("result");
        let f = |key: &str| result.and_then(|r| r.get(key)).and_then(Value::as_f64).unwrap_or(0.0);
        let s = |key: &str| {
            result.and_then(|r| r.get(key)).and_then(Value::as_str).map(str::to_string).unwrap_or_default()
        };
        let i = |key: &str| result.and_then(|r| r.get(key)).and_then(Value::as_i64).unwrap_or(0);
        snapshots.push(DriftSummary {
            id: snapshot.id.clone(),
            created_at: snapshot.created_at.clone(),
            vol_before: f("vol_before"),
            vol_after: f("vol_after"),
            vol_change_pct: f("vol_change_pct"),
            regime_before: s("regime_before"),
            regime_after: s("regime_after"),
            days_elapsed: i("days_elapsed"),
        });
        if snapshots.len() >= query.limit {
            break;
        }
    }

    Ok(Json(DriftListResponse { snapshots }))
}

// ---------------------------------------------------------------------
// Upstox OAuth
// ---------------------------------------------------------------------

const UPSTOX_AUTHORIZE_URL: &str = "https://api.upstox.com/v2/login/authorization/dialog";

/// `GET /auth/upstox/login`: redirects to Upstox's OAuth 2.0 authorization
/// dialog. 503 if Upstox credentials aren't configured in this environment
/// (see `UpstoxConfig::is_configured`) -- local dev without credentials
/// fails cleanly here instead of panicking or producing a broken redirect.
///
/// **Judgment call**: the spec's own URL host (`api.upstox.com`, given
/// verbatim in its URL-construction section) and its test description
/// (asserting the redirect's `Location` contains `accounts.upstox.com`)
/// disagree. Followed the explicit URL, which also matches Upstox's real,
/// documented OAuth endpoint -- flagged in this session's report rather
/// than silently picking one.
pub async fn get_upstox_login(State(state): State<AppState>) -> Response {
    let Some(api_key) = state.upstox_config.api_key.as_deref() else {
        return ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "not_configured", "Upstox OAuth not configured")
            .into_response();
    };

    let oauth_state = crate::upstox::insert_state(&state.upstox_state_map);
    let url = format!(
        "{UPSTOX_AUTHORIZE_URL}?response_type=code&client_id={}&redirect_uri={}&state={}",
        urlencoding::encode(api_key),
        urlencoding::encode(&state.upstox_config.redirect_uri),
        urlencoding::encode(&oauth_state),
    );

    (StatusCode::FOUND, [(header::LOCATION, url)]).into_response()
}

#[derive(Deserialize)]
pub struct UpstoxCallbackQuery {
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub state: Option<String>,
}

#[derive(Serialize)]
pub struct UpstoxImportResponse {
    pub portfolio: Portfolio,
    pub holdings_count: usize,
    pub data_as_of: String,
}

/// `GET /auth/upstox/callback`: receives Upstox's redirect after the user
/// logs in (`?code=...&state=...`), exchanges the one-time `code` for an
/// access token, fetches holdings, and returns them as a `Portfolio`. The
/// access token is used exactly once for the holdings fetch and never
/// persisted -- this endpoint is stateless and read-only with respect to
/// the user's Upstox account.
pub async fn get_upstox_callback(
    State(state): State<AppState>,
    Query(query): Query<UpstoxCallbackQuery>,
) -> Result<Json<UpstoxImportResponse>, ApiError> {
    let code = query
        .code
        .ok_or_else(|| ApiError::bad_request("invalid_callback", "missing \"code\" query parameter"))?;
    let oauth_state = query
        .state
        .ok_or_else(|| ApiError::bad_request("invalid_callback", "missing \"state\" query parameter"))?;

    if !crate::upstox::validate_and_consume_state(&state.upstox_state_map, &oauth_state) {
        return Err(ApiError::bad_request(
            "invalid_state",
            "Invalid or expired OAuth state. Please try again.",
        ));
    }

    let (api_key, api_secret) = match (&state.upstox_config.api_key, &state.upstox_config.api_secret) {
        (Some(k), Some(s)) => (k.clone(), s.clone()),
        _ => {
            return Err(ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "not_configured",
                "Upstox OAuth not configured",
            ))
        }
    };

    let access_token = state
        .upstox_client
        .exchange_code_for_token(&code, &api_key, &api_secret, &state.upstox_config.redirect_uri)
        .await
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, "upstox_error", e.to_string()))?;

    let holdings = state
        .upstox_client
        .fetch_holdings(&access_token)
        .await
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, "upstox_error", e.to_string()))?;

    let portfolio = crate::upstox::holdings_to_portfolio(&holdings).map_err(|e| match e {
        crate::upstox::UpstoxError::InsufficientHoldings => {
            ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "upstox_error", e.to_string())
        }
        other => ApiError::new(StatusCode::BAD_GATEWAY, "upstox_error", other.to_string()),
    })?;

    let holdings_count = portfolio.holdings.len();
    Ok(Json(UpstoxImportResponse {
        portfolio,
        holdings_count,
        data_as_of: chrono::Utc::now().to_rfc3339(),
    }))
}

/// `GET /auth/upstox/status`: whether Upstox credentials are configured in
/// this environment, so the UI can decide whether to show the "Connect
/// with Upstox" button at all.
pub async fn get_upstox_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "configured": state.upstox_config.is_configured() }))
}
