mod backend;
mod error;
mod logging;
mod pdf;
mod routes;
mod upload;
mod validate;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};

use backend::{Backend, NoAiBackend, RealBackend};
use routes::AppState;
use store::SnapshotStore;

const DEFAULT_SNAPSHOT_DB_PATH: &str = "./data/snapshots.db";

#[tokio::main]
async fn main() {
    init_tracing();

    let db_path =
        std::env::var("SNAPSHOT_DB_PATH").unwrap_or_else(|_| DEFAULT_SNAPSHOT_DB_PATH.to_string());
    let store = match SnapshotStore::open(&db_path) {
        Ok(store) => Arc::new(store),
        Err(err) => {
            tracing::error!(error = %err, %db_path, "failed to open snapshot store");
            std::process::exit(1);
        }
    };

    let backend: Arc<dyn Backend> = match agent::gemini::HttpGeminiClient::new() {
        Ok(gemini) => {
            tracing::info!("GEMINI_API_KEY found — AI features enabled (/ask, narration)");
            Arc::new(RealBackend::new(gemini, store.clone()))
        }
        Err(agent::gemini::GeminiError::MissingApiKey) => {
            tracing::warn!(
                "GEMINI_API_KEY not set — starting in compute-only mode. \
                 /experiment returns EvidenceTrace without AI narration; \
                 /ask returns 503. Set GEMINI_API_KEY and restart to enable AI features."
            );
            Arc::new(NoAiBackend {
                store: store.clone(),
            })
        }
        Err(err) => {
            tracing::error!(error = %err, "failed to initialise Gemini client: {err}");
            std::process::exit(1);
        }
    };

    let state = AppState { backend, store };

    let app = build_router(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let addr = format!("0.0.0.0:{port}");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|err| panic!("failed to bind {addr}: {err}"));
    tracing::info!(%addr, "listening");

    axum::serve(listener, app).await.expect("server error");
}

fn build_router(state: AppState) -> Router {
    // Permissive by design: this API has no cookie/session-based auth (no
    // credentials to leak cross-origin), and the frontend is served from a
    // different origin than the Cloud Run API in at least one deployment
    // shape (see the README) -- so every origin/method/header is allowed
    // rather than hard-coding a single expected frontend origin.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(routes::health))
        .route("/scenarios", get(routes::get_scenarios))
        .route("/experiment", post(routes::post_experiment))
        .route("/ask", post(routes::post_ask))
        .route("/report/:result_id", get(routes::get_report))
        .route("/execution-trace/:id", get(routes::get_execution_trace))
        .route("/drift", get(routes::get_drift))
        .route("/portfolio/upload", post(upload::post_portfolio_upload))
        .fallback(routes::static_handler)
        .layer(axum::middleware::from_fn(logging::log_requests))
        .layer(cors)
        .with_state(state)
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;

    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

#[cfg(test)]
mod tests;
