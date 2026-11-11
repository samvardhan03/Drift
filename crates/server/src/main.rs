mod backend;
mod error;
mod logging;
mod pdf;
mod routes;
mod upload;
mod upstox;
mod validate;
mod visualization;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};

use backend::{Backend, NoAiBackend, RealBackend};
use routes::{AppState, UpstoxConfig};
use store::SnapshotStore;
use upstox::{HttpUpstoxClient, UpstoxClient};

const DEFAULT_SNAPSHOT_DB_PATH: &str = "./data/snapshots.db";

/// Local default; set UPSTOX_REDIRECT_URI to your deployment's public
/// callback URL (and register the same URL in the Upstox developer console).
const DEFAULT_UPSTOX_REDIRECT_URI: &str = "http://localhost:8080/auth/upstox/callback";

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

    let (backend, gemini_configured): (Arc<dyn Backend>, bool) = match agent::gemini::HttpGeminiClient::new() {
        Ok(gemini) => {
            tracing::info!("GEMINI_API_KEY found — AI features enabled (/ask, narration)");
            (Arc::new(RealBackend::new(gemini, store.clone())), true)
        }
        Err(agent::gemini::GeminiError::MissingApiKey) => {
            tracing::warn!(
                "GEMINI_API_KEY not set — starting in compute-only mode. \
                 /experiment returns EvidenceTrace without AI narration; \
                 /ask returns 503. Set GEMINI_API_KEY and restart to enable AI features."
            );
            (Arc::new(NoAiBackend { store: store.clone() }), false)
        }
        Err(err) => {
            tracing::error!(error = %err, "failed to initialise Gemini client: {err}");
            std::process::exit(1);
        }
    };

    let upstox_api_key = std::env::var("UPSTOX_API_KEY").ok().filter(|s| !s.is_empty());
    let upstox_api_secret = std::env::var("UPSTOX_API_SECRET").ok().filter(|s| !s.is_empty());
    let upstox_redirect_uri =
        std::env::var("UPSTOX_REDIRECT_URI").unwrap_or_else(|_| DEFAULT_UPSTOX_REDIRECT_URI.to_string());
    let upstox_config = UpstoxConfig {
        api_key: upstox_api_key,
        api_secret: upstox_api_secret,
        redirect_uri: upstox_redirect_uri,
    };
    if upstox_config.is_configured() {
        tracing::info!("Upstox OAuth: configured");
    } else {
        tracing::info!("Upstox OAuth: not configured (UPSTOX_API_KEY missing)");
    }
    let upstox_client: Arc<dyn UpstoxClient> = Arc::new(HttpUpstoxClient::new());
    let upstox_state_map = Arc::new(Mutex::new(HashMap::new()));

    // Yahoo Finance throttles the default reqwest User-Agent, so ISIN lookups
    // present a browser-like one.
    let isin_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .user_agent("Mozilla/5.0")
        .build()
        .expect("failed to build ISIN HTTP client");

    let state = AppState {
        backend,
        store,
        upstox_config,
        upstox_client,
        upstox_state_map,
        semaphore: Arc::new(tokio::sync::Semaphore::new(routes::MAX_CONCURRENT_ASKS)),
        capacity: routes::MAX_CONCURRENT_ASKS,
        ask_queue_timeout: routes::ASK_QUEUE_TIMEOUT,
        // false when GEMINI_API_KEY is absent (compute-only mode).
        gemini_configured,
        isin_client,
        isin_config: compute::isin::ResolverConfig::default(),
    };

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

    tokio::spawn(warm_market_data_cache());

    axum::serve(listener, app)
        .await
        .expect("server error");
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
        .route("/auth/upstox/login", get(routes::get_upstox_login))
        .route("/auth/upstox/callback", get(routes::get_upstox_callback))
        .route("/auth/upstox/status", get(routes::get_upstox_status))
        .fallback(routes::static_handler)
        .layer(axum::middleware::from_fn(error::ensure_json_errors))
        .layer(axum::middleware::from_fn(logging::log_requests))
        .layer(cors)
        .with_state(state)
}

/// Tickers most portfolios and every experiment need: common large caps
/// plus the five factor series.
const WARMUP_TICKERS: [&str; 15] = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "LT.NS", "ITC.NS",
    "KOTAKBANK.NS", "BHARTIARTL.NS", "TMPV.NS", "^NSEI", "INR=X", "BZ=F", "GC=F", "^NSEBANK",
];

/// Pre-fetches `WARMUP_TICKERS` into the on-disk price cache shortly after
/// startup so the first real request doesn't pay the Yahoo round trips. A
/// failure for any ticker is logged and skipped -- never fatal.
async fn warm_market_data_cache() {
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    let outcome = tokio::task::spawn_blocking(|| {
        let dir = std::path::Path::new(compute::dispatch::CACHE_DIR);
        let mut failed = Vec::new();
        for ticker in WARMUP_TICKERS {
            if let Err(err) = compute::data::load_series(dir, ticker, false) {
                failed.push(format!("{ticker}: {err}"));
            }
        }
        failed
    })
    .await;
    match outcome {
        Ok(failed) if failed.is_empty() => tracing::info!("Market data cache warmed"),
        Ok(failed) => tracing::warn!(?failed, "Market data cache warmup incomplete"),
        Err(err) => tracing::warn!(error = %err, "Market data cache warmup task failed"),
    }
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;

    tracing_subscriber::fmt()
        .json()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();
}

#[cfg(test)]
mod tests;
