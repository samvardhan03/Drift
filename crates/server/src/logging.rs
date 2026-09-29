//! Request logging middleware: method, path, status, latency in ms, as a
//! single tracing event per request (JSON-formatted by the subscriber
//! configured in `main::init_tracing`).

use std::time::Instant;

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;

pub async fn log_requests(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let start = Instant::now();

    let response = next.run(request).await;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    let status = response.status().as_u16();
    tracing::info!(
        %method,
        %path,
        status,
        latency_ms,
        "request"
    );

    response
}
