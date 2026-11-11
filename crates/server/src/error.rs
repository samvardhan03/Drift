//! JSON error responses. Every error path returns `{ "error": string,
//! "code": string }` with the appropriate HTTP status, per spec.

use axum::extract::{FromRequest, Request};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::de::DeserializeOwned;

use crate::backend::BackendError;

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
    /// Extra top-level fields merged into the `{error, code}` response
    /// object (e.g. `portfolio/upload`'s `columns_found`/`missing`) --
    /// `None` for every error that only ever needed the base two fields.
    pub extra: Option<serde_json::Value>,
}

/// User-facing messages for the 503/500 codes. The detail behind a 500 or
/// 503 (raw Rust error text, upstream bodies) is logged, never returned.
pub const MSG_SERVICE_BUSY: &str =
    "The service is handling many requests. Please wait 30 seconds and try again.";
pub const MSG_DATA_UNAVAILABLE: &str =
    "Market data is temporarily unavailable. Please try again in a moment.";
pub const MSG_AI_UNAVAILABLE: &str = "The AI service is temporarily busy. Please try again in a moment.";
pub const MSG_INTERNAL: &str = "Something went wrong on our end. Please try again.";

/// `retry_after_seconds` in a `service_busy` body, and its `Retry-After` header.
pub const RETRY_AFTER_SECONDS: u64 = 30;

impl ApiError {
    /// 503 `service_busy`: the `/ask` concurrency limiter's queue timed out.
    pub fn service_busy() -> Self {
        ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "service_busy",
            message: MSG_SERVICE_BUSY.to_string(),
            extra: Some(serde_json::json!({ "retry_after_seconds": RETRY_AFTER_SECONDS })),
        }
    }

    pub fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
            extra: None,
        }
    }

    pub fn not_found(code: &'static str, message: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::NOT_FOUND,
            code,
            message: message.into(),
            extra: None,
        }
    }

    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        ApiError {
            status,
            code,
            message: message.into(),
            extra: None,
        }
    }

    /// Like `bad_request`, but with extra fields merged into the response
    /// JSON alongside `error`/`code`. `extra` must serialize to a JSON
    /// object -- its keys become top-level response fields.
    pub fn bad_request_with_extra(code: &'static str, message: impl Into<String>, extra: serde_json::Value) -> Self {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
            extra: Some(extra),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(mut self) -> Response {
        // Internal failures never expose their raw detail to the caller.
        if matches!(self.code, "internal_error" | "compute_error") {
            tracing::error!(code = self.code, detail = %self.message, "internal error");
            self.message = MSG_INTERNAL.to_string();
            self.status = StatusCode::INTERNAL_SERVER_ERROR;
        }
        let retry_after = self
            .extra
            .as_ref()
            .and_then(|e| e.get("retry_after_seconds"))
            .and_then(serde_json::Value::as_u64);
        let mut body = serde_json::json!({
            "error": self.message,
            "code": self.code,
        });
        if let Some(extra) = self.extra {
            if let (Some(base), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
                base.extend(extra.clone());
            }
        }
        let mut response = (self.status, Json(body)).into_response();
        if let Some(secs) = retry_after {
            response.headers_mut().insert(axum::http::header::RETRY_AFTER, secs.into());
        }
        response
    }
}

impl From<BackendError> for ApiError {
    fn from(err: BackendError) -> Self {
        match err {
            BackendError::Unrecognised(message) => ApiError {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                code: "unrecognised_request",
                message,
                extra: None,
            },
            // `/ask` intercepts `Redirect` and answers 200 itself; this arm
            // only exists for any other route that might surface one.
            BackendError::Redirect(message) => ApiError {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                code: "unrecognised_request",
                message,
                extra: None,
            },
            BackendError::UnresolvedTicker(message) => ApiError {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                code: "unresolved_ticker",
                message,
                extra: None,
            },
            BackendError::InsufficientData(message) => ApiError {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                code: "insufficient_data",
                message,
                extra: None,
            },
            BackendError::DataUnavailable(detail) => {
                tracing::warn!(%detail, "market data unavailable");
                ApiError {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    code: "data_unavailable",
                    message: MSG_DATA_UNAVAILABLE.to_string(),
                    extra: None,
                }
            }
            BackendError::Compute(message) => ApiError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                code: "compute_error",
                message,
                extra: None,
            },
            BackendError::GeminiUnavailable(detail) => {
                tracing::warn!(%detail, "gemini unavailable");
                ApiError {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    code: "ai_unavailable",
                    message: MSG_AI_UNAVAILABLE.to_string(),
                    extra: None,
                }
            }
            BackendError::Internal(message) => ApiError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                code: "internal_error",
                message,
                extra: None,
            },
        }
    }
}

/// A `Json<T>` extractor whose rejection (malformed JSON, wrong content
/// type, missing/mistyped fields) is reported in the same `{error, code}`
/// shape as every other error response, instead of axum's default plain
/// text.
pub struct AppJson<T>(pub T);

#[async_trait::async_trait]
impl<T, S> FromRequest<S> for AppJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(AppJson(value)),
            Err(rejection) => Err(ApiError {
                status: rejection.status(),
                code: "invalid_json",
                message: rejection.body_text(),
                extra: None,
            }),
        }
    }
}

/// Safety net: any error response a handler or extractor produced that is
/// not already JSON (axum's plain-text extractor rejections, a bare 405,
/// a body-limit 413, ...) is rewritten into the standard `{error, code}`
/// shape, so no error path ever returns an empty body or raw text.
pub async fn ensure_json_errors(req: Request, next: axum::middleware::Next) -> Response {
    let response = next.run(req).await;
    let status = response.status();
    if !(status.is_client_error() || status.is_server_error()) {
        return response;
    }
    let is_json = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("application/json"));
    if is_json {
        return response;
    }
    let (code, fallback) = match status {
        StatusCode::NOT_FOUND => ("not_found", "Not found."),
        StatusCode::METHOD_NOT_ALLOWED => ("method_not_allowed", "That method is not allowed here."),
        StatusCode::PAYLOAD_TOO_LARGE => ("payload_too_large", "The request body is too large."),
        StatusCode::UNSUPPORTED_MEDIA_TYPE => ("unsupported_media_type", "Unsupported content type."),
        s if s.is_server_error() => ("internal_error", MSG_INTERNAL),
        _ => ("bad_request", "The request could not be processed."),
    };
    let (parts, body) = response.into_parts();
    let text = axum::body::to_bytes(body, 8 * 1024).await.unwrap_or_default();
    let text = String::from_utf8_lossy(&text).trim().to_string();
    let message = if status.is_server_error() {
        if !text.is_empty() {
            tracing::error!(%status, detail = %text, "non-JSON server error");
        }
        MSG_INTERNAL.to_string()
    } else if text.is_empty() {
        fallback.to_string()
    } else {
        text
    };
    let mut out = (parts.status, Json(serde_json::json!({ "error": message, "code": code }))).into_response();
    for name in [axum::http::header::ALLOW, axum::http::header::RETRY_AFTER] {
        if let Some(v) = parts.headers.get(&name) {
            out.headers_mut().insert(name, v.clone());
        }
    }
    out
}
