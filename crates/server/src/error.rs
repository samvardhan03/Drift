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
}

impl ApiError {
    pub fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
        }
    }

    pub fn not_found(code: &'static str, message: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::NOT_FOUND,
            code,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(serde_json::json!({
            "error": self.message,
            "code": self.code,
        }));
        (self.status, body).into_response()
    }
}

impl From<BackendError> for ApiError {
    fn from(err: BackendError) -> Self {
        match err {
            BackendError::Unrecognised(message) => ApiError {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                code: "unrecognised_request",
                message,
            },
            BackendError::Compute(message) => ApiError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                code: "compute_error",
                message,
            },
            BackendError::GeminiUnavailable(message) => ApiError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                code: "gemini_unavailable",
                message,
            },
            BackendError::Internal(message) => ApiError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                code: "internal_error",
                message,
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
            }),
        }
    }
}
