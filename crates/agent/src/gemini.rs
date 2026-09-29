//! A minimal async Gemini client: request/response types for the
//! `generateContent` endpoint, function-calling support, and a retrying
//! HTTP transport. `GeminiClient` is a trait so `parse`/`narrate` can be
//! tested against a mock implementation with no network access.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";
// Cheap models for the two non-narration calls; the narration call keeps the
// stronger flash model since its output is user-facing prose that also has
// to survive the grounding check in `grounding::grounded_narrate`.
//
// Both gemini-2.5-flash-lite and gemini-2.5-flash are documented as still
// valid but return live 404s for this project ("no longer available to new
// users... use models/gemini-3.5-flash-lite" / "...models/gemini-3.8-flash"
// — confirmed 2026-09-26), so this uses Google's own suggested replacements
// instead.
pub const MODEL_PARSE: &str = "gemini-3.5-flash-lite";
pub const MODEL_SUGGEST: &str = "gemini-3.5-flash-lite";
pub const MODEL_NARRATE: &str = "gemini-3.8-flash";
const MAX_ATTEMPTS: u32 = 3;

#[derive(Debug, Error)]
pub enum GeminiError {
    #[error("GEMINI_API_KEY environment variable is not set")]
    MissingApiKey,
    #[error("http request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("gemini returned status {status}: {body}")]
    Status { status: u16, body: String },
    #[error("failed to parse gemini response as json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("gemini response had no candidates")]
    NoCandidates,
}

/// One turn's content: a role (absent for `system_instruction`) plus parts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<Part>,
}

/// A single part of a `Content`. Gemini's wire format has at most one of
/// these fields set per part, so this mirrors that as optional fields
/// rather than a tagged enum (which would not round-trip Gemini's actual
/// JSON shape).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

impl Part {
    pub fn text(text: impl Into<String>) -> Self {
        Part {
            text: Some(text.into()),
            function_call: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    /// JSON Schema (as accepted by Gemini's function-calling subset) for
    /// the function's arguments.
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

impl GeminiRequest {
    /// A single user-turn request with a system instruction and no tools
    /// (the narration call shape).
    pub fn user_turn(system_prompt: impl Into<String>, user_text: impl Into<String>) -> Self {
        GeminiRequest {
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![Part::text(user_text)],
            }],
            system_instruction: Some(Content {
                role: None,
                parts: vec![Part::text(system_prompt)],
            }),
            tools: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Candidate {
    pub content: Content,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeminiResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
}

impl GeminiResponse {
    /// The first candidate's first part, whichever of text/function_call it has.
    pub fn first_part(&self) -> Option<&Part> {
        self.candidates.first()?.content.parts.first()
    }
}

/// Anything that can execute a Gemini `generateContent` call. A trait so
/// tests can supply a mock implementation with canned responses instead of
/// making network calls.
#[async_trait::async_trait]
pub trait GeminiClient: Send + Sync {
    async fn generate(
        &self,
        model: &str,
        request: &GeminiRequest,
    ) -> Result<GeminiResponse, GeminiError>;
}

/// The real client: POSTs to Gemini's `generateContent` endpoint, retrying
/// up to `MAX_ATTEMPTS` times with exponential backoff on 429/503, on a
/// transport-level send failure (DNS hiccup, connection reset, TLS
/// handshake failure), and on a response body read/decode failure --
/// caught live in this session's verification, all three occurring on an
/// otherwise-successful run against a real network. Previously only the
/// 429/503 case retried; a `send()`/body-read failure propagated
/// immediately via `?`, with no retry at all, even though it's no less
/// transient than a 503.
pub struct HttpGeminiClient {
    http: reqwest::Client,
    api_key: String,
}

impl HttpGeminiClient {
    /// Reads `GEMINI_API_KEY` from the environment.
    pub fn new() -> Result<Self, GeminiError> {
        let api_key = std::env::var("GEMINI_API_KEY").map_err(|_| GeminiError::MissingApiKey)?;
        Ok(HttpGeminiClient {
            http: reqwest::Client::new(),
            api_key,
        })
    }
}

#[async_trait::async_trait]
impl GeminiClient for HttpGeminiClient {
    async fn generate(
        &self,
        model: &str,
        request: &GeminiRequest,
    ) -> Result<GeminiResponse, GeminiError> {
        let url = format!("{API_BASE}/{model}:generateContent");
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let retry_or_return = |err: GeminiError, attempt: u32| async move {
                if attempt < MAX_ATTEMPTS {
                    let backoff = Duration::from_millis(250 * 2u64.pow(attempt - 1));
                    tokio::time::sleep(backoff).await;
                    None
                } else {
                    Some(err)
                }
            };

            // The API key is sent as a header, not a `?key=...` query
            // parameter: `reqwest::Error`'s `Display` (surfaced via
            // `GeminiError::Http`) includes the request URL on a
            // transport-level failure (DNS, TLS, connect timeout, etc.),
            // which would otherwise leak the key into logs/error
            // responses. Gemini's `generateContent` accepts either form;
            // this sidesteps the leak vector entirely rather than relying
            // on scrubbing every place an error might surface.
            let sent = self
                .http
                .post(&url)
                .header("x-goog-api-key", &self.api_key)
                .json(request)
                .send()
                .await;
            let resp = match sent {
                Ok(resp) => resp,
                Err(e) => match retry_or_return(GeminiError::Http(e), attempt).await {
                    None => continue,
                    Some(e) => return Err(e),
                },
            };
            let status = resp.status();
            if status.is_success() {
                let text = match resp.text().await {
                    Ok(text) => text,
                    Err(e) => match retry_or_return(GeminiError::Http(e), attempt).await {
                        None => continue,
                        Some(e) => return Err(e),
                    },
                };
                let body: GeminiResponse = serde_json::from_str(&text)?;
                return Ok(body);
            }
            let retryable = status.as_u16() == 429 || status.as_u16() == 503;
            if retryable && attempt < MAX_ATTEMPTS {
                let backoff = Duration::from_millis(250 * 2u64.pow(attempt - 1));
                tokio::time::sleep(backoff).await;
                continue;
            }
            let body = resp.text().await.unwrap_or_default();
            return Err(GeminiError::Status {
                status: status.as_u16(),
                body,
            });
        }
    }
}

/// Calls `client.generate`, a thin free function matching the spec's
/// `generate(client, request) -> Result<GeminiResponse, GeminiError>` shape.
pub async fn generate<C: GeminiClient>(
    client: &C,
    model: &str,
    request: &GeminiRequest,
) -> Result<GeminiResponse, GeminiError> {
    client.generate(model, request).await
}
