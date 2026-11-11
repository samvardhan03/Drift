//! A minimal async Gemini client: request/response types for the
//! `generateContent` endpoint, function-calling support, and a retrying
//! HTTP transport. `GeminiClient` is a trait so `parse`/`narrate` can be
//! tested against a mock implementation with no network access.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";
// gemini-2.5-flash-lite (tried first, per this session's cost-reduction
// pass) still 404s for this project -- "no longer available to new
// users... use models/gemini-3.5-flash-lite" -- confirmed live again
// 2026-10-01. gemini-3.1-flash-lite ($0.25/$1.50 per 1M, 5x cheaper than
// the narrate model this replaces) returned 200 live, so all three calls
// (parse, suggest, *and* narrate) now use it -- confirmed via a live /ask
// that narration quality and grounding are unaffected (see this session's
// report). Previously the narrate call alone used a stronger model
// (gemini-3.8-flash); that tradeoff is gone now that flash-lite's output
// has been verified to hold up for user-facing narration too.
pub const MODEL_PARSE: &str = "gemini-3.1-flash-lite";
pub const MODEL_SUGGEST: &str = "gemini-3.1-flash-lite";
pub const MODEL_NARRATE: &str = "gemini-3.1-flash-lite";
/// Retries after the first attempt (so up to 6 requests in all), with
/// exponential backoff of 2s, 4s, 8s, 16s, 32s between them.
const MAX_RETRIES: u32 = 5;
const DEFAULT_BACKOFF_BASE: Duration = Duration::from_secs(2);
/// Per-request timeout. Without one, a hung connection would never surface
/// as a retryable timeout. 30s keeps one call's worst case (6 x 30s plus
/// 62s of backoff) inside Cloud Run's 300s request limit.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

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
    /// Every retry of a retryable failure (429, 500, 503, timeout or
    /// transport error) was exhausted.
    #[error("gemini unavailable after {attempts} attempts: {last_error}")]
    Unavailable { attempts: u32, last_error: String },
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
/// up to `MAX_RETRIES` times with exponential backoff (2s, 4s, 8s, 16s,
/// 32s) on 429/500/503, on a timeout or transport-level send failure (DNS
/// hiccup, connection reset, TLS handshake failure), and on a response
/// body read failure, then returning `GeminiError::Unavailable` --
/// caught live in this session's verification, all three occurring on an
/// otherwise-successful run against a real network. Previously only the
/// 429/503 case retried; a `send()`/body-read failure propagated
/// immediately via `?`, with no retry at all, even though it's no less
/// transient than a 503.
pub struct HttpGeminiClient {
    http: reqwest::Client,
    api_key: String,
    api_base: String,
    backoff_base: Duration,
}

impl HttpGeminiClient {
    /// Reads `GEMINI_API_KEY` from the environment.
    pub fn new() -> Result<Self, GeminiError> {
        let api_key = std::env::var("GEMINI_API_KEY").map_err(|_| GeminiError::MissingApiKey)?;
        Ok(HttpGeminiClient {
            http: reqwest::Client::builder().timeout(REQUEST_TIMEOUT).build()?,
            api_key,
            api_base: API_BASE.to_string(),
            backoff_base: DEFAULT_BACKOFF_BASE,
        })
    }

    /// Points the client at a different endpoint with a different backoff
    /// base (tests only: a local stub server and millisecond backoff).
    #[doc(hidden)]
    pub fn with_endpoint_for_test(mut self, api_base: impl Into<String>, backoff_base: Duration) -> Self {
        self.api_base = api_base.into();
        self.backoff_base = backoff_base;
        self
    }

    fn backoff(&self, retry: u32) -> Duration {
        self.backoff_base * 2u32.pow(retry - 1)
    }
}

fn is_retryable_status(status: u16) -> bool {
    matches!(status, 429 | 500 | 503)
}

#[async_trait::async_trait]
impl GeminiClient for HttpGeminiClient {
    async fn generate(
        &self,
        model: &str,
        request: &GeminiRequest,
    ) -> Result<GeminiResponse, GeminiError> {
        let url = format!("{}/{model}:generateContent", self.api_base);
        let mut retries = 0u32;
        loop {
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

            // `Err` = a retryable failure (its description, with the URL
            // stripped so the key can never leak); `Ok` = a final outcome.
            let failure: String = match sent {
                Err(e) => e.without_url().to_string(),
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.text().await {
                            Ok(text) => return Ok(serde_json::from_str(&text)?),
                            Err(e) => e.without_url().to_string(),
                        }
                    } else if is_retryable_status(status.as_u16()) {
                        format!("status {}", status.as_u16())
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        return Err(GeminiError::Status { status: status.as_u16(), body });
                    }
                }
            };

            if retries >= MAX_RETRIES {
                return Err(GeminiError::Unavailable { attempts: retries + 1, last_error: failure });
            }
            retries += 1;
            tokio::time::sleep(self.backoff(retries)).await;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A stub Gemini endpoint: answers the i-th request with `statuses[i]`
    /// (the last status repeats); 200 carries a minimal valid body.
    fn stub(statuses: Vec<u16>) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut conn) = conn else { return };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                // Read headers, then the declared body, so the client never
                // sees a reset while still writing.
                loop {
                    let n = conn.read(&mut chunk).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    if let Some(idx) = text.find("\r\n\r\n") {
                        let len = text[..idx]
                            .lines()
                            .find_map(|l| l.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()))
                            .unwrap_or(0);
                        if buf.len() >= idx + 4 + len {
                            break;
                        }
                    }
                }
                let i = counter.fetch_add(1, Ordering::SeqCst);
                let status = *statuses.get(i).or(statuses.last()).unwrap();
                let body = if status == 200 {
                    r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]}}]}"#
                } else {
                    r#"{"error":"nope"}"#
                };
                let _ = write!(
                    conn,
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        (base, hits)
    }

    fn client(base: String) -> HttpGeminiClient {
        HttpGeminiClient {
            http: reqwest::Client::new(),
            api_key: "k".to_string(),
            api_base: base,
            backoff_base: Duration::from_millis(1),
        }
    }

    fn req() -> GeminiRequest {
        GeminiRequest::user_turn("s", "u")
    }

    #[test]
    fn backoff_sequence_is_2_4_8_16_32_seconds() {
        let c = HttpGeminiClient::new_for_backoff_test();
        let secs: Vec<u64> = (1..=5).map(|r| c.backoff(r).as_secs()).collect();
        assert_eq!(secs, vec![2, 4, 8, 16, 32]);
    }

    impl HttpGeminiClient {
        fn new_for_backoff_test() -> Self {
            client(String::new()).with_endpoint_for_test("", DEFAULT_BACKOFF_BASE)
        }
    }

    #[tokio::test]
    async fn retries_429_500_503_then_succeeds() {
        let (base, hits) = stub(vec![429, 500, 503, 200]);
        let resp = client(base).generate("m", &req()).await.unwrap();
        assert_eq!(resp.first_part().unwrap().text.as_deref(), Some("ok"));
        assert_eq!(hits.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn five_retries_then_unavailable() {
        let (base, hits) = stub(vec![503]);
        let err = client(base).generate("m", &req()).await.unwrap_err();
        assert!(matches!(err, GeminiError::Unavailable { attempts: 6, .. }), "{err:?}");
        assert_eq!(hits.load(Ordering::SeqCst), 6);
    }

    #[tokio::test]
    async fn non_retryable_status_fails_immediately() {
        let (base, hits) = stub(vec![400]);
        let err = client(base).generate("m", &req()).await.unwrap_err();
        assert!(matches!(err, GeminiError::Status { status: 400, .. }));
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_request_timeout_is_retried_then_unavailable() {
        // Accepts connections but never answers.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            let mut held = Vec::new();
            for conn in listener.incoming() {
                held.push(conn);
            }
        });
        let c = HttpGeminiClient {
            http: reqwest::Client::builder().timeout(Duration::from_millis(50)).build().unwrap(),
            ..client(base)
        };
        let err = c.generate("m", &req()).await.unwrap_err();
        assert!(matches!(err, GeminiError::Unavailable { attempts: 6, .. }), "{err:?}");
    }
}
