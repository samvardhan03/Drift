//! A mock `GeminiClient` for tests: returns a queue of canned responses
//! and counts how many times `generate` was called, with no network
//! access.
#![allow(dead_code)]

use std::sync::Mutex;

use agent::gemini::{Candidate, Content, FunctionCall, GeminiClient, GeminiError, GeminiRequest, GeminiResponse, Part};

pub struct MockGeminiClient {
    responses: Mutex<Vec<Result<GeminiResponse, String>>>,
    calls: Mutex<usize>,
    /// Every request `generate` was called with, oldest first, so tests can
    /// inspect exactly what was sent (e.g. conversation-history ordering).
    requests: Mutex<Vec<GeminiRequest>>,
    /// Every model string `generate` was called with, oldest first.
    models: Mutex<Vec<String>>,
}

impl MockGeminiClient {
    /// `responses` are returned in order, oldest first, one per call to
    /// `generate`. Panics (via the returned error) if `generate` is called
    /// more times than there are queued responses.
    pub fn new(responses: Vec<GeminiResponse>) -> Self {
        MockGeminiClient {
            responses: Mutex::new(responses.into_iter().map(Ok).rev().collect()),
            calls: Mutex::new(0),
            requests: Mutex::new(Vec::new()),
            models: Mutex::new(Vec::new()),
        }
    }

    pub fn call_count(&self) -> usize {
        *self.calls.lock().unwrap()
    }

    /// The most recent request `generate` was called with.
    pub fn last_request(&self) -> GeminiRequest {
        self.requests
            .lock()
            .unwrap()
            .last()
            .cloned()
            .expect("MockGeminiClient: generate was never called")
    }

    /// The most recent model string `generate` was called with.
    pub fn last_model(&self) -> String {
        self.models
            .lock()
            .unwrap()
            .last()
            .cloned()
            .expect("MockGeminiClient: generate was never called")
    }
}

#[async_trait::async_trait]
impl GeminiClient for MockGeminiClient {
    async fn generate(
        &self,
        model: &str,
        request: &GeminiRequest,
    ) -> Result<GeminiResponse, GeminiError> {
        *self.calls.lock().unwrap() += 1;
        self.requests.lock().unwrap().push(request.clone());
        self.models.lock().unwrap().push(model.to_string());
        let mut queue = self.responses.lock().unwrap();
        match queue.pop() {
            Some(Ok(resp)) => Ok(resp),
            Some(Err(_)) | None => panic!("MockGeminiClient: no more queued responses"),
        }
    }
}

/// Builds a `GeminiResponse` with one candidate whose single part is a
/// function call (`name`) with the given `args`. `name` should be one of
/// `agent::schema::{FACTOR_SHOCK_FUNCTION, RISK_DECOMPOSITION_FUNCTION,
/// CVAR_REBALANCE_FUNCTION}`.
pub fn function_call_response(name: &str, args: serde_json::Value) -> GeminiResponse {
    GeminiResponse {
        candidates: vec![Candidate {
            content: Content {
                role: Some("model".to_string()),
                parts: vec![Part {
                    text: None,
                    function_call: Some(FunctionCall {
                        name: name.to_string(),
                        args,
                    }),
                }],
            },
            finish_reason: Some("STOP".to_string()),
        }],
    }
}

/// Builds a `GeminiResponse` with one candidate whose single part is plain
/// text (the "I can't extract this" / narration response shape).
pub fn text_response(text: impl Into<String>) -> GeminiResponse {
    GeminiResponse {
        candidates: vec![Candidate {
            content: Content {
                role: Some("model".to_string()),
                parts: vec![Part::text(text)],
            },
            finish_reason: Some("STOP".to_string()),
        }],
    }
}
