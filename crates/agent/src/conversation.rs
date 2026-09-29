//! A single turn of prior conversation, as carried across requests by the
//! caller (`server`'s `/ask`) so `parse`/`narrate` can resolve references
//! to earlier turns ("now try with 25% turnover") without the caller
//! re-stating context.

use serde::{Deserialize, Serialize};

use crate::gemini::{Content, Part};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// `"user"` or `"assistant"`.
    pub role: String,
    pub content: String,
}

impl ConversationTurn {
    pub fn user(content: impl Into<String>) -> Self {
        ConversationTurn {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        ConversationTurn {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Maps our `"user"`/`"assistant"` roles onto Gemini's own `"user"`/`"model"`
/// role vocabulary.
pub fn turn_to_content(turn: &ConversationTurn) -> Content {
    let role = if turn.role == "assistant" { "model" } else { "user" };
    Content {
        role: Some(role.to_string()),
        parts: vec![Part::text(turn.content.clone())],
    }
}
