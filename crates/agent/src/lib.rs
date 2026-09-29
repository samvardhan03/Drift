//! The LLM-facing agent layer: NL request -> `compute::Experiment` ->
//! `EvidenceTrace` -> grounded plain-language narration.

pub mod conversation;
pub mod execution_trace;
pub mod gemini;
pub mod grounding;
pub mod narrate;
pub mod orchestrator;
pub mod parse;
pub mod pipeline;
pub mod schema;
pub mod suggest;

pub use conversation::ConversationTurn;
pub use execution_trace::{AgentExecutionTrace, GroundingStatus, ToolResult};
pub use gemini::{GeminiClient, GeminiError, GeminiRequest, GeminiResponse, HttpGeminiClient};
pub use grounding::{grounded_narrate, GroundedNarration};
pub use narrate::{narrate, NarrateError};
pub use orchestrator::{plan_tools, run_tool_plans, OrchestratorError, RiskTool, ToolPlan};
pub use parse::{parse_experiment, ParseError};
pub use pipeline::{run, PipelineError, PipelineResult};
pub use schema::experiment_function_declarations;
pub use suggest::{suggest_follow_up, SuggestError};
