//! `AgentExecutionTrace`: a record of one `/ask` orchestration run itself
//! (the planning call, each tool's execution, the combined narration, and
//! grounding outcome) -- distinct from `compute::trace::EvidenceTrace`,
//! which records one *experiment's* evidence. Persisted by the server
//! alongside the risk snapshot so `GET /execution-trace/{id}` can return
//! exactly what the orchestrator did for a given `/ask` call, independent
//! of the risk snapshot it produced.

use serde::{Deserialize, Serialize};

use crate::orchestrator::ToolPlan;

/// One planned tool's execution outcome. `trace_id` is empty and `error`
/// is set when `success` is `false` -- a failed tool never produces an
/// `EvidenceTrace` to point at.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool: String,
    pub trace_id: String,
    pub latency_ms: u64,
    pub success: bool,
    #[serde(default)]
    pub error: Option<String>,
}

/// The outcome of `agent::grounding::grounded_narrate_many`'s retry loop.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GroundingStatus {
    pub passed: bool,
    pub warnings: Vec<String>,
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentExecutionTrace {
    pub id: String,
    pub created_at: String,
    pub user_message: String,
    /// The planning call's raw text response, verbatim -- kept even when
    /// it failed to parse as a `Vec<ToolPlan>` (see
    /// `orchestrator::plan_tools`'s fallback), so a failed plan is still
    /// diagnosable from this trace alone.
    pub planning_response_raw: String,
    pub tool_plans: Vec<ToolPlan>,
    pub tool_results: Vec<ToolResult>,
    pub narration: String,
    pub grounding_status: GroundingStatus,
    pub suggestion: String,
    pub total_latency_ms: u64,
    /// Total Gemini calls this `/ask` made: 1 planning call, `1 +
    /// grounding_status.retry_count` narration calls, and 1 suggestion
    /// call.
    pub gemini_calls: u32,
}
