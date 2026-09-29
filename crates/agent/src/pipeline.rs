//! The single entry point the `server` crate calls: NL message + portfolio
//! in, one or more planned tool calls run against the compute layer, a
//! single grounded narration covering all of them, and an
//! `AgentExecutionTrace` recording the orchestration itself, out.

use compute::context::ExperimentContext;
use compute::experiments::{Experiment, Portfolio};
use compute::trace::EvidenceTrace;
use thiserror::Error;

use crate::conversation::ConversationTurn;
use crate::execution_trace::{AgentExecutionTrace, GroundingStatus};
use crate::gemini::GeminiClient;
use crate::grounding::GroundedNarration;
use crate::narrate::NarrateError;
use crate::orchestrator::{plan_tools, run_tool_plans, ToolPlan};
use crate::suggest::suggest_follow_up;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("planning error: {0}")]
    Orchestrator(#[from] crate::orchestrator::OrchestratorError),
    #[error("compute error: {0}")]
    Compute(#[from] compute::ComputeError),
    #[error("narration error: {0}")]
    Narrate(#[from] NarrateError),
    #[error("suggestion error: {0}")]
    Suggest(#[from] crate::suggest::SuggestError),
    /// No planned tool produced a trace (every one failed) -- there is
    /// nothing left to narrate or persist. Carries the per-tool errors.
    #[error("every planned tool failed: {0:?}")]
    AllToolsFailed(Vec<String>),
}

#[derive(Clone)]
pub struct PipelineResult {
    /// The first planned tool's `Experiment`/`EvidenceTrace` -- kept for
    /// backward compatibility with callers written against the single-tool
    /// pipeline (e.g. `server::routes::post_ask`'s existing snapshot
    /// persistence). A multi-tool run's full detail lives in `traces`.
    pub experiment: Experiment,
    pub trace: EvidenceTrace,
    /// Every tool's `EvidenceTrace`, in plan order (length 1 for a
    /// single-tool run).
    pub traces: Vec<EvidenceTrace>,
    pub tool_plans: Vec<ToolPlan>,
    pub narration: GroundedNarration,
    /// This turn's narration, as an `assistant` `ConversationTurn` ready
    /// for the caller to append to its `conversation_history` before the
    /// next request.
    pub assistant_turn: ConversationTurn,
    /// One follow-up question a risk manager would naturally ask next
    /// (see `suggest`). Not grounding-checked -- it's a question, not a
    /// factual claim about the trace.
    pub suggestion: String,
    pub execution_trace: AgentExecutionTrace,
}

/// plan -> run each planned tool -> narrate all of them together (with a
/// cross-trace grounding check) -> suggest a follow-up.
/// `conversation_history` (if any) is threaded through the planning and
/// narration Gemini calls so multi-turn references resolve correctly; an
/// empty history behaves exactly as before it existed. `ctx` carries the
/// snapshot store (read by `risk_drift`, and written mid-plan when a
/// `current_risk` immediately precedes a `risk_drift` -- see
/// `orchestrator::run_tool_plans`) and the optional passive/active policy.
pub async fn run<C: GeminiClient>(
    client: &C,
    user_message: &str,
    portfolio: Portfolio,
    conversation_history: &[ConversationTurn],
    ctx: &ExperimentContext,
) -> Result<PipelineResult, PipelineError> {
    let total_started = std::time::Instant::now();
    let mut gemini_calls: u32 = 0;

    let (tool_plans, planning_response_raw) =
        plan_tools(client, user_message, conversation_history).await?;
    gemini_calls += 1;

    let portfolio_for_tools = portfolio.clone();
    let ctx_for_tools = ctx.clone();
    let tool_plans_for_compute = tool_plans.clone();
    let (traces, tool_results) = tokio::task::spawn_blocking(move || {
        run_tool_plans(&tool_plans_for_compute, &portfolio_for_tools, &ctx_for_tools)
    })
    .await
    .expect("run_tool_plans task panicked");

    if traces.is_empty() {
        let errors: Vec<String> = tool_results.into_iter().filter_map(|r| r.error).collect();
        return Err(PipelineError::AllToolsFailed(errors));
    }

    let summary = traces.iter().map(experiment_summary).collect::<Vec<_>>().join(" ");
    let (narration_result, suggestion_result) = tokio::join!(
        crate::grounding::grounded_narrate_many(client, &traces, conversation_history),
        suggest_follow_up(client, &summary),
    );
    let (narration, narration_retries) = narration_result?;
    gemini_calls += 1 + narration_retries;
    let suggestion = suggestion_result?;
    gemini_calls += 1;

    let assistant_turn = ConversationTurn::assistant(narration.narration.clone());
    let grounding_status = GroundingStatus {
        passed: narration.grounding_warnings.is_empty(),
        warnings: narration.grounding_warnings.clone(),
        retry_count: narration_retries,
    };
    let execution_trace = AgentExecutionTrace {
        id: compute::trace::new_trace_id(),
        created_at: chrono::Utc::now().to_rfc3339(),
        user_message: user_message.to_string(),
        planning_response_raw,
        tool_plans: tool_plans.clone(),
        tool_results,
        narration: narration.narration.clone(),
        grounding_status,
        suggestion: suggestion.clone(),
        total_latency_ms: total_started.elapsed().as_millis() as u64,
        gemini_calls,
    };

    // `experiment` (for backward compat) is reconstructed from the first
    // trace's own recorded `inputs` + experiment tag, not re-derived from
    // the plan -- simplest way to get a real `Experiment` value without
    // threading one back out of `run_tool_plans`, which only returns
    // traces (a failed tool has no `Experiment` to alias anyway).
    let experiment = experiment_from_trace(&traces[0])?;

    Ok(PipelineResult {
        experiment,
        trace: traces[0].clone(),
        traces,
        tool_plans,
        narration,
        assistant_turn,
        suggestion,
        execution_trace,
    })
}

/// Reconstructs the `Experiment` that produced `trace`, from its own
/// recorded `experiment` tag + `inputs` (both already stored on every
/// trace). Used only for `PipelineResult::experiment`'s backward-compat
/// alias -- the trace itself is the authoritative record either way.
fn experiment_from_trace(trace: &EvidenceTrace) -> Result<Experiment, PipelineError> {
    let mut map = trace.inputs.as_object().cloned().unwrap_or_default();
    map.insert("type".to_string(), serde_json::Value::String(trace.experiment.clone()));
    serde_json::from_value(serde_json::Value::Object(map))
        .map_err(|e| PipelineError::Compute(compute::ComputeError::InvalidInput(e.to_string())))
}

/// A short plain-text summary of `trace` (experiment type + one key output
/// number), used as `suggest_follow_up`'s input instead of the narration --
/// the two Gemini calls run concurrently (see `run`), so suggest can't wait
/// on narrate's output.
fn experiment_summary(trace: &EvidenceTrace) -> String {
    let result = trace.outputs.get("result");
    let number = |path: &[&str]| -> Option<f64> {
        let mut v = result?;
        for key in path {
            v = v.get(key)?;
        }
        v.as_f64()
    };
    match trace.experiment.as_str() {
        "FactorShock" => match number(&["portfolio_pnl_inr"]) {
            Some(pnl) => format!("FactorShock experiment result: portfolio P&L is {pnl:.2} INR."),
            None => "FactorShock experiment result.".to_string(),
        },
        "RiskDecomposition" => match number(&["portfolio_vol_annualized"]) {
            Some(vol) => {
                format!("RiskDecomposition experiment result: annualised portfolio vol is {vol:.6}.")
            }
            None => "RiskDecomposition experiment result.".to_string(),
        },
        "CvarRebalance" => {
            let before = number(&["stats_before", "historical_cvar"]);
            let after = number(&["stats_after", "historical_cvar"]);
            match (before, after) {
                (Some(b), Some(a)) => format!(
                    "CvarRebalance experiment result: historical CVaR went from {b:.6} to {a:.6}."
                ),
                _ => "CvarRebalance experiment result.".to_string(),
            }
        }
        "PortfolioPerformance" => match number(&["total_return"]) {
            Some(r) => format!(
                "PortfolioPerformance experiment result: total return over the window is {r:.4}."
            ),
            None => "PortfolioPerformance experiment result.".to_string(),
        },
        "RiskDrift" => match number(&["vol_change_pct"]) {
            Some(pct) => format!("RiskDrift experiment result: portfolio vol changed by {pct:.2}%."),
            None => "RiskDrift experiment result.".to_string(),
        },
        "ReverseStress" => match number(&["mahalanobis_severity"]) {
            Some(severity) => format!(
                "ReverseStress experiment result: minimum-severity breaching shock has Mahalanobis severity {severity:.2}."
            ),
            None => "ReverseStress experiment result.".to_string(),
        },
        "PolicyCheck" => {
            let all_passed = result.and_then(|r| r.get("policy_result")).and_then(|r| r.get("all_passed")).and_then(|v| v.as_bool());
            match all_passed {
                Some(true) => "PolicyCheck experiment result: portfolio passes all policy checks.".to_string(),
                Some(false) => "PolicyCheck experiment result: portfolio breaches at least one policy check.".to_string(),
                None => "PolicyCheck experiment result.".to_string(),
            }
        }
        other => format!("{other} experiment result."),
    }
}

/// Runs the compute-layer path (data fetch + model fit + the appropriate
/// experiment) for `experiment`. Synchronous/blocking (see the module-level
/// note in `run`'s body about `spawn_blocking`); exposed so `server`'s
/// direct `/experiment` route can reuse it without a Gemini round trip.
///
/// A thin wrapper over `compute::dispatch::run_experiment` -- the fetch/
/// fit/dispatch logic itself lives in `compute` now, not here, since
/// `RiskDrift` needs read access to the snapshot store (`ctx.store`) to
/// resolve its baseline, and `store` is a dependency of `compute`, not of
/// this (narrower, Gemini-facing) crate.
pub fn compute_trace(
    experiment: &Experiment,
    portfolio: &Portfolio,
    ctx: &ExperimentContext,
) -> compute::Result<EvidenceTrace> {
    compute::dispatch::run_experiment(experiment, portfolio, ctx)
}
