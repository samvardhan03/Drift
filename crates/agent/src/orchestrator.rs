//! Multi-tool orchestrator: a single Gemini "planning" call decides which
//! of the eight `RiskTool`s to run (and with what parameters) for a given
//! natural-language message, each planned tool is executed sequentially
//! against the compute layer, and a single narration call covers every
//! tool's evidence together, grounded against their combined numeric
//! leaves. Replaces `parse::parse_experiment` as `/ask`'s entry point
//! (single-function-call -> multi-tool) -- `parse` itself is unchanged and
//! still used by `POST /experiment`'s direct (non-`/ask`) path.

use compute::context::ExperimentContext;
use compute::experiments::{
    CvarRebalanceInput, Experiment, FactorShockInput, PolicyCheckInput, Portfolio,
    PortfolioPerformanceInput, ReverseStressInput, RiskDecompositionInput, RiskDriftInput,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::conversation::{turn_to_content, ConversationTurn};
use crate::gemini::{Content, GeminiClient, GeminiRequest, Part, MODEL_PARSE};

/// The eight tools the planner can choose from. `historical_stress` is
/// the only one with no matching `compute::experiments::Experiment`
/// variant of its own -- it resolves to a `FactorShock` built from one of
/// `compute::scenarios::all_scenarios()`'s fixed shock sets, with the
/// resulting trace's `scenario_provenance` attached afterward (see
/// `execute_one`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskTool {
    CurrentRisk,
    RiskDrift,
    FactorShock,
    ReverseStress,
    HistoricalStress,
    CvarRebalance,
    PolicyCheck,
    PortfolioPerformance,
}

impl RiskTool {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "current_risk" => RiskTool::CurrentRisk,
            "risk_drift" => RiskTool::RiskDrift,
            "factor_shock" => RiskTool::FactorShock,
            "reverse_stress" => RiskTool::ReverseStress,
            "historical_stress" => RiskTool::HistoricalStress,
            "cvar_rebalance" => RiskTool::CvarRebalance,
            "policy_check" => RiskTool::PolicyCheck,
            "portfolio_performance" => RiskTool::PortfolioPerformance,
            _ => return None,
        })
    }
}

/// One planned tool call, as the planning Gemini response is expected to
/// name it: `{"tool": "risk_drift", "params": {...}, "reason": "..."}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPlan {
    pub tool: String,
    #[serde(default)]
    pub params: Value,
    #[serde(default)]
    pub reason: String,
}

/// The planning call's system prompt.
///
/// **Judgment call**: this session's own planning prompt (as originally
/// specified) was lost to context compaction partway through this session
/// and could not be recovered verbatim -- reconstructed here from the
/// spec's described contract (eight named tools, each tool's params, plain
/// JSON-array output with no function-calling) rather than risk presenting
/// a paraphrase as a verbatim quote. Flagged explicitly in this session's
/// report; every other verbatim prompt in this codebase (narration,
/// suggest, parse) is unaffected.
pub const PLANNING_SYSTEM_PROMPT: &str = "You are a planning engine for a portfolio risk copilot. \
Given the user's message, decide which risk tools to run, in what order, and with what \
parameters. Available tools and their params:
- current_risk: current portfolio volatility and risk decomposition. params: {frequency?, window?}
- risk_drift: how portfolio risk has changed since a prior snapshot. params: {baseline_snapshot_id?, frequency?, window?}
- factor_shock: apply a hypothetical shock to one or more factors (MARKET, USDINR, BRENT, GOLD_USD, RATES_PROXY). params: {shocks_pct: {FACTOR: percent}, propagate?}
- historical_stress: replay a named historical scenario. params: {scenario_id} where scenario_id is one of covid_crash, ilfs_contagion, taper_tantrum_2013
- reverse_stress: find the smallest shock that would breach a loss threshold. params: {loss_threshold_inr, factor_bounds?}
- cvar_rebalance: propose a rebalance that reduces tail risk (CVaR) within a turnover budget. params: {turnover_limit, confidence_level?, per_name_cap?, commission_bps?}
- policy_check: check the portfolio against risk limits. params: {policy: {max_vol_annualized?, max_cvar_95?, max_factor_contribution_share?, max_position_weight?}}
- portfolio_performance: realized historical performance over a trailing window. params: {frequency?, window?}
- decline: the message has nothing to do with this portfolio's risk, performance, or a market \
scenario (e.g. small talk, general knowledge, something unrelated like the weather). params: {}. \
reason must be exactly the one-sentence, polite decline to show the user directly (not a note to \
yourself) -- e.g. \"I can only help with questions about your portfolio's risk and performance.\" \
This must be the only entry in the array when used.

- If the user asks about a specific stock, best/worst performer, or individual holding returns, use \
portfolio_performance -- it includes per-holding data.
- If the user asks a follow-up that references a prior result ('now reduce it', 'what about a bigger \
crash', 'which stock is dragging me down'), infer the experiment from context -- do not ask for \
clarification.
- If the user asks what to do, what action to take, or how to fix their portfolio, select \
cvar_rebalance.
- If the user expresses concern about a market event ('what if RBI raises rates', 'what about the US \
election', 'crude is spiking'), select factor_shock with the relevant factor shocked.

Respond with ONLY a JSON array, no prose, no markdown code fences: \
[{\"tool\": <tool name>, \"params\": <object>, \"reason\": <one short sentence>}, ...]. \
Plan exactly one tool unless the user's message clearly asks for more than one distinct thing \
(e.g. both a hypothetical shock and a policy check, or both current risk and how it has changed \
since last time). When a later tool depends on an earlier one's result, order the earlier one \
first (e.g. current_risk before risk_drift, if both are needed). Never invent a parameter value \
the user's message does not support -- omit it and let the tool use its own default. \
Portfolio holdings and weights are supplied separately; never include a \"portfolio\" field.";

#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("gemini error: {0}")]
    Gemini(#[from] crate::gemini::GeminiError),
    #[error("failed to (de)serialize tool params: {0}")]
    Serialize(#[from] serde_json::Error),
    /// The planning call decided the message doesn't relate to the
    /// portfolio at all (see `PLANNING_SYSTEM_PROMPT`'s `decline` tool) --
    /// the model's own one-sentence decline, to return to the caller
    /// as-is. Mirrors `parse::ParseError::Unrecognised`'s contract from
    /// the pre-orchestrator single-tool pipeline (a 422, not a 500 --
    /// see `server::backend`'s `From` impl).
    #[error("{0}")]
    Unrecognised(String),
}

/// Runs the planning call and parses its response into `Vec<ToolPlan>`,
/// falling back to a single `current_risk` plan (recording the raw
/// response either way) if the response isn't valid JSON, isn't an array,
/// or is empty. Returns `(plans, raw_response_text)` -- except when the
/// plan is a single `decline` entry (see `PLANNING_SYSTEM_PROMPT`), which
/// returns `Err(OrchestratorError::Unrecognised)` instead: an off-topic
/// message runs no tool at all, rather than falling back to one.
pub async fn plan_tools<C: GeminiClient>(
    client: &C,
    user_message: &str,
    conversation_history: &[ConversationTurn],
) -> Result<(Vec<ToolPlan>, String), OrchestratorError> {
    let mut contents: Vec<Content> = conversation_history.iter().map(turn_to_content).collect();
    contents.push(Content {
        role: Some("user".to_string()),
        parts: vec![Part::text(user_message)],
    });
    let request = GeminiRequest {
        contents,
        system_instruction: Some(Content {
            role: None,
            parts: vec![Part::text(PLANNING_SYSTEM_PROMPT)],
        }),
        tools: None,
    };

    let response = client.generate(MODEL_PARSE, &request).await?;
    let raw = response
        .first_part()
        .and_then(|p| p.text.clone())
        .unwrap_or_default();

    let plans = parse_plan_response(&raw);
    match plans {
        Some(plans) if plans.len() == 1 && plans[0].tool == "decline" => {
            Err(OrchestratorError::Unrecognised(plans[0].reason.clone()))
        }
        Some(plans) if !plans.is_empty() => Ok((plans, raw)),
        _ => Ok((
            vec![ToolPlan {
                tool: "current_risk".to_string(),
                params: serde_json::json!({}),
                reason: "fallback: planning response could not be parsed".to_string(),
            }],
            raw,
        )),
    }
}

/// Strips a markdown code fence if present (Gemini sometimes wraps JSON in
/// ```json ... ``` despite being told not to), then parses the remainder
/// as a `Vec<ToolPlan>`.
fn parse_plan_response(raw: &str) -> Option<Vec<ToolPlan>> {
    let trimmed = raw.trim();
    let trimmed = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    let trimmed = trimmed.strip_suffix("```").unwrap_or(trimmed).trim();
    serde_json::from_str::<Vec<ToolPlan>>(trimmed).ok()
}

fn inject_portfolio(params: &Value, portfolio: &Portfolio) -> Value {
    let mut map = match params {
        Value::Object(map) => map.clone(),
        _ => serde_json::Map::new(),
    };
    map.insert(
        "portfolio".to_string(),
        serde_json::to_value(portfolio).expect("Portfolio always serializes"),
    );
    Value::Object(map)
}

/// Builds the `Experiment` a plan's tool+params actually dispatch to.
/// `baseline_override` is `risk_drift`'s chained `current_risk` snapshot
/// id (see `run_tool_plans`'s doc) -- only used when the plan itself
/// didn't already specify `baseline_snapshot_id`.
fn build_experiment(
    tool: RiskTool,
    params: &Value,
    portfolio: &Portfolio,
    ctx: &ExperimentContext,
    baseline_override: Option<String>,
) -> Result<Experiment, OrchestratorError> {
    Ok(match tool {
        RiskTool::CurrentRisk => Experiment::RiskDecomposition(serde_json::from_value::<
            RiskDecompositionInput,
        >(inject_portfolio(params, portfolio))?),
        RiskTool::FactorShock => Experiment::FactorShock(serde_json::from_value::<FactorShockInput>(
            inject_portfolio(params, portfolio),
        )?),
        RiskTool::HistoricalStress => {
            let scenario_id = params
                .get("scenario_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let scenario = compute::scenarios::all_scenarios()
                .iter()
                .find(|s| s.id == scenario_id);
            let (shocks_pct, propagate): (std::collections::BTreeMap<String, f64>, bool) = match scenario {
                Some(s) => (
                    s.shocks_pct.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
                    s.propagate,
                ),
                // Unknown/missing scenario_id: fall through to a no-op
                // shock set rather than failing the whole plan -- the
                // per-tool ToolResult still records this tool ran, just
                // with an empty (zero-loss) shock, which is a visible,
                // diagnosable signal in the trace rather than a hard error.
                None => (Default::default(), true),
            };
            let mut map = match params {
                Value::Object(map) => map.clone(),
                _ => serde_json::Map::new(),
            };
            map.insert("shocks_pct".to_string(), serde_json::to_value(&shocks_pct)?);
            map.insert("propagate".to_string(), serde_json::json!(propagate));
            Experiment::FactorShock(serde_json::from_value::<FactorShockInput>(inject_portfolio(
                &Value::Object(map),
                portfolio,
            ))?)
        }
        RiskTool::CvarRebalance => Experiment::CvarRebalance(serde_json::from_value::<
            CvarRebalanceInput,
        >(inject_portfolio(params, portfolio))?),
        RiskTool::PortfolioPerformance => Experiment::PortfolioPerformance(serde_json::from_value::<
            PortfolioPerformanceInput,
        >(inject_portfolio(params, portfolio))?),
        RiskTool::RiskDrift => {
            let mut input = serde_json::from_value::<RiskDriftInput>(params.clone())?;
            if input.baseline_snapshot_id.is_none() {
                input.baseline_snapshot_id = baseline_override;
            }
            Experiment::RiskDrift(input)
        }
        RiskTool::ReverseStress => {
            Experiment::ReverseStress(serde_json::from_value::<ReverseStressInput>(params.clone())?)
        }
        RiskTool::PolicyCheck => {
            let mut map = match params {
                Value::Object(map) => map.clone(),
                _ => serde_json::Map::new(),
            };
            if !map.contains_key("policy") {
                let policy = ctx.policy.clone().unwrap_or_default();
                map.insert("policy".to_string(), serde_json::to_value(policy)?);
            }
            Experiment::PolicyCheck(serde_json::from_value::<PolicyCheckInput>(Value::Object(map))?)
        }
    })
}

/// Builds the minimal `store::RiskSnapshot` needed to persist `trace` mid-
/// plan purely so a later `risk_drift` in the same plan can chain against
/// it via a real snapshot id (see `run_tool_plans`). Deliberately a
/// smaller subset of fields than `server::routes::snapshot_from_trace`
/// (narration/suggestion aren't known yet at this point in the pipeline);
/// the server's own post-`/ask` persistence still separately stores the
/// full, narrated snapshot afterward.
fn snapshot_from_trace(
    trace: &compute::trace::EvidenceTrace,
    ctx: &ExperimentContext,
) -> Result<store::RiskSnapshot, OrchestratorError> {
    Ok(store::RiskSnapshot {
        id: String::new(),
        created_at: String::new(),
        portfolio_hash: ctx.portfolio_hash.clone(),
        experiment_type: trace.experiment.clone(),
        engine_version: trace.engine_version.clone(),
        regime_label: trace.model_params.regime_state.as_ref().map(|r| r.current_label.to_string()),
        smoothed_probs: trace.model_params.regime_state.as_ref().map(|r| r.smoothed_probs),
        portfolio_vol_annualized: trace
            .outputs
            .get("result")
            .and_then(|r| r.get("portfolio_vol_annualized"))
            .and_then(Value::as_f64),
        cvar_historical: None,
        trace_json: serde_json::to_string(trace)?,
        narration: None,
        suggestion: None,
        grounding_warnings: None,
    })
}

/// One tool's outcome: either its `EvidenceTrace` or the compute error it
/// failed with (kept as a string -- see `execution_trace::ToolResult`).
pub enum ToolOutcome {
    Trace(Box<compute::trace::EvidenceTrace>),
    Error(String),
}

/// Runs `plan` (an unrecognised `plan.tool` name is also an error outcome,
/// not a panic). Synchronous/blocking, same reasoning as
/// `compute::dispatch::run_experiment` -- callers run this via
/// `spawn_blocking`.
fn execute_one(
    plan: &ToolPlan,
    portfolio: &Portfolio,
    ctx: &ExperimentContext,
    baseline_override: Option<String>,
) -> ToolOutcome {
    let Some(tool) = RiskTool::parse(&plan.tool) else {
        return ToolOutcome::Error(format!("unknown tool {:?}", plan.tool));
    };
    let experiment = match build_experiment(tool, &plan.params, portfolio, ctx, baseline_override) {
        Ok(e) => e,
        Err(e) => return ToolOutcome::Error(e.to_string()),
    };
    match compute::dispatch::run_experiment(&experiment, portfolio, ctx) {
        Ok(mut trace) => {
            if tool == RiskTool::HistoricalStress {
                let scenario_id = plan.params.get("scenario_id").and_then(Value::as_str).unwrap_or_default();
                if let Some(s) = compute::scenarios::all_scenarios().iter().find(|s| s.id == scenario_id) {
                    trace.scenario_provenance = Some(compute::trace::ScenarioProvenance {
                        scenario_id: s.id.to_string(),
                        scenario_name: s.name.to_string(),
                        date_range: s.date_range.to_string(),
                        description: s.description.to_string(),
                    });
                }
            }
            ToolOutcome::Trace(Box::new(trace))
        }
        Err(e) => ToolOutcome::Error(e.to_string()),
    }
}

/// Runs every plan in `plans` sequentially (not concurrently -- a later
/// plan may depend on an earlier one's persisted snapshot, see below).
/// Returns `(traces, tool_results)`, both in plan order; a failed tool
/// contributes a `ToolResult` with `success: false` and no trace, but does
/// not stop later tools from running.
///
/// **Chaining**: when a `current_risk` plan is immediately followed by a
/// `risk_drift` plan in the same request, `current_risk`'s trace is
/// persisted to `ctx.store` right away (purely to obtain a real snapshot
/// id -- `risk_drift` can only resolve a baseline by store lookup, not
/// from an in-memory trace) and that id is threaded into the `risk_drift`
/// plan's `baseline_snapshot_id`, unless the plan already specified one.
/// This is the one cross-tool dependency the planning prompt is told to
/// order for; every other tool pair is independent.
pub fn run_tool_plans(
    plans: &[ToolPlan],
    portfolio: &Portfolio,
    ctx: &ExperimentContext,
) -> (Vec<compute::trace::EvidenceTrace>, Vec<crate::execution_trace::ToolResult>) {
    let mut traces = Vec::new();
    let mut results = Vec::new();
    let mut current_risk_snapshot_id: Option<String> = None;

    for (i, plan) in plans.iter().enumerate() {
        let started = std::time::Instant::now();
        let baseline_override = if plan.tool == "risk_drift" {
            current_risk_snapshot_id.clone()
        } else {
            None
        };
        let outcome = execute_one(plan, portfolio, ctx, baseline_override);
        let latency_ms = started.elapsed().as_millis() as u64;

        match outcome {
            ToolOutcome::Trace(trace) => {
                let next_is_risk_drift =
                    plans.get(i + 1).map(|p| p.tool == "risk_drift").unwrap_or(false);
                if plan.tool == "current_risk" && next_is_risk_drift {
                    if let Ok(snapshot) = snapshot_from_trace(&trace, ctx) {
                        if let Ok(id) = ctx.store.insert(&snapshot) {
                            current_risk_snapshot_id = Some(id);
                        }
                    }
                }
                results.push(crate::execution_trace::ToolResult {
                    tool: plan.tool.clone(),
                    trace_id: trace.id.clone(),
                    latency_ms,
                    success: true,
                    error: None,
                });
                traces.push(*trace);
            }
            ToolOutcome::Error(error) => {
                results.push(crate::execution_trace::ToolResult {
                    tool: plan.tool.clone(),
                    trace_id: String::new(),
                    latency_ms,
                    success: false,
                    error: Some(error),
                });
            }
        }
    }

    (traces, results)
}
