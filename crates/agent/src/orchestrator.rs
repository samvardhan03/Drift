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
This must be the only entry in the array when used. Only use this for questions completely \
unrelated to finance, investing, markets, or portfolio risk. Never use this for scenario \
questions, historical market events, or hypothetical market moves.

RULE 0 -- Greeting detection (applies before every other rule):
If the user message is a greeting with no portfolio question (hi, hello, hey, good morning, good \
evening, namaste, hola, thanks, thank you, ok, okay, sure, got it), use the decline tool with \
exactly this reason: 'Hello! I'm your portfolio risk analyst. Ask me anything about your \
portfolio -- try \"What is my risk right now?\" or \"What would a market crash do to me?\"' \
Do not run any experiment for a greeting.

ROUTING RULE 0 -- Stock vs Macro distinction (applies before every other rule):
- Macro factors (route to factor_shock): Market/Nifty/index, crude oil/Brent/petrol, rupee/INR/\
dollar/currency, gold, interest rates/RBI/repo rate, inflation, bond yields.
- Individual stocks (NEVER route to factor_shock or historical_stress): any company name, brand \
name, or NSE/BSE ticker symbol that is not a macro factor. Examples: Ratnaveer, NSIL, TCS, \
Reliance, Zydus, Infosys, HDFC, any stock ticker.
- If the user asks 'what if [STOCK] goes up/down by X%' where STOCK is an individual company, use \
decline with this exact reason (substituting the company name for [STOCK]): 'I cannot model \
individual stock price movements -- the system analyses macro factor risks, not individual stock \
forecasts. What I can do instead: 1. Show how [STOCK] has historically performed in your \
portfolio. 2. Show what a broader market drop would do to your entire portfolio. 3. Identify \
which macro factors affect your portfolio most. Which would be most useful?'
- NEVER use historical_stress unless the CURRENT user message explicitly names a historical event \
(COVID, IL&FS, taper tantrum, 2008 crisis). Do not infer a historical scenario from earlier \
conversation turns: a question about crude oil after an IL&FS question is a factor_shock, not \
historical_stress.

Rules for tool selection:

- General risk questions ('what is my risk', 'analyse my portfolio', 'how am I positioned') -> \
current_risk
- Questions about what changed, drift, comparison to before -> risk_drift (always run current_risk \
first if risk_drift is selected)
- Questions about commodity prices or macro:
  'crude', 'oil', 'petrol', 'diesel', 'Brent' going up/spike/rise -> factor_shock BRENT: +20.0
  'crude', 'oil' going down/fall/crash -> factor_shock BRENT: -20.0
  'rupee', 'INR', 'dollar' weakening/depreciating -> factor_shock USDINR: +5.0
  'gold' rising -> factor_shock GOLD_USD: +10.0
  'gold' falling -> factor_shock GOLD_USD: -10.0
  'interest rates', 'RBI hike', 'repo rate' rising -> factor_shock RATES_PROXY: +5.0
  'market crash', 'Nifty falls', 'bear market' -> factor_shock MARKET: -20.0 (use the user's own \
percentage when they give one, e.g. 'Nifty falls 15%' -> MARKET: -15.0)
- Questions about market-wide events or scenarios ('what if', 'what would happen if', 'what would \
X have done', 'impact of', 'effect of') where the subject is a macro factor or a named historical \
event -> factor_shock or historical_stress. These are ALWAYS portfolio risk questions.
- Named historical events (only when the current message names them):
  'COVID', 'covid crash', 'March 2020' -> historical_stress covid_crash
  'IL&FS', 'ILFS', 'NBFC crisis' -> historical_stress ilfs_contagion
  'taper tantrum', 'taper', '2013 crisis' -> historical_stress taper_tantrum_2013
  '2008 crisis' (no preset exists) -> factor_shock MARKET: -20.0
- Questions about tail risk, worst case, loss limit, wipe out, drawdown limit -> reverse_stress
- Questions about rebalancing, optimising, reducing risk, fixing the portfolio -> cvar_rebalance
- Questions about policy limits, risk limits, compliance, within limits -> policy_check
- Questions about portfolio performance, returns, how did I do, profit/loss, best/worst stock, \
which stock is dragging -> portfolio_performance
- When the user asks about a specific stock in their portfolio ('how is X doing', 'what about X', \
'tell me about X in my portfolio', 'should I sell X'): select portfolio_performance AND include \
{'focus_holding': '[TICKER]'} in its params. Extract the company name or ticker from the user \
message and give it as an NSE symbol (add .NS if missing). Examples: 'Zydus Wellness' -> \
'ZYDUSWELL.NS', 'Ratnaveer' -> 'RATNAVEER.NS', 'NSIL' -> 'NSIL.NS'. The focus_holding tells the \
narration step to answer specifically about that stock. End the narration with: 'Note: this is \
historical risk analysis, not investment advice.'
- If the user asks which stocks in a sector (PSU, banking, IT, pharma) are riskiest, best or worst: \
select portfolio_performance with NO focus_holding and include {'sector_question': true} in its \
params. The system has no sector classification data; never guess which holdings belong to a sector.
- If the user asks what their CVaR is, their tail risk, or their 95%/99% loss estimate: select \
cvar_rebalance with {'turnover_limit': 0.0, 'per_name_cap': 1.0, 'confidence_level': 0.95} (use 0.99 \
when they ask for 99%). turnover_limit 0.0 means the portfolio is not changed (and per_name_cap 1.0 \
removes the position cap that would otherwise force trades), so this only reports the current CVaR. \
Do NOT select current_risk for CVaR questions.
- The 'window' parameter must always be an integer (number of days), never a date string or date \
range. Valid: 252. Invalid: '2020-03-01:2020-04-01'.
- If the user asks a follow-up that references a prior result ('now reduce it', 'what about a \
bigger crash'), infer the experiment from context -- do not ask for clarification.

SCOPE BOUNDARIES. In scope (run a tool): portfolio risk, volatility, factor exposure; macro shock \
impact; historical scenario impact; CVaR rebalancing; policy checks; portfolio performance and \
returns; a specific holding's performance (portfolio_performance + focus_holding).
Out of scope (use decline; its reason is shown to the user, so write exactly the message below, \
substituting the company name for X):
- Valuation ('Is X overvalued?', 'What is the P/E of X?', 'Is X a good buy?'): 'I analyse \
portfolio risk using price data, not fundamental valuation. For P/E ratios and valuation, check \
Screener.in or Tickertape.'
- Stock price predictions ('Will X reach Y?', 'Where will X be in 6 months?'): 'I cannot forecast \
stock prices. I can show you how X has contributed to your portfolio risk historically, or what a \
broader market move would do to you.'
- Individual stock price impact ('What if X goes up/down by Y%?'): the ROUTING RULE 0 message.
- News and current events ('Why did X fall today?', 'What happened to X?'): 'I don't have \
real-time news. I can show you X's historical return in your portfolio or run a stress test.'
- Anything with no connection to finance or investing (weather, sports, cooking, entertainment): \
'I can only help with portfolio risk analysis.'
- When uncertain between two tools, pick the one that gives more information. Never decline a \
question about portfolio risk, returns, macro factors or a named historical event.

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
    /// The planner chose `decline`: the message is out of scope (an
    /// individual-stock price question, valuation, prediction, news, or
    /// non-finance). The payload is the redirect text to show the user. Not
    /// an error from the caller's point of view -- `server` returns it as a
    /// 200 with `is_redirect: true`.
    #[error("{0}")]
    Declined(String),
}

/// Runs the planning call and parses its response into `Vec<ToolPlan>`,
/// falling back to a single `current_risk` plan (recording the raw
/// response either way) if the response isn't valid JSON, isn't an array,
/// or is empty. Returns `(plans, raw_response_text)` -- except when the
/// plan is a single `decline` entry (see `PLANNING_SYSTEM_PROMPT`), which
/// returns `Err(OrchestratorError::Declined)` instead: an off-topic
/// message runs no tool at all, rather than falling back to one.
pub async fn plan_tools<C: GeminiClient>(
    client: &C,
    user_message: &str,
    conversation_history: &[ConversationTurn],
) -> Result<(Vec<ToolPlan>, String), OrchestratorError> {
    plan_tools_for(client, user_message, conversation_history, None).await
}

/// The holdings block prepended to the planning prompt: lets the planner
/// tell "a stock in your portfolio" from "a stock you don't hold".
fn holdings_block(portfolio: &Portfolio) -> String {
    let list: Vec<String> =
        portfolio.holdings.iter().map(|h| format!("{} ({:.1}%)", h.ticker, h.weight * 100.0)).collect();
    format!(
        "PORTFOLIO HOLDINGS: {}\n\
         If the user asks about a specific stock that is NOT in this list, use the decline tool with this reason \
         (substituting the stock's name): '[STOCK] is not in your current portfolio.' Match company names to \
         these tickers loosely ('Zydus Wellness' is ZYDUSWELL, 'Ratnaveer Precision' is RATNAVEER, 'Bank of \
         Maharashtra' is MAHABANK) before concluding a stock is missing. This applies only to questions about one \
         named company, never to macro factors, the market, or the portfolio as a whole.\n\n",
        list.join(", ")
    )
}

/// `plan_tools` with the portfolio's holdings shown to the planner (see
/// `holdings_block`); `None` plans without them.
pub async fn plan_tools_for<C: GeminiClient>(
    client: &C,
    user_message: &str,
    conversation_history: &[ConversationTurn],
    portfolio: Option<&Portfolio>,
) -> Result<(Vec<ToolPlan>, String), OrchestratorError> {
    let system_prompt = match portfolio {
        Some(p) => format!("{}{}", holdings_block(p), PLANNING_SYSTEM_PROMPT),
        None => PLANNING_SYSTEM_PROMPT.to_string(),
    };
    let mut contents: Vec<Content> = conversation_history.iter().map(turn_to_content).collect();
    contents.push(Content {
        role: Some("user".to_string()),
        parts: vec![Part::text(user_message)],
    });
    let request = GeminiRequest {
        contents,
        system_instruction: Some(Content {
            role: None,
            parts: vec![Part::text(system_prompt)],
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
            Err(OrchestratorError::Declined(plans[0].reason.clone()))
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

/// Maps the planner's `focus_holding` guess (a company name or symbol the
/// model extracted from the user's message, e.g. `RATNAVEERP.NS`) onto an
/// actual holding of `portfolio`: exact symbol match first, then a prefix
/// match in either direction on the bare symbol (at least 4 characters, so
/// `RATNAVEERP` still finds `RATNAVEER.NS`). `None` if nothing matches --
/// the caller then drops the focus rather than narrate about a stock the
/// user doesn't hold.
pub fn resolve_focus_holding(raw: &str, portfolio: &Portfolio) -> Option<String> {
    let bare = |t: &str| -> String {
        let up = t.trim().to_uppercase();
        up.strip_suffix(".NS").or_else(|| up.strip_suffix(".BO")).unwrap_or(&up).to_string()
    };
    let want = bare(raw);
    if want.is_empty() {
        return None;
    }
    let held: Vec<(&String, String)> = portfolio.holdings.iter().map(|h| (&h.ticker, bare(&h.ticker))).collect();
    if let Some((t, _)) = held.iter().find(|(_, b)| *b == want) {
        return Some((*t).clone());
    }
    let prefix = |a: &str, b: &str| a.len() >= 4 && b.starts_with(a);
    let mut near = held.iter().filter(|(_, b)| prefix(&want, b) || prefix(b, &want));
    match (near.next(), near.next()) {
        (Some((t, _)), None) => Some((*t).clone()),
        _ => None,
    }
}

/// Phrases that unambiguously ask about intraday movement.
const INTRADAY_PHRASES: &[&str] = &["today", "this morning", "this afternoon", "intraday"];
/// Price-movement words (whole words: a prefix like "ris" would match
/// "risk"). "right now" only counts as an intraday question next to one of
/// these ("what is my risk right now" is just "currently").
const MOVEMENT_WORDS: &[&str] = &[
    "fell", "fall", "falls", "falling", "fallen", "drop", "dropped", "dropping", "crash", "crashed", "crashing",
    "surge", "surged", "surging", "rose", "rise", "rises", "rising", "up", "down",
];

fn is_intraday_question(message: &str) -> bool {
    let lower = message.to_lowercase();
    if INTRADAY_PHRASES.iter().any(|p| lower.contains(p)) {
        return true;
    }
    lower.contains("right now")
        && lower
            .split(|c: char| !c.is_alphanumeric())
            .any(|w| MOVEMENT_WORDS.contains(&w))
}

/// Marks the first plan `realtime_caveat: true` when the message asks about
/// an intraday move -- the data is daily history, so the narration must say
/// it can't speak to today's prices. Returns whether it did.
pub fn apply_realtime_caveat(plans: &mut [ToolPlan], user_message: &str) -> bool {
    if !is_intraday_question(user_message) {
        return false;
    }
    let Some(plan) = plans.first_mut() else { return false };
    if let Value::Object(map) = &mut plan.params {
        map.insert("realtime_caveat".to_string(), Value::Bool(true));
    } else if plan.params.is_null() {
        plan.params = serde_json::json!({ "realtime_caveat": true });
    } else {
        return false;
    }
    true
}

/// Whether the planner flagged this as a "which PSU/banking/IT stock..."
/// question (see the prompt's sector rule).
pub fn is_sector_question(plans: &[ToolPlan]) -> bool {
    plans.iter().any(|p| p.params.get("sector_question").and_then(Value::as_bool) == Some(true))
}

/// The first plan's `focus_holding` (if any), resolved against `portfolio`
/// and written back into that plan's params (or removed when it matches no
/// holding). Returns the resolved ticker.
pub fn apply_focus_holding(plans: &mut [ToolPlan], portfolio: &Portfolio) -> Option<String> {
    let mut resolved = None;
    for plan in plans.iter_mut() {
        let Some(raw) = plan.params.get("focus_holding").and_then(Value::as_str).map(str::to_string) else {
            continue;
        };
        let found = resolve_focus_holding(&raw, portfolio);
        if let Value::Object(map) = &mut plan.params {
            match &found {
                Some(t) => map.insert("focus_holding".to_string(), Value::String(t.clone())),
                None => map.remove("focus_holding"),
            };
        }
        if resolved.is_none() {
            resolved = found;
        }
    }
    resolved
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

/// `build_experiment`, but a planner response whose params don't deserialize
/// (e.g. `window: "2020-03-01:2020-04-01"`) never fails the request: the tool
/// is retried with empty params, i.e. every default. Only `scenario_id` is
/// kept for `historical_stress`, since without it the run would silently be
/// a zero-loss no-op. If even the defaults can't build (a tool with required
/// params, such as `factor_shock`'s shocks), the original error stands.
pub fn build_experiment_or_defaults(
    tool: RiskTool,
    tool_name: &str,
    params: &Value,
    portfolio: &Portfolio,
    ctx: &ExperimentContext,
    baseline_override: Option<String>,
) -> Result<Experiment, OrchestratorError> {
    match build_experiment(tool, params, portfolio, ctx, baseline_override.clone()) {
        Ok(e) => Ok(e),
        Err(first) => {
            tracing::warn!("Tool params deserialization failed for {tool_name}, retrying with defaults: {first}");
            let mut defaults = serde_json::Map::new();
            if tool == RiskTool::HistoricalStress {
                if let Some(id) = params.get("scenario_id") {
                    defaults.insert("scenario_id".to_string(), id.clone());
                }
            }
            build_experiment(tool, &Value::Object(defaults), portfolio, ctx, baseline_override).map_err(|_| first)
        }
    }
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

/// One tool's failure: a structured `compute::ComputeError` when the
/// failure came from `dispatch::run_experiment`, or a plain message for
/// anything else (an unrecognised tool name, bad params). Kept distinct
/// from a plain `String` (unlike `execution_trace::ToolResult::error`,
/// which always renders to one via `Display`) so `pipeline::PipelineError::
/// AllToolsFailed` can reclassify a single compute failure into its proper
/// HTTP status (see `server::backend`'s `From<ComputeError>` impl) instead
/// of collapsing every tool failure into a generic 500.
#[derive(Debug)]
pub enum ToolError {
    Compute(compute::ComputeError),
    Other(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::Compute(e) => write!(f, "{e}"),
            ToolError::Other(s) => write!(f, "{s}"),
        }
    }
}

/// One tool's outcome: either its `EvidenceTrace` or the error it failed
/// with.
pub enum ToolOutcome {
    Trace(Box<compute::trace::EvidenceTrace>),
    Error(ToolError),
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
        return ToolOutcome::Error(ToolError::Other(format!("unknown tool {:?}", plan.tool)));
    };
    let experiment = match build_experiment_or_defaults(tool, &plan.tool, &plan.params, portfolio, ctx, baseline_override) {
        Ok(e) => e,
        Err(e) => return ToolOutcome::Error(ToolError::Other(e.to_string())),
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
        Err(e) => ToolOutcome::Error(ToolError::Compute(e)),
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
) -> (
    Vec<compute::trace::EvidenceTrace>,
    Vec<crate::execution_trace::ToolResult>,
    Vec<ToolError>,
) {
    let mut traces = Vec::new();
    let mut results = Vec::new();
    let mut errors = Vec::new();
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
                    error: Some(error.to_string()),
                });
                errors.push(error);
            }
        }
    }

    (traces, results, errors)
}
