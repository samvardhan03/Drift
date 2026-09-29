//! CvarRebalance: Rockafellar-Uryasev CVaR-minimizing long-only rebalance
//! over historical scenarios (raw simple returns of the holdings, per the
//! design note — not factor-model-simulated). See the README's "CvarRebalance"
//! section for the full formulation and the rationale for each choice below.

use std::collections::BTreeMap;

use good_lp::{clarabel, variable, Expression, ProblemVariables, ResolutionError, Solution, SolverModel};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use nalgebra::DVector;

use crate::data::DataQuality;
use crate::error::{ComputeError, Result};
use crate::experiments::{log_to_simple, Holding, Portfolio};
use crate::model::{Frequency, ModelConfig};
use crate::policy::{evaluate_policy, PolicyResult, RiskPolicy};
use crate::regime::RegimeState;
use crate::trace::{DataWindow, EvidenceTrace, InvariantCheck, ModelParams};

/// Max re-solve attempts `run_cvar_rebalance`'s remediation loop takes,
/// each one tightening `confidence_level` by 0.01, per spec.
const MAX_REMEDIATION_ITERATIONS: u32 = 5;
const REMEDIATION_CONFIDENCE_STEP: f64 = 0.01;

fn default_confidence_level() -> f64 {
    0.95
}

/// Applied when the caller omits `per_name_cap` entirely (e.g. a `/ask`
/// request where the user never mentioned a cap).
pub const DEFAULT_PER_NAME_CAP: f64 = 0.20;

/// Commission is charged on the traded (turnover) value; 10 bps (0.10%) is
/// a reasonable blended default for Indian equity brokerage + STT + other
/// statutory charges on a delivery trade, but is always caller-overridable.
fn default_commission_bps() -> f64 {
    10.0
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CvarRebalanceInput {
    /// Current holdings (w0) and total portfolio value.
    pub portfolio: Portfolio,
    /// Confidence level beta for CVaR/VaR (e.g. 0.95 = worst 5% tail).
    #[serde(default = "default_confidence_level")]
    pub confidence_level: f64,
    /// Per-name maximum weight (fraction, e.g. 0.20 for 20%). `None` when
    /// the caller (or the parsed NL request) didn't specify one; defaults
    /// to `DEFAULT_PER_NAME_CAP` (0.20), recorded as `model_params.cap_source`
    /// in the trace (`"user-specified"` vs. `"server-default-0.20"`).
    #[serde(default)]
    pub per_name_cap: Option<f64>,
    /// Maximum turnover, Sum_i |w_i - w0_i| (fraction, both buys and sells
    /// counted; e.g. 0.30 allows up to 30% of the portfolio to trade).
    pub turnover_limit: f64,
    /// Commission rate in basis points, charged on `turnover * total_value_inr`.
    #[serde(default = "default_commission_bps")]
    pub commission_bps: f64,
    #[serde(default)]
    pub frequency: Frequency,
    /// Scenario window in periods; omit for the full available history.
    #[serde(default)]
    pub window: Option<usize>,
    /// If set: after the LP solves, `evaluate_policy` runs on the proposed
    /// (`weights_after`) portfolio. If breaches remain, `confidence_level`
    /// is tightened by `REMEDIATION_CONFIDENCE_STEP` and the LP is re-run,
    /// up to `MAX_REMEDIATION_ITERATIONS` times -- see
    /// `run_cvar_rebalance`'s remediation loop.
    #[serde(default)]
    pub policy: Option<RiskPolicy>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CvarPortfolioStats {
    /// Historical VaR at `confidence_level`: the `tail_scenario_count`-th
    /// worst historical loss (decimal, positive = a loss).
    pub historical_var: f64,
    /// Historical CVaR at `confidence_level`: the mean of the
    /// `tail_scenario_count` worst historical losses (decimal, positive =
    /// a loss). Computed directly from the scenario matrix and a weight
    /// vector, independent of the LP's own reported objective value.
    pub historical_cvar: f64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CvarRebalanceOutput {
    /// "optimal" (LP solved), "infeasible" (pre-solve check or solver
    /// reported infeasible), or "solver_error" (any other non-optimal
    /// solver status).
    pub status: String,
    /// Human-readable explanation when `status != "optimal"`.
    pub diagnostics: Option<String>,
    pub confidence_level: f64,
    pub scenario_count: usize,
    /// `round(scenario_count * (1 - confidence_level))`, clamped to >= 1:
    /// how many worst-case historical scenarios the CVaR average is taken
    /// over.
    pub tail_scenario_count: usize,
    pub weights_before: BTreeMap<String, f64>,
    pub weights_after: Option<BTreeMap<String, f64>>,
    pub stats_before: CvarPortfolioStats,
    pub stats_after: Option<CvarPortfolioStats>,
    /// Sum_i |w_i - w0_i|, only present when `status == "optimal"`.
    pub turnover: Option<f64>,
    pub commission_cost_inr: Option<f64>,
    /// The LP's own optimal objective value (zeta* + (1/k) * Sum u_s*),
    /// reported alongside `stats_after.historical_cvar` (computed
    /// independently from the scenario matrix) so the two can be compared.
    pub lp_objective_cvar: Option<f64>,
    /// Annualized portfolio vol under the current-regime factor covariance
    /// (`sqrt(w' Sigma_regime w)`), for `weights_before`. An informational
    /// cross-check against the historical-scenario CVaR/VaR above, not used
    /// in the LP/feasibility.
    pub regime_portfolio_vol_annualized_before: Option<f64>,
    /// Same as `regime_portfolio_vol_annualized_before`, for `weights_after`
    /// (only when `status == "optimal"`).
    pub regime_portfolio_vol_annualized_after: Option<f64>,
    /// `Some` only when `input.policy` was set. Evaluated against
    /// `weights_before` -- always computable regardless of `status`,
    /// unlike everything else below (which needs an optimal solve).
    pub policy_result_before: Option<PolicyResult>,
    /// `Some` only when `input.policy` was set *and* `status == "optimal"`:
    /// the policy result against the *final* weights (after any
    /// remediation attempts).
    pub policy_result_after: Option<PolicyResult>,
    /// Rule names that were breaching right after the initial solve but
    /// are not breaching in `policy_result_after`.
    pub policy_breaches_resolved: Vec<String>,
    /// Rule names still breaching in `policy_result_after`.
    pub policy_breaches_remaining: Vec<String>,
    /// Number of confidence-level-tightening re-solves attempted (0 if no
    /// policy was attached, or the initial solve already satisfied it).
    pub remediation_iterations: u32,
}

/// Historical VaR/CVaR of the scenario-weighted loss distribution
/// `Loss_s = -Sum_i w_i * R[s][i]`, for the worst `k` scenarios.
fn historical_stats(scenarios: &[Vec<f64>], w: &[f64], k: usize) -> CvarPortfolioStats {
    let mut losses: Vec<f64> = scenarios
        .iter()
        .map(|row| {
            let portfolio_return: f64 = row.iter().zip(w.iter()).map(|(r, wi)| r * wi).sum();
            -portfolio_return
        })
        .collect();
    losses.sort_by(|a, b| b.partial_cmp(a).expect("returns are never NaN"));
    let k = k.min(losses.len()).max(1);
    let historical_var = losses[k - 1];
    let historical_cvar = losses[..k].iter().sum::<f64>() / k as f64;
    CvarPortfolioStats {
        historical_var,
        historical_cvar,
    }
}

pub fn run_cvar_rebalance(
    data_quality: &DataQuality,
    data: &crate::data::MarketData,
    input: &CvarRebalanceInput,
) -> Result<(CvarRebalanceOutput, EvidenceTrace)> {
    let tickers = input.portfolio.tickers();
    if tickers.is_empty() {
        return Err(ComputeError::InvalidInput(
            "portfolio must have at least one holding".to_string(),
        ));
    }
    let w0: Vec<f64> = input.portfolio.holdings.iter().map(|h| h.weight).collect();
    let n = tickers.len();

    let series: Vec<&Vec<f64>> = tickers
        .iter()
        .map(|t| {
            data.stock_returns
                .get(t)
                .ok_or_else(|| ComputeError::Model(format!("missing return series for {t}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let total_obs = series[0].len();
    let window = input.window.unwrap_or(total_obs).min(total_obs).max(1);
    let start = total_obs - window;

    // scenarios[s][i] = simple return of stock i in scenario s.
    let scenarios: Vec<Vec<f64>> = (0..window)
        .map(|s| (0..n).map(|i| log_to_simple(series[i][start + s])).collect())
        .collect();
    let scenario_count = window;

    let beta = input.confidence_level;
    if !(0.0..1.0).contains(&beta) {
        return Err(ComputeError::InvalidInput(format!(
            "confidence_level must be in [0, 1), got {beta}"
        )));
    }
    let k = ((scenario_count as f64) * (1.0 - beta)).round().max(1.0) as usize;

    let stats_before = historical_stats(&scenarios, &w0, k);
    let weights_before: BTreeMap<String, f64> =
        tickers.iter().cloned().zip(w0.iter().copied()).collect();

    // Informational only: always fits a regime-conditional factor model
    // purely to report regime_portfolio_vol_annualized_{before,after} as a
    // parametric cross-check alongside the historical-scenario CVaR/VaR.
    // Never feeds the LP or the feasibility checks below. Uses the same
    // `window` already resolved above (the CVaR scenario window, bounded by
    // available data) rather than a fixed default, so the regime read
    // reflects the same span the CVaR analysis itself runs over. `fit_hmm`
    // needs >= 90 observations (`regime::N_STATES * 30`); on a window
    // narrower than that (or any other fit failure) this degrades to `None`
    // rather than failing the whole experiment, since it's informational
    // only, not load-bearing for the LP/feasibility above.
    let regime_model =
        crate::model::fit_factor_model(data, &tickers, ModelConfig::new(window, input.frequency)).ok();
    let regime_state: Option<RegimeState> = regime_model.as_ref().and_then(|m| m.regime_state.clone());
    let regime_fallback_warnings: Vec<String> =
        regime_model.as_ref().map(|m| m.regime_fallback_warnings.clone()).unwrap_or_default();
    let regime_vol = |weights: &BTreeMap<String, f64>| -> Option<f64> {
        let m = regime_model.as_ref()?;
        // m.factor_covariance_daily (and hence stock_covariance()) is
        // already the *current* regime's F, per fit_factor_model.
        let sigma = m.stock_covariance();
        let w = DVector::from_iterator(m.tickers.len(), m.tickers.iter().map(|t| weights[t]));
        let variance = (w.transpose() * &sigma * &w)[(0, 0)];
        Some(variance.max(0.0).sqrt())
    };
    let regime_portfolio_vol_annualized_before = regime_vol(&weights_before);

    // Builds a `Portfolio` from a `{ticker: weight}` map (as `weights_before`/
    // `weights_after` are), reusing the caller's own `total_value_inr` --
    // used to hand `evaluate_policy` a portfolio at either point in time.
    // Iterates `tickers` (the original portfolio's order, matching
    // `regime_model.tickers`/`beta_matrix()` row order) rather than the
    // `BTreeMap`'s own (alphabetical) order -- `evaluate_policy` needs
    // `portfolio.tickers() == model.tickers` positionally, not just as a
    // set.
    let build_portfolio = |weights: &BTreeMap<String, f64>| -> Portfolio {
        Portfolio {
            holdings: tickers.iter().map(|t| Holding { ticker: t.clone(), weight: weights[t] }).collect(),
            total_value_inr: input.portfolio.total_value_inr,
        }
    };
    // Always computable regardless of whether the LP itself later succeeds
    // (see `CvarRebalanceOutput::policy_result_before`'s doc) -- degrades
    // to `None` only if `regime_model` itself failed to fit (see its own
    // doc above), same as the informational vol check just above.
    let policy_result_before: Option<PolicyResult> = match (&input.policy, &regime_model) {
        (Some(policy), Some(model)) => Some(evaluate_policy(policy, &build_portfolio(&weights_before), model, data)?),
        _ => None,
    };

    let (cap, cap_source) = match input.per_name_cap {
        Some(c) => (c, "user-specified"),
        None => (DEFAULT_PER_NAME_CAP, "server-default-0.20"),
    };

    let data_window = DataWindow {
        frequency: input.frequency,
        window_periods: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    };
    let model_params = ModelParams {
        frequency: input.frequency,
        window_periods: window,
        // CvarRebalance uses raw historical scenarios, not the fitted
        // factor model (see the design note); these fields don't apply.
        factor_names: Vec::new(),
        shrinkage_intensity: 0.0,
        annualization_factor: 1.0,
        regime_state,
        regime_fallback_warnings,
        cap_source: Some(cap_source.to_string()),
    };

    let make_trace = |output: &CvarRebalanceOutput, invariants: Vec<InvariantCheck>| -> Result<EvidenceTrace> {
        Ok(EvidenceTrace {
            id: crate::trace::new_trace_id(),
            experiment: "CvarRebalance".to_string(),
            inputs: serde_json::to_value(input)?,
            data_as_of: crate::trace::data_as_of(&data_window),
            data_window: data_window.clone(),
            data_quality: data_quality.clone(),
            model_params: model_params.clone(),
            outputs: serde_json::json!({
                "result": output,
                "note": "Scenarios are simple returns (exp(log) - 1) of the holdings' own \
                         historical log returns, not factor-model-simulated. model_params' \
                         factor_names/shrinkage_intensity do not apply to this experiment.",
            }),
            invariants,
            engine_version: crate::trace::engine_version(),
            engine_commit: crate::trace::engine_commit(),
            scenario_provenance: None,
            parent_trace_ids: Vec::new(),
            baseline_model_params: None,
            policy_result: None,
        })
    };

    // --- Pre-solve feasibility check (necessary conditions; the LP solve
    // itself remains the authoritative feasibility check) ---
    if cap <= 0.0 || cap > 1.0 {
        return Err(ComputeError::InvalidInput(format!(
            "per_name_cap must be in (0, 1], got {cap}"
        )));
    }
    if cap * (n as f64) < 1.0 - 1e-9 {
        let diagnostics = format!(
            "per_name_cap ({cap}) * n_stocks ({n}) = {:.4} < 1: even at every name's cap, \
             weights cannot sum to 1 under a long-only portfolio.",
            cap * (n as f64)
        );
        let output = infeasible_output(input, scenario_count, k, weights_before.clone(), stats_before.clone(), regime_portfolio_vol_annualized_before, policy_result_before.clone(), diagnostics.clone());
        let trace = make_trace(
            &output,
            vec![InvariantCheck {
                name: "pre-solve: per_name_cap * n_stocks >= 1".to_string(),
                passed: false,
                tolerance: 1e-9,
                detail: diagnostics,
            }],
        )?;
        return Ok((output, trace));
    }
    // Necessary (not sufficient) lower bound: names already over cap must
    // sell down to at least their cap, and that sold capital must be
    // bought back elsewhere to keep Sum w = 1, so turnover is at least
    // twice the total forced-sell amount.
    let min_required_sells: f64 = w0.iter().map(|&wi0| (wi0 - cap).max(0.0)).sum();
    if 2.0 * min_required_sells > input.turnover_limit + 1e-9 {
        let diagnostics = format!(
            "names over per_name_cap ({cap}) require selling at least {min_required_sells:.4} \
             of portfolio value, which must be redeployed elsewhere to keep weights summing to \
             1 -- a minimum turnover of {:.4}, exceeding turnover_limit ({}).",
            2.0 * min_required_sells,
            input.turnover_limit
        );
        let output = infeasible_output(input, scenario_count, k, weights_before.clone(), stats_before.clone(), regime_portfolio_vol_annualized_before, policy_result_before.clone(), diagnostics.clone());
        let trace = make_trace(
            &output,
            vec![InvariantCheck {
                name: "pre-solve: minimum turnover to satisfy caps <= turnover_limit".to_string(),
                passed: false,
                tolerance: 1e-9,
                detail: diagnostics,
            }],
        )?;
        return Ok((output, trace));
    }

    // --- Build and solve the LP ---
    let mut vars = ProblemVariables::new();
    let w: Vec<_> = (0..n).map(|_| vars.add(variable().min(0.0).max(cap))).collect();
    let buy: Vec<_> = (0..n).map(|_| vars.add(variable().min(0.0))).collect();
    let sell: Vec<_> = (0..n).map(|_| vars.add(variable().min(0.0))).collect();
    let zeta = vars.add_variable();
    let u: Vec<_> = (0..scenario_count).map(|_| vars.add(variable().min(0.0))).collect();

    let mut objective = Expression::from(zeta);
    for &uv in &u {
        objective += uv * (1.0 / k as f64);
    }

    let mut model = vars.minimise(objective).using(clarabel);

    let sum_w: Expression = w.iter().map(|&v| Expression::from(v)).sum();
    model = model.with(sum_w.eq(1.0));

    let turnover_expr: Expression = buy.iter().chain(sell.iter()).map(|&v| Expression::from(v)).sum();
    model = model.with(turnover_expr.leq(input.turnover_limit));

    for i in 0..n {
        // w_i - buy_i + sell_i = w0_i  <=>  w_i = w0_i + buy_i - sell_i
        let expr = Expression::from(w[i]) - Expression::from(buy[i]) + Expression::from(sell[i]);
        model = model.with(expr.eq(w0[i]));
    }

    for s in 0..scenario_count {
        // u_s + Sum_i R[s][i] * w_i + zeta >= 0
        let mut expr = Expression::from(u[s]) + Expression::from(zeta);
        for (&wi, &ri) in w.iter().zip(scenarios[s].iter()) {
            expr += wi * ri;
        }
        model = model.with(expr.geq(0.0));
    }

    let solution = match model.solve() {
        Ok(sol) => sol,
        Err(e) => {
            let (status, diagnostics) = classify_resolution_error(&e);
            let output = infeasible_output_with_status(
                input, scenario_count, k, weights_before.clone(), stats_before.clone(), regime_portfolio_vol_annualized_before, policy_result_before.clone(), status, diagnostics.clone(),
            );
            let trace = make_trace(
                &output,
                vec![InvariantCheck {
                    name: "solver status == optimal".to_string(),
                    passed: false,
                    tolerance: 0.0,
                    detail: diagnostics,
                }],
            )?;
            return Ok((output, trace));
        }
    };

    let w_star: Vec<f64> = w.iter().map(|&v| solution.value(v)).collect();
    let zeta_star = solution.value(zeta);
    let u_star: Vec<f64> = u.iter().map(|&v| solution.value(v)).collect();
    let lp_objective_cvar = zeta_star + u_star.iter().sum::<f64>() / k as f64;

    let weights_after: BTreeMap<String, f64> =
        tickers.iter().cloned().zip(w_star.iter().copied()).collect();
    let stats_after = historical_stats(&scenarios, &w_star, k);
    let turnover: f64 = w_star
        .iter()
        .zip(w0.iter())
        .map(|(after, before)| (after - before).abs())
        .sum();
    let commission_cost_inr =
        turnover * input.portfolio.total_value_inr * (input.commission_bps / 10_000.0);

    let sum_w_star: f64 = w_star.iter().sum();
    let tolerance = 1e-6;
    let invariants = vec![
        InvariantCheck::approx_eq("sum(weights_after) == 1", sum_w_star, 1.0, 1e-9),
        InvariantCheck {
            name: "turnover <= turnover_limit + tolerance".to_string(),
            passed: turnover <= input.turnover_limit + 1e-6,
            tolerance: 1e-6,
            detail: format!("turnover={turnover:.6} turnover_limit={}", input.turnover_limit),
        },
        InvariantCheck::approx_eq(
            "LP objective == directly-computed historical CVaR of the solution",
            lp_objective_cvar,
            stats_after.historical_cvar,
            tolerance,
        ),
    ];

    let regime_portfolio_vol_annualized_after = regime_vol(&weights_after);

    // --- Policy-aware remediation ---
    // Only attempted when `input.policy` is set *and* a factor model was
    // available to evaluate it against (see `policy_result_before`'s own
    // doc for why that second condition can fail). Each attempt re-solves
    // the *entire* LP at a tightened `confidence_level` via a recursive
    // call to this same function (with `policy: None`, so that call can't
    // itself recurse) -- reusing the full solve rather than trying to
    // extract a re-runnable "core" out of the large stateful LP-building
    // block above. `final_*` fields below start as the initial solve's
    // own results and are overwritten by whichever remediation attempt
    // last produced an optimal solve (which may still be the initial one,
    // if remediation never ran or every retry failed to solve).
    let mut final_confidence_level = beta;
    let mut final_weights_after = weights_after.clone();
    let mut final_stats_after = stats_after.clone();
    let mut final_turnover = Some(turnover);
    let mut final_commission_cost_inr = Some(commission_cost_inr);
    let mut final_lp_objective_cvar = Some(lp_objective_cvar);
    let mut final_regime_vol_after = regime_portfolio_vol_annualized_after;
    let mut policy_result_after: Option<PolicyResult> = None;
    let mut policy_breaches_resolved: Vec<String> = Vec::new();
    let mut policy_breaches_remaining: Vec<String> = Vec::new();
    let mut remediation_iterations = 0u32;

    if let (Some(policy), Some(model)) = (&input.policy, &regime_model) {
        let mut current_result = evaluate_policy(policy, &build_portfolio(&final_weights_after), model, data)?;
        let initial_breaches: std::collections::BTreeSet<String> =
            current_result.checks.iter().filter(|c| !c.passed).map(|c| c.rule.clone()).collect();

        while !current_result.all_passed && remediation_iterations < MAX_REMEDIATION_ITERATIONS {
            remediation_iterations += 1;
            let candidate_confidence = (final_confidence_level + REMEDIATION_CONFIDENCE_STEP).min(0.999);
            let mut retry_input = input.clone();
            retry_input.confidence_level = candidate_confidence;
            retry_input.policy = None;
            let (retry_output, _) = run_cvar_rebalance(data_quality, data, &retry_input)?;
            if retry_output.status != "optimal" {
                // Can't remediate further at a tighter confidence level;
                // stop and report whatever the last *successful* attempt
                // (or the original solve) achieved.
                break;
            }
            let retry_weights = retry_output.weights_after.clone().expect("optimal status implies weights_after");
            current_result = evaluate_policy(policy, &build_portfolio(&retry_weights), model, data)?;

            final_confidence_level = candidate_confidence;
            final_weights_after = retry_weights;
            final_stats_after = retry_output.stats_after.expect("optimal status implies stats_after");
            final_turnover = retry_output.turnover;
            final_commission_cost_inr = retry_output.commission_cost_inr;
            final_lp_objective_cvar = retry_output.lp_objective_cvar;
            final_regime_vol_after = retry_output.regime_portfolio_vol_annualized_after;
        }

        let final_breaches: std::collections::BTreeSet<String> =
            current_result.checks.iter().filter(|c| !c.passed).map(|c| c.rule.clone()).collect();
        policy_breaches_resolved = initial_breaches.difference(&final_breaches).cloned().collect();
        policy_breaches_remaining = final_breaches.into_iter().collect();
        policy_result_after = Some(current_result);
    }

    let output = CvarRebalanceOutput {
        status: "optimal".to_string(),
        diagnostics: None,
        confidence_level: final_confidence_level,
        scenario_count,
        tail_scenario_count: k,
        weights_before,
        weights_after: Some(final_weights_after),
        stats_before,
        stats_after: Some(final_stats_after),
        turnover: final_turnover,
        commission_cost_inr: final_commission_cost_inr,
        lp_objective_cvar: final_lp_objective_cvar,
        regime_portfolio_vol_annualized_before,
        regime_portfolio_vol_annualized_after: final_regime_vol_after,
        policy_result_before,
        policy_result_after,
        policy_breaches_resolved,
        policy_breaches_remaining,
        remediation_iterations,
    };
    let trace = make_trace(&output, invariants)?;
    Ok((output, trace))
}

fn classify_resolution_error(e: &ResolutionError) -> (String, String) {
    match e {
        ResolutionError::Infeasible => (
            "infeasible".to_string(),
            "solver reported the problem as infeasible".to_string(),
        ),
        other => ("solver_error".to_string(), format!("solver error: {other}")),
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn infeasible_output(
    input: &CvarRebalanceInput,
    scenario_count: usize,
    k: usize,
    weights_before: BTreeMap<String, f64>,
    stats_before: CvarPortfolioStats,
    regime_portfolio_vol_annualized_before: Option<f64>,
    policy_result_before: Option<PolicyResult>,
    diagnostics: String,
) -> CvarRebalanceOutput {
    infeasible_output_with_status(
        input,
        scenario_count,
        k,
        weights_before,
        stats_before,
        regime_portfolio_vol_annualized_before,
        policy_result_before,
        "infeasible".to_string(),
        diagnostics,
    )
}

#[allow(clippy::too_many_arguments)]
fn infeasible_output_with_status(
    input: &CvarRebalanceInput,
    scenario_count: usize,
    k: usize,
    weights_before: BTreeMap<String, f64>,
    stats_before: CvarPortfolioStats,
    regime_portfolio_vol_annualized_before: Option<f64>,
    policy_result_before: Option<PolicyResult>,
    status: String,
    diagnostics: String,
) -> CvarRebalanceOutput {
    CvarRebalanceOutput {
        status,
        diagnostics: Some(diagnostics),
        confidence_level: input.confidence_level,
        scenario_count,
        tail_scenario_count: k,
        weights_before,
        weights_after: None,
        stats_before,
        stats_after: None,
        turnover: None,
        commission_cost_inr: None,
        lp_objective_cvar: None,
        regime_portfolio_vol_annualized_before,
        regime_portfolio_vol_annualized_after: None,
        policy_result_before,
        policy_result_after: None,
        policy_breaches_resolved: Vec::new(),
        policy_breaches_remaining: Vec::new(),
        remediation_iterations: 0,
    }
}
