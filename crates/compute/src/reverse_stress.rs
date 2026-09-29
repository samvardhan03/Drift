//! ReverseStress: solves for the minimum-severity factor shock that
//! breaches a specified portfolio loss threshold. Inverse of `FactorShock`
//! (which asks "what happens under shock X"): this asks "what's the
//! smallest shock that causes at least L rupees of loss."
//!
//! The exact problem (minimise `s^T F^-1 s` subject to the *true* nonlinear
//! `P&L(s) <= -L` and per-factor box bounds) is a nonconvex NLP. This
//! solves it in two stages, per the spec:
//!
//! 1. **KKT closed form** on the *linearised* problem (`P&L(s) ~= p^T s`,
//!    exact for the quadratic-objective/linear-constraint QP that results):
//!    `s* = -L * (F p) / (p^T F p)`, clipped to box bounds.
//! 2. **Projected gradient descent** on the true quadratic objective,
//!    refining that starting point. "Projection" here does two things each
//!    step: clip to box bounds, then rescale the (fixed-direction) point by
//!    the smallest factor that makes the *true* nonlinear `P&L` breach `-L`
//!    exactly (a 1-D bisection along the ray from the origin -- see
//!    `project_to_feasible`). This is not a general nonlinear-constraint
//!    projection, but it is exact for this problem's actual geometry
//!    (P&L is monotonic in shock magnitude along any fixed "bad" direction),
//!    and it's what keeps every iterate exactly feasible rather than only
//!    approximately so.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::FACTOR_NAMES;
use crate::error::{ComputeError, Result};
use crate::experiments::{log_to_simple, simple_to_log, Portfolio};
use crate::format::format_inr;
use crate::model::{FactorModel, Frequency};
use crate::trace::{DataWindow, EvidenceTrace, InvariantCheck, ModelParams};

const MAX_GRADIENT_ITERS: u32 = 200;
const GRADIENT_NORM_TOLERANCE: f64 = 1e-6;
const OBJECTIVE_CHANGE_TOLERANCE: f64 = 1e-8;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReverseStressInput {
    /// Positive INR loss magnitude that defines the breach: `P&L <=
    /// -loss_threshold_inr` is a breach.
    pub loss_threshold_inr: f64,
    /// Per-factor `(lower, upper)` bounds on the shock, in simple %
    /// (e.g. `(-50.0, 0.0)`). Omit (or `null`) for
    /// `default_factor_bounds()`.
    #[serde(default)]
    pub factor_bounds: Option<BTreeMap<String, (f64, f64)>>,
    #[serde(default)]
    pub window: Option<usize>,
    #[serde(default)]
    pub frequency: Option<String>,
}

/// Plausible historical shock ranges for Indian markets, used when the
/// caller doesn't specify `factor_bounds` (or leaves individual factors
/// out of a partial map -- each factor is resolved independently).
pub fn default_factor_bounds() -> BTreeMap<String, (f64, f64)> {
    [
        ("MARKET", (-40.0, 0.0)),
        ("USDINR", (-5.0, 20.0)),
        ("BRENT", (-60.0, 100.0)),
        ("GOLD_USD", (-20.0, 40.0)),
        ("RATES_PROXY", (-10.0, 10.0)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ReverseStressOutputs {
    pub shock_vector: BTreeMap<String, f64>,
    pub shock_vector_log: BTreeMap<String, f64>,
    pub mahalanobis_severity: f64,
    pub severity_label: String,
    pub portfolio_pnl_inr: f64,
    pub loss_threshold_inr: f64,
    pub holding_pnl: BTreeMap<String, f64>,
    /// Attribution of `P&L`'s log-space equivalent (`Sum_i value_i *
    /// beta_ik * s_k`) by factor -- exact in log-return space (same
    /// convention `FactorShock::factor_attribution_log_inr` uses), so it
    /// doesn't sum exactly to `portfolio_pnl_inr` once the `exp(.) - 1`
    /// conversion is applied. See the recorded invariant for the residual.
    pub factor_attribution: BTreeMap<String, f64>,
    /// Top 3 holdings by loss magnitude, most negative first.
    pub most_vulnerable_holdings: Vec<String>,
    pub solver_status: String,
    pub n_iterations: u32,
    pub gradient_norm_final: f64,
    /// `abs(nonlinear P&L - linear P&L)` at the solution.
    pub linearisation_error_inr: f64,
    pub factor_bounds_used: BTreeMap<String, (f64, f64)>,
}

/// "within 1σ" | "1–2σ" | "2–3σ" | ">3σ", bucketed on the *unsquared*
/// Mahalanobis distance (`sqrt(s^T F^-1 s)`) so the boundaries at 1/2/3
/// mean what they say (a squared quadratic form would need boundaries at
/// 1/4/9 for the same sigma levels).
pub fn severity_label(mahalanobis_severity: f64) -> String {
    if mahalanobis_severity < 1.0 {
        "within 1\u{3c3}".to_string()
    } else if mahalanobis_severity < 2.0 {
        "1\u{2013}2\u{3c3}".to_string()
    } else if mahalanobis_severity < 3.0 {
        "2\u{2013}3\u{3c3}".to_string()
    } else {
        ">3\u{3c3}".to_string()
    }
}

/// The analytical KKT solution to `minimise s^T F^-1 s subject to p^T s =
/// -loss_threshold` (i.e. ignoring box bounds): `s* = -L * (F p) / (p^T F
/// p)`. Exposed (not just inlined) so it can be unit-tested directly
/// against a hand-built `F`/`p` -- see `reverse_stress_tests.rs`'s
/// single-factor case.
pub fn kkt_shock(f: &DMatrix<f64>, p: &DVector<f64>, loss_threshold: f64) -> DVector<f64> {
    let f_p = f * p;
    let denom = p.dot(&f_p);
    &f_p * (-loss_threshold / denom)
}

/// `sqrt(s^T F^-1 s)`. `Err` only if `f` is singular (shouldn't happen for
/// a Ledoit-Wolf-shrunk covariance, which is always PD).
pub fn mahalanobis_severity(s: &DVector<f64>, f: &DMatrix<f64>) -> Result<f64> {
    let f_inv = f
        .clone()
        .try_inverse()
        .ok_or_else(|| ComputeError::Model("factor covariance is singular; cannot compute Mahalanobis severity".to_string()))?;
    let quadratic = (s.transpose() * &f_inv * s)[(0, 0)];
    Ok(quadratic.max(0.0).sqrt())
}

/// Finds the smallest `t >= 0` such that `pnl_fn(t * s) <= -loss_threshold`,
/// by expanding-then-bisecting -- i.e. the point on the ray through `s`
/// closest to the origin (least severe) that still breaches. Assumes
/// `pnl_fn` is monotonically non-increasing in `t` along this ray (true for
/// any "bad" direction actually produced by this module's own solver;
/// not enforced for an arbitrary `s`).
fn scale_to_breach(s: &DVector<f64>, loss_threshold: f64, pnl_fn: &dyn Fn(&DVector<f64>) -> f64) -> f64 {
    if s.norm() < 1e-12 {
        return 1.0;
    }
    let mut t_lo = 0.0_f64;
    let mut t_hi = 1.0_f64;
    for _ in 0..60 {
        if pnl_fn(&(s * t_hi)) <= -loss_threshold {
            break;
        }
        t_hi *= 1.5;
        if t_hi > 1e6 {
            break;
        }
    }
    for _ in 0..60 {
        let t_mid = 0.5 * (t_lo + t_hi);
        if pnl_fn(&(s * t_mid)) <= -loss_threshold {
            t_hi = t_mid;
        } else {
            t_lo = t_mid;
        }
    }
    t_hi
}

/// Alternates box-bound clipping and `scale_to_breach` until both are
/// (near-)simultaneously satisfied, or `max_rounds` is exhausted. Neither
/// operation alone guarantees the other still holds afterward (clipping
/// can pull a scaled point back out of breach; rescaling can push a
/// clipped point back out of bounds), so this is a small fixed-point loop
/// over the two, not a single pass.
#[allow(clippy::too_many_arguments)]
fn project_to_feasible(
    s: &DVector<f64>,
    lb: &DVector<f64>,
    ub: &DVector<f64>,
    loss_threshold: f64,
    pnl_fn: &dyn Fn(&DVector<f64>) -> f64,
    max_rounds: u32,
) -> DVector<f64> {
    let k = s.len();
    let mut current = DVector::from_fn(k, |i, _| s[i].clamp(lb[i], ub[i]));
    for _ in 0..max_rounds {
        let t = scale_to_breach(&current, loss_threshold, pnl_fn);
        let scaled = &current * t;
        let clipped = DVector::from_fn(k, |i, _| scaled[i].clamp(lb[i], ub[i]));
        let delta = (&clipped - &current).norm();
        current = clipped;
        if delta < 1e-9 {
            break;
        }
    }
    current
}

pub fn run_reverse_stress(
    data_quality: &crate::data::DataQuality,
    data_window: DataWindow,
    model: &FactorModel,
    portfolio: &Portfolio,
    input: &ReverseStressInput,
) -> Result<(ReverseStressOutputs, EvidenceTrace)> {
    if portfolio.tickers() != model.tickers {
        return Err(ComputeError::InvalidInput(
            "portfolio holdings and fitted model tickers must match 1:1, in order".to_string(),
        ));
    }
    if input.loss_threshold_inr <= 0.0 {
        return Err(ComputeError::InvalidInput(format!(
            "loss_threshold_inr must be > 0, got {}",
            input.loss_threshold_inr
        )));
    }
    let loss_threshold = input.loss_threshold_inr;

    let factor_names: Vec<String> = FACTOR_NAMES.iter().map(|s| s.to_string()).collect();
    let k = factor_names.len();

    let defaults = default_factor_bounds();
    let given = input.factor_bounds.clone().unwrap_or_default();
    let mut factor_bounds_used: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for name in &factor_names {
        let (lb, ub) = given.get(name).copied().unwrap_or(defaults[name]);
        if lb > ub {
            return Err(ComputeError::InvalidInput(format!(
                "factor_bounds for {name}: lower bound {lb} exceeds upper bound {ub}"
            )));
        }
        factor_bounds_used.insert(name.clone(), (lb, ub));
    }
    let lb_log: DVector<f64> =
        DVector::from_iterator(k, factor_names.iter().map(|n| simple_to_log(factor_bounds_used[n].0 / 100.0)));
    let ub_log: DVector<f64> =
        DVector::from_iterator(k, factor_names.iter().map(|n| simple_to_log(factor_bounds_used[n].1 / 100.0)));

    let w = portfolio.weights();
    let b = model.beta_matrix();
    let total_value = portfolio.total_value_inr;
    // p_k = Sum_i w_i * beta_ik * total_value_inr: the linear ("small
    // shock") sensitivity of P&L to factor k's log shock.
    let p: DVector<f64> = (b.transpose() * &w) * total_value;
    let f_annual = model.factor_covariance();

    // --- Feasibility: worst-case (max-loss) corner of the box, linearised ---
    let s_worst: DVector<f64> = DVector::from_fn(k, |i, _| if p[i] >= 0.0 { lb_log[i] } else { ub_log[i] });
    let max_loss_linear_pnl = p.dot(&s_worst);
    if max_loss_linear_pnl > -loss_threshold {
        let max_feasible_loss = -max_loss_linear_pnl;
        return Err(ComputeError::ReverseStressInfeasible(format!(
            "The loss threshold {} cannot be breached within the specified factor bounds. \
             Maximum feasible loss is {}.",
            format_inr(loss_threshold),
            format_inr(max_feasible_loss)
        )));
    }

    let nonlinear_pnl = |s: &DVector<f64>| -> f64 {
        portfolio
            .holdings
            .iter()
            .zip(model.fits.iter())
            .map(|(holding, fit)| {
                let value_i = holding.weight * total_value;
                let log_return: f64 = fit.betas.iter().zip(s.iter()).map(|(beta, sk)| beta * sk).sum();
                value_i * (log_return.exp() - 1.0)
            })
            .sum()
    };

    let f_inv = f_annual
        .clone()
        .try_inverse()
        .ok_or_else(|| ComputeError::Model("factor covariance is singular; cannot compute Mahalanobis severity".to_string()))?;
    let objective = |s: &DVector<f64>| -> f64 { (s.transpose() * &f_inv * s)[(0, 0)] };

    // --- Step 1: KKT closed form, then project onto the true feasible set ---
    let s0 = kkt_shock(&f_annual, &p, loss_threshold);
    let mut s = project_to_feasible(&s0, &lb_log, &ub_log, loss_threshold, &nonlinear_pnl, 20);
    let mut current_obj = objective(&s);

    // --- Step 2: projected gradient descent, refining severity ---
    let mut n_iterations = 0u32;
    let mut gradient_norm_final = (&f_inv * &s * 2.0).norm();
    let mut solver_status = "optimal".to_string();

    if gradient_norm_final >= GRADIENT_NORM_TOLERANCE {
        let mut converged = false;
        for iter in 1..=MAX_GRADIENT_ITERS {
            n_iterations = iter;
            let grad = &f_inv * &s * 2.0;
            gradient_norm_final = grad.norm();
            if gradient_norm_final < GRADIENT_NORM_TOLERANCE {
                converged = true;
                break;
            }
            let grad_norm_sq = grad.norm_squared();
            let mut step = 1.0_f64;
            let mut accepted = None;
            for _ in 0..30 {
                let candidate = &s - &grad * step;
                let candidate = project_to_feasible(&candidate, &lb_log, &ub_log, loss_threshold, &nonlinear_pnl, 20);
                let candidate_obj = objective(&candidate);
                // Armijo sufficient-decrease condition (beta=0.5, c=1e-4).
                if candidate_obj <= current_obj - 1e-4 * step * grad_norm_sq {
                    accepted = Some((candidate, candidate_obj));
                    break;
                }
                step *= 0.5;
            }
            let Some((s_new, obj_new)) = accepted else {
                // No Armijo-accepted step found even at the smallest
                // tried step size -- already essentially at a local
                // optimum of the feasible set.
                converged = true;
                break;
            };
            let obj_change = (current_obj - obj_new).abs();
            s = s_new;
            current_obj = obj_new;
            if obj_change < OBJECTIVE_CHANGE_TOLERANCE {
                converged = true;
                break;
            }
        }
        solver_status = if converged { "converged".to_string() } else { "max_iter".to_string() };
    }

    // Final defensive snap: guarantees the hard breach invariant holds even
    // if the loop above exited (Armijo stall, or MAX_GRADIENT_ITERS) at a
    // point that satisfies bounds but not-quite the true nonlinear
    // constraint to full precision.
    s = project_to_feasible(&s, &lb_log, &ub_log, loss_threshold, &nonlinear_pnl, 20);

    let mahalanobis = mahalanobis_severity(&s, &f_annual)?;
    let severity = severity_label(mahalanobis);
    let portfolio_pnl_inr = nonlinear_pnl(&s);
    let linear_pnl = p.dot(&s);
    let linearisation_error_inr = (portfolio_pnl_inr - linear_pnl).abs();

    let mut holding_pnl: BTreeMap<String, f64> = BTreeMap::new();
    let mut factor_attribution: BTreeMap<String, f64> = factor_names.iter().map(|n| (n.clone(), 0.0)).collect();
    for (holding, fit) in portfolio.holdings.iter().zip(model.fits.iter()) {
        let value_i = holding.weight * total_value;
        let log_return: f64 = fit.betas.iter().zip(s.iter()).map(|(beta, sk)| beta * sk).sum();
        let pnl_i = value_i * (log_return.exp() - 1.0);
        holding_pnl.insert(holding.ticker.clone(), pnl_i);
        for (kk, name) in factor_names.iter().enumerate() {
            *factor_attribution.get_mut(name).unwrap() += value_i * fit.betas[kk] * s[kk];
        }
    }

    let mut by_loss: Vec<(&String, &f64)> = holding_pnl.iter().collect();
    by_loss.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
    let most_vulnerable_holdings: Vec<String> = by_loss.iter().take(3).map(|(t, _)| (*t).clone()).collect();

    let shock_vector: BTreeMap<String, f64> =
        factor_names.iter().enumerate().map(|(i, n)| (n.clone(), log_to_simple(s[i]) * 100.0)).collect();
    let shock_vector_log: BTreeMap<String, f64> =
        factor_names.iter().enumerate().map(|(i, n)| (n.clone(), s[i])).collect();

    let breach_ok = portfolio_pnl_inr <= -loss_threshold * 0.999;
    let attribution_sum: f64 = factor_attribution.values().sum();
    let bounds_ok = factor_names.iter().enumerate().all(|(i, n)| {
        let (lb, ub) = factor_bounds_used[n];
        let lb_l = simple_to_log(lb / 100.0);
        let ub_l = simple_to_log(ub / 100.0);
        s[i] >= lb_l - 1e-6 && s[i] <= ub_l + 1e-6
    });

    let invariants = vec![
        InvariantCheck {
            name: "portfolio_pnl_inr <= -loss_threshold_inr * 0.999".to_string(),
            passed: breach_ok,
            tolerance: loss_threshold * 0.001,
            detail: format!("portfolio_pnl_inr={portfolio_pnl_inr:.4} loss_threshold_inr={loss_threshold:.4}"),
        },
        InvariantCheck::approx_eq(
            "sum(factor_attribution) ~= portfolio_pnl_inr (approximate: log-space attribution vs. \
             exp-converted simple P&L, per FactorShock's own convention)",
            attribution_sum,
            portfolio_pnl_inr,
            loss_threshold * 0.01,
        ),
        InvariantCheck {
            name: "shock_vector within factor_bounds_used".to_string(),
            passed: bounds_ok,
            tolerance: 1e-6,
            detail: format!("shock_vector_log={shock_vector_log:?}"),
        },
        InvariantCheck {
            name: "mahalanobis_severity > 0".to_string(),
            passed: mahalanobis > 0.0,
            tolerance: 0.0,
            detail: format!("mahalanobis_severity={mahalanobis}"),
        },
    ];

    let output = ReverseStressOutputs {
        shock_vector,
        shock_vector_log,
        mahalanobis_severity: mahalanobis,
        severity_label: severity,
        portfolio_pnl_inr,
        loss_threshold_inr: loss_threshold,
        holding_pnl,
        factor_attribution,
        most_vulnerable_holdings,
        solver_status,
        n_iterations,
        gradient_norm_final,
        linearisation_error_inr,
        factor_bounds_used,
    };

    let trace = EvidenceTrace {
        id: crate::trace::new_trace_id(),
        experiment: "ReverseStress".to_string(),
        inputs: serde_json::to_value(input)?,
        data_as_of: crate::trace::data_as_of(&data_window),
        data_window,
        data_quality: data_quality.clone(),
        model_params: ModelParams {
            frequency: model.frequency,
            window_periods: model.window,
            factor_names,
            shrinkage_intensity: model.shrinkage_intensity,
            annualization_factor: model.frequency.annualization_factor(),
            regime_state: model.regime_state.clone(),
            regime_fallback_warnings: model.regime_fallback_warnings.clone(),
            cap_source: None,
        },
        outputs: serde_json::json!({ "result": output }),
        invariants,
        engine_version: crate::trace::engine_version(),
        engine_commit: crate::trace::engine_commit(),
        scenario_provenance: None,
        parent_trace_ids: Vec::new(),
        baseline_model_params: None,
        policy_result: None,
    };

    Ok((output, trace))
}

pub(crate) fn resolved_frequency(input: &ReverseStressInput) -> Result<Frequency> {
    Frequency::from_optional_str(input.frequency.as_deref())
}
