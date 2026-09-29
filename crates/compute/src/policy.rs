//! Deterministic policy engine: checks a portfolio (already fit against a
//! `FactorModel`) against configurable risk limits. Every rule is a plain
//! arithmetic comparison against a quantity already computable from the
//! fitted model / historical data -- no optimisation, no LLM, no new data
//! fetch beyond what the caller already has on hand.
//!
//! `evaluate_policy` deliberately does **not** take a `store` handle: the
//! spec's own evaluation rules ("compute CVaR directly from historical
//! scenarios... do not run the LP") never actually need one -- CVaR is
//! computed directly from the same historical return series
//! `CvarRebalance` uses, not from a stored snapshot or a live LP solve. See
//! the README's judgment-call note.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::{DataQuality, MarketData, FACTOR_NAMES};
use crate::error::{ComputeError, Result};
use crate::experiments::{log_to_simple, run_factor_shock, FactorShockInput, Portfolio};
use crate::model::FactorModel;
use crate::trace::{DataWindow, EvidenceTrace, InvariantCheck, ModelParams};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct RiskPolicy {
    /// e.g. `0.18` = 18% annualized.
    #[serde(default)]
    pub max_vol_annualized: Option<f64>,
    /// Historical CVaR at 95% confidence, as a fraction of portfolio value
    /// (e.g. `0.05` = 5%), computed directly from historical scenarios --
    /// never via `CvarRebalance`'s LP.
    #[serde(default)]
    pub max_cvar_95: Option<f64>,
    /// e.g. `0.85` = no single factor's Euler contribution may exceed 85%
    /// of total portfolio vol.
    #[serde(default)]
    pub max_factor_contribution_share: Option<f64>,
    /// e.g. `0.20` = no single holding may exceed 20% of portfolio weight.
    #[serde(default)]
    pub max_position_weight: Option<f64>,
    /// Used only by `CvarRebalance`'s remediation loop (the turnover
    /// budget it may spend re-solving); `PolicyCheck` doesn't consume this
    /// field -- weight changes aren't something a passive check can do
    /// anything about.
    #[serde(default)]
    pub max_turnover: Option<f64>,
    #[serde(default)]
    pub max_loss_under_scenarios: Option<Vec<ScenarioLimit>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ScenarioLimit {
    /// Matches one of `compute::scenarios::all_scenarios()`'s ids.
    pub scenario_id: String,
    /// e.g. `0.15` = max 15% portfolio loss under this scenario.
    pub max_loss_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct PolicyCheck {
    pub rule: String,
    pub limit: f64,
    pub actual: f64,
    pub passed: bool,
    pub breach_magnitude: f64,
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct PolicyResult {
    pub checks: Vec<PolicyCheck>,
    pub all_passed: bool,
    pub breach_count: usize,
    pub most_severe_breach: Option<PolicyCheck>,
}

fn make_check(rule: &str, limit: f64, actual: f64, evidence: String) -> PolicyCheck {
    let passed = actual <= limit;
    PolicyCheck {
        rule: rule.to_string(),
        limit,
        actual,
        passed,
        breach_magnitude: if passed { 0.0 } else { (actual - limit).abs() },
        evidence,
    }
}

fn pct(v: f64) -> String {
    format!("{:.1}%", v * 100.0)
}

/// Portfolio annualized vol under the (already regime-conditional, per
/// `compute`'s always-on regime) fitted `model` -- same formula as
/// `RiskDecomposition`'s `portfolio_vol_annualized`, no refit needed.
pub fn portfolio_vol_annualized(portfolio: &Portfolio, model: &FactorModel) -> f64 {
    let w = portfolio.weights();
    let sigma = model.stock_covariance();
    let variance = (w.transpose() * &sigma * &w)[(0, 0)];
    variance.max(0.0).sqrt()
}

/// `(factor_name, share)` for the single factor with the largest Euler
/// contribution share (`contribution_k / portfolio_vol`); `share` is
/// signed (not absolute value) since a policy limit is about a dominant
/// *positive* risk driver, not a large diversifying (negative) one.
pub fn max_factor_contribution_share(portfolio: &Portfolio, model: &FactorModel) -> (String, f64) {
    let w = portfolio.weights();
    let b = model.beta_matrix();
    let f_annual = model.factor_covariance();
    let x = b.transpose() * &w;
    let fx = &f_annual * &x;
    let vol = portfolio_vol_annualized(portfolio, model);

    let factor_names = FACTOR_NAMES;
    let mut best_name = factor_names[0].to_string();
    let mut best_share = f64::MIN;
    for (i, name) in factor_names.iter().enumerate() {
        let share = if vol > 0.0 { (x[i] * fx[i] / vol) / vol } else { 0.0 };
        if share > best_share {
            best_share = share;
            best_name = name.to_string();
        }
    }
    (best_name, best_share)
}

pub fn max_position_weight(portfolio: &Portfolio) -> f64 {
    portfolio.holdings.iter().map(|h| h.weight).fold(f64::MIN, f64::max)
}

/// Historical CVaR at 95% confidence, computed directly from the
/// portfolio's own simple returns over the *full* available history (same
/// series `CvarRebalance` uses when its own `window` is omitted) -- no LP,
/// no store access.
pub fn portfolio_cvar_95(portfolio: &Portfolio, data: &MarketData) -> Result<f64> {
    let tickers = portfolio.tickers();
    let w: Vec<f64> = portfolio.holdings.iter().map(|h| h.weight).collect();
    let series: Vec<&Vec<f64>> = tickers
        .iter()
        .map(|t| {
            data.stock_returns
                .get(t)
                .ok_or_else(|| ComputeError::Model(format!("missing return series for {t}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let scenario_count = series[0].len();
    let k = ((scenario_count as f64) * 0.05).round().max(1.0) as usize;

    let mut losses: Vec<f64> = (0..scenario_count)
        .map(|s| {
            let portfolio_return: f64 =
                (0..tickers.len()).map(|i| w[i] * log_to_simple(series[i][s])).sum();
            -portfolio_return
        })
        .collect();
    losses.sort_by(|a, b| b.partial_cmp(a).expect("returns are never NaN"));
    let k = k.min(losses.len()).max(1);
    Ok(losses[..k].iter().sum::<f64>() / k as f64)
}

/// Runs a `FactorShock` with `scenario_id`'s fixed shocks (`propagate:
/// false`, per spec) and returns the resulting loss as a fraction of
/// portfolio value (`abs(portfolio_pnl_inr) / total_value_inr`).
pub fn scenario_loss_pct(
    data_quality: &DataQuality,
    data_window: DataWindow,
    portfolio: &Portfolio,
    model: &FactorModel,
    scenario_id: &str,
) -> Result<f64> {
    let scenario = crate::scenarios::all_scenarios()
        .iter()
        .find(|s| s.id == scenario_id)
        .ok_or_else(|| ComputeError::InvalidInput(format!("unknown scenario id {scenario_id:?}")))?;

    let shocks_pct: BTreeMap<String, f64> =
        scenario.shocks_pct.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    let input = FactorShockInput {
        portfolio: portfolio.clone(),
        shocks_pct,
        propagate: false,
        linear_approximation: false,
        frequency: model.frequency,
        window: Some(model.window),
    };
    let (output, _trace) = run_factor_shock(data_quality, data_window, model, &input)?;
    Ok(output.portfolio_pnl_inr.abs() / portfolio.total_value_inr.abs())
}

/// Evaluates every rule `policy` sets (skipping any left `None`) against
/// `portfolio`/`model`/`data`, returning the aggregate `PolicyResult`.
pub fn evaluate_policy(
    policy: &RiskPolicy,
    portfolio: &Portfolio,
    model: &FactorModel,
    data: &MarketData,
) -> Result<PolicyResult> {
    let mut checks = Vec::new();

    if let Some(limit) = policy.max_vol_annualized {
        let vol = portfolio_vol_annualized(portfolio, model);
        checks.push(make_check(
            "max_vol_annualized",
            limit,
            vol,
            format!("Portfolio annualized volatility is {} (limit: {})", pct(vol), pct(limit)),
        ));
    }

    if let Some(limit) = policy.max_cvar_95 {
        let cvar = portfolio_cvar_95(portfolio, data)?;
        checks.push(make_check(
            "max_cvar_95",
            limit,
            cvar,
            format!("Portfolio historical CVaR (95%) is {} (limit: {})", pct(cvar), pct(limit)),
        ));
    }

    if let Some(limit) = policy.max_factor_contribution_share {
        let (factor, share) = max_factor_contribution_share(portfolio, model);
        checks.push(make_check(
            "max_factor_contribution_share",
            limit,
            share,
            format!("{factor} factor contributes {} of vol (limit: {})", pct(share), pct(limit)),
        ));
    }

    if let Some(limit) = policy.max_position_weight {
        let weight = max_position_weight(portfolio);
        let ticker = portfolio
            .holdings
            .iter()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())
            .map(|h| h.ticker.clone())
            .unwrap_or_default();
        checks.push(make_check(
            "max_position_weight",
            limit,
            weight,
            format!("Largest position ({ticker}) is {} of portfolio (limit: {})", pct(weight), pct(limit)),
        ));
    }

    if let Some(scenario_limits) = &policy.max_loss_under_scenarios {
        let window = model.window;
        let data_window = DataWindow {
            frequency: model.frequency,
            window_periods: window,
            start: data.dates[data.dates.len() - window],
            end: *data.dates.last().unwrap(),
        };
        for scenario_limit in scenario_limits {
            let loss_pct = scenario_loss_pct(
                &data.quality,
                data_window.clone(),
                portfolio,
                model,
                &scenario_limit.scenario_id,
            )?;
            checks.push(make_check(
                &format!("max_loss_under_scenarios[{}]", scenario_limit.scenario_id),
                scenario_limit.max_loss_pct,
                loss_pct,
                format!(
                    "Under scenario '{}', portfolio would lose {} (limit: {})",
                    scenario_limit.scenario_id,
                    pct(loss_pct),
                    pct(scenario_limit.max_loss_pct)
                ),
            ));
        }
    }

    let breach_count = checks.iter().filter(|c| !c.passed).count();
    let all_passed = breach_count == 0;
    let most_severe_breach = checks
        .iter()
        .filter(|c| !c.passed)
        .max_by(|a, b| a.breach_magnitude.partial_cmp(&b.breach_magnitude).unwrap())
        .cloned();

    Ok(PolicyResult { checks, all_passed, breach_count, most_severe_breach })
}

// ---------------------------------------------------------------------
// PolicyCheck experiment
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PolicyCheckInput {
    pub policy: RiskPolicy,
    #[serde(default)]
    pub window: Option<usize>,
    #[serde(default)]
    pub frequency: Option<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct PolicyCheckOutputs {
    pub policy_result: PolicyResult,
    pub regime_label: String,
    pub portfolio_vol: f64,
    pub portfolio_cvar_95: f64,
    pub max_position_weight: f64,
    pub max_factor_share: f64,
    /// `scenario_id -> loss_pct`, only for scenarios named in
    /// `policy.max_loss_under_scenarios` (empty if that field is `None`).
    pub scenario_losses: BTreeMap<String, f64>,
}

pub fn run_policy_check(
    data_quality: &DataQuality,
    data_window: DataWindow,
    data: &MarketData,
    model: &FactorModel,
    portfolio: &Portfolio,
    input: &PolicyCheckInput,
) -> Result<(PolicyCheckOutputs, EvidenceTrace)> {
    if portfolio.tickers() != model.tickers {
        return Err(ComputeError::InvalidInput(
            "portfolio holdings and fitted model tickers must match 1:1, in order".to_string(),
        ));
    }

    let policy_result = evaluate_policy(&input.policy, portfolio, model, data)?;

    let regime_label = model
        .regime_state
        .as_ref()
        .map(|r| r.current_label.to_string())
        .ok_or_else(|| ComputeError::InvalidInput("factor model has no regime_state".to_string()))?;
    let portfolio_vol = portfolio_vol_annualized(portfolio, model);
    let cvar_95 = portfolio_cvar_95(portfolio, data)?;
    let (_, max_factor_share) = max_factor_contribution_share(portfolio, model);
    let max_position = max_position_weight(portfolio);

    let mut scenario_losses = BTreeMap::new();
    if let Some(scenario_limits) = &input.policy.max_loss_under_scenarios {
        for scenario_limit in scenario_limits {
            let loss_pct = scenario_loss_pct(
                data_quality,
                data_window.clone(),
                portfolio,
                model,
                &scenario_limit.scenario_id,
            )?;
            scenario_losses.insert(scenario_limit.scenario_id.clone(), loss_pct);
        }
    }

    let output = PolicyCheckOutputs {
        policy_result,
        regime_label,
        portfolio_vol,
        portfolio_cvar_95: cvar_95,
        max_position_weight: max_position,
        max_factor_share,
        scenario_losses,
    };

    let invariants = vec![InvariantCheck {
        name: "breach_count == checks.iter().filter(!passed).count()".to_string(),
        passed: output.policy_result.breach_count == output.policy_result.checks.iter().filter(|c| !c.passed).count(),
        tolerance: 0.0,
        detail: format!("breach_count={}", output.policy_result.breach_count),
    }];

    let trace = EvidenceTrace {
        id: crate::trace::new_trace_id(),
        experiment: "PolicyCheck".to_string(),
        inputs: serde_json::to_value(input)?,
        data_as_of: crate::trace::data_as_of(&data_window),
        data_window,
        data_quality: data_quality.clone(),
        model_params: ModelParams {
            frequency: model.frequency,
            window_periods: model.window,
            factor_names: FACTOR_NAMES.iter().map(|s| s.to_string()).collect(),
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
