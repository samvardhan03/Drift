//! PortfolioPerformance: realized historical performance of the portfolio's
//! own current weights, held constant (rebalanced back to target weights
//! every period -- a "constant-mix" simplification, see the judgment call
//! below) over the trailing window, computed directly from each holding's
//! own historical log returns. Deliberately independent of the factor
//! model: this answers "how has my portfolio actually performed," not a
//! hypothetical shock (FactorShock) or a model-implied risk decomposition
//! (RiskDecomposition) -- neither of those computes realized returns.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::DataQuality;
use crate::error::{ComputeError, Result};
use crate::experiments::{log_to_simple, Portfolio};
use crate::format::{round_2dp, to_pct_2dp};
use crate::model::Frequency;
use crate::regime;
use crate::trace::{DataWindow, EvidenceTrace, InvariantCheck, ModelParams};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PortfolioPerformanceInput {
    pub portfolio: Portfolio,
    #[serde(default)]
    pub frequency: Frequency,
    /// Trailing window in periods at `frequency`. Defaults to
    /// `frequency.default_window()` (252 daily, 156 weekly) when omitted.
    #[serde(default)]
    pub window: Option<usize>,
}

impl PortfolioPerformanceInput {
    pub fn resolved_window(&self) -> usize {
        self.window.unwrap_or_else(|| self.frequency.default_window())
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct PortfolioPerformanceOutput {
    pub start_value_inr: f64,
    pub end_value_inr: f64,
    /// Cumulative simple return over the window: `end_value/start_value - 1`.
    pub total_return: f64,
    pub annualized_return: f64,
    /// Realized volatility of the portfolio's own period returns over the
    /// window -- distinct from `RiskDecomposition`'s factor-model-implied
    /// `portfolio_vol_annualized`, which is a different (model-based, not
    /// historical-sample) quantity.
    pub annualized_vol_realized: f64,
    /// Most negative peak-to-trough decline over the window, as a fraction
    /// (e.g. -0.18 for an 18% drawdown from the running high).
    pub max_drawdown: f64,
    pub best_period_return: f64,
    pub worst_period_return: f64,
    /// The market regime ("Bull"/"Bear"/"Crisis") at the end of the window,
    /// informational only -- this experiment never fits a factor model, so
    /// the regime is read from a standalone HMM fit on the window's MARKET
    /// series (see `run_portfolio_performance`), not from any covariance
    /// used above. `None` only if that fit itself failed (e.g. the window
    /// is narrower than `regime::fit_hmm`'s minimum observation count).
    pub regime_label: Option<String>,
    /// `total_return * 100`, rounded to 2dp -- narration must never state a
    /// raw decimal fraction as a percentage (Gemini otherwise has no
    /// grounded number to cite for "−19.9%" and either invents one or
    /// states the fraction itself, e.g. "-0.199%"). See the equivalent
    /// fields below and `RiskDecompositionOutput`/`ReverseStressOutputs`
    /// for the same fix applied elsewhere.
    pub total_return_pct: f64,
    /// `annualized_return * 100`, rounded to 2dp.
    pub annualized_return_pct: f64,
    /// `annualized_vol_realized * 100`, rounded to 2dp.
    pub annualized_vol_pct: f64,
    /// `max_drawdown * 100`, rounded to 2dp.
    pub max_drawdown_pct: f64,
    /// Per-holding performance over the same window/date range as the
    /// portfolio-level figures above, keyed by ticker -- lets narration
    /// name a specific stock ("what is the most underperforming stock?")
    /// without a separate experiment.
    pub holding_returns: BTreeMap<String, HoldingPerformance>,
    /// Ticker with the highest `total_return_pct` in `holding_returns`.
    pub best_performer: String,
    /// Ticker with the lowest `total_return_pct` in `holding_returns`.
    pub worst_performer: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct HoldingPerformance {
    pub total_return_pct: f64,
    pub annualized_return_pct: f64,
    pub annualized_vol_pct: f64,
    pub max_drawdown_pct: f64,
    /// `weight * total_return_pct`, rounded to 2dp -- this holding's own
    /// share of the portfolio's total return, not its return in isolation
    /// (e.g. a holding that returned 50% at 5% weight contributed 2.5
    /// points of the portfolio's total return, not 50).
    pub contribution_to_portfolio_return_pct: f64,
}

/// `(total_return, annualized_vol, max_drawdown)` for one holding's own
/// log-return series, over the same `[start, start+window)` slice and
/// `ann_factor` as the portfolio-level computation in
/// `run_portfolio_performance` -- same constant-mix-irrelevant math as the
/// portfolio level (a single holding has no rebalancing to simplify away),
/// just applied to one series instead of the weighted sum of all of them.
fn holding_performance(series: &[f64], start: usize, window: usize, ann_factor: f64) -> (f64, f64, f64) {
    let period_returns: Vec<f64> =
        (0..window).map(|s| log_to_simple(series[start + s])).collect();

    let mut cumulative = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut max_drawdown = 0.0_f64;
    for &r in &period_returns {
        cumulative *= 1.0 + r;
        peak = peak.max(cumulative);
        max_drawdown = max_drawdown.min(cumulative / peak - 1.0);
    }
    let total_return = cumulative - 1.0;

    let mean_r: f64 = period_returns.iter().sum::<f64>() / window as f64;
    let variance: f64 = period_returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>()
        / (window.max(2) - 1) as f64;
    let annualized_vol = (variance * ann_factor).sqrt();

    (total_return, annualized_vol, max_drawdown)
}

pub fn run_portfolio_performance(
    data_quality: &DataQuality,
    data_window: DataWindow,
    data: &crate::data::MarketData,
    input: &PortfolioPerformanceInput,
) -> Result<(PortfolioPerformanceOutput, EvidenceTrace)> {
    let tickers = input.portfolio.tickers();
    if tickers.is_empty() {
        return Err(ComputeError::InvalidInput(
            "portfolio must have at least one holding".to_string(),
        ));
    }
    let weights = input.portfolio.weights();
    let window = data_window.window_periods;

    let series: Vec<&Vec<f64>> = tickers
        .iter()
        .map(|t| {
            data.stock_returns
                .get(t)
                .ok_or_else(|| ComputeError::Model(format!("missing return series for {t}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let total_obs = series[0].len();
    if window > total_obs {
        return Err(ComputeError::InvalidInput(format!(
            "requested window ({window}) exceeds available observations ({total_obs})"
        )));
    }
    let start = total_obs - window;

    // Constant-mix simplification: each period's portfolio return is the
    // *current* weights applied to that period's per-holding simple
    // return, as if rebalanced back to target weights every period --
    // not a literal buy-and-hold share-count simulation (only aligned log
    // returns survive past the data layer, not raw per-share prices, so
    // there is no drift-with-no-rebalancing path available without
    // fetching prices separately; see the judgment call in the report).
    let mut period_returns: Vec<f64> = Vec::with_capacity(window);
    for s in 0..window {
        let r: f64 = (0..tickers.len())
            .map(|i| weights[i] * log_to_simple(series[i][start + s]))
            .sum();
        period_returns.push(r);
    }

    let start_value_inr = input.portfolio.total_value_inr;
    let mut cumulative = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut max_drawdown = 0.0_f64;
    let mut best = f64::MIN;
    let mut worst = f64::MAX;
    for &r in &period_returns {
        cumulative *= 1.0 + r;
        peak = peak.max(cumulative);
        let drawdown = cumulative / peak - 1.0;
        max_drawdown = max_drawdown.min(drawdown);
        best = best.max(r);
        worst = worst.min(r);
    }
    let end_value_inr = start_value_inr * cumulative;
    let total_return = cumulative - 1.0;

    let ann_factor = input.frequency.annualization_factor();
    let annualized_return = (1.0 + total_return).powf(ann_factor / window as f64) - 1.0;

    let mean_r: f64 = period_returns.iter().sum::<f64>() / window as f64;
    let variance: f64 = period_returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>()
        / (window.max(2) - 1) as f64;
    let annualized_vol_realized = (variance * ann_factor).sqrt();

    // Informational only (see PortfolioPerformanceOutput::regime_label
    // doc): this experiment never fits a factor model, so the regime read
    // here is a standalone HMM fit on the window's own MARKET series, not
    // derived from anything above. Degrades to `None` rather than failing
    // the whole experiment if the window is too narrow for `fit_hmm`'s
    // minimum observation count.
    // "MARKET" is the factor *label* (FACTOR_NAMES[0]), not the raw ticker
    // `data::MARKET` ("^NSEI") -- `factor_returns` is keyed by the former.
    let nsei_window: Option<&Vec<f64>> = data.factor_returns.get("MARKET");
    let regime_state = nsei_window.and_then(|series| {
        let series_start = series.len().checked_sub(window)?;
        regime::fit_hmm(&series[series_start..]).ok().map(|(_, state)| state)
    });
    let regime_label = regime_state.as_ref().map(|s| s.current_label.to_string());

    let holding_returns: BTreeMap<String, HoldingPerformance> = tickers
        .iter()
        .enumerate()
        .map(|(i, ticker)| {
            let (total_return_i, annualized_vol_i, max_drawdown_i) =
                holding_performance(series[i], start, window, ann_factor);
            let annualized_return_i = (1.0 + total_return_i).powf(ann_factor / window as f64) - 1.0;
            let total_return_pct = to_pct_2dp(total_return_i);
            (
                ticker.clone(),
                HoldingPerformance {
                    total_return_pct,
                    annualized_return_pct: to_pct_2dp(annualized_return_i),
                    annualized_vol_pct: to_pct_2dp(annualized_vol_i),
                    max_drawdown_pct: to_pct_2dp(max_drawdown_i),
                    contribution_to_portfolio_return_pct: round_2dp(weights[i] * total_return_pct),
                },
            )
        })
        .collect();

    let best_performer = tickers
        .iter()
        .max_by(|a, b| {
            holding_returns[*a].total_return_pct.total_cmp(&holding_returns[*b].total_return_pct)
        })
        .expect("tickers is non-empty")
        .clone();
    let worst_performer = tickers
        .iter()
        .min_by(|a, b| {
            holding_returns[*a].total_return_pct.total_cmp(&holding_returns[*b].total_return_pct)
        })
        .expect("tickers is non-empty")
        .clone();

    let output = PortfolioPerformanceOutput {
        start_value_inr,
        end_value_inr,
        total_return,
        annualized_return,
        annualized_vol_realized,
        max_drawdown,
        best_period_return: best,
        worst_period_return: worst,
        regime_label,
        total_return_pct: to_pct_2dp(total_return),
        annualized_return_pct: to_pct_2dp(annualized_return),
        annualized_vol_pct: to_pct_2dp(annualized_vol_realized),
        max_drawdown_pct: to_pct_2dp(max_drawdown),
        holding_returns,
        best_performer,
        worst_performer,
    };

    let invariant = InvariantCheck::approx_eq(
        "end_value_inr == start_value_inr * (1 + total_return)",
        end_value_inr,
        start_value_inr * (1.0 + total_return),
        1e-6 * start_value_inr.max(1.0),
    );

    let model_params = ModelParams {
        frequency: input.frequency,
        window_periods: window,
        // This experiment never fits a factor model -- these fields don't
        // apply (same convention `CvarRebalance` already uses for its own
        // raw-historical-scenario path).
        factor_names: Vec::new(),
        shrinkage_intensity: 0.0,
        annualization_factor: ann_factor,
        regime_state,
        regime_fallback_warnings: Vec::new(),
        cap_source: None,
    };

    let trace = EvidenceTrace {
        id: crate::trace::new_trace_id(),
        experiment: "PortfolioPerformance".to_string(),
        inputs: serde_json::to_value(input)?,
        data_as_of: crate::trace::data_as_of(&data_window),
        data_window,
        data_quality: data_quality.clone(),
        model_params,
        outputs: serde_json::json!({
            "result": output,
            "note": "Constant-mix simplification: each period's portfolio return is the \
                     current weights applied to that period's per-holding simple return, as \
                     if rebalanced back to target weights every period -- not a literal \
                     buy-and-hold share-count simulation.",
        }),
        invariants: vec![invariant],
        engine_version: crate::trace::engine_version(),
        engine_commit: crate::trace::engine_commit(),
        scenario_provenance: None,
        parent_trace_ids: Vec::new(),
        baseline_model_params: None,
        policy_result: None,
    };

    Ok((output, trace))
}
