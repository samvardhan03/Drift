//! Daily portfolio value series for charting, built from data already in
//! memory during every experiment run (aligned returns, weights, and the
//! fitted regime's Viterbi path) and attached to `EvidenceTrace`.

use chrono::NaiveDate;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::MarketData;
use crate::experiments::Portfolio;
use crate::regime::{self, RegimeState};

/// Default number of points in the series (1 year of daily data).
pub const DEFAULT_HISTORY_DAYS: usize = 252;
/// Hard cap on the number of points (5 years of daily data).
pub const MAX_HISTORY_DAYS: usize = 1260;
const REGIME_NAMES: [&str; 3] = ["Bull", "Bear", "Crisis"];

/// All arrays have length `window_days`. Index 0 is the base date (value =
/// `total_value_inr`, return 0); index `t >= 1` applies that date's
/// weighted simple return.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PortfolioHistory {
    /// ISO-8601 dates (`YYYY-MM-DD`), one per period in the window.
    pub dates: Vec<String>,
    /// Portfolio value in INR, compounded from `total_value_inr`.
    pub portfolio_value: Vec<f64>,
    /// Cumulative return from day 0, in percent, 2dp.
    pub portfolio_return_pct: Vec<f64>,
    /// "Bull" | "Bear" | "Crisis" per day (Viterbi-decoded).
    pub regime_sequence: Vec<String>,
    /// Drawdown from running peak, in percent, 2dp, always <= 0.
    pub drawdown_pct: Vec<f64>,
    pub window_days: usize,
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Builds the history over the trailing `window` points of `data`
/// (clamped to `MAX_HISTORY_DAYS`).
///
/// Per-holding log returns are converted to simple returns
/// (`exp(r) - 1`) before weighting. A holding with less history than the
/// window contributes 0 before its first observation (series are
/// right-aligned). `regime` supplies the Viterbi path, right-aligned to
/// the return series; the window is shortened to the path's coverage
/// (`len + 1` points) rather than inventing regimes for uncovered days.
/// Day 0 (no return of its own) takes day 1's regime.
///
/// `None` if there is too little data (< 2 points) or no regime path.
pub fn compute_portfolio_history(
    data: &MarketData,
    portfolio: &Portfolio,
    regime: &RegimeState,
    window: usize,
) -> Option<PortfolioHistory> {
    let path = &regime.viterbi_sequence;
    let n_returns_avail = data.dates.len().checked_sub(1)?;
    let points = window
        .clamp(2, MAX_HISTORY_DAYS)
        .min(n_returns_avail + 1)
        .min(path.len() + 1);
    if points < 2 {
        return None;
    }
    let n_ret = points - 1;
    let dates_slice = &data.dates[data.dates.len() - points..];

    // Per-holding simple returns for the trailing n_ret periods; missing
    // history (short-history holdings) counts as 0.
    let mut port_returns = vec![0.0_f64; n_ret];
    for h in &portfolio.holdings {
        let series = data.stock_returns.get(&h.ticker)?;
        for (i, slot) in port_returns.iter_mut().enumerate() {
            // Index of this period within `series` (right-aligned).
            let from_end = n_ret - i;
            if let Some(idx) = series.len().checked_sub(from_end) {
                *slot += h.weight * (series[idx].exp() - 1.0);
            }
        }
    }

    let base = portfolio.total_value_inr;
    let mut values = Vec::with_capacity(points);
    values.push(base);
    for r in &port_returns {
        values.push(values.last().unwrap() * (1.0 + r));
    }

    let mut peak = f64::NEG_INFINITY;
    let mut drawdown = Vec::with_capacity(points);
    for &v in &values {
        peak = peak.max(v);
        drawdown.push(round2((v / peak - 1.0) * 100.0).min(0.0));
    }
    let returns_pct: Vec<f64> = values.iter().map(|v| round2((v / base - 1.0) * 100.0)).collect();

    let regime_path = &path[path.len() - n_ret..];
    let name = |s: u8| REGIME_NAMES[(s as usize).min(REGIME_NAMES.len() - 1)].to_string();
    let mut regimes = Vec::with_capacity(points);
    regimes.push(name(regime_path[0]));
    regimes.extend(regime_path.iter().map(|&s| name(s)));

    Some(PortfolioHistory {
        dates: dates_slice.iter().map(NaiveDate::to_string).collect(),
        portfolio_value: values,
        portfolio_return_pct: returns_pct,
        regime_sequence: regimes,
        drawdown_pct: drawdown,
        window_days: points,
    })
}

/// The regime to use for `history`: the trace's own fitted state when
/// present, else a standalone HMM fit on the trailing `window` MARKET
/// returns (as `performance` does).
pub fn regime_for_history(existing: Option<&RegimeState>, data: &MarketData, window: usize) -> Option<RegimeState> {
    if let Some(r) = existing {
        return Some(r.clone());
    }
    let series = data.factor_returns.get("MARKET")?;
    let start = series.len().checked_sub(window)?;
    regime::fit_hmm(&series[start..]).ok().map(|(_, s)| s)
}
