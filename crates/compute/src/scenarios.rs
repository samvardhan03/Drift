//! Historical scenario presets: fixed shock sets for well-known market
//! events, for direct use as `FactorShockInput::shocks_pct`. These are
//! static reference data (not fit from live data), exposed read-only via
//! `all_scenarios()` and the server's `GET /scenarios`.
//!
//! Also: `ShockContext`/`compute_shock_context`, which answer a different
//! question -- not "what shock set corresponds to a known historical
//! event" but "how extreme, empirically, is *this* MARKET shock (from any
//! `FactorShock` run, named scenario or caller-specified) relative to the
//! market's own historical return distribution."

use std::collections::BTreeMap;

use chrono::NaiveDate;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct HistoricalScenario {
    pub id: &'static str,
    pub name: &'static str,
    pub date_range: &'static str,
    pub description: &'static str,
    /// Simple % returns (not log), keyed by factor name (see
    /// `data::FACTOR_NAMES`).
    pub shocks_pct: BTreeMap<&'static str, f64>,
    pub propagate: bool,
}

fn shocks(pairs: &[(&'static str, f64)]) -> BTreeMap<&'static str, f64> {
    pairs.iter().copied().collect()
}

fn scenarios() -> Vec<HistoricalScenario> {
    vec![
        HistoricalScenario {
            id: "covid_crash",
            name: "COVID Crash (Mar 2020)",
            date_range: "Feb 19 \u{2013} Mar 23, 2020",
            description: "Nifty fell 38% in 33 days; crude collapsed on the OPEC+ breakdown; INR hit 76.",
            shocks_pct: shocks(&[
                ("MARKET", -38.0),
                ("BRENT", -55.0),
                ("USDINR", 8.5),
                ("GOLD_USD", 3.0),
                ("RATES_PROXY", -6.0),
            ]),
            propagate: false,
        },
        HistoricalScenario {
            id: "ilfs_contagion",
            name: "IL&FS Contagion (Sep\u{2013}Oct 2018)",
            date_range: "Sep 21 \u{2013} Oct 26, 2018",
            description: "IL&FS default triggered NBFC liquidity freeze; Nifty fell 15%; INR hit 74 on oil+EM selloff.",
            shocks_pct: shocks(&[
                ("MARKET", -15.0),
                ("USDINR", 7.0),
                ("BRENT", 15.0),
                ("GOLD_USD", 2.5),
                ("RATES_PROXY", 4.0),
            ]),
            propagate: false,
        },
        HistoricalScenario {
            id: "taper_tantrum_2013",
            name: "Taper Tantrum (May\u{2013}Aug 2013)",
            date_range: "May 22 \u{2013} Aug 28, 2013",
            description: "Fed taper signal sent INR to 68, Nifty fell 12%, gold sold off as dollar surged.",
            shocks_pct: shocks(&[
                ("MARKET", -12.0),
                ("USDINR", 18.0),
                ("GOLD_USD", -18.0),
                ("BRENT", -5.0),
                ("RATES_PROXY", 5.0),
            ]),
            propagate: false,
        },
    ]
}

/// The fixed set of historical scenarios, built once and cached.
pub fn all_scenarios() -> &'static [HistoricalScenario] {
    use std::sync::OnceLock;
    static SCENARIOS: OnceLock<Vec<HistoricalScenario>> = OnceLock::new();
    SCENARIOS.get_or_init(scenarios)
}

/// Empirical/statistical context for a `FactorShock` run's MARKET shock,
/// relative to the market's own historical return distribution over the
/// fitted window. Only ever computed for the MARKET factor (see
/// `compute_shock_context`'s doc) -- other factors don't get this
/// treatment, per spec.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ShockContext {
    /// Empirical percentile (0-100) of the shock in a normal distribution
    /// fit to the historical MARKET log returns, e.g. a -20% shock might
    /// be the 99.2nd percentile loss.
    pub market_shock_percentile: f64,
    /// How many standard deviations the shock is from the historical mean.
    pub market_shock_sigma: f64,
    /// How many historical observations in the fitted window had
    /// `|return| >= |shock|`.
    pub historical_occurrences: u32,
    /// ISO-8601 date of the most recent such observation, `None` if there
    /// were none.
    pub last_occurrence_date: Option<String>,
    /// `historical_occurrences / years_in_window` -- "this has happened
    /// ~X times per year historically."
    pub annualized_probability: f64,
    /// "extremely rare (>3\u{3c3})" | "rare (2-3\u{3c3})" |
    /// "uncommon (1-2\u{3c3})" | "within normal range (<1\u{3c3})".
    pub context_label: String,
}

/// Abramowitz & Stegun 7.1.26 rational approximation of the error
/// function, max absolute error ~1.5e-7 -- plenty for a statistical
/// context label, and avoids pulling in a stats crate for one call site.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;
    let t = 1.0 / (1.0 + P * x);
    let poly = ((((A5 * t + A4) * t) + A3) * t + A2) * t + A1;
    let y = 1.0 - poly * t * (-x * x).exp();
    sign * y
}

/// CDF of `Normal(mean, std^2)` at `x`.
fn normal_cdf(x: f64, mean: f64, std: f64) -> f64 {
    let z = (x - mean) / (std.max(1e-12) * std::f64::consts::SQRT_2);
    0.5 * (1.0 + erf(z))
}

/// Builds `ShockContext` for a MARKET log-return shock of `shock_log`
/// (whether caller-specified or conditionally implied -- see
/// `experiments::run_factor_shock`), against the historical MARKET
/// log-return series `market_log_returns` (the fitted window, one entry
/// per trading day) and its paired `dates` (same length, the trailing date
/// of each return -- i.e. `MarketData.dates[1..]`, not the full calendar).
///
/// Fits a normal distribution (mean, std) to `market_log_returns` and
/// reads the shock's percentile/sigma off that fit; counts empirical
/// occurrences and the annualized rate directly from the data (not the
/// normal-distribution fit -- real market returns are fat-tailed, so the
/// empirical count is the more honest "how often has this actually
/// happened" number, while the percentile/sigma give an interpretable
/// "how extreme is this, in standard statistical terms" framing).
pub fn compute_shock_context(market_log_returns: &[f64], dates: &[NaiveDate], shock_log: f64) -> ShockContext {
    let n = market_log_returns.len().max(1);
    let mean = market_log_returns.iter().sum::<f64>() / n as f64;
    let variance = market_log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.max(1e-12).sqrt();

    let sigma = (shock_log - mean) / std;
    let percentile = normal_cdf(shock_log, mean, std) * 100.0;

    let threshold = shock_log.abs();
    let mut historical_occurrences = 0u32;
    let mut last_occurrence_date: Option<NaiveDate> = None;
    for (i, &r) in market_log_returns.iter().enumerate() {
        if r.abs() >= threshold {
            historical_occurrences += 1;
            if let Some(&d) = dates.get(i) {
                if last_occurrence_date.is_none_or(|cur| d > cur) {
                    last_occurrence_date = Some(d);
                }
            }
        }
    }

    let years_in_window = market_log_returns.len() as f64 / 252.0;
    let annualized_probability =
        if years_in_window > 0.0 { historical_occurrences as f64 / years_in_window } else { 0.0 };

    let abs_sigma = sigma.abs();
    let context_label = if abs_sigma >= 3.0 {
        "extremely rare (>3\u{3c3})"
    } else if abs_sigma >= 2.0 {
        "rare (2-3\u{3c3})"
    } else if abs_sigma >= 1.0 {
        "uncommon (1-2\u{3c3})"
    } else {
        "within normal range (<1\u{3c3})"
    }
    .to_string();

    ShockContext {
        market_shock_percentile: percentile,
        market_shock_sigma: sigma,
        historical_occurrences,
        last_occurrence_date: last_occurrence_date.map(|d| d.format("%Y-%m-%d").to_string()),
        annualized_probability,
        context_label,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::FACTOR_NAMES;

    #[test]
    fn all_scenarios_returns_exactly_three() {
        assert_eq!(all_scenarios().len(), 3);
    }

    #[test]
    fn every_shock_key_is_a_valid_factor_name() {
        for scenario in all_scenarios() {
            for key in scenario.shocks_pct.keys() {
                assert!(
                    FACTOR_NAMES.contains(key),
                    "scenario {} has unknown factor key {key}",
                    scenario.id
                );
            }
        }
    }

    fn synthetic_market_returns(n: usize, std: f64, seed: u64) -> (Vec<f64>, Vec<NaiveDate>) {
        struct Rng(u64);
        impl Rng {
            fn next_signed(&mut self) -> f64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                let unit = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
                unit * 2.0 - 1.0
            }
        }
        let mut rng = Rng(seed.max(1));
        let returns: Vec<f64> = (0..n).map(|_| rng.next_signed() * std).collect();
        let start = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let dates: Vec<NaiveDate> = (0..n).map(|i| start + chrono::Duration::days(i as i64)).collect();
        (returns, dates)
    }

    #[test]
    fn a_zero_sigma_shock_has_percentile_near_50() {
        let (returns, dates) = synthetic_market_returns(500, 0.01, 1);
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let context = compute_shock_context(&returns, &dates, mean);
        assert!(
            (context.market_shock_percentile - 50.0).abs() < 1.0,
            "percentile {} not close to 50",
            context.market_shock_percentile
        );
    }

    #[test]
    fn a_plus_3_sigma_shock_is_labelled_extremely_rare() {
        let (returns, dates) = synthetic_market_returns(500, 0.01, 2);
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std = variance.sqrt();
        let shock = mean + 3.0 * std;
        let context = compute_shock_context(&returns, &dates, shock);
        assert_eq!(context.context_label, "extremely rare (>3\u{3c3})");
    }

    #[test]
    fn historical_occurrences_is_non_negative_and_annualized_probability_is_non_negative() {
        let (returns, dates) = synthetic_market_returns(500, 0.01, 3);
        let context = compute_shock_context(&returns, &dates, 0.05);
        assert!(context.annualized_probability >= 0.0);
        // u32 is already non-negative; this just documents the invariant.
        let _: u32 = context.historical_occurrences;
    }
}
