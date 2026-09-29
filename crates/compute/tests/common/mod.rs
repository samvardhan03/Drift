//! Shared synthetic-data helpers for the compute test suite. Builds
//! `MarketData` directly (bypassing the network-fetching data layer) so
//! model/experiment tests are deterministic and hermetic.
//!
//! Not every integration test binary uses every helper here (each `mod
//! common;` is compiled per-binary), hence the blanket allow.
#![allow(dead_code)]

use std::collections::BTreeMap;

use chrono::NaiveDate;
use compute::data::{DataQuality, MarketData, SeriesQuality, FACTOR_NAMES};

/// Tiny deterministic PRNG (xorshift64*) so tests don't need a `rand` dep.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    /// Uniform-ish value in [-1, 1].
    pub fn next_signed(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        unit * 2.0 - 1.0
    }
}

fn business_days(n: usize) -> Vec<NaiveDate> {
    let mut dates = Vec::with_capacity(n);
    let mut d = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
    while dates.len() < n {
        // Skip weekends to look calendar-plausible; exact dates don't
        // matter for these tests, only that they're strictly increasing.
        if d.format("%u").to_string() != "6" && d.format("%u").to_string() != "7" {
            dates.push(d);
        }
        d = d.succ_opt().unwrap();
    }
    dates
}

/// Builds `n_obs` days of synthetic factor returns (small, noise-scale
/// values) and one stock whose returns are generated *exactly* from
/// `intercept + betas . factors + noise`, so a fitted OLS should recover
/// `betas`/`intercept` within a small tolerance.
pub fn synthetic_single_stock(
    ticker: &str,
    intercept: f64,
    betas: &[f64],
    n_obs: usize,
    noise_scale: f64,
    seed: u64,
) -> MarketData {
    assert_eq!(betas.len(), FACTOR_NAMES.len());
    let mut rng = Rng::new(seed);

    let dates = business_days(n_obs + 1);

    let mut factor_returns: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut factor_series: Vec<Vec<f64>> = Vec::with_capacity(FACTOR_NAMES.len());
    for (k, name) in FACTOR_NAMES.iter().enumerate() {
        let scale = 0.01 + 0.002 * k as f64;
        let series: Vec<f64> = (0..n_obs).map(|_| rng.next_signed() * scale).collect();
        factor_returns.insert(name.to_string(), series.clone());
        factor_series.push(series);
    }

    let stock_series: Vec<f64> = (0..n_obs)
        .map(|t| {
            let systematic: f64 = betas
                .iter()
                .zip(factor_series.iter())
                .map(|(b, f)| b * f[t])
                .sum();
            intercept + systematic + rng.next_signed() * noise_scale
        })
        .collect();

    let mut stock_returns = BTreeMap::new();
    stock_returns.insert(ticker.to_string(), stock_series);

    let quality = DataQuality {
        date_range_start: dates[0],
        date_range_end: *dates.last().unwrap(),
        trading_days: dates.len(),
        per_series: vec![SeriesQuality {
            ticker: ticker.to_string(),
            raw_observations: n_obs + 1,
            forward_filled_days: 0,
            dropped_days: 0,
        }],
    };

    MarketData {
        dates,
        stock_returns,
        factor_returns,
        quality,
    }
}

/// Builds a `MarketData` directly from caller-specified per-stock log
/// return series (all series must have equal length `n_obs`). Factor
/// returns are filled with zeros (of the same length) since this is meant
/// for tests that only exercise stock-return-driven logic (e.g.
/// CvarRebalance, which uses raw historical stock returns, not the factor
/// model).
pub fn market_data_from_log_returns(stocks: &[(&str, Vec<f64>)]) -> MarketData {
    let n_obs = stocks[0].1.len();
    assert!(
        stocks.iter().all(|(_, r)| r.len() == n_obs),
        "all stock return series must have equal length"
    );
    let dates = business_days(n_obs + 1);

    let mut stock_returns = BTreeMap::new();
    let mut per_series = Vec::new();
    for (ticker, returns) in stocks {
        stock_returns.insert(ticker.to_string(), returns.clone());
        per_series.push(SeriesQuality {
            ticker: ticker.to_string(),
            raw_observations: n_obs + 1,
            forward_filled_days: 0,
            dropped_days: 0,
        });
    }

    let mut factor_returns = BTreeMap::new();
    for name in FACTOR_NAMES {
        factor_returns.insert(name.to_string(), vec![0.0; n_obs]);
    }

    let quality = DataQuality {
        date_range_start: dates[0],
        date_range_end: *dates.last().unwrap(),
        trading_days: dates.len(),
        per_series,
    };

    MarketData {
        dates,
        stock_returns,
        factor_returns,
        quality,
    }
}

/// Extends `synthetic_single_stock`'s factor data with additional stocks,
/// each following `intercept + betas . factors + noise` with its own beta
/// vector (used for multi-stock portfolio tests).
pub fn synthetic_multi_stock(
    stocks: &[(&str, f64, [f64; 5])],
    n_obs: usize,
    noise_scale: f64,
    seed: u64,
) -> MarketData {
    let mut rng = Rng::new(seed);
    let dates = business_days(n_obs + 1);

    let mut factor_returns: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut factor_series: Vec<Vec<f64>> = Vec::with_capacity(FACTOR_NAMES.len());
    for (k, name) in FACTOR_NAMES.iter().enumerate() {
        let scale = 0.01 + 0.002 * k as f64;
        let series: Vec<f64> = (0..n_obs).map(|_| rng.next_signed() * scale).collect();
        factor_returns.insert(name.to_string(), series.clone());
        factor_series.push(series);
    }

    let mut stock_returns = BTreeMap::new();
    let mut per_series = Vec::new();
    for (ticker, intercept, betas) in stocks {
        let series: Vec<f64> = (0..n_obs)
            .map(|t| {
                let systematic: f64 = betas
                    .iter()
                    .zip(factor_series.iter())
                    .map(|(b, f)| b * f[t])
                    .sum();
                intercept + systematic + rng.next_signed() * noise_scale
            })
            .collect();
        stock_returns.insert(ticker.to_string(), series);
        per_series.push(SeriesQuality {
            ticker: ticker.to_string(),
            raw_observations: n_obs + 1,
            forward_filled_days: 0,
            dropped_days: 0,
        });
    }

    let quality = DataQuality {
        date_range_start: dates[0],
        date_range_end: *dates.last().unwrap(),
        trading_days: dates.len(),
        per_series,
    };

    MarketData {
        dates,
        stock_returns,
        factor_returns,
        quality,
    }
}
