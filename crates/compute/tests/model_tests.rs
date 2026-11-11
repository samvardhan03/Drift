mod common;

use std::collections::BTreeMap;

use chrono::NaiveDate;
use compute::data::{DataQuality, MarketData, SeriesQuality, FACTOR_NAMES};
use compute::model::{fit_factor_model, ledoit_wolf_shrink_identity, Frequency, ModelConfig};
use compute::ComputeError;
use nalgebra::DMatrix;

/// Builds a hermetic `MarketData` with `n_factor_obs` factor observations
/// (long enough for a real regime fit) and two stock tickers whose own
/// return series may have different lengths, to exercise
/// `fit_factor_model`'s per-ticker windowing (see its doc) independent of
/// the data layer's own calendar alignment (`data::load_aligned_prices`).
fn two_stock_market_data(n_factor_obs: usize, long_obs: usize, short_obs: usize) -> MarketData {
    let mut rng = common::Rng::new(123);
    let mut factor_returns: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for name in FACTOR_NAMES {
        factor_returns.insert(
            name.to_string(),
            (0..n_factor_obs).map(|_| rng.next_signed() * 0.01).collect(),
        );
    }

    let mut stock_returns = BTreeMap::new();
    stock_returns.insert("LONG".to_string(), (0..long_obs).map(|_| rng.next_signed() * 0.01).collect());
    stock_returns.insert("SHORT".to_string(), (0..short_obs).map(|_| rng.next_signed() * 0.01).collect());

    let start = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
    let dates: Vec<NaiveDate> = (0..=n_factor_obs).map(|i| start + chrono::Duration::days(i as i64)).collect();
    let quality = DataQuality {
        date_range_start: dates[0],
        date_range_end: *dates.last().unwrap(),
        trading_days: dates.len(),
        per_series: vec![
            SeriesQuality {
                ticker: "LONG".to_string(),
                raw_observations: long_obs + 1,
                forward_filled_days: 0,
                dropped_days: 0,
            },
            SeriesQuality {
                ticker: "SHORT".to_string(),
                raw_observations: short_obs + 1,
                forward_filled_days: 0,
                dropped_days: 0,
            },
        ],
    };

    MarketData { dates, stock_returns, factor_returns, quality }
}

#[test]
fn ols_recovers_known_betas() {
    let true_intercept = 0.0003;
    let true_betas = [1.1, -0.4, 0.15, 0.05, 0.6];
    let data = common::synthetic_single_stock(
        "TEST",
        true_intercept,
        &true_betas,
        400,
        0.0005, // small noise relative to signal
        42,
    );

    let tickers = vec!["TEST".to_string()];
    let model = fit_factor_model(&data, &tickers, ModelConfig::new(252, Frequency::Daily))
        .expect("fit should succeed");

    let fit = &model.fits[0];
    assert_eq!(fit.betas.len(), 5);
    for (recovered, expected) in fit.betas.iter().zip(true_betas.iter()) {
        assert!(
            (recovered - expected).abs() < 0.05,
            "recovered beta {recovered} too far from expected {expected}"
        );
    }
    assert!(
        (fit.intercept - true_intercept).abs() < 0.001,
        "recovered intercept {} too far from expected {true_intercept}",
        fit.intercept
    );
    assert!(fit.r_squared > 0.9, "R^2 {} unexpectedly low", fit.r_squared);
}

#[test]
fn ledoit_wolf_shrinkage_in_unit_interval_and_psd() {
    let true_betas = [0.8, 0.1, -0.2, 0.3, -0.1];
    let data = common::synthetic_single_stock("TEST", 0.0, &true_betas, 300, 0.001, 7);
    let tickers = vec!["TEST".to_string()];
    let model = fit_factor_model(&data, &tickers, ModelConfig::new(252, Frequency::Daily)).unwrap();

    assert!(
        (0.0..=1.0).contains(&model.shrinkage_intensity),
        "shrinkage intensity {} out of [0,1]",
        model.shrinkage_intensity
    );

    let f = model.factor_covariance_daily.clone();
    assert!(is_symmetric(&f, 1e-9), "F is not symmetric");
    let eig = f.symmetric_eigenvalues();
    for lambda in eig.iter() {
        assert!(*lambda >= -1e-8, "F has a negative eigenvalue: {lambda}");
    }
}

#[test]
fn ledoit_wolf_shrinks_pure_noise_toward_identity() {
    // For near-uncorrelated, near-equal-variance data, shrinkage should be
    // substantial (sample covariance is noisy relative to the target).
    let mut rng = common::Rng::new(99);
    let t = 60; // short window -> noisy sample covariance -> more shrinkage
    let n = 5;
    let data = DMatrix::from_fn(t, n, |_, _| rng.next_signed() * 0.01);
    let (shrunk, intensity) = ledoit_wolf_shrink_identity(&data);
    assert!(intensity > 0.0, "expected nonzero shrinkage on a short noisy window");
    assert!(is_symmetric(&shrunk, 1e-9));
}

/// Requires network access to Yahoo Finance (real NSEI + factor + stock
/// data), so it's `#[ignore]`d by default -- run explicitly with
/// `cargo test -p compute --test model_tests -- --ignored`. Every other
/// test in this crate is hermetic (synthetic data only); this is the one
/// deliberate exception the checkpoint spec asks for ("PSD ... on real
/// NSEI data").
#[test]
#[ignore]
fn regime_conditional_f_is_psd_for_all_three_regimes_on_real_nsei_data() {
    let cache_dir = std::path::Path::new("data/cache");
    let tickers = vec!["RELIANCE.NS".to_string(), "TCS.NS".to_string()];
    let data = compute::data::load_market_data(cache_dir, &tickers, false, Frequency::Daily)
        .expect("live Yahoo fetch failed");
    let model = fit_factor_model(&data, &tickers, ModelConfig::new(252, Frequency::Daily))
        .expect("regime-conditional fit failed on real data");

    let regime_fs = model
        .regime_factor_covariance_daily
        .as_ref()
        .expect("regime-conditioning is unconditional, regime_factor_covariance_daily must be Some");

    for (regime_idx, f) in regime_fs.iter().enumerate() {
        assert!(is_symmetric(f, 1e-9), "F for regime {regime_idx} is not symmetric");
        let eig = f.clone().symmetric_eigenvalues();
        for lambda in eig.iter() {
            assert!(
                *lambda >= -1e-8,
                "F for regime {regime_idx} has a negative eigenvalue: {lambda}"
            );
        }
    }
}

/// A ticker with 45 return observations (above `MIN_TICKER_OBSERVATIONS`=30,
/// below the configured 252-day window) is fit on its own 45-day window
/// rather than failing the whole request -- see `fit_factor_model`'s
/// per-ticker windowing doc.
#[test]
fn a_ticker_with_45_observations_uses_a_45_day_window_and_is_recorded_as_short_history() {
    let data = two_stock_market_data(300, 300, 45);
    let tickers = vec!["LONG".to_string(), "SHORT".to_string()];
    let model = fit_factor_model(&data, &tickers, ModelConfig::new(252, Frequency::Daily))
        .expect("a 45-obs ticker should fit, not hard-fail");

    let short_fit = model.fits.iter().find(|f| f.ticker == "SHORT").unwrap();
    assert_eq!(short_fit.n_obs, 45);
    let long_fit = model.fits.iter().find(|f| f.ticker == "LONG").unwrap();
    assert_eq!(long_fit.n_obs, 252, "an established ticker must keep the full configured window");

    assert_eq!(model.short_history_tickers, vec![("SHORT".to_string(), 45)]);
}

/// A ticker with fewer than `MIN_TICKER_OBSERVATIONS` (30) return
/// observations hard-fails the whole request with
/// `ComputeError::InsufficientData`, naming the ticker and its observation
/// count.
#[test]
fn a_ticker_with_20_observations_hard_fails_with_insufficient_data() {
    let data = two_stock_market_data(300, 300, 20);
    let tickers = vec!["LONG".to_string(), "SHORT".to_string()];
    let result = fit_factor_model(&data, &tickers, ModelConfig::new(252, Frequency::Daily));

    let Err(err) = result else {
        panic!("a 20-obs ticker must hard-fail");
    };
    match err {
        ComputeError::InsufficientData(msg) => {
            assert!(msg.contains("SHORT"), "message should name the ticker: {msg}");
            assert!(msg.contains("20"), "message should state the observation count: {msg}");
        }
        other => panic!("expected ComputeError::InsufficientData, got {other:?}"),
    }
}

/// When the *shared* factor history itself is too thin (here, only 70
/// total factor observations -- below `MIN_MODEL_WINDOW`=90, independent of
/// any single ticker's own history), the whole request fails with a clear
/// `ComputeError::InsufficientData`, not an opaque regime-fit error.
#[test]
fn an_effective_window_below_the_model_floor_hard_fails_with_a_clear_message() {
    let data = two_stock_market_data(70, 300, 300);
    let tickers = vec!["LONG".to_string(), "SHORT".to_string()];
    let result = fit_factor_model(&data, &tickers, ModelConfig::new(252, Frequency::Daily));

    let Err(err) = result else {
        panic!("a factor history shorter than the model floor must hard-fail");
    };
    match err {
        ComputeError::InsufficientData(msg) => {
            assert!(msg.contains("70"), "message should state the available observations: {msg}");
        }
        other => panic!("expected ComputeError::InsufficientData, got {other:?}"),
    }
}

fn is_symmetric(m: &DMatrix<f64>, tol: f64) -> bool {
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            if (m[(i, j)] - m[(j, i)]).abs() > tol {
                return false;
            }
        }
    }
    true
}
