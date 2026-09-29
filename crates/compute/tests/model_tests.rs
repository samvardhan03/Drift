mod common;

use compute::model::{fit_factor_model, ledoit_wolf_shrink_identity, Frequency, ModelConfig};
use nalgebra::DMatrix;

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
