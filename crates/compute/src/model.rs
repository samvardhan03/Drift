//! Factor model: per-stock OLS on the five fixed factors, Ledoit-Wolf
//! shrinkage of the factor covariance matrix, and the resulting stock
//! covariance Sigma = B F B^T + D.

use nalgebra::{DMatrix, DVector};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::{MarketData, FACTOR_NAMES};
use crate::error::{ComputeError, Result};
use crate::regime::{self, RegimeState};

/// Minimum observations a regime needs, within the fitted window, for its
/// own Ledoit-Wolf factor covariance to be used; below this, that
/// regime's covariance falls back to the full-window `F` and a warning is
/// recorded (see `fit_factor_model`'s `regime_covariance` handling).
pub const MIN_REGIME_OBSERVATIONS: usize = 30;

/// Return frequency the factor model is fit at. Non-overlapping: `Weekly`
/// returns are computed between successive week-end closes, not a rolling
/// 5-day window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
pub enum Frequency {
    #[default]
    Daily,
    Weekly,
}

impl Frequency {
    /// Periods per year at this frequency. This is the single place the
    /// annualization factor is defined: everything upstream (OLS,
    /// Ledoit-Wolf shrinkage) is computed on period returns at whatever
    /// frequency was chosen, and `annualize_matrix`/`annualize_scalar` are
    /// the only functions that scale a period (co)variance into annual
    /// terms, using this value.
    pub fn annualization_factor(&self) -> f64 {
        match self {
            Frequency::Daily => 252.0,
            Frequency::Weekly => 52.0,
        }
    }

    /// Default trailing window, in periods at this frequency: 252 trading
    /// days, or 156 weeks (~3 years, chosen to keep a comparable number of
    /// independent observations to the daily 252-day/~1yr default given
    /// weekly data's lower observation density).
    pub fn default_window(&self) -> usize {
        match self {
            Frequency::Daily => 252,
            Frequency::Weekly => 156,
        }
    }

    /// Parses the `"daily"`/`"weekly"` (case-insensitive) string form some
    /// experiment inputs (`RiskDriftInput`, `ReverseStressInput`) use
    /// instead of the enum directly (so the field can default to `None` ->
    /// `Daily` without schemars needing a default-valued enum). `None`
    /// (the field omitted) -> `Daily`.
    pub fn from_optional_str(s: Option<&str>) -> Result<Frequency> {
        match s {
            None => Ok(Frequency::Daily),
            Some(s) if s.eq_ignore_ascii_case("daily") => Ok(Frequency::Daily),
            Some(s) if s.eq_ignore_ascii_case("weekly") => Ok(Frequency::Weekly),
            Some(other) => Err(ComputeError::InvalidInput(format!(
                "frequency must be \"daily\" or \"weekly\", got {other:?}"
            ))),
        }
    }
}

pub fn annualize_scalar(period_variance: f64, frequency: Frequency) -> f64 {
    period_variance * frequency.annualization_factor()
}

pub fn annualize_matrix(period_covariance: &DMatrix<f64>, frequency: Frequency) -> DMatrix<f64> {
    period_covariance * frequency.annualization_factor()
}

/// serde `default =` helper: default trailing window for experiment inputs
/// (daily convention; callers on `Frequency::Weekly` should override).
pub fn default_window() -> usize {
    Frequency::Daily.default_window()
}

/// Per-stock OLS fit against the five fixed factors.
#[derive(Debug, Clone)]
pub struct StockFit {
    pub ticker: String,
    pub intercept: f64,
    /// Betas in `FACTOR_NAMES` order.
    pub betas: Vec<f64>,
    /// Daily residual variance (unannualized).
    pub residual_variance_daily: f64,
    pub r_squared: f64,
    pub n_obs: usize,
}

/// `fit_factor_model`'s configuration: the trailing window and its
/// frequency. The factor covariance is always conditioned on the current
/// market regime (see `regime` module doc) — there is no longer a flag to
/// opt out of it.
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    pub window: usize,
    pub frequency: Frequency,
}

impl ModelConfig {
    pub fn new(window: usize, frequency: Frequency) -> Self {
        ModelConfig { window, frequency }
    }
}

/// The fitted factor model for a set of stocks over a trailing window.
pub struct FactorModel {
    pub frequency: Frequency,
    pub window: usize,
    pub tickers: Vec<String>,
    pub fits: Vec<StockFit>,
    /// Per-period (daily or weekly, per `frequency`) factor covariance
    /// after Ledoit-Wolf shrinkage, in `FACTOR_NAMES` order (rows/cols).
    /// This is `F` for the *current* regime (see
    /// `regime_factor_covariance_daily`) — every consumer of
    /// `factor_covariance()`/`stock_covariance()` is therefore always
    /// regime-conditional.
    pub factor_covariance_daily: DMatrix<f64>,
    pub shrinkage_intensity: f64,
    /// Always `Some` after `fit_factor_model` — regime-conditioning is
    /// unconditional.
    pub regime_state: Option<RegimeState>,
    /// Always `Some` after `fit_factor_model`: the per-period, per-regime
    /// factor covariance in Bull/Bear/Crisis order (index 0/1/2), each
    /// either fit on that regime's own days within the window or, if it had
    /// fewer than `MIN_REGIME_OBSERVATIONS`, a copy of the full-window `F`
    /// (see `regime_fallback_warnings`).
    pub regime_factor_covariance_daily: Option<[DMatrix<f64>; 3]>,
    pub regime_fallback_warnings: Vec<String>,
}

impl FactorModel {
    /// Beta matrix B (n_stocks x n_factors), `FACTOR_NAMES` column order.
    pub fn beta_matrix(&self) -> DMatrix<f64> {
        let n = self.fits.len();
        let k = FACTOR_NAMES.len();
        DMatrix::from_fn(n, k, |i, j| self.fits[i].betas[j])
    }

    /// Diagonal matrix D of per-period residual variances (n_stocks x n_stocks).
    pub fn residual_matrix_daily(&self) -> DMatrix<f64> {
        let n = self.fits.len();
        DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                self.fits[i].residual_variance_daily
            } else {
                0.0
            }
        })
    }

    /// Annualized stock covariance Sigma = B F B^T + D.
    pub fn stock_covariance(&self) -> DMatrix<f64> {
        let b = self.beta_matrix();
        let f = &self.factor_covariance_daily;
        let d = self.residual_matrix_daily();
        let period_sigma = &b * f * b.transpose() + d;
        annualize_matrix(&period_sigma, self.frequency)
    }

    /// Annualized factor covariance F.
    pub fn factor_covariance(&self) -> DMatrix<f64> {
        annualize_matrix(&self.factor_covariance_daily, self.frequency)
    }

    /// Annualized factor covariance for a specific regime (0=Bull,
    /// 1=Bear, 2=Crisis), independent of which regime is "current". Always
    /// `Some` after `fit_factor_model`.
    pub fn factor_covariance_for_regime(&self, regime: usize) -> Option<DMatrix<f64>> {
        self.regime_factor_covariance_daily
            .as_ref()
            .map(|fs| annualize_matrix(&fs[regime], self.frequency))
    }

    /// Annualized stock covariance Sigma = B F_regime B^T + D, for a
    /// specific regime (see `factor_covariance_for_regime`). `D` (specific
    /// risk) is unchanged — it's always the full-window residual variance,
    /// per the checkpoint spec.
    pub fn stock_covariance_for_regime(&self, regime: usize) -> Option<DMatrix<f64>> {
        let f = self.regime_factor_covariance_daily.as_ref()?;
        let b = self.beta_matrix();
        let d = self.residual_matrix_daily();
        let period_sigma = &b * &f[regime] * b.transpose() + d;
        Some(annualize_matrix(&period_sigma, self.frequency))
    }

    /// Factor correlation matrix, in `FACTOR_NAMES` order. Scale-free: the
    /// annualization factor cancels in the ratio, so this is identical
    /// whether computed from the per-period or annualized covariance.
    pub fn factor_correlation(&self) -> DMatrix<f64> {
        let f = &self.factor_covariance_daily;
        let n = f.nrows();
        let sd: Vec<f64> = (0..n).map(|i| f[(i, i)].max(0.0).sqrt()).collect();
        DMatrix::from_fn(n, n, |i, j| {
            if sd[i] > 0.0 && sd[j] > 0.0 {
                f[(i, j)] / (sd[i] * sd[j])
            } else {
                0.0
            }
        })
    }
}

/// Ordinary least squares with intercept: y ~ 1 + X, solved via the
/// Moore-Penrose pseudo-inverse (SVD) for numerical robustness on
/// possibly collinear factor windows.
fn ols_fit(y: &DVector<f64>, factors: &DMatrix<f64>) -> Result<(f64, Vec<f64>, f64, f64)> {
    let n = y.len();
    let k = factors.ncols();
    if n <= k + 1 {
        return Err(ComputeError::Model(format!(
            "not enough observations ({n}) for {k} factors plus intercept"
        )));
    }

    let mut design = DMatrix::from_element(n, k + 1, 1.0);
    for i in 0..n {
        for j in 0..k {
            design[(i, j + 1)] = factors[(i, j)];
        }
    }

    let svd = design.clone().svd(true, true);
    let coeffs = svd
        .solve(y, 1e-12)
        .map_err(|e| ComputeError::Model(format!("OLS solve failed: {e}")))?;

    let intercept = coeffs[0];
    let betas: Vec<f64> = (0..k).map(|j| coeffs[j + 1]).collect();

    let fitted = &design * &coeffs;
    let residuals = y - &fitted;
    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let y_mean = y.mean();
    let tss: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let r_squared = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };
    // Residual variance uses n - k - 1 degrees of freedom (intercept + k betas).
    let dof = (n - k - 1).max(1) as f64;
    let residual_variance = rss / dof;

    Ok((intercept, betas, residual_variance, r_squared))
}

/// Ledoit-Wolf shrinkage of the sample covariance of `data` (T x N, rows =
/// observations) toward a scaled-identity target `mu * I`, following
/// Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix".
///
/// Returns `(shrunk_covariance, shrinkage_intensity)`.
pub fn ledoit_wolf_shrink_identity(data: &DMatrix<f64>) -> (DMatrix<f64>, f64) {
    let t = data.nrows() as f64;
    let n = data.ncols();

    // Demean each column (factor) over the window.
    let means: Vec<f64> = (0..n)
        .map(|j| data.column(j).iter().sum::<f64>() / t)
        .collect();
    let mut centered = data.clone();
    for j in 0..n {
        for i in 0..data.nrows() {
            centered[(i, j)] -= means[j];
        }
    }

    // Sample covariance S = (1/T) X'X.
    let sample_cov: DMatrix<f64> = (&centered.transpose() * &centered) / t;

    // Target: mu * I, mu = trace(S) / N.
    let mu = sample_cov.trace() / n as f64;
    let target = DMatrix::identity(n, n) * mu;

    // pi_hat: average over t of ||x_t x_t' - S||_F^2.
    let mut pi_sum = 0.0;
    for i in 0..data.nrows() {
        let row = centered.row(i).transpose();
        let outer = &row * row.transpose();
        let diff = &outer - &sample_cov;
        pi_sum += diff.iter().map(|v| v * v).sum::<f64>();
    }
    let pi_hat = pi_sum / t;

    // gamma_hat = ||S - target||_F^2 (rho_hat = 0 for an identity target:
    // the target has no estimation error and zero off-diagonal covariance
    // with the sample covariance's off-diagonal entries).
    let diff_target = &sample_cov - &target;
    let gamma_hat = diff_target.iter().map(|v| v * v).sum::<f64>();

    let shrinkage = if gamma_hat > 0.0 {
        (pi_hat / (t * gamma_hat)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let shrunk = &target * shrinkage + &sample_cov * (1.0 - shrinkage);
    (shrunk, shrinkage)
}

/// Splits `factors_window` (window x n_factors) into per-regime
/// sub-matrices by `viterbi_sequence` (one label per row, 0/1/2) and
/// Ledoit-Wolf-shrinks each. A regime with fewer than
/// `MIN_REGIME_OBSERVATIONS` rows falls back to `full_window_f` and gets a
/// warning string pushed instead. Factored out of
/// `fit_factor_model_with_config` so the fallback path can be tested with
/// an injected/synthetic `viterbi_sequence`, independent of whether a real
/// HMM fit happens to produce a regime that small.
fn regime_conditional_factor_covariance(
    factors_window: &DMatrix<f64>,
    viterbi_sequence: &[u8],
    full_window_f: &DMatrix<f64>,
) -> ([DMatrix<f64>; 3], Vec<String>) {
    let k = factors_window.ncols();
    let mut per_regime: [DMatrix<f64>; 3] = std::array::from_fn(|_| full_window_f.clone());
    let mut warnings = Vec::new();

    for (regime_idx, label) in regime::REGIME_LABELS.iter().enumerate() {
        let row_indices: Vec<usize> = (0..viterbi_sequence.len())
            .filter(|&t| viterbi_sequence[t] as usize == regime_idx)
            .collect();
        if row_indices.len() < MIN_REGIME_OBSERVATIONS {
            warnings.push(format!(
                "regime_{regime_idx} ({label}) has only {} observations, fell back to \
                 full-window covariance",
                row_indices.len()
            ));
            // per_regime[regime_idx] already initialized to the full-window F.
            continue;
        }
        let regime_matrix =
            DMatrix::from_fn(row_indices.len(), k, |i, j| factors_window[(row_indices[i], j)]);
        let (f_k, _shrinkage_k) = ledoit_wolf_shrink_identity(&regime_matrix);
        per_regime[regime_idx] = f_k;
    }

    (per_regime, warnings)
}

/// Fits the factor model for `tickers` over the trailing `window` periods
/// (trading days or weeks, per `frequency`) of `data` (the most recent
/// `window` return observations, which must already be at `frequency`),
/// per `config` (see `ModelConfig`). Always fits a 3-state HMM on the
/// window's `MARKET` (Nifty) returns and splits the factor covariance by
/// Viterbi-assigned regime — see `regime` module doc and the
/// `regime_factor_covariance_daily`/`regime_state` fields.
pub fn fit_factor_model(
    data: &MarketData,
    tickers: &[String],
    config: ModelConfig,
) -> Result<FactorModel> {
    let ModelConfig { window, frequency } = config;

    let factor_matrices: Vec<&Vec<f64>> = FACTOR_NAMES
        .iter()
        .map(|name| {
            data.factor_returns
                .get(*name)
                .ok_or_else(|| ComputeError::Model(format!("missing factor series {name}")))
        })
        .collect::<Result<Vec<_>>>()?;

    let total_obs = factor_matrices[0].len();
    if total_obs < window {
        return Err(ComputeError::Model(format!(
            "only {total_obs} return observations available, need {window} for the trailing window"
        )));
    }
    let start = total_obs - window;

    let k = FACTOR_NAMES.len();
    let factors_window = DMatrix::from_fn(window, k, |i, j| factor_matrices[j][start + i]);

    let (full_window_factor_covariance_daily, shrinkage_intensity) =
        ledoit_wolf_shrink_identity(&factors_window);

    // FACTOR_NAMES[0] == "MARKET" == Nifty (^NSEI); factor_matrices[0] is
    // that column, already sliced to the same [start, start+window) window
    // as everything else here.
    let nsei_window: Vec<f64> = factor_matrices[0][start..start + window].to_vec();
    let (_hmm, state) = regime::fit_hmm(&nsei_window)?;

    let (per_regime, regime_fallback_warnings) = regime_conditional_factor_covariance(
        &factors_window,
        &state.viterbi_sequence,
        &full_window_factor_covariance_daily,
    );
    let factor_covariance_daily = per_regime[state.current_regime as usize].clone();
    let regime_factor_covariance_daily = Some(per_regime);
    let regime_state = Some(state);

    let mut fits = Vec::with_capacity(tickers.len());
    for ticker in tickers {
        let series = data
            .stock_returns
            .get(ticker)
            .ok_or_else(|| ComputeError::Model(format!("missing return series for {ticker}")))?;
        if series.len() < window {
            return Err(ComputeError::Model(format!(
                "ticker {ticker} has only {} observations, need {window}",
                series.len()
            )));
        }
        let series_start = series.len() - window;
        let y = DVector::from_fn(window, |i, _| series[series_start + i]);

        let (intercept, betas, residual_variance_daily, r_squared) =
            ols_fit(&y, &factors_window)?;

        fits.push(StockFit {
            ticker: ticker.clone(),
            intercept,
            betas,
            residual_variance_daily,
            r_squared,
            n_obs: window,
        });
    }

    Ok(FactorModel {
        frequency,
        window,
        tickers: tickers.to_vec(),
        fits,
        factor_covariance_daily,
        shrinkage_intensity,
        regime_state,
        regime_factor_covariance_daily,
        regime_fallback_warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic small PRNG, matching the pattern used elsewhere in
    /// this crate's tests (no `rand` dependency).
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

    fn random_factor_matrix(rows: usize, cols: usize, seed: u64) -> DMatrix<f64> {
        let mut rng = Rng(seed.max(1));
        DMatrix::from_fn(rows, cols, |_, _| rng.next_signed() * 0.01)
    }

    /// A regime with < MIN_REGIME_OBSERVATIONS rows must fall back to the
    /// full-window F and record a warning naming that regime — tested with
    /// an injected `viterbi_sequence` rather than relying on a real HMM fit
    /// happening to produce such a skewed split.
    #[test]
    fn fallback_warning_fires_for_a_regime_with_too_few_observations() {
        let window = 100;
        let k = FACTOR_NAMES.len();
        let factors_window = random_factor_matrix(window, k, 7);
        let (full_window_f, _) = ledoit_wolf_shrink_identity(&factors_window);

        // Bull (0): 60 obs, Bear (1): 25 obs (< 30, should fall back),
        // Crisis (2): 15 obs (< 30, should fall back).
        let mut viterbi_sequence = vec![0u8; 60];
        viterbi_sequence.extend(std::iter::repeat_n(1u8, 25));
        viterbi_sequence.extend(std::iter::repeat_n(2u8, 15));
        assert_eq!(viterbi_sequence.len(), window);

        let (per_regime, warnings) =
            regime_conditional_factor_covariance(&factors_window, &viterbi_sequence, &full_window_f);

        assert_eq!(warnings.len(), 2, "expected exactly 2 fallback warnings, got {warnings:?}");
        assert!(warnings[0].contains("regime_1") && warnings[0].contains("Bear") && warnings[0].contains("25"));
        assert!(warnings[1].contains("regime_2") && warnings[1].contains("Crisis") && warnings[1].contains("15"));

        // Regime 0 (Bull, 60 obs >= 30) should NOT equal the full-window F
        // (it's fit on its own, distinct data); regimes 1 and 2 (fallback)
        // should equal the full-window F exactly.
        assert_ne!(per_regime[0], full_window_f);
        assert_eq!(per_regime[1], full_window_f);
        assert_eq!(per_regime[2], full_window_f);
    }

    #[test]
    fn no_fallback_warnings_when_every_regime_has_enough_observations() {
        let window = 90;
        let k = FACTOR_NAMES.len();
        let factors_window = random_factor_matrix(window, k, 11);
        let (full_window_f, _) = ledoit_wolf_shrink_identity(&factors_window);

        let viterbi_sequence: Vec<u8> = (0..window).map(|i| (i / 30) as u8).collect();
        let (_per_regime, warnings) =
            regime_conditional_factor_covariance(&factors_window, &viterbi_sequence, &full_window_f);
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    }
}
