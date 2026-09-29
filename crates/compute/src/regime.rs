// Index loops throughout this module pair `j`/`k` across N_STATES x
// N_STATES matrices (transition, xi) matching the math notation in the
// doc comments; clippy's enumerate()/zip() rewrites for that pattern are
// harder to read, not easier, so they're disabled for the whole file
// rather than case-by-case.
#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]

//! Regime detection: a 3-state Gaussian Hidden Markov Model fit on Nifty
//! (`^NSEI`) daily log returns via Baum-Welch (implemented from scratch —
//! no external HMM crate), used to split the factor covariance by market
//! regime (`model::fit_factor_model`'s `regime_covariance` option).
//!
//! States are always reported in ascending emission-variance order —
//! 0 = Bull (lowest vol), 1 = Bear (medium vol), 2 = Crisis (highest vol) —
//! regardless of which internal state index the Baum-Welch fit happened to
//! converge to for which regime; the relabelling step at the end of
//! `fit_hmm` is what guarantees that determinism (see its doc comment).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{ComputeError, Result};

const N_STATES: usize = 3;
const MAX_ITER: u32 = 500;
const LOG_LIKELIHOOD_TOLERANCE: f64 = 1e-6;
/// Floor for emission variance and forward/backward normalizers, so a
/// degenerate (near-zero-variance) cluster or a vanishing likelihood can't
/// produce a division by zero or an infinite Gaussian density.
const VARIANCE_FLOOR: f64 = 1e-10;
const PROB_FLOOR: f64 = 1e-300;

pub const REGIME_LABELS: [&str; N_STATES] = ["Bull", "Bear", "Crisis"];

/// The fitted HMM's own parameters, in the final (variance-sorted)
/// Bull/Bear/Crisis state order.
#[derive(Debug, Clone)]
pub struct HmmModel {
    /// Initial-state distribution, pi\[k\].
    pub initial: [f64; N_STATES],
    /// Transition matrix, transition\[j\]\[k\] = P(state_{t+1}=k | state_t=j).
    pub transition: [[f64; N_STATES]; N_STATES],
    /// Per-state Gaussian emission mean.
    pub means: [f64; N_STATES],
    /// Per-state Gaussian emission variance.
    pub variances: [f64; N_STATES],
    pub log_likelihood: f64,
    pub n_iter: u32,
}

/// Constant `RegimeState::smoothing_note` value (see its field doc).
const SMOOTHING_NOTE: &str = "full-history smoothed, not suitable for live trading signals";

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct RegimeState {
    /// 0 = Bull, 1 = Bear, 2 = Crisis.
    pub current_regime: u8,
    pub current_label: &'static str,
    /// gamma_T(k): the smoothed regime-probability distribution at the
    /// final observation.
    pub smoothed_probs: [f64; N_STATES],
    /// The full Viterbi-decoded state sequence, one entry per input
    /// observation, in the same variance-sorted state order.
    pub viterbi_sequence: Vec<u8>,
    pub obs_count_per_regime: [usize; N_STATES],
    pub log_likelihood: f64,
    pub n_iter: u32,
    pub smoothing_note: &'static str,
}

/// Hand-written (not derived): `current_label`/`smoothing_note` are
/// `&'static str`, which `#[derive(Deserialize)]` cannot produce (it would
/// need to borrow from the deserializer's input, not `'static`). Round-trips
/// through an owned-`String` shadow struct instead, mapping `current_label`
/// back onto one of `REGIME_LABELS`'s `'static` entries (round-tripping
/// `EvidenceTrace` through JSON is exactly what `store::SnapshotStore`'s
/// persistence, and `GET /report/{id}`'s reconstruction of it, both need).
impl<'de> Deserialize<'de> for RegimeState {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RegimeStateOwned {
            current_regime: u8,
            current_label: String,
            smoothed_probs: [f64; N_STATES],
            viterbi_sequence: Vec<u8>,
            obs_count_per_regime: [usize; N_STATES],
            log_likelihood: f64,
            n_iter: u32,
            #[allow(dead_code)]
            smoothing_note: String,
        }
        let owned = RegimeStateOwned::deserialize(deserializer)?;
        let current_label = REGIME_LABELS
            .iter()
            .find(|&&label| label == owned.current_label)
            .copied()
            .ok_or_else(|| serde::de::Error::custom(format!("unknown regime label {:?}", owned.current_label)))?;
        Ok(RegimeState {
            current_regime: owned.current_regime,
            current_label,
            smoothed_probs: owned.smoothed_probs,
            viterbi_sequence: owned.viterbi_sequence,
            obs_count_per_regime: owned.obs_count_per_regime,
            log_likelihood: owned.log_likelihood,
            n_iter: owned.n_iter,
            smoothing_note: SMOOTHING_NOTE,
        })
    }
}

fn gaussian_pdf(x: f64, mean: f64, variance: f64) -> f64 {
    let variance = variance.max(VARIANCE_FLOOR);
    let exponent = -(x - mean).powi(2) / (2.0 * variance);
    exponent.exp() / (2.0 * std::f64::consts::PI * variance).sqrt()
}

/// Deterministic k-means (k=3) initialization.
///
/// Clusters on `|data|` (magnitude), not on the raw signed values: HMM
/// regimes here are meant to separate by *volatility*, and daily return
/// regimes are all roughly zero-mean, so clustering on signed values just
/// splits points into "very negative" / "near zero" / "very positive"
/// buckets by direction — which has nothing to do with which volatility
/// regime a point belongs to, and was confirmed live to make Baum-Welch
/// converge to a poor local optimum (two of the three fitted states ended
/// up with similar variances, differentiated mostly by mean, on synthetic
/// data with three well-separated *true* variances and zero true mean
/// everywhere). Clustering on magnitude directly targets spread instead.
/// Final per-state means/variances are still computed from the original
/// signed data within each magnitude-assigned cluster.
fn kmeans_init(data: &[f64]) -> ([f64; N_STATES], [f64; N_STATES]) {
    let n = data.len();
    let magnitudes: Vec<f64> = data.iter().map(|x| x.abs()).collect();
    let mut sorted = magnitudes.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut centers = [sorted[n / 6], sorted[n / 2], sorted[(5 * n) / 6]];

    let mut assignments = vec![0usize; n];
    for _ in 0..100 {
        let mut changed = false;
        for (i, &m) in magnitudes.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = f64::INFINITY;
            for (k, &c) in centers.iter().enumerate() {
                let d = (m - c).powi(2);
                if d < best_dist {
                    best_dist = d;
                    best = k;
                }
            }
            if assignments[i] != best {
                changed = true;
                assignments[i] = best;
            }
        }
        let mut sums = [0.0; N_STATES];
        let mut counts = [0usize; N_STATES];
        for (i, &m) in magnitudes.iter().enumerate() {
            sums[assignments[i]] += m;
            counts[assignments[i]] += 1;
        }
        for k in 0..N_STATES {
            if counts[k] > 0 {
                centers[k] = sums[k] / counts[k] as f64;
            }
        }
        if !changed {
            break;
        }
    }

    let mut mean_sums = [0.0; N_STATES];
    let mut counts = [0usize; N_STATES];
    for (i, &x) in data.iter().enumerate() {
        let k = assignments[i];
        mean_sums[k] += x;
        counts[k] += 1;
    }
    let mut means = [0.0; N_STATES];
    for k in 0..N_STATES {
        if counts[k] > 0 {
            means[k] = mean_sums[k] / counts[k] as f64;
        }
    }

    let mut var_sums = [0.0; N_STATES];
    for (i, &x) in data.iter().enumerate() {
        let k = assignments[i];
        var_sums[k] += (x - means[k]).powi(2);
    }
    let overall_mean: f64 = data.iter().sum::<f64>() / n as f64;
    let overall_var: f64 =
        data.iter().map(|x| (x - overall_mean).powi(2)).sum::<f64>() / n as f64;
    let mut variances = [0.0; N_STATES];
    for k in 0..N_STATES {
        variances[k] = if counts[k] > 1 {
            (var_sums[k] / counts[k] as f64).max(VARIANCE_FLOOR)
        } else {
            // A cluster with 0 or 1 points has no meaningful within-cluster
            // variance; fall back to the whole series' variance so it's at
            // least a plausible starting guess rather than ~0.
            overall_var.max(VARIANCE_FLOOR)
        };
    }
    (means, variances)
}

/// Scaled forward pass (Rabiner 1989 scaling): returns
/// `(alpha_hat, c)` where `alpha_hat[t][k] = P(state_t=k | O_1..O_t)`
/// (sums to 1 over `k` at every `t`) and `c[t]` is the normalizer at `t`,
/// with `log P(O_1..O_T) = sum_t ln(c[t])`.
fn forward(
    data: &[f64],
    initial: &[f64; N_STATES],
    transition: &[[f64; N_STATES]; N_STATES],
    means: &[f64; N_STATES],
    variances: &[f64; N_STATES],
) -> (Vec<[f64; N_STATES]>, Vec<f64>) {
    let t_n = data.len();
    let mut alpha = vec![[0.0; N_STATES]; t_n];
    let mut c = vec![0.0; t_n];

    for k in 0..N_STATES {
        alpha[0][k] = initial[k] * gaussian_pdf(data[0], means[k], variances[k]);
    }
    c[0] = alpha[0].iter().sum::<f64>().max(PROB_FLOOR);
    for k in 0..N_STATES {
        alpha[0][k] /= c[0];
    }

    for t in 1..t_n {
        for k in 0..N_STATES {
            let predicted: f64 = (0..N_STATES).map(|j| alpha[t - 1][j] * transition[j][k]).sum();
            alpha[t][k] = predicted * gaussian_pdf(data[t], means[k], variances[k]);
        }
        c[t] = alpha[t].iter().sum::<f64>().max(PROB_FLOOR);
        for k in 0..N_STATES {
            alpha[t][k] /= c[t];
        }
    }

    (alpha, c)
}

/// Scaled backward pass, using the same `c` scale factors the forward pass
/// produced. `beta[t][k] = beta_true(t,k) / prod_{s=t+1}^{T} c[s]`, chosen
/// so that `alpha[t][k] * beta[t][k] == gamma_t(k)` exactly (see the
/// derivation in the module's design notes / commit message — the
/// standard Rabiner-scaling identity), with no further rescaling needed.
fn backward(
    data: &[f64],
    transition: &[[f64; N_STATES]; N_STATES],
    means: &[f64; N_STATES],
    variances: &[f64; N_STATES],
    c: &[f64],
) -> Vec<[f64; N_STATES]> {
    let t_n = data.len();
    let mut beta = vec![[0.0; N_STATES]; t_n];
    beta[t_n - 1] = [1.0; N_STATES];

    for t in (0..t_n - 1).rev() {
        for k in 0..N_STATES {
            let s: f64 = (0..N_STATES)
                .map(|j| transition[k][j] * gaussian_pdf(data[t + 1], means[j], variances[j]) * beta[t + 1][j])
                .sum();
            beta[t][k] = s / c[t + 1];
        }
    }

    beta
}

struct EStepResult {
    /// gamma[t][k] = P(state_t = k | all observations).
    gamma: Vec<[f64; N_STATES]>,
    /// xi_sum[j][k] = sum_{t=0}^{T-2} P(state_t=j, state_{t+1}=k | O).
    xi_sum: [[f64; N_STATES]; N_STATES],
}

fn e_step(
    data: &[f64],
    alpha: &[[f64; N_STATES]],
    beta: &[[f64; N_STATES]],
    transition: &[[f64; N_STATES]; N_STATES],
    means: &[f64; N_STATES],
    variances: &[f64; N_STATES],
    c: &[f64],
) -> EStepResult {
    let t_n = data.len();
    let mut gamma = vec![[0.0; N_STATES]; t_n];
    for t in 0..t_n {
        let mut row = [0.0; N_STATES];
        for k in 0..N_STATES {
            row[k] = alpha[t][k] * beta[t][k];
        }
        let sum: f64 = row.iter().sum::<f64>().max(PROB_FLOOR);
        for k in 0..N_STATES {
            gamma[t][k] = row[k] / sum;
        }
    }

    let mut xi_sum = [[0.0; N_STATES]; N_STATES];
    for t in 0..t_n - 1 {
        let mut xi_t = [[0.0; N_STATES]; N_STATES];
        let mut total = 0.0;
        for j in 0..N_STATES {
            for k in 0..N_STATES {
                let v = alpha[t][j]
                    * transition[j][k]
                    * gaussian_pdf(data[t + 1], means[k], variances[k])
                    * beta[t + 1][k]
                    / c[t + 1];
                xi_t[j][k] = v;
                total += v;
            }
        }
        let total = total.max(PROB_FLOOR);
        for j in 0..N_STATES {
            for k in 0..N_STATES {
                xi_sum[j][k] += xi_t[j][k] / total;
            }
        }
    }

    EStepResult { gamma, xi_sum }
}

fn m_step(
    data: &[f64],
    e: &EStepResult,
) -> ([f64; N_STATES], [[f64; N_STATES]; N_STATES], [f64; N_STATES], [f64; N_STATES]) {
    let t_n = data.len();

    let mut initial = [0.0; N_STATES];
    for k in 0..N_STATES {
        initial[k] = e.gamma[0][k];
    }

    let mut gamma_sum_excl_last = [0.0; N_STATES];
    for t in 0..t_n - 1 {
        for j in 0..N_STATES {
            gamma_sum_excl_last[j] += e.gamma[t][j];
        }
    }
    let mut transition = [[0.0; N_STATES]; N_STATES];
    for j in 0..N_STATES {
        let denom = gamma_sum_excl_last[j].max(PROB_FLOOR);
        for k in 0..N_STATES {
            transition[j][k] = e.xi_sum[j][k] / denom;
        }
    }

    let mut means = [0.0; N_STATES];
    let mut gamma_sum_all = [0.0; N_STATES];
    for t in 0..t_n {
        for k in 0..N_STATES {
            gamma_sum_all[k] += e.gamma[t][k];
            means[k] += e.gamma[t][k] * data[t];
        }
    }
    for k in 0..N_STATES {
        means[k] /= gamma_sum_all[k].max(PROB_FLOOR);
    }

    let mut variances = [0.0; N_STATES];
    for t in 0..t_n {
        for k in 0..N_STATES {
            variances[k] += e.gamma[t][k] * (data[t] - means[k]).powi(2);
        }
    }
    for k in 0..N_STATES {
        variances[k] = (variances[k] / gamma_sum_all[k].max(PROB_FLOOR)).max(VARIANCE_FLOOR);
    }

    (initial, transition, means, variances)
}

/// Log-space Viterbi: the single most likely state sequence given the
/// converged model parameters.
fn viterbi(
    data: &[f64],
    initial: &[f64; N_STATES],
    transition: &[[f64; N_STATES]; N_STATES],
    means: &[f64; N_STATES],
    variances: &[f64; N_STATES],
) -> Vec<usize> {
    let t_n = data.len();
    let ln_floor = PROB_FLOOR.ln();
    let mut delta = vec![[0.0; N_STATES]; t_n];
    let mut psi = vec![[0usize; N_STATES]; t_n];

    for k in 0..N_STATES {
        delta[0][k] =
            initial[k].max(PROB_FLOOR).ln() + gaussian_pdf(data[0], means[k], variances[k]).max(PROB_FLOOR).ln();
    }

    for t in 1..t_n {
        for k in 0..N_STATES {
            let mut best_val = f64::NEG_INFINITY;
            let mut best_j = 0;
            for j in 0..N_STATES {
                let v = delta[t - 1][j] + transition[j][k].max(PROB_FLOOR).ln();
                if v > best_val {
                    best_val = v;
                    best_j = j;
                }
            }
            let emit = gaussian_pdf(data[t], means[k], variances[k]).max(PROB_FLOOR).ln();
            delta[t][k] = best_val.max(ln_floor) + emit;
            psi[t][k] = best_j;
        }
    }

    let mut path = vec![0usize; t_n];
    path[t_n - 1] = (0..N_STATES)
        .max_by(|&a, &b| delta[t_n - 1][a].partial_cmp(&delta[t_n - 1][b]).unwrap())
        .unwrap();
    for t in (0..t_n - 1).rev() {
        path[t] = psi[t + 1][path[t + 1]];
    }
    path
}

/// Fits a 3-state Gaussian HMM on `nsei_returns` via Baum-Welch, then
/// relabels states into ascending-variance (Bull/Bear/Crisis) order.
///
/// The relabelling is what makes the result deterministic regardless of
/// which internal state index Baum-Welch happened to converge to for
/// which regime: after fitting, states are sorted purely by their final
/// emission variance and every output (`initial`, `transition`'s rows and
/// columns, `means`, `variances`, `gamma`'s columns, the Viterbi path's
/// labels) is permuted through that same sort order before being returned.
/// Two fits that converge to the "same" model up to a permutation of state
/// indices therefore always report identical Bull/Bear/Crisis labels.
pub fn fit_hmm(nsei_returns: &[f64]) -> Result<(HmmModel, RegimeState)> {
    if nsei_returns.len() < 3 * 30 {
        return Err(ComputeError::Model(format!(
            "fit_hmm needs at least {} observations for 3 states with a usable sample per \
             state, got {}",
            3 * 30,
            nsei_returns.len()
        )));
    }

    let (mut means, mut variances) = kmeans_init(nsei_returns);
    let mut initial = [1.0 / N_STATES as f64; N_STATES];
    let mut transition = [[1.0 / N_STATES as f64; N_STATES]; N_STATES];

    let mut prev_ll = f64::NEG_INFINITY;
    let mut n_iter = 0u32;
    let mut final_ll = f64::NEG_INFINITY;

    for iter in 1..=MAX_ITER {
        let (alpha, c) = forward(nsei_returns, &initial, &transition, &means, &variances);
        let ll: f64 = c.iter().map(|v| v.ln()).sum();
        let beta = backward(nsei_returns, &transition, &means, &variances, &c);
        let e = e_step(nsei_returns, &alpha, &beta, &transition, &means, &variances, &c);
        let (new_initial, new_transition, new_means, new_variances) = m_step(nsei_returns, &e);

        n_iter = iter;
        final_ll = ll;
        let improved = ll - prev_ll;
        initial = new_initial;
        transition = new_transition;
        means = new_means;
        variances = new_variances;

        if iter > 1 && improved.abs() < LOG_LIKELIHOOD_TOLERANCE {
            break;
        }
        prev_ll = ll;
    }

    // Recompute with the final converged parameters so log_likelihood,
    // gamma (for smoothed_probs), and the Viterbi path all reflect the
    // same model that's actually being returned.
    let (alpha, c) = forward(nsei_returns, &initial, &transition, &means, &variances);
    final_ll = c.iter().map(|v| v.ln()).sum::<f64>().max(final_ll.min(f64::MAX));
    let beta = backward(nsei_returns, &transition, &means, &variances, &c);
    let e = e_step(nsei_returns, &alpha, &beta, &transition, &means, &variances, &c);
    let viterbi_path = viterbi(nsei_returns, &initial, &transition, &means, &variances);

    // Relabel by ascending variance.
    let mut order: [usize; N_STATES] = [0, 1, 2];
    order.sort_by(|&a, &b| variances[a].partial_cmp(&variances[b]).unwrap());
    // rank[old_index] = new_index
    let mut rank = [0usize; N_STATES];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        rank[old_idx] = new_idx;
    }

    let sorted_means: [f64; N_STATES] = std::array::from_fn(|new_k| means[order[new_k]]);
    let sorted_variances: [f64; N_STATES] = std::array::from_fn(|new_k| variances[order[new_k]]);
    let sorted_initial: [f64; N_STATES] = std::array::from_fn(|new_k| initial[order[new_k]]);
    let sorted_transition: [[f64; N_STATES]; N_STATES] =
        std::array::from_fn(|new_j| std::array::from_fn(|new_k| transition[order[new_j]][order[new_k]]));

    let t_n = nsei_returns.len();
    // `order[new_k] = old_k`, so this reindexes the final gamma row from
    // old (fit-time) state indices into the sorted Bull/Bear/Crisis order.
    let smoothed_probs: [f64; N_STATES] = std::array::from_fn(|new_k| e.gamma[t_n - 1][order[new_k]]);

    let viterbi_sequence: Vec<u8> = viterbi_path.iter().map(|&old| rank[old] as u8).collect();
    let mut obs_count_per_regime = [0usize; N_STATES];
    for &s in &viterbi_sequence {
        obs_count_per_regime[s as usize] += 1;
    }

    let current_regime = (0..N_STATES)
        .max_by(|&a, &b| smoothed_probs[a].partial_cmp(&smoothed_probs[b]).unwrap())
        .unwrap() as u8;

    let model = HmmModel {
        initial: sorted_initial,
        transition: sorted_transition,
        means: sorted_means,
        variances: sorted_variances,
        log_likelihood: final_ll,
        n_iter,
    };

    let state = RegimeState {
        current_regime,
        current_label: REGIME_LABELS[current_regime as usize],
        smoothed_probs,
        viterbi_sequence,
        obs_count_per_regime,
        log_likelihood: final_ll,
        n_iter,
        smoothing_note: SMOOTHING_NOTE,
    };

    Ok((model, state))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_returns(seed: u64) -> Vec<f64> {
        // Deterministic xorshift PRNG, matching compute's other tests'
        // pattern (no `rand` dependency).
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> f64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                let unit = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
                unit * 2.0 - 1.0
            }
        }
        let mut rng = Rng(seed.max(1));
        let mut out = Vec::with_capacity(300);
        for _ in 0..100 {
            out.push(rng.next() * 0.003); // low vol
        }
        for _ in 0..100 {
            out.push(rng.next() * 0.012); // medium vol
        }
        for _ in 0..100 {
            out.push(rng.next() * 0.035); // high vol
        }
        out
    }

    #[test]
    fn gamma_sums_to_one_at_every_t() {
        let data = synthetic_returns(1);
        let (model, _state) = fit_hmm(&data).unwrap();
        let (alpha, c) = forward(&data, &model.initial, &model.transition, &model.means, &model.variances);
        let beta = backward(&data, &model.transition, &model.means, &model.variances, &c);
        for t in 0..data.len() {
            let sum: f64 = (0..N_STATES).map(|k| alpha[t][k] * beta[t][k]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "gamma at t={t} sums to {sum}, expected ~1.0"
            );
        }
    }

    #[test]
    fn viterbi_recovers_synthetic_segments_with_over_85_percent_accuracy() {
        let data = synthetic_returns(2);
        let (_model, state) = fit_hmm(&data).unwrap();

        // Segment 0..100 should be mostly Bull (0), 100..200 mostly Bear
        // (1), 200..300 mostly Crisis (2).
        for (segment, expected) in [(0..100, 0u8), (100..200, 1u8), (200..300, 2u8)] {
            let correct = segment.clone().filter(|&i| state.viterbi_sequence[i] == expected).count();
            let accuracy = correct as f64 / segment.len() as f64;
            assert!(
                accuracy > 0.85,
                "segment {segment:?} (expected regime {expected}) only {:.1}% correctly assigned",
                accuracy * 100.0
            );
        }
    }

    #[test]
    fn state_labels_are_variance_ordered_regardless_of_segment_order_in_the_data() {
        // Same three variance levels, different orderings in time: labels
        // must still come out Bull (lowest var) < Bear < Crisis (highest
        // var) regardless of which chunk appears first/last/middle.
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> f64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                let unit = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
                unit * 2.0 - 1.0
            }
        }
        let mut rng = Rng(42);
        let low: Vec<f64> = (0..100).map(|_| rng.next() * 0.003).collect();
        let med: Vec<f64> = (0..100).map(|_| rng.next() * 0.012).collect();
        let high: Vec<f64> = (0..100).map(|_| rng.next() * 0.035).collect();

        // Order: high, low, medium (deliberately not sorted by variance).
        let mut data = Vec::new();
        data.extend_from_slice(&high);
        data.extend_from_slice(&low);
        data.extend_from_slice(&med);

        let (model, _state) = fit_hmm(&data).unwrap();
        assert!(
            model.variances[0] < model.variances[1] && model.variances[1] < model.variances[2],
            "variances not ascending: {:?}",
            model.variances
        );
    }

    #[test]
    fn errors_on_too_few_observations() {
        let data = vec![0.001; 10];
        assert!(fit_hmm(&data).is_err());
    }
}
