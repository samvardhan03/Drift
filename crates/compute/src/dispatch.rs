//! The single entry point that runs any `Experiment` end to end: data
//! fetch, model fit, and the appropriate experiment function, returning its
//! `EvidenceTrace`. Moved here (from `agent::pipeline::compute_trace`,
//! which is now a thin wrapper around this) because `RiskDrift` needs read
//! access to the snapshot store to resolve its baseline, and the `store`
//! crate is a dependency of `compute`, not of the (Gemini-facing) `agent`
//! crate's narrower concerns.

use std::path::Path;

use crate::context::ExperimentContext;
use crate::data::MarketData;
use crate::experiments::Experiment;
use crate::model::Frequency;
use crate::trace::{DataWindow, EvidenceTrace};
use crate::Result;

/// Where fetched price series are cached on disk.
pub const CACHE_DIR: &str = "data/cache";

fn build_data_window(data: &MarketData, window: usize, frequency: Frequency) -> DataWindow {
    DataWindow {
        frequency,
        window_periods: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    }
}

/// If `ctx.policy` is set, evaluates it against `portfolio`/`model`/`data`
/// and attaches the result to `trace.policy_result`; a no-op otherwise.
/// Used for the "passive" policy check (see `run_experiment`'s doc) --
/// `PolicyCheck` and `CvarRebalance` handle policy themselves and never
/// call this.
fn maybe_attach_policy(
    mut trace: EvidenceTrace,
    ctx: &ExperimentContext,
    portfolio: &crate::experiments::Portfolio,
    data: &MarketData,
    model: &crate::model::FactorModel,
) -> Result<EvidenceTrace> {
    if let Some(policy) = &ctx.policy {
        trace.policy_result = Some(crate::policy::evaluate_policy(policy, portfolio, model, data)?);
    }
    Ok(trace)
}

/// Runs `experiment` (data fetch + model fit + the appropriate experiment
/// function) and returns its `EvidenceTrace`. Synchronous/blocking --
/// callers on an async executor (e.g. `agent::pipeline::run`) should run it
/// via `tokio::task::spawn_blocking`.
///
/// `portfolio` is `experiment`'s portfolio, always -- for every variant
/// except `RiskDrift`/`ReverseStress`/`PolicyCheck`, this is the same
/// portfolio already embedded in the variant's own input
/// (`FactorShockInput::portfolio` etc.), so it's slightly redundant there;
/// those three variants' inputs carry no `portfolio` field of their own,
/// so this is their only source of one.
///
/// `ctx.policy`, when set, gets a **passive** policy check attached to
/// `trace.policy_result` for every variant except `PolicyCheck` (which
/// *is* the policy check) and `CvarRebalance` (which runs its own
/// before/after check as part of remediation -- see `cvar::run_cvar_rebalance`).
/// **Known gap**: `RiskDrift` fits its own "current" factor model
/// internally (`drift::run_risk_drift`) and doesn't expose it back to this
/// dispatcher, so the passive check is skipped for it -- a caller wanting
/// a policy check alongside `RiskDrift` should make a separate
/// `PolicyCheck` request instead.
pub fn run_experiment(
    experiment: &Experiment,
    portfolio: &crate::experiments::Portfolio,
    ctx: &ExperimentContext,
) -> Result<EvidenceTrace> {
    let (mut trace, data) = run_experiment_inner(experiment, portfolio, ctx)?;
    // `RiskDrift` fits its own models internally and doesn't hand back its
    // `MarketData`; reload it (from the disk cache) for the history only.
    let data = match data {
        Some(d) => Some(d),
        None => crate::data::load_market_data(
            Path::new(CACHE_DIR),
            &portfolio.tickers(),
            false,
            trace.model_params.frequency,
        )
        .ok(),
    };
    if let Some(data) = data {
        trace.portfolio_history = build_history(&trace, &data, portfolio);
    }
    Ok(trace)
}

/// `portfolio_history` for `trace`'s window, or `None` if no regime path
/// is available (see `portfolio_history::compute_portfolio_history`).
fn build_history(
    trace: &EvidenceTrace,
    data: &MarketData,
    portfolio: &crate::experiments::Portfolio,
) -> Option<crate::portfolio_history::PortfolioHistory> {
    use crate::portfolio_history as ph;
    let regime = ph::regime_for_history(
        trace.model_params.regime_state.as_ref(),
        data,
        trace.model_params.frequency.default_window(),
    )?;
    ph::compute_portfolio_history(data, portfolio, &regime, ph::DEFAULT_HISTORY_DAYS)
}

fn run_experiment_inner(
    experiment: &Experiment,
    portfolio: &crate::experiments::Portfolio,
    ctx: &ExperimentContext,
) -> Result<(EvidenceTrace, Option<MarketData>)> {
    let cache_dir = Path::new(CACHE_DIR);
    match experiment {
        Experiment::FactorShock(input) => {
            let tickers = input.portfolio.tickers();
            let window = input.resolved_window();
            let data = crate::data::load_market_data(cache_dir, &tickers, false, input.frequency)?;
            let model = crate::model::fit_factor_model(
                &data,
                &tickers,
                crate::model::ModelConfig::new(window, input.frequency),
            )?;
            let data_window = build_data_window(&data, window, input.frequency);
            let (_, trace) = crate::experiments::run_factor_shock(&data.quality, data_window, &data, &model, input)?;
            let trace = maybe_attach_policy(trace, ctx, &input.portfolio, &data, &model)?;
            Ok((trace, Some(data)))
        }
        Experiment::RiskDecomposition(input) => {
            let tickers = input.portfolio.tickers();
            let window = input.resolved_window();
            let data = crate::data::load_market_data(cache_dir, &tickers, false, input.frequency)?;
            let model = crate::model::fit_factor_model(
                &data,
                &tickers,
                crate::model::ModelConfig::new(window, input.frequency),
            )?;
            let data_window = build_data_window(&data, window, input.frequency);
            let (_, trace) =
                crate::experiments::run_risk_decomposition(&data.quality, data_window, &data, &model, input)?;
            let trace = maybe_attach_policy(trace, ctx, &input.portfolio, &data, &model)?;
            Ok((trace, Some(data)))
        }
        Experiment::CvarRebalance(input) => {
            let tickers = input.portfolio.tickers();
            let data = crate::data::load_market_data(cache_dir, &tickers, false, input.frequency)?;
            let (_, trace) = crate::cvar::run_cvar_rebalance(&data.quality, &data, input)?;
            Ok((trace, Some(data)))
        }
        Experiment::PortfolioPerformance(input) => {
            let tickers = input.portfolio.tickers();
            let window = input.resolved_window();
            let data = crate::data::load_market_data(cache_dir, &tickers, false, input.frequency)?;
            let data_window = build_data_window(&data, window, input.frequency);
            let (_, trace) =
                crate::performance::run_portfolio_performance(&data.quality, data_window, &data, input)?;
            if ctx.policy.is_some() {
                // PortfolioPerformance never otherwise fits a factor model;
                // one is fit here purely for the passive policy check, so
                // it's skipped entirely (no extra fit/cost) when no policy
                // is attached.
                let model = crate::model::fit_factor_model(
                    &data,
                    &tickers,
                    crate::model::ModelConfig::new(window, input.frequency),
                )?;
                let trace = maybe_attach_policy(trace, ctx, &input.portfolio, &data, &model)?;
                Ok((trace, Some(data)))
            } else {
                Ok((trace, Some(data)))
            }
        }
        Experiment::RiskDrift(input) => {
            let (_, trace) = crate::drift::run_risk_drift(cache_dir, portfolio, input, ctx)?;
            Ok((trace, None))
        }
        Experiment::ReverseStress(input) => {
            let tickers = portfolio.tickers();
            let frequency = crate::reverse_stress::resolved_frequency(input)?;
            let window = input.window.unwrap_or_else(|| frequency.default_window());
            let data = crate::data::load_market_data(cache_dir, &tickers, false, frequency)?;
            let model = crate::model::fit_factor_model(&data, &tickers, crate::model::ModelConfig::new(window, frequency))?;
            let data_window = build_data_window(&data, window, frequency);
            let (_, trace) =
                crate::reverse_stress::run_reverse_stress(&data.quality, data_window, &model, portfolio, input)?;
            let trace = maybe_attach_policy(trace, ctx, portfolio, &data, &model)?;
            Ok((trace, Some(data)))
        }
        Experiment::PolicyCheck(input) => {
            let tickers = portfolio.tickers();
            let frequency = crate::model::Frequency::from_optional_str(input.frequency.as_deref())?;
            let window = input.window.unwrap_or_else(|| frequency.default_window());
            let data = crate::data::load_market_data(cache_dir, &tickers, false, frequency)?;
            let model = crate::model::fit_factor_model(&data, &tickers, crate::model::ModelConfig::new(window, frequency))?;
            let data_window = build_data_window(&data, window, frequency);
            let (_, trace) =
                crate::policy::run_policy_check(&data.quality, data_window, &data, &model, portfolio, input)?;
            Ok((trace, Some(data)))
        }
    }
}
