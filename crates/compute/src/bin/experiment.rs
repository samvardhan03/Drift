use std::path::PathBuf;

use clap::Parser;
use compute::cvar::run_cvar_rebalance;
use compute::data::load_market_data;
use compute::experiments::{run_factor_shock, run_risk_decomposition, Experiment};
use compute::model::{fit_factor_model, ModelConfig};
use compute::performance::run_portfolio_performance;
use compute::trace::DataWindow;

/// Runs a single experiment (FactorShock, RiskDecomposition, or
/// CvarRebalance) described by a JSON file and prints its Evidence Trace to
/// stdout.
#[derive(Parser, Debug)]
struct Args {
    /// Path to a JSON file containing a tagged `Experiment` value.
    input: PathBuf,

    /// Directory used to cache fetched price series.
    #[arg(long, default_value = "data/cache")]
    cache_dir: PathBuf,

    /// Refetch price series instead of reading from cache.
    #[arg(long)]
    refresh: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let raw = std::fs::read_to_string(&args.input)?;
    let experiment: Experiment = serde_json::from_str(&raw)?;

    if matches!(experiment, Experiment::RiskDrift(_)) {
        return Err("RiskDrift needs a running SnapshotStore with prior experiment history to \
                     diff against, which this one-shot CLI doesn't provide -- run it via the \
                     server's POST /experiment instead."
            .into());
    }
    if matches!(experiment, Experiment::ReverseStress(_) | Experiment::PolicyCheck(_)) {
        return Err("ReverseStress/PolicyCheck have no \"portfolio\" field of their own (the \
                     portfolio is a separate parameter at the server/dispatch layer, not part \
                     of the JSON tagged-union input this CLI deserializes) -- run them via the \
                     server's POST /experiment instead."
            .into());
    }

    if let Experiment::CvarRebalance(input) = &experiment {
        let tickers = input.portfolio.tickers();
        let data = load_market_data(&args.cache_dir, &tickers, args.refresh, input.frequency)?;
        let (_, trace) = run_cvar_rebalance(&data.quality, &data, input)?;
        println!("{}", serde_json::to_string_pretty(&trace)?);
        return Ok(());
    }

    if let Experiment::PortfolioPerformance(input) = &experiment {
        let tickers = input.portfolio.tickers();
        let window = input.resolved_window();
        let data = load_market_data(&args.cache_dir, &tickers, args.refresh, input.frequency)?;
        let data_window = DataWindow {
            frequency: input.frequency,
            window_periods: window,
            start: data.dates[data.dates.len() - window],
            end: *data.dates.last().unwrap(),
        };
        let (_, trace) = run_portfolio_performance(&data.quality, data_window, &data, input)?;
        println!("{}", serde_json::to_string_pretty(&trace)?);
        return Ok(());
    }

    let (portfolio, window, frequency) = match &experiment {
        Experiment::FactorShock(i) => (&i.portfolio, i.resolved_window(), i.frequency),
        Experiment::RiskDecomposition(i) => (&i.portfolio, i.resolved_window(), i.frequency),
        Experiment::CvarRebalance(_) | Experiment::PortfolioPerformance(_) | Experiment::RiskDrift(_) | Experiment::ReverseStress(_) | Experiment::PolicyCheck(_) => {
            unreachable!("handled above")
        }
    };

    let tickers = portfolio.tickers();
    let data = load_market_data(&args.cache_dir, &tickers, args.refresh, frequency)?;
    let model = fit_factor_model(&data, &tickers, ModelConfig::new(window, frequency))?;

    let data_window = DataWindow {
        frequency,
        window_periods: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    };

    let trace = match &experiment {
        Experiment::FactorShock(input) => {
            let (_, trace) = run_factor_shock(&data.quality, data_window, &model, input)?;
            trace
        }
        Experiment::RiskDecomposition(input) => {
            let (_, trace) = run_risk_decomposition(&data.quality, data_window, &model, input)?;
            trace
        }
        Experiment::CvarRebalance(_) | Experiment::PortfolioPerformance(_) | Experiment::RiskDrift(_) | Experiment::ReverseStress(_) | Experiment::PolicyCheck(_) => {
            unreachable!("handled above")
        }
    };

    println!("{}", serde_json::to_string_pretty(&trace)?);
    Ok(())
}
