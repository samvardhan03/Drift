//! Data layer: fetches daily adjusted-close series from Yahoo Finance,
//! caches them to disk, aligns them onto the NSE trading calendar, and
//! derives the daily- or weekly-frequency log-return series used by the
//! factor model (see `to_returns`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use serde::Deserialize;

use crate::error::{ComputeError, Result};

/// Market factor ticker (Nifty 50).
pub const MARKET: &str = "^NSEI";
/// USD/INR ticker.
pub const USDINR: &str = "INR=X";
/// Brent crude futures ticker.
pub const BRENT: &str = "BZ=F";
/// Gold futures ticker, quoted in USD (hence the `GOLD_USD` factor label —
/// it is *not* INR-denominated gold; see `to_returns`'s doc for the
/// derived INR-gold move some experiments compute).
pub const GOLD: &str = "GC=F";
/// Nifty Bank ticker, used only to build the rates proxy factor.
pub const BANK: &str = "^NSEBANK";

/// Synthetic factor label: NIFTY BANK return minus NIFTY 50 return, used as
/// a proxy for rate-sensitivity exposure. This label (not `^NSEBANK`) is
/// what appears in betas, factor covariance, and all trace output.
pub const RATES_PROXY: &str = "RATES_PROXY";

/// Fixed, ordered list of factor names as they appear in every beta vector,
/// factor covariance matrix, and trace. Order matters: index `k` here is
/// index `k` everywhere else in the model.
pub const FACTOR_NAMES: [&str; 5] = ["MARKET", "USDINR", "BRENT", "GOLD_USD", RATES_PROXY];

/// Tickers that must be fetched to build the five named factors above.
/// (RATES_PROXY is derived, not fetched directly.)
pub const FACTOR_TICKERS: [&str; 5] = [MARKET, USDINR, BRENT, GOLD, BANK];

/// Maximum number of consecutive trading days a non-NSE series may be
/// forward-filled across a gap in the master (NSEI) calendar. Dates beyond
/// this are dropped for that series rather than filled.
pub const MAX_FORWARD_FILL_DAYS: usize = 3;

/// A raw price series as fetched (or loaded from cache): dates paired with
/// adjusted close prices, ascending by date.
#[derive(Debug, Clone)]
pub struct PriceSeries {
    pub ticker: String,
    pub dates: Vec<NaiveDate>,
    pub closes: Vec<f64>,
}

/// Per-series data-quality accounting, surfaced in every Evidence Trace.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SeriesQuality {
    pub ticker: String,
    /// Raw observations returned by the source (or cache) before alignment.
    pub raw_observations: usize,
    /// Number of master-calendar dates filled forward for this series.
    pub forward_filled_days: usize,
    /// Number of master-calendar dates dropped for this series because the
    /// gap exceeded `MAX_FORWARD_FILL_DAYS`.
    pub dropped_days: usize,
}

/// Aggregate data-quality report for a loaded dataset.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct DataQuality {
    pub date_range_start: NaiveDate,
    pub date_range_end: NaiveDate,
    pub trading_days: usize,
    pub per_series: Vec<SeriesQuality>,
}

/// Aligned, return-space dataset ready for factor-model fitting.
///
/// `dates` is the master NSE calendar at the chosen `frequency` (length =
/// returns rows + 1, since a return needs a prior price): every daily
/// trading day for `Frequency::Daily`, or the last NSE trading day of each
/// week for `Frequency::Weekly`. `stock_returns` and `factor_returns` are
/// keyed by ticker (stocks) / factor name (factors) and are all the same
/// length, aligned to `dates[1..]`.
pub struct MarketData {
    pub dates: Vec<NaiveDate>,
    pub stock_returns: BTreeMap<String, Vec<f64>>,
    pub factor_returns: BTreeMap<String, Vec<f64>>,
    pub quality: DataQuality,
}

/// Daily-aligned adjusted closes for every requested stock and factor
/// ticker, on the NSE (`^NSEI`) master calendar, forward-filled per
/// `align_to_calendar`. This is the frequency-independent intermediate:
/// `MarketData` (daily or weekly log returns) is derived from it by
/// `to_returns`.
pub struct AlignedPrices {
    pub dates: Vec<NaiveDate>,
    pub stock_closes: BTreeMap<String, Vec<f64>>,
    pub factor_closes: BTreeMap<String, Vec<f64>>,
    pub quality: DataQuality,
}

fn sanitize_for_filename(ticker: &str) -> String {
    ticker
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn cache_path(cache_dir: &Path, ticker: &str) -> PathBuf {
    cache_dir.join(format!("{}.csv", sanitize_for_filename(ticker)))
}

fn write_cache(cache_dir: &Path, series: &PriceSeries) -> Result<()> {
    std::fs::create_dir_all(cache_dir)?;
    let path = cache_path(cache_dir, &series.ticker);
    let mut writer = csv::Writer::from_path(&path)?;
    writer.write_record(["date", "close"])?;
    for (d, c) in series.dates.iter().zip(series.closes.iter()) {
        writer.write_record([d.format("%Y-%m-%d").to_string(), c.to_string()])?;
    }
    writer.flush()?;
    Ok(())
}

fn read_cache(cache_dir: &Path, ticker: &str) -> Result<Option<PriceSeries>> {
    let path = cache_path(cache_dir, ticker);
    if !path.exists() {
        return Ok(None);
    }
    let mut reader = csv::Reader::from_path(&path)?;
    let mut dates = Vec::new();
    let mut closes = Vec::new();
    for result in reader.records() {
        let record = result?;
        let date = NaiveDate::parse_from_str(&record[0], "%Y-%m-%d")
            .map_err(|e| ComputeError::Data(format!("bad cached date for {ticker}: {e}")))?;
        let close: f64 = record[1]
            .parse()
            .map_err(|e| ComputeError::Data(format!("bad cached close for {ticker}: {e}")))?;
        dates.push(date);
        closes.push(close);
    }
    Ok(Some(PriceSeries {
        ticker: ticker.to_string(),
        dates,
        closes,
    }))
}

#[derive(Debug, Deserialize)]
struct YahooChartResponse {
    chart: YahooChart,
}

#[derive(Debug, Deserialize)]
struct YahooChart {
    result: Option<Vec<YahooResult>>,
    error: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct YahooResult {
    timestamp: Option<Vec<i64>>,
    indicators: YahooIndicators,
}

#[derive(Debug, Deserialize)]
struct YahooIndicators {
    adjclose: Option<Vec<YahooAdjClose>>,
    quote: Vec<YahooQuote>,
}

#[derive(Debug, Deserialize)]
struct YahooAdjClose {
    adjclose: Vec<Option<f64>>,
}

#[derive(Debug, Deserialize)]
struct YahooQuote {
    close: Vec<Option<f64>>,
}

/// Fetches a full daily adjusted-close history for `ticker` from Yahoo
/// Finance's chart endpoint. Uses a 5-year range at daily interval, which
/// comfortably covers any trailing window the factor model uses.
pub fn fetch_yahoo_chart(ticker: &str) -> Result<PriceSeries> {
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?range=5y&interval=1d",
        urlencode_ticker(ticker)
    );
    let client = reqwest::blocking::Client::builder()
        .user_agent("drift-risk-copilot/0.1 (+compute-data-layer)")
        .build()?;
    let resp: YahooChartResponse = client.get(&url).send()?.error_for_status()?.json()?;

    if let Some(err) = resp.chart.error {
        if !err.is_null() {
            return Err(ComputeError::Data(format!(
                "yahoo chart error for {ticker}: {err}"
            )));
        }
    }

    let result = resp
        .chart
        .result
        .and_then(|mut r| if r.is_empty() { None } else { Some(r.remove(0)) })
        .ok_or_else(|| ComputeError::Data(format!("no chart result for {ticker}")))?;

    let timestamps = result
        .timestamp
        .ok_or_else(|| ComputeError::Data(format!("no timestamps for {ticker}")))?;

    let closes: Vec<Option<f64>> = if let Some(adj) = result.indicators.adjclose {
        adj.into_iter()
            .next()
            .map(|a| a.adjclose)
            .ok_or_else(|| ComputeError::Data(format!("no adjclose for {ticker}")))?
    } else {
        result
            .indicators
            .quote
            .into_iter()
            .next()
            .map(|q| q.close)
            .ok_or_else(|| ComputeError::Data(format!("no close for {ticker}")))?
    };

    let mut dates = Vec::new();
    let mut prices = Vec::new();
    for (ts, close) in timestamps.into_iter().zip(closes) {
        if let Some(c) = close {
            let date = chrono::DateTime::from_timestamp(ts, 0)
                .ok_or_else(|| ComputeError::Data(format!("bad timestamp for {ticker}")))?
                .date_naive();
            dates.push(date);
            prices.push(c);
        }
    }

    Ok(PriceSeries {
        ticker: ticker.to_string(),
        dates,
        closes: prices,
    })
}

fn urlencode_ticker(ticker: &str) -> String {
    // Tickers only ever contain a small, known character set (letters,
    // digits, '.', '^', '='), so a minimal manual encoding is sufficient
    // and avoids pulling in a URL-encoding dependency for one call site.
    ticker.replace('^', "%5E").replace('=', "%3D")
}

/// Loads (from cache, or fetching + caching on `refresh`/cache-miss) the raw
/// price series for `ticker`.
pub fn load_series(cache_dir: &Path, ticker: &str, refresh: bool) -> Result<PriceSeries> {
    if !refresh {
        if let Some(cached) = read_cache(cache_dir, ticker)? {
            return Ok(cached);
        }
    }
    let fetched = fetch_yahoo_chart(ticker)?;
    write_cache(cache_dir, &fetched)?;
    Ok(fetched)
}

/// Aligns a raw series onto `calendar` (ascending, deduplicated master
/// dates), forward-filling gaps of up to `MAX_FORWARD_FILL_DAYS` calendar
/// slots and dropping the series' value on dates beyond that. Returns the
/// aligned closes (`None` where dropped) plus the fill/drop counts.
fn align_to_calendar(
    series: &PriceSeries,
    calendar: &[NaiveDate],
) -> (Vec<Option<f64>>, usize, usize) {
    let mut by_date: BTreeMap<NaiveDate, f64> = BTreeMap::new();
    for (d, c) in series.dates.iter().zip(series.closes.iter()) {
        by_date.insert(*d, *c);
    }

    let mut aligned = Vec::with_capacity(calendar.len());
    let mut last_value: Option<f64> = None;
    let mut gap_len = 0usize;
    let mut fills = 0usize;
    let mut drops = 0usize;

    for date in calendar {
        if let Some(v) = by_date.get(date) {
            aligned.push(Some(*v));
            last_value = Some(*v);
            gap_len = 0;
        } else {
            gap_len += 1;
            if gap_len <= MAX_FORWARD_FILL_DAYS {
                if let Some(v) = last_value {
                    aligned.push(Some(v));
                    fills += 1;
                } else {
                    aligned.push(None);
                    drops += 1;
                }
            } else {
                aligned.push(None);
                drops += 1;
            }
        }
    }

    (aligned, fills, drops)
}

fn log_returns(prices: &[f64]) -> Vec<f64> {
    prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect()
}

/// Loads and aligns daily adjusted closes for everything needed to fit the
/// factor model: the given stock tickers (holdings) plus the five fixed
/// factor tickers, on the NSE (`^NSEI`) trading calendar, forward-filled
/// per `align_to_calendar`. Frequency-independent; see `to_returns` for the
/// daily/weekly log-return derivation.
pub fn load_aligned_prices(
    cache_dir: &Path,
    stock_tickers: &[String],
    refresh: bool,
) -> Result<AlignedPrices> {
    let market_series = load_series(cache_dir, MARKET, refresh)?;
    let mut calendar = market_series.dates.clone();
    calendar.sort();
    calendar.dedup();

    let mut per_series_quality = Vec::new();
    let mut aligned_closes: BTreeMap<String, Vec<Option<f64>>> = BTreeMap::new();

    let mut all_tickers: Vec<String> = stock_tickers.to_vec();
    for t in FACTOR_TICKERS {
        all_tickers.push(t.to_string());
    }

    for ticker in &all_tickers {
        let series = if ticker == MARKET {
            market_series.clone()
        } else {
            load_series(cache_dir, ticker, refresh)?
        };
        let (aligned, fills, drops) = align_to_calendar(&series, &calendar);
        per_series_quality.push(SeriesQuality {
            ticker: ticker.clone(),
            raw_observations: series.dates.len(),
            forward_filled_days: fills,
            dropped_days: drops,
        });
        aligned_closes.insert(ticker.clone(), aligned);
    }

    // Trim leading calendar dates where any series is still None (before its
    // first observed price) so every series has a valid price for the whole
    // remaining window.
    let mut start_idx = 0usize;
    'outer: for i in 0..calendar.len() {
        for closes in aligned_closes.values() {
            if closes[i].is_none() {
                start_idx = i + 1;
                continue 'outer;
            }
        }
        break;
    }
    let calendar = calendar[start_idx..].to_vec();
    for closes in aligned_closes.values_mut() {
        *closes = closes[start_idx..].to_vec();
    }

    let mut stock_closes = BTreeMap::new();
    for ticker in stock_tickers {
        let closes = aligned_closes
            .get(ticker)
            .ok_or_else(|| ComputeError::Data(format!("missing series for {ticker}")))?;
        let closes: Vec<f64> = closes
            .iter()
            .map(|c| c.expect("trimmed to a common non-None prefix"))
            .collect();
        stock_closes.insert(ticker.clone(), closes);
    }

    let mut factor_closes = BTreeMap::new();
    for ticker in FACTOR_TICKERS {
        let closes = aligned_closes
            .get(ticker)
            .ok_or_else(|| ComputeError::Data(format!("missing factor series for {ticker}")))?;
        let closes: Vec<f64> = closes
            .iter()
            .map(|c| c.expect("trimmed to a common non-None prefix"))
            .collect();
        factor_closes.insert(ticker.to_string(), closes);
    }

    let quality = DataQuality {
        date_range_start: *calendar.first().unwrap(),
        date_range_end: *calendar.last().unwrap(),
        trading_days: calendar.len(),
        per_series: per_series_quality,
    };

    Ok(AlignedPrices {
        dates: calendar,
        stock_closes,
        factor_closes,
        quality,
    })
}

/// Keeps only the last daily observation in each ISO (year, week) group,
/// i.e. the last NSE trading day of each week (usually Friday, but a
/// holiday-shortened week keeps whatever day is last). Non-overlapping by
/// construction: each week contributes exactly one point.
fn weekly_resample_indices(dates: &[NaiveDate]) -> Vec<usize> {
    use chrono::Datelike;
    let mut indices = Vec::new();
    for i in 0..dates.len() {
        let this_week = dates[i].iso_week();
        let is_last_of_week = match dates.get(i + 1) {
            Some(next) => next.iso_week() != this_week,
            None => true,
        };
        if is_last_of_week {
            indices.push(i);
        }
    }
    indices
}

/// Derives daily or (non-overlapping, weekly-resampled) log-return series
/// from `prices`, computing `RATES_PROXY` = NIFTY BANK return - NIFTY 50
/// return at that same frequency.
type CloseSeriesByTicker = BTreeMap<String, Vec<f64>>;

pub fn to_returns(prices: &AlignedPrices, frequency: crate::model::Frequency) -> MarketData {
    let (dates, stock_closes, factor_closes, quality): (
        Vec<NaiveDate>,
        CloseSeriesByTicker,
        CloseSeriesByTicker,
        DataQuality,
    ) = match frequency {
        crate::model::Frequency::Daily => (
            prices.dates.clone(),
            prices.stock_closes.clone(),
            prices.factor_closes.clone(),
            prices.quality.clone(),
        ),
        crate::model::Frequency::Weekly => {
            let idx = weekly_resample_indices(&prices.dates);
            let dates: Vec<NaiveDate> = idx.iter().map(|&i| prices.dates[i]).collect();
            let resample = |series: &Vec<f64>| -> Vec<f64> {
                idx.iter().map(|&i| series[i]).collect()
            };
            let stock_closes = prices
                .stock_closes
                .iter()
                .map(|(k, v)| (k.clone(), resample(v)))
                .collect();
            let factor_closes = prices
                .factor_closes
                .iter()
                .map(|(k, v)| (k.clone(), resample(v)))
                .collect();
            let quality = DataQuality {
                date_range_start: *dates.first().unwrap(),
                date_range_end: *dates.last().unwrap(),
                trading_days: dates.len(),
                per_series: prices.quality.per_series.clone(),
            };
            (dates, stock_closes, factor_closes, quality)
        }
    };

    let mut stock_returns = BTreeMap::new();
    for (ticker, closes) in &stock_closes {
        stock_returns.insert(ticker.clone(), log_returns(closes));
    }

    let market_ret = log_returns(&factor_closes[MARKET]);
    let usdinr_ret = log_returns(&factor_closes[USDINR]);
    let brent_ret = log_returns(&factor_closes[BRENT]);
    let gold_ret = log_returns(&factor_closes[GOLD]);
    let bank_ret = log_returns(&factor_closes[BANK]);

    let rates_proxy_ret: Vec<f64> = bank_ret
        .iter()
        .zip(market_ret.iter())
        .map(|(b, m)| b - m)
        .collect();

    let mut factor_returns = BTreeMap::new();
    factor_returns.insert("MARKET".to_string(), market_ret);
    factor_returns.insert("USDINR".to_string(), usdinr_ret);
    factor_returns.insert("BRENT".to_string(), brent_ret);
    factor_returns.insert("GOLD_USD".to_string(), gold_ret);
    factor_returns.insert(RATES_PROXY.to_string(), rates_proxy_ret);

    MarketData {
        dates,
        stock_returns,
        factor_returns,
        quality,
    }
}

/// Convenience wrapper: loads aligned daily prices and derives returns at
/// `frequency` in one call.
pub fn load_market_data(
    cache_dir: &Path,
    stock_tickers: &[String],
    refresh: bool,
    frequency: crate::model::Frequency,
) -> Result<MarketData> {
    let prices = load_aligned_prices(cache_dir, stock_tickers, refresh)?;
    Ok(to_returns(&prices, frequency))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weekly_resample_keeps_last_trading_day_of_each_week() {
        // Mon 2024-01-01 .. Fri 2024-01-05 (full week), then Mon 2024-01-08,
        // Tue 2024-01-09 (a holiday-shortened second week ending Tuesday).
        let dates = vec![
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 4).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 5).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 8).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 9).unwrap(),
        ];
        let idx = weekly_resample_indices(&dates);
        let kept: Vec<NaiveDate> = idx.iter().map(|&i| dates[i]).collect();
        assert_eq!(
            kept,
            vec![
                NaiveDate::from_ymd_opt(2024, 1, 5).unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 9).unwrap(),
            ]
        );
    }
}
