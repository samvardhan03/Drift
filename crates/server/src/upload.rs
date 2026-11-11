//! `POST /portfolio/upload`: turns a CSV or XLSX file into a `Portfolio`
//! the caller can hand straight to `/experiment` or `/ask`.
//!
//! Column detection recognises the header conventions of 9 major Indian
//! brokers' holdings exports (not just this codebase's own canonical
//! `ticker`/`weight`/`shares`/`avg_price_inr` names) -- known formats, for
//! reference and as what this module's tests are built against:
//!
//! | Broker            | Ticker column        | Quantity column       | Price column      | Notes                              |
//! |--------------------|----------------------|------------------------|--------------------|-------------------------------------|
//! | Zerodha (Kite/Console) | `Instrument`      | `Qty.`                 | `Avg. cost`        | ~3 metadata rows above the header   |
//! | Upstox             | `Symbol`/`trading_symbol` | `Quantity`/`Quantity Available` | `Average Price` | tickers may carry an `NSE:` prefix |
//! | Groww              | `Symbol`              | `Units`                | `Average Buy Price` |                                    |
//! | Angel One          | `Symbol`/`Scrip Name` | `Net Quantity`         | `Avg. Buy Price`   |                                     |
//! | HDFC Securities    | `Symbol`              | `Quantity`             | `Avg Rate`         | multiple title rows above header    |
//! | ICICI Direct       | `Stock Name`          | `Quantity`             | `Average Rate`     | report header rows above data       |
//! | 5Paisa             | `Symbol`              | `Qty`                  | `Avg Price`        |                                     |
//! | Motilal Oswal      | `Scrip Name`          | `Qty`                  | `Buy Avg Price`    |                                     |
//! | Kotak Securities   | `Scrip`               | `Quantity`             | `Average Price`    |                                     |

use std::collections::BTreeMap;
use std::io::Cursor;

use axum::extract::{Multipart, State};
use calamine::{open_workbook_from_rs, Data, DataType, Reader, Xlsx};
use compute::experiments::{Holding, Portfolio};
use serde::Serialize;

use crate::error::ApiError;
use crate::routes::AppState;
use crate::validate::validate_portfolio;

/// Weight-based CSV/XLSX uploads carry no portfolio value at all (only
/// ticker + weight), so there is nothing to derive `total_value_inr` from.
/// Defaults to this codebase's existing demo convention (the same value
/// `sample_portfolio` helpers across the test suite use) — callers that
/// care about the real value should overwrite it in the returned
/// `Portfolio` before using it.
const DEFAULT_TOTAL_VALUE_INR: f64 = 1_000_000.0;

/// How many leading rows the header-row scan checks before giving up (see
/// `find_header_row_index`).
const MAX_HEADER_SCAN_ROWS: usize = 25;

// Alias lists are matched in the order given -- `find_column` tries each
// alias against the header row in turn, so when a file's headers happen to
// match more than one alias (unlikely, but e.g. a file with both "symbol"
// and "ticker" columns), the earlier-listed alias wins.

const TICKER_ALIASES: &[&str] = &[
    "ticker",
    "symbol",
    "instrument",
    "stock",
    "trading_symbol",
    "trading symbol",
    "scrip",
    "script",
    "isin",
    "isin code",
    "stock name",
    "company name",
    "name",
    "security name",
    "security",
    "share name",
    "scrip name",
    "stock symbol",
    "nse symbol",
    "bse symbol",
    "equity",
];

const QUANTITY_ALIASES: &[&str] = &[
    "quantity available",
    "qty",
    "qty.",
    "quantity",
    "net quantity",
    "net qty",
    "net qty.",
    "shares",
    "units",
    "holdings",
    "volume",
    "quantity long term",
    "long term quantity",
    "free quantity",
    "saleable quantity",
    "closing balance",
    "balance quantity",
    "total quantity",
    "gross quantity",
];

/// Includes `wap`/`weighted average price`/etc: some exports carry the
/// current market price rather than the original purchase price. Using it
/// still produces an internally-consistent *current* weight split (qty *
/// price / total) -- it just means the derived `total_value_inr` reflects
/// today's value rather than cost basis, which is arguably the more
/// useful number for a risk tool anyway.
const PRICE_ALIASES: &[&str] = &[
    "avg_price_inr",
    "avg. cost",
    "avg cost",
    "average cost",
    "average price",
    "avg price",
    "avg. price",
    "avg_price",
    "average_price",
    "avg rate",
    "average rate",
    "average buy price",
    "avg. buy price",
    "avg buy price",
    "buy avg",
    // Motilal Oswal's documented export uses this exact phrase; not in
    // the originally-specified alias list, added so that broker's own
    // stated format actually parses (see this session's report).
    "buy avg price",
    "avg. buy rate",
    "cost price",
    "purchase price",
    "buy price",
    "avg purchase price",
    "average purchase price",
    "weighted average price",
    "wap",
    "avg. cost price",
    "cost per share",
    "book value per share",
    "book value",
];

const WEIGHT_ALIASES: &[&str] = &[
    "weight",
    "weight%",
    "weight %",
    "allocation",
    "allocation%",
    "allocation %",
    "portfolio weight",
    "portfolio %",
    "%",
    "percentage",
    "percent",
];

/// Row-skip values (see `should_skip_row`), beyond "empty" and "starts
/// with a digit" which get their own checks.
const SKIP_ROW_VALUES: &[&str] = &[
    "total",
    "grand total",
    "sub total",
    "subtotal",
    "net total",
    "overall total",
    // A broker's footer sometimes repeats the header row verbatim.
    "instrument",
    "symbol",
    "stock",
    "-",
    "--",
    "n/a",
    "na",
    "nil",
];

#[derive(Debug, Serialize)]
pub struct UploadResponse {
    pub portfolio: Portfolio,
    pub layout_detected: &'static str,
    /// `"RELIANCE" -> "RELIANCE.NS"` for every ticker that got normalised
    /// (suffix appended, exchange prefix/"-EQ" stripped, or flagged as an
    /// ISIN needing manual mapping); empty if none needed it.
    pub tickers_normalised: Vec<String>,
    pub row_count: usize,
    /// Holdings dropped because their computed weight was 0.0 or below
    /// 1e-6 after normalisation -- not returned in `portfolio.holdings`.
    pub skipped_zero_weight: usize,
    /// `ISIN -> resolved symbol` for every ISIN in the file that was
    /// converted to an NSE/BSE symbol; empty for a symbol-based file.
    pub isins_resolved: BTreeMap<String, String>,
    /// Holdings left out of `portfolio`, each with a user-facing reason and
    /// a category (`delisted`/`non_equity`/`sme`/`not_found`/`timeout`).
    pub isins_skipped: Vec<compute::isin::IsinSkipReason>,
    /// True when skipped holdings forced the remaining weights to be
    /// rescaled to sum to 1 (and, for value-based files, the total value
    /// to shrink to the retained holdings' value).
    pub weights_renormalized: bool,
    /// Holdings in the file (after dropping zero-weight rows), before any
    /// ISIN was skipped.
    pub original_holding_count: usize,
    pub retained_holding_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Layout {
    Weight,
    Value,
}

type Record = BTreeMap<String, String>;

/// Resolved column names (as they actually appear in this file's header
/// row) for each role the parser understands, plus which `Layout` that
/// implies. `quantity`/`price` are `None` under `Layout::Weight` and vice
/// versa for `weight` under `Layout::Value`.
struct ColumnMap {
    ticker: String,
    quantity: Option<String>,
    price: Option<String>,
    weight: Option<String>,
    layout: Layout,
}

pub async fn post_portfolio_upload(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<axum::Json<UploadResponse>, ApiError> {
    let mut filename: Option<String> = None;
    let mut bytes: Option<Vec<u8>> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::bad_request("invalid_upload", format!("malformed multipart body: {e}")))?
    {
        if field.name() == Some("file") {
            filename = field.file_name().map(|s| s.to_string());
            let data = field
                .bytes()
                .await
                .map_err(|e| ApiError::bad_request("invalid_upload", format!("failed to read file bytes: {e}")))?;
            bytes = Some(data.to_vec());
        }
    }

    let filename =
        filename.ok_or_else(|| ApiError::bad_request("invalid_upload", "missing a \"file\" form field"))?;
    let bytes = bytes.ok_or_else(|| ApiError::bad_request("invalid_upload", "missing a \"file\" form field"))?;

    if bytes.is_empty() {
        return Err(ApiError::bad_request("empty_file", "uploaded file is empty"));
    }

    let lower_name = filename.to_lowercase();
    let rows = if lower_name.ends_with(".csv") {
        read_csv_rows(&bytes)?
    } else if lower_name.ends_with(".xlsx") {
        read_xlsx_rows(&bytes)?
    } else {
        return Err(ApiError::bad_request(
            "unsupported_file_type",
            format!("unsupported file type for \"{filename}\"; expected .csv or .xlsx"),
        ));
    };

    if rows.is_empty() {
        return Err(ApiError::bad_request("empty_file", "file contains no data rows"));
    }

    let records = records_from_rows(&rows)?;
    if records.is_empty() {
        return Err(ApiError::bad_request("empty_file", "file contains no data rows below the header"));
    }

    let headers: Vec<String> = records[0].keys().cloned().collect();
    let columns = resolve_columns(&headers)?;
    let (holdings, tickers_normalised, total_value_inr) = match columns.layout {
        Layout::Weight => build_weight_based(&records, &columns)?,
        Layout::Value => build_value_based(&records, &columns)?,
    };

    // A holding whose weight rounds to ~0 (a zero-share row, a rounding
    // artifact, ...) isn't a real position -- drop it rather than
    // returning a portfolio entry that will just fail downstream
    // validation (every weight must be > 0) or add noise with no actual
    // risk contribution.
    let holdings_before = holdings.len();
    let holdings: Vec<Holding> = holdings.into_iter().filter(|h| h.weight.abs() > 1e-6).collect();
    let skipped_zero_weight = holdings_before - holdings.len();

    let original_holding_count = holdings.len();
    let mut total_value_inr = total_value_inr;
    let mut isins_resolved = BTreeMap::new();
    let mut isins_skipped = Vec::new();
    let mut weights_renormalized = false;

    let isin_holdings: Vec<(String, f64)> = holdings
        .iter()
        .filter(|h| is_isin(&h.ticker))
        .map(|h| (h.ticker.clone(), h.weight * total_value_inr))
        .collect();
    let holdings = if isin_holdings.is_empty() {
        holdings
    } else {
        let batch = compute::isin::resolve_isins_batch_with(&state.isin_config, &isin_holdings, &state.isin_client).await;
        isins_resolved = batch.resolved.iter().map(|(i, s, _)| (i.clone(), s.clone())).collect();
        isins_skipped = batch.skipped;

        let kept = apply_isin_resolution(holdings, &isins_resolved);
        let kept_fraction: f64 = kept.iter().map(|h| h.weight).sum();
        if kept.len() < 2 {
            // Nothing (or one holding) usable. If the cause was transient
            // lookups failing, say "try again" instead of blaming the file.
            if isins_skipped.iter().any(|s| s.category == "timeout") {
                return Err(ApiError::new(
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    "data_unavailable",
                    crate::error::MSG_DATA_UNAVAILABLE,
                ));
            }
            let reasons: Vec<String> = isins_skipped.iter().map(|s| s.reason.clone()).collect();
            return Err(ApiError {
                status: axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                code: "insufficient_holdings",
                message: format!(
                    "Fewer than 2 of your holdings could be analysed after resolving ISINs ({} of {} kept). {}",
                    kept.len(),
                    original_holding_count,
                    reasons.join(" ")
                ),
                extra: Some(serde_json::json!({ "isins_skipped": isins_skipped })),
            });
        }
        if !isins_skipped.is_empty() {
            weights_renormalized = true;
            // A value-based file's total is the sum of its holdings' values,
            // so it shrinks with them; a weight-based file's is a placeholder.
            if matches!(columns.layout, Layout::Value) {
                total_value_inr *= kept_fraction;
            }
        }
        renormalize(kept)
    };

    if holdings.len() < 2 {
        return Err(ApiError::new(
            axum::http::StatusCode::UNPROCESSABLE_ENTITY,
            "insufficient_holdings",
            format!("portfolio must have at least 2 holdings after filtering, got {}", holdings.len()),
        ));
    }

    let retained_holding_count = holdings.len();
    let round2 = |x: f64| (x * 100.0).round() / 100.0;
    let portfolio = Portfolio { holdings, total_value_inr: round2(total_value_inr) };
    validate_portfolio(&portfolio)?;

    Ok(axum::Json(UploadResponse {
        portfolio,
        layout_detected: match columns.layout {
            Layout::Weight => "weight",
            Layout::Value => "value",
        },
        tickers_normalised,
        row_count: original_holding_count,
        skipped_zero_weight,
        isins_resolved,
        isins_skipped,
        weights_renormalized,
        original_holding_count,
        retained_holding_count,
    }))
}

/// Replaces each ISIN holding with its resolved symbol and drops the ones
/// with none; if two rows end up on the same symbol their weights are
/// merged. Other holdings pass through untouched, in order.
fn apply_isin_resolution(holdings: Vec<Holding>, resolved: &BTreeMap<String, String>) -> Vec<Holding> {
    let mut out: Vec<Holding> = Vec::with_capacity(holdings.len());
    for h in holdings {
        let ticker = if is_isin(&h.ticker) {
            match resolved.get(&h.ticker) {
                Some(symbol) => symbol.clone(),
                None => continue,
            }
        } else {
            h.ticker
        };
        match out.iter_mut().find(|e| e.ticker == ticker) {
            Some(existing) => existing.weight += h.weight,
            None => out.push(Holding { ticker, weight: h.weight }),
        }
    }
    out
}

/// Rescales weights to sum to exactly 1 over the holdings kept.
fn renormalize(mut holdings: Vec<Holding>) -> Vec<Holding> {
    let sum: f64 = holdings.iter().map(|h| h.weight).sum();
    if sum > 0.0 {
        for h in &mut holdings {
            h.weight /= sum;
        }
    }
    holdings
}

/// Case-insensitive header lookup: `field()` on a `Record` (whose keys are
/// already lowercased by `records_from_rows`).
fn field<'a>(record: &'a Record, name: &str) -> Option<&'a str> {
    record.get(name).map(|s| s.as_str())
}

/// The first alias (in list order) that matches one of `headers` (both
/// sides already lowercase/trimmed by the caller).
fn find_column(headers: &[String], aliases: &[&str]) -> Option<String> {
    aliases.iter().find_map(|alias| headers.iter().find(|h| h.as_str() == *alias).cloned())
}

fn resolve_columns(headers: &[String]) -> Result<ColumnMap, ApiError> {
    let ticker = find_column(headers, TICKER_ALIASES);
    let weight = find_column(headers, WEIGHT_ALIASES);
    let quantity = find_column(headers, QUANTITY_ALIASES);
    let price = find_column(headers, PRICE_ALIASES);

    let Some(ticker) = ticker else {
        return Err(unsupported_format_error(headers, "ticker"));
    };

    // Weight takes precedence when a file happens to carry both a weight
    // column and quantity+price columns -- it's the more direct signal of
    // intended allocation, not a derived one.
    if let Some(weight) = weight {
        return Ok(ColumnMap { ticker, quantity: None, price: None, weight: Some(weight), layout: Layout::Weight });
    }

    match (quantity, price) {
        (Some(quantity), Some(price)) => Ok(ColumnMap {
            ticker,
            quantity: Some(quantity),
            price: Some(price),
            weight: None,
            layout: Layout::Value,
        }),
        (None, _) => Err(unsupported_format_error(headers, "quantity")),
        (Some(_), None) => Err(unsupported_format_error(headers, "price")),
    }
}

fn unsupported_format_error(headers: &[String], missing: &'static str) -> ApiError {
    let columns = headers.join(", ");
    ApiError::bad_request_with_extra(
        "unsupported_format",
        format!(
            "Could not parse portfolio file. Columns found: [{columns}]. Could not identify: {missing} \
             column.\n\nSupported formats: Zerodha, Upstox, Groww, Angel One, HDFC Securities, ICICI \
             Direct, 5Paisa, Motilal Oswal, Kotak Securities.\n\nIf your broker is not listed, use our \
             template:\nCSV with columns: ticker, weight (e.g. RELIANCE.NS, 0.15)\nor: ticker, shares, \
             avg_price_inr"
        ),
        serde_json::json!({
            "columns_found": headers,
            "missing": missing,
        }),
    )
}

/// Whether a row should be skipped rather than treated as a holding: an
/// empty ticker, a summary/total row, a repeated header row, a
/// placeholder value ("-", "n/a", "nil", ...), or a row whose ticker cell
/// starts with a digit (seen in some exports' footnote/disclaimer rows).
fn should_skip_row(raw_ticker: &str) -> bool {
    let trimmed = raw_ticker.trim();
    if trimmed.is_empty() {
        return true;
    }
    let lower = trimmed.to_lowercase();
    if SKIP_ROW_VALUES.contains(&lower.as_str()) {
        return true;
    }
    trimmed.chars().next().is_some_and(|c| c.is_ascii_digit())
}

fn parse_number(record: &Record, name: &str, row_index: usize) -> Result<f64, ApiError> {
    let raw = field(record, name)
        .ok_or_else(|| ApiError::bad_request("missing_columns", format!("row {row_index}: missing \"{name}\"")))?;
    raw.trim().parse::<f64>().map_err(|_| {
        ApiError::bad_request("invalid_upload", format!("row {row_index}: \"{name}\" is not a number: {raw:?}"))
    })
}

fn is_isin(ticker: &str) -> bool {
    ticker.len() == 12 && ticker.starts_with("IN") && ticker.chars().all(|c| c.is_ascii_alphanumeric())
}

/// Normalises a raw ticker cell value, in order:
/// 1. trim whitespace
/// 2. uppercase
/// 3. strip a leading exchange prefix (`NSE:`, `BSE:`, `NSE/`, `BSE/`)
/// 4. strip a trailing `-EQ` suffix (`RELIANCE-EQ` -> `RELIANCE`)
/// 5. if it already ends in `.NS` or `.BO`, stop here
/// 6. if it looks like an ISIN (`IN` + 10 more alphanumerics, 12 chars
///    total), stop here too -- an ISIN isn't a trading symbol this
///    codebase (or Yahoo Finance) can resolve on its own, so it's left
///    as-is with a note flagging that it needs manual mapping, rather
///    than silently appending `.NS` to something that isn't a symbol
/// 7. otherwise, append `.NS`
///
/// Returns `(normalised_ticker, Some(note) if it changed or needs
/// attention)`.
fn normalise_ticker(raw: &str) -> (String, Option<String>) {
    let original = raw.trim();
    let mut ticker = original.to_uppercase();

    for prefix in ["NSE:", "BSE:", "NSE/", "BSE/"] {
        if let Some(rest) = ticker.strip_prefix(prefix) {
            ticker = rest.to_string();
            break;
        }
    }
    if let Some(stripped) = ticker.strip_suffix("-EQ") {
        ticker = stripped.to_string();
    }

    if ticker.ends_with(".NS") || ticker.ends_with(".BO") {
        let note = if ticker != original { Some(format!("{original} -> {ticker}")) } else { None };
        return (ticker, note);
    }

    if is_isin(&ticker) {
        return (ticker.clone(), Some(format!("{ticker}: ISIN, needs manual ticker mapping")));
    }

    let normalised = format!("{ticker}.NS");
    (normalised.clone(), Some(format!("{original} -> {normalised}")))
}

fn build_weight_based(records: &[Record], columns: &ColumnMap) -> Result<(Vec<Holding>, Vec<String>, f64), ApiError> {
    let weight_col = columns.weight.as_deref().expect("Layout::Weight always has a weight column");
    let mut holdings = Vec::with_capacity(records.len());
    let mut tickers_normalised = Vec::new();

    for (i, record) in records.iter().enumerate() {
        let raw_ticker = field(record, &columns.ticker)
            .ok_or_else(|| ApiError::bad_request("missing_columns", format!("row {i}: missing ticker column")))?;
        if should_skip_row(raw_ticker) {
            continue;
        }
        let (ticker, note) = normalise_ticker(raw_ticker);
        if let Some(note) = note {
            tickers_normalised.push(note);
        }
        let weight = parse_number(record, weight_col, i)?;
        holdings.push(Holding { ticker, weight });
    }

    Ok((holdings, tickers_normalised, DEFAULT_TOTAL_VALUE_INR))
}

fn build_value_based(records: &[Record], columns: &ColumnMap) -> Result<(Vec<Holding>, Vec<String>, f64), ApiError> {
    let quantity_col = columns.quantity.as_deref().expect("Layout::Value always has a quantity column");
    let price_col = columns.price.as_deref().expect("Layout::Value always has a price column");
    let mut rows = Vec::with_capacity(records.len());
    let mut tickers_normalised = Vec::new();

    for (i, record) in records.iter().enumerate() {
        let raw_ticker = field(record, &columns.ticker)
            .ok_or_else(|| ApiError::bad_request("missing_columns", format!("row {i}: missing ticker column")))?;
        if should_skip_row(raw_ticker) {
            continue;
        }
        let (ticker, note) = normalise_ticker(raw_ticker);
        if let Some(note) = note {
            tickers_normalised.push(note);
        }
        let shares = parse_number(record, quantity_col, i)?;
        let avg_price_inr = parse_number(record, price_col, i)?;
        rows.push((ticker, shares * avg_price_inr));
    }

    let total_value_inr: f64 = rows.iter().map(|(_, v)| v).sum();
    let round6 = |x: f64| (x * 1_000_000.0).round() / 1_000_000.0;
    let mut weights: Vec<f64> = rows
        .iter()
        .map(|(_, value_inr)| if total_value_inr > 0.0 { round6(value_inr / total_value_inr) } else { 0.0 })
        .collect();
    // Rounding each weight to 6dp independently can leave the reported
    // weights summing to slightly more/less than 1.0 once enough holdings
    // are involved -- absorb the residual into the last holding so the
    // sum is exact to float precision rather than left to chance.
    if total_value_inr > 0.0 {
        let rounded_sum: f64 = weights.iter().sum();
        if let Some(last) = weights.last_mut() {
            *last = round6(*last + (1.0 - rounded_sum));
        }
    }
    let holdings: Vec<Holding> =
        rows.into_iter().zip(weights).map(|((ticker, _), weight)| Holding { ticker, weight }).collect();

    Ok((holdings, tickers_normalised, total_value_inr))
}

/// Reads every row of a CSV file as raw string cells, with no header
/// interpretation -- `records_from_rows` (shared with XLSX) does the
/// header-row scan instead, so a broker CSV export with metadata rows
/// above its real header is handled the same way an XLSX one is.
fn read_csv_rows(bytes: &[u8]) -> Result<Vec<Vec<String>>, ApiError> {
    // `flexible(true)`: metadata/title rows above the real header (see
    // `find_header_row_index`'s doc) routinely have a different field
    // count than the data rows below them (e.g. a single-cell report
    // title above a 3-column table) -- the csv crate's default strict
    // mode treats that as a malformed-row error, which would reject every
    // broker CSV export with metadata rows before we even get a chance to
    // scan past them.
    let mut reader = csv::ReaderBuilder::new().has_headers(false).flexible(true).from_reader(bytes);
    let mut rows = Vec::new();
    for result in reader.records() {
        let row = result.map_err(|e| ApiError::bad_request("invalid_upload", format!("malformed CSV row: {e}")))?;
        rows.push(row.iter().map(|cell| cell.to_string()).collect());
    }
    Ok(rows)
}

fn read_xlsx_rows(bytes: &[u8]) -> Result<Vec<Vec<String>>, ApiError> {
    let cursor = Cursor::new(bytes.to_vec());
    let mut workbook: Xlsx<_> = open_workbook_from_rs(cursor)
        .map_err(|e| ApiError::bad_request("invalid_upload", format!("failed to open XLSX file: {e}")))?;

    let sheet_name = workbook
        .sheet_names()
        .first()
        .cloned()
        .ok_or_else(|| ApiError::bad_request("invalid_upload", "XLSX file has no sheets"))?;
    let range = workbook
        .worksheet_range(&sheet_name)
        .map_err(|e| ApiError::bad_request("invalid_upload", format!("failed to read XLSX sheet: {e}")))?;

    Ok(range.rows().map(|row| row.iter().map(cell_to_string).collect()).collect())
}

/// Scans the first `MAX_HEADER_SCAN_ROWS` rows for the real header row:
/// the first row containing at least one recognised ticker-column name
/// *and* at least one recognised quantity-or-weight-column name. Broker
/// exports (Zerodha, ICICI Direct, HDFC Securities, ...) routinely prepend
/// metadata rows (account name, report title, date range) above the
/// actual table, so row 0 can't always be trusted as the header.
fn find_header_row_index(rows: &[Vec<String>]) -> Option<usize> {
    rows.iter().take(MAX_HEADER_SCAN_ROWS).position(|row| {
        let lowered: Vec<String> = row.iter().map(|c| c.trim().to_lowercase()).collect();
        let has_ticker = lowered.iter().any(|c| TICKER_ALIASES.contains(&c.as_str()));
        let has_quantity_or_weight = lowered
            .iter()
            .any(|c| QUANTITY_ALIASES.contains(&c.as_str()) || WEIGHT_ALIASES.contains(&c.as_str()));
        has_ticker && has_quantity_or_weight
    })
}

fn records_from_rows(rows: &[Vec<String>]) -> Result<Vec<Record>, ApiError> {
    let Some(header_idx) = find_header_row_index(rows) else {
        // No row in the scanned window had a recognisable ticker *and*
        // quantity-or-weight column together -- report whatever columns
        // row 0 has (the most likely candidate a human would call "the
        // header"), same as `resolve_columns`'s own "ticker not found"
        // error shape, for a consistent response either way.
        let headers: Vec<String> = rows[0].iter().map(|c| c.trim().to_lowercase()).collect();
        return Err(unsupported_format_error(&headers, "ticker"));
    };

    let headers: Vec<String> = rows[header_idx].iter().map(|c| c.trim().to_lowercase()).collect();
    let mut records = Vec::new();
    for row in &rows[header_idx + 1..] {
        let mut record = Record::new();
        for (header, value) in headers.iter().zip(row.iter()) {
            record.insert(header.clone(), value.trim().to_string());
        }
        records.push(record);
    }
    Ok(records)
}

fn cell_to_string(cell: &Data) -> String {
    match cell {
        Data::String(s) => s.trim().to_string(),
        Data::Float(f) => {
            // Numeric cells (e.g. a "weight" column typed as a number by
            // Excel) round-trip through the same string parsing path as
            // CSV, rather than adding a separate numeric branch everywhere
            // a column is read.
            let mut s = format!("{f}");
            if s.ends_with(".0") {
                s.truncate(s.len() - 2);
            }
            s
        }
        Data::Int(i) => i.to_string(),
        _ => cell.as_string().unwrap_or_default(),
    }
}

#[cfg(test)]
mod isin_unit_tests {
    use super::*;

    fn h(t: &str, w: f64) -> Holding {
        Holding { ticker: t.to_string(), weight: w }
    }

    #[test]
    fn skipping_holdings_renormalizes_weights_to_one() {
        let resolved = BTreeMap::from([("INE002A01018".to_string(), "RELIANCE.NS".to_string())]);
        let kept = apply_isin_resolution(
            vec![h("INE002A01018", 0.3), h("TCS.NS", 0.2), h("INE999X99999", 0.4), h("INFY.NS", 0.1)],
            &resolved,
        );
        let tickers: Vec<&str> = kept.iter().map(|x| x.ticker.as_str()).collect();
        assert_eq!(tickers, vec!["RELIANCE.NS", "TCS.NS", "INFY.NS"]);
        let kept = renormalize(kept);
        let sum: f64 = kept.iter().map(|x| x.weight).sum();
        assert!((sum - 1.0).abs() < 1e-6, "{sum}");
        assert!((kept[0].weight - 0.5).abs() < 1e-12); // 0.3 / 0.6
    }

    #[test]
    fn two_isins_resolving_to_one_symbol_merge_their_weights() {
        let resolved = BTreeMap::from([
            ("INE000000001".to_string(), "AAA.NS".to_string()),
            ("INE000000002".to_string(), "AAA.NS".to_string()),
        ]);
        let kept = apply_isin_resolution(vec![h("INE000000001", 0.25), h("INE000000002", 0.25), h("TCS.NS", 0.5)], &resolved);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].weight - 0.5).abs() < 1e-12);
    }
}
