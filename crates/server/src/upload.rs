//! `POST /portfolio/upload`: turns a CSV or XLSX file into a `Portfolio`
//! the caller can hand straight to `/experiment` or `/ask`.

use std::collections::BTreeMap;
use std::io::Cursor;

use axum::extract::Multipart;
use calamine::{open_workbook_from_rs, Data, DataType, Reader, Xlsx};
use compute::experiments::{Holding, Portfolio};
use serde::Serialize;

use crate::error::ApiError;
use crate::validate::validate_portfolio;

/// Weight-based CSV/XLSX uploads carry no portfolio value at all (only
/// ticker + weight), so there is nothing to derive `total_value_inr` from.
/// Defaults to this codebase's existing demo convention (the same value
/// `sample_portfolio` helpers across the test suite use) — callers that
/// care about the real value should overwrite it in the returned
/// `Portfolio` before using it.
const DEFAULT_TOTAL_VALUE_INR: f64 = 1_000_000.0;

#[derive(Debug, Serialize)]
pub struct UploadResponse {
    pub portfolio: Portfolio,
    pub layout_detected: &'static str,
    /// `"RELIANCE" -> "RELIANCE.NS"` for every ticker that got the `.NS`
    /// suffix appended; empty if none needed it.
    pub tickers_normalised: Vec<String>,
    pub row_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Layout {
    Weight,
    Value,
}

type Record = BTreeMap<String, String>;

pub async fn post_portfolio_upload(mut multipart: Multipart) -> Result<axum::Json<UploadResponse>, ApiError> {
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
    let records = if lower_name.ends_with(".csv") {
        parse_csv(&bytes)?
    } else if lower_name.ends_with(".xlsx") {
        parse_xlsx(&bytes)?
    } else {
        return Err(ApiError::bad_request(
            "unsupported_file_type",
            format!("unsupported file type for \"{filename}\"; expected .csv or .xlsx"),
        ));
    };

    if records.is_empty() {
        return Err(ApiError::bad_request("empty_file", "file contains no data rows"));
    }

    let layout = detect_layout(&records[0])?;
    let (holdings, tickers_normalised, total_value_inr) = match layout {
        Layout::Weight => build_weight_based(&records)?,
        Layout::Value => build_value_based(&records)?,
    };

    let row_count = holdings.len();
    let portfolio = Portfolio { holdings, total_value_inr };
    validate_portfolio(&portfolio)?;

    Ok(axum::Json(UploadResponse {
        portfolio,
        layout_detected: match layout {
            Layout::Weight => "weight",
            Layout::Value => "value",
        },
        tickers_normalised,
        row_count,
    }))
}

/// Case-insensitive header lookup: `field()` on a `Record` (whose keys are
/// already lowercased in `parse_csv`/`parse_xlsx`).
fn field<'a>(record: &'a Record, name: &str) -> Option<&'a str> {
    record.get(name).map(|s| s.as_str())
}

fn detect_layout(first_record: &Record) -> Result<Layout, ApiError> {
    let has = |name: &str| field(first_record, name).is_some();
    if has("ticker") && has("weight") {
        Ok(Layout::Weight)
    } else if has("ticker") && has("shares") && has("avg_price_inr") {
        Ok(Layout::Value)
    } else {
        Err(ApiError::bad_request(
            "missing_columns",
            "expected either [ticker, weight] or [ticker, shares, avg_price_inr] columns",
        ))
    }
}

fn parse_number(record: &Record, name: &str, row_index: usize) -> Result<f64, ApiError> {
    let raw = field(record, name)
        .ok_or_else(|| ApiError::bad_request("missing_columns", format!("row {row_index}: missing \"{name}\"")))?;
    raw.trim().parse::<f64>().map_err(|_| {
        ApiError::bad_request("invalid_upload", format!("row {row_index}: \"{name}\" is not a number: {raw:?}"))
    })
}

/// Appends `.NS` to a ticker that has no exchange suffix and looks like a
/// bare NSE symbol (all uppercase letters/digits, no `.`). Returns
/// `(normalised_ticker, Some("ORIGINAL -> NORMALISED") if it changed)`.
fn normalise_ticker(raw: &str) -> (String, Option<String>) {
    let ticker = raw.trim();
    let looks_bare_nse =
        !ticker.contains('.') && !ticker.is_empty() && ticker.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit());
    if looks_bare_nse {
        let normalised = format!("{ticker}.NS");
        (normalised.clone(), Some(format!("{ticker} -> {normalised}")))
    } else {
        (ticker.to_string(), None)
    }
}

fn build_weight_based(records: &[Record]) -> Result<(Vec<Holding>, Vec<String>, f64), ApiError> {
    let mut holdings = Vec::with_capacity(records.len());
    let mut tickers_normalised = Vec::new();

    for (i, record) in records.iter().enumerate() {
        let raw_ticker = field(record, "ticker")
            .ok_or_else(|| ApiError::bad_request("missing_columns", format!("row {i}: missing \"ticker\"")))?;
        let (ticker, note) = normalise_ticker(raw_ticker);
        if let Some(note) = note {
            tickers_normalised.push(note);
        }
        let weight = parse_number(record, "weight", i)?;
        holdings.push(Holding { ticker, weight });
    }

    Ok((holdings, tickers_normalised, DEFAULT_TOTAL_VALUE_INR))
}

fn build_value_based(records: &[Record]) -> Result<(Vec<Holding>, Vec<String>, f64), ApiError> {
    let mut rows = Vec::with_capacity(records.len());
    let mut tickers_normalised = Vec::new();

    for (i, record) in records.iter().enumerate() {
        let raw_ticker = field(record, "ticker")
            .ok_or_else(|| ApiError::bad_request("missing_columns", format!("row {i}: missing \"ticker\"")))?;
        let (ticker, note) = normalise_ticker(raw_ticker);
        if let Some(note) = note {
            tickers_normalised.push(note);
        }
        let shares = parse_number(record, "shares", i)?;
        let avg_price_inr = parse_number(record, "avg_price_inr", i)?;
        rows.push((ticker, shares * avg_price_inr));
    }

    let total_value_inr: f64 = rows.iter().map(|(_, v)| v).sum();
    let round6 = |x: f64| (x * 1_000_000.0).round() / 1_000_000.0;
    let holdings: Vec<Holding> = rows
        .into_iter()
        .map(|(ticker, value_inr)| Holding {
            ticker,
            weight: if total_value_inr > 0.0 { round6(value_inr / total_value_inr) } else { 0.0 },
        })
        .collect();

    Ok((holdings, tickers_normalised, total_value_inr))
}

fn parse_csv(bytes: &[u8]) -> Result<Vec<Record>, ApiError> {
    let mut reader = csv::ReaderBuilder::new().has_headers(true).from_reader(bytes);
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| ApiError::bad_request("invalid_upload", format!("failed to read CSV headers: {e}")))?
        .iter()
        .map(|h| h.trim().to_lowercase())
        .collect();

    let mut records = Vec::new();
    for result in reader.records() {
        let row = result.map_err(|e| ApiError::bad_request("invalid_upload", format!("malformed CSV row: {e}")))?;
        let mut record = Record::new();
        for (header, value) in headers.iter().zip(row.iter()) {
            record.insert(header.clone(), value.trim().to_string());
        }
        records.push(record);
    }
    Ok(records)
}

fn parse_xlsx(bytes: &[u8]) -> Result<Vec<Record>, ApiError> {
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

    let mut rows = range.rows();
    let header_row = match rows.next() {
        Some(row) => row,
        None => return Ok(Vec::new()),
    };
    let headers: Vec<String> = header_row.iter().map(|cell| cell_to_string(cell).to_lowercase()).collect();

    let mut records = Vec::new();
    for row in rows {
        let mut record = Record::new();
        for (header, cell) in headers.iter().zip(row.iter()) {
            record.insert(header.clone(), cell_to_string(cell));
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
