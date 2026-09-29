//! Hermetic tests for `EvidenceTrace` v2's new fields' helper functions
//! (`compute::trace::{data_as_of, engine_commit, new_trace_id}`) -- pure
//! functions, no data fetch or model fit needed.

use chrono::NaiveDate;
use compute::model::Frequency;
use compute::trace::{data_as_of, engine_commit, new_trace_id, DataWindow};

fn sample_data_window() -> DataWindow {
    DataWindow {
        frequency: Frequency::Daily,
        window_periods: 252,
        start: NaiveDate::from_ymd_opt(2025, 9, 16).unwrap(),
        end: NaiveDate::from_ymd_opt(2026, 9, 24).unwrap(),
    }
}

#[test]
fn data_as_of_is_the_data_windows_end_date_as_an_iso8601_utc_timestamp() {
    let value = data_as_of(&sample_data_window());
    assert_eq!(value, "2026-09-24T00:00:00Z");
    // Must parse as a valid RFC 3339 / ISO 8601 timestamp.
    chrono::DateTime::parse_from_rfc3339(&value).expect("data_as_of must be a valid RFC 3339 timestamp");
}

#[test]
fn engine_commit_is_never_empty() {
    // Falls back to "unknown" when VERGEN_GIT_SHA wasn't set at build time
    // (e.g. a source tarball with no `.git` directory) -- either way, the
    // field itself must never be an empty string.
    assert!(!engine_commit().is_empty());
}

#[test]
fn engine_commit_never_leaks_vergens_own_placeholder_string() {
    // Regression test for a bug caught live against a real Cloud Run
    // deploy (no `.git` directory in the Docker build context): vergen's
    // own failure-mode placeholder, "VERGEN_IDEMPOTENT_OUTPUT", must never
    // be exposed as-is -- it should normalize to "unknown" like any other
    // undetermined-commit case.
    assert_ne!(engine_commit(), "VERGEN_IDEMPOTENT_OUTPUT");
}

#[test]
fn new_trace_id_produces_distinct_ids() {
    let a = new_trace_id();
    let b = new_trace_id();
    assert_ne!(a, b);
    assert!(!a.is_empty());
}
