mod support;

use agent::grounding::{check_grounding, extract_numbers, grounded_narrate, numeric_leaves};
use support::{text_response, MockGeminiClient};

#[test]
fn lakh_notation_is_normalised_to_absolute_rupees() {
    let numbers = extract_numbers("The portfolio lost \u{20b9}11.7 lakh this week.");
    assert_eq!(numbers.len(), 1);
    assert!((numbers[0].value - 1_170_000.0).abs() < 1e-6);
    assert_eq!(numbers[0].raw_text, "\u{20b9}11.7 lakh");
}

#[test]
fn percent_and_negative_sign_are_normalised_to_a_decimal_fraction() {
    // Unicode minus, as Gemini sometimes emits.
    let numbers = extract_numbers("Market shock: \u{2212}12% implied a portfolio loss.");
    assert_eq!(numbers.len(), 1);
    assert!((numbers[0].value - (-0.12)).abs() < 1e-9);
}

/// Regression test for a second bug caught in this session's live
/// verification: `NARRATE_SYSTEM_PROMPT` mandates the abbreviated `₹X.XL`
/// / `₹X.XCr` forms (not the spelled-out word "lakh"/"crore") for amounts
/// above ₹1L, but the regex used to only recognize "lakh"/"crore" --
/// every compliant narration was spuriously flagged as ungrounded.
#[test]
fn abbreviated_l_and_cr_suffixes_are_recognised() {
    let numbers = extract_numbers("Your portfolio would lose approximately \u{20b9}12.0L, or \u{20b9}1.2Cr in a worse case.");
    assert_eq!(numbers.len(), 2);
    assert!((numbers[0].value - 1_200_000.0).abs() < 1e-6, "got {}", numbers[0].value);
    assert!((numbers[1].value - 12_000_000.0).abs() < 1e-6, "got {}", numbers[1].value);
}

#[test]
fn plain_number_with_thousands_separators_is_normalised() {
    let numbers = extract_numbers("Portfolio P&L: -1,175,389 INR.");
    assert_eq!(numbers.len(), 1);
    assert!((numbers[0].value - (-1_175_389.0)).abs() < 1e-6);
}

/// Regression test for a bug caught in this session's live verification:
/// Gemini writes a negative rupee amount as "−₹11,95,069" (minus, then
/// the rupee sign, then Indian-grouped digits) -- the sign must still be
/// picked up even though it precedes ₹ rather than the digits themselves,
/// or a real loss gets misread as a positive number and wrongly flagged
/// as ungrounded.
#[test]
fn a_minus_sign_before_the_rupee_symbol_is_still_applied() {
    let numbers = extract_numbers("Your portfolio would lose approximately \u{2212}\u{20b9}11,95,069.");
    assert_eq!(numbers.len(), 1);
    assert!((numbers[0].value - (-1_195_069.0)).abs() < 1e-6, "got {}", numbers[0].value);
}

#[test]
fn numeric_leaves_flattens_nested_trace_json() {
    let value = serde_json::json!({
        "a": 1.5,
        "b": { "c": -2.0, "d": [3, 4.25] },
        "e": "not a number",
        "f": null,
        "g": true,
    });
    let mut leaves = numeric_leaves(&value);
    leaves.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(leaves, vec![-2.0, 1.5, 3.0, 4.25]);
}

/// Regression test for a bug caught live verifying this session's
/// conversational narration style: "Your portfolio lost 17.8%" states the
/// magnitude as a positive number and conveys the sign through the word
/// "lost", not a literal minus sign -- `check_grounding` must still match
/// it against a negative trace value like `total_return_pct: -17.8`.
#[test]
fn a_positive_stated_magnitude_matches_a_negative_trace_value() {
    let trace_numbers = vec![-0.178, -0.204];
    let narration = "Your portfolio lost 17.8% with a maximum drawdown of 20.4%.";
    let check = check_grounding(narration, &trace_numbers);
    assert!(check.passed(), "unmatched: {:?}", check.unmatched);
    assert_eq!(check.matched.len(), 2);
}

#[test]
fn narration_with_all_matching_numbers_passes() {
    let trace_numbers = vec![-0.12, 0.20, -1_175_389.13, 252.0];
    let narration =
        "Given shocks of -12% and 20% imply a portfolio loss of -1,175,389 over the 252-day window.";
    let check = check_grounding(narration, &trace_numbers);
    assert!(check.passed(), "unmatched: {:?}", check.unmatched);
    assert_eq!(check.matched.len(), 4);
}

#[test]
fn injected_unmatched_number_fails_and_is_recorded_with_its_token() {
    let trace_numbers = vec![-0.12, 0.20];
    let narration = "Given a -12% shock, the portfolio loses 47% -- a number nowhere in the trace.";
    let check = check_grounding(narration, &trace_numbers);
    assert!(!check.passed());
    assert_eq!(check.unmatched.len(), 1);
    assert_eq!(check.unmatched[0].raw_text, "47%");
    assert!((check.unmatched[0].value - 0.47).abs() < 1e-9);
}

#[tokio::test]
async fn retry_path_is_invoked_exactly_once_on_a_single_failure() {
    let trace = sample_trace();
    let trace_numbers = numeric_leaves(&serde_json::to_value(&trace).unwrap());
    // Sanity: 0.1552 is one of RiskDecomposition's own numbers below.
    assert!(trace_numbers.iter().any(|&n| (n - 0.1552).abs() < 1e-9));

    // First response has an unmatched number (99%); second (the retry) is clean.
    let client = MockGeminiClient::new(vec![
        text_response("Vol is 0.1552, but also 99% which is not grounded."),
        text_response("Vol is 0.1552 annualised."),
    ]);

    let result = grounded_narrate(&client, &trace, &[]).await.unwrap();
    assert_eq!(client.call_count(), 2, "expected exactly one retry (2 calls total)");
    assert!(result.grounding_warnings.is_empty());
    assert_eq!(result.narration, "Vol is 0.1552 annualised.");
}

#[tokio::test]
async fn still_failing_after_max_retries_surfaces_warnings_without_suppressing_the_narration() {
    let trace = sample_trace();
    let client = MockGeminiClient::new(vec![
        text_response("Vol is 99% (attempt 1)."),
        text_response("Vol is 99% (attempt 2)."),
        text_response("Vol is 99% (attempt 3)."),
    ]);

    let result = grounded_narrate(&client, &trace, &[]).await.unwrap();
    assert_eq!(client.call_count(), 3, "1 initial + 2 retries = 3 calls");
    assert!(!result.grounding_warnings.is_empty());
    assert_eq!(result.narration, "Vol is 99% (attempt 3).");
}

fn sample_trace() -> compute::trace::EvidenceTrace {
    use chrono::NaiveDate;
    use compute::data::DataQuality;
    use compute::model::Frequency;
    use compute::trace::{DataWindow, EvidenceTrace, ModelParams};

    EvidenceTrace {
        id: compute::trace::new_trace_id(),
        experiment: "RiskDecomposition".to_string(),
        inputs: serde_json::json!({}),
        data_as_of: "2026-09-24T00:00:00Z".to_string(),
        data_window: DataWindow {
            frequency: Frequency::Daily,
            window_periods: 252,
            start: NaiveDate::from_ymd_opt(2025, 9, 16).unwrap(),
            end: NaiveDate::from_ymd_opt(2026, 9, 24).unwrap(),
        },
        data_quality: DataQuality {
            date_range_start: NaiveDate::from_ymd_opt(2021, 9, 27).unwrap(),
            date_range_end: NaiveDate::from_ymd_opt(2026, 9, 24).unwrap(),
            trading_days: 1234,
            per_series: vec![],
        },
        model_params: ModelParams {
            frequency: Frequency::Daily,
            window_periods: 252,
            factor_names: vec!["MARKET".to_string()],
            shrinkage_intensity: 0.0374,
            annualization_factor: 252.0,
            regime_state: None,
            regime_fallback_warnings: vec![],
            cap_source: None,
        },
        outputs: serde_json::json!({ "result": { "portfolio_vol_annualized": 0.1552 } }),
        invariants: vec![],
        engine_version: "0.1.0".to_string(),
        engine_commit: compute::trace::engine_commit(),
        scenario_provenance: None,
        parent_trace_ids: Vec::new(),
        baseline_model_params: None,
        policy_result: None,
    }
}
