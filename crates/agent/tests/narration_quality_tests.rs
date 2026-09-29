//! Quality checks on the new conversational narration/suggestion style
//! (`NARRATE_SYSTEM_PROMPT`/`SUGGEST_SYSTEM_PROMPT`), asserted against real
//! narrations captured from a live Gemini run against a 10-stock Nifty
//! portfolio (see this session's live-verification report) -- these are
//! properties the *model* must satisfy, not something this crate's own
//! code enforces, so a canned fixture is what's actually being checked,
//! not a mocked round-trip through our own logic.

/// A regex matching 7 or more consecutive digits -- narration must always
/// express a rupee amount in Indian notation (₹X.XL/₹X.XCr) instead of a
/// raw 7-digit figure (see `NARRATE_SYSTEM_PROMPT`'s formatting rules).
fn contains_seven_plus_digit_number(text: &str) -> bool {
    let mut run = 0;
    for c in text.chars() {
        if c.is_ascii_digit() {
            run += 1;
            if run >= 7 {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// Trace field names narration must never leak verbatim (it must describe
/// what a number means, not name the field it came from).
const LEAKED_FIELD_SUFFIXES: &[&str] = &["_pct", "_inr", "_log", "_annualized"];

fn contains_a_leaked_field_name(text: &str) -> bool {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .any(|word| LEAKED_FIELD_SUFFIXES.iter().any(|suffix| word.ends_with(suffix)))
}

/// Captured live via `POST /ask` "How is my portfolio doing?" against a
/// 10-stock Nifty portfolio (RELIANCE.NS, HDFCBANK.NS, ICICIBANK.NS,
/// INFY.NS, TCS.NS, LT.NS, ITC.NS, KOTAKBANK.NS, BHARTIARTL.NS, TMPV.NS),
/// with LT.NS as `best_performer` and TMPV.NS as `worst_performer` in the
/// underlying `PortfolioPerformanceOutput`.
const LIVE_PORTFOLIO_PERFORMANCE_NARRATION: &str = "Your portfolio lost 17.8% over the period, bringing total value down to \u{20b9}82.2L while enduring a maximum drawdown of 20.4%. LT.NS was your best performer, whereas TMPV.NS was your worst performer and the primary drag on the book. You took 14.7% annualised volatility for a \u{2212}17.8% return \u{2014} taking on that level of risk was clearly not rewarded. While the underlying environment remains classified as a Bull regime, holding unhedged high-beta single names is severely eroding your capital base. I recommend eliminating TMPV.NS from the portfolio immediately and reallocating that capital into lower-volatility core holdings.";

#[test]
fn portfolio_performance_narration_names_best_and_worst_performer() {
    assert!(
        LIVE_PORTFOLIO_PERFORMANCE_NARRATION.contains("LT.NS"),
        "narration should name the best_performer ticker"
    );
    assert!(
        LIVE_PORTFOLIO_PERFORMANCE_NARRATION.contains("TMPV.NS"),
        "narration should name the worst_performer ticker"
    );
}

#[test]
fn narration_contains_no_seven_plus_digit_raw_number() {
    assert!(
        !contains_seven_plus_digit_number(LIVE_PORTFOLIO_PERFORMANCE_NARRATION),
        "narration should use Indian-notation abbreviations (\u{20b9}X.XL/\u{20b9}X.XCr), never a raw 7+ digit rupee figure"
    );
    // Regression: the detector itself must actually catch a raw figure.
    assert!(contains_seven_plus_digit_number("your loss is 1195069 rupees"));
}

#[test]
fn narration_contains_no_leaked_field_names() {
    assert!(
        !contains_a_leaked_field_name(LIVE_PORTFOLIO_PERFORMANCE_NARRATION),
        "narration should describe what a number means, not the trace field it came from"
    );
    // Regression: the detector itself must actually catch a leaked field
    // name in each of the four suffixes this session's fields use.
    assert!(contains_a_leaked_field_name("total_return_pct was -17.8"));
    assert!(contains_a_leaked_field_name("portfolio_pnl_inr is negative"));
    assert!(contains_a_leaked_field_name("portfolio_log_pnl_inr you cited is wrong"));
    assert!(contains_a_leaked_field_name("portfolio_vol_annualized rose"));
}

/// Suggestions captured live from the same session's `POST /ask` calls
/// (`suggest::SUGGEST_SYSTEM_PROMPT`'s new "action, not question" style).
const LIVE_SUGGESTIONS: &[&str] = &[
    "Decompose the -17.76% return to isolate the top three portfolio detractors.",
    "Decompose the negative return to isolate the specific sector driving the drawdown.",
    "Decompose the P&L by factor to identify the primary shock driver.",
    "Recalibrate the covariance matrix using rolling 90-day returns.",
];

#[test]
fn suggestions_are_under_15_words_and_are_not_phrased_as_a_question() {
    for suggestion in LIVE_SUGGESTIONS {
        let word_count = suggestion.split_whitespace().count();
        assert!(word_count < 15, "{suggestion:?} has {word_count} words, expected under 15");
        assert!(
            !suggestion.trim_end().ends_with('?'),
            "{suggestion:?} is phrased as a question, expected an imperative action"
        );
    }
}
