//! Shared INR formatting: Indian lakh/crore digit grouping (last 3 digits,
//! then groups of 2), used both when a trace field carries a pre-formatted
//! string for the agent's narration to cite, and by `server`'s PDF report.
//!
//! Also `to_pct_2dp`: the shared decimal-fraction -> percent conversion
//! every experiment's `_pct` output fields use, so narration always has an
//! exact number to cite for a percentage instead of having to multiply a
//! raw fraction by 100 itself (and risk citing it unconverted, e.g.
//! "0.113%" instead of "11.3%").

/// Rounds `fraction * 100` to 2 decimal places, e.g. `0.11304 -> 11.3`.
pub fn to_pct_2dp(fraction: f64) -> f64 {
    (fraction * 100.0 * 100.0).round() / 100.0
}

/// Rounds an already-percent-scaled value to 2 decimal places, e.g.
/// `-1.7764 -> -1.78` -- for a value that's a product of other `_pct`
/// fields (already percent-scaled), where `to_pct_2dp` would incorrectly
/// multiply by 100 again.
pub fn round_2dp(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// Formats `value` as a whole-rupee amount with Indian digit grouping, e.g.
/// `1_177_846.39 -> "\u{20b9}11,77,846"`. Negative values use the Unicode
/// minus sign (U+2212), not an ASCII hyphen, to match how `agent::grounding`
/// already normalizes narration text. Fractional rupees are truncated
/// (rounded to the nearest whole rupee), since PDF/narration display never
/// needs paisa-level precision.
pub fn format_inr(value: f64) -> String {
    let rounded = value.round();
    let negative = rounded < 0.0;
    let digits = format!("{:.0}", rounded.abs());

    let grouped = group_indian(&digits);

    if negative {
        format!("\u{2212}\u{20b9}{grouped}")
    } else {
        format!("\u{20b9}{grouped}")
    }
}

/// Indian digit grouping: the last 3 digits form one group, then every
/// remaining pair of digits (from the right) forms its own group, e.g.
/// `"1177846" -> "11,77,846"`.
fn group_indian(digits: &str) -> String {
    let len = digits.len();
    if len <= 3 {
        return digits.to_string();
    }
    let (head, tail) = digits.split_at(len - 3);
    let mut groups: Vec<String> = Vec::new();
    let head_bytes = head.as_bytes();
    let mut i = head_bytes.len();
    while i > 2 {
        groups.push(head[i - 2..i].to_string());
        i -= 2;
    }
    groups.push(head[..i].to_string());
    groups.reverse();
    groups.push(tail.to_string());
    groups.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_value_with_lakh_grouping() {
        assert_eq!(format_inr(1_177_846.39), "\u{20b9}11,77,846");
    }

    #[test]
    fn negative_value_uses_unicode_minus() {
        assert_eq!(format_inr(-1_177_846.39), "\u{2212}\u{20b9}11,77,846");
    }

    #[test]
    fn value_under_lakh_still_groups_thousands() {
        assert_eq!(format_inr(3_000.0), "\u{20b9}3,000");
    }

    #[test]
    fn value_under_a_thousand_is_ungrouped() {
        assert_eq!(format_inr(100.0), "\u{20b9}100");
    }

    #[test]
    fn to_pct_2dp_converts_and_rounds() {
        assert_eq!(to_pct_2dp(0.11304), 11.3);
        assert_eq!(to_pct_2dp(-0.19899), -19.9);
        assert_eq!(to_pct_2dp(0.0), 0.0);
    }
}
