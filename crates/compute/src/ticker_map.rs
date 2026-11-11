//! Resolves a broker-reported (or user-typed) ticker symbol to the exact
//! string Yahoo Finance's chart API expects. Pure, stateless, data-layer
//! concern: only the string sent to Yahoo (and used as the cache filename)
//! is affected -- `MarketData`/`AlignedPrices` stay keyed by the original,
//! unresolved ticker string, so callers like `dispatch.rs` never need to
//! know resolution happened at all.

/// NSE "series" suffixes that mark a special trading series (block deal,
/// trade-to-trade, etc.) rather than a different company -- stripped before
/// a mapping-table lookup, and before falling back to the generic `.NS`
/// append.
const SERIES_SUFFIXES: &[&str] = &["-BE", "-BZ", "-SM", "-ST"];

/// Known NSE symbol variants where the broker-reported/plain symbol
/// doesn't match Yahoo Finance's own symbol for that company. Keys are
/// matched case-insensitively (callers uppercase first) and without a
/// trailing ".NS" (stripped before lookup, since some inputs arrive
/// already-suffixed -- see `resolve_ticker`).
const STATIC_MAPPINGS: &[(&str, &str)] = &[
    ("DIL-BZ", "DALBHARAT.NS"),
    ("M&M", "M&M.NS"),
    ("M&MFIN", "M&MFIN.NS"),
    ("L&TFH", "LTFH.NS"),
    ("L&T", "LT.NS"),
    ("NIFTY 50", "^NSEI"),
    // Yahoo accepts this hyphenated form directly -- do NOT de-hyphenate.
    ("BAJAJ-AUTO", "BAJAJ-AUTO.NS"),
    ("HDFCAMC", "HDFCAMC.NS"),
    // Yahoo's symbol is RATNAVEER.NS (RATNAVEERP.NS 404s -- checked live);
    // the "P" spelling is a common mistaken variant, so map it too.
    ("RATNAVEER", "RATNAVEER.NS"),
    ("RATNAVEERP", "RATNAVEER.NS"),
    // Hyphenated symbols Yahoo only knows in hyphenated form (the generic
    // de-hyphenation step would otherwise break them).
    ("MCDOWELL-N", "MCDOWELL-N.NS"),
    ("HDFCLIFE", "HDFCLIFE.NS"),
    ("SBILIFE", "SBILIFE.NS"),
    ("ICICIGI", "ICICIGI.NS"),
    ("ICICIPRULI", "ICICIPRULI.NS"),
    ("NAUKRI", "NAUKRI.NS"),
    ("PIIND", "PIIND.NS"),
    ("TATACOMM", "TATACOMM.NS"),
    ("TATAELXSI", "TATAELXSI.NS"),
    ("TATAMTRDVR", "TATAMTRDVR.NS"),
];

/// Whether `raw` looks like an ISIN (12 alphanumeric characters starting
/// with "IN", e.g. `INE672A01026`) rather than an NSE trading symbol.
pub fn is_isin(raw: &str) -> bool {
    let t = raw.trim();
    t.len() == 12 && t.to_uppercase().starts_with("IN") && t.chars().all(|c| c.is_ascii_alphanumeric())
}

/// The user-facing error for a holding given as an ISIN. Callers check
/// `is_isin` *before* `resolve_ticker`, which would otherwise happily turn
/// an ISIN into a nonexistent "INE672A01026.NS" symbol.
pub fn isin_error(raw: &str) -> crate::ComputeError {
    crate::ComputeError::UnresolvedTicker(format!(
        "Your portfolio contains ISIN codes ({raw}) instead of NSE ticker symbols. \
         Please re-upload using NSE symbols like RELIANCE.NS, TATAMOTORS.NS. \
         Most brokers let you export by symbol instead of ISIN."
    ))
}

fn lookup(symbol: &str) -> Option<&'static str> {
    STATIC_MAPPINGS
        .iter()
        .find(|(k, _)| *k == symbol)
        .map(|(_, v)| *v)
}

/// Resolves `raw` to the ticker string Yahoo Finance's chart API expects,
/// in this order:
///
/// 1. Static mapping exact match (uppercased, trailing ".NS" stripped).
/// 2. Strip a known series suffix (`-BE`/`-BZ`/`-SM`/`-ST`) and re-check
///    the mapping table.
/// 3. Strip an exchange prefix (`NSE:`/`BSE:`).
/// 4. If hyphenated, try the de-hyphenated form.
/// 5. Append `.NS` if no recognised suffix is already present.
/// 6. Return.
pub fn resolve_ticker(raw: &str) -> String {
    let upper = raw.trim().to_uppercase();

    // Already a Yahoo-native non-NSE symbol (an index or a BSE-suffixed
    // ticker) -- nothing to resolve.
    if upper.starts_with('^') || upper.ends_with(".BO") {
        return upper;
    }

    let bare = upper.strip_suffix(".NS").unwrap_or(&upper);

    // Step 1: static mapping exact match.
    if let Some(mapped) = lookup(bare) {
        return mapped.to_string();
    }

    // Step 2: strip a known series suffix, re-check the mapping table.
    for suffix in SERIES_SUFFIXES {
        if let Some(stripped) = bare.strip_suffix(suffix) {
            if let Some(mapped) = lookup(stripped) {
                return mapped.to_string();
            }
            return finish(stripped);
        }
    }

    finish(bare)
}

/// Steps 3-6: strip an exchange prefix, try de-hyphenating, append `.NS`.
fn finish(ticker: &str) -> String {
    let mut t = ticker;
    for prefix in ["NSE:", "BSE:"] {
        if let Some(rest) = t.strip_prefix(prefix) {
            t = rest;
            break;
        }
    }

    // Step 4: if hyphenated, try the de-hyphenated form -- some NSE symbols
    // are listed under both forms depending on the data source, and a
    // hyphen Yahoo doesn't recognise would otherwise 404.
    let deduped = if t.contains('-') {
        t.replace('-', "")
    } else {
        t.to_string()
    };

    // Step 5: append ".NS" if no recognised suffix is already present.
    if deduped.ends_with(".NS") || deduped.ends_with(".BO") {
        deduped
    } else {
        format!("{deduped}.NS")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isin_detection() {
        assert!(is_isin("INE672A01026"));
        assert!(is_isin("ine672a01026"));
        assert!(!is_isin("INFY"));
        assert!(!is_isin("INE672A0102")); // 11 chars
        assert!(!is_isin("INE672A01026.NS"));
        assert!(!is_isin("RELIANCE.NS"));
    }

    #[test]
    fn isin_error_names_the_code_and_the_fix() {
        match isin_error("INE672A01026") {
            crate::ComputeError::UnresolvedTicker(m) => {
                assert!(m.contains("INE672A01026") && m.contains("RELIANCE.NS"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn added_mappings_resolve() {
        for (raw, want) in [
            ("RATNAVEER", "RATNAVEER.NS"),
            ("RATNAVEER.NS", "RATNAVEER.NS"),
            ("RATNAVEERP", "RATNAVEER.NS"),
            ("MCDOWELL-N", "MCDOWELL-N.NS"),
            ("TATAMTRDVR", "TATAMTRDVR.NS"),
            ("HDFCLIFE", "HDFCLIFE.NS"),
            ("L&T", "LT.NS"),
            ("L&TFH", "LTFH.NS"),
        ] {
            assert_eq!(resolve_ticker(raw), want, "{raw}");
        }
    }

    #[test]
    fn dil_bz_resolves_to_dalbharat() {
        assert_eq!(resolve_ticker("DIL-BZ"), "DALBHARAT.NS");
    }

    #[test]
    fn dil_bz_resolves_even_when_already_ns_suffixed() {
        // upload.rs's own normalisation may have already appended ".NS" to
        // a bare broker-reported symbol before this ever reaches the data
        // layer.
        assert_eq!(resolve_ticker("DIL-BZ.NS"), "DALBHARAT.NS");
    }

    #[test]
    fn ampersand_tickers_resolve_via_static_mapping() {
        assert_eq!(resolve_ticker("M&M"), "M&M.NS");
        assert_eq!(resolve_ticker("M&MFIN"), "M&MFIN.NS");
        assert_eq!(resolve_ticker("L&TFH"), "LTFH.NS");
        assert_eq!(resolve_ticker("L&T"), "LT.NS");
    }

    #[test]
    fn nifty_50_resolves_to_the_index_ticker() {
        assert_eq!(resolve_ticker("NIFTY 50"), "^NSEI");
    }

    #[test]
    fn bajaj_auto_is_not_de_hyphenated() {
        assert_eq!(resolve_ticker("BAJAJ-AUTO"), "BAJAJ-AUTO.NS");
    }

    #[test]
    fn hdfcamc_resolves_via_static_mapping() {
        assert_eq!(resolve_ticker("HDFCAMC"), "HDFCAMC.NS");
    }

    #[test]
    fn bare_symbol_defaults_to_ns_suffix() {
        assert_eq!(resolve_ticker("RELIANCE"), "RELIANCE.NS");
    }

    #[test]
    fn exchange_prefix_is_stripped() {
        assert_eq!(resolve_ticker("NSE:RELIANCE"), "RELIANCE.NS");
        assert_eq!(resolve_ticker("BSE:RELIANCE"), "RELIANCE.NS");
    }

    #[test]
    fn unmapped_hyphenated_symbol_is_de_hyphenated() {
        assert_eq!(resolve_ticker("SOME-MADEUP"), "SOMEMADEUP.NS");
    }

    #[test]
    fn unmapped_series_suffix_is_stripped_then_suffixed() {
        assert_eq!(resolve_ticker("SOMETICKER-BE"), "SOMETICKER.NS");
    }

    #[test]
    fn already_ns_suffixed_unmapped_symbol_passes_through() {
        assert_eq!(resolve_ticker("RELIANCE.NS"), "RELIANCE.NS");
    }

    #[test]
    fn index_ticker_passes_through_unchanged() {
        assert_eq!(resolve_ticker("^NSEI"), "^NSEI");
    }

    #[test]
    fn bse_suffixed_ticker_passes_through_unchanged() {
        assert_eq!(resolve_ticker("500325.BO"), "500325.BO");
    }
}
