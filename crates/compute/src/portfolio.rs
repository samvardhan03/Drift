//! Portfolio identity: a stable hash of a set of (ticker, weight) pairs,
//! used as the join key `store::SnapshotStore` uses to find prior snapshots
//! for "the same portfolio" (see `SnapshotStore::latest_for_portfolio`).

use sha2::{Digest, Sha256};

/// SHA-256 of `"TICKER1:w1,TICKER2:w2,..."`, tickers sorted ascending so the
/// hash is independent of input order, hex-encoded lowercase. Weights are
/// formatted with `{}` (Rust's default `f64` `Display`), so the same weight
/// value always formats identically regardless of which holding it arrived
/// attached to.
pub fn portfolio_hash(holdings: &[(String, f64)]) -> String {
    let mut sorted: Vec<&(String, f64)> = holdings.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let joined = sorted
        .iter()
        .map(|(ticker, weight)| format!("{ticker}:{weight}"))
        .collect::<Vec<_>>()
        .join(",");

    let mut hasher = Sha256::new();
    hasher.update(joined.as_bytes());
    hex_encode(&hasher.finalize())
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_independent_of_holding_order() {
        let a = vec![
            ("RELIANCE.NS".to_string(), 0.5),
            ("HDFCBANK.NS".to_string(), 0.5),
        ];
        let b = vec![
            ("HDFCBANK.NS".to_string(), 0.5),
            ("RELIANCE.NS".to_string(), 0.5),
        ];
        assert_eq!(portfolio_hash(&a), portfolio_hash(&b));
    }

    #[test]
    fn different_weights_hash_differently() {
        let a = vec![("RELIANCE.NS".to_string(), 0.5)];
        let b = vec![("RELIANCE.NS".to_string(), 0.6)];
        assert_ne!(portfolio_hash(&a), portfolio_hash(&b));
    }
}
