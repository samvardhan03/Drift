//! Upstox OAuth 2.0: "Connect with Upstox" -> authorization redirect ->
//! callback -> one-time access token -> holdings fetch -> `Portfolio`.
//! Backend-only this session; the UI wiring (the actual "Connect" button
//! and where the returned portfolio lands in the builder) is out of scope
//! here.
//!
//! `UpstoxClient` is a trait (mirroring `agent::gemini::GeminiClient`'s
//! pattern) so the OAuth callback's token-exchange/holdings-fetch calls
//! can be exercised in tests with no real Upstox credentials or network
//! access.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use compute::experiments::{Holding, Portfolio};
use serde::Deserialize;
use thiserror::Error;

pub const STATE_TTL: Duration = Duration::from_secs(600);

#[derive(Debug, Error)]
pub enum UpstoxError {
    /// Token-exchange failed; the message is Upstox's own response body
    /// verbatim (or a transport-level error string), per spec -- returned
    /// to the caller as-is for debuggability, not wrapped/paraphrased.
    #[error("{0}")]
    TokenExchange(String),
    /// Holdings fetch failed; same "surface verbatim" contract.
    #[error("{0}")]
    HoldingsFetch(String),
    /// Fewer than 2 NSE_EQ holdings remained after filtering.
    #[error("Your Upstox account has fewer than 2 NSE equity holdings. Please add holdings and try again.")]
    InsufficientHoldings,
}

/// One holding as Upstox's `GET /v2/portfolio/long-term-holdings` response
/// represents it. Only the fields this module actually uses are modeled;
/// the real response has more (isin, last_price, pnl, ...).
#[derive(Debug, Clone, Deserialize)]
pub struct UpstoxHolding {
    pub trading_symbol: String,
    pub exchange: String,
    pub quantity: f64,
    pub average_price: f64,
}

#[derive(Debug, Deserialize)]
struct UpstoxHoldingsEnvelope {
    #[serde(default)]
    data: Vec<UpstoxHolding>,
}

/// Anything that can perform the two Upstox API calls the OAuth callback
/// needs. A trait so tests can supply a mock with no network access, same
/// pattern as `agent::gemini::GeminiClient`.
#[async_trait::async_trait]
pub trait UpstoxClient: Send + Sync {
    async fn exchange_code_for_token(
        &self,
        code: &str,
        client_id: &str,
        client_secret: &str,
        redirect_uri: &str,
    ) -> Result<String, UpstoxError>;

    async fn fetch_holdings(&self, access_token: &str) -> Result<Vec<UpstoxHolding>, UpstoxError>;
}

/// The real client: POSTs/GETs Upstox's actual API.
pub struct HttpUpstoxClient {
    http: reqwest::Client,
}

impl HttpUpstoxClient {
    pub fn new() -> Self {
        HttpUpstoxClient { http: reqwest::Client::new() }
    }
}

impl Default for HttpUpstoxClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl UpstoxClient for HttpUpstoxClient {
    async fn exchange_code_for_token(
        &self,
        code: &str,
        client_id: &str,
        client_secret: &str,
        redirect_uri: &str,
    ) -> Result<String, UpstoxError> {
        let params = [
            ("code", code),
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("redirect_uri", redirect_uri),
            ("grant_type", "authorization_code"),
        ];
        let resp = self
            .http
            .post("https://api.upstox.com/v2/login/authorization/token")
            .header("accept", "application/json")
            .form(&params)
            .send()
            .await
            .map_err(|e| UpstoxError::TokenExchange(e.to_string()))?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| UpstoxError::TokenExchange(e.to_string()))?;
        if !status.is_success() {
            return Err(UpstoxError::TokenExchange(text));
        }
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|_| UpstoxError::TokenExchange(text.clone()))?;
        value
            .get("access_token")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or(UpstoxError::TokenExchange(text))
    }

    async fn fetch_holdings(&self, access_token: &str) -> Result<Vec<UpstoxHolding>, UpstoxError> {
        let resp = self
            .http
            .get("https://api.upstox.com/v2/portfolio/long-term-holdings")
            .header("accept", "application/json")
            .bearer_auth(access_token)
            .send()
            .await
            .map_err(|e| UpstoxError::HoldingsFetch(e.to_string()))?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| UpstoxError::HoldingsFetch(e.to_string()))?;
        if !status.is_success() {
            return Err(UpstoxError::HoldingsFetch(text));
        }
        let envelope: UpstoxHoldingsEnvelope =
            serde_json::from_str(&text).map_err(|_| UpstoxError::HoldingsFetch(text))?;
        Ok(envelope.data)
    }
}

/// Filters to `NSE_EQ` holdings, converts each to a `.NS`-suffixed ticker,
/// and derives weights from `quantity * average_price` (rounded to 6dp).
///
/// **Judgment call**: rounding each weight to 6dp independently can leave
/// the reported weights summing to slightly more/less than 1.0 once enough
/// holdings are involved (each rounding step can move the sum by up to
/// 5e-7, and those can compound) -- this session's own test requires the
/// sum to be within 1e-6 of 1.0, tighter than that worst case. The
/// remainder is absorbed into the last holding's weight (adjusted, then
/// itself rounded to 6dp) so the reported weights always sum to 1.0 to
/// float precision, rather than leaving that to chance.
pub fn holdings_to_portfolio(holdings: &[UpstoxHolding]) -> Result<Portfolio, UpstoxError> {
    let filtered: Vec<&UpstoxHolding> = holdings.iter().filter(|h| h.exchange == "NSE_EQ").collect();
    if filtered.len() < 2 {
        return Err(UpstoxError::InsufficientHoldings);
    }

    let values: Vec<f64> = filtered.iter().map(|h| h.quantity * h.average_price).collect();
    let total_value_inr: f64 = values.iter().sum();

    let raw_weights: Vec<f64> = values
        .iter()
        .map(|&v| if total_value_inr > 0.0 { v / total_value_inr } else { 0.0 })
        .collect();
    let mut rounded_weights: Vec<f64> =
        raw_weights.iter().map(|w| (w * 1_000_000.0).round() / 1_000_000.0).collect();
    let rounded_sum: f64 = rounded_weights.iter().sum();
    if let Some(last) = rounded_weights.last_mut() {
        *last = ((*last + (1.0 - rounded_sum)) * 1_000_000.0).round() / 1_000_000.0;
    }

    let out_holdings: Vec<Holding> = filtered
        .iter()
        .zip(rounded_weights)
        .map(|(h, weight)| Holding { ticker: format!("{}.NS", h.trading_symbol), weight })
        .collect();

    Ok(Portfolio { holdings: out_holdings, total_value_inr })
}

/// Inserts a fresh random state UUID into `map` (stamped with `Instant::
/// now()`) and returns it. No cleanup task removes expired entries on a
/// timer -- `validate_and_consume` below checks expiry (and removes the
/// entry either way) at callback time instead, per spec; the map only
/// ever holds one entry per in-flight OAuth login, so unbounded growth
/// isn't a practical concern at this scale.
pub fn insert_state(map: &Arc<Mutex<HashMap<String, Instant>>>) -> String {
    let state = uuid::Uuid::new_v4().to_string();
    map.lock().unwrap().insert(state.clone(), Instant::now());
    state
}

/// One-time use: removes `state` from `map` regardless of outcome, and
/// returns whether it was present and not yet expired (`STATE_TTL`).
pub fn validate_and_consume_state(map: &Arc<Mutex<HashMap<String, Instant>>>, state: &str) -> bool {
    let inserted_at = map.lock().unwrap().remove(state);
    match inserted_at {
        Some(t) => t.elapsed() < STATE_TTL,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn holding(symbol: &str, exchange: &str, quantity: f64, average_price: f64) -> UpstoxHolding {
        UpstoxHolding {
            trading_symbol: symbol.to_string(),
            exchange: exchange.to_string(),
            quantity,
            average_price,
        }
    }

    #[test]
    fn filters_out_non_nse_eq_holdings() {
        let holdings = vec![
            holding("RELIANCE", "NSE_EQ", 10.0, 2500.0),
            holding("TCS", "NSE_EQ", 5.0, 3800.0),
            holding("SOMEUS", "NASDAQ", 3.0, 100.0),
        ];
        let portfolio = holdings_to_portfolio(&holdings).unwrap();
        assert_eq!(portfolio.holdings.len(), 2);
        assert!(portfolio.holdings.iter().all(|h| h.ticker != "SOMEUS.NS"));
    }

    #[test]
    fn fewer_than_two_nse_eq_holdings_is_an_error() {
        let holdings = vec![holding("RELIANCE", "NSE_EQ", 10.0, 2500.0), holding("SOMEUS", "NASDAQ", 3.0, 100.0)];
        let err = holdings_to_portfolio(&holdings).unwrap_err();
        assert!(matches!(err, UpstoxError::InsufficientHoldings));
    }

    #[test]
    fn weights_sum_to_one_within_tolerance() {
        // Values chosen so raw division doesn't round evenly at 6dp.
        let holdings = vec![
            holding("A", "NSE_EQ", 7.0, 111.11),
            holding("B", "NSE_EQ", 13.0, 222.22),
            holding("C", "NSE_EQ", 3.0, 333.33),
            holding("D", "NSE_EQ", 9.0, 77.77),
            holding("E", "NSE_EQ", 1.0, 999.99),
        ];
        let portfolio = holdings_to_portfolio(&holdings).unwrap();
        let sum: f64 = portfolio.holdings.iter().map(|h| h.weight).sum();
        assert!((sum - 1.0).abs() < 1e-6, "weights summed to {sum}, expected ~1.0");
    }

    #[test]
    fn ticker_gets_ns_suffix_and_value_is_quantity_times_average_price() {
        let holdings = vec![holding("RELIANCE", "NSE_EQ", 10.0, 2500.0), holding("TCS", "NSE_EQ", 5.0, 3800.0)];
        let portfolio = holdings_to_portfolio(&holdings).unwrap();
        assert_eq!(portfolio.total_value_inr, 10.0 * 2500.0 + 5.0 * 3800.0);
        assert!(portfolio.holdings.iter().any(|h| h.ticker == "RELIANCE.NS"));
        assert!(portfolio.holdings.iter().any(|h| h.ticker == "TCS.NS"));
    }

    #[test]
    fn state_round_trips_and_is_one_time_use() {
        let map = Arc::new(Mutex::new(HashMap::new()));
        let state = insert_state(&map);
        assert!(validate_and_consume_state(&map, &state));
        // Second consume of the same state must fail -- one-time use.
        assert!(!validate_and_consume_state(&map, &state));
    }

    #[test]
    fn unknown_state_is_invalid() {
        let map = Arc::new(Mutex::new(HashMap::new()));
        assert!(!validate_and_consume_state(&map, "not-a-real-state"));
    }
}
