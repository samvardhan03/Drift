//! Portfolio validation shared by `/experiment` and `/ask`. Tickers are
//! deliberately not validated here — `compute::data` already errors
//! clearly on a bad ticker, and duplicating that check would just mean two
//! places to keep in sync.

use compute::experiments::Portfolio;

use crate::error::ApiError;

const WEIGHT_SUM_TOLERANCE: f64 = 0.01;

pub fn validate_portfolio(portfolio: &Portfolio) -> Result<(), ApiError> {
    if portfolio.holdings.len() < 2 {
        return Err(ApiError::bad_request(
            "invalid_portfolio",
            format!(
                "portfolio must have at least 2 holdings, got {}",
                portfolio.holdings.len()
            ),
        ));
    }
    if portfolio.total_value_inr <= 0.0 {
        return Err(ApiError::bad_request(
            "invalid_portfolio",
            format!(
                "total_value_inr must be > 0, got {}",
                portfolio.total_value_inr
            ),
        ));
    }
    if let Some(bad) = portfolio.holdings.iter().find(|h| h.weight <= 0.0) {
        return Err(ApiError::bad_request(
            "invalid_portfolio",
            format!(
                "all weights must be > 0, but {} has weight {}",
                bad.ticker, bad.weight
            ),
        ));
    }
    let sum: f64 = portfolio.holdings.iter().map(|h| h.weight).sum();
    if (sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
        return Err(ApiError::bad_request(
            "invalid_portfolio",
            format!("weights must sum to 1.0 (+/- {WEIGHT_SUM_TOLERANCE}), got {sum}"),
        ));
    }
    Ok(())
}
