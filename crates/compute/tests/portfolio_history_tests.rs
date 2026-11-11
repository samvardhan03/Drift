use std::collections::BTreeMap;

use chrono::{Duration, NaiveDate};
use compute::data::{DataQuality, MarketData};
use compute::experiments::{Holding, Portfolio};
use compute::portfolio_history::{compute_portfolio_history, MAX_HISTORY_DAYS};
use compute::regime::RegimeState;

fn regime(path: Vec<u8>) -> RegimeState {
    RegimeState {
        current_regime: *path.last().unwrap(),
        current_label: "Bull",
        smoothed_probs: [1.0, 0.0, 0.0],
        viterbi_sequence: path,
        obs_count_per_regime: [0, 0, 0],
        log_likelihood: -1.0,
        n_iter: 1,
        smoothing_note: "",
        transition_matrix: [[0.0; 3]; 3],
        regime_forecast: vec![],
    }
}

/// `n_ret` returns per ticker; A has a full series, B (if `short_b`) only half.
fn market(n_ret: usize, short_b: bool) -> MarketData {
    let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let dates: Vec<NaiveDate> = (0..=n_ret as i64).map(|i| start + Duration::days(i)).collect();
    let a: Vec<f64> = (0..n_ret).map(|i| if i % 3 == 0 { -0.03 } else { 0.012 }).collect();
    let b_len = if short_b { n_ret / 2 } else { n_ret };
    let b: Vec<f64> = (0..b_len).map(|i| if i % 5 == 0 { -0.04 } else { 0.008 }).collect();
    let mut stock_returns = BTreeMap::new();
    stock_returns.insert("A".to_string(), a);
    stock_returns.insert("B".to_string(), b);
    MarketData {
        quality: DataQuality {
            date_range_start: dates[0],
            date_range_end: *dates.last().unwrap(),
            trading_days: dates.len(),
            per_series: vec![],
        },
        dates,
        stock_returns,
        factor_returns: BTreeMap::new(),
    }
}

fn portfolio() -> Portfolio {
    Portfolio {
        holdings: vec![
            Holding { ticker: "A".to_string(), weight: 0.6 },
            Holding { ticker: "B".to_string(), weight: 0.4 },
        ],
        total_value_inr: 1_000_000.0,
    }
}

fn path(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 3) as u8).collect()
}

#[test]
fn arrays_share_one_length_and_start_at_base() {
    let h = compute_portfolio_history(&market(300, false), &portfolio(), &regime(path(300)), 252).unwrap();
    assert_eq!(h.window_days, 252);
    assert_eq!(h.dates.len(), 252);
    assert_eq!(h.portfolio_value.len(), 252);
    assert_eq!(h.portfolio_return_pct.len(), 252);
    assert_eq!(h.regime_sequence.len(), 252);
    assert_eq!(h.drawdown_pct.len(), 252);
    assert_eq!(h.portfolio_return_pct[0], 0.0);
    assert_eq!(h.portfolio_value[0], 1_000_000.0);
    assert_eq!(h.drawdown_pct[0], 0.0);
}

#[test]
fn compounds_weighted_simple_returns() {
    let h = compute_portfolio_history(&market(10, false), &portfolio(), &regime(path(10)), 252).unwrap();
    // First return of the 10: A = -0.03, B = -0.04 (i = 0).
    let r = 0.6 * ((-0.03_f64).exp() - 1.0) + 0.4 * ((-0.04_f64).exp() - 1.0);
    assert!((h.portfolio_value[1] - 1_000_000.0 * (1.0 + r)).abs() < 1e-6);
}

#[test]
fn drawdown_and_return_invariants_hold() {
    let h = compute_portfolio_history(&market(400, false), &portfolio(), &regime(path(400)), 252).unwrap();
    for t in 0..h.window_days {
        assert!(h.drawdown_pct[t] <= 0.0, "drawdown > 0 at {t}");
        // Peak >= day-0 value, so drawdown from peak is never better than
        // return from the start.
        assert!(h.drawdown_pct[t] <= h.portfolio_return_pct[t], "drawdown above return at {t}");
    }
}

#[test]
fn regime_labels_are_only_the_three_names() {
    let h = compute_portfolio_history(&market(300, false), &portfolio(), &regime(path(300)), 252).unwrap();
    assert!(h.regime_sequence.iter().all(|r| matches!(r.as_str(), "Bull" | "Bear" | "Crisis")));
    assert!(h.regime_sequence.iter().any(|r| r == "Crisis"));
}

#[test]
fn window_shrinks_to_regime_path_coverage() {
    let h = compute_portfolio_history(&market(300, false), &portfolio(), &regime(path(100)), 252).unwrap();
    assert_eq!(h.window_days, 101);
}

#[test]
fn window_is_capped_and_limited_by_data() {
    let h = compute_portfolio_history(&market(1500, false), &portfolio(), &regime(path(1500)), 5000).unwrap();
    assert_eq!(h.window_days, MAX_HISTORY_DAYS);
    let h = compute_portfolio_history(&market(50, false), &portfolio(), &regime(path(300)), 252).unwrap();
    assert_eq!(h.window_days, 51);
}

#[test]
fn short_history_holding_contributes_zero_before_it_starts() {
    let h = compute_portfolio_history(&market(20, true), &portfolio(), &regime(path(20)), 252).unwrap();
    // B has 10 returns; for the first 10 periods only A moves the portfolio.
    let r = 0.6 * ((-0.03_f64).exp() - 1.0);
    assert!((h.portfolio_value[1] - 1_000_000.0 * (1.0 + r)).abs() < 1e-6);
}

#[test]
fn missing_holding_or_too_little_data_is_none() {
    let mut p = portfolio();
    p.holdings.push(Holding { ticker: "ZZZ".to_string(), weight: 0.1 });
    assert!(compute_portfolio_history(&market(50, false), &p, &regime(path(50)), 252).is_none());
    assert!(compute_portfolio_history(&market(0, false), &portfolio(), &regime(vec![0]), 252).is_none());
}
