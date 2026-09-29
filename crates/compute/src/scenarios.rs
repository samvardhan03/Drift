//! Historical scenario presets: fixed shock sets for well-known market
//! events, for direct use as `FactorShockInput::shocks_pct`. These are
//! static reference data (not fit from live data), exposed read-only via
//! `all_scenarios()` and the server's `GET /scenarios`.

use std::collections::BTreeMap;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct HistoricalScenario {
    pub id: &'static str,
    pub name: &'static str,
    pub date_range: &'static str,
    pub description: &'static str,
    /// Simple % returns (not log), keyed by factor name (see
    /// `data::FACTOR_NAMES`).
    pub shocks_pct: BTreeMap<&'static str, f64>,
    pub propagate: bool,
}

fn shocks(pairs: &[(&'static str, f64)]) -> BTreeMap<&'static str, f64> {
    pairs.iter().copied().collect()
}

fn scenarios() -> Vec<HistoricalScenario> {
    vec![
        HistoricalScenario {
            id: "covid_crash",
            name: "COVID Crash (Mar 2020)",
            date_range: "Feb 19 \u{2013} Mar 23, 2020",
            description: "Nifty fell 38% in 33 days; crude collapsed on the OPEC+ breakdown; INR hit 76.",
            shocks_pct: shocks(&[
                ("MARKET", -38.0),
                ("BRENT", -55.0),
                ("USDINR", 8.5),
                ("GOLD_USD", 3.0),
                ("RATES_PROXY", -6.0),
            ]),
            propagate: false,
        },
        HistoricalScenario {
            id: "ilfs_contagion",
            name: "IL&FS Contagion (Sep\u{2013}Oct 2018)",
            date_range: "Sep 21 \u{2013} Oct 26, 2018",
            description: "IL&FS default triggered NBFC liquidity freeze; Nifty fell 15%; INR hit 74 on oil+EM selloff.",
            shocks_pct: shocks(&[
                ("MARKET", -15.0),
                ("USDINR", 7.0),
                ("BRENT", 15.0),
                ("GOLD_USD", 2.5),
                ("RATES_PROXY", 4.0),
            ]),
            propagate: false,
        },
        HistoricalScenario {
            id: "taper_tantrum_2013",
            name: "Taper Tantrum (May\u{2013}Aug 2013)",
            date_range: "May 22 \u{2013} Aug 28, 2013",
            description: "Fed taper signal sent INR to 68, Nifty fell 12%, gold sold off as dollar surged.",
            shocks_pct: shocks(&[
                ("MARKET", -12.0),
                ("USDINR", 18.0),
                ("GOLD_USD", -18.0),
                ("BRENT", -5.0),
                ("RATES_PROXY", 5.0),
            ]),
            propagate: false,
        },
    ]
}

/// The fixed set of historical scenarios, built once and cached.
pub fn all_scenarios() -> &'static [HistoricalScenario] {
    use std::sync::OnceLock;
    static SCENARIOS: OnceLock<Vec<HistoricalScenario>> = OnceLock::new();
    SCENARIOS.get_or_init(scenarios)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::FACTOR_NAMES;

    #[test]
    fn all_scenarios_returns_exactly_three() {
        assert_eq!(all_scenarios().len(), 3);
    }

    #[test]
    fn every_shock_key_is_a_valid_factor_name() {
        for scenario in all_scenarios() {
            for key in scenario.shocks_pct.keys() {
                assert!(
                    FACTOR_NAMES.contains(key),
                    "scenario {} has unknown factor key {key}",
                    scenario.id
                );
            }
        }
    }
}
