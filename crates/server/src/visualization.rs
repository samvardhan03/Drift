//! Chart-ready visualization data derived from a completed `EvidenceTrace`,
//! for `/ask`'s `AskResponse::visualization` field. Reads directly off the
//! trace's `outputs.result` JSON (the same `serde_json::Value` every
//! experiment's typed output struct serializes into) rather than
//! deserializing back into each experiment's own Rust type -- none of
//! those output structs derive `Deserialize` (they're serialize-only,
//! trace-output types), and re-deriving it on all seven just for this
//! would be a much larger change than reading the handful of fields each
//! chart actually needs, the same pattern `routes::get_drift` already uses
//! for `RiskDrift` summaries.

use compute::data::FACTOR_NAMES;
use compute::portfolio_history::PortfolioHistory;
use compute::trace::EvidenceTrace;
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize)]
pub struct VisualizationData {
    /// "factor_breakdown" | "scenario_impact" | "cvar_comparison" |
    /// "performance_breakdown" | "risk_drift" | "reverse_stress" |
    /// "policy_check" -- one per experiment type, see `build_visualization`.
    pub chart_type: String,
    pub charts: Vec<Chart>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Chart {
    /// Unique within the response.
    pub id: String,
    pub title: String,
    /// "bar" | "donut" | "comparison" | "gauge" | "timeseries".
    pub chart_kind: String,
    /// Chart-specific shape; see each `build_*` function's doc for what it
    /// contains.
    pub data: Value,
    /// One sentence to show above the chart.
    pub insight: String,
}

fn num(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

fn opt_num(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

fn text(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or_default().to_string()
}

fn flag(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Builds `AskResponse::visualization` from a completed experiment's
/// `EvidenceTrace`, or `None` if `trace.experiment` isn't one of the seven
/// recognised types (there are none today -- `Experiment` is a closed enum
/// with exactly these variants -- but `trace.experiment` is a plain
/// `String`, not the enum itself, so this stays defensive rather than
/// assuming it). `POST /experiment` never calls this (no narration there
/// either; see `routes::post_experiment`'s doc).
pub fn build_visualization(trace: &EvidenceTrace) -> Option<VisualizationData> {
    let result = trace.outputs.get("result")?;
    let mut viz = match trace.experiment.as_str() {
        "RiskDecomposition" => Some(build_risk_decomposition(result)),
        "FactorShock" => Some(build_factor_shock(result)),
        "CvarRebalance" => Some(build_cvar_rebalance(result)),
        "ReverseStress" => Some(build_reverse_stress(result)),
        "PolicyCheck" => Some(build_policy_check(result)),
        "RiskDrift" => Some(build_risk_drift(result)),
        "PortfolioPerformance" => Some(build_portfolio_performance(result)),
        _ => None,
    }?;
    if let Some(history) = &trace.portfolio_history {
        viz.charts.insert(0, build_portfolio_history_chart(history));
    }
    Some(viz)
}

/// "portfolio_history" (always first when the trace carries a history):
/// the daily portfolio value series, plus total-return and max-drawdown
/// summaries. All arrays have length `window_days`.
fn build_portfolio_history_chart(h: &PortfolioHistory) -> Chart {
    let total_return_pct = h.portfolio_return_pct.last().copied().unwrap_or(0.0);
    let max_drawdown_pct = h.drawdown_pct.iter().copied().fold(0.0_f64, f64::min);
    let span = if h.window_days >= 250 {
        "the past year".to_string()
    } else {
        format!("the past {} trading days", h.window_days)
    };
    let verb = if total_return_pct >= 0.0 { "gained" } else { "lost" };
    Chart {
        id: "portfolio_history".to_string(),
        title: "Portfolio value over time".to_string(),
        chart_kind: "timeseries".to_string(),
        data: json!({
            "dates": h.dates,
            "portfolio_value": h.portfolio_value,
            "portfolio_return_pct": h.portfolio_return_pct,
            "regime_sequence": h.regime_sequence,
            "drawdown_pct": h.drawdown_pct,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "window_days": h.window_days,
        }),
        insight: format!("Portfolio {verb} {}% over {span}", total_return_pct.abs()),
    }
}

fn color_for_factor(name: &str) -> &'static str {
    match name {
        "MARKET" => "#2563EB",
        "USDINR" => "#F59E0B",
        "BRENT" => "#EF4444",
        "GOLD_USD" => "#8B5CF6",
        "RATES_PROXY" => "#10B981",
        _ => "#64748B",
    }
}

/// Chart 1 ("factor_contributions"): each `by_factor` entry's
/// `fraction_of_vol_pct`, plus `specific_risk_fraction_of_vol_pct` as a
/// trailing "Specific Risk" bar.
/// Chart 2 ("vol_forecast"): `garch_forecast.current_vol_annualized` plus
/// each of its `forecasts` entries, all as percentages.
fn build_risk_decomposition(result: &Value) -> VisualizationData {
    let by_factor = result.get("by_factor").and_then(Value::as_array).cloned().unwrap_or_default();

    let mut labels: Vec<String> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    let mut colors: Vec<&'static str> = Vec::new();
    for factor in &by_factor {
        let name = text(factor, "factor");
        colors.push(color_for_factor(&name));
        labels.push(name);
        values.push(num(factor, "fraction_of_vol_pct"));
    }
    labels.push("Specific Risk".to_string());
    values.push(num(result, "specific_risk_fraction_of_vol_pct"));
    colors.push("#64748B");

    let (top_label, top_value) = labels
        .iter()
        .zip(values.iter())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(l, v)| (l.clone(), *v))
        .unwrap_or_default();

    let chart1 = Chart {
        id: "factor_contributions".to_string(),
        title: "What's driving your risk".to_string(),
        chart_kind: "bar".to_string(),
        data: json!({
            "labels": labels,
            "values": values,
            "colors": colors,
            "unit": "%",
        }),
        insight: format!("{top_label} dominates at {top_value}% of total vol"),
    };

    let garch = result.get("garch_forecast").cloned().unwrap_or(Value::Null);
    let current_vol_pct = round2(num(&garch, "current_vol_annualized") * 100.0);
    let forecasts = garch.get("forecasts").and_then(Value::as_array).cloned().unwrap_or_default();

    let mut vf_labels = vec!["Current".to_string()];
    let mut vf_values = vec![current_vol_pct];
    let mut highlight_index = 0usize;
    let mut direction_20d = "stable".to_string();
    for (i, f) in forecasts.iter().enumerate() {
        let horizon = f.get("horizon_days").and_then(Value::as_u64).unwrap_or(0);
        vf_labels.push(format!("{horizon} days"));
        vf_values.push(round2(num(f, "vol_annualized") * 100.0));
        if horizon == 20 {
            highlight_index = i + 1; // +1 for the leading "Current" entry
            direction_20d = text(f, "vol_direction");
        }
    }
    let direction_word = match direction_20d.as_str() {
        "increasing" => "increase",
        "decreasing" => "decrease",
        _ => "remain stable",
    };

    let chart2 = Chart {
        id: "vol_forecast".to_string(),
        title: "Volatility outlook".to_string(),
        chart_kind: "bar".to_string(),
        data: json!({
            "labels": vf_labels,
            "values": vf_values,
            "unit": "%",
            "highlight_index": highlight_index,
        }),
        insight: format!("Volatility forecast to {direction_word} over 20 days"),
    };

    VisualizationData { chart_type: "factor_breakdown".to_string(), charts: vec![chart1, chart2] }
}

/// Chart 1 ("shock_impact"): `per_holding[].pnl_inr`, plus
/// `portfolio_pnl_inr` as the total.
/// Chart 2 ("factor_attribution"): `factor_attribution_log_inr` per
/// factor, in `FACTOR_NAMES` order -- the only factor-level INR
/// attribution the trace carries; it's in log-return space (see
/// `FactorShockOutput::factor_attribution_log_inr`'s doc), not a literal
/// simple-return INR split, which is the nearest available number to what
/// this chart asks for (a judgment call; see the report).
fn build_factor_shock(result: &Value) -> VisualizationData {
    let per_holding = result.get("per_holding").and_then(Value::as_array).cloned().unwrap_or_default();
    let labels: Vec<String> = per_holding.iter().map(|h| text(h, "ticker")).collect();
    let values: Vec<f64> = per_holding.iter().map(|h| num(h, "pnl_inr")).collect();
    let formatted: Vec<String> = values.iter().map(|v| compute::format::format_inr(*v)).collect();
    let total = num(result, "portfolio_pnl_inr");

    let most_affected = labels
        .iter()
        .zip(values.iter())
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(l, _)| l.clone())
        .unwrap_or_default();
    let direction = if total < 0.0 { "loss" } else { "gain" };

    let chart1 = Chart {
        id: "shock_impact".to_string(),
        title: "Portfolio impact by holding".to_string(),
        chart_kind: "bar".to_string(),
        data: json!({
            "labels": labels,
            "values": values,
            "formatted": formatted,
            "unit": "INR",
            "total": total,
            "shocks_pct": result.get("given_shocks_pct").cloned().unwrap_or_else(|| json!({})),
        }),
        insight: format!(
            "Total {direction} of {}, {most_affected} most affected",
            compute::format::format_inr(total)
        ),
    };

    let attribution = result.get("factor_attribution_log_inr").cloned().unwrap_or(Value::Null);
    let mut fa_labels = Vec::new();
    let mut fa_values = Vec::new();
    for name in FACTOR_NAMES {
        if let Some(v) = attribution.get(name).and_then(Value::as_f64) {
            fa_labels.push(name.to_string());
            fa_values.push(v);
        }
    }
    let fa_formatted: Vec<String> = fa_values.iter().map(|v| compute::format::format_inr(*v)).collect();

    let abs_sum: f64 = fa_values.iter().map(|v| v.abs()).sum();
    let (top_factor, top_abs) = fa_labels
        .iter()
        .zip(fa_values.iter())
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(l, v)| (l.clone(), v.abs()))
        .unwrap_or_default();
    let top_share_pct = if abs_sum > 0.0 { round2(top_abs / abs_sum * 100.0) } else { 0.0 };

    let chart2 = Chart {
        id: "factor_attribution".to_string(),
        title: "Which factors drove the loss".to_string(),
        chart_kind: "bar".to_string(),
        data: json!({
            "labels": fa_labels,
            "values": fa_values,
            "formatted": fa_formatted,
            "unit": "INR",
        }),
        insight: format!("{top_factor} factor accounts for {top_share_pct}% of the loss"),
    };

    VisualizationData { chart_type: "scenario_impact".to_string(), charts: vec![chart1, chart2] }
}

/// Chart 1 ("rebalance_comparison"): CVaR and regime vol before/after. If
/// the LP didn't solve (`status != "optimal"`), there is no "after" -- the
/// "after" values fall back to "before" (no change occurred), which keeps
/// the chart shape stable rather than making it optional (a judgment
/// call; see the report).
fn build_cvar_rebalance(result: &Value) -> VisualizationData {
    let status = text(result, "status");
    let stats_before = result.get("stats_before").cloned().unwrap_or(Value::Null);
    let cvar_before = round2(num(&stats_before, "historical_cvar") * 100.0);
    let cvar_after = result
        .get("stats_after")
        .and_then(|v| opt_num(v, "historical_cvar"))
        .map(|v| round2(v * 100.0))
        .unwrap_or(cvar_before);

    let vol_before = round2(opt_num(result, "regime_portfolio_vol_annualized_before").unwrap_or(0.0) * 100.0);
    let vol_after = opt_num(result, "regime_portfolio_vol_annualized_after")
        .map(|v| round2(v * 100.0))
        .unwrap_or(vol_before);

    let turnover_pct = round2(opt_num(result, "turnover").unwrap_or(0.0) * 100.0);
    let commission_inr = opt_num(result, "commission_cost_inr").unwrap_or(0.0);
    let commission_formatted = compute::format::format_inr(commission_inr);

    let data = json!({
        "metrics": [
            {"label": "CVaR (95%)", "before": cvar_before, "after": cvar_after, "unit": "%", "lower_is_better": true},
            {"label": "Portfolio Vol", "before": vol_before, "after": vol_after, "unit": "%", "lower_is_better": true},
        ],
        "turnover_pct": turnover_pct,
        "commission_inr": commission_inr,
        "commission_formatted": commission_formatted,
    });

    let insight = if status == "optimal" {
        format!("CVaR reduced from {cvar_before}% to {cvar_after}%, turnover cost {commission_formatted}")
    } else {
        format!("CVaR unchanged at {cvar_before}% -- rebalance was {status}")
    };

    let chart = Chart {
        id: "rebalance_comparison".to_string(),
        title: "Risk reduction from rebalancing".to_string(),
        chart_kind: "comparison".to_string(),
        data,
        insight,
    };

    VisualizationData { chart_type: "cvar_comparison".to_string(), charts: vec![chart] }
}

/// Chart 1 ("stress_shocks"): `shock_vector` per factor, in `FACTOR_NAMES`
/// order, plus `severity_label`/a color bucketed off it. The spec's
/// "[historical context]" insight clause has no backing field on
/// `ReverseStressOutputs` (unlike `FactorShock`'s `shock_historical_context`),
/// so it's a qualitative gloss of the severity bucket itself, not a
/// fabricated number (a judgment call; see the report).
fn build_reverse_stress(result: &Value) -> VisualizationData {
    let shock_vector = result.get("shock_vector").cloned().unwrap_or(Value::Null);
    let mut labels = Vec::new();
    let mut values = Vec::new();
    for name in FACTOR_NAMES {
        if let Some(v) = shock_vector.get(name).and_then(Value::as_f64) {
            labels.push(name.to_string());
            values.push(v);
        }
    }

    let severity = text(result, "severity_label");
    let (severity_color, context) = match severity.as_str() {
        "within 1\u{3c3}" => ("#10B981", "shocks of this size occur routinely in historical data"),
        "1\u{2013}2\u{3c3}" => ("#F59E0B", "an uncommon but not extreme historical move"),
        "2\u{2013}3\u{3c3}" => ("#EF4444", "a rare move, in line with past major market stress events"),
        _ => ("#7F1D1D", "an extremely rare move, beyond most historical market stress episodes"),
    };

    let chart = Chart {
        id: "stress_shocks".to_string(),
        title: "Minimum shock to breach your loss limit".to_string(),
        chart_kind: "bar".to_string(),
        data: json!({
            "labels": labels,
            "values": values,
            "unit": "%",
            "severity": severity,
            "severity_color": severity_color,
        }),
        insight: format!("A {severity} event \u{2014} {context}"),
    };

    VisualizationData { chart_type: "reverse_stress".to_string(), charts: vec![chart] }
}

/// Chart 1 ("policy_compliance"): `policy_result.checks[]`, limit/actual
/// converted from fraction to percent (every `PolicyCheck` rule uses the
/// same 0-1 fraction convention, see `policy::pct`'s doc).
fn build_policy_check(result: &Value) -> VisualizationData {
    let policy_result = result.get("policy_result").cloned().unwrap_or(Value::Null);
    let checks_in = policy_result.get("checks").and_then(Value::as_array).cloned().unwrap_or_default();

    let checks: Vec<Value> = checks_in
        .iter()
        .map(|c| {
            json!({
                "rule": text(c, "rule"),
                "limit": round2(num(c, "limit") * 100.0),
                "actual": round2(num(c, "actual") * 100.0),
                "passed": flag(c, "passed"),
                "unit": "%",
            })
        })
        .collect();

    let passed_count = checks.iter().filter(|c| c["passed"].as_bool().unwrap_or(false)).count();
    let total = checks.len();

    let chart = Chart {
        id: "policy_compliance".to_string(),
        title: "Policy check results".to_string(),
        chart_kind: "bar".to_string(),
        data: json!({ "checks": checks }),
        insight: format!("{passed_count} of {total} checks passed"),
    };

    VisualizationData { chart_type: "policy_check".to_string(), charts: vec![chart] }
}

/// Chart 1 ("risk_drift"): vol before/after, regime before/after. Uses
/// `baseline_created_at` (an actual stored timestamp) for the insight's
/// "[date]", not a vaguer "N days ago" -- the spec's own field list
/// (`days_elapsed`) has no date string, but the trace carries one.
fn build_risk_drift(result: &Value) -> VisualizationData {
    let vol_before = round2(num(result, "vol_before") * 100.0);
    let vol_after = round2(num(result, "vol_after") * 100.0);
    let days_elapsed = result.get("days_elapsed").and_then(Value::as_i64).unwrap_or(0);
    let regime_before = text(result, "regime_before");
    let regime_after = text(result, "regime_after");
    let regime_changed = flag(result, "regime_changed");
    let baseline_created_at = text(result, "baseline_created_at");

    let change = round2(vol_after - vol_before);
    let direction = if change >= 0.0 { "increased" } else { "decreased" };

    let data = json!({
        "metrics": [
            {"label": "Portfolio Vol", "before": vol_before, "after": vol_after, "unit": "%", "lower_is_better": true},
        ],
        "days_elapsed": days_elapsed,
        "regime_before": regime_before,
        "regime_after": regime_after,
        "regime_changed": regime_changed,
    });

    let chart = Chart {
        id: "risk_drift".to_string(),
        title: "How your risk has changed".to_string(),
        chart_kind: "comparison".to_string(),
        data,
        insight: format!("Vol {direction} by {}% since {baseline_created_at}", change.abs()),
    };

    VisualizationData { chart_type: "risk_drift".to_string(), charts: vec![chart] }
}

/// Chart 1 ("holding_returns"): `holding_returns[].total_return_pct`,
/// sorted descending.
/// Chart 2 ("portfolio_summary"): total return / vol / max drawdown as
/// three plain metrics (no "scatter" chart kind exists in this response
/// shape -- `comparison`'s `metrics` list is the closest fit to the
/// spec's "risk vs return scatter hint", which otherwise names fields
/// that don't match `Chart::data`'s `metrics`-list convention used
/// everywhere else in this file; a judgment call, see the report).
fn build_portfolio_performance(result: &Value) -> VisualizationData {
    let holding_returns = result.get("holding_returns").cloned().unwrap_or(Value::Null);
    let mut entries: Vec<(String, f64)> = holding_returns
        .as_object()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), num(v, "total_return_pct"))).collect())
        .unwrap_or_default();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let labels: Vec<String> = entries.iter().map(|(k, _)| k.clone()).collect();
    let values: Vec<f64> = entries.iter().map(|(_, v)| *v).collect();
    let colors: Vec<&'static str> = values.iter().map(|v| if *v >= 0.0 { "#10B981" } else { "#EF4444" }).collect();

    let best = text(result, "best_performer");
    let worst = text(result, "worst_performer");
    let best_ret = entries.iter().find(|(k, _)| *k == best).map(|(_, v)| *v).unwrap_or(0.0);
    let worst_ret = entries.iter().find(|(k, _)| *k == worst).map(|(_, v)| *v).unwrap_or(0.0);

    let chart1 = Chart {
        id: "holding_returns".to_string(),
        title: "Return by holding".to_string(),
        chart_kind: "bar".to_string(),
        data: json!({ "labels": labels, "values": values, "unit": "%", "colors": colors }),
        insight: format!("Best: {best} (+{best_ret}%), Worst: {worst} ({worst_ret}%)"),
    };

    let total_return_pct = num(result, "total_return_pct");
    let vol_pct = num(result, "annualized_vol_pct");
    let drawdown_pct = num(result, "max_drawdown_pct");
    let rewarded = if total_return_pct >= 0.0 { "rewarded" } else { "not rewarded" };

    let chart2 = Chart {
        id: "portfolio_summary".to_string(),
        title: "Portfolio summary".to_string(),
        chart_kind: "comparison".to_string(),
        data: json!({
            "metrics": [
                {"label": "Total Return", "value": total_return_pct, "unit": "%"},
                {"label": "Annualised Vol", "value": vol_pct, "unit": "%"},
                {"label": "Max Drawdown", "value": drawdown_pct, "unit": "%"},
            ],
        }),
        insight: format!("Returned {total_return_pct}% for {vol_pct}% annualised vol \u{2014} risk was {rewarded}"),
    };

    VisualizationData { chart_type: "performance_breakdown".to_string(), charts: vec![chart1, chart2] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use compute::data::DataQuality;
    use compute::model::Frequency;
    use compute::trace::{DataWindow, ModelParams};

    fn trace_with(experiment: &str, result: Value) -> EvidenceTrace {
        EvidenceTrace {
            id: "test-trace".to_string(),
            experiment: experiment.to_string(),
            inputs: json!({}),
            data_as_of: "2026-01-01T00:00:00Z".to_string(),
            data_window: DataWindow {
                frequency: Frequency::Daily,
                window_periods: 252,
                start: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
                end: NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
            },
            data_quality: DataQuality {
                date_range_start: NaiveDate::from_ymd_opt(2021, 1, 1).unwrap(),
                date_range_end: NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
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
                short_history_tickers: vec![],
            },
            outputs: json!({ "result": result }),
            invariants: vec![],
            engine_version: "0.1.0".to_string(),
            engine_commit: "test".to_string(),
            scenario_provenance: None,
            parent_trace_ids: Vec::new(),
            baseline_model_params: None,
            policy_result: None,
            portfolio_history: None,
        }
    }

    fn risk_decomposition_result() -> Value {
        json!({
            "portfolio_vol_annualized": 0.136,
            "portfolio_vol_annualized_pct": 13.6,
            "by_factor": [
                {"factor": "MARKET", "contribution": 0.09, "fraction_of_vol": 0.663, "fraction_of_vol_pct": 66.3},
                {"factor": "USDINR", "contribution": 0.002, "fraction_of_vol": 0.012, "fraction_of_vol_pct": 1.2},
                {"factor": "BRENT", "contribution": 0.001, "fraction_of_vol": 0.008, "fraction_of_vol_pct": 0.8},
                {"factor": "GOLD_USD", "contribution": 0.0005, "fraction_of_vol": 0.004, "fraction_of_vol_pct": 0.4},
                {"factor": "RATES_PROXY", "contribution": 0.005, "fraction_of_vol": 0.036, "fraction_of_vol_pct": 3.6},
            ],
            "specific_risk_contribution": 0.041,
            "specific_risk_fraction_of_vol": 0.302,
            "specific_risk_fraction_of_vol_pct": 30.2,
            "garch_forecast": {
                "omega": 1e-6,
                "alpha": 0.06,
                "beta": 0.92,
                "persistence": 0.98,
                "long_run_vol_annualized": 0.15,
                "current_vol_annualized": 0.136,
                "forecasts": [
                    {"horizon_days": 5, "vol_annualized": 0.138, "vol_direction": "stable"},
                    {"horizon_days": 10, "vol_annualized": 0.141, "vol_direction": "increasing"},
                    {"horizon_days": 20, "vol_annualized": 0.145, "vol_direction": "increasing"},
                    {"horizon_days": 60, "vol_annualized": 0.152, "vol_direction": "increasing"},
                ],
                "converged": true,
                "log_likelihood": 4100.0,
            },
        })
    }

    #[test]
    fn risk_decomposition_produces_two_charts_with_expected_ids() {
        let trace = trace_with("RiskDecomposition", risk_decomposition_result());
        let viz = build_visualization(&trace).expect("RiskDecomposition should produce a visualization");
        assert_eq!(viz.chart_type, "factor_breakdown");
        let ids: Vec<&str> = viz.charts.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["factor_contributions", "vol_forecast"]);
    }

    #[test]
    fn risk_decomposition_chart_values_match_the_trace() {
        let trace = trace_with("RiskDecomposition", risk_decomposition_result());
        let viz = build_visualization(&trace).unwrap();

        let chart1 = &viz.charts[0];
        assert_eq!(chart1.data["labels"], json!(["MARKET", "USDINR", "BRENT", "GOLD_USD", "RATES_PROXY", "Specific Risk"]));
        assert_eq!(chart1.data["values"], json!([66.3, 1.2, 0.8, 0.4, 3.6, 30.2]));
        assert!(chart1.insight.contains("MARKET"));
        assert!(chart1.insight.contains("66.3"));

        let chart2 = &viz.charts[1];
        assert_eq!(chart2.data["labels"], json!(["Current", "5 days", "10 days", "20 days", "60 days"]));
        assert_eq!(chart2.data["values"], json!([13.6, 13.8, 14.1, 14.5, 15.2]));
        assert_eq!(chart2.data["highlight_index"], json!(3));
        assert!(chart2.insight.contains("increase"));
    }

    fn factor_shock_result() -> Value {
        json!({
            "per_holding": [
                {"ticker": "RELIANCE.NS", "value_inr": 400000.0, "pnl_inr": -245000.0},
                {"ticker": "HDFCBANK.NS", "value_inr": 300000.0, "pnl_inr": -198000.0},
            ],
            "portfolio_pnl_inr": -443000.0,
            "given_shocks_pct": {"MARKET": -15.0, "BRENT": 20.0},
            "factor_attribution_log_inr": {
                "MARKET": -980000.0,
                "USDINR": 23000.0,
                "BRENT": -45000.0,
                "GOLD_USD": -12000.0,
                "RATES_PROXY": -8000.0,
            },
        })
    }

    #[test]
    fn factor_shock_produces_two_charts() {
        let trace = trace_with("FactorShock", factor_shock_result());
        let viz = build_visualization(&trace).expect("FactorShock should produce a visualization");
        assert_eq!(viz.chart_type, "scenario_impact");
        let ids: Vec<&str> = viz.charts.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["shock_impact", "factor_attribution"]);
    }

    #[test]
    fn factor_shock_chart_values_match_the_trace() {
        let trace = trace_with("FactorShock", factor_shock_result());
        let viz = build_visualization(&trace).unwrap();

        let chart1 = &viz.charts[0];
        assert_eq!(chart1.data["labels"], json!(["RELIANCE.NS", "HDFCBANK.NS"]));
        assert_eq!(chart1.data["values"], json!([-245000.0, -198000.0]));
        assert_eq!(chart1.data["total"], json!(-443000.0));
        assert_eq!(chart1.data["shocks_pct"], json!({"MARKET": -15.0, "BRENT": 20.0}));
        assert!(chart1.insight.contains("RELIANCE.NS"));

        let chart2 = &viz.charts[1];
        assert_eq!(chart2.data["labels"], json!(["MARKET", "USDINR", "BRENT", "GOLD_USD", "RATES_PROXY"]));
        assert_eq!(chart2.data["values"], json!([-980000.0, 23000.0, -45000.0, -12000.0, -8000.0]));
        assert!(chart2.insight.contains("MARKET"));
    }

    fn cvar_rebalance_result() -> Value {
        json!({
            "status": "optimal",
            "stats_before": {"historical_var": 0.025, "historical_cvar": 0.0204},
            "stats_after": {"historical_var": 0.02, "historical_cvar": 0.0187},
            "regime_portfolio_vol_annualized_before": 0.136,
            "regime_portfolio_vol_annualized_after": 0.121,
            "turnover": 0.20,
            "commission_cost_inr": 2000.0,
        })
    }

    #[test]
    fn cvar_rebalance_produces_one_chart() {
        let trace = trace_with("CvarRebalance", cvar_rebalance_result());
        let viz = build_visualization(&trace).expect("CvarRebalance should produce a visualization");
        assert_eq!(viz.chart_type, "cvar_comparison");
        assert_eq!(viz.charts.len(), 1);
        assert_eq!(viz.charts[0].id, "rebalance_comparison");
        let metrics = viz.charts[0].data["metrics"].as_array().unwrap();
        assert_eq!(metrics[0]["before"], json!(2.04));
        assert_eq!(metrics[0]["after"], json!(1.87));
        assert_eq!(viz.charts[0].data["turnover_pct"], json!(20.0));
    }

    #[test]
    fn unrecognised_experiment_type_returns_none() {
        let trace = trace_with("SomethingElse", json!({}));
        assert!(build_visualization(&trace).is_none());
    }

    #[test]
    fn every_recognised_experiment_type_produces_at_least_one_chart() {
        let cases = [
            ("RiskDecomposition", risk_decomposition_result()),
            ("FactorShock", factor_shock_result()),
            ("CvarRebalance", cvar_rebalance_result()),
            (
                "ReverseStress",
                json!({
                    "shock_vector": {"MARKET": -4.67, "USDINR": 0.68, "BRENT": 1.80, "GOLD_USD": -0.62, "RATES_PROXY": 0.88},
                    "severity_label": "within 1\u{3c3}",
                }),
            ),
            (
                "PolicyCheck",
                json!({
                    "policy_result": {
                        "checks": [
                            {"rule": "max_vol_annualized", "limit": 0.18, "actual": 0.136, "passed": true},
                        ],
                    },
                }),
            ),
            (
                "RiskDrift",
                json!({
                    "vol_before": 0.121, "vol_after": 0.136, "days_elapsed": 7,
                    "regime_before": "Bull", "regime_after": "Bull", "regime_changed": false,
                    "baseline_created_at": "2026-01-01T00:00:00Z",
                }),
            ),
            (
                "PortfolioPerformance",
                json!({
                    "total_return_pct": -17.76, "annualized_vol_pct": 14.65, "max_drawdown_pct": -20.43,
                    "best_performer": "TATACAP.NS", "worst_performer": "IRFC.NS",
                    "holding_returns": {
                        "TATACAP.NS": {"total_return_pct": 31.2},
                        "IRFC.NS": {"total_return_pct": -18.6},
                    },
                }),
            ),
        ];
        for (experiment, result) in cases {
            let trace = trace_with(experiment, result);
            let viz = build_visualization(&trace).unwrap_or_else(|| panic!("{experiment} should produce a visualization"));
            assert!(!viz.charts.is_empty(), "{experiment} produced an empty charts list");
        }
    }

    fn history(returns: &[f64]) -> PortfolioHistory {
        let mut values = vec![1_000_000.0];
        for r in returns {
            values.push(values.last().unwrap() * (1.0 + r));
        }
        let mut peak = f64::MIN;
        let drawdown = values
            .iter()
            .map(|v| {
                peak = peak.max(*v);
                round2((v / peak - 1.0) * 100.0)
            })
            .collect();
        PortfolioHistory {
            dates: (0..values.len()).map(|i| format!("2025-01-{:02}", i + 1)).collect(),
            portfolio_return_pct: values.iter().map(|v| round2((v / 1_000_000.0 - 1.0) * 100.0)).collect(),
            regime_sequence: vec!["Bull".to_string(); values.len()],
            portfolio_value: values,
            drawdown_pct: drawdown,
            window_days: returns.len() + 1,
        }
    }

    #[test]
    fn portfolio_history_chart_is_first_for_every_experiment_type() {
        for exp in [
            "RiskDecomposition",
            "FactorShock",
            "CvarRebalance",
            "ReverseStress",
            "PolicyCheck",
            "RiskDrift",
            "PortfolioPerformance",
        ] {
            let mut trace = trace_with(exp, json!({}));
            trace.portfolio_history = Some(history(&[0.01, -0.02, 0.005]));
            let viz = build_visualization(&trace).unwrap_or_else(|| panic!("{exp}: no visualization"));
            assert_eq!(viz.charts[0].id, "portfolio_history", "{exp}");
            assert_eq!(viz.charts[0].chart_kind, "timeseries", "{exp}");
            assert_eq!(viz.charts.iter().filter(|c| c.id == "portfolio_history").count(), 1, "{exp}");
        }
    }

    #[test]
    fn portfolio_history_chart_data_and_summaries() {
        let mut trace = trace_with("RiskDecomposition", risk_decomposition_result());
        let n_before = build_visualization(&trace).unwrap().charts.len();
        trace.portfolio_history = Some(history(&[0.10, -0.20, 0.05]));
        let viz = build_visualization(&trace).unwrap();
        assert_eq!(viz.charts.len(), n_before + 1);
        let d = &viz.charts[0].data;
        let last = trace.portfolio_history.as_ref().unwrap().portfolio_return_pct.last().copied().unwrap();
        assert!((d["total_return_pct"].as_f64().unwrap() - last).abs() < 0.01);
        assert_eq!(d["window_days"], json!(4));
        for key in ["dates", "portfolio_value", "portfolio_return_pct", "regime_sequence", "drawdown_pct"] {
            assert_eq!(d[key].as_array().unwrap().len(), 4, "{key}");
        }
        assert_eq!(d["max_drawdown_pct"], json!(-20.0));
        assert_eq!(viz.charts[0].insight, "Portfolio lost 7.6% over the past 4 trading days");
    }

    #[test]
    fn no_history_means_no_history_chart() {
        let trace = trace_with("RiskDecomposition", risk_decomposition_result());
        let viz = build_visualization(&trace).unwrap();
        assert!(viz.charts.iter().all(|c| c.id != "portfolio_history"));
    }
}
