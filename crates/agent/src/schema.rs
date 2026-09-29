//! Produces the function declarations Gemini uses for structured
//! extraction, derived from `compute::experiments`' `schemars` JSON
//! Schemas.
//!
//! One function per experiment type (`run_factor_shock`,
//! `run_risk_decomposition`, `run_cvar_rebalance`), not one `run_experiment`
//! function with a `oneOf`-tagged-union parameter schema. The latter was
//! the original design and is what `compute::experiments::Experiment`'s own
//! `#[serde(tag = "type")]` shape naturally maps to, but it does not work
//! in practice: confirmed live against Gemini that the model reliably
//! omits the `"type"` discriminator field from a `oneOf` branch's
//! `functionCall.args`, e.g. `"missing field \`type\`"` on every real
//! extraction attempt. Separate functions sidestep the problem entirely —
//! the function *name* Gemini chooses to call is the discriminator, which
//! is exactly what function-calling models are built to get right, instead
//! of also having to correctly populate an artificial enum-valued field
//! inside a `oneOf` schema.

use std::collections::HashMap;

use compute::experiments::{
    CvarRebalanceInput, FactorShockInput, PolicyCheckInput, PortfolioPerformanceInput,
    ReverseStressInput, RiskDecompositionInput, RiskDriftInput,
};
use serde_json::Value;

use crate::gemini::FunctionDeclaration;

pub const FACTOR_SHOCK_FUNCTION: &str = "run_factor_shock";
pub const RISK_DECOMPOSITION_FUNCTION: &str = "run_risk_decomposition";
pub const CVAR_REBALANCE_FUNCTION: &str = "run_cvar_rebalance";
pub const PORTFOLIO_PERFORMANCE_FUNCTION: &str = "run_portfolio_performance";
pub const RISK_DRIFT_FUNCTION: &str = "run_risk_drift";
pub const REVERSE_STRESS_FUNCTION: &str = "run_reverse_stress";
pub const POLICY_CHECK_FUNCTION: &str = "run_policy_check";

/// Fields the caller supplies out-of-band (portfolio holdings/weights) and
/// that are therefore stripped from the schema shown to Gemini, so a small
/// model isn't nudged into inventing a `portfolio` object from prose that
/// never mentions tickers or weights. `parse::parse_experiment` always
/// overwrites this field with the caller's real portfolio before
/// deserializing the function-call args, regardless of whether Gemini
/// included it.
const CALLER_SUPPLIED_FIELDS: &[&str] = &["portfolio"];

/// The three function declarations passed to Gemini for NL -> Experiment
/// parsing (see the module doc for why three, not one `oneOf`-typed one).
pub fn experiment_function_declarations() -> Vec<FunctionDeclaration> {
    vec![
        FunctionDeclaration {
            name: FACTOR_SHOCK_FUNCTION.to_string(),
            description: "Parse the user's request into FactorShock parameters: shocks \
                (as simple percent returns, e.g. -12.0 for -12%) applied to one or more of \
                MARKET, USDINR, BRENT, GOLD_USD, RATES_PROXY, and whether to propagate the \
                shock to unspecified factors. Portfolio holdings and weights are provided \
                separately; only extract the shock parameters from the user's text. Use this \
                when the user asks what happens to their portfolio under a hypothetical market \
                move (a crash, a rate move, a commodity move, etc.)."
                .to_string(),
            parameters: sanitized_schema::<FactorShockInput>(),
        },
        FunctionDeclaration {
            name: RISK_DECOMPOSITION_FUNCTION.to_string(),
            description: "Parse the user's request into RiskDecomposition parameters. \
                Portfolio holdings and weights are provided separately; only extract the \
                (optional) frequency/window parameters from the user's text, if any are \
                mentioned. Use this when the user asks about their current risk level, \
                volatility, or where their risk is concentrated (by stock or by factor) \
                without describing a hypothetical shock or a rebalance."
                .to_string(),
            parameters: sanitized_schema::<RiskDecompositionInput>(),
        },
        FunctionDeclaration {
            name: CVAR_REBALANCE_FUNCTION.to_string(),
            description: "Parse the user's request into CvarRebalance parameters: \
                confidence_level (default 0.95), per_name_cap, turnover_limit, and \
                commission_bps. Portfolio holdings and weights are provided separately; only \
                extract these parameters from the user's text. Use this when the user asks to \
                reduce tail risk, rebalance, or cut CVaR, especially if they mention a turnover \
                or risk budget."
                .to_string(),
            parameters: sanitized_schema::<CvarRebalanceInput>(),
        },
        FunctionDeclaration {
            name: PORTFOLIO_PERFORMANCE_FUNCTION.to_string(),
            description: "Parse the user's request into PortfolioPerformance parameters. \
                Portfolio holdings and weights are provided separately; only extract the \
                (optional) frequency/window parameters from the user's text, if any are \
                mentioned. Use this when the user asks how their portfolio has actually been \
                doing, its returns, gains/losses, or drawdown over some recent period -- \
                anything about realized historical performance, not a hypothetical shock, a \
                current risk breakdown, or a rebalance."
                .to_string(),
            parameters: sanitized_schema::<PortfolioPerformanceInput>(),
        },
        FunctionDeclaration {
            name: RISK_DRIFT_FUNCTION.to_string(),
            description: "Compare current portfolio risk against a prior snapshot. Use when \
                the user asks what has changed in their risk, whether risk has increased, or \
                how their exposure has shifted over time. Leave baseline_snapshot_id empty to \
                use the most recent prior snapshot automatically."
                .to_string(),
            parameters: sanitized_schema::<RiskDriftInput>(),
        },
        FunctionDeclaration {
            name: REVERSE_STRESS_FUNCTION.to_string(),
            description: "Find the smallest market shock that would cause the portfolio to \
                lose at least a specified amount. Use when the user asks what scenario would \
                wipe out X rupees, what shock breaks their portfolio, or how severe a crash \
                needs to be to breach a loss limit. loss_threshold_inr is a positive number in \
                INR."
                .to_string(),
            parameters: sanitized_schema::<ReverseStressInput>(),
        },
        FunctionDeclaration {
            name: POLICY_CHECK_FUNCTION.to_string(),
            description: "Check whether a portfolio complies with specified risk limits. Use \
                when the user asks whether their portfolio is within risk limits, whether they \
                are breaching any policy, or whether their risk is acceptable. All policy \
                fields are optional \u{2014} omit any limit the user does not mention."
                .to_string(),
            parameters: sanitized_schema::<PolicyCheckInput>(),
        },
    ]
}

fn sanitized_schema<T: schemars::JsonSchema>() -> serde_json::Value {
    let schema = schemars::schema_for!(T);
    let mut value = serde_json::to_value(schema).expect("schemars output is always valid JSON");
    strip_fields(&mut value, CALLER_SUPPLIED_FIELDS);
    sanitize_for_gemini(value)
}

/// Reshapes a `schemars`-generated JSON Schema into what Gemini's
/// `generateContent` function-calling API actually accepts. Verified
/// against a live deployment: without this, Gemini rejects the schema
/// outright (400 `INVALID_ARGUMENT`) for using `$schema`/`definitions`/
/// `$ref` (draft-07 features Gemini's schema proto doesn't have),
/// `additionalProperties` (used by schemars for `BTreeMap<String, f64>`
/// fields like `shocks_pct` — not supported; degrades to an unconstrained
/// `object`, which is fine here since the description already says what
/// keys are valid), and `"type": ["integer", "null"]` for `Option<T>`
/// fields (Gemini's proto field is a single enum value, not a list — the
/// `null` branch is dropped, relying on `required` to encode optionality
/// instead of an explicit nullable type).
fn sanitize_for_gemini(mut schema: Value) -> Value {
    let definitions: HashMap<String, Value> = match &mut schema {
        Value::Object(map) => {
            let defs = map.remove("definitions").or_else(|| map.remove("$defs"));
            map.remove("$schema");
            match defs {
                Some(Value::Object(defs_map)) => defs_map.into_iter().collect(),
                _ => HashMap::new(),
            }
        }
        _ => HashMap::new(),
    };
    resolve_and_clean(&mut schema, &definitions);
    schema
}

fn resolve_and_clean(value: &mut Value, defs: &HashMap<String, Value>) {
    match value {
        Value::Object(map) => {
            // Inline `$ref` pointers into `definitions`/`$defs` (Gemini has
            // no concept of a schema reference); replaces this node
            // entirely, so nothing else in this branch applies afterward.
            if let Some(Value::String(r)) = map.get("$ref") {
                if let Some(name) = r
                    .strip_prefix("#/definitions/")
                    .or_else(|| r.strip_prefix("#/$defs/"))
                {
                    if let Some(resolved) = defs.get(name) {
                        let mut resolved = resolved.clone();
                        resolve_and_clean(&mut resolved, defs);
                        *value = resolved;
                        return;
                    }
                }
            }

            // Collapse `allOf: [single_schema]` (schemars' encoding of a
            // `$ref` with sibling keywords, e.g. `{"allOf": [{"$ref": ...}],
            // "default": ...}`) by merging the resolved schema's keys in
            // alongside the siblings already present.
            if let Some(Value::Array(items)) = map.remove("allOf") {
                for mut item in items {
                    resolve_and_clean(&mut item, defs);
                    if let Value::Object(inner) = item {
                        for (k, v) in inner {
                            map.entry(k).or_insert(v);
                        }
                    }
                }
            }

            // Gemini's schema proto has no open-map equivalent; drop it and
            // fall back to an unconstrained `object` (the field's
            // `description` carries the "what goes in here" guidance).
            map.remove("additionalProperties");

            // `Option<T>` fields serialize as `"type": ["T", "null"]`;
            // Gemini's `type` is a single scalar. Optionality is still
            // conveyed via `required` (or its absence), so just drop the
            // `null` branch.
            if let Some(Value::Array(types)) = map.get("type").cloned() {
                match types.into_iter().find(|t| t.as_str() != Some("null")) {
                    Some(t) => {
                        map.insert("type".to_string(), t);
                    }
                    None => {
                        map.remove("type");
                    }
                }
            }

            for child in map.values_mut() {
                resolve_and_clean(child, defs);
            }
        }
        Value::Array(items) => {
            for item in items {
                resolve_and_clean(item, defs);
            }
        }
        _ => {}
    }
}

/// Recursively removes `fields` from every `properties` map and `required`
/// array found anywhere in a JSON Schema value (including inside
/// `definitions`/`$defs`, reached by the generic recursion below).
fn strip_fields(value: &mut serde_json::Value, fields: &[&str]) {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(serde_json::Value::Object(properties)) = map.get_mut("properties") {
                for field in fields {
                    properties.remove(*field);
                }
            }
            if let Some(serde_json::Value::Array(required)) = map.get_mut("required") {
                required.retain(|v| !v.as_str().is_some_and(|s| fields.contains(&s)));
            }
            for child in map.values_mut() {
                strip_fields(child, fields);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                strip_fields(item, fields);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for a live-caught bug: Gemini's `generateContent`
    /// rejected the unsanitized schemars output with a 400 (`$schema`,
    /// `definitions`, `$ref`, `additionalProperties` unrecognized; `type`
    /// as a list not accepted). Asserts none of those survive anywhere in
    /// any of the three function declarations' schemas.
    #[test]
    fn sanitized_schemas_have_no_gemini_incompatible_keywords() {
        for decl in experiment_function_declarations() {
            assert_no_incompatible_keywords(&decl.parameters);
        }
    }

    /// Regression test for a second live-caught bug: a single
    /// `run_experiment` function with a `oneOf`-tagged-union parameter
    /// schema made Gemini omit the `"type"` discriminator field. None of
    /// the three per-experiment schemas should have a `"type"` *property*
    /// named literally `"type"` (the tag was only ever injected by the
    /// outer `Experiment` enum, not present on the inner Input structs
    /// these schemas are generated from), nor a `oneOf` at the top level.
    #[test]
    fn function_schemas_have_no_type_discriminator_or_oneof() {
        for decl in experiment_function_declarations() {
            let params = decl.parameters.as_object().expect("object schema");
            assert!(
                !params.contains_key("oneOf"),
                "{}: schema has a oneOf at the top level",
                decl.name
            );
            if let Some(Value::Object(properties)) = params.get("properties") {
                assert!(
                    !properties.contains_key("type"),
                    "{}: schema has a \"type\" property (the tagged-union discriminator)",
                    decl.name
                );
            }
        }
    }

    fn assert_no_incompatible_keywords(value: &Value) {
        match value {
            Value::Object(map) => {
                assert!(!map.contains_key("$schema"), "found $schema: {value}");
                assert!(!map.contains_key("$ref"), "found $ref: {value}");
                assert!(!map.contains_key("definitions"), "found definitions: {value}");
                assert!(!map.contains_key("$defs"), "found $defs: {value}");
                assert!(
                    !map.contains_key("additionalProperties"),
                    "found additionalProperties: {value}"
                );
                assert!(!map.contains_key("allOf"), "found allOf: {value}");
                if let Some(t) = map.get("type") {
                    assert!(!t.is_array(), "found array-valued \"type\": {value}");
                }
                for child in map.values() {
                    assert_no_incompatible_keywords(child);
                }
            }
            Value::Array(items) => {
                for item in items {
                    assert_no_incompatible_keywords(item);
                }
            }
            _ => {}
        }
    }
}
