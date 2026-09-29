//! Runs the full pipeline (parse -> compute -> grounded narrate) for the
//! FactorShock 10-stock example, using a scripted mock Gemini client since
//! no GEMINI_API_KEY / network access to Gemini is available in this
//! environment. The compute step itself still hits live Yahoo data,
//! exactly as the `experiment` CLI does. Not part of the crate's public
//! API; a one-off demo for the checkpoint report.

use std::sync::Mutex;

use agent::gemini::{Candidate, Content, GeminiClient, GeminiError, GeminiRequest, GeminiResponse, Part};
use compute::experiments::{Holding, Portfolio};

struct ScriptedClient {
    responses: Mutex<Vec<GeminiResponse>>,
}

#[async_trait::async_trait]
impl GeminiClient for ScriptedClient {
    async fn generate(
        &self,
        _model: &str,
        _request: &GeminiRequest,
    ) -> Result<GeminiResponse, GeminiError> {
        Ok(self
            .responses
            .lock()
            .unwrap()
            .pop()
            .expect("ScriptedClient: ran out of scripted responses"))
    }
}

fn text(t: &str) -> GeminiResponse {
    GeminiResponse {
        candidates: vec![Candidate {
            content: Content {
                role: Some("model".to_string()),
                parts: vec![Part::text(t)],
            },
            finish_reason: Some("STOP".to_string()),
        }],
    }
}

#[tokio::main]
async fn main() {
    let portfolio = Portfolio {
        holdings: vec![
            Holding { ticker: "RELIANCE.NS".to_string(), weight: 0.15 },
            Holding { ticker: "HDFCBANK.NS".to_string(), weight: 0.13 },
            Holding { ticker: "ICICIBANK.NS".to_string(), weight: 0.12 },
            Holding { ticker: "INFY.NS".to_string(), weight: 0.10 },
            Holding { ticker: "TCS.NS".to_string(), weight: 0.10 },
            Holding { ticker: "LT.NS".to_string(), weight: 0.09 },
            Holding { ticker: "ITC.NS".to_string(), weight: 0.09 },
            Holding { ticker: "KOTAKBANK.NS".to_string(), weight: 0.08 },
            Holding { ticker: "BHARTIARTL.NS".to_string(), weight: 0.08 },
            Holding { ticker: "TMPV.NS".to_string(), weight: 0.06 },
        ],
        total_value_inr: 10_000_000.0,
    };

    let narration_text = "A -12% shock to MARKET combined with a +20% shock to BRENT produces a \
portfolio loss of approximately -1,177,846 INR on this ten-stock Nifty portfolio. Because the \
user specified only these two factors, the remaining three factors are model-estimated from \
this portfolio's return history via the factor covariance: USDINR is implied to move +2.53%, \
GOLD_USD -6.36%, and RATES_PROXY -1.80%, each shown separately from the two given shocks above. \
These implied moves are not user inputs; they follow from the historical correlation between \
MARKET, BRENT and the other factors. The loss is dominated by the MARKET shock, given the \
portfolio's substantial equity beta exposure.";

    let plan_json = serde_json::json!([{
        "tool": "factor_shock",
        "params": { "shocks_pct": { "MARKET": -12.0, "BRENT": 20.0 }, "propagate": true },
        "reason": "user asked for a hypothetical market + oil shock",
    }])
    .to_string();

    let client = ScriptedClient {
        responses: Mutex::new(vec![
            // Popped last-in-first-out, so this list is in reverse call
            // order: plan (JSON text), then narrate (text), then suggest
            // (text).
            text("What if I cut my turnover budget to 20% instead?"),
            text(narration_text),
            text(&plan_json),
        ]),
    };

    let store = std::sync::Arc::new(store::SnapshotStore::open(":memory:").expect("in-memory store always opens"));
    let holdings: Vec<(String, f64)> =
        portfolio.holdings.iter().map(|h| (h.ticker.clone(), h.weight)).collect();
    let ctx = compute::context::ExperimentContext {
        store,
        portfolio_hash: compute::portfolio::portfolio_hash(&holdings),
        policy: None,
    };
    let result = agent::pipeline::run(
        &client,
        "what if the market drops 12% and brent jumps 20%?",
        portfolio,
        &[],
        &ctx,
    )
    .await
    .expect("pipeline run failed");

    println!("=== experiment (parsed) ===");
    println!("{}", serde_json::to_string_pretty(&result.experiment).unwrap());
    println!("\n=== trace.outputs ===");
    println!("{}", serde_json::to_string_pretty(&result.trace.outputs).unwrap());
    println!("\n=== trace.invariants ===");
    println!("{}", serde_json::to_string_pretty(&result.trace.invariants).unwrap());
    println!("\n=== narration ===");
    println!("{}", result.narration.narration);
    println!("\n=== grounding_warnings ===");
    println!("{:?}", result.narration.grounding_warnings);
    println!("\n=== suggestion ===");
    println!("{}", result.suggestion);
}
