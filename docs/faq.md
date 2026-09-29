# FAQ

**Q: Do I need a Gemini API key to run Drift?**  
No. Without `GEMINI_API_KEY` the server starts in compute-only mode: `POST /experiment` returns a full `EvidenceTrace`, and `POST /ask` returns a 503. You only need the key for AI narration and natural-language queries. A free-tier Google AI Studio key is sufficient for development.

**Q: What data source does the open-source version use?**  
Yahoo Finance via the `yahoofinance` crate. Data is cached as CSV files in `data/cache/` on first fetch. NSE calendar alignment is applied automatically.

**Q: Is this suitable for live trading?**  
No. Drift is a research tool. It computes target portfolio weights; it does not execute trades, connect to a broker, or have access to your account. See the disclaimer in `NOTICE`.

**Q: What is the Drift Engine?**  
The Drift Engine is a proprietary hosted service that provides additional quantitative features on top of the open-core stack. It is not included in this repository. Contact the maintainers for access.

**Q: How does the evidence trace work?**  
Every `POST /experiment` returns an `EvidenceTrace` JSON object. This object records every intermediate computed value — factor scores, regime state, covariance matrix, and the final result — alongside the raw inputs that produced them. The AI narration in `POST /ask` is verified against this trace before being returned: every number mentioned in the AI's text must appear verbatim in the trace.

**Q: What does "lookahead-safe" mean?**  
All factor scores and HMM regime labels are computed using only data available at the point in time being evaluated. The regime model uses causal (one-sided) filtering. This is enforced structurally and checked in CI.

**Q: The server fails to start. What do I do?**  
Run with `RUST_LOG=debug` for verbose output. Common causes: SQLite can't create the snapshot DB file (check `SNAPSHOT_DB_PATH` permissions); port 8080 in use (set `PORT`).

**Q: Can I contribute a new experiment type?**  
Yes — see `CONTRIBUTING.md`. Add it to the `Experiment` enum in `crates/compute/src/experiments.rs`, implement the compute logic, and add a synthetic-data test in `crates/compute/tests/`.
