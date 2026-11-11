# FAQ

**Do I need a Gemini API key?**
No. Without `GEMINI_API_KEY` the server starts in compute-only mode: `POST /experiment`, `GET /scenarios`, portfolio uploads and PDF reports all work over the API (the built-in UI is conversational and needs the key), and `POST /ask` returns `503` with code `ai_not_configured`. `GET /health` shows `"gemini": "not_configured"`. A free Google AI Studio key is enough for development. Export it in your shell; never commit it.

**Where does market data come from?**
Yahoo Finance's public chart API, fetched over HTTPS and cached as CSV under `data/cache/`. Prices are aligned to the NSE trading calendar. Common large caps and the five factor series are pre-fetched a few seconds after startup so the first request is fast. Yahoo is free but rate-limited; if you see `429`-style errors, wait and retry, or rely on the cache.

**Why is `data_as_of` old?**
Once a ticker is cached, the server reuses that file and does not re-fetch it. On platforms with ephemeral disks (Cloud Run, Fly machines) the cache resets on each cold start, so this rarely shows; on a long-running self-hosted server, prices stay at the date of the first fetch. Every trace reports `data_as_of`, so it is visible. To refresh, stop the server, delete `data/cache/*.csv`, and restart (the startup warmup re-fetches common tickers). The CLI accepts `--refresh`.

**Which broker exports can I upload?**
CSV or XLSX holdings exports from Zerodha, Upstox, Groww, Angel One, HDFC Securities, ICICI Direct, 5Paisa, Motilal Oswal and Kotak Securities, plus a plain `ticker,weight` (or `ticker,shares,avg_price_inr`) file. ISIN-identified holdings are resolved to NSE/BSE symbols; non-equity instruments (ETFs, mutual funds, bonds) and unresolvable ones are reported as skipped and the remaining weights renormalised.

**How do I enable Upstox import?**
Create an app in the Upstox developer console, set `UPSTOX_API_KEY`, `UPSTOX_API_SECRET` and `UPSTOX_REDIRECT_URI` (the same callback URL you registered), and restart. `GET /auth/upstox/status` tells you whether it's configured.

**How does grounded narration work?**
`/ask` plans one or more experiments, runs them, and asks the model to narrate *only* values in the resulting traces. Before the answer is returned, a numeric check confirms every number quoted appears in the trace, and directional/recommendation checks flag text that describes a value backwards or names a holding the trace doesn't single out. Failing narrations are retried; any remaining warnings are returned alongside the answer and stored with the trace.

**Is the open core lookahead-safe?**
Not in the backtesting sense, and it doesn't claim to be. The open core is a *current-portfolio risk* tool: models are fit on a trailing window ending at the latest available price, and every trace records its exact `data_window` and `data_as_of`. Regime labels come from Viterbi decoding over that whole window, which is appropriate for describing the window but is not a point-in-time signal. Lookahead-safe walk-forward validation is a Drift Engine capability.

**Is this suitable for trading?**
No. Drift is a research tool. It computes risk and suggested weights; it never connects to an order system or places trades. See the disclaimer in `NOTICE`.

**What is the Drift Engine?**
A proprietary, hosted service with additional research capabilities (multi-scale path-stability features, regime-aware factor blending, lookahead-safe walk-forward validation with deflated Sharpe). It is not part of this repository.

**The server won't start.**
Run with `RUST_LOG=debug`. Common causes: the directory for `SNAPSHOT_DB_PATH` isn't writable, or the port is in use (set `PORT`).

**How do I add an experiment type?**
Add a variant to `Experiment` in `crates/compute/src/experiments.rs`, wire it into `compute::dispatch`, and add a synthetic-data test under `crates/compute/tests/`. To make it reachable from `/ask`, add a function declaration in `crates/agent/src/schema.rs`. See `CONTRIBUTING.md`.
