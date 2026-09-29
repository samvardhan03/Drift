# drift-risk-copilot

Portfolio risk copilot for BFSI users (AI Builder Cup 2026). A Rust
workspace: `compute` fetches market data, fits a factor risk model, and
runs portfolio-risk experiments, returning a fully-cited `EvidenceTrace`
for every result; `agent` turns a natural-language request into one of
those experiments via Gemini, runs it, and narrates the result back with a
verbatim-number grounding check; `server` is an axum binary (`/health`,
`/experiment`, `/ask`, plus an embedded single-page UI) that calls
`agent::pipeline::run`, with a Dockerfile and Cloud Run config to deploy
it. The backend is deployable and demo-ready as of this checkpoint.

## Workspace layout

```
crates/
  compute/   # data layer, factor model, experiments, evidence trace
    src/data.rs         Yahoo fetch + cache + NSE calendar alignment + log returns
    src/model.rs         OLS factor fits, Ledoit-Wolf shrinkage, stock covariance, regime-conditional F
    src/regime.rs          3-state Gaussian HMM (Baum-Welch, Viterbi) for market-regime detection
    src/experiments.rs    FactorShock, RiskDecomposition
    src/cvar.rs            CvarRebalance: Rockafellar-Uryasev LP via good_lp + clarabel
    src/drift.rs            RiskDrift: diffs current risk against a stored baseline snapshot
    src/reverse_stress.rs    ReverseStress: minimum-severity shock that breaches a loss threshold
    src/policy.rs            RiskPolicy/PolicyResult; PolicyCheck experiment; evaluate_policy
    src/dispatch.rs          run_experiment: the one fetch+fit+dispatch entry point for every experiment type
    src/context.rs            ExperimentContext (SnapshotStore + portfolio_hash), used by RiskDrift
    src/portfolio.rs           portfolio_hash: order-independent SHA-256 of a portfolio's holdings
    src/trace.rs          EvidenceTrace and its sub-structs
    src/scenarios.rs       Fixed historical-scenario presets (COVID crash, IL&FS, taper tantrum)
    src/bin/experiment.rs CLI: runs one experiment from a JSON file
    examples/              FactorShock / RiskDecomposition / CvarRebalance inputs, 10-stock Nifty portfolio
    tests/                  Synthetic-data unit/integration tests (no network required)
  agent/     # NL -> Experiment -> EvidenceTrace -> grounded narration
    src/gemini.rs        Async Gemini client: request/response types, retrying HTTP transport
    src/conversation.rs   ConversationTurn (user/assistant) + Gemini role mapping
    src/schema.rs         JSON Schema (via schemars) for the three per-experiment function declarations
    src/parse.rs           NL -> Experiment via a Gemini function-calling turn, conversation-history aware
    src/narrate.rs          EvidenceTrace -> plain-language narration via Gemini, conversation-history aware
    src/grounding.rs        Verbatim-number check on narration vs. trace, with retry
    src/suggest.rs           One proactive follow-up question via a plain-text Gemini call
    src/pipeline.rs          agent::pipeline::run: the one function `server` calls
    examples/demo_pipeline.rs  One-off demo: mocked Gemini + a real compute call (see below)
    tests/                    Mocked-Gemini unit/integration tests (no network to Gemini)
  server/    # axum HTTP API + embedded UI
    src/main.rs           Router, tracing setup, GEMINI_API_KEY + SnapshotStore startup checks
    src/routes.rs          /health, /scenarios, /experiment, /ask, /report/{id}, static-file fallback handlers
    src/upload.rs            POST /portfolio/upload: CSV/XLSX -> Portfolio
    src/backend.rs          Backend trait (RealBackend wraps compute+agent) + error mapping
    src/validate.rs          Portfolio validation shared by /experiment, /ask, and /portfolio/upload
    src/error.rs             ApiError ({error, code} JSON responses) + AppJson extractor
    src/logging.rs            Request logging middleware (method, path, status, latency)
    src/pdf.rs                GET /report/{id}: renders a PDF from a stored EvidenceTrace
    src/tests.rs               Route tests against a MockBackend (no network)
    static/index.html         Embedded single-page UI (include_str!, no build step)
  store/     # persistent SQLite-backed risk-snapshot store
    src/lib.rs             SnapshotStore + RiskSnapshot; rusqlite (bundled feature, no external sqlite3 needed)
data/cache/  # cached raw price CSVs (gitignored; fetched on first run)
data/snapshots.db (or $SNAPSHOT_DB_PATH)  # SQLite risk-snapshot store
Dockerfile     # multi-stage build -> gcr.io/distroless/cc-debian12
cloudrun.yaml  # Cloud Run service config
```

## Running an experiment

```
cargo run -p compute --bin experiment -- crates/compute/examples/factor_shock_nifty10.json
cargo run -p compute --bin experiment -- crates/compute/examples/risk_decomposition_nifty10.json
cargo run -p compute --bin experiment -- crates/compute/examples/cvar_rebalance_nifty10.json
```

Add `--refresh` to refetch price series instead of reading `data/cache/`.
Each run prints a JSON `EvidenceTrace` to stdout.

> Note: the frequency comparison below was run before the `GOLD` factor
> label was renamed to `GOLD_USD` (see "Trace additions for
> explainability"); `GOLD` in this section refers to what the trace now
> calls `GOLD_USD`. The numbers themselves are unaffected by the rename.

## Return frequency

`FactorShockInput`/`RiskDecompositionInput` take an optional `frequency`
(`"Daily"` | `"Weekly"`, default `"Daily"` — unchanged from the first
checkpoint) and an optional `window` (periods at that frequency; omit to
get `frequency.default_window()`: 252 for Daily, 156 for Weekly).
`Weekly` returns are **non-overlapping**, computed between successive
last-NSE-trading-day-of-the-week closes (`data::weekly_resample_indices`),
not a rolling 5-day window. Annualization (252 vs. 52) is still derived
from a single function, `model::annualize_matrix`/`annualize_scalar`,
now parameterized by `Frequency` instead of a bare constant.

**`examples/*_weekly.json`** run the same 10-stock portfolio at
`Weekly`/156 for comparison against the `Daily`/252 default.

### Daily vs. Weekly comparison (10-stock Nifty portfolio, live data, run 2026-09-24)

Betas on USDINR / BRENT / GOLD / RATES_PROXY (MARKET omitted — large and
stable across both frequencies, ~0.7-1.1 for all ten names):

| ticker | USDINR (D) | USDINR (W) | BRENT (D) | BRENT (W) | GOLD (D) | GOLD (W) | RATES_PROXY (D) | RATES_PROXY (W) |
|---|---|---|---|---|---|---|---|---|
| RELIANCE.NS | -0.059 | -0.505 | -0.011 | 0.046 | -0.016 | -0.083 | -0.732 | -0.597 |
| HDFCBANK.NS | 0.021 | 0.166 | 0.016 | -0.021 | -0.038 | 0.017 | 0.668 | 1.080 |
| ICICIBANK.NS | 0.205 | -0.384 | -0.012 | 0.004 | -0.033 | -0.031 | 0.868 | 0.928 |
| INFY.NS | -0.207 | -0.156 | 0.022 | 0.002 | -0.051 | -0.117 | -1.342 | -1.257 |
| TCS.NS | -0.364 | 0.085 | -0.002 | 0.022 | -0.052 | 0.029 | -1.333 | -0.974 |
| LT.NS | 0.093 | 0.106 | -0.003 | -0.005 | 0.020 | -0.019 | -0.037 | 0.292 |
| ITC.NS | -0.050 | 0.388 | -0.004 | -0.023 | -0.046 | 0.074 | -0.573 | -0.573 |
| KOTAKBANK.NS | 0.244 | -0.109 | -0.002 | 0.005 | 0.037 | 0.127 | 0.416 | 0.413 |
| BHARTIARTL.NS | 0.067 | -0.268 | 0.036 | 0.042 | 0.035 | 0.009 | -0.547 | -0.580 |
| TMPV.NS | 0.083 | 1.225 | -0.052 | -0.086 | -0.060 | -0.111 | -0.011 | -0.760 |

**Takeaway:** the USDINR and RATES_PROXY betas are noticeably less stable
across frequency than MARKET (expected — currency and rate-proxy signal is
noisier per-name and weekly regressions have ~6x fewer observations per
year); BRENT and GOLD betas are small and noisy at both frequencies (no
stock in this portfolio has meaningfully commodity-linked earnings).
Shrinkage intensity is higher weekly (0.053) than daily (0.038), consistent
with a noisier per-period factor covariance estimate needing more
shrinkage toward the target.

Factor correlation matrix (Daily / Weekly, same order MARKET, USDINR,
BRENT, GOLD, RATES_PROXY):

```
Daily                                    Weekly
         MKT   USD   BRT   GLD   RTP              MKT   USD   BRT   GLD   RTP
MKT    1.00 -0.28 -0.27  0.24  0.20     MKT     1.00 -0.23 -0.29  0.04  0.04
USD   -0.28  1.00  0.23 -0.19 -0.02     USD    -0.23  1.00  0.24 -0.04 -0.04
BRT   -0.27  0.23  1.00 -0.17 -0.19     BRT    -0.29  0.24  1.00 -0.08 -0.14
GLD    0.24 -0.19 -0.17  1.00  0.11     GLD     0.04 -0.04 -0.08  1.00 -0.03
RTP    0.20 -0.02 -0.19  0.11  1.00     RTP     0.04 -0.04 -0.14 -0.03  1.00
```

MARKET/USDINR and MARKET/BRENT correlations are stable in sign and
magnitude across frequency (~-0.23 to -0.29); GOLD's and RATES_PROXY's
correlations with everything else shrink toward zero weekly, consistent
with those being the noisiest factor pair at daily frequency (aliased
short-horizon noise that partially cancels over a week).

FactorShock (Nifty -12%, Brent +20%, propagate=true) implied moves:

| factor | Daily | Weekly |
|---|---|---|
| USDINR | +2.42% | +1.63% |
| GOLD | -6.34% | -1.12% |
| RATES_PROXY | -1.79% | -0.67% |
| portfolio P&L (INR) | -1,175,389 | -1,132,802 |

RiskDecomposition shares (`fraction_of_vol`):

| | Daily | Weekly |
|---|---|---|
| portfolio vol (annualized) | 15.52% | 13.50% |
| MARKET | 82.4% | 82.2% |
| specific risk | 18.8% | 17.4% |
| USDINR / BRENT / GOLD / RATES_PROXY (combined) | -1.2% | 0.4% |

Vol estimates are reasonably close (15.5% vs 13.5%); the systematic/specific
split is nearly identical (~82/18 both ways), which is the most
frequency-robust number here. **Not changing the default** pending review —
Daily/252 stays the default per the spec until you confirm one way or the
other; Weekly's much smaller effective sample (260 weeks in a 5y fetch vs.
1235 days) makes its factor-covariance and small-beta estimates visibly
noisier, which shows up as instability in the USDINR/RATES_PROXY betas
above.

### `INR=X` vs. `USDINR=X`

Checked both tickers against the `^NSEI` master calendar over the same 5y
fetch: identical raw observation counts (1300), identical missing-date sets
relative to NSEI (149 dates each), and identical gap-run-length histograms
(all singleton 1-day gaps, no clustering — `Counter({1: 149})` for both).
No behavioral difference; kept `INR=X` (already in use, and the shorter of
the two equivalent tickers).

## Trace additions for explainability

`FactorShockOutput` (nested in every `FactorShock` `EvidenceTrace`) now
also carries:

- `factor_correlation`: the fit-time factor correlation matrix
  (`FACTOR_NAMES` order), so a reader can see e.g. MARKET/BRENT correlation
  without recomputing it from the covariance.
- `conditional_coefficients`: `F_uk * F_kk^-1`, keyed
  `implied_factor -> { given_factor: coefficient }`. Each implied move is
  exactly `Sum_k coefficient_k * given_log_shock_k`, so a reader can
  attribute, say, "why did GOLD_USD move -6.6% log?" to specific coefficient
  x given-shock products instead of trusting an opaque number.
- `gold_inr_implied_move`: gold priced in INR is `GOLD_USD * USDINR`, so its
  log return is the sum of the `GOLD_USD` and `USDINR` log shocks (given or
  implied, whichever applies). The fitted `GOLD_USD` factor alone excludes
  the rupee move a domestic gold holder actually realizes, so this is
  reported as a separate, clearly-labelled derived field rather than
  folded into `GOLD_USD`.
- `model_params.frequency` / `model_params.window_periods` (added in the
  frequency changes above) record which frequency/window the fit used.

The `GOLD` factor label is renamed `GOLD_USD` everywhere (`FACTOR_NAMES`,
`factor_returns` keys, `shocks_pct` keys, betas, trace output) to make
explicit that it is USD-denominated gold, not INR-denominated gold — see
`gold_inr_implied_move` above for the INR-denominated derived move. The
Yahoo ticker constant (`data::GOLD`, `"GC=F"`) is unchanged; only the
factor *label* moved.

## Tests

```
cargo test --workspace
```

All tests use synthetic in-memory `MarketData` (see `tests/common/mod.rs`) —
no network access required.

## Dependencies and why

| Crate | Why |
|---|---|
| `nalgebra` | Matrix algebra for OLS (SVD-based pseudo-inverse solve), Ledoit-Wolf shrinkage, and portfolio covariance/Euler decomposition. |
| `serde` / `serde_json` | Serialize every experiment input/output and the `EvidenceTrace` to JSON. |
| `schemars` (with `chrono` feature) | JSON Schema derivation for `Experiment` and trace types, for the future agent/tool-calling layer. |
| `chrono` | Calendar-correct date handling for the NSE trading calendar and price series. |
| `thiserror` | Structured `ComputeError` variants instead of stringly-typed errors. |
| `reqwest` (blocking + async, `rustls-tls`, `default-features = false`) | HTTP client for Yahoo Finance's chart endpoint (blocking, `compute`) and Gemini's `generateContent` endpoint (async, `agent`). Switched from the default `native-tls`/OpenSSL backend to `rustls-tls` in the server checkpoint — see "Docker" below for why. |
| `csv` | Reading/writing the on-disk price cache. |
| `clap` (derive) | CLI argument parsing (`--refresh`, `--cache-dir`) for the `experiment` binary — not in the original justified list, added because the spec requires a `--refresh` flag and hand-rolled arg parsing would be worse than a one-line derive. |
| `good_lp` (`clarabel` backend only, `default-features = false`) | The Rockafellar-Uryasev LP for `CvarRebalance`. `clarabel` is a pure-Rust interior-point solver (no C/C++ toolchain or system solver binary needed), matching the design note's preference for build simplicity over a HiGHS/CBC binding. |
| `tokio` (`rt-multi-thread`, `macros`, `time`) | Async runtime for `agent`'s Gemini calls (`reqwest`'s async client) and retry backoff (`tokio::time::sleep`); `rt-multi-thread`/`macros` also back `#[tokio::main]`/`#[tokio::test]`. |
| `async-trait` | `agent::gemini::GeminiClient` is an async trait (needed so `parse`/`narrate`/`pipeline` can be generic over a real HTTP client or a test mock); stable Rust doesn't yet support `async fn` in traits used as trait objects/generically without this. |
| `regex` | Number extraction in `agent::grounding` (lakh/percent/plain numeric tokens) — a hand-rolled parser would be far more error-prone for this than a well-tested regex engine. |
| `axum` | The HTTP framework for `server` — spec-named, and a natural fit given `tokio`/`tower` are already in the dependency tree via `reqwest`/`agent`. |
| `tracing` / `tracing-subscriber` (`json`, `env-filter`) | Structured JSON request logging (method, path, status, latency), `RUST_LOG`-overridable level, per spec. |
| `tower` (dev-only, `util`) / `http-body-util` (dev-only) | `ServiceExt::oneshot` and response-body reading for `server`'s route tests, run in-process against the axum `Router` with no real network listener. |

## Judgment calls

- **Ticker substitution:** `TATAMOTORS.NS` 404s on Yahoo's chart endpoint
  (Tata Motors demerged its commercial-vehicle business in 2025); the
  example portfolio uses `TMPV.NS` (Tata Motors Passenger Vehicles, the
  surviving Yahoo-listed entity) instead. This is a data-availability
  substitution for the example only, not a code change.
- **Calendar trimming:** after forward-filling non-NSE series, any leading
  calendar dates where a series still has no observed *or* filled price
  (i.e. before that series' first print) are trimmed from the whole
  dataset, so every series has a valid price for the entire surviving
  window. This is not explicitly specified but is required for `MarketData`
  to have rectangular, gap-free return arrays.
- **OLS solver:** used SVD-based pseudo-inverse (`nalgebra`'s `svd().solve`)
  rather than a normal-equations solve, for numerical stability if factors
  are ever near-collinear over a given window.
- **Ledoit-Wolf target:** implemented the identity-target (not the
  single-index or constant-correlation target) variant of Ledoit & Wolf
  (2004), since the spec asks for "scaled identity" specifically; `rho_hat`
  is taken as 0, which holds for that target (no off-diagonal estimation
  error to correlate against).
- **Annualization:** applied in exactly one place —
  `model::annualize_matrix` / `annualize_scalar` — called only when
  building the final `Sigma` / `F` used by experiments; the OLS fit and
  Ledoit-Wolf shrinkage themselves operate entirely on daily returns.
- **`RATES_PROXY` sign:** defined as `^NSEBANK` return minus `^NSEI` return
  (bank-sector excess return over the market), matching the checkpoint spec
  verbatim.
- **FactorShock attribution vs. per-holding P&L:** both are reported in INR
  and are cross-checked as invariants (`sum(per_holding.pnl_inr) ==
  portfolio_pnl_inr` and `sum(factor_attribution_inr) ==
  portfolio_pnl_inr`), which hold exactly because the shock model is linear
  with no intercept term (noted explicitly in every trace's `outputs.note`).

## CvarRebalance

Implemented in `src/cvar.rs` via `good_lp` + the `clarabel` backend (pure
Rust, no external solver binary), per the design note above with these
amendments:

**Formulation.** Rockafellar-Uryasev CVaR minimization, but with the
`1/(S(1-beta))` coefficient replaced by `1/k` where `k =
round(S*(1-beta))` is an **integer tail scenario count** rather than the
continuous `S(1-beta)`. For equally-weighted historical scenarios this
makes the LP exactly equivalent to "minimize the average of the k worst
historical losses" — at optimum, `zeta*` is exactly the k-th worst loss
(VaR) and the objective is exactly the mean of the k worst losses (CVaR),
which is what lets the "LP objective == directly-computed CVaR" invariant
below hold to solver tolerance (~1e-10) rather than only approximately.

```
minimize   zeta + (1/k) * sum_s u_s
subject to u_s >= -(r_s . w) - zeta,   u_s >= 0,        for all s
           sum_i w_i = 1
           0 <= w_i <= per_name_cap                      (long-only + per-name cap)
           w_i = w0_i + buy_i - sell_i,  buy_i, sell_i >= 0
           sum_i (buy_i + sell_i) <= turnover_limit       (turnover, linearized)
```

**Scenarios.** Simple returns (`exp(log) - 1`) of the holdings' own
historical log returns (`data::MarketData.stock_returns`) — raw historical,
not factor-model-simulated, per the design note's recommendation. Full
available history by default; `window` (periods) is configurable.

**Historical VaR/CVaR, computed independently of the LP.** For a weight
vector `w` (before or after), `historical_stats` sorts the `k` worst
scenario losses `Loss_s = -(r_s . w)` directly from the scenario matrix and
reports `historical_var` (the k-th worst loss) and `historical_cvar` (their
mean) — a fresh computation from data + weights, not a copy of the solver's
reported objective, so `lp_objective_cvar` and `stats_after.historical_cvar`
are independent cross-checks of each other (see invariants below).

**Pre-solve feasibility check** (before ever calling the solver):
1. `per_name_cap * n_stocks >= 1` — otherwise long-only weights can never
   sum to 1.
2. A necessary lower bound on turnover: names already over `per_name_cap`
   must sell down to it, and — since weights must still sum to 1 — that
   sold capital must be bought back elsewhere, so turnover is at least
   `2 * sum_i max(0, w0_i - per_name_cap)`. If that exceeds `turnover_limit`,
   report infeasible without solving. (This is a *necessary*, not
   *sufficient*, condition; the LP solve remains the authoritative
   feasibility check for anything this doesn't catch.)

Either pre-solve failure, or the solver itself returning
`ResolutionError::Infeasible` / any other non-optimal status, produces a
structured result rather than a thrown error: `CvarRebalanceOutput.status`
(`"optimal"` | `"infeasible"` | `"solver_error"`) plus `diagnostics: Option<String>`,
both inside a normal `Ok(...)` `EvidenceTrace` — so a caller always gets a
citable trace, even for a failed rebalance, with the failing check recorded
as a `passed: false` invariant.

**Invariants:** weights sum to 1 (1e-9), turnover <= `turnover_limit` +
1e-6, and LP objective == directly-computed historical CVaR of the solution
(1e-6) — all three checked in `run_cvar_rebalance` and included in every
`EvidenceTrace.invariants`.

### Sample trace (10-stock example, cap 20%, turnover 30%, beta 0.95)

```
scenario_count: 1232, tail_scenario_count: 62
stats_before: { historical_var: 0.0136, historical_cvar: 0.0204 }
stats_after:  { historical_var: 0.0128, historical_cvar: 0.0187 }
lp_objective_cvar: 0.018721952952451708   (matches historical_cvar to 1.4e-14)
turnover: 0.300 (binding at the limit)
commission_cost_inr: 3000.0  (0.30 * 1e7 * 10bps)
weights_after: TMPV.NS -> ~0 (cut essentially to zero), ITC.NS -> 0.181,
               BHARTIARTL.NS -> 0.139 (both bid up), others near-unchanged
invariants: all 3 passed
```

Full trace: `cargo run -p compute --bin experiment -- crates/compute/examples/cvar_rebalance_nifty10.json`.

### Judgment calls specific to CvarRebalance

- **`commission_bps` default:** 10 bps (0.10%), a reasonable blended
  estimate for Indian equity delivery trades (brokerage + STT + other
  statutory charges); always caller-overridable, no default was specified
  in the brief.
- **`per_name_cap` applies uniformly to every name**, including in the
  `n * cap >= 1` feasibility check — with few holdings and a tight cap,
  this can force residual weight onto a name the optimizer would otherwise
  zero out (confirmed in `heavy_tail_asset_is_cut_to_near_zero_when_turnover_allows`,
  where the cap had to be raised to 1.0 to let the test isolate the
  CVaR-driven effect from the cap-driven one).
- **`model_params.factor_names`/`shrinkage_intensity` don't apply** to this
  experiment (no factor model is fit); left as an empty vec / 0.0 rather
  than adding an experiment-specific trace variant, with an explicit note
  in `outputs.note` saying so.

## Agent pipeline (`agent::pipeline::run`)

`compute` is unchanged in this checkpoint. `agent` adds the NL -> Experiment
-> EvidenceTrace -> grounded-narration pipeline `server` will call:

```
agent::pipeline::run(client, user_message, portfolio)
  -> parse::parse_experiment   (1 Gemini call, function-calling)
  -> compute_trace              (real compute::data/model/experiments/cvar call,
                                  on a blocking thread via tokio::task::spawn_blocking)
  -> grounding::grounded_narrate (1-3 Gemini calls: narrate, then up to 2 grounding retries)
```

### Gemini client (`agent::gemini`)

`GeminiClient` is an async trait with one method, `generate`; `HttpGeminiClient`
is the real POST-to-`generateContent` implementation (API key from
`GEMINI_API_KEY`, retrying up to 3 attempts with exponential backoff — 250ms,
500ms — on HTTP 429/503, surfacing any other status as `GeminiError::Status`).
Being a trait (not a concrete struct) is what makes `parse`/`narrate`/`pipeline`
testable without network access: tests supply a `MockGeminiClient` with a
queue of canned responses instead.

### Schema (`agent::schema`)

`experiment_json_schema()` derives a JSON Schema from `compute::experiments::Experiment`
via `schemars::schema_for!`, then strips the `portfolio` field from every
variant's `properties`/`required` (recursively, including inside `definitions`)
before it's shown to Gemini — see judgment calls below for why.

### Grounding check (`agent::grounding`)

The core piece. `extract_numbers` recognizes three token shapes, tried in
priority order (most specific first) via one alternation-based regex, so
e.g. `"₹11.7 lakh"` is consumed whole rather than also matching `"11.7"`
generically:

1. **Lakh:** `(?:₹\s*)?(-?[\d,]+(?:\.\d+)?)\s*lakh\b` → `value * 100_000`.
2. **Percent:** `(-?[\d,]+(?:\.\d+)?)\s*%` → `value / 100`.
3. **Plain:** `(?:₹\s*)?(-?[\d,]+(?:\.\d+)?)` → `value` as-is.

The minus sign accepts both ASCII `-` and Unicode `−` (U+2212), since
Gemini (and Indian financial prose generally) uses both; commas are
stripped before parsing, which normalizes both Western (`1,175,389`) and
Indian (`11,75,389`) digit grouping identically. `numeric_leaves` flattens
every numeric JSON leaf out of `serde_json::to_value(&trace)` (recursively;
strings/bools/nulls ignored). A narration number matches a trace number if
`|a - b| / max(|a|, |b|, 1e-9) <= 0.02` (the `1e-9` floor avoids
division-by-zero when both are ~0, without changing behavior anywhere the
spec's 2% figure actually matters).

`grounded_narrate` calls `narrate`, checks, and — if any number is
unmatched — retries up to twice with the failing tokens named in an
appended system instruction, per the spec's exact retry wording. If still
failing after 2 retries (3 calls total), it returns the last narration with
`grounding_warnings` populated rather than suppressing the response.

### Sample `PipelineResult` — FactorShock, live 10-stock portfolio

No `GEMINI_API_KEY` is available in this environment, so `agent::pipeline::run`
below used a **scripted mock Gemini client** (`crates/agent/examples/demo_pipeline.rs`)
for the two Gemini calls, while the compute step hit live Yahoo data exactly
as the `experiment` CLI does. The narration text was written by hand,
honoring the narrate system prompt's rules, then run through the *real*
`grounding::check_grounding` (not mocked) — this is a demonstration of the
grounding machinery on a genuine trace, not a live Gemini call:

```
$ cargo run -p agent --example demo_pipeline
```

**Parsed experiment:** `FactorShock { shocks_pct: {MARKET: -12.0, BRENT: 20.0}, propagate: true, ... }`
(portfolio injected from the caller, not from Gemini's args).

**Trace summary** (`trace.outputs.result`, full JSON via the command above):

```
given_shocks:    MARKET  -12.00% (simple)     BRENT  +20.00% (simple)
implied_shocks:  USDINR  +2.53%                GOLD_USD  -6.36%             RATES_PROXY  -1.80%
portfolio_pnl_inr: -1,177,846.39
invariants: sum(per_holding.pnl_inr) == portfolio_pnl_inr        -> passed
            sum(factor_attribution_log_inr) == portfolio_log_pnl_inr -> passed
```

**Narration:**

> A -12% shock to MARKET combined with a +20% shock to BRENT produces a
> portfolio loss of approximately -1,177,846 INR on this ten-stock Nifty
> portfolio. Because the user specified only these two factors, the
> remaining three factors are model-estimated from this portfolio's return
> history via the factor covariance: USDINR is implied to move +2.53%,
> GOLD_USD -6.36%, and RATES_PROXY -1.80%, each shown separately from the
> two given shocks above. These implied moves are not user inputs; they
> follow from the historical correlation between MARKET, BRENT and the
> other factors. The loss is dominated by the MARKET shock, given the
> portfolio's substantial equity beta exposure.

**`grounding_warnings`: `[]`** — every number in the narration matched a
trace value on the first attempt; no retry was needed. See "flag
immediately if grounding_warnings fires" below.

### Judgment calls

- **`portfolio` stripped from the schema Gemini sees**, not just documented
  in the function description: with `Portfolio` present as a required field
  in each variant's schema but the model told "don't extract this," a small
  model can still feel obligated to invent a plausible-looking (wrong)
  portfolio object, wasting tokens and risking a parse failure if its
  shape is malformed. Stripping it removes the temptation entirely; the
  real portfolio is always spliced into the function-call args
  (`args["portfolio"] = ...`) before deserializing into `Experiment`,
  overwriting whatever Gemini did or didn't include.
- **2% relative tolerance, not absolute:** an absolute tolerance would be
  either too loose for small numbers (betas, shrinkage intensities are
  often < 0.1) or too tight for large INR amounts (portfolio values in the
  millions), so every comparison is scaled by the larger of the two
  magnitudes (floored at `1e-9` to stay finite at/near zero).
- **Number-matching, not phrase-matching:** the grounding check verifies
  every *number* the narration states is real, not that the *sentence*
  containing it is accurate (e.g. it can't catch a narration that swaps
  which factor a correct number belongs to). This matches the spec's
  literal ask ("every number... must appear verbatim") but is worth naming
  as a limitation — a stronger check would need entity/number pairing,
  out of scope here.
- **Blocking compute on `spawn_blocking`:** `compute`'s data/model/CVaR
  path is synchronous (blocking `reqwest`, CPU-bound linear algebra/LP
  solve); `pipeline::run` moves it to a blocking thread rather than making
  `compute` itself async, since `compute` has no other reason to depend on
  an async runtime and the CLI (`experiment`) needs to stay synchronous too.
- **`commission_bps`/`window`/`frequency` defaults are unchanged** from the
  compute-layer checkpoints; `agent` doesn't override or second-guess them.

## Server (`server::main`)

```
GET  /health              -> { status: "ok", version } (200)
POST /experiment           -> { portfolio, experiment } in, EvidenceTrace out
POST /ask                    -> { portfolio, message } in, AskResponse out
GET  /report/{result_id}      -> PDF, rendered from a stored EvidenceTrace
POST /portfolio/upload         -> multipart CSV/XLSX in, { portfolio, ... } out
GET  /*                          -> embedded single-page UI (static/index.html)
```

Both `POST /experiment` and `POST /ask` now also insert a `RiskSnapshot`
into the persistent `store::SnapshotStore` on every successful call (see
"PDF reports, snapshot store, and CVaR cap defaulting" below) — `/ask`
returns the row's id as `result_id`; `/experiment`'s response shape is
unchanged (still the bare `EvidenceTrace`), so its snapshot is a side
effect, not something the caller gets an id for directly.

### `POST /portfolio/upload` (`server::upload`)

Accepts `multipart/form-data` with one field, `file` (`.csv` or `.xlsx`,
first sheet only for XLSX). Two accepted column layouts, detected from the
header row (case-insensitive): `ticker,weight` (weight-based) or
`ticker,shares,avg_price_inr` (value-based — `value_inr = shares *
avg_price_inr` per holding, `weight = value_inr / total_value_inr` rounded
to 6 decimals, `total_value_inr` recomputed as the sum). A bare NSE-looking
ticker with no exchange suffix (all uppercase letters/digits, no `.`) gets
`.NS` appended, reported back in `tickers_normalised`. Runs the same
`validate_portfolio` check as `/experiment`/`/ask` before responding, so a
malformed upload (weights not summing to 1.0, fewer than 2 holdings, an
unsupported extension, missing columns, or an empty file) gets the same
400 `{error, code}` shape those routes use.

**Judgment call:** weight-based CSVs carry no portfolio value at all (only
ticker + weight), so there's nothing to derive `total_value_inr` from.
Defaults to 1,000,000 INR — this codebase's existing demo convention
elsewhere (`sample_portfolio` helpers throughout the test suite) — which
the caller should overwrite before using the returned `Portfolio` for
anything that cares about real INR amounts (P&L, commission cost, etc.).

`ExperimentRequest.experiment` is the tagged `Experiment` variant's JSON
*minus* `portfolio` (e.g. `{"type": "FactorShock", "shocks_pct": {...}}`) —
the same "caller supplies portfolio separately" pattern as `agent::parse`
(§ Agent pipeline). The route handler splices `req.portfolio` into that
JSON before deserializing into `compute::experiments::Experiment`, so a
caller never has to repeat the portfolio inside the experiment object.

**Validation** (`validate::validate_portfolio`, shared by both POST routes):
at least 2 holdings, every weight > 0, `total_value_inr` > 0, weights sum
to `1.0 +/- 0.01`. Tickers are deliberately not checked — `compute::data`
already errors clearly on a bad one, and a second ticker-format check here
would just be one more place to keep in sync. Every failure returns 400
with `{"error": "...", "code": "invalid_portfolio"}`.

**Error mapping** (`backend::BackendError` -> `error::ApiError`):

| Backend error | HTTP status | `code` |
|---|---|---|
| `ParseError::Unrecognised` (the model's one-sentence explanation) | 422 | `unrecognised_request` |
| any `compute::ComputeError` | 500 | `compute_error` |
| any Gemini error other than a missing API key (already refused at startup) | 503 | `gemini_unavailable` |
| malformed/missing-field JSON body (`AppJson`'s rejection) | 400 | `invalid_json` |
| anything else unexpected | 500 | `internal_error` |

**Testability:** routes depend on a `Backend` trait (`run_experiment`,
`run_ask`), not on `compute`/`agent` directly. `RealBackend` wraps live
calls (via `tokio::task::spawn_blocking` for the blocking compute path);
`src/tests.rs` supplies a `MockBackend` with canned results, so the 5
required tests need no network access to Yahoo Finance or Gemini.

**Logging:** one `tracing::info!` event per request (`method`, `path`,
`status`, `latency_ms`), emitted by a small `axum::middleware::from_fn`
wrapper rather than `tower_http::trace::TraceLayer`, so the exact fields
logged match the spec precisely instead of `TraceLayer`'s span-based
defaults. `tracing_subscriber::fmt().json()`, level from `RUST_LOG`
(default `info`).

### UI (`static/index.html`)

Single file, embedded via `include_str!` — no build step, no npm, no
external fonts/scripts, plain system fonts, `#2563EB` as the one accent
colour, two-column layout above 900px. Portfolio builder (add/remove
ticker/weight rows + a client-side Validate button running the same rule
as the server's `validate_portfolio`), a prompt box with a Direct toggle
that reveals structured params per experiment type, a submit button with a
spinner, a narration panel (yellow banner when `grounding_warnings` is
non-empty, exact wording per spec), and a collapsible evidence-trace
`<pre>` block with copy-to-clipboard.

## Sample responses

### `GET /health`

```json
{"status":"ok","version":"0.1.0"}
```

(Live, from a running binary — `GEMINI_API_KEY=<dummy> PORT=8099 ./target/debug/server`.)

### `POST /ask` (mocked pipeline — see the agent checkpoint's note on no live Gemini access)

Same approach as the agent checkpoint's demo: `src/tests.rs` used a
`MockBackend` returning a `PipelineResult` built from a real FactorShock
input and a hand-written, grounding-checked narration.

```json
{
  "experiment": {
    "type": "FactorShock",
    "portfolio": {
      "holdings": [
        {"ticker": "RELIANCE.NS", "weight": 0.6},
        {"ticker": "TCS.NS", "weight": 0.4}
      ],
      "total_value_inr": 1000000.0
    },
    "shocks_pct": {"BRENT": 20.0, "MARKET": -12.0},
    "propagate": true,
    "linear_approximation": false,
    "frequency": "Daily",
    "window": null
  },
  "trace": {
    "experiment": "RiskDecomposition",
    "data_window": {"frequency": "Daily", "window_periods": 252, "start": "2025-09-16", "end": "2026-09-24"},
    "data_quality": {"date_range_start": "2021-09-27", "date_range_end": "2026-09-24", "trading_days": 1234, "per_series": []},
    "model_params": {"frequency": "Daily", "window_periods": 252, "factor_names": ["MARKET"], "shrinkage_intensity": 0.0374, "annualization_factor": 252.0},
    "inputs": {},
    "outputs": {"result": {"portfolio_vol_annualized": 0.1552}},
    "invariants": [],
    "engine_version": "0.1.0"
  },
  "narration": "A -12% shock to MARKET combined with a +20% shock to BRENT produces a portfolio loss on this portfolio. USDINR, GOLD_USD and RATES_PROXY move as implied, model-estimated moves, shown separately from the two given shocks.",
  "grounding_warnings": []
}
```

(`trace` here is a small stand-in fixture from the test suite, not a full
live trace — the point of this response is the `AskResponse` *shape*, not
new trace content; a full trace looks exactly like the ones in
`crates/compute/examples/*.json`.)

## Docker

```
docker build -t drift-risk-copilot:server .
docker run -e GEMINI_API_KEY=<key> -p 8080:8080 drift-risk-copilot:server
```

Verified locally (`docker build` + `docker run` + a real `GET /health`
against the running container). Final image: **~12.5MB** (`docker save
drift-risk-copilot:server | wc -c` = 13,155,328 bytes; `docker images`'
own ~62MB figure includes buildx provenance/attestation metadata that
isn't part of the runtime image) — comfortably under the 50MB target.

### Build failures hit along the way (reported verbatim, per instructions)

**1. `rust:1.82-slim` as originally specified:**

```
error: failed to parse manifest at `/usr/local/cargo/registry/.../clap_lex-1.1.1/Cargo.toml`
Caused by:
  feature `edition2024` is required
  The package requires the Cargo feature called `edition2024`, but that feature is not
  stabilized in this version of Cargo (1.82.0 (8f40fc59f 2024-08-21)).
```

**2. `rust:1.85-slim`** (edition2024 stabilized in Cargo 1.85, tried next):

```
error: rustc 1.85.1 is not supported by the following packages:
  icu_collections@2.3.0 requires rustc 1.88
  icu_locale_core@2.3.0 requires rustc 1.88
  icu_normalizer@2.3.0 requires rustc 1.88
  ... (icu_normalizer_data, icu_properties, icu_properties_data, icu_provider @ rustc 1.88)
  idna_adapter@1.2.2 requires rustc 1.86
```

**3. `rust:1.90-slim`** — built successfully.

### Judgment calls

- **Rust version bumped from 1.82 to 1.90 (not the spec's exact pin).**
  This `Cargo.lock` was generated with a current toolchain (rustc 1.95),
  which resolved several transitive dependencies (`clap_lex`, the
  `icu_*` family via `idna`/`url`) to versions with an MSRV well above
  1.82. Re-pinning those dependencies to older, 1.82-compatible versions
  was the other option, but would mean carrying a second, artificially
  old dependency set for Docker only, diverging from what's actually
  tested locally — worse than moving the build image forward to a
  version that supports the lockfile that's actually shipped. 1.90 is
  the smallest bump that got a clean build in this environment (verified
  by trying 1.82, then 1.85, then 1.90 — see above).
- **`rustls-tls` instead of the default `native-tls`/OpenSSL backend for
  `reqwest`** (workspace-wide, both `compute` and `agent`): this is what
  actually makes `gcr.io/distroless/cc-debian12` viable as specified.
  `distroless/cc` ships glibc + libstdc++ but **no OpenSSL** — a normal
  `reqwest` build (native-tls) dynamically links `libssl.so`/`libcrypto.so`
  at runtime and would fail to start in that image. The alternative the
  checkpoint explicitly offered — static musl linking — would need either
  `openssl-sys`'s `vendored` feature (a full C build of OpenSSL inside
  the musl cross-build, plus a musl target + `musl-tools`) or switching to
  `rustls-tls` anyway; since `rustls-tls` alone already solves the
  OpenSSL-in-distroless problem with a normal glibc build and zero extra
  toolchain setup, it's the cleaner of the two options the spec allowed
  ("pick whichever is cleaner"). Confirmed working end-to-end: `--refresh`
  against live Yahoo Finance still succeeds after the switch (rustls
  validates Yahoo's cert chain fine via `webpki-roots`).
- **The `touch` before the final `cargo build` in the Dockerfile is
  load-bearing, not decorative** (see the Dockerfile's own comment for the
  full story): without it, `cargo build --release -p server` after copying
  the *real* source over the dummy-stub source finished in 0.08s doing
  nothing, and the resulting image's `/server` silently ran the dummy
  `fn main() {}` — exit code 0, no log output, port never opened. Caught by
  actually running the built container and hitting `/health`, not by
  trusting a "Finished" message. `find ... -exec touch {} +` on the real
  source before rebuilding fixes it (BuildKit's `COPY` doesn't always
  advance mtimes past cargo's fingerprint records from the earlier dummy
  build).
- **Dummy stubs cover every declared target in every workspace member's
  Cargo.toml** (`compute`'s `lib.rs` *and* its `bin/experiment.rs`,
  `agent`'s `lib.rs`, `server`'s `main.rs`), not just the crate(s) actually
  being built — Cargo parses every workspace member's manifest (for
  lockfile/dependency-graph resolution) even when building a single
  package with `-p`, and errors if a declared target's source file is
  missing.

## Cloud Run deploy

```
gcloud run services replace cloudrun.yaml --region=asia-south1
gcloud run services add-iam-policy-binding drift-risk-copilot \
  --region=asia-south1 --member=allUsers --role=roles/run.invoker
```

`cloudrun.yaml` references the image as `gcr.io/PROJECT_ID/drift-risk-copilot:latest`
(substitute the real project ID, and push the image built above to that
path first) and reads `GEMINI_API_KEY` from Secret Manager
(`secretKeyRef: {name: drift-gemini-key, key: latest}` — the secret must
exist and the Cloud Run service's runtime service account needs
`roles/secretmanager.secretAccessor` on it before `services replace` will
succeed). `minScale: 0` / `maxScale: 3`, 512Mi/1 CPU, `timeoutSeconds: 300`
(bumped from the originally-specified 60 -- confirmed live that a real
`/ask` request hit Cloud Run's own gateway timeout at exactly 60.1s: `/ask`
can chain up to 4 sequential Gemini round trips -- parse, then narrate,
then up to 2 grounding-retry narrate calls, each itself retrying up to 3x
internally on 429/503 -- which routinely exceeds 60s under real API
latency/rate-limiting, well beyond just the cold-start compute call the
original 60s was sized for).

## Regime-conditional factor covariance (`compute::regime`, `model::ModelConfig`)

A 3-state Gaussian HMM (`compute::regime`, Baum-Welch fit from scratch, no
external HMM crate) on Nifty (`^NSEI`) daily log returns, used to split the
factor covariance `F` by market regime instead of always pooling the full
window.

**Always-on as of the session that added persistent snapshots and portfolio
upload**: regime-conditioning used to be an opt-in `regime_covariance: bool`
field on `FactorShockInput`/`RiskDecompositionInput`/`CvarRebalanceInput`,
defaulting to `false` everywhere. That field is gone -- every request now
gets a regime-conditional fit unconditionally, and every `EvidenceTrace`'s
`model_params.regime_state` is always populated (never `None`). The
mechanics below (states, fallback, per-experiment wiring) are unchanged;
only the "was this requested" branch was removed.

### The model

- **States, always reported in ascending-emission-variance order**: 0 =
  Bull (lowest vol), 1 = Bear (medium), 2 = Crisis (highest). The relabel
  happens once, after Baum-Welch converges, by sorting the three fitted
  states by their final emission variance — so which *internal* state index
  the EM fit happened to land on for "the high-vol regime" never matters;
  the label always does.
- **Forward/backward**: scaled (Rabiner 1989) — `alpha_hat_t(k)` normalized
  to sum to 1 at every `t`, with `log P(O) = sum_t ln(c_t)`; `beta` scaled
  by the *same* `c` array so `alpha_hat_t(k) * beta_t(k) == gamma_t(k)`
  exactly, verified by a test that checks this sums to 1 (within 1e-9) at
  every `t`, not just the final one.
- **Convergence**: log-likelihood improvement `< 1e-6` or 500 iterations.
- **Viterbi**: log-space, for the full regime-assignment sequence used to
  split factor returns.
- `smoothed_probs` (`gamma_T`, the *current* regime distribution) come with
  `smoothing_note: "full-history smoothed, not suitable for live trading
  signals"` — a full-history smoother uses future information (everything
  up to `T`) to estimate the state at `T`, which is fine as a point-in-time
  snapshot but not what a causal, real-time signal would look like.

### Wiring into the factor model

`model::ModelConfig { window, frequency }` is the only fit configuration
now -- `fit_factor_model_with_config` was renamed to `fit_factor_model` and
the old non-regime `fit_factor_model(data, tickers, window, frequency)`
thin wrapper was removed, since there is no longer a non-regime path to
wrap. Every caller (`agent::pipeline`, `compute`'s own CLI, `cvar`) was
updated to the new signature.

The HMM fits on the *same* window's `MARKET` factor series (already
Nifty's own log returns, no separate fetch), factor returns are split by
the Viterbi sequence, and each regime gets its own Ledoit-Wolf `F_k`.
`FactorModel.factor_covariance_daily` — the field every existing
`factor_covariance()`/`stock_covariance()` call already reads — is always
`F_{current_regime}`, so **RiskDecomposition needs zero special-casing of
its own math**: it just calls those methods, unconditionally regime-aware.
All three regimes' `F_k` remain available via `factor_covariance_for_regime(k)`
/ `stock_covariance_for_regime(k)`, which is what `FactorShock`'s
`crisis_comparison` uses.

**Fallback**: a regime with fewer than `MIN_REGIME_OBSERVATIONS` (30) days
in the window uses the full-window `F` instead of its own (too few
observations to shrink meaningfully), and a warning is recorded in
`model_params.regime_fallback_warnings`. This is a real, live-observed
case, not just a hypothetical — see the live run below.

### Per-experiment behaviour

- **RiskDecomposition**: vol and Euler contributions always use
  `F_{current_regime}` (see above). `model_params.regime_state` records
  which regime.
- **FactorShock**: whenever the current regime isn't already Crisis,
  `outputs.result.crisis_comparison` reruns the same shock (including
  conditional propagation) using `F_crisis` instead of `F_current`, so a
  reader can see "how much worse would this look under crisis-regime
  correlations" without a second request. Absent (not zeroed) when the
  current regime already is Crisis, since that comparison would be a no-op.
- **CvarRebalance**: per the design note, the LP and feasibility checks
  always use historical scenarios directly, *never* a factor-model
  covariance — so regime can't gate optimality. It instead always fits a
  regime-conditional factor model purely to report
  `regime_portfolio_vol_annualized_{before,after}` (`sqrt(w' Sigma_regime
  w)` for `weights_before`/`weights_after`) as a parametric cross-check
  alongside the historical CVaR/VaR, with `model_params.regime_state`
  recording which regime. This fit uses the CVaR experiment's own resolved
  scenario `window` (not a separate fixed default), and degrades to `None`
  regime fields (rather than failing the whole request) if that window is
  narrower than the HMM's 90-observation minimum.
- **PortfolioPerformance**: never fits a factor model (unchanged), but now
  reports `outputs.result.regime_label` (informational only) from a
  standalone HMM fit on the window's own MARKET series, and populates
  `model_params.regime_state` the same way. `None` only if that standalone
  fit itself fails.

### Live run (10-stock Nifty portfolio)

**HMM fit**: `n_iter: 329`, `log_likelihood: 884.996`.

**Current regime**: **Bull** — `smoothed_probs: [0.907, 0.090, 0.003]`
(Bull/Bear/Crisis), `obs_count_per_regime: [184, 10, 58]` (Viterbi, out of
the 252-day window).

**Fallback warning actually fired** (not just tested synthetically):
`"regime_1 (Bear) has only 10 observations, fell back to full-window
covariance"` — Bear was too thin a slice of this particular 252-day window
to shrink its own `F`.

**RiskDecomposition, `portfolio_vol_annualized`**, captured from the
session that first added regime-conditioning (back when it was still
opt-in via `regime_covariance: bool`; the flag itself is gone now, see
above, but the comparison is still the right mental model for what
regime-conditioning does):

| | value |
|---|---|
| full-window `F` (no regime split) | 0.1549 |
| regime-conditional `F` (Bull) | 0.1174 |

Meaningfully lower under the Bull-regime `F` than the full-window `F`, as
expected — the window's Bear/Crisis days pull the full-window covariance up.

**FactorShock (Nifty −12%, Brent +20%)**, current regime Bull, so
`crisis_comparison` is present:

| | current regime (Bull) | `crisis_comparison` |
|---|---|---|
| `portfolio_pnl_inr` | −1,224,017 | −1,164,320 |
| implied `USDINR` | +2.05% | +2.17% |
| implied `GOLD_USD` | −4.15% | −6.02% |
| implied `RATES_PROXY` | +0.02% | −2.38% |

Both invariants passed in both runs. Reproduce: `cargo run --bin
experiment -- crates/compute/examples/risk_decomposition_nifty10_regime.json`
/ `factor_shock_nifty10_regime.json`.

### Tests

`compute::regime`'s own unit tests (synthetic 3-segment low/medium/high-vol
data): `gamma` sums to 1 at every `t`; Viterbi recovers each segment with
>85% accuracy; state labels come out variance-ordered regardless of which
order the segments appear in the data (a deliberately *not* variance-sorted
order — high, low, medium). `model`'s own tests inject a synthetic Viterbi
sequence directly (rather than coaxing a real HMM fit into an unlucky
split) to test the <30-obs fallback deterministically. `experiment_tests.rs`
covers RiskDecomposition always using a regime-conditional `F` that differs
across regimes, and FactorShock's `crisis_comparison` presence/absence by
regime. Two tests need network and are `#[ignore]`d by default: `F` is PSD
for all three regimes on real NSEI data (`cargo test -p compute --test
model_tests -- --ignored`), and every experiment type's trace has a
non-null `regime_state` on real data (`cargo test -p compute --test
always_on_regime_tests -- --ignored`). Both confirmed passing.

### Judgment calls

- **k-means init clusters on `|returns|`, not raw signed returns** — found
  live, not anticipated: since all three regimes are roughly zero-mean,
  clustering on signed values just splits points by *direction*
  ("very negative" / "near zero" / "very positive"), which has nothing to
  do with volatility regime. This made Baum-Welch converge to a poor local
  optimum on the very first synthetic test run (two of three fitted states
  ended up with similar variances, differentiated mostly by mean, on data
  with three well-separated *true* variances and zero true mean
  everywhere). Clustering on magnitude fixed it immediately; final
  per-state means/variances are still computed from the original signed
  data within each magnitude-assigned cluster.
- **30-observation fallback threshold**: not derived from anything more
  principled than "Ledoit-Wolf shrinkage needs enough observations to
  estimate a 5x5 sample covariance's off-diagonal structure at all" — 30
  points for 5 factors is already a thin sample (6 obs/factor), but Ledoit-
  Wolf shrinkage is specifically designed to be robust in exactly that
  small-T regime (it's the paper's whole point), so this is closer to "no
  smaller than this" than a precisely justified number. The live run above
  shows it firing in practice (Bear regime, 10 obs), which is reassuring
  that the threshold isn't so low it never triggers.
- **CvarRebalance's regime output is informational-only by design**,
  per the checkpoint spec's explicit "not in the LP itself" — it would be
  straightforward to instead use `Sigma_regime` in the pre-solve
  feasibility checks too, but those checks don't reference any covariance
  at all currently (they're pure cap/turnover arithmetic), so doing that
  would be a bigger, unrequested change to what "feasible" means for this
  experiment.
- **Regime HMM window for CvarRebalance now reuses the CVaR experiment's own
  resolved scenario `window`**, not a separate fixed `frequency.default_window()`
  (252 daily) as in the original opt-in design. That fixed default broke
  the moment regime-conditioning became unconditional, since it required
  every ticker to have >= 252 observations even when the caller's own
  `window`/available data was shorter (hit immediately by this crate's
  existing short-synthetic-data CVaR tests, all of which use ~100
  observations). Reusing the resolved window also has the advantage of
  keeping the reported regime consistent with the same span the CVaR
  analysis itself runs over, rather than an unrelated hardcoded lookback.
- **CvarRebalance's regime fit degrades to `None` on failure** (including
  "window narrower than 90 observations", `regime::fit_hmm`'s minimum)
  instead of failing the whole request, since it's explicitly informational
  and not load-bearing for the LP/feasibility checks. The equivalent
  standalone fit in `PortfolioPerformance` (for `regime_label`) does the
  same. `FactorShock`/`RiskDecomposition` do *not* have this fallback —
  regime-conditioning is load-bearing there (`crisis_comparison`, and the
  vol/Euler-contribution math itself), so a fit failure there is a genuine
  error, propagated with `?` from `fit_factor_model`.

## Historical scenario presets and multi-turn `/ask`

### `GET /scenarios` (`compute::scenarios`)

Three fixed historical-scenario presets (`compute::scenarios::all_scenarios()`),
static reference data (not fit from live data): COVID Crash (Mar 2020),
IL&FS Contagion (Sep-Oct 2018), Taper Tantrum (May-Aug 2013). Each is a
`shocks_pct` map in the same simple-% units `FactorShockInput` already
takes, `propagate: false` since all five factors are given (propagation
would be a no-op). No portfolio or auth needed -- served directly off
static data. Verified with the server running locally:

```json
[
  {
    "id": "covid_crash",
    "name": "COVID Crash (Mar 2020)",
    "date_range": "Feb 19 – Mar 23, 2020",
    "shocks_pct": {"BRENT": -55.0, "GOLD_USD": 3.0, "MARKET": -38.0, "RATES_PROXY": -6.0, "USDINR": 8.5},
    "propagate": false
  },
  {
    "id": "ilfs_contagion",
    "name": "IL&FS Contagion (Sep–Oct 2018)",
    "shocks_pct": {"BRENT": 15.0, "GOLD_USD": 2.5, "MARKET": -15.0, "RATES_PROXY": 4.0, "USDINR": 7.0},
    "propagate": false
  },
  {
    "id": "taper_tantrum_2013",
    "name": "Taper Tantrum (May–Aug 2013)",
    "shocks_pct": {"BRENT": -5.0, "GOLD_USD": -18.0, "MARKET": -12.0, "RATES_PROXY": 5.0, "USDINR": 18.0},
    "propagate": false
  }
]
```

(Full descriptions/date ranges omitted above for brevity; see `crates/compute/src/scenarios.rs`.)

### Multi-turn `/ask` (`agent::conversation`, `agent::suggest`)

`AskRequest` gains an optional `conversation_history: Vec<ConversationTurn>`
(`role: "user" | "assistant"`, default empty -- an empty history produces
the exact same Gemini request shape as before this checkpoint, verified by
`empty_conversation_history_matches_pre_existing_request_shape`). Both the
`parse` (function-calling) and `narrate` Gemini calls prepend history as
prior turns before their own current-turn message, with `"assistant"`
mapped onto Gemini's own `"model"` role (`conversation::turn_to_content`).
This lets a second request like "now try with 25% turnover" resolve
against the first request's portfolio/experiment context without the
caller re-stating it, and lets narration refer back to a prior result
("compared to the previous scenario..."). **The grounding check itself is
unchanged** -- it only ever validates the current turn's narration against
the current turn's trace, never anything from history.

`AskResponse` gains `assistant_turn` (this turn's narration, pre-wrapped as
a `ConversationTurn` the caller appends to its own history for the next
request) and `suggestion` (see below).

### Proactive follow-up (`agent::suggest`)

After grounded narration, `pipeline::run` makes one more Gemini call
(`suggest::suggest_follow_up`) -- plain text, no function calling, no
grounding check (a question isn't a factual claim to verify) -- asking for
one actionable follow-up question a risk manager would naturally ask next.
Verified with a mocked pipeline end-to-end
(`cargo run -p agent --example demo_pipeline`):

> "A -12% shock to MARKET combined with a +20% shock to BRENT produces a
> portfolio loss of approximately -1,177,846 INR on this ten-stock Nifty
> portfolio. ... The loss is dominated by the MARKET shock, given the
> portfolio's substantial equity beta exposure."
>
> **suggestion:** "What if I cut my turnover budget to 20% instead?"

### Tests

`compute::scenarios`: exactly 3 scenarios; every `shocks_pct` key is a
valid `FACTOR_NAMES` entry. `agent`: `parse_experiment` passes
`conversation_history` as prior turns in the correct order (asserted via
`MockGeminiClient::last_request()`, a new test-support accessor that
records every request sent, not just responses returned); an empty history
produces the pre-existing single-turn request shape; `suggest_follow_up`
returns the mock's text as-is, including an empty string without erroring.
`server`: `GET /scenarios` returns 200 and an array of 3, each with
`id`/`name`/`shocks_pct`; `POST /ask` with a non-empty
`conversation_history` succeeds and the mock backend's received history is
asserted directly (2 turns, correct roles/content) via a `MockBackend` held
outside the router as `Arc<MockBackend>` (not just `Arc<dyn Backend>`) so
the test can inspect what it captured after the request completes.

### Judgment calls

- **`suggest` failures propagate as a `PipelineError`/500**, same as
  parse/narrate, rather than degrading `/ask` to a response with an empty
  `suggestion` on Gemini failure -- consistent with how this codebase
  already treats every other Gemini-dependent step as load-bearing, not
  best-effort, and keeps `BackendError`'s existing 503-on-Gemini-failure
  mapping meaningful for this call too.
- **`suggest_follow_up` is a free function** (`agent::suggest`), not a
  method needing an accumulating "grounding" abstraction, since per spec
  it deliberately has none of grounding's retry/verification machinery --
  reusing that machinery would have been the wrong shape for a call that
  isn't checking a factual claim.
- **A live two-turn `/ask` exchange against real Gemini was not run in
  this session** -- no `GEMINI_API_KEY` is available in this environment,
  and fetching the deployed Cloud Run service's key from Secret Manager to
  run one was declined (see report). `GET /scenarios` and the mocked
  end-to-end pipeline (above) were both verified live/running instead;
  `suggest`'s added latency (one more sequential Gemini call in `/ask`,
  per spec "sequential is fine") is therefore also unmeasured against real
  API latency in this session.

## PDF reports, snapshot store, and CVaR cap defaulting

### Persistent risk snapshot store (`store::SnapshotStore`)

`ResultStore` (an in-memory, fixed-capacity-20 `VecDeque`) has been
replaced by `store::SnapshotStore`, a SQLite-backed store in its own crate
(`crates/store`, using `rusqlite`'s `bundled` feature so no external
`sqlite3` needs to be installed on the distroless runtime image -- SQLite
is compiled from source as part of the crate build). Both `POST /experiment`
and `POST /ask` now insert a `RiskSnapshot` row on every successful call
(`POST /experiment` didn't store anything at all before this); `POST /ask`
still returns the row's id as `result_id` in `AskResponse`.

A `RiskSnapshot` holds the full `EvidenceTrace` as `trace_json`, plus a
handful of fields pulled out of it for fast querying without re-parsing
JSON: `portfolio_hash` (see below), `experiment_type`, `engine_version`,
`regime_label`/`smoothed_probs` (from `model_params.regime_state`, which
every experiment type now always populates -- see "Regime-conditional
factor covariance" below), `portfolio_vol_annualized` (`RiskDecomposition`
only), and `cvar_historical` (`CvarRebalance` only). It deliberately does
**not** persist narration or the follow-up suggestion -- those are
`/ask`-pipeline-only text, not part of the evidence trace, and adding them
would mean widening the fixed schema.

**Judgment call:** since the persisted schema has no room for narration,
`GET /report/{id}`'s PDF (`server::pdf`, below) now renders purely from the
trace -- the old "Analysis" section (the narration paragraph + grounding
warning) is gone. A report requested for a `POST /experiment` result was
always narration-less (there's no Gemini call on that path); the same is
now also true for a `POST /ask` result once it's fetched back out of the
store. The live narration text is still returned synchronously in `/ask`'s
own JSON response -- only the *stored, re-fetchable-later* copy lost it.

**Persistence caveat (Cloud Run):** the store's file path comes from
`SNAPSHOT_DB_PATH` (`main.rs`), defaulting to `/data/snapshots.db`. On
Cloud Run, `/data` is the container instance's own ephemeral local disk: it
survives across requests on a single warm instance, but is never shared
between instances or revisions, and is wiped on a cold start or
scale-to-zero (this service's `cloudrun.yaml` sets `minScale: "0"`, so
scale-to-zero is the normal idle state, not an edge case). Snapshots are
therefore best-effort recent history, not a durable audit log -- acceptable
for a hackathon-scale demo, but a real deployment should point
`SNAPSHOT_DB_PATH` at a mounted persistent volume (e.g. a GCS FUSE mount)
or replace `SnapshotStore`'s backing store with a managed database
(Cloud SQL) instead.

### Portfolio identity (`compute::portfolio::portfolio_hash`)

`SnapshotStore::latest_for_portfolio` looks up prior snapshots for "the
same portfolio" by `portfolio_hash`: SHA-256 of `"TICKER:weight,..."` pairs
sorted by ticker ascending, hex-encoded. Order-independent by construction
(sorted before hashing), so the same holdings always hash identically
regardless of what order the caller listed them in.

### `GET /report/{result_id}` (`server::pdf`)

A single-page A4 PDF built on `printpdf = "=0.12.8"` (pinned exactly, see
"Judgment calls" below for why the version matters here more than usual).
404 (`result_not_found`) if the id was never stored. Renders from the
`EvidenceTrace` alone (see the judgment call above for why narration is no
longer part of this). Layout: header (title + experiment type/timestamp,
rule), Key Numbers (a 3-column table, one row set per experiment type --
P&L/given+implied shocks/regime/crisis P&L for FactorShock; vol/top-2
factors/specific risk/regime for RiskDecomposition; CVaR before/after/
reduction/turnover/commission for CvarRebalance), Evidence Trace (model
params, data quality, invariants with pass/fail), footer (fixed
regime-smoothing disclaimer). Single-pass, no pagination, per spec.

### Shared INR formatting (`compute::format::format_inr`)

`format_inr(1_177_846.39) == "\u{20b9}11,77,846"` -- Indian lakh grouping
(last 3 digits, then pairs), Unicode minus (not ASCII hyphen) for
negatives. Lives in `compute` (not `server`) since it's used both there
(the PDF) and in `compute` itself: `FactorShockOutput` gained
`formatted_pnl_inr: String`, a pre-formatted sibling of the existing raw
`portfolio_pnl_inr: f64` field, so the agent's narration has a ready-to-
cite string. The raw numeric field is unchanged and is still what
grounding checks against.

### CvarRebalance `per_name_cap` defaulting

`CvarRebalanceInput::per_name_cap` is now `Option<f64>` (was a required
`f64`); when omitted, `run_cvar_rebalance` defaults it to
`cvar::DEFAULT_PER_NAME_CAP` (0.20) and records which happened as
`model_params.cap_source`: `"user-specified"` or `"server-default-0.20"`.
`parse::PARSE_SYSTEM_PROMPT` gained a line telling Gemini to omit
`per_name_cap` entirely when the user doesn't mention a cap (rather than
guessing a number), which is exactly what makes the Option/default path
reachable from a real `/ask` request.

**Judgment call, and a real spec/reality mismatch**: the checkpoint spec
asked for this defaulting to live "in server/src/routes.rs, ... after
deserialising AskRequest". That's not actually reachable there --
`AskRequest` carries the raw NL `message`, not a parsed `Experiment`;
parsing happens inside `agent::pipeline::run` (via Gemini), which
`routes.rs` never sees mid-flight. I implemented the default (and
`cap_source` recording) inside `compute::cvar::run_cvar_rebalance` itself
instead, which is the one place that's guaranteed to run for *every* path
that reaches CvarRebalance -- `/ask`, `/experiment`, and the `experiment`
CLI alike -- so the behavior the spec actually wants (a missing cap
defaults to 0.20, and the trace says which) holds everywhere, not just
`/ask`. Tested at the compute level (`cvar_tests.rs`), not as a `/ask`
integration test, since `server`'s `MockBackend` returns canned results
regardless of input and never touches real `compute` code.

### Parallelising narrate + suggest (`agent::pipeline`)

`pipeline::run` now runs `grounded_narrate` and `suggest_follow_up`
concurrently via `tokio::join!`, instead of narrate-then-suggest in
sequence. This required `suggest_follow_up` to stop taking the
narration text as input (narrate hasn't necessarily finished when suggest
starts) -- it now takes a short plain-text `experiment_summary` built
directly from the trace (`pipeline::experiment_summary`: experiment type +
one key output number, e.g. `"CvarRebalance experiment result: historical
CVaR went from 0.0204 to 0.0190."`), which is available as soon as the
compute step finishes, before either Gemini call starts.

**Live latency comparison**, same two-turn exchange as the previous
checkpoint's report (10-stock Nifty portfolio, real Gemini):

| | previous checkpoint (sequential) | this checkpoint (parallel) |
|---|---|---|
| Turn 1: "Where is my risk concentrated?" | 14.1s | 7.7s |
| Turn 2: rebalance request | ~14-20s (had transient retries) | 7.2s |

Turn 1 (clean, no retries, in both runs) is the fairest comparison: **14.1s
-> 7.7s, about 45% faster**, consistent with removing one whole sequential
Gemini round trip's worth of latency (narrate and suggest now overlap
instead of stacking).

### Tests

`store::tests`: insert 21 items into a capacity-20 store, the oldest (index
0) is evicted, the remaining 20 are all retrievable. `server::tests`:
`GET /report/{id}` after a real `POST /ask` returns 200,
`Content-Type: application/pdf`, and a body starting with the `%PDF`
signature; `GET /report/{unknown-uuid}` returns 404; `POST /ask`'s
response includes `result_id`. `compute::cvar_tests`: an omitted
`per_name_cap` resolves to 0.20 and every post-rebalance weight respects
it, with `cap_source == "server-default-0.20"`; a given `per_name_cap`
records `cap_source == "user-specified"`. `format::tests`: the four cases
from the spec (`1_177_846.39`, `-1_177_846.39`, `3_000.0`, `100.0`).

### Judgment calls, and a printpdf finding I want to flag explicitly

**`printpdf` resolved to `0.12.8`, a complete API rewrite** from the
`PdfLayerReference`-based API most printpdf tutorials/examples still show.
0.12 is `Op`-list based (`PdfDocument::new`, `PdfPage::new(w, h, ops)`,
`doc.with_pages(...).save(...)`) with no `PdfLayerReference`/
`add_builtin_font` returning a font reference to call `.use_text()` on --
I read the crate's own source (`~/.cargo/registry/.../printpdf-0.12.8/src`)
and its `examples/text.rs` to confirm the actual current shape before
writing `server::pdf`, rather than assuming the older API. This version
also pulls in a much heavier dependency tree than the name suggests
(`azul-core`, `azul-layout`, `hyphenation`, `rust-fontconfig`) for an
**optional HTML/CSS-to-PDF layout engine that `server::pdf` never uses** --
only the low-level `Op` primitives (text positioning, `DrawLine`). Pinned
exactly (`=0.12.8`) per spec, and worth pinning exactly given how much the
API moved between versions.

**Confirmed layout/encoding bug, found live, not hypothetical**: printpdf's
built-in fonts use `/Encoding /WinAnsiEncoding` (CP1252, single-byte).
Traced it to the byte level in a real generated PDF's content stream:
any character outside that repertoire -- ₹ (rupee), the Unicode minus
− that `format_inr` uses for negative amounts, ✓/✗, ⚠ -- is
silently written as a literal `?` glyph, no error, no warning. First
observed as `"Commission Cost | ?2,000 | ?"` in a live-generated report.
Per this checkpoint's explicit instruction ("flag immediately ... don't
work around silently"), I stopped and reported this verbatim before doing
anything else, including the specific finding that a *negative* rupee
figure (the common case -- most FactorShock P&L in this app is a loss)
would have rendered as `"??11,77,846"`, not `"\u{2212}\u{20b9}11,77,846"`.
Given the choice between an ASCII fallback (no new dependency), embedding
a real Unicode font (rejected by the checkpoint spec's own "no font files"
instruction), or shipping the `?`s, the chosen fix -- confirmed by the
person running this session -- is a PDF-only ASCII substitution
(`server::pdf::pdf_safe`: ₹->`"Rs."`, −->`"-"`, ✓->`"Y"`,
✗->`"N"`, ⚠->`"!"`), applied once at the single funnel point every
piece of PDF text passes through (`Page::text`). `compute::format_inr`
itself, and every other consumer of it (the trace field, JSON API
responses), is untouched -- this is a PDF-rendering-only substitution, not
a change to what the app reports elsewhere. Re-verified live afterward: a
real negative-P&L FactorShock report now shows `"-Rs.11,77,846"` and `"Y"`/
`"N"` for invariants, correctly.

**Word-wrap uses an approximate per-character width heuristic**
(`server::pdf::char_width_em`), not real Helvetica AFM metrics --
printpdf's builtin fonts expose no width/measurement API (only externally
loaded `ParsedFont`s carry `glyph_widths`). The heuristic (narrow
punctuation/`i`/`l` ~0.28em, wide `m`/`w`/uppercase ~0.7-0.83em, ~0.52em
default, bold +5%) is close enough that wrapping at 170mm didn't visibly
break in any of the live reports generated this session, but it is an
approximation, not a measurement, and I'm flagging it as such rather than
presenting it as exact.

**Table cells are not word-wrapped** (only the Analysis section's
narration is) -- a long `Evidence Trace` invariant `detail` string (these
can run 60-70 characters, e.g. `"lhs=-1177846.374721108703
rhs=-1177846.374721108703 abs_diff=0.000e0"`) draws starting at the
Detail column's x-position and extends past the page's right margin
uncorrected, rather than wrapping or truncating. Confirmed via `qpdf
--check` (structurally valid) and `pdftotext`, not visually re-rendered
end-to-end in an actual PDF viewer this session. Left as-is rather than
building a general cell-wrapping table renderer, consistent with the
spec's own "single-pass layout -- no pagination needed for a demo" scope,
but noted here rather than silently accepted.

**Report generation timestamp is wall-clock, not experiment-run time**:
`EvidenceTrace` carries no "when was this computed" field, so the header's
timestamp is `chrono::Utc::now()` at PDF-render time (which can be later
than when the underlying `/ask` actually ran, if the result sat in the
store for a while). Good enough for a demo; would need a real field on
`EvidenceTrace` to be exact.

## Fourth experiment: PortfolioPerformance

Added in response to a real production report: a user asked "How is my
portfolio performing right now in terms of recent market trends?" via the
live UI and got a 422 `unrecognised_request` -- not a parsing bug, but a
genuine capability gap. The other three experiments answer "what if"
(FactorShock), "where's my current risk" (RiskDecomposition), and "how do
I rebalance" (CvarRebalance); none of them compute realized historical
returns, so `agent::parse` correctly had nothing to map that question to.

`compute::performance::run_portfolio_performance` (`PortfolioPerformance`,
a fourth `Experiment` variant) closes the gap: `total_return`,
`annualized_return`, `annualized_vol_realized` (realized, from the actual
period-by-period return series -- distinct from RiskDecomposition's
factor-model-*implied* `portfolio_vol_annualized`), `max_drawdown`,
`best_period_return`, `worst_period_return`, `start_value_inr`/
`end_value_inr`. Wired through the same path every other experiment uses:
`agent::schema` (`run_portfolio_performance` function declaration),
`agent::parse` (system prompt + match arm), `agent::pipeline::compute_trace`,
`server::pdf` (a `PortfolioPerformance` Key Numbers table), and the UI
(needed no changes at all -- narration/suggestion/download/trace rendering
is already experiment-agnostic).

**Judgment call: constant-mix, not literal buy-and-hold.** Each period's
portfolio return is the *current* target weights applied to that period's
per-holding simple return, as if rebalanced back to target weights every
period, rather than a literal share-count simulation that lets weights
drift with prices. This is a real simplification (it modestly
understates/overstates drift-related effects vs. true buy-and-hold), but
it's the only option available without a new data source: past the data
layer (`compute::data`), only aligned *log returns* survive, not raw
per-share prices, and every other experiment in this codebase already
treats "the portfolio's weights" as the snapshot to analyze, not a
historical share-count schedule to reconstruct. Documented in the trace's
`outputs.note` field, not just here.

**Live-verified** (real Gemini, real Yahoo data): the exact failing
question now returns 200 with `experiment.type: "PortfolioPerformance"`
and a real narrated answer (`total_return -16.27%`, `max_drawdown -19.66%`
on the 10-stock Nifty portfolio's trailing 252 trading days). PDF report
generation also confirmed working for the new experiment type. Tests: 3
new `compute` tests (constant-return compounding is exact, a V-shaped
return path is correctly flagged as a deep drawdown despite ending near
zero, and the trace's own value invariant holds for a multi-holding
portfolio) and 1 new `agent` parse test (mocked Gemini calling
`run_portfolio_performance`). Postman: a new "11. Portfolio Performance"
folder, including the exact regression case (this question used to 422,
now returns 200) -- full collection re-run clean, 30/30 requests, 92/92
assertions.

## Fifth experiment: RiskDrift (`compute::drift`)

Explains what changed in a portfolio's risk between two points in time,
by diffing the current factor-model fit against a prior stored
`RiskDecomposition` snapshot: volatility, factor contributions (Euler),
portfolio-level factor betas, factor correlations, market regime, and
specific-risk share. `RiskDriftInput` carries no `portfolio` field of its
own (unlike every other experiment) -- it's diffing "the current
portfolio" against a *stored* baseline, not two portfolios given inline --
so the portfolio now flows into the compute layer as an explicit parameter
everywhere, not just embedded in each variant's own input.

### `ExperimentContext` and the new dispatch layer (`compute::dispatch`, `compute::context`)

`RiskDrift` needs read access to `store::SnapshotStore` to resolve its
baseline, so `store` is now a dependency of `compute` (previously only
`server` depended on it). The per-variant fetch/fit/dispatch logic that
used to live in `agent::pipeline::compute_trace` moved into
`compute::dispatch::run_experiment(experiment, portfolio, ctx)` --
`agent::pipeline::compute_trace` is now a thin wrapper around it.
`ExperimentContext { store, portfolio_hash }` is threaded through for
every experiment type but only `RiskDrift` actually reads it; the other
four ignore it (reserved for a later session's policy checks, per the
original spec). This cascaded through:
- `agent::pipeline::run` gained a `store: Arc<SnapshotStore>` parameter.
- `server::backend::Backend::run_experiment` gained a `portfolio: Portfolio`
  parameter (needed since `RiskDriftInput` has none of its own to read).
- `server::backend::RealBackend` gained a `store` field, built once in
  `main.rs` and shared with `AppState`.

### Baseline compatibility: `portfolio_betas` and `factor_correlation`

Diffing betas/correlations requires the *baseline's own* per-stock betas
and factor correlation matrix -- neither was ever persisted anywhere in
`EvidenceTrace` before this session (only `by_factor`'s Euler
contributions and `portfolio_vol_annualized` were). `RiskDecompositionOutput`
gained two new fields, `portfolio_betas: BTreeMap<String, f64>` (portfolio-level
factor exposure, `Sum_i w_i * beta_ik`) and `factor_correlation: CorrelationMatrix`,
populated by `run_risk_decomposition`. **This means a `RiskDecomposition`
snapshot stored *before* this change cannot serve as a `RiskDrift`
baseline** -- `run_risk_drift` detects the absence of these fields (an
empty map/matrix when read back from JSON) and returns a clear
`ComputeError::InvalidInput` naming the stored snapshot's `engine_version`
and asking the caller to rerun `RiskDecomposition`, rather than silently
producing zeroed deltas.

### Judgment calls

- **Baseline resolution when `baseline_snapshot_id` is omitted uses the
  single most recent stored snapshot**, not "the second entry" of a
  2-item `latest_for_portfolio` query as an earlier draft of this spec
  described. At the point baseline resolution runs, `RiskDrift`'s own
  result has not yet been stored (that happens afterward, via the same
  generic `SnapshotStore::insert` path `/experiment`/`/ask` already use
  for every experiment type -- `RiskDrift` needed no special insertion
  code of its own), so there is no "current" entry in the store to skip
  past yet. Taking the second entry literally would require *two* prior
  snapshots to exist before `RiskDrift` could run even once, which
  contradicts the required live-run flow (a single prior
  `RiskDecomposition` snapshot must be usable immediately as a baseline
  for the very next `RiskDrift` call) -- confirmed by the live run below,
  which does exactly that.
- **`run_risk_drift` is split into a thin data-fetching wrapper and a
  hermetic `compute_risk_drift`** (pure diff logic: baseline snapshot +
  an already-computed "current" `RiskDecompositionOutput`/`EvidenceTrace`
  in, `RiskDriftOutputs`/`EvidenceTrace` out), matching every other
  experiment function's convention of taking pre-fetched data/model
  rather than reaching out to the network itself. This is what makes the
  five required `compute` tests possible without live Yahoo data.
- **`GET /drift` filters client-side over a capped recent window**
  (`DRIFT_SCAN_WINDOW = 1000`) rather than the store having a "list
  recent of this experiment type" query -- out of this session's declared
  scope ("no changes to the store crate"). A portfolio with more than
  1000 *non*-`RiskDrift` snapshots since its oldest relevant `RiskDrift`
  one could miss older entries; fine for a hackathon-scale demo, but a
  real "risk drift history" UI would want the store itself to support
  filtering by `experiment_type`.
- **`ComputeError::NoPriorSnapshot` maps to 422**, not the generic 500
  every other compute error gets (`server::backend`'s
  `From<compute::ComputeError>` now matches on this variant specifically)
  -- matching how `ParseError::Unrecognised` (a request problem, not an
  internal failure) is already mapped.
- **`days_elapsed` compares data-window end dates, not wall-clock
  `created_at` timestamps** -- it answers "how much did the underlying
  market data advance between the two fits," which is what a risk-drift
  narrative actually cares about, not how long ago someone happened to
  click a button.

### Live run (10-stock... actually 2-stock demo portfolio, RELIANCE.NS/TCS.NS)

1. `POST /experiment` `RiskDecomposition` (window 252) to create a
   baseline: `200`, regime `Bull`, `portfolio_vol_annualized: 0.18447555218792452`.
2. `POST /experiment` `RiskDrift` with `baseline_snapshot_id: null`,
   run seconds later against the same live data: `200`,
   `vol_before: 0.18447555218792452`, `vol_after: 0.18447555218792452`,
   `vol_change_pct: 0.0`, `regime_before: "Bull"`, `regime_after: "Bull"`,
   `top_growing_risk_factor: "USDINR"`. Vol and regime are identical
   because the underlying market data hadn't changed in the few seconds
   between the two calls -- exactly the expected, correct result, not a
   bug (a real drift narrative would show non-trivial deltas across a
   longer gap between snapshots).
3. `GET /drift?portfolio=<hash>` (hash computed the same way
   `compute::portfolio::portfolio_hash` does): `200`, `snapshots` contains
   exactly the one `RiskDrift` result from step 2, with the same
   `vol_before`/`vol_after`/`regime_before`/`regime_after` fields.

### Tests

`compute`: 5 new hermetic tests in `drift_tests.rs` against
`compute_risk_drift` directly (no network) -- `vol_change_abs`/
`vol_change_pct` computed correctly; the Euler-additivity residual is
recorded (not errored) when factor deltas don't sum exactly to
`vol_change_abs`; `regime_changed`/`regime_worsened` true/false in both
directions; `risk_became_more_concentrated`'s 5-percentage-point
threshold (50%->60% true, 50%->52% false); and
`run_risk_drift` (the thin wrapper, exercising real baseline resolution
against an empty in-memory store) returns `ComputeError::NoPriorSnapshot`
with the exact required message when nothing exists yet. `agent`: 2 new
`parse_tests.rs` cases (`RiskDrift` with a `null` and with a specific
`baseline_snapshot_id`). `server`: 4 new `tests.rs` cases (`RiskDrift`
with a pre-inserted baseline returns 200 with a non-null `vol_change_abs`;
no prior snapshot returns 422 with the exact message; `GET /drift` with
no snapshots returns `{"snapshots": []}`, 200; `GET /drift` after
inserting two `RiskDrift` snapshots -- plus one deliberately-inserted
`RiskDecomposition` snapshot for the same portfolio, to confirm it's
excluded -- returns exactly the 2 summaries, with no `trace_json`/
`outputs` field present).

## Sixth experiment: ReverseStress (`compute::reverse_stress`)

The inverse of `FactorShock`: instead of "what happens under shock X," this
asks "what's the *smallest* market shock that would cause at least L
rupees of loss." Solves for the minimum-Mahalanobis-severity factor shock
`s` (in log-return space) such that `P&L(s) <= -loss_threshold_inr`,
subject to per-factor box bounds (default: plausible historical ranges for
Indian markets -- `MARKET [-40, 0]`, `USDINR [-5, 20]`, `BRENT [-60, 100]`,
`GOLD_USD [-20, 40]`, `RATES_PROXY [-10, 10]`, all in simple %). Like
`RiskDrift`, `ReverseStressInput` has no `portfolio` field of its own --
the portfolio arrives via `compute::dispatch::run_experiment`'s explicit
`portfolio` parameter (the infrastructure Session 2 built for exactly this
shape of input), so this session needed **no changes to `store`, `agent`
pipeline, or server routing** beyond the new `Experiment` variant and its
own dispatch-layer fetch/fit case.

### The math

`P&L(s) = Sum_i w_i * (exp(Sum_k beta_ik * s_k) - 1) * total_value_inr` --
identical to `FactorShock`'s formula. `s^T F^-1 s` (the squared Mahalanobis
distance under the current-regime factor covariance `F`) is the severity to
minimise; `mahalanobis_severity` in the output is its square root, so the
`"within 1σ"`/`"1–2σ"`/`"2–3σ"`/`">3σ"` labels' boundaries at exactly 1/2/3
mean what they say (bucketing the *squared* form would need boundaries at
1/4/9 for the same sigma levels -- see the judgment call below).

Solved in two stages:
1. **KKT closed form** on the linearised problem (`P&L(s) ~= p^T s`, where
   `p_k = Sum_i w_i * beta_ik * total_value_inr` is the linear P&L
   sensitivity to factor `k`): `s* = -L * (F p) / (p^T F p)` --
   `compute::reverse_stress::kkt_shock`, unit-tested directly against a
   hand-built diagonal `F`/single-nonzero-entry `p` (the "single-factor"
   case: `s* = -L/p_k`, `severity = |s*|/sqrt(F_kk)`, both exact).
2. **Projected gradient descent** refines that starting point against the
   *true* (nonlinear) objective and constraint. "Projection" does two
   things every step: clip to box bounds, then rescale the point (fixed
   direction, 1-D bisection search on the scale factor) to the smallest
   magnitude that still makes the *true* nonlinear `P&L` breach `-L`
   exactly -- not a general nonlinear-constraint projection, but exact for
   this problem's actual geometry, since `P&L` is monotonic in shock
   magnitude along any "bad" direction. Armijo backtracking (`β=0.5`,
   `c=1e-4`), up to 200 iterations, gradient-norm (`<1e-6`) or
   objective-change (`<1e-8`) convergence.
3. **Feasibility pre-check**: the worst-case corner of the box (each
   factor set to whichever bound maximises linearised loss) must itself
   breach `-L`, or the request is infeasible -- `ComputeError::
   ReverseStressInfeasible` with the exact max feasible loss, before any
   optimisation is attempted.

### Judgment calls

- **`mahalanobis_severity` is the *unsquared* Mahalanobis distance**
  (`sqrt(s^T F^-1 s)`), even though the spec's formulation names `s^T F^-1
  s` itself "the Mahalanobis severity" and calls it the objective to
  minimise. Both are literally true (the objective minimised is the
  squared form; the reported/labelled quantity is its square root) -- the
  σ-bucket boundaries at exactly 1, 2, 3 only make sense for the unsquared
  distance (a 2σ point has squared-form value 4, not 2), so that's what's
  reported and bucketed.
- **`solver_status: "converged"` means "the projected-gradient search
  could no longer find a strictly-better feasible point," not "the raw
  unconstrained gradient norm fell below tolerance."** Confirmed live (see
  below): the live run converged in a single iteration with
  `gradient_norm_final: 11.78` -- nowhere near the `1e-6` tolerance. This
  is expected, not a bug: near the projected boundary, the *projection*
  operator (clip + rescale-to-breach) dominates a small unconstrained
  gradient step, so Armijo backtracking can exhaust its 30 attempts
  without finding a strictly-decreasing candidate even though the
  underlying point is already the true constrained optimum. A raw gradient
  norm is the right diagnostic for an *unconstrained* problem; for this
  projected one, "no further Armijo-acceptable improvement" is the more
  meaningful stopping signal, which is why it's still labelled
  `"converged"` rather than `"max_iter"`.
- **Bounds and the true breach constraint are enforced by an alternating
  fixed-point loop** (`project_to_feasible`: clip to bounds, then rescale
  to the true constraint boundary, repeat up to 20 rounds), not a single
  pass -- clipping can pull a rescaled point back out of breach, and
  rescaling can push a clipped point back out of bounds, so neither
  operation alone is sufficient. In every test and the live run this
  converges within a handful of rounds; a final defensive
  `project_to_feasible` call after the gradient loop exits (regardless of
  why it exited) guarantees the hard breach invariant
  (`portfolio_pnl_inr <= -loss_threshold_inr * 0.999`) holds even if the
  refinement loop itself stalled early.
- **`factor_attribution` sums to the log-space P&L, not
  `portfolio_pnl_inr` exactly** -- same convention as `FactorShock`'s own
  `factor_attribution_log_inr`: `Sum_i value_i * beta_ik * s_k` is exact in
  log-return space, but doesn't equal the `exp(.) - 1`-converted simple
  P&L once the shock is large (the conversion is convex). Recorded as an
  approximate invariant (tolerance `1%` of `loss_threshold_inr`), never
  errored, per spec.

### Live run (10-stock Nifty portfolio, `loss_threshold_inr: 500000`, `factor_bounds: null`)

| field | value |
|---|---|
| `shock_vector` | `MARKET -4.67%`, `BRENT +1.80%`, `GOLD_USD -0.62%`, `RATES_PROXY +0.88%`, `USDINR +0.68%` |
| `mahalanobis_severity` | `0.534` |
| `severity_label` | `"within 1σ"` |
| `portfolio_pnl_inr` | `-500,000.00` (breaches the ₹5L threshold exactly) |
| `most_vulnerable_holdings` | `["RELIANCE.NS", "HDFCBANK.NS", "LT.NS"]` |
| `solver_status` | `"converged"` (`n_iterations: 1`, see the judgment call above re: `gradient_norm_final`) |
| `linearisation_error_inr` | `13,629.20` |

A modest ~4.7% Nifty fall (well within 1 standard deviation of this
portfolio's own factor covariance) is enough to breach a ₹5L loss on this
₹1Cr portfolio -- an unsurprising, sanity-checking result (5% of ₹1Cr is
exactly ₹5L, and a ~5% market move is not a tail event for Nifty).

### Tests

`compute`: 5 new hermetic tests in `reverse_stress_tests.rs` -- the
single-factor KKT closed form matches exactly (`1e-6`); `severity_label`'s
four buckets; infeasibility returns the correct max feasible loss for an
artificially tight-bounded scenario; the solution respects
caller-specified bounds; and a 10-synthetic-stock portfolio's gradient
descent both converges and genuinely breaches the threshold. `agent`: 2
new `parse_tests.rs` cases (`ReverseStress` with `null` and with explicit
`factor_bounds`). `server`: 2 new `tests.rs` cases (valid input returns 200
with a negative `portfolio_pnl_inr`; an infeasible threshold returns 422
with the exact infeasibility message).

## Policy Engine (`compute::policy`) and constraint-aware remediation

A deterministic risk-limit checker (`RiskPolicy` -> `PolicyResult`), plus
two ways to use it: a standalone seventh experiment type (`PolicyCheck`),
and an extension to `CvarRebalance` that tries to *fix* policy breaches by
re-solving at a tighter CVaR confidence level. No optimisation and no LLM
call happens inside the policy engine itself -- every rule is a plain
comparison against a quantity already computable from a fitted
`FactorModel` and the historical return series.

### Rules and what each one actually computes

- `max_vol_annualized`: the same `sqrt(w' Sigma w)` formula
  `RiskDecomposition` uses, against the already-fitted (always
  regime-conditional) model -- no refit.
- `max_cvar_95`: historical CVaR at a **fixed** 95% confidence, computed
  directly from the portfolio's own full-history simple returns -- never
  via `CvarRebalance`'s LP, and never by reading a stored snapshot (see
  the judgment call below on why `evaluate_policy` doesn't take a `store`
  parameter at all, despite an earlier draft of this spec suggesting one).
- `max_factor_contribution_share`: the largest *signed* (not absolute
  value) single-factor Euler contribution share -- a large negative
  (diversifying) contributor is never flagged as "dominant," matching what
  a risk manager actually means by the limit.
- `max_position_weight`: `max(w_i)` across holdings.
- `max_loss_under_scenarios`: for each named scenario, a real
  `FactorShock` run (`propagate: false`, per spec) against
  `compute::scenarios::all_scenarios()`'s fixed shock set, loss reported
  as `abs(portfolio_pnl_inr) / total_value_inr`.

### `PolicyCheck`, the seventh experiment type

Like `RiskDrift`/`ReverseStress`, `PolicyCheckInput` has no `portfolio`
field of its own -- it reuses the same dispatch-level `portfolio`
parameter those two already established. `PolicyCheckOutputs` always
reports the four "actual" values (`portfolio_vol`, `portfolio_cvar_95`,
`max_position_weight`, `max_factor_share`) regardless of whether the
matching `RiskPolicy` field was set, so a caller can see where they stand
even for limits they didn't ask about.

### Constraint-aware remediation (`CvarRebalance`'s new `policy` field)

`CvarRebalanceInput.policy: Option<RiskPolicy>` triggers, after the LP
solves: evaluate the policy against the proposed weights; if breaches
remain, tighten `confidence_level` by `0.01` and re-solve, up to 5 times,
keeping whichever attempt's weights are reported (`weights_after`,
`stats_after`, `turnover`, etc. all reflect the *final* remediation
attempt, not the original solve -- `confidence_level` in the output is
likewise the final, tightened value, not the one the caller originally
sent). Implemented as a recursive call to `run_cvar_rebalance` itself
(each retry's own `input.policy` is `None`, so it can't recurse further) --
reusing the entire existing LP-building/solving path rather than
extracting a separate "core solver" out of it.

### Judgment calls

- **`evaluate_policy` takes no `store` parameter**, despite an earlier
  draft of this spec's own function signature listing one ("needed for
  CVaR: run CvarRebalance or read from latest snapshot"). The spec's own
  evaluation logic for `max_cvar_95` says the opposite explicitly ("compute
  CVaR directly from historical scenarios... do not run the LP"), which
  needs a `MarketData` reference, not a store handle -- so `store` was
  dropped and `data: &MarketData` added instead. `evaluate_policy` is
  synchronous, hermetic, and needs nothing beyond what a fitted model
  already carries.
- **Passive policy attachment is skipped for `RiskDrift`.** `dispatch::
  run_experiment` attaches a passive `policy_result` to every experiment
  except `PolicyCheck`/`CvarRebalance` (which handle policy themselves),
  but `RiskDrift` fits its "current" factor model *internally*
  (`drift::run_risk_drift`) and never exposes it back to the dispatcher --
  re-fitting it a second time purely for a passive check felt like the
  wrong trade-off for this session. A caller wanting a policy check
  alongside `RiskDrift` should make a separate `PolicyCheck` request.
  `PortfolioPerformance` similarly never otherwise fits a factor model, so
  one is fit there *only* when a policy is actually attached (skipped
  entirely, no extra cost, when it isn't).
- **Live-caught bug, fixed before the live runs below**: `run_cvar_rebalance`'s
  internal `build_portfolio` closure (builds a `Portfolio` from a
  `weights_before`/`weights_after` map to hand to `evaluate_policy`)
  originally iterated the map directly -- but `weights_before`/`_after` are
  `BTreeMap<String, f64>`, hence *alphabetically* ordered, while
  `evaluate_policy` needs `portfolio.tickers() == model.tickers`
  *positionally*, not just as a set (the fitted model's `beta_matrix()`
  rows are in the original portfolio's ticker order). Every existing test
  happened to use already-alphabetical tickers (`AAA`/`BBB`), so this
  passed the full test suite before failing immediately on the first real
  10-ticker Nifty portfolio (`RELIANCE.NS` first, not alphabetically
  first) with `"portfolio holdings and fitted model tickers must match
  1:1, in order"`. Fixed by iterating the *original* `tickers` list
  instead of the map; a new regression test
  (`cvar_rebalance_with_policy_does_not_care_about_ticker_alphabetical_order`)
  uses deliberately reverse-alphabetical tickers so this can't silently
  regress again.
- **Tightening `confidence_level` is not guaranteed to resolve every
  breach, and the live run below confirms it doesn't always** -- CVaR
  minimization at a higher confidence level only changes the
  historical-scenario tail the LP optimizes over; it has no direct
  mechanical relationship to a factor-shock scenario limit
  (`max_loss_under_scenarios`) or a factor-concentration limit
  (`max_factor_contribution_share`). The live run's `max_position_weight`
  breach happened to resolve (a side effect of `per_name_cap` binding in
  the same LP, not of confidence-level tightening itself), while the
  `covid_crash` scenario-loss breach did not, even after all 5 iterations
  reached the confidence-level ceiling (`0.999`). This is exactly why the
  output separately reports `policy_breaches_resolved`/`_remaining` rather
  than a single pass/fail bit -- remediation is a best-effort heuristic,
  not a guaranteed fix.

### Live run A: `PolicyCheck` on the 10-stock Nifty portfolio

Policy: `max_vol_annualized: 0.14`, `max_cvar_95: 0.03`,
`max_factor_contribution_share: 0.80`, `max_position_weight: 0.12`,
`max_loss_under_scenarios: [{covid_crash, max_loss_pct: 0.30}]`.

`all_passed: false`, `breach_count: 2`:

| rule | actual | limit | passed |
|---|---|---|---|
| `max_vol_annualized` | 11.7% | 14.0% | ✓ |
| `max_cvar_95` | 2.0% | 3.0% | ✓ |
| `max_factor_contribution_share` (MARKET) | 64.4% | 80.0% | ✓ |
| `max_position_weight` (RELIANCE.NS) | 15.0% | 12.0% | ✗ (breach 3.0pp) |
| `max_loss_under_scenarios[covid_crash]` | 37.6% | 30.0% | ✗ (breach 7.6pp) |

`most_severe_breach`: `max_loss_under_scenarios[covid_crash]` (larger
breach magnitude). Note the portfolio's live vol (11.7%) is well under the
14% limit and under the 15.5% figure an earlier session's live run
captured -- real market data moves between sessions; this isn't a
discrepancy, it's what "live" means.

### Live run B: `CvarRebalance` with the same policy attached

`turnover_limit: 0.4`, `per_name_cap: 0.12`, same policy as above.

`status: "optimal"`, `remediation_iterations: 5`, final
`confidence_level: 0.999` (tightened from 0.95, hit the ceiling without
resolving everything):

- `policy_breaches_resolved`: `[]`
- `policy_breaches_remaining`: `["max_loss_under_scenarios[covid_crash]"]`

The `max_position_weight` breach present in `policy_result_before`
disappeared from `policy_result_after` -- but, per the judgment call
above, that's `per_name_cap: 0.12` doing its job in the LP itself (every
`weights_after` entry sits at essentially exactly 0.12), not the
confidence-level tightening. The `covid_crash` scenario-loss breach
remained unresolved through all 5 iterations, demonstrating the
remediation loop's real limitation rather than a contrived one.
