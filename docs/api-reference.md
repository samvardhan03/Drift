# Drift HTTP API Reference

Reference for building a frontend or client against the Drift server.

## Base URL
LOCAL: http://localhost:8080
DEPLOYED: https://YOUR_DEPLOYMENT_HOST (see docs/deploy.md)

Frontends typically read this from an environment variable such as
NEXT_PUBLIC_API_BASE_URL.

## CORS
The backend allows all origins. No proxy needed.

---

## Endpoints

### GET /health
Returns: { "status": "ok", "version": "0.1.0" }
Use on page load to show the "● API Live" / "● API Offline" 
indicator in the navbar.

---

### GET /scenarios
Returns the 3 historical scenario presets.
Response shape:
[
  {
    "id": "covid_crash",
    "name": "COVID Crash (Mar 2020)",
    "date_range": "Feb 19 – Mar 23, 2020",
    "description": "Nifty fell 38% in 33 days...",
    "shocks_pct": { "MARKET": -38.0, "BRENT": -55.0, ... },
    "propagate": false
  },
  { "id": "ilfs_contagion", ... },
  { "id": "taper_tantrum_2013", ... }
]
Use to populate the three scenario shortcut buttons.
When a scenario button is clicked, pre-fill the chat input 
with: "What would the [scenario.name] have done to my portfolio?"
and submit immediately.

---

### POST /portfolio/upload
Upload a CSV or XLSX file to auto-populate the portfolio builder.
Request: multipart/form-data, field name "file"

CSV format accepted (two layouts):
  Layout A — weight-based:
    ticker,weight
    RELIANCE.NS,0.10
    HDFCBANK.NS,0.10

  Layout B — value-based:
    ticker,shares,avg_price_inr
    RELIANCE.NS,10,2850.00

Response (200):
{
  "portfolio": {
    "holdings": [
      { "ticker": "RELIANCE.NS", "weight": 0.10 },
      ...
    ],
    "total_value_inr": 10000000.0
  },
  "layout_detected": "weight",
  "tickers_normalised": [],
  "row_count": 10
}
On error: 400 with { "error": "...", "code": "..." }

After a successful upload, populate the portfolio builder 
with the returned holdings and total_value_inr.

---

### GET /auth/upstox/status
Returns: { "configured": true }
Call on page load. If configured is false, hide the 
"Connect with Upstox" button entirely.

### GET /auth/upstox/login
Redirect the user's browser to this URL directly:
  window.location.href = `${API_BASE_URL}/auth/upstox/login`
The backend handles the OAuth redirect to Upstox.
Do NOT fetch this as an API call — it must be a full 
browser navigation.

### GET /auth/upstox/callback
The backend handles this — the frontend never calls it 
directly. After Upstox redirects back, the backend 
returns JSON directly to the browser:
{
  "portfolio": { "holdings": [...], "total_value_inr": ... },
  "holdings_count": 10,
  "data_as_of": "2026-09-28T00:00:00Z"
}
To handle this: on page load, check if the current URL 
path is /auth/upstox/callback. If it is, read the 
response body (the backend already rendered it as JSON), 
parse it, populate the portfolio builder, then redirect 
to / using window.history.replaceState.

---

### POST /ask  ← MOST IMPORTANT ENDPOINT
This is the core endpoint. Call it on every user message.

Request body:
{
  "portfolio": {
    "holdings": [
      { "ticker": "RELIANCE.NS", "weight": 0.10 },
      { "ticker": "HDFCBANK.NS", "weight": 0.10 }
      // ... at least 2 holdings
    ],
    "total_value_inr": 10000000.0
  },
  "message": "What would a Nifty crash do to my portfolio?",
  "conversation_history": [
    // All prior turns in this session
    // Start with [] on first message
    // Append each turn after a successful response
    { "role": "user", "content": "Prior user message" },
    { "role": "assistant", "content": "Prior AI response" }
  ]
}

Response (200):
{
  "experiment": { "type": "FactorShock", ... },
  "trace": { ... },  // full EvidenceTrace — use for PDF
  "narration": "Your portfolio would lose approximately ₹20.6L...",
  "grounding_warnings": [],  // empty = all good
  "result_id": "uuid-for-pdf-download",
  "assistant_turn": {
    "role": "assistant",
    "content": "Your portfolio would lose approximately ₹20.6L..."
  },
  "suggestion": "Run a stress test for the IL&FS scenario",
  "agent_execution_trace": {
    "id": "uuid",
    "tool_plans": [
      { "tool": "factor_shock", "reason": "..." }
    ],
    "gemini_calls": 3,
    "total_latency_ms": 7200
  }
}

After a successful response:
1. Append to conversation_history:
   { "role": "user", "content": <the message you sent> }
   then response.assistant_turn
2. Render response.narration as the AI message in chat
3. Show response.suggestion as a tappable chip below 
   the AI message — clicking it submits that text as 
   the next message
4. If response.grounding_warnings.length > 0, show a 
   yellow banner: "⚠ Some numbers could not be verified"
5. Store response.result_id for the PDF download button
6. Show regime badge from:
   response.trace.model_params.regime_state.current_label
   "Bull" → green, "Bear" → amber, "Crisis" → red

On error 422: show in chat as a soft grey message:
  "I can only help with portfolio risk questions."
On error 5xx: show red banner "Something went wrong. Try again."
Do NOT append failed turns to conversation_history.

---

### GET /report/{result_id}
Download a PDF report for a completed /ask result.
result_id comes from the /ask response.

Call as:
  const response = await fetch(`${API_BASE_URL}/report/${resultId}`)
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `drift-report-${experimentType}-${Date.now()}.pdf`
  a.click()

Show a "↓ PDF" icon button on each AI message that has 
a result_id. Only show it after the response is received, 
not while loading.

---

### GET /execution-trace/{id}
Returns the full agent execution trace for an /ask result.
id = response.agent_execution_trace.id from /ask response.

Optional — show a "View Trace" link in a developer/debug 
panel. Not needed for the main demo flow.

---

### POST /experiment  ← Direct experiment, no AI narration
Use only for the "Direct" mode if you implement it.
Request:
{
  "portfolio": { ... },
  "experiment": {
    "type": "RiskDecomposition"
    // or FactorShock, CvarRebalance, 
    // ReverseStress, PolicyCheck, 
    // RiskDrift, PortfolioPerformance
  }
}
Returns the raw EvidenceTrace. No narration, no suggestion.
Most users should use /ask instead.

---

## Portfolio validation rules
Enforce these client-side before any API call:
- At least 2 holdings
- All weights > 0
- Weights sum to 1.0 ± 0.01 (show live indicator)
- total_value_inr > 0

Show inline errors — do not submit if validation fails.

## IMPORTANT: Weight format
The API always expects weights as decimals summing to 1.0 (e.g. 0.10 for 10%).

When displaying weights in the UI from an upload response, multiply by 100 for display only.
When sending weights to /ask or /experiment, always divide the displayed value by 100 first.

Example:
  API returns: weight: 0.3797
  Display as: 37.97%
  Send back:  weight: 0.3797 (not 37.97)

A weight sent back as a raw percentage (e.g. 37.97 instead of 0.3797) fails the "all weights > 0%"
validation above in a confusing way — the weight itself isn't 0%, but the sum of all weights will be
wildly over 1.0 ± 0.01, since every holding was multiplied by 100 again. If you see that failure, check
this conversion first.

---

## Default portfolio (pre-load on first visit)
{
  "holdings": [
    { "ticker": "RELIANCE.NS", "weight": 0.10 },
    { "ticker": "HDFCBANK.NS", "weight": 0.10 },
    { "ticker": "ICICIBANK.NS", "weight": 0.10 },
    { "ticker": "INFY.NS",     "weight": 0.10 },
    { "ticker": "TCS.NS",      "weight": 0.10 },
    { "ticker": "LT.NS",       "weight": 0.10 },
    { "ticker": "ITC.NS",      "weight": 0.10 },
    { "ticker": "KOTAKBANK.NS","weight": 0.10 },
    { "ticker": "BHARTIARTL.NS","weight": 0.10 },
    { "ticker": "TMPV.NS",     "weight": 0.10 }
  ],
  "total_value_inr": 10000000.0
}

---

## Conversation history management
// Initialise
let conversationHistory = []

// On each successful /ask response:
conversationHistory.push(
  { role: "user", content: userMessage },
  response.assistant_turn  // { role: "assistant", content: "..." }
)

// Pass on every /ask call:
body.conversation_history = conversationHistory

// On "Clear conversation" button:
conversationHistory = []

// Keep the last 20 turns max to avoid token bloat:
if (conversationHistory.length > 20) {
  conversationHistory = conversationHistory.slice(-20)
}

---

## Visualization Data (charts for "View Details")

Every successful POST /ask response includes a `visualization` field with
pre-computed, chart-ready data. Use this to render charts in a "View
Details" expandable panel below the narration, above the Evidence Trace.

### Response shape

{
  "narration": "...",
  "visualization": {
    "chart_type": "factor_breakdown",
    "charts": [
      {
        "id": "factor_contributions",
        "title": "What's driving your risk",
        "chart_kind": "bar",
        "insight": "MARKET dominates at 66.3% of total volatility",
        "data": {
          "labels": ["MARKET", "RATES_PROXY", "USDINR", "BRENT", "GOLD_USD", "Specific Risk"],
          "values": [66.3, 3.6, 1.2, 0.8, 0.4, 30.2],
          "colors": ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#64748B"],
          "unit": "%"
        }
      }
    ]
  }
}

visualization is null for POST /experiment responses — only /ask returns it.
(The example above omits the portfolio_history chart for brevity.)

### portfolio_history chart (always first)

Every /ask response includes a timeseries chart as the first chart in
charts[] (omitted only if market data or a regime path was unavailable).

  id: "portfolio_history"
  chart_kind: "timeseries"

  data fields:
    dates[]                ISO-8601 date strings
    portfolio_value[]      INR value per day
    portfolio_return_pct[] cumulative return %
    regime_sequence[]      "Bull"|"Bear"|"Crisis"
    drawdown_pct[]         drawdown from peak % (always <= 0)
    total_return_pct       summary: total return
    max_drawdown_pct       summary: max drawdown
    window_days            length of all arrays

All arrays are the same length (window_days). The default window is 252
periods; it is shorter when the regime fit or price history covers fewer
(e.g. a custom shorter analysis window), capped at 1260. The same series
is also on the trace as `portfolio_history`.

Recommended rendering:
- Line chart: dates[] on x-axis, portfolio_return_pct[] on y-axis
- Color the background by regime_sequence[]
  (green for Bull, amber for Bear, red for Crisis)
- Show drawdown as a shaded area below zero
- Add a hover tooltip showing date, value in ₹, return %, and regime

For candlestick/OHLC views of individual holdings, fetch Yahoo Finance
directly in the frontend (the backend does not return OHLC data).

### chart_kind values and how to render each

"bar"
  Fields: labels[], values[], colors[], unit
  Render as a vertical or horizontal bar chart.
  If colors[] is present, use per-bar colours.
  If values contain negatives, use red (#EF4444) for negative bars and
  green (#10B981) for positive bars, overriding colors[].
  Show unit as axis label.
  PolicyCheck's "policy_compliance" chart is also chart_kind "bar", but
  its data shape is `checks[]` (rule, limit, actual, passed, unit), not
  labels[]/values[] — branch on the presence of data.checks, not on
  chart_kind alone, and render it as a list of progress bars instead:
  passed=true → green bar; passed=false → red bar, show limit as a marker.

"comparison"
  Fields: metrics[] where each metric has:
    label, before, after, unit, lower_is_better
  Render as side-by-side before/after cards.
  If lower_is_better and after < before: highlight after in green.
  If lower_is_better and after > before: highlight after in red.
  Some comparison charts (PortfolioPerformance's "portfolio_summary") carry
  single-value metrics instead — each metric then has label, value, unit
  with no before/after/lower_is_better. Render those as plain stat cards.

### FactorShock unit guide

Use these fields for display (percentage points):
  experiment.shocks_pct          -> input shocks
  outputs.given_shocks_pct       -> confirmed inputs
  outputs.implied_shocks_pct     -> model-estimated
  visualization.data.shocks_pct  -> chart display

Do NOT use for display (these are fractions):
  outputs.given_shocks.simple    -> internal use
  outputs.given_shocks.log       -> internal use
  outputs.implied_shocks.simple/log -> internal

### Charts returned per experiment type

RiskDecomposition (2 charts):
  1. id: "factor_contributions"
     chart_kind: "bar"
     Shows factor risk breakdown as % of vol.
     "Specific Risk" is always the last bar.

  2. id: "vol_forecast"
     chart_kind: "bar"
     Shows current vol + GARCH forecasts at 5, 10, 20, 60 days.
     data.highlight_index marks the 20-day bar.

FactorShock (2 charts):
  1. id: "shock_impact"
     chart_kind: "bar"
     Per-holding P&L in INR.
     data.formatted[] has pre-formatted ₹ strings.
     data.total is the portfolio-level P&L.
     data.shocks_pct is the given shocks in percentage points, for shock chips.

  2. id: "factor_attribution"
     chart_kind: "bar"
     Per-factor P&L attribution in INR.
     Positive values = factors that helped.
     Negative values = factors that hurt.

CvarRebalance (1 chart):
  1. id: "rebalance_comparison"
     chart_kind: "comparison"
     Before/after CVaR and vol.
     Also has: turnover_pct, commission_inr, commission_formatted at the
     top level of data{}.

ReverseStress (1 chart):
  1. id: "stress_shocks"
     chart_kind: "bar"
     The minimum shock vector per factor.
     data.severity and data.severity_color for the severity badge.
     Colors: #10B981 within-1σ, #F59E0B 1-2σ, #EF4444 2-3σ, #7F1D1D >3σ

PolicyCheck (1 chart):
  1. id: "policy_compliance"
     chart_kind: "bar"
     data.checks[] — one entry per policy rule checked (rule, limit,
     actual, passed, unit). Render as a list of progress bars per the
     "bar" section above, not a generic labels[]/values[] bar chart.

RiskDrift (1 chart):
  1. id: "risk_drift"
     chart_kind: "comparison"
     Before/after vol, plus days_elapsed, regime_before, regime_after,
     regime_changed.

PortfolioPerformance (2 charts):
  1. id: "holding_returns"
     chart_kind: "bar"
     Per-holding total return %, sorted descending.
     Positive bars green, negative bars red.

  2. id: "portfolio_summary"
     chart_kind: "comparison"
     Single-value metrics (no before/after).
     Each metric has: label, value, unit.

### UI placement

Show charts in a collapsible "View Details ▾" panel below the narration
in each AI message. Collapsed by default on mobile, expanded on desktop.

Always show chart.insight as a subtitle above each chart — it is a
one-sentence human summary.

Render charts in the array order returned.

If visualization is null: no "View Details" panel — show only narration
and Evidence Trace.

### Colour conventions (use exactly)

#2563EB  blue     primary / market factor
#10B981  green    positive / safe / bull regime
#F59E0B  amber    warning / bear regime
#EF4444  red      danger / loss / crisis
#8B5CF6  purple   secondary factors
#64748B  grey     specific risk / neutral

### Suggested libraries

Chart.js or Recharts — the data format works with both out of the box.

For Chart.js:
  labels[] → labels
  values[] → data
  colors[] → backgroundColor

For Recharts:
  Map labels[i] + values[i] → { name: labels[i], value: values[i] }

### Key things that will break if done wrong

1. Do not parse the trace JSON to build charts — use only the
   visualization field. The trace is for the Evidence Trace panel only.
2. Do not render charts for /experiment responses — visualization is null
   there.
3. Always check visualization !== null before rendering the View Details
   panel.
4. data.formatted[] strings are already in Indian ₹ notation — use them
   directly, do not reformat the raw values[] for display.
5. For comparison charts with single-value metrics (no before/after),
   render each metric as a stat card, not a progress bar.
6. PolicyCheck's chart_kind is "bar", not "gauge" — do not switch on
   chart_kind to detect it. Switch on the presence of data.checks[]
   instead (every other "bar" chart carries data.labels[]/data.values[],
   never data.checks[]).

---

## Redirects (is_redirect)

Some questions are out of scope: a price scenario for an individual stock
("what if Ratnaveer goes up 10%?"), valuation ("is TCS a good buy?"),
price predictions, "why did X fall today?", or anything not about
finance. They are not errors. `POST /ask` answers HTTP 200 with:

  is_redirect: true
  narration:   the redirect text (what the system can do instead)
  suggestion:  a follow-up chip, as usual
  assistant_turn: the same text as an assistant turn

and `experiment`, `trace`, `result_id`, `agent_execution_trace` and
`visualization` are all null. Every normal response has
`is_redirect: false`.

When is_redirect is true: render narration as a soft grey AI message, not
an error banner (no red, no retry button). Still show the suggestion chip,
and append assistant_turn to conversation_history as normal. Never read
`trace`/`result_id` without checking for null first (no PDF report exists
for a redirect). Questions about *macro* factors ("what if Nifty falls
15%?", "what if crude spikes?") are NOT redirected; they run a shock.

## ISIN support

Brokers that export ISINs (Kotak Securities, HDFC Securities, ICICI Direct)
are supported. `POST /portfolio/upload` resolves each ISIN to an NSE (or,
failing that, BSE `.BO`) symbol via Yahoo Finance and never returns
`isin_not_supported` any more. Common large caps are answered from a
built-in cache without a network call.

Upload response fields added:

  isins_resolved          { "INE002A01018": "RELIANCE.NS", ... }
  isins_skipped           [ { isin, reason, category } ]   // [] if none
  weights_renormalized    true if skipped holdings forced a rescale
  original_holding_count  holdings in the file
  retained_holding_count  holdings in portfolio.holdings
  row_count               same as original_holding_count

`portfolio` contains only the retained holdings: weights sum to 1 over
them, and for value-based files total_value_inr is the retained value.

## ISIN resolution warnings

After a successful upload, check response.isins_skipped.

If isins_skipped.length > 0, show a yellow warning banner above the
portfolio:
  "1 holding was skipped:" (when exactly one) or
  "X holdings were skipped:" (when more than one)
   • [reason for each]
   Weights have been adjusted accordingly."

Categories and icons:
  delisted   → ⚠ yellow
  non_equity → ℹ blue     (mutual funds, bonds/debentures, SGBs)
  sme        → ⚠ yellow
  not_found  → ⚠ yellow
  timeout    → 🔄 grey (show retry button)

When nothing usable remains, the upload fails instead of succeeding:
  422 insufficient_holdings   fewer than 2 holdings could be analysed.
                              error.error lists every reason, and
                              isins_skipped is included in the body —
                              show it as a red error: "None of your
                              holdings could be resolved. Please check
                              your export format."
  503 data_unavailable        the lookups themselves failed (Yahoo
                              timed out / rate-limited). Show a retry
                              button; the file is not at fault.

Notes:
- A symbol may end in `.BO` (BSE) when Yahoo lists the stock only there.
- ETFs resolve like stocks. Mutual funds, bonds and gold bonds are skipped
  with category non_equity.
- `delisted` is reserved: Yahoo gives no delisting signal, so an unknown
  ISIN is reported as not_found ("recently listed or non-standard").

## Error handling for users

Every error response is JSON: `{ "error": "...", "code": "..." }`. Show
these messages based on `code`:

service_busy (503):
  Show countdown: "High demand right now — retrying in Xs" and auto-retry
  after retry_after_seconds. Do NOT show "Something went wrong."

data_unavailable (503):
  Show: "Market data is temporarily unavailable. Please try again."
  Add a retry button.

ai_unavailable (503):
  Show: "AI service is busy. Please try again in a moment."
  Add a retry button.

isin_not_supported: no longer returned (ISINs are resolved automatically;
  see "ISIN support").

unresolved_ticker (422, on ask/experiment):
  Show the error.error field directly — it names the ticker and explains
  the fix.

insufficient_data (422): show error.error directly.
unrecognised_request (422): show error.error directly.

For all 503 responses with retry_after_seconds:
  Auto-retry after that many seconds. Show a countdown timer, not an
  error message. After 2 auto-retries, show the error message with a
  manual retry button.

(`ai_unavailable` replaces the former `gemini_unavailable` code. 500s —
`internal_error`/`compute_error` — always carry the same generic message:
"Something went wrong on our end. Please try again.")

### GET /health additions

`{"status","version","gemini":"configured"|"not_configured",
"upstox":"configured"|"not_configured","active_requests":N,"capacity":8}`

## Regime badge
Source: response.trace.model_params.regime_state.current_label
Values: "Bull" | "Bear" | "Crisis"
Show as a small pill in the chat header, updated after 
each /ask response.
Colours: Bull #10B981, Bear #F59E0B, Crisis #EF4444

---

## Loading states
- Disable send button and scenario buttons during a request
- Show a pulsing indicator in the chat area
- Expected response time: 5–10 seconds (Gemini calls)
- Do not show a timeout error until 60 seconds

---

## Key things that will break if done wrong
1. Not sending conversation_history — the AI loses context 
   and "what should I do about it?" won't work
2. Not appending assistant_turn after each response — 
   same problem
3. Fetching /auth/upstox/login instead of redirecting — 
   OAuth will fail silently
4. Calling /report/{id} before result_id exists — 
   will 404
5. Submitting with weights not summing to 1.0 — 
   will 400
6. Sending fewer than 2 holdings — will 400

---

## Testing the integration
Before building any UI, verify the API is live:
  curl http://localhost:8080/health
Expected: {"status":"ok","version":"0.1.0"}

Then verify /ask works:
  curl -X POST \
    http://localhost:8080/ask \
    -H "Content-Type: application/json" \
    -d '{
      "portfolio": {
        "holdings": [
          {"ticker":"RELIANCE.NS","weight":0.5},
          {"ticker":"HDFCBANK.NS","weight":0.5}
        ],
        "total_value_inr": 1000000
      },
      "message": "What is my risk?",
      "conversation_history": []
    }'
Expected: 200 with narration field non-empty.
