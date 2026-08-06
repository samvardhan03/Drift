#!/usr/bin/env bash
# verify_showcase.sh — smoke-test every demo route against the running API
# Usage: ./scripts/verify_showcase.sh [BASE_URL]
# Defaults to http://localhost:8000

set -euo pipefail

BASE="${1:-http://localhost:8000}"

ok() { echo "[OK]  $1"; }
fail() { echo "[FAIL] $1"; exit 1; }

# health
STATUS=$(curl -sf "$BASE/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['mode'])")
[ "$STATUS" = "showcase" ] && ok "GET /health" || fail "GET /health mode != showcase"

# screener GET
curl -sf "$BASE/screen" | python3 -c "
import sys,json; d=json.load(sys.stdin)
assert d['universe'] == 'nifty50', f\"universe={d['universe']}\"
assert len(d['results']) > 0, 'no results'
" && ok "GET /screen" || fail "GET /screen"

# screener POST
curl -sf -X POST "$BASE/screen" -H "Content-Type: application/json" -d '{}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['universe']=='nifty50'" \
  && ok "POST /screen" || fail "POST /screen"

# signals
curl -sf -X POST "$BASE/signals/compute" \
  -H "Content-Type: application/json" \
  -d '{"tickers":["TCS","INFY"],"benchmark":"^NSEI"}' \
  | python3 -c "
import sys,json; d=json.load(sys.stdin)
assert d['regime']['label'] in ('bull','bear','sideways'), f\"label={d['regime']['label']}\"
" && ok "POST /signals/compute" || fail "POST /signals/compute"

# portfolio optimise
curl -sf -X POST "$BASE/portfolio/optimise" \
  -H "Content-Type: application/json" \
  -d '{"tickers":["TCS","INFY","HDFCBANK"]}' \
  | python3 -c "
import sys,json; d=json.load(sys.stdin)
total = sum(w['weight'] for w in d['weights'])
assert abs(total - 1.0) < 0.01, f'weights sum={total}'
" && ok "POST /portfolio/optimise" || fail "POST /portfolio/optimise"

# portfolio analyze
curl -sf -X POST "$BASE/portfolio/analyze" \
  -H "Content-Type: application/json" \
  -d '{"weights":{"TCS":0.5,"INFY":0.5}}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['n_holdings']>0" \
  && ok "POST /portfolio/analyze" || fail "POST /portfolio/analyze"

# risk decompose
curl -sf -X POST "$BASE/risk/decompose" \
  -H "Content-Type: application/json" \
  -d '{"weights":{"TCS":0.5,"INFY":0.5}}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['annualised_vol']>0" \
  && ok "POST /risk/decompose" || fail "POST /risk/decompose"

# risk stress
curl -sf -X POST "$BASE/risk/stress" \
  -H "Content-Type: application/json" \
  -d '{"weights":{"TCS":0.5,"INFY":0.5}}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'results' in d" \
  && ok "POST /risk/stress" || fail "POST /risk/stress"

# backtest run + poll
JOB=$(curl -sf -X POST "$BASE/backtest/run" \
  -H "Content-Type: application/json" \
  -d '{"tickers":["TCS","INFY","HDFCBANK"]}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
sleep 1
curl -sf "$BASE/backtest/$JOB" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='done'" \
  && ok "POST /backtest/run + GET /backtest/{id}" || fail "POST /backtest/run"

echo ""
echo "All showcase routes verified."
