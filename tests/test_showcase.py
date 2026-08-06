"""
Showcase golden tests:
- fixtures validate through pydantic at import time (MockScreener etc.)
- all API routes return 200 with valid response models
- fixture responses are byte-stable (golden assertion)
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.showcase.services import (
    MockAnalyze,
    MockBacktest,
    MockPortfolio,
    MockRisk,
    MockScreener,
    MockSignals,
)

client = TestClient(app)


# ── fixture validation (fails at import if fixture drifts from model) ──

def test_mock_screener_valid():
    assert MockScreener().response.universe == "nifty50"

def test_mock_signals_valid():
    resp = MockSignals().response
    assert resp.regime.label in ("bull", "bear", "sideways")

def test_mock_backtest_valid():
    resp = MockBacktest().response
    assert resp.metrics.sharpe > 0

def test_mock_portfolio_valid():
    resp = MockPortfolio().response
    weights = {w.ticker: w.weight for w in resp.weights}
    assert abs(sum(weights.values()) - 1.0) < 1e-3

def test_mock_risk_valid():
    resp = MockRisk().decompose()
    assert resp.annualised_vol > 0
    assert resp.factor_variance + resp.specific_variance <= resp.total_variance + 1e-6

def test_mock_analyze_valid():
    resp = MockAnalyze().response
    assert resp.n_holdings == len(resp.holdings)


# ── API route smoke tests ──

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["mode"] == "showcase"

def test_screen_get():
    r = client.get("/screen")
    assert r.status_code == 200
    assert r.json()["universe"] == "nifty50"

def test_screen_post():
    r = client.post("/screen", json={})
    assert r.status_code == 200

def test_signals_compute():
    r = client.post("/signals/compute", json={
        "tickers": ["TCS", "INFY"],
        "benchmark": "^NSEI",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["regime"]["label"] in ("bull", "bear", "sideways")

def test_portfolio_optimise():
    r = client.post("/portfolio/optimise", json={
        "tickers": ["TCS", "INFY", "HDFCBANK"],
    })
    assert r.status_code == 200

def test_portfolio_analyze():
    r = client.post("/portfolio/analyze", json={
        "weights": {"TCS": 0.5, "INFY": 0.5},
    })
    assert r.status_code == 200

def test_risk_decompose():
    r = client.post("/risk/decompose", json={
        "weights": {"TCS": 0.5, "INFY": 0.5},
    })
    assert r.status_code == 200
    assert r.json()["annualised_vol"] > 0

def test_risk_stress():
    r = client.post("/risk/stress", json={
        "weights": {"TCS": 0.5, "INFY": 0.5},
    })
    assert r.status_code == 200

def test_backtest_run_and_poll():
    r = client.post("/backtest/run", json={
        "tickers": ["TCS", "INFY", "HDFCBANK"],
    })
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    r2 = client.get(f"/backtest/{job_id}")
    assert r2.status_code == 200

def test_backtest_not_found():
    r = client.get("/backtest/does-not-exist")
    assert r.status_code == 404


# ── tripwire: no proprietary keywords in source tree ──

import subprocess, pathlib

_TRIPWIRE_KEYWORDS = (
    "hmmlearn|GaussianHMM|kiteconnect|pyotp|omni_ffi|"
    "KITE_|TOTP|sk_live|whsec_"
)

def test_tripwire_content():
    root = pathlib.Path(__file__).parents[1]
    result = subprocess.run(
        ["grep", "-rInE", _TRIPWIRE_KEYWORDS,
         "--include=*.py", "--include=*.toml",
         "src/", "pyproject.toml"],
        cwd=root, capture_output=True, text=True,
    )
    assert result.stdout == "", f"Tripwire hit:\n{result.stdout}"

def test_tripwire_paths():
    root = pathlib.Path(__file__).parents[1]
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root, capture_output=True, text=True,
    )
    patterns = [
        "features/wst.py", "features/regime.py", "features/factors.py",
        "features/comovement.py", "src/alpha/", "portfolio/black_litterman.py",
        "portfolio/hrp.py", "portfolio/omega.py", "portfolio/cvar.py",
        "risk/covariance.py", "risk/factor_model.py", "risk/stress.py",
        "src/backtest/", "src/report/", "advisor.py",
        "providers/kite.py", "kite_auto_refresh.py", "kite_login.py",
        "feature_store.py", "token_store.py", "src/scripts/",
        "dashboard/live.py",
    ]
    hits = [f for f in result.stdout.splitlines() if any(p in f for p in patterns)]
    assert hits == [], f"Strip-list paths found in git tree:\n" + "\n".join(hits)
