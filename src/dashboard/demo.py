"""
Drift showcase demo data — loads precomputed fixtures from src/showcase/fixtures/
instead of running the proprietary engine. All numeric values are synthetic.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_FIXTURE_DIR = Path(__file__).parents[2] / "src" / "showcase" / "fixtures"
_PARQUET_PATH = Path(__file__).parents[3] / "data" / "synthetic" / "ohlcv_demo.parquet"

DEMO_TICKERS = [
    "TCS", "INFY", "HDFCBANK", "ICICIBANK", "RELIANCE",
    "HCLTECH", "WIPRO", "SBIN", "ONGC", "ADANIENT",
]


def _load(name: str) -> dict:
    d = json.loads((_FIXTURE_DIR / name).read_text())
    d.pop("_comment", None)
    return d


def load_demo_data() -> dict:
    screener  = _load("screener_nifty50.json")
    signals   = _load("signals_nifty50.json")
    backtest  = _load("backtest_default.json")
    portfolio = _load("portfolio_default.json")
    risk      = _load("risk_default.json")

    weights_hrp = {w["ticker"]: w["weight"] for w in portfolio["weights"]}

    ec_raw = backtest["equity_curve"]
    equity_curve = pd.Series(
        list(ec_raw.values()),
        index=pd.to_datetime(list(ec_raw.keys())),
        name="equity_curve",
    )

    dd_raw = backtest["drawdown"]
    drawdown = pd.Series(
        list(dd_raw.values()),
        index=pd.to_datetime(list(dd_raw.keys())),
        name="drawdown",
    )

    regime_labels = pd.Series(
        [screener["current_regime"]] * len(equity_curve),
        index=equity_curve.index,
        name="regime",
    )

    factor_ic = pd.DataFrame({
        "momentum": [0.063, 0.058, 0.071, 0.052, 0.067, 0.061, 0.048, 0.073, 0.056, 0.069],
        "quality":  [0.054, 0.061, 0.049, 0.066, 0.058, 0.053, 0.071, 0.047, 0.062, 0.055],
        "value":    [0.041, 0.038, 0.045, 0.039, 0.043, 0.037, 0.048, 0.034, 0.042, 0.040],
        "beta":     [0.021, 0.018, 0.024, 0.019, 0.022, 0.017, 0.025, 0.015, 0.023, 0.020],
        "size":     [0.031, 0.029, 0.033, 0.028, 0.032, 0.027, 0.034, 0.026, 0.030, 0.028],
    }, index=pd.bdate_range("2023-07-01", periods=10, freq="21B"))

    ohlcv: pd.DataFrame | None = None
    try:
        ohlcv = pd.read_parquet(_PARQUET_PATH)
    except Exception:
        pass

    return {
        "tickers":        DEMO_TICKERS,
        "screener":       screener,
        "signals":        signals,
        "backtest":       backtest,
        "weights_hrp":    weights_hrp,
        "risk":           risk,
        "equity_curve":   equity_curve,
        "drawdown":       drawdown,
        "regime_labels":  regime_labels,
        "factor_ic":      factor_ic,
        "ohlcv":          ohlcv,
        "current_regime": screener["current_regime"],
        "regime_conf":    screener["regime_confidence"],
        "as_of":          date.fromisoformat(screener["as_of"]),
    }
