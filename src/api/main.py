"""
Drift showcase API server

This is a public demonstration shell. All analytics responses are precomputed
fixtures generated on synthetic data. The production inference engine (regime
models, signal construction, portfolio optimisation) is proprietary and not
included.

Routes (no authentication required):
    GET  /health
    GET  /screen        — NSE factor screener (Nifty 50 demo)
    POST /screen
    POST /signals/compute
    POST /portfolio/optimise
    POST /portfolio/analyze
    POST /risk/decompose
    POST /risk/stress
    POST /backtest/run
    GET  /backtest/{job_id}
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import backtest, portfolio, risk, screener, signals, analyze

_DEMO_BANNER = (
    "**Demo build** — All responses are precomputed fixtures generated on synthetic "
    "data. The production inference engine (regime models, signal construction, "
    "portfolio optimisation) is proprietary and not included in this repository."
)

app = FastAPI(
    title="Drift API — Showcase",
    description=_DEMO_BANNER,
    version="0.3.0-showcase",
)

_origins = [
    "http://localhost:3000",
    "http://localhost:8501",
    "https://drift-site-livid.vercel.app",
]
_extra = os.environ.get("DRIFT_CORS_ORIGINS", "")
_origins += [o.strip() for o in _extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "mode": "showcase",
        "note": "Demo build — synthetic fixture data only.",
    }


app.include_router(screener.router)
app.include_router(signals.router)
app.include_router(portfolio.router)
app.include_router(risk.router)
app.include_router(backtest.router)
app.include_router(analyze.router)
