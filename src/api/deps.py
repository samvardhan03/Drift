"""
FastAPI dependency accessors for Drift showcase — returns mock services.
All routes in the showcase are open (no API key required).
"""

from __future__ import annotations

from fastapi import Request

from src.showcase.services import (
    MockAnalyze,
    MockBacktest,
    MockPortfolio,
    MockRisk,
    MockScreener,
    MockSignals,
    get_mock_analyze,
    get_mock_backtest,
    get_mock_portfolio,
    get_mock_risk,
    get_mock_screener,
    get_mock_signals,
)


def get_screener(_: Request = None) -> MockScreener:
    return get_mock_screener()


def get_signals(_: Request = None) -> MockSignals:
    return get_mock_signals()


def get_backtest(_: Request = None) -> MockBacktest:
    return get_mock_backtest()


def get_portfolio(_: Request = None) -> MockPortfolio:
    return get_mock_portfolio()


def get_risk(_: Request = None) -> MockRisk:
    return get_mock_risk()


def get_analyze(_: Request = None) -> MockAnalyze:
    return get_mock_analyze()
