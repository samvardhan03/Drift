"""
Mock services for Drift showcase — load and validate fixture JSON through the
same pydantic models the production routers use. A fixture that fails
validation crashes at import, loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.api.models import (
    BacktestResponse,
    OptimiseResponse,
    PortfolioAnalysisResponse,
    RiskResponse,
    SignalResponse,
    StressResponse,
)
from src.api.screener_models import ScreenResponse

_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    path = _FIXTURE_DIR / name
    data = json.loads(path.read_text())
    data.pop("_comment", None)
    return data


class MockScreener:
    response: ScreenResponse = ScreenResponse.model_validate(_load("screener_nifty50.json"))

    def get(self) -> ScreenResponse:
        return self.response


class MockSignals:
    response: SignalResponse = SignalResponse.model_validate(_load("signals_nifty50.json"))

    def get(self) -> SignalResponse:
        return self.response


class MockBacktest:
    response: BacktestResponse = BacktestResponse.model_validate(_load("backtest_default.json"))

    def get(self) -> BacktestResponse:
        return self.response


class MockPortfolio:
    response: OptimiseResponse = OptimiseResponse.model_validate(_load("portfolio_default.json"))

    def get(self) -> OptimiseResponse:
        return self.response


class MockRisk:
    decompose_response: RiskResponse = RiskResponse.model_validate(_load("risk_default.json"))
    stress_response: StressResponse  = StressResponse.model_validate(_load("stress_default.json"))

    def decompose(self) -> RiskResponse:
        return self.decompose_response

    def stress(self) -> StressResponse:
        return self.stress_response


class MockAnalyze:
    response: PortfolioAnalysisResponse = PortfolioAnalysisResponse.model_validate(
        _load("analyze_default.json")
    )

    def get(self) -> PortfolioAnalysisResponse:
        return self.response


_screener  = MockScreener()
_signals   = MockSignals()
_backtest  = MockBacktest()
_portfolio = MockPortfolio()
_risk      = MockRisk()
_analyze   = MockAnalyze()


def get_mock_screener()  -> MockScreener:  return _screener
def get_mock_signals()   -> MockSignals:   return _signals
def get_mock_backtest()  -> MockBacktest:  return _backtest
def get_mock_portfolio() -> MockPortfolio: return _portfolio
def get_mock_risk()      -> MockRisk:      return _risk
def get_mock_analyze()   -> MockAnalyze:   return _analyze
