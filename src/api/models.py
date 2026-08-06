from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── date-range base ──────────────────────────────────────────────

class DateRangeRequest(BaseModel):
    """
    Base for all requests that need a date window.

    start_date defaults to 2 years ago so callers don't have to pass it
    for exploratory requests. All child models can override the default
    if they need a different lookback (e.g. BacktestRequest uses 3 years).
    """
    start_date: date = Field(
        default_factory=lambda: date.today() - timedelta(days=730)
    )
    end_date: date = Field(default_factory=date.today)

    @model_validator(mode="after")
    def validate_date_range(self):
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


# ─── shared ───────────────────────────────────────────────────────

class TickerScore(BaseModel):
    ticker:    str
    scores:    dict[str, float]       # factor_name → [-1, +1]
    composite: float | None = None    # IC-weighted blend, null if not fitted


class RegimeInfo(BaseModel):
    label:         Literal["bull", "bear", "sideways"]
    probabilities: dict[str, float]   # {"bull": 0.8, "bear": 0.1, "sideways": 0.1}


# ─── signals ──────────────────────────────────────────────────────

class SignalRequest(DateRangeRequest):
    tickers:   list[str] = Field(..., min_length=1, max_length=200)
    benchmark: str       = "^NSEI"    # Nifty 50 — matches Kite default universe
    provider:  str       = "kite"     # matches Step 1 data provider

    @field_validator("tickers")
    @classmethod
    def unique_tickers(cls, tickers: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
        if not cleaned:
            raise ValueError("at least one non-empty ticker is required")
        return cleaned

    model_config = {"json_schema_extra": {"example": {
        "tickers":   ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
        "benchmark": "^NSEI",
        "provider":  "kite",
    }}}


class SignalResponse(BaseModel):
    as_of:   date
    regime:  RegimeInfo
    signals: list[TickerScore]
    factors: list[str]                # which factors were computed


# ─── portfolio ────────────────────────────────────────────────────

class OptimiseRequest(DateRangeRequest):
    tickers:    list[str] = Field(..., min_length=2, max_length=200)
    method:     Literal["hrp", "black_litterman", "cvar"] = "hrp"
    alpha:      float = Field(0.95, ge=0.9, le=0.99)   # CVaR confidence
    max_weight: float = Field(0.20, ge=0.05, le=1.0)
    benchmark:  str   = "^NSEI"
    provider:   str   = "kite"

    @field_validator("tickers")
    @classmethod
    def unique_tickers(cls, tickers: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
        if len(cleaned) < 2:
            raise ValueError("at least two distinct tickers are required")
        return cleaned

    @model_validator(mode="after")
    def validate_weight_cap(self):
        if self.method in {"black_litterman", "cvar"}:
            if len(self.tickers) * self.max_weight < 1:
                raise ValueError("max_weight is infeasible for the number of tickers")
        return self


class WeightItem(BaseModel):
    ticker: str
    weight: float


class OptimiseResponse(BaseModel):
    method:      str
    weights:     list[WeightItem]
    effective_n: float               # 1 / sum(w²) — diversification measure
    as_of:       date


# ─── risk ─────────────────────────────────────────────────────────

class RiskRequest(DateRangeRequest):
    weights:   dict[str, float] = Field(..., min_length=1, max_length=200)
    benchmark: str = "^NSEI"
    provider:  str = "kite"

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        cleaned = {t.strip().upper(): w for t, w in weights.items()}
        if any(not t for t in cleaned):
            raise ValueError("ticker names must not be empty")
        if any(w < 0 for w in cleaned.values()):
            raise ValueError("weights must be non-negative")
        if sum(cleaned.values()) <= 0:
            raise ValueError("weights must sum to a positive value")
        return cleaned


class RiskResponse(BaseModel):
    annualised_vol:    float
    factor_variance:   float
    specific_variance: float
    factor_contrib:    dict[str, float]
    total_variance:    float


class StressResult(BaseModel):
    scenario:     str
    total_return: float
    max_drawdown: float
    ann_vol:      float
    cvar_daily:   float
    n_days:       int


class StressResponse(BaseModel):
    weights: dict[str, float]
    results: list[StressResult]


# ─── backtest ─────────────────────────────────────────────────────

class BacktestRequest(DateRangeRequest):
    """
    Backtest needs more history than signals — default lookback is 3 years
    so the walk-forward engine has enough warmup data to be meaningful.
    """
    tickers:        list[str] = Field(..., min_length=2, max_length=200)
    method:         Literal["hrp", "cvar"] = "hrp"
    rebalance_freq: int   = Field(21, ge=1, le=252)
    commission_bps: float = Field(5.0, ge=0, le=100)
    benchmark:      str   = "^NSEI"
    provider:       str   = "kite"

    # Override the base default to 3 years for more backtest history
    start_date: date = Field(
        default_factory=lambda: date.today() - timedelta(days=1095)
    )

    @field_validator("tickers")
    @classmethod
    def unique_tickers(cls, tickers: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
        if len(cleaned) < 2:
            raise ValueError("at least two distinct tickers are required")
        return cleaned


class BacktestMetrics(BaseModel):
    total_return: float
    ann_return:   float
    ann_vol:      float
    sharpe:       float
    sortino:      float
    max_drawdown: float
    psr:          float
    dsr:          float
    n_days:       int


class BacktestResponse(BaseModel):
    metrics:      BacktestMetrics
    equity_curve: dict[str, float]   # date_str → cumulative return
    drawdown:     dict[str, float]
    avg_turnover: float

# ─── portfolio analysis ───────────────────────────────────────────

class PortfolioAnalyzeRequest(BaseModel):
    weights:   dict[str, float] = Field(..., min_length=2, max_length=50)
    benchmark: str  = "^NSEI"
    provider:  str  = "kite"

    @field_validator("weights")
    @classmethod
    def validate_and_normalise(cls, weights: dict[str, float]) -> dict[str, float]:
        cleaned = {t.strip().upper(): w for t, w in weights.items()}
        if any(w < 0 for w in cleaned.values()):
            raise ValueError("weights must be non-negative")
        total = sum(cleaned.values())
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        return {t: w / total for t, w in cleaned.items()}


class HoldingDetail(BaseModel):
    ticker:         str
    weight:         float
    factor_scores:  dict[str, float]
    composite:      float | None
    vol_contrib:    float        # fraction of portfolio variance
    is_helping:     bool         # composite > 0.05
    risk_flags:     list[str]


class ConcentrationMetrics(BaseModel):
    hhi:              float   # Herfindahl-Hirschman Index
    effective_n:      float   # 1/HHI
    max_weight:       float
    top3_weight:      float


class CorrelationCluster(BaseModel):
    tickers:   list[str]
    avg_corr:  float
    note:      str


class RebalanceSuggestion(BaseModel):
    suggested_weights:       dict[str, float]
    method:                  str
    effective_n_current:     float
    effective_n_suggested:   float
    improvement:             float   # effective_n_suggested - effective_n_current


class RegimeImpact(BaseModel):
    current_regime:     str
    confidence:         float
    portfolio_score:    float   # weighted avg composite in this regime
    active_factors:     list[str]
    suppressed_factors: list[str]
    recommendation:     str


class PortfolioAnalysisResponse(BaseModel):
    as_of:               date
    n_holdings:          int
    annualised_vol:      float
    max_drawdown:        float
    sharpe:              float | None
    effective_n:         float
    current_regime:      str
    regime_confidence:   float
    holdings:            list[HoldingDetail]
    factor_exposure:     dict[str, float]
    concentration:       ConcentrationMetrics
    correlation_clusters: list[CorrelationCluster]
    risk_decomposition:  RiskResponse
    stress_results:      list[StressResult]
    rebalance:           RebalanceSuggestion
    warnings:            list[str]
    regime_impact:       RegimeImpact