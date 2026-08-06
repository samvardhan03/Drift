from __future__ import annotations
from datetime import date
from typing import Literal
from pydantic import BaseModel, Field


class ScreenRequest(BaseModel):
    universe:        Literal["nifty50", "nifty100", "banknifty", "custom"] = "nifty50"
    custom_tickers:  list[str] = Field(default=[], max_length=50)
    sort_by:         Literal["composite", "momentum", "quality", "value", "risk"] = "composite"
    sector:          str | None = None
    min_composite:   float = -1.0
    max_volatility:  float | None = None
    benchmark:       str = "^NSEI"
    provider:        str = "kite"


class ScreenerRow(BaseModel):
    rank:               int
    ticker:             str
    sector:             str
    composite_score:    float
    factor_scores:      dict[str, float]
    annualised_vol:     float
    regime_compatible:  bool
    classification:     Literal["candidate", "watchlist", "avoid"]
    why:                str
    risk_note:          str


class ScreenResponse(BaseModel):
    as_of:               date
    universe:            str
    total_in_universe:   int
    total_passed_filters: int
    shown:               int
    is_truncated:        bool
    current_regime:      str
    regime_confidence:   float
    results:             list[ScreenerRow]
    skipped_tickers:     list[str]
    note:                str
    data_mode:           Literal["precomputed", "live"] = "live"
    snapshot_created_utc: str | None = None