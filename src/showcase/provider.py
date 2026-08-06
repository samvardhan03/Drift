"""
SyntheticProvider — serves bundled synthetic OHLCV from
data/synthetic/ohlcv_demo.parquet under the same loader contract:
MultiIndex(date, ticker), columns open/high/low/close/volume.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_PARQUET_PATH = Path(__file__).parents[3] / "data" / "synthetic" / "ohlcv_demo.parquet"


class SyntheticProvider:
    _df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_parquet(_PARQUET_PATH)
        return self._df

    def equity_ohlcv(
        self,
        tickers: list[str],
        **kwargs,
    ) -> pd.DataFrame:
        df = self.load()
        available = df.index.get_level_values("ticker").unique()
        requested = [t for t in tickers if t in available]
        if not requested:
            return df.iloc[:0]
        return df[df.index.get_level_values("ticker").isin(requested)]
