from typing import Literal

Dataset = Literal[
    "equity_ohlcv",
    "fundamentals_income",
    "fundamentals_balance",
    "fundamentals_ratios",
    "options_chains",
    "news",
]
 
Interval = Literal["1d", "1wk", "1mo"]
 
Periodicity = Literal["annual", "quarterly"]