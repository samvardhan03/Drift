"""
NSE universe definitions for the Drift screener.

Ticker symbols are in Kite Connect format — bare NSE trading symbols
without any .NS or .BO suffix.

These lists reflect approximate Nifty compositions as of mid-2026.
Verify against nseindia.com → Indices → Constituents for current members.
"""

from __future__ import annotations

from typing import Literal

UniverseName = Literal["nifty50", "nifty100", "banknifty", "custom"]

# ── Nifty 50 ─────────────────────────────────────────────────────
NIFTY50_TICKERS: list[str] = [
    "ADANIENT",   "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJFINSV", "BAJFINANCE", "BHARTIARTL", "BPCL",       "BRITANNIA",
    "CIPLA",      "COALINDIA",  "DIVISLAB",   "DRREDDY",    "EICHERMOT",
    "GRASIM",     "HCLTECH",    "HDFCBANK",   "HDFCLIFE",   "HEROMOTOCO",
    "HINDALCO",   "HINDUNILVR", "ICICIBANK",  "INDUSINDBK", "INFY",
    "ITC",        "JSWSTEEL",   "KOTAKBANK",  "LT",         "MARUTI",
    "M&M",         "NESTLEIND",  "NTPC",       "ONGC",       "POWERGRID",
    "RELIANCE",   "SBILIFE",    "SBIN",       "SUNPHARMA",  "TATACONSUM",
    "TATAMOTORS", "TATASTEEL",  "TCS",        "TECHM",      "TITAN",
    "ULTRACEMCO", "WIPRO",      "BAJAJAUTO",  "SHREECEM",   "TRENT",
]

# ── Bank Nifty ───────────────────────────────────────────────────
BANKNIFTY_TICKERS: list[str] = [
    "HDFCBANK",   "ICICIBANK",   "KOTAKBANK",  "AXISBANK",
    "SBIN",       "INDUSINDBK",  "BANDHANBNK", "FEDERALBNK",
    "IDFCFIRSTB", "AUBANK",      "PNB",        "BANKBARODA",
]

# ── Nifty Next 50 (added to Nifty50 = Nifty100) ──────────────────
NIFTY_NEXT50_TICKERS: list[str] = [
    "ABB",        "AMBUJACEM",  "DMART",      "BERGEPAINT",  "BOSCHLTD",
    "CHOLAFIN",   "COLPAL",     "CONCOR",     "CUMMINSIND",  "DABUR",
    "DLF",        "GODREJCP",   "GODREJPROP", "HAL",         "HAVELLS",
    "ICICIGI",    "ICICIPRULI", "INDUSTOWER", "IRCTC",       "LTIM",
    "LUPIN",      "MARICO",     "MCDOWELL-N", "MOTHERSON",   "MUTHOOTFIN",
    "NAUKRI",     "NHPC",       "NMDC",       "PAGEIND",     "PIIND",
    "PIDILITIND", "RECLTD",     "SAIL",       "SIEMENS",     "SRF",
    "TORNTPHARM", "TVSMOTOR",   "UBL",        "VEDL",        "VOLTAS",
    "ZYDUSLIFE",  "ALKEM",      "LICI",       "ZOMATO",      "NYKAA",
    "PATANJALI",  "POLICYBZR",  "PAYTM",      "TATAPOWER",   "LODHA",
]

NIFTY100_TICKERS: list[str] = list(
    dict.fromkeys(NIFTY50_TICKERS + NIFTY_NEXT50_TICKERS)
)

# ── Sector map ───────────────────────────────────────────────────
SECTOR_MAP: dict[str, str] = {
    "RELIANCE": "Energy",     "ONGC": "Energy",       "BPCL": "Energy",
    "COALINDIA": "Energy",    "NMDC": "Materials",    "VEDL": "Materials",
    "SAIL": "Materials",      "TATAPOWER": "Utilities",
    "HDFCBANK": "Financials", "ICICIBANK": "Financials","KOTAKBANK": "Financials",
    "AXISBANK": "Financials", "SBIN": "Financials",   "INDUSINDBK": "Financials",
    "BANDHANBNK": "Financials","FEDERALBNK": "Financials","IDFCFIRSTB": "Financials",
    "AUBANK": "Financials",   "PNB": "Financials",    "BANKBARODA": "Financials",
    "BAJFINANCE": "Financials","BAJAJFINSV": "Financials","HDFCLIFE": "Financials",
    "SBILIFE": "Financials",  "ICICIGI": "Financials","ICICIPRULI": "Financials",
    "CHOLAFIN": "Financials", "MUTHOOTFIN": "Financials","LICI": "Financials",
    "RECLTD": "Financials",
    "TCS": "Technology",      "INFY": "Technology",   "HCLTECH": "Technology",
    "WIPRO": "Technology",    "TECHM": "Technology",  "LTIM": "Technology",
    "NAUKRI": "Technology",   "POLICYBZR": "Technology","PAYTM": "Technology",
    "NYKAA": "Technology",    "ZOMATO": "Technology",
    "MARUTI": "Consumer Discretionary","TATAMOTORS": "Consumer Discretionary",
    "M&M": "Consumer Discretionary",    "BAJAJAUTO": "Consumer Discretionary",
    "HEROMOTOCO": "Consumer Discretionary","EICHERMOT": "Consumer Discretionary",
    "TVSMOTOR": "Consumer Discretionary","TITAN": "Consumer Discretionary",
    "TRENT": "Consumer Discretionary", "DMART": "Consumer Discretionary",
    "PAGEIND": "Consumer Discretionary","BERGEPAINT": "Consumer Discretionary",
    "ASIANPAINT": "Consumer Discretionary","VOLTAS": "Consumer Discretionary",
    "HINDUNILVR": "Consumer Staples","ITC": "Consumer Staples",
    "NESTLEIND": "Consumer Staples", "BRITANNIA": "Consumer Staples",
    "DABUR": "Consumer Staples",     "MARICO": "Consumer Staples",
    "COLPAL": "Consumer Staples",    "GODREJCP": "Consumer Staples",
    "TATACONSUM": "Consumer Staples","UBL": "Consumer Staples",
    "MCDOWELL-N": "Consumer Staples","PATANJALI": "Consumer Staples",
    "SUNPHARMA": "Healthcare",  "DRREDDY": "Healthcare",  "CIPLA": "Healthcare",
    "DIVISLAB": "Healthcare",   "APOLLOHOSP": "Healthcare","LUPIN": "Healthcare",
    "TORNTPHARM": "Healthcare", "ALKEM": "Healthcare",    "ZYDUSLIFE": "Healthcare",
    "LT": "Industrials",        "ADANIPORTS": "Industrials","CONCOR": "Industrials",
    "HAL": "Industrials",       "IRCTC": "Industrials",   "SIEMENS": "Industrials",
    "ABB": "Industrials",       "CUMMINSIND": "Industrials","BOSCHLTD": "Industrials",
    "MOTHERSON": "Industrials", "HAVELLS": "Industrials",
    "TATASTEEL": "Materials",   "JSWSTEEL": "Materials",  "HINDALCO": "Materials",
    "GRASIM": "Materials",      "ULTRACEMCO": "Materials","AMBUJACEM": "Materials",
    "SHREECEM": "Materials",    "PIIND": "Materials",     "SRF": "Materials",
    "PIDILITIND": "Materials",
    "DLF": "Real Estate",       "GODREJPROP": "Real Estate","LODHA": "Real Estate",
    "NTPC": "Utilities",        "POWERGRID": "Utilities", "NHPC": "Utilities",
    "BHARTIARTL": "Communication","INDUSTOWER": "Communication",
    "ADANIENT": "Conglomerate",
}


def get_universe(name: UniverseName) -> list[str]:
    return {
        "nifty50":   NIFTY50_TICKERS,
        "nifty100":  NIFTY100_TICKERS,
        "banknifty": BANKNIFTY_TICKERS,
        "custom":    [],
    }[name]


def get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker, "Other")


SCREENER_FREE_LIMIT  = 10
SCREENER_PRO_LIMIT   = 500