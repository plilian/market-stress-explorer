from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_ohlcv_yfinance(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data w yfinance.
    Returns a df with columns:
    ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] and DatetimeIndex.
    """
    ticker = (ticker or "").strip()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker={ticker}, period={period}, interval={interval}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.dropna().copy()
    return df
