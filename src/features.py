from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out[price_col].pct_change()
    return out


def add_rolling_volatility(df: pd.DataFrame, ret_col: str = "ret", window: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["volatility"] = out[ret_col].rolling(window).std()
    return out


def add_volume_shock(df: pd.DataFrame, vol_col: str = "Volume", window: int = 30) -> pd.DataFrame:
    out = df.copy()
    denom = out[vol_col].rolling(window).mean()
    out["vol_shock"] = out[vol_col] / denom
    return out


def add_liquidity_proxy(df: pd.DataFrame, vol_col: str = "Volume") -> pd.DataFrame:
    """
    Liquidity proxy = Volume / (High-Low)
    """
    out = df.copy()
    out["range"] = (out["High"] - out["Low"]).replace(0, np.nan)
    out["liq_proxy"] = out[vol_col] / out["range"]
    return out


def add_momentum_decay(df: pd.DataFrame, price_col: str = "Close", short: int = 5, long: int = 20) -> pd.DataFrame:
    """
    Momentum decay proxy = (short return) / (long return)
    """
    out = df.copy()
    out["ret_short"] = out[price_col].pct_change(short)
    out["ret_long"] = out[price_col].pct_change(long)
    out["mom_decay"] = out["ret_short"] / out["ret_long"].replace(0, np.nan)
    return out
