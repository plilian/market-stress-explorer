from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(series: pd.Series) -> pd.Series:
    """
    Stable z-score: ignores inf and handles zero std
    """
    s = series.replace([np.inf, -np.inf], np.nan)
    mu = s.mean(skipna=True)
    sigma = s.std(skipna=True)

    if sigma is None or np.isnan(sigma) or sigma == 0:
        return pd.Series(0.0, index=series.index)

    return (s - mu) / sigma


def compute_stress_score(
    df: pd.DataFrame,
    w_vol: float = 0.4,
    w_volshock: float = 0.4,
    w_liq: float = 0.2,
) -> pd.DataFrame:
    """
    Stress Score = w_vol * z(volatility) + w_volshock * z(vol_shock) + w_liq * z(liq_proxy)
    """
    out = df.copy()

    needed = ["volatility", "vol_shock", "liq_proxy"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns required for stress score: {missing}")

    out["vol_z"] = zscore(out["volatility"])
    out["volshock_z"] = zscore(out["vol_shock"])
    out["liq_z"] = zscore(out["liq_proxy"])

    out["stress_score"] = (
        w_vol * out["vol_z"] +
        w_volshock * out["volshock_z"] +
        w_liq * out["liq_z"]
    )

    return out


def add_regime(df: pd.DataFrame, stress_col: str = "stress_score") -> pd.DataFrame:
    """
    Simple regimes:
    Normal <= 1.5
    Stress  (1.5, 2.5]
    Extreme > 2.5
    You can change! :)
    """
    out = df.copy()
    out["regime"] = "Normal"
    out.loc[out[stress_col] > 1.5, "regime"] = "Stress"
    out.loc[out[stress_col] > 2.5, "regime"] = "Extreme"
    return out
