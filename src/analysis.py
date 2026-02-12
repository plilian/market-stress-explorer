from __future__ import annotations

import numpy as np
import pandas as pd


def add_forward_metrics(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Adds forward-looking metrics to help analyze whether stress relates to future instability.
    forward returns: 1d, 5d, 20d
    forward volatility: std of returns over next 20 days (shifted)
    """
    out = df.copy()

    out["ret"] = out[price_col].pct_change()

    for h in [1, 5, 20]:
        out[f"fwd_ret_{h}d"] = out[price_col].shift(-h) / out[price_col] - 1.0
    out["fwd_vol_20d"] = out["ret"].rolling(20).std().shift(-20)

    return out


def regime_summary(df: pd.DataFrame, regime_col: str = "regime") -> pd.DataFrame:
    """
    Returns a small table: count and percentage per regime
    """
    counts = df[regime_col].value_counts(dropna=False)
    perc = (counts / counts.sum() * 100).round(1)
    res = pd.DataFrame({"days": counts, "pct": perc}).reset_index().rename(columns={"index": "regime"})
    return res


def top_stress_events(
    df: pd.DataFrame,
    stress_col: str = "stress_score",
    n: int = 5,
) -> pd.DataFrame:
    """
    top N stress dates w basic context metrics.
    """
    cols = [stress_col, "Close"]
    extra = [c for c in ["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d", "fwd_vol_20d", "regime"] if c in df.columns]
    cols += extra

    tmp = df[cols].dropna(subset=[stress_col]).sort_values(stress_col, ascending=False).head(n).copy()
    tmp = tmp.reset_index().rename(columns={"index": "date"})
    return tmp


def correlation_snapshot(df: pd.DataFrame, stress_col: str = "stress_score") -> pd.DataFrame:
    """
    Quick correlations between stress / forward metrics.
    """
    candidates = ["fwd_vol_20d", "fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d"]
    rows = []
    for c in candidates:
        if c in df.columns:
            sub = df[[stress_col, c]].dropna()
            if len(sub) >= 30:
                corr = sub[stress_col].corr(sub[c])
                rows.append({"metric": c, "corr_with_stress": float(np.round(corr, 4))})
    return pd.DataFrame(rows)

def compute_thresholds(
    stress_series: pd.Series,
    method: str = "percentile",
    p_stress: float = 0.85,
    p_extreme: float = 0.95,
    k_stress: float = 1.0,
    k_extreme: float = 2.0,
) -> tuple[float, float]:
    import numpy as np

    s = stress_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 50:
        mu = float(s.mean()) if len(s) else 0.0
        sigma = float(s.std()) if len(s) else 1.0
        return mu + 1.0 * sigma, mu + 2.0 * sigma

    method = (method or "").lower().strip()

    if method == "std":
        mu = float(s.mean())
        sigma = float(s.std())
        return mu + k_stress * sigma, mu + k_extreme * sigma

    stress_thr = float(s.quantile(p_stress))
    extreme_thr = float(s.quantile(p_extreme))

    if extreme_thr <= stress_thr:
        extreme_thr = stress_thr + 1e-6

    return stress_thr, extreme_thr

