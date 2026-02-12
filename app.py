from __future__ import annotations

import streamlit as st
import pandas as pd

from src.data import fetch_ohlcv_yfinance
from src.features import (
    add_returns,
    add_rolling_volatility,
    add_volume_shock,
    add_liquidity_proxy,
)
from src.scoring import compute_stress_score, add_regime
from src.viz import plot_price_and_stress, plot_feature_breakdown
from src.analysis import (
    add_forward_metrics,
    regime_summary,
    top_stress_events,
    correlation_snapshot,
    compute_thresholds
)

st.set_page_config(page_title="Market Stress Explorer", layout="wide")


@st.cache_data(ttl=3600)
def load_and_compute(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = fetch_ohlcv_yfinance(ticker=ticker, period=period, interval=interval)

    df = add_returns(df)
    df = add_rolling_volatility(df, window=20)
    df = add_volume_shock(df, window=30)
    df = add_liquidity_proxy(df)

    df = compute_stress_score(df)
    df = add_regime(df)

    # forward metrics can create NaNs at the tail; keep them
    df = add_forward_metrics(df)

    # only ensure core values exist
    df = df.dropna(subset=["Close", "stress_score"]).copy()
    return df


def about_tab():
    st.header("About")
    st.write("""
This project is built and maintained by **Parham Lilian**.

I’m a **Data & Analytics Developer** focused on building practical data tools:
- Signal engineering & time-series analytics
- Market / incentive / behavior systems
- Data products (apps, bots, dashboards)
""")
    st.link_button("Connect on LinkedIn", "https://linkedin.com/in/parhamlilian")

    st.divider()
    st.subheader("What this tool does")
    st.write("""
**Market Stress Explorer** is a multi-factor **multi-asset** regime detection tool.

It does NOT try to predict price.
It detects unstable regimes using:
- Volatility expansion
- Volume shock
- Liquidity proxy

Supported markets (examples):
- Crypto: BTC-USD, ETH-USD
- Stocks: AAPL, NVDA
- Indices: ^GSPC, ^IXIC
- FX: EURUSD=X, USDTRY=X
- Commodities: GC=F (Gold), CL=F (Oil)

Metrics are normalized and combined into a single stress score.
""")

    st.subheader("Disclaimer")
    st.write("Analytics/monitoring only. Not financial advice.")


def main():
    st.title("Market Stress Explorer")
    st.caption("Multi-asset market stress & regime detection tool (Crypto, Stocks, Indices, FX, Commodities) using OHLCV data.")

    with st.sidebar:
        st.header("Inputs")

        with st.form("controls", clear_on_submit=False):
            ticker = st.text_input(
                "Ticker",
                value="BTC-USD",
                help="Works across asset classes via Yahoo Finance. Examples: BTC-USD, ETH-USD, AAPL, NVDA, SPY, ^GSPC, EURUSD=X, GC=F"
            )
            period = st.selectbox("Period", ["6mo", "1y", "2y", "5y", "10y"], index=2)
            interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

            with st.expander("Example tickers (multi-asset)"):
                st.write("""
**Crypto:** BTC-USD, ETH-USD, SOL-USD  
**Stocks:** AAPL, NVDA, TSLA, SPY  
**Indices:** ^GSPC (S&P 500), ^IXIC (NASDAQ)  
**FX:** EURUSD=X, USDTRY=X  
**Commodities:** GC=F (Gold), CL=F (Oil)
""")

            st.divider()
            st.subheader("Regime thresholds")

            threshold_mode = st.radio(
                "Threshold mode",
                ["Auto (Percentiles)", "Auto (Mean±Std)", "Manual"],
                index=0
            )

            # Show only relevant controls
            if threshold_mode == "Manual":
                stress_thr_manual = st.slider("Stress Threshold (manual)", 0.0, 5.0, 1.5, 0.1)
                extreme_thr_manual = st.slider("Extreme Threshold (manual)", 0.0, 8.0, 2.5, 0.1)
                p_stress = p_extreme = k_stress = k_extreme = None
            elif threshold_mode == "Auto (Percentiles)":
                p_stress = st.slider("Stress percentile (P)", 0.60, 0.95, 0.85, 0.01)
                p_extreme = st.slider("Extreme percentile (P)", 0.70, 0.99, 0.95, 0.01)
                stress_thr_manual = extreme_thr_manual = None
                k_stress = k_extreme = None
            else:  # Auto (Mean±Std)
                k_stress = st.slider("Stress = mean + k·std", 0.5, 3.0, 1.0, 0.1)
                k_extreme = st.slider("Extreme = mean + k·std", 1.0, 5.0, 2.0, 0.1)
                stress_thr_manual = extreme_thr_manual = None
                p_stress = p_extreme = None

            run = st.form_submit_button("Run")

    tabs = st.tabs(["Dashboard", "Analysis", "Data", "About"])

    if not run and "df" not in st.session_state:
        with tabs[0]:
            st.info("Set inputs and click **Run**.")
        with tabs[3]:
            about_tab()
        return

    if run:
        try:
            df = load_and_compute(ticker, period, interval).copy()

            # Decide thresholds
            if threshold_mode == "Auto (Mean±Std)":
                stress_thr, extreme_thr = compute_thresholds(
                    df["stress_score"],
                    method="std",
                    k_stress=float(k_stress),
                    k_extreme=float(k_extreme),
                )
            elif threshold_mode == "Auto (Percentiles)":
                stress_thr, extreme_thr = compute_thresholds(
                    df["stress_score"],
                    method="percentile",
                    p_stress=float(p_stress),
                    p_extreme=float(p_extreme),
                )
            else:
                stress_thr, extreme_thr = float(stress_thr_manual), float(extreme_thr_manual)

            # Apply regimes
            df["regime"] = "Normal"
            df.loc[df["stress_score"] > float(stress_thr), "regime"] = "Stress"
            df.loc[df["stress_score"] > float(extreme_thr), "regime"] = "Extreme"

            # Store
            st.session_state["df"] = df
            st.session_state["ticker"] = ticker
            st.session_state["threshold_mode"] = threshold_mode
            st.session_state["stress_thr"] = float(stress_thr)
            st.session_state["extreme_thr"] = float(extreme_thr)

        except Exception as e:
            with tabs[0]:
                st.error(f"Error: {e}")
            with tabs[3]:
                about_tab()
            return

    df = st.session_state["df"]
    ticker = st.session_state.get("ticker", "BTC-USD")

    if df.empty:
        with tabs[0]:
            st.warning("Not enough data to compute the signal. Try a larger period or smaller interval.")
        with tabs[3]:
            about_tab()
        return

    latest = df.iloc[-1]

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Price", f"{latest['Close']:.2f}")
        c2.metric("Stress Score", f"{latest['stress_score']:.2f}")
        c3.metric("Regime", str(latest.get("regime", "N/A")))
        c4.metric("Rows", len(df))

        st.caption(
            f"Thresholds ({st.session_state.get('threshold_mode','N/A')}): "
            f"Stress > {st.session_state.get('stress_thr', 0.0):.3f} | "
            f"Extreme > {st.session_state.get('extreme_thr', 0.0):.3f}"
        )

        with st.expander("Stress score distribution (quick check)"):
            st.write(df["stress_score"].describe())

        st.divider()
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(plot_price_and_stress(df), use_container_width=True)
        with right:
            st.plotly_chart(plot_feature_breakdown(df), use_container_width=True)

    with tabs[1]:
        st.header("Result Analysis")

        lookback = st.selectbox("Lookback Window", [30, 60, 90, 180], index=1)
        df_recent = df.tail(lookback).copy()

        if df_recent.empty:
            st.warning("Not enough rows for the selected lookback window.")
        else:
            st.subheader("Regime Breakdown (lookback)")
            rs = regime_summary(df_recent)
            st.dataframe(rs, use_container_width=True)

            st.divider()
            st.subheader("Top Stress Events")
            topn = st.slider("Top N Events", 3, 10, 5)
            te = top_stress_events(df, n=topn)
            st.dataframe(te, use_container_width=True)

            st.divider()
            st.subheader("Stress vs Future Instability (quick diagnostic)")
            cs = correlation_snapshot(df)
            if cs.empty:
                st.info("Not enough data to compute correlations (try larger period or smaller interval).")
            else:
                st.dataframe(cs, use_container_width=True)

            st.divider()
            st.subheader("Copy-ready summary")
            normal_pct = float(rs.loc[rs["regime"] == "Normal", "pct"].values[0]) if (rs["regime"] == "Normal").any() else 0.0
            stress_pct = float(rs.loc[rs["regime"] == "Stress", "pct"].values[0]) if (rs["regime"] == "Stress").any() else 0.0
            extreme_pct = float(rs.loc[rs["regime"] == "Extreme", "pct"].values[0]) if (rs["regime"] == "Extreme").any() else 0.0

            summary = (
                f"Lookback: last {lookback} sessions\n"
                f"Current regime: {latest.get('regime','N/A')} | Stress score: {latest['stress_score']:.2f}\n"
                f"Thresholds ({st.session_state.get('threshold_mode','N/A')}): "
                f"Stress > {st.session_state.get('stress_thr', 0.0):.3f} | "
                f"Extreme > {st.session_state.get('extreme_thr', 0.0):.3f}\n"
                f"Regime mix: Normal {normal_pct:.1f}% | Stress {stress_pct:.1f}% | Extreme {extreme_pct:.1f}%\n"
                f"Avg stress (lookback): {df_recent['stress_score'].mean():.2f} | Max: {df_recent['stress_score'].max():.2f}"
            )
            st.code(summary, language="text")

    with tabs[2]:
        st.subheader("Data Preview")
        st.dataframe(df.tail(300), use_container_width=True)

        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button("Download CSV", csv, f"{ticker}_stress.csv", "text/csv")

    with tabs[3]:
        about_tab()


if __name__ == "__main__":
    main()
