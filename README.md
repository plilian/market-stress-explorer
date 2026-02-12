# 🚀 Market Stress Explorer

**Market Stress Explorer** is a multi-asset market regime detection tool built from raw OHLCV data.

Instead of predicting price, the tool focuses on detecting **market instability regimes** using a multi-factor stress signal.

---

## 🧠 Concept

The stress signal is built from three core market microstructure proxies:

- **Volatility Expansion** → Rolling standard deviation of returns  
- **Volume Shock** → Volume vs rolling mean volume  
- **Liquidity Proxy** → Volume / Price Range (High - Low)  

Each component is normalized and combined into a unified **Stress Score**.

The system then classifies market regimes into:
- Normal
- Stress
- Extreme

Thresholds can be:
- Auto-calibrated via Percentiles
- Auto-calibrated via Mean ± Std
- Manually defined

---

## 🌍 Multi-Asset Support

The tool works across asset classes via Yahoo Finance:

| Asset Class | Examples |
|---|---|
| Crypto | BTC-USD, ETH-USD |
| Stocks | AAPL, NVDA, TSLA |
| Indices | ^GSPC, ^IXIC |
| FX | EURUSD=X, USDTRY=X |
| Commodities | GC=F (Gold), CL=F (Oil) |

---

## 📊 Features

- Multi-factor stress signal engineering
- Auto adaptive regime thresholds
- Forward stress diagnostics
- Stress event detection
- Regime distribution analysis
- Multi-asset compatibility
- Streamlit interactive dashboard

---

## 🛠 Tech Stack

- Python
- Pandas / NumPy
- Plotly
- Streamlit
- Time-Series Signal Engineering

---

## ⚙️ Local Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
