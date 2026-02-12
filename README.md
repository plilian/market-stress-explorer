# Market Stress Explorer

A simple multi-factor stress signal built from OHLCV data:
- Volatility (rolling std of returns)
- Volume shock (volume / rolling mean volume)
- Liquidity proxy (volume / (high-low))

## Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt