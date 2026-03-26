from fastapi import FastAPI, Query
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI(
    title="Technical Indicators API",
    description="Optimized API for RSI and EMA (lightweight)",
    version="2.0.0"
)

LOOKBACK_CANDLES = 250  # enough for EMA 200

# =========================
# Indicator Functions
# =========================

def calculate_ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain_series = pd.Series(gain, index=series.index)
    loss_series = pd.Series(loss, index=series.index)

    avg_gain = gain_series.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss_series.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# =========================
# Optimized Data Fetch
# =========================

def fetch_minimal_data(ticker: str, interval: str):
    ticker_obj = yf.Ticker(ticker)

    # Always fetch small safe window
    df = ticker_obj.history(interval=interval, period="5d")

    df.dropna(inplace=True)

    # Trim to only required candles
    return df.tail(LOOKBACK_CANDLES)


# =========================
# SINGLE OPTIMIZED ENDPOINT
# =========================

@app.get("/technical_snapshot")
def get_technical_snapshot(
    ticker: str = Query(...),
    interval: str = Query("5m")
):
    df = fetch_minimal_data(ticker, interval)

    close = df["Close"]

    # Indicators
    rsi = calculate_rsi(close, 14).iloc[-1]

    ema_9 = calculate_ema(close, 9).iloc[-1]
    ema_50 = calculate_ema(close, 50).iloc[-1]
    ema_100 = calculate_ema(close, 100).iloc[-1]
    ema_200 = calculate_ema(close, 200).iloc[-1]

    price = close.iloc[-1]

    return {
        "ticker": ticker,
        "interval": interval,
        "price": round(float(price), 2),
        "rsi_14": round(float(rsi), 2),
        "ema_9": round(float(ema_9), 2),
        "ema_50": round(float(ema_50), 2),
        "ema_100": round(float(ema_100), 2),
        "ema_200": round(float(ema_200), 2)
    }
