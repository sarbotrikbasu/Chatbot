from fastapi import FastAPI, Query
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI(
    title="Technical Indicators API",
    description="Optimized API for RSI and EMA (minimal data, no overflow)",
    version="2.0.0"
)

# =========================
# CONFIG
# =========================
LOOKBACK_CANDLES = 250  # enough for EMA 200


# =========================
# INDICATOR FUNCTIONS
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
# OPTIMIZED DATA FETCH
# =========================

def fetch_minimal_data(ticker: str, interval: str):
    try:
        ticker_obj = yf.Ticker(ticker)

        # Fetch small safe dataset
        df = ticker_obj.history(interval=interval, period="5d")

        if df.empty:
            return None

        df.dropna(inplace=True)

        # CRITICAL: trim BEFORE calculations
        return df.tail(LOOKBACK_CANDLES)

    except Exception:
        return None


# =========================
# SINGLE SNAPSHOT ENDPOINT
# =========================

@app.get("/technical_snapshot")
def get_technical_snapshot(
    ticker: str = Query(..., description="Stock ticker (e.g., RELIANCE.NS)"),
    interval: str = Query("5m", description="Timeframe (1m, 5m, 15m, etc.)")
):
    df = fetch_minimal_data(ticker, interval)

    if df is None or len(df) < 50:
        return {
            "error": "Insufficient data or invalid ticker"
        }

    close = df["Close"]

    try:
        # Indicators
        rsi_14 = calculate_rsi(close, 14).iloc[-1]

        ema_9 = calculate_ema(close, 9).iloc[-1]
        ema_50 = calculate_ema(close, 50).iloc[-1]
        ema_100 = calculate_ema(close, 100).iloc[-1]
        ema_200 = calculate_ema(close, 200).iloc[-1]

        price = close.iloc[-1]

        # Ensure pure Python floats (important for serialization)
        return {
            "ticker": ticker,
            "interval": interval,
            "price": float(round(price, 2)),
            "rsi_14": float(round(rsi_14, 2)),
            "ema_9": float(round(ema_9, 2)),
            "ema_50": float(round(ema_50, 2)),
            "ema_100": float(round(ema_100, 2)),
            "ema_200": float(round(ema_200, 2))
        }

    except Exception:
        return {
            "error": "Indicator calculation failed"
        }


# =========================
# OPTIONAL: INDIVIDUAL EMA
# =========================

@app.get("/ema")
def get_ema(
    ticker: str,
    interval: str = "5m",
    ema_length: int = 20
):
    df = fetch_minimal_data(ticker, interval)

    if df is None:
        return {"error": "Data fetch failed"}

    close = df["Close"]

    try:
        ema_value = calculate_ema(close, ema_length).iloc[-1]

        return {
            "ticker": ticker,
            "ema_length": ema_length,
            "interval": interval,
            "value": float(round(ema_value, 4))
        }

    except Exception:
        return {"error": "EMA calculation failed"}


# =========================
# OPTIONAL: INDIVIDUAL RSI
# =========================

@app.get("/rsi")
def get_rsi(
    ticker: str,
    interval: str = "5m",
    rsi_length: int = 14
):
    df = fetch_minimal_data(ticker, interval)

    if df is None:
        return {"error": "Data fetch failed"}

    close = df["Close"]

    try:
        rsi_value = calculate_rsi(close, rsi_length).iloc[-1]

        return {
            "ticker": ticker,
            "rsi_length": rsi_length,
            "interval": interval,
            "value": float(round(rsi_value, 2))
        }

    except Exception:
        return {"error": "RSI calculation failed"}
