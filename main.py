from fastapi import FastAPI, Query
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI(
    title="Technical Indicators API",
    description="API for RSI and EMA using yfinance",
    version="1.0.0"
)

# =========================
# Utility Functions (TOOLS)
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


def fetch_data(ticker: str, interval: str, period: str):
    df = yf.download(ticker, interval=interval, period=period)
    df.dropna(inplace=True)
    return df


# =========================
# API ENDPOINTS
# =========================

@app.get("/ema")
def get_ema(
    ticker: str = Query(..., description="Stock ticker (e.g., AAPL, RELIANCE.NS)"),
    interval: str = Query("1d", description="Data interval (1m, 5m, 1h, 1d)"),
    period: str = Query("1mo", description="Lookback period (1d, 5d, 1mo, etc.)"),
    ema_length: int = Query(20, description="EMA length")
):
    df = fetch_data(ticker, interval, period)

    df["EMA"] = calculate_ema(df["Close"], ema_length)

    latest_value = df["EMA"].iloc[-1]

    return {
        "ticker": ticker,
        "indicator": "EMA",
        "ema_length": ema_length,
        "interval": interval,
        "value": round(float(latest_value), 4)
    }


@app.get("/rsi")
def get_rsi(
    ticker: str = Query(..., description="Stock ticker"),
    interval: str = Query("1d"),
    period: str = Query("1mo"),
    rsi_length: int = Query(14)
):
    df = fetch_data(ticker, interval, period)

    df["RSI"] = calculate_rsi(df["Close"], rsi_length)

    latest_value = df["RSI"].iloc[-1]

    return {
        "ticker": ticker,
        "indicator": "RSI",
        "rsi_length": rsi_length,
        "interval": interval,
        "value": round(float(latest_value), 2)
    }
