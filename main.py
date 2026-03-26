from fastapi import FastAPI, Query
import yfinance as yf
import pandas as pd

app = FastAPI(
    title="Technical Indicators API",
    description="Ultra-optimized API for RSI & EMA (minimal data, single call)",
    version="3.0.0"
)

# =========================
# CONFIG
# =========================
LOOKBACK_CANDLES = 220  # enough for EMA 200 + buffer

# =========================
# INDICATOR FUNCTIONS
# =========================
def calculate_ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# =========================
# MINIMAL DATA FETCH
# =========================
def fetch_minimal_data(ticker: str, interval: str):
    try:
        df = yf.download(
            tickers=ticker,
            interval=interval,
            period="2d",   # minimal safe fetch
            progress=False
        )

        if df.empty:
            return None

        # Keep only required column
        df = df[["Close"]]

        # Trim BEFORE calculations
        df = df.tail(LOOKBACK_CANDLES)

        # Remove index (prevents serialization overhead)
        df.reset_index(drop=True, inplace=True)

        return df

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
        return {"error": "Insufficient data or invalid ticker"}

    close = df["Close"]

    try:
        # ===== Indicators =====
        rsi = calculate_rsi(close, 14).iloc[-1]

        ema9 = calculate_ema(close, 9).iloc[-1]
        ema50 = calculate_ema(close, 50).iloc[-1]
        ema100 = calculate_ema(close, 100).iloc[-1]
        ema200 = calculate_ema(close, 200).iloc[-1]

        price = close.iloc[-1]

        # ===== Minimal Response =====
        return {
            "ticker": ticker,
            "price": float(round(price, 2)),
            "rsi": float(round(rsi, 2)),
            "ema9": float(round(ema9, 2)),
            "ema50": float(round(ema50, 2)),
            "ema100": float(round(ema100, 2)),
            "ema200": float(round(ema200, 2))
        }

    except Exception:
        return {"error": "Indicator calculation failed"}
