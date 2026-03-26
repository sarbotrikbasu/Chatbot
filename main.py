from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from typing import Optional

app = FastAPI(
    title="Stock Indicators API",
    description="API to calculate the latest RSI and EMA for any stock ticker, designed for ChatGPT Custom Actions.",
    version="1.0.0",
   
    servers=[{"url": "https://chatbot-beryl-one-50.vercel.app/"}]
)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # Wilder's moving average
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

class IndicatorResponse(BaseModel):
    symbol: str
    indicator: str
    period: int
    latest_value: float

@app.get("/rsi", response_model=IndicatorResponse, summary="Calculate RSI", description="Calculates the Relative Strength Index for a given stock ticker.")
def get_rsi(
    symbol: str = Query(..., description="Stock ticker symbol (e.g., AAPL, MSFT, ^NSEI)"),
    period: int = Query(14, description="Period for RSI calculation (default: 14)")
):
    try:
        stock = yf.Ticker(symbol)
        # Fetch enough historical data to calculate RSI
        hist = stock.history(period="1y") 
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Ticker '{symbol}' not found or no data available.")
        
        prices = hist['Close']
        if len(prices) < period * 2:
            raise HTTPException(status_code=400, detail="Not enough data to calculate reliable RSI.")
            
        rsi_series = calculate_rsi(prices, period)
        latest_rsi = rsi_series.iloc[-1]
        
        if pd.isna(latest_rsi):
            raise HTTPException(status_code=500, detail="Failed to calculate RSI value.")
            
        return {
            "symbol": symbol,
            "indicator": "RSI",
            "period": period,
            "latest_value": round(float(latest_rsi), 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ema", response_model=IndicatorResponse, summary="Calculate EMA", description="Calculates the Exponential Moving Average for a given stock ticker.")
def get_ema(
    symbol: str = Query(..., description="Stock ticker symbol (e.g., AAPL)"),
    period: int = Query(20, description="Period for EMA calculation (default: 20)")
):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Ticker '{symbol}' not found or no data available.")
            
        prices = hist['Close']
        if len(prices) < period:
            raise HTTPException(status_code=400, detail="Not enough data to calculate EMA.")
            
        ema_series = prices.ewm(span=period, adjust=False).mean()
        latest_ema = ema_series.iloc[-1]
        
        if pd.isna(latest_ema):
            raise HTTPException(status_code=500, detail="Failed to calculate EMA value.")
            
        return {
            "symbol": symbol,
            "indicator": "EMA",
            "period": period,
            "latest_value": round(float(latest_ema), 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", summary="Health Check")
def health_check():
    return {"status": "ok"}
