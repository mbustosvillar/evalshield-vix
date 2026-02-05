import os
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class MarketDataProvider(ABC):
    @abstractmethod
    def get_latest_context(self) -> dict:
        pass

class MockMarketDataProvider(MarketDataProvider):
    """Provides synthetic data for testing/offline mode."""
    def get_latest_context(self) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "spy_price": 500.0,
            "vix_current": 18.5,
            "vix_3m": 20.0,
            "vvix_current": 95.0,
            "skew_current": 125.0,
            "vix_term_structure": 0.925,  # VIX/VIX3M
            "yield_curve_spread": 0.15,
            "portfolio": {
                "GLD": {"price": 180.0, "change_pct": -1.18},
                "SLV": {"price": 22.0, "change_pct": -12.19},
                "URA": {"price": 28.0, "change_pct": -4.60},
                "COPX": {"price": 35.0, "change_pct": -4.87},
                "LMT": {"price": 450.0, "change_pct": 1.43},
                "VST": {"price": 50.0, "change_pct": -1.00},
                "TEM": {"price": 15.0, "change_pct": -0.86}
            },
            "is_mock": True
        }

class LiveMarketDataProvider(MarketDataProvider):
    """Fetches real-time market data via Yahoo Finance with robust fallbacks."""
    
    def get_latest_context(self) -> dict:
        try:
            # Macro + Portfolio Tickers
            macro_tickers = ["SPY", "^VIX", "VIX3M", "^VVIX", "^SKEW", "^TNX"]
            portfolio_tickers = ["GLD", "SLV", "URA", "COPX", "LMT", "VST", "TEM"]
            all_tickers = macro_tickers + portfolio_tickers
            
            data = yf.download(all_tickers, period="2d", interval="1d", progress=False)
            
            if data.empty:
                raise ValueError("YFinance returned empty data (Rate limit?)")

            # Latest Close
            latest = data['Close'].iloc[-1]
            
            # Calculate daily change for portfolio
            prev_close = data['Close'].iloc[-2]
            portfolio_data = {}
            for ticker in portfolio_tickers:
                price = float(latest[ticker])
                prev = float(prev_close[ticker])
                change = ((price - prev) / prev) * 100
                portfolio_data[ticker] = {"price": round(price, 2), "change_pct": round(change, 2)}

            context = {
                "timestamp": datetime.now().isoformat(),
                "spy_price": float(latest['SPY']),
                "vix_current": float(latest['^VIX']),
                "vix_3m": float(latest['VIX3M']),
                "vvix_current": float(latest['^VVIX']),
                "skew_current": float(latest['^SKEW']),
                "vix_term_structure": float(latest['^VIX'] / latest['VIX3M']),
                "yield_curve_spread": float(latest['^TNX'] / 100.0), # Simplified proxy
                "portfolio": portfolio_data,
                "is_mock": False,
                "provider": "yfinance_live"
            }
            return context

        except Exception as e:
            print(f"[WARN] Live market data failed: {e}. Using MOCK data.")
            return MockMarketDataProvider().get_latest_context()

class AlphaVantageDataProvider(MarketDataProvider):
    """Fetches high-reliability data via Alpha Vantage (Real-time)."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")

    def get_latest_context(self) -> dict:
        if not self.api_key:
            return LiveMarketDataProvider().get_latest_context()
            
        try:
            import requests
            base_url = "https://www.alphavantage.co/query"
            
            # Fetch SPY
            resp = requests.get(base_url, params={
                "function": "GLOBAL_QUOTE",
                "symbol": "SPY",
                "apikey": self.api_key
            }).json()
            
            spy_price = float(resp.get("Global Quote", {}).get("05. price", 500.0))
            
            # Hybridize: Alpha Vantage for SPY, yfinance for the rest
            context = LiveMarketDataProvider().get_latest_context()
            context['spy_price'] = spy_price
            context['provider'] = "alpha_vantage_hybrid"
            return context

        except Exception as e:
            print(f"[WARN] Alpha Vantage failed: {e}. Fallback to yfinance.")
            return LiveMarketDataProvider().get_latest_context()

def get_provider(mode: str = "auto") -> MarketDataProvider:
    if mode == "live":
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            return AlphaVantageDataProvider()
        return LiveMarketDataProvider()
    return MockMarketDataProvider()
