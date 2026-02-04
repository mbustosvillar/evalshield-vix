"""
data_collector.py
=================
Historical data factory for Layer 2 Transformer training.
Fetches 10-30 years of data for VIX, Term Structure, VVIX, SKEW, Yields, and Tickers.
Handles rate limits and merges data into a clean training CSV.
"""

import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
import sys
import argparse

# CONFIG
TICKERS = ["SPY", "^VIX", "^VIX3M", "^VVIX", "^SKEW", "^TNX", "^TYX"] 
START_DATE = "1994-01-01" # ~30 years
OUTPUT_FILE = "historical_training_data.csv"

def fetch_with_retry(ticker, start, end):
    for attempt in range(5):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df[["Close"]]
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            print(f"[WARN] Failed {ticker} attempt {attempt+1}: {e}", file=sys.stderr)
            time.sleep(15)
    return None

def generate_mock_history(days=7500):
    """Generates 30 years of synthetic data with realistic correlations."""
    print(f"[INFO] Generating {days} days of synthetic historical data...")
    end_date = datetime.now(timezone.utc)
    dates = pd.date_range(end=end_date, periods=days, freq="B")
    
    np.random.seed(42)
    # Latent "Panic" factor (random walk)
    panic = np.cumsum(np.random.randn(days) * 0.1)
    
    data = {
        "spy": 100 * np.exp(np.cumsum(np.random.randn(days) * 0.01 + 0.0002) - 0.2 * panic),
        "vix": 15 + 5 * panic + np.random.randn(days) * 2,
        "vix3m": 18 + 4 * panic + np.random.randn(days) * 1.5,
        "vvix": 90 + 10 * panic + np.random.randn(days) * 5,
        "skew": 120 + 5 * panic + np.random.randn(days) * 3,
        "tnx": 3.5 - 0.1 * panic + np.random.randn(days) * 0.1,
        "tyx": 4.0 - 0.05 * panic + np.random.randn(days) * 0.1
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Clip VIX and other non-negative indicators
    df["vix"] = df["vix"].clip(lower=9)
    df["vix3m"] = df["vix3m"].clip(lower=10)
    df["vvix"] = df["vvix"].clip(lower=60)
    
    df["vix_term_structure"] = df["vix"] / df["vix3m"]
    
    # Labeling
    df["fwd_ret_21d"] = df["spy"].shift(-21) / df["spy"] - 1
    # Crash label: drawdown in next 30 days
    df["is_crash_30d"] = (df["spy"].rolling(30).min().shift(-30) / df["spy"] - 1 < -0.05).astype(int)
    
    return df.dropna()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock:
        master_df = generate_mock_history()
    else:
        print(f"[INFO] Initializing historical data collection from {START_DATE}...")
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        master_df = pd.DataFrame()

        for ticker in TICKERS:
            print(f"[INFO] Fetching {ticker}...")
            df = fetch_with_retry(ticker, START_DATE, end_date)
            if df is not None:
                df.columns = [ticker.replace("^", "").lower()]
                if master_df.empty:
                    master_df = df
                else:
                    master_df = master_df.join(df, how="outer")
            else:
                print(f"[ERROR] Could not fetch {ticker} after multiple retries.")
            time.sleep(5)

        print("[INFO] Engineering basic historical features...")
        if "vix" in master_df.columns and "vix3m" in master_df.columns:
            master_df["vix_term_structure"] = master_df["vix"] / master_df["vix3m"]
        
        if "spy" in master_df.columns:
            master_df["fwd_ret_21d"] = master_df["spy"].shift(-21) / master_df["spy"] - 1
            master_df["is_crash_30d"] = (master_df["spy"].rolling(30).min().shift(-30) / master_df["spy"] - 1 < -0.05).astype(int)
        
        master_df.dropna(subset=["spy"], inplace=True)

    master_df.to_csv(OUTPUT_FILE)
    print(f"[SUCCESS] Historical data saved to {OUTPUT_FILE} ({len(master_df)} rows)")

if __name__ == "__main__":
    main()
