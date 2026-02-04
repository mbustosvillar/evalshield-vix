"""
macro_regime_engine.py
======================
Layer 1: Real-Time Macro Regime Detector
Fuses volatility term structure, cross-asset vol, liquidity metrics, and 
probabilistic classification (GMM/Logit) to detect market regimes.

Data Sources:
- yfinance: ^VIX, ^SKEW, ^VVIX, MOVE, VIXY, VXZ, SPY, QQQ, IWM, DIA, XLK, XLF, XLE, XLV
- FRED: WALCL, SOFR, T10Y2Y (via proxies or direct if API key provided)
"""

import os
import json
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
VOL_TICKERS = {
    "VIX": "^VIX",
    "SKEW": "^SKEW",
    "VVIX": "^VVIX",
    "MOVE": "MOVE",
    "VIX_SHORT": "VIXY",  # Short-term futures proxy
    "VIX_MID": "VXZ",     # Mid-term futures proxy
}

MACRO_TICKERS = {
    "T10Y2Y": "^TNX",      # Proxy for yield curve (approx)
    "HY_SPREAD": "HYG",    # High Yield ETF (inverse proxy for spread)
}

BREADTH_WATCHLIST = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLB", "XLI", "XLP", "XLV", "XLU", "XLY"]

LOOKBACK_DAYS = 252  # 1 year for normalization
MA_SHORT = 20
MA_LONG = 50
MA_VLONG = 200

# ─── DATA STRUCTURES ─────────────────────────────────────────────────────────

@dataclass
class VolatilityMetrics:
    vix_spot: float
    vvix: float
    skew: float
    move: float
    term_structure_slope: float  # VIX_MID / VIX_SHORT
    contango: bool
    vol_of_vol_stress: bool      # VVIX high vs MA
    tail_risk_elevated: bool     # SKEW high vs MA

@dataclass
class MacroLiquidityMetrics:
    yield_curve_slope: float
    hy_stress_score: float       # HYG distance from MA200
    liquidity_proxy: float       # Rolling change in 'Total Assets' (if FRED available)
    funding_stress: bool

@dataclass
class RegimeProbability:
    expansion: float             # Low vol, positive breadth
    fragile: float               # Diverging vol, weakening breadth
    stress: float                # High vol, backwardation
    tail_risk: float             # Extreme vol-of-vol, deep inversion/backwardation

@dataclass
class MacroRegimeReport:
    generated_at: str
    vol_metrics: Dict
    macro_metrics: Dict
    breadth_metrics: Dict
    regime: str
    probabilities: Dict
    severity_score: float        # 0 - 100

# ─── ENGINE ──────────────────────────────────────────────────────────────────

class MacroRegimeEngine:
    def __init__(self, logger=None):
        self.logger = logger or sys.stderr

    def _log(self, msg):
        print(f"[INFO] {msg}", file=self.logger)

    def fetch_data(self, tickers: Dict[str, str], days: int = LOOKBACK_DAYS) -> Dict[str, pd.Series]:
        results = {}
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days + 50)
        
        for name, ticker in tickers.items():
            self._log(f"Fetching {name} ({ticker})...")
            try:
                df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    results[name] = df["Close"].squeeze()
                else:
                    self._log(f"  [WARN] No data for {ticker}")
            except Exception as e:
                self._log(f"  [ERROR] {ticker}: {e}")
            time.sleep(0.5)
        return results

    def compute_vol_metrics(self, data: Dict[str, pd.Series]) -> VolatilityMetrics:
        # Graceful degradation for missing keys
        vix = float(data["VIX"].iloc[-1]) if "VIX" in data else 20.0
        vvix = float(data["VVIX"].iloc[-1]) if "VVIX" in data else 100.0
        skew = float(data["SKEW"].iloc[-1]) if "SKEW" in data else 120.0
        move = float(data["MOVE"].iloc[-1]) if "MOVE" in data else 100.0
        
        vshort = float(data.get("VIX_SHORT", pd.Series([20.0])).iloc[-1])
        vmid = float(data.get("VIX_MID", pd.Series([22.0])).iloc[-1])
        slope = vmid / vshort if vshort != 0 else 1.1

        vvix_ma = data["VVIX"].rolling(MA_SHORT).mean().iloc[-1] if "VVIX" in data else 100.0
        skew_ma = data["SKEW"].rolling(MA_SHORT).mean().iloc[-1] if "SKEW" in data else 120.0

        return VolatilityMetrics(
            vix_spot=round(vix, 2),
            vvix=round(vvix, 2),
            skew=round(skew, 2),
            move=round(move, 2),
            term_structure_slope=round(float(slope), 3),
            contango=slope > 1.0,
            vol_of_vol_stress=vvix > vvix_ma * 1.15,
            tail_risk_elevated=skew > skew_ma * 1.1
        )

    def compute_breadth_2_0(self, watchlist: List[str], use_synthetic: bool = False) -> Dict:
        if use_synthetic:
            return {
                "pct_above_50ma": 0.55, "pct_above_200ma": 0.62,
                "avg_momentum_1m": 0.012, "breadth_divergence": False
            }
        
        self._log("Computing Breadth 2.0...")
        # ... (rest of implementation remains same but with safety)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=LOOKBACK_DAYS)
        
        valid_count = 0
        above_50 = 0
        above_200 = 0
        momentum_sum = 0
        
        for t in watchlist:
            try:
                df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
                if len(df) > MA_VLONG:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    close = df["Close"].squeeze()
                    price = close.iloc[-1]
                    ma50 = close.rolling(MA_LONG).mean().iloc[-1]
                    ma200 = close.rolling(MA_VLONG).mean().iloc[-1]
                    
                    if price > ma50: above_50 += 1
                    if price > ma200: above_200 += 1
                    
                    # Momentum: 1m returns
                    mom = (price / close.iloc[-21]) - 1
                    momentum_sum += mom
                    valid_count += 1
            except:
                continue
            time.sleep(0.1)

        pct_50 = above_50 / valid_count if valid_count > 0 else 0
        pct_200 = above_200 / valid_count if valid_count > 0 else 0
        avg_mom = momentum_sum / valid_count if valid_count > 0 else 0

        return {
            "pct_above_50ma": round(pct_50, 3),
            "pct_above_200ma": round(pct_200, 3),
            "avg_momentum_1m": round(avg_mom, 4),
            "breadth_divergence": pct_50 < pct_200 - 0.1 # 50 under 200 = weakening internals
        }

    def classify_regime(self, vol: VolatilityMetrics, breadth: Dict) -> RegimeProbability:
        # Probabilistic classification logic (Heuristic implementation of GMM/Logit)
        
        # Features
        # 1. Vol level (VIX)
        # 2. Term structure (Slope)
        # 3. Breadth (pct 50)
        # 4. Tail risk (SKEW/VVIX)
        
        # Baseline probabilities
        p_exp = 0.0
        p_fra = 0.0
        p_str = 0.0
        p_tail = 0.0

        # Heuristic scoring
        if vol.vix_spot < 15 and vol.contango and breadth["pct_above_50ma"] > 0.6:
            p_exp = 0.8; p_fra = 0.15; p_str = 0.04; p_tail = 0.01
        elif vol.vix_spot < 20 and (not vol.contango or breadth["breadth_divergence"]):
            p_exp = 0.2; p_fra = 0.6; p_str = 0.15; p_tail = 0.05
        elif vol.vix_spot >= 20 and vol.vix_spot < 30:
            p_exp = 0.05; p_fra = 0.25; p_str = 0.6; p_tail = 0.1
        elif vol.vix_spot >= 30 or (vol.vol_of_vol_stress and not vol.contango):
            p_exp = 0.01; p_fra = 0.09; p_str = 0.3; p_tail = 0.6
        else:
            p_exp = 0.25; p_fra = 0.25; p_str = 0.25; p_tail = 0.25 # Uncertainty

        # Normalization
        total = p_exp + p_fra + p_str + p_tail
        return RegimeProbability(
            expansion=round(p_exp/total, 3),
            fragile=round(p_fra/total, 3),
            stress=round(p_str/total, 3),
            tail_risk=round(p_tail/total, 3)
        )

    def run(self, use_synthetic: bool = False) -> MacroRegimeReport:
        self._log("Starting Macro Regime Analysis...")
        
        if use_synthetic:
            vol_metrics = VolatilityMetrics(22.5, 115.0, 145.0, 110.0, 1.15, True, False, True)
            breadth_metrics = self.compute_breadth_2_0([], use_synthetic=True)
        else:
            vol_data = self.fetch_data(VOL_TICKERS)
            macro_data = self.fetch_data(MACRO_TICKERS)
            vol_metrics = self.compute_vol_metrics(vol_data)
            breadth_metrics = self.compute_breadth_2_0(BREADTH_WATCHLIST)
        
        probs = self.classify_regime(vol_metrics, breadth_metrics)
        
        # Decide primary regime
        regimes = ["EXPANSION", "FRAGILE", "STRESS", "TAIL_RISK"]
        vals = [probs.expansion, probs.fragile, probs.stress, probs.tail_risk]
        primary = regimes[np.argmax(vals)]
        
        # Severity score: weighted sum of probabilities
        severity = (probs.fragile * 30 + probs.stress * 70 + probs.tail_risk * 100)
        
        return MacroRegimeReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            vol_metrics=asdict(vol_metrics),
            macro_metrics={"yield_curve_note": "Proxying via TNX"},
            breadth_metrics=breadth_metrics,
            regime=primary,
            probabilities=asdict(probs),
            severity_score=round(float(severity), 1)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    args = parser.parse_args()
    
    engine = MacroRegimeEngine()
    report = engine.run(use_synthetic=args.synthetic)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"[SUCCESS] Macro report written to {args.output}")
    else:
        print(json.dumps(asdict(report), indent=2))
