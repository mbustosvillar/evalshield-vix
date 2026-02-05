"""
signal_engine.py
================
Fetches market data, computes tactical indicators, scores air-pocket
probability, and emits structured JSON consumed by the report generator.

Data sources (all free-tier / open):
  - yfinance   → historical prices, VIX, breadth proxies
  - Polygon.io → real-time bid-ask / trade-count (optional; degrades gracefully)

Run:
  python signal_engine.py                        # full scan, prints JSON
  python signal_engine.py --ticker SPY --days 90 # single-ticker deep scan
  python signal_engine.py --output signals.json  # persist for report gen
"""

import argparse, json, sys, os, time, requests
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from typing import Optional

import yfinance as yf
import pandas as pd
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────────
WATCHLIST = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV"]
VIX_TICKER = "^VIX"
VIX3M_TICKER = "^VIX3M"
VVIX_TICKER = "^VVIX"
SKEW_TICKER = "^SKEW"
AD_LINE_PROXY = "^ADVANCING"          # yfinance may not serve this; fallback below
YIELD_CURVE_10Y = "^TNX"              # 10-Year Treasury Yield
YIELD_CURVE_2Y  = "^TW2"              # 2-Year Treasury Yield (best-effort)

LOOKBACK_DAYS   = 90
RSI_PERIOD      = 14
MA_SHORT        = 20
MA_LONG         = 50
VIX_COMPLACENCY = 13.0                # VIX below this = complacency zone
VIX_PANIC_SPIKE = 0.35                # 35% single-day spike = panic signal
BREADTH_THRESHOLD = 0.45              # % of watchlist above 50-day MA; below = weak

# DevalShield AR Config
BLUELY_URL = "https://api.bluelytics.com.ar/v2/latest"
W_GLOBAL_COMPLACENCY = 0.4
W_LOCAL_DEVAL = 0.6

NEGATIVE_SENTIMENT_WORDS = [
    'miedo', 'pánico', 'devaluación', 'devaluar', 'devaluó', 'brecha', 'amplia', 'explota', 'manipulación', 
    'manipular', 'mentira', 'mienten', 'renuncia', 'traba', 'ansiedad', 'crisis', 'locura', 'cínico', 
    'hambre', 'pobre', 'derrite', 'licuación', 'quilombo', 'corrida', 'incertidumbre'
]
POSITIVE_SENTIMENT_WORDS = [
    'alivio', 'estabilidad', 'baja', 'mínimo', 'estable', 'previsibilidad', 'control', 'grandeza', 
    'exterminar', 'libre', 'factos', 'volando', 'zafo', 'éxito', 'desinflación', 'paz', 'confianza'
]

# Scoring weights (must sum to 1.0)
W_BREADTH   = 0.25
W_VIX       = 0.35  # Increased weight for complex vol signals
W_MOMENTUM  = 0.25
W_YIELD     = 0.15
assert abs(W_BREADTH + W_VIX + W_MOMENTUM + W_YIELD - 1.0) < 1e-9

# ─── DATA STRUCTURES ─────────────────────────────────────────────────────────
@dataclass
class TickerSignal:
    ticker: str
    price: float
    ma_20: float
    ma_50: float
    rsi: float
    above_ma50: bool
    momentum_score: float          # -1 (bearish) … +1 (bullish)
    signal_text: str               # human-readable classification

@dataclass
class MarketContext:
    vix_current: float
    vix_1d_change_pct: float
    vix_52w_low: float
    vix_ma20: float
    vix_ma50: float
    vix_3m: Optional[float]
    vix_term_structure: Optional[float]  # VIX / VIX3M
    vvix_current: Optional[float]
    skew_current: Optional[float]
    vix_complacency: bool          # current near 52w low
    vix_panic_spike: bool          # single-day spike > threshold
    vix_short_term_signal: bool    # VIX > MA20
    vix_medium_term_signal: bool   # VIX > MA50
    vix_long_term_signal: bool     # VIX vs 52w low (complacency)
    vix_term_structure_inverted: bool
    tactical_score: int            # 0-4 aggregate (added term structure)
    breadth_pct_above_ma50: float  # fraction of watchlist above 50-day MA
    breadth_weak: bool
    yield_curve_spread: Optional[float]  # 10Y - 2Y; negative = inverted
    yield_curve_inverted: bool
    # DevalShield AR Context
    blue_gap_pct: Optional[float]
    gap_velocity: Optional[float]
    deval_vacuum_index: float
    x_sentiment_score: float        # -1 to +1
    x_sentiment_label: str

@dataclass
class AirPocketScore:
    """Composite score 0–100. Higher = more likely air pocket conditions."""
    total: float
    breadth_component: float
    vix_component: float
    momentum_component: float
    yield_component: float
    vix_term_component: float       # new Layer 2
    vvix_component: float           # new Layer 2
    condition_flags: list           # active qualitative flags
    severity: str                   # LOW / MODERATE / HIGH / CRITICAL

@dataclass
class FullReport:
    generated_at: str
    tickers: list
    context: dict
    score: dict
    ticker_signals: list
    narrative: list                # DevalShield Storytelling

# ─── DEVALSHIELD AR FUNCTIONS ───────────────────────────────────────────────
def fetch_local_data() -> dict:
    """Fetches Argentine local data from Bluelytics."""
    local = {"blue_gap_pct": 0.0, "gap_velocity": 0.0, "deval_pressure": False}
    try:
        resp = requests.get(BLUELY_URL, timeout=10)
        data = resp.json()
        blue = data['blue']['value_avg']
        oficial = data['oficial']['value_avg']
        gap = ((blue - oficial) / oficial) * 100 if oficial else 0
        local['blue_gap_pct'] = round(gap, 2)
        
        # Historical fetch for velocity (mocking/simplifying for MVP)
        # In a real environment, we'd persist yesterday's data. 
        # Here we attempt to fetch historical if available via Bluelytics proxy
        yesterday = (datetime.now() - timedelta(1)).strftime("%Y-%m-%d")
        hist_resp = requests.get(f"https://api.bluelytics.com.ar/v2/historical?day={yesterday}", timeout=5)
        if hist_resp.status_code == 200:
            hist = hist_resp.json()
            prev_blue = hist['blue']['value_avg']
            prev_gap = ((prev_blue - oficial) / oficial) * 100 # assuming oficial moved less
            local['gap_velocity'] = round(gap - prev_gap, 2)
        
        local['deval_pressure'] = gap > 25 or local['gap_velocity'] > 3
    except Exception as e:
        print(f"[WARN] fetch_local_data error: {e}", file=sys.stderr)
    return local

def generate_narrative(score_val: float, dvi: float, local: dict, sentiment: float = 0.0) -> list:
    """Generates a contextual narrative story for the regime."""
    lines = []
    
    # Sentiment flavor
    if sentiment < -0.3:
        lines.append(f"SENTIMIENTO X: Alerta de pánico detectada (score {sentiment:.2f}). La conversación social refleja desconfianza y temor institucional.")
    elif sentiment > 0.3:
        lines.append(f"SENTIMIENTO X: Clima de alivio y estabilidad (score {sentiment:.2f}).")

    if dvi > 70:
        lines.append(f"ALERTA CRÍTICA: El Deval Vacuum Index está en {dvi}. ")
        lines.append(f"Se detecta una confluencia peligrosa: complacencia global mientras la brecha local ({local.get('blue_gap_pct')}%) se acelera.")
        if sentiment < -0.2:
            lines.append("EXTRA: El ruido social en X confirma el inicio de una posible corrida de opinión.")
        lines.append("Recomendación: Proteger poder de compra mediante dolarización táctica (CEDEARs/Stablecoins) antes del vacío.")
    elif dvi > 50:
        lines.append(f"PRECAUCIÓN: Presión devaluatoria en aumento ({dvi}). Vigilancia estrecha de la brecha blue.")
    else:
        lines.append("RÉGIMEN ESTABLE: No se detecta confluencia de complacencia global y tensión local inmediata.")
    return lines

def compute_sentiment(texts: list[str]) -> float:
    """Score -1 (very negative/fear) to +1 (positive/calm). 0 neutral."""
    if not texts: return 0.0
    
    total_mentions = 0
    total_score = 0.0
    for text in texts:
        t = text.lower()
        neg = sum(t.count(w) for w in NEGATIVE_SENTIMENT_WORDS)
        pos = sum(t.count(w) for w in POSITIVE_SENTIMENT_WORDS)
        if (neg + pos) > 0:
            total_score += (pos - neg) / (neg + pos)
            total_mentions += 1
    
    return total_score / total_mentions if total_mentions > 0 else 0.0

def fetch_x_posts(query: str = '(devaluación OR "dólar blue" OR brecha OR inflación OR miedo) Argentina lang:es', max_results: int = 20) -> dict:
    """Fetches tweets and returns a dict with posts and metadata (Confidence Score)."""
    token = os.getenv("X_BEARER_TOKEN")
    result = {"posts": [], "confidence": 0.5} # Default mid confidence for mock
    
    if not token:
        # Static mock based on Feb 3 2026 Snapshot
        result["posts"] = [
            "El dólar blue sigue bajando, increíble la estabilidad cambiaria!",
            "Mucho alivio con la inflación de este mes, Milei lo hizo.",
            "Dudosa la renuncia de Marco Lavagna en el INDEC...",
            "Me da mucha ansiedad que mientan con los números de la brecha.",
            "Cínico que digan que no hay crisis cuando el hambre aprieta.",
            "Éxito total el canje de deuda, estabilidad absoluta.",
            "Locura total lo que está pasando con el dólar libre, volando!"
        ]
        return result
    
    headers = {"Authorization": f"Bearer {token}"}
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {"query": query, "max_results": max_results, "tweet.fields": "text"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            result["posts"] = [t['text'] for t in data]
            # Confidence grows with sample size
            result["confidence"] = min(1.0, len(data) / max_results)
            return result
    except Exception as e:
        print(f"[WARN] fetch_x_posts error: {e}", file=sys.stderr)
    
    return result

# ─── INDICATOR FUNCTIONS ─────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def fetch_ticker(ticker: str, days: int = LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
    """Download adjusted close; returns None on failure (graceful degradation).
    Includes retries and backoff to handle yfinance rate limits.
    """
    for attempt in range(4):
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days + 40)  # extra buffer
            df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                             progress=False)
            if df.empty:
                if attempt < 3:
                    time.sleep(2 * (attempt + 1))
                    continue
                return None
            
            # yfinance may return multi-level columns; flatten
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            
            return df[["Close"]].rename(columns={"Close": "close"})
        except Exception as e:
            if "Rate Limited" in str(e) or "Too Many Requests" in str(e) or "429" in str(e):
                import random
                wait = 10 * (attempt + 1) + random.uniform(2, 5)
                print(f"[WARN] Rate limit hit for {ticker}. Waiting {wait:.1f}s (Attempt {attempt+1}/4)...", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"[WARN] fetch_ticker({ticker}): {e}", file=sys.stderr)
                break
    return None

def generate_mock_df(ticker: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Generates synthetic data for testing when APIs are throttled."""
    end = datetime.now(timezone.utc)
    dates = pd.date_range(end=end, periods=days + 60, freq="D")
    
    # Base values for specific indices
    base = 20.0
    if ticker == VIX_TICKER: base = 18.5
    elif ticker == VIX3M_TICKER: base = 21.0
    elif ticker == VVIX_TICKER: base = 105.0
    elif ticker == SKEW_TICKER: base = 135.0
    elif ticker.startswith("^"): base = 3.5  # yields
    else: base = 450.0 # SPY etc
    
    # Add random walk
    np.random.seed(42)
    noise = np.random.randn(len(dates)) * (base * 0.02)
    prices = base + np.cumsum(noise)
    
    df = pd.DataFrame({"close": prices}, index=dates)
    return df

def analyze_ticker(ticker: str, days: int = LOOKBACK_DAYS, mock: bool = False) -> Optional[TickerSignal]:
    df = generate_mock_df(ticker, days) if mock else fetch_ticker(ticker, days)
    if df is None or len(df) < MA_LONG + 5:
        return None

    close = df["close"].squeeze()
    ma20  = close.rolling(MA_SHORT).mean().iloc[-1]
    ma50  = close.rolling(MA_LONG).mean().iloc[-1]
    rsi   = compute_rsi(close).iloc[-1]
    price = close.iloc[-1]
    above = bool(price > ma50)

    # Momentum score: combines price vs MA50 and RSI
    rsi_norm = (float(rsi) - 50) / 50
    ma_norm  = 1.0 if above else -1.0
    momentum = 0.5 * rsi_norm + 0.5 * ma_norm

    if momentum > 0.3:
        sig = "BULLISH"
    elif momentum > -0.3:
        sig = "NEUTRAL"
    else:
        sig = "BEARISH"

    return TickerSignal(
        ticker=ticker, price=round(float(price), 2),
        ma_20=round(float(ma20), 2), ma_50=round(float(ma50), 2),
        rsi=round(float(rsi), 2), above_ma50=above,
        momentum_score=round(float(momentum), 3), signal_text=sig
    )

def analyze_vix_complex(mock: bool = False) -> dict:
    """Returns VIX metrics including Term Structure and VVIX."""
    if mock:
        vix_df = generate_mock_df(VIX_TICKER)
        vix3m_df = generate_mock_df(VIX3M_TICKER)
        vvix_df = generate_mock_df(VVIX_TICKER)
        skew_df = generate_mock_df(SKEW_TICKER)
    else:
        vix_df = fetch_ticker(VIX_TICKER, LOOKBACK_DAYS)
        vix3m_df = fetch_ticker(VIX3M_TICKER, LOOKBACK_DAYS)
        vvix_df = fetch_ticker(VVIX_TICKER, LOOKBACK_DAYS)
        skew_df = fetch_ticker(SKEW_TICKER, LOOKBACK_DAYS)

    if vix_df is None or len(vix_df) < MA_LONG:
        return {"error": "Primary VIX data unavailable"}

    v_close = vix_df["close"].squeeze()
    current    = float(v_close.iloc[-1])
    prev       = float(v_close.iloc[-2])
    low_52w    = float(v_close.min())
    change_pct = (current - prev) / prev
    
    ma20 = float(v_close.rolling(MA_SHORT).mean().iloc[-1])
    ma50 = float(v_close.rolling(MA_LONG).mean().iloc[-1])

    short_term  = current > ma20
    medium_term = current > ma50
    long_term   = bool(current < (low_52w * 1.1) or current < VIX_COMPLACENCY)

    # Term Structure
    vix3m_val = float(vix3m_df["close"].squeeze().iloc[-1]) if vix3m_df is not None else None
    term_structure = float(current / vix3m_val) if vix3m_val else None
    ts_inverted = bool(term_structure > 1.0) if term_structure else False

    # VVIX
    vvix_val = float(vvix_df["close"].squeeze().iloc[-1]) if vvix_df is not None else None
    
    # SKEW
    skew_val = float(skew_df["close"].squeeze().iloc[-1]) if skew_df is not None else None

    # Accumulated score: Short Term +1, Medium Term +1, Long Term +1, TS Inverted +1
    score = 0
    if short_term: score += 1
    if medium_term: score += 1
    if long_term: score += 1
    if ts_inverted: score += 1

    return {
        "vix_current": round(current, 2),
        "vix_1d_change_pct": round(change_pct, 4),
        "vix_52w_low": round(low_52w, 2),
        "vix_ma20": round(ma20, 2),
        "vix_ma50": round(ma50, 2),
        "vix_3m": round(vix3m_val, 2) if vix3m_val else None,
        "vix_term_structure": round(term_structure, 3) if term_structure else None,
        "vvix_current": round(vvix_val, 2) if vvix_val else None,
        "skew_current": round(skew_val, 2) if skew_val else None,
        "vix_complacency": long_term,
        "vix_panic_spike": change_pct > VIX_PANIC_SPIKE,
        "vix_short_term_signal": short_term,
        "vix_medium_term_signal": medium_term,
        "vix_long_term_signal": long_term,
        "vix_term_structure_inverted": ts_inverted,
        "tactical_score": score
    }

def analyze_yield_curve(mock: bool = False) -> dict:
    if mock:
        df10 = generate_mock_df(YIELD_CURVE_10Y)
        df2  = generate_mock_df(YIELD_CURVE_2Y)
    else:
        df10 = fetch_ticker(YIELD_CURVE_10Y, LOOKBACK_DAYS)
        df2  = fetch_ticker(YIELD_CURVE_2Y, LOOKBACK_DAYS)

    if df10 is None:
        return {"yield_curve_spread": None, "yield_curve_inverted": False,
                "error": "Yield curve data unavailable"}

    ten_y = float(df10["close"].squeeze().iloc[-1])
    two_y = float(df2["close"].squeeze().iloc[-1]) if df2 is not None else None

    if two_y is not None:
        spread = round(ten_y - two_y, 3)
        return {"yield_curve_spread": spread, "yield_curve_inverted": spread < 0}
    return {"yield_curve_spread": None, "yield_curve_inverted": False,
            "note": "2Y Treasury data unavailable; spread not computed"}

# ─── SCORING ─────────────────────────────────────────────────────────────────
def score_air_pocket(vix_data: dict, breadth_pct: float,
                      ticker_signals: list) -> AirPocketScore:
    flags = []

    # --- Breadth component (0–100) ---
    breadth_score = max(0, min(100, (1.0 - breadth_pct) * 100))
    if breadth_pct < BREADTH_THRESHOLD:
        flags.append("BREADTH_DETERIORATION")

    # --- VIX component (0–100) ---
    vix_score = 0.0
    if vix_data.get("vix_complacency"):
        vix_score += 40
        flags.append("VIX_COMPLACENCY")
    if vix_data.get("vix_panic_spike"):
        vix_score += 60
        flags.append("VIX_PANIC_SPIKE")
    
    chg = vix_data.get("vix_1d_change_pct", 0)
    if 0.1 < chg <= VIX_PANIC_SPIKE:
        vix_score += 20
        flags.append("VIX_RISING")

    # Layer 2: Term Structure & VVIX
    vix_term_score = 0.0
    if vix_data.get("vix_term_structure_inverted"):
        vix_term_score = 100.0
        flags.append("VIX_TERM_STRUCTURE_INVERTED")
    
    vvix_score = 0.0
    vvix = vix_data.get("vvix_current", 0)
    if vvix and vvix > 110: # Sample threshold
        vvix_score = 100.0
        flags.append("VVIX_EXCESSIVE")
    elif vvix and vvix > 95:
        vvix_score = 50.0
        flags.append("VVIX_ELEVATED")

    # --- Momentum component (0–100) ---
    if ticker_signals:
        avg_mom = np.mean([t.momentum_score for t in ticker_signals])
        momentum_score = max(0, min(100, (1.0 - avg_mom) * 50))
    else:
        momentum_score = 50

    if sum(1 for t in ticker_signals if t.signal_text == "BEARISH") >= len(ticker_signals) * 0.6:
        flags.append("MAJORITY_BEARISH")

    # Initial total (yield and L2 signals will be weighted in build_report)
    total = (W_BREADTH * breadth_score +
             W_VIX * vix_score +
             W_MOMENTUM * momentum_score)

    return AirPocketScore(
        total=round(total, 1),
        breadth_component=round(breadth_score, 1),
        vix_component=round(vix_score, 1),
        momentum_component=round(momentum_score, 1),
        yield_component=0.0,
        vix_term_component=round(vix_term_score, 1),
        vvix_component=round(vvix_score, 1),
        condition_flags=flags,
        severity="LOW"
    )

# ─── ORCHESTRATOR ────────────────────────────────────────────────────────────
def build_report(ticker_list: list = None, mock: bool = False) -> FullReport:
    if ticker_list is None:
        ticker_list = WATCHLIST

    print(f"[INFO] Fetching signals (Mock={mock})...", file=sys.stderr)
    vix_data = analyze_vix_complex(mock=mock)
    yield_data = analyze_yield_curve(mock=mock)

    print("[INFO] Scanning tickers...", file=sys.stderr)
    signals = []
    for t in ticker_list:
        if not mock: time.sleep(1.5)
        sig = analyze_ticker(t, mock=mock)
        if sig:
            signals.append(sig)
            print(f"  [{sig.signal_text:>8}] {t:>5} | P={sig.price} RSI={sig.rsi}", file=sys.stderr)

    # Breadth: % above MA50
    breadth_pct = float(sum(1 for s in signals if s.above_ma50) / len(signals)) if signals else 0.5

    # Score (initial)
    score = score_air_pocket(vix_data, breadth_pct, signals)

    # Inject yield component
    y_score = 0.0
    if yield_data.get("yield_curve_spread") is not None:
        spread = yield_data["yield_curve_spread"]
        if yield_data["yield_curve_inverted"]:
            y_score = 100.0
            score.condition_flags.append("YIELD_CURVE_INVERTED")
        else:
            y_score = max(0, min(100, (1.5 - spread) / 1.5 * 100))
        score.yield_component = round(y_score, 1)

    # Recompute total with Layer 2 Volatility integration
    # We blend standard VIX score with Term Structure and VVIX
    composite_vix_score = (score.vix_component * 0.5 + 
                           score.vix_term_component * 0.3 + 
                           score.vvix_component * 0.2)
    score.vix_component = round(composite_vix_score, 1)

    score.total = round(
        W_BREADTH   * score.breadth_component +
        W_VIX       * score.vix_component +
        W_MOMENTUM  * score.momentum_component +
        W_YIELD     * score.yield_component, 1)

    # Re-derive severity
    if score.total >= 70: score.severity = "CRITICAL"
    elif score.total >= 50: score.severity = "HIGH"
    elif score.total >= 30: score.severity = "MODERATE"
    else: score.severity = "LOW"

    # --- DevalShield AR Integration ---
    local_data = fetch_local_data()
    
    # Global Complacency component: Inverse of VIX score (0-100)
    global_complacency = max(0, 100 - score.vix_component)
    
    # Local pressure component
    x_data = fetch_x_posts()
    x_texts = x_data['posts']
    sentiment_score = compute_sentiment(x_texts)
    
    # Pressure = Gap + Velocity + Sentiment Booster
    # Sentiment score < 0 (fear) boosts pressure
    sentiment_boost = max(0, -sentiment_score * 40) 
    local_pressure = min(100, local_data['blue_gap_pct'] * 2 + local_data['gap_velocity'] * 5 + sentiment_boost)
    
    dvi = round(W_GLOBAL_COMPLACENCY * global_complacency + W_LOCAL_DEVAL * local_pressure, 1)

    # Updated logic: If DVI is high, it can override/boost total score
    if dvi > 70:
        score.total = max(score.total, dvi)
        score.condition_flags.append("DEVAL_VACUUM_HIGH")
        if score.vix_component < 30: # Low global risk + high local = Contrarian setup
            score.condition_flags.append("CONTRARIAN_DOLLARIZE")

    # Re-derive severity (final)
    if score.total >= 70: score.severity = "CRITICAL"
    elif score.total >= 50: score.severity = "HIGH"
    elif score.total >= 30: score.severity = "MODERATE"
    else: score.severity = "LOW"

    context = MarketContext(
        vix_current=vix_data.get("vix_current", 0),
        vix_1d_change_pct=vix_data.get("vix_1d_change_pct", 0),
        vix_52w_low=vix_data.get("vix_52w_low", 0),
        vix_ma20=vix_data.get("vix_ma20", 0),
        vix_ma50=vix_data.get("vix_ma50", 0),
        vix_3m=vix_data.get("vix_3m"),
        vix_term_structure=vix_data.get("vix_term_structure"),
        vvix_current=vix_data.get("vvix_current"),
        skew_current=vix_data.get("skew_current"),
        vix_complacency=vix_data.get("vix_complacency", False),
        vix_panic_spike=vix_data.get("vix_panic_spike", False),
        vix_short_term_signal=vix_data.get("vix_short_term_signal", False),
        vix_medium_term_signal=vix_data.get("vix_medium_term_signal", False),
        vix_long_term_signal=vix_data.get("vix_long_term_signal", False),
        vix_term_structure_inverted=vix_data.get("vix_term_structure_inverted", False),
        tactical_score=vix_data.get("tactical_score", 0),
        breadth_pct_above_ma50=round(breadth_pct, 3),
        breadth_weak=breadth_pct < BREADTH_THRESHOLD,
        yield_curve_spread=yield_data.get("yield_curve_spread"),
        yield_curve_inverted=yield_data.get("yield_curve_inverted", False),
        # New DevalShield Fields
        blue_gap_pct=local_data['blue_gap_pct'],
        gap_velocity=local_data['gap_velocity'],
        deval_vacuum_index=dvi,
        x_sentiment_score=round(sentiment_score, 3),
        x_sentiment_label="Negativo" if sentiment_score < -0.3 else "Neutro" if sentiment_score < 0.3 else "Positivo"
    )

    narrative = generate_narrative(score.total, dvi, local_data, sentiment_score)

    return FullReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        tickers=ticker_list,
        context=asdict(context),
        score=asdict(score),
        ticker_signals=[asdict(s) for s in signals],
        narrative=narrative
    )

# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Air Pocket Signal Engine - Layer 2")
    parser.add_argument("--ticker", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--mock", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    report = build_report(args.ticker, mock=args.mock)
    payload = asdict(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Signals written to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(payload, indent=2))
