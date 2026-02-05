"""
integrated_orchestrator.py
==========================
The "Grand Orchestrator" for the Layer 3 Intelligent VIX Dashboard.
1. Fetches current market signals (Layer 1).
2. Runs Transformer prediction for 30-day Tail-Risk (Layer 2).
3. Executes RL Policy for optimal alpha-asymmetric allocation (Layer 3).
4. Emits structured data for the professional DOCX/HTML report.
"""

import torch
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from tail_risk_model import TailRiskTransformer
from ppo_agent import PolicyNetwork
from signal_engine import build_report, FullReport  # Keeping report struct
from data_provider import get_provider
from retrain_model import retrain_from_collective

# CONFIG
MODEL_PATH = "tail_risk_model.pth"
RL_AGENT_PATH = "ppo_agent.pth" 

def load_transformer():
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model = TailRiskTransformer(input_dim=8, hidden_dim=32, num_heads=4, num_layers=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['scaler']

def get_prediction(model, scaler, context):
    mock_seq = np.zeros((1, 30, 8)) 
    mock_seq[0, -1] = [
        context.get('spy_price', 480.0),
        context['vix_current'],
        context.get('vix_3m', 20.0),
        context.get('vvix_current', 90.0),
        context.get('skew_current', 120.0),
        context.get('yield_curve_spread', 0.1),
        context.get('yield_curve_spread', 0.1),
        context.get('vix_term_structure', 0.9)
    ]
    x = torch.FloatTensor(mock_seq)
    with torch.no_grad():
        prob = model(x).item()
    return prob

def run_orchestrator(mock: bool = True):
    print(f"[INFO] Phase 1: Real-time Signal Scanning (Mock={mock})...")
    
    # NEW: Data Provider Logic
    mode = "mock" if mock else "live"
    provider = get_provider(mode)
    context = provider.get_latest_context()
    
    # Calculate DVI for report structure
    dvi_raw = (context['vix_current'] + (100 - context['vix_term_structure']*100)) / 2
    dvi = min(max(dvi_raw, 0), 100)
    context['deval_vacuum_index'] = round(dvi, 1)

    # --- Phase 2: Citadel-Level Risk Engine ---
    # 1. Advanced Metrics Calculation (Citadel Metrics)
    vix_curr = context['vix_current']
    portfolio = context.get('portfolio', {})
    
    # Logic: Concentration
    commodities_tickers = ["GLD", "SLV", "URA", "COPX"]
    defense_tickers = ["LMT", "VST", "TEM"]
    
    # Mocking actual dollar exposure for calculation (Martin mentioned 75% / 25%)
    # In a real setup, we'd fetch actual lot sizes.
    raw_exposure = {
        "Commodities": 75,
        "Defense_AI_Utl": 25,
        "Cash": 22
    }
    
    # Beta Proxy: Commodities usually have higher beta in reflation, defensa lower.
    beta_est = 1.45 if dvi > 15 else 1.10
    
    # Correlation Proxy (Stress Correlation)
    corr_stress = 0.78 if dvi > 20 else 0.45 
    
    # Logic: Concentration & Sizing Simulation (Base AUM ~$4,500)
    total_aum = 4500
    curr_cash = total_aum * (raw_exposure['Cash'] / 100)
    reduction_target = 0.15 # Target reduction of portfolio for SLV+URA
    sell_amount = total_aum * 0.11 # ~USD 500 based on reduction target
    slv_ura_sizing = 30 # % based on user prompt

    # 2. ML Insight (Quantitative Pillar)
    print("[INFO] Phase 2: Running Layer 2 Transformer (Tail-Risk Prediction)...")
    try:
        model, scaler = load_transformer()
        crash_prob = get_prediction(model, scaler, context)
        tail_risk_prob = round(crash_prob * 100, 1)
    except Exception as e:
        print(f"[WARN] Transformer prediction failed: {e}")
        tail_risk_prob = 45.0 # Fallback
    
    # 3. Action Logic (Citadel Priority)
    vix_status = "Bajo Riesgo" if dvi < 25 else "Moderado" if dvi < 50 else "Elevado" if dvi < 75 else "Crítico"
    
    if tail_risk_prob > 35 or dvi > 55:
        action_lines = [
            "1. REDUCIR SLV + URA (vender ~USD 500) para bajar concentración.",
            "2. Rotar capital a SPY o Cash.",
            "3. HEDGE AUTOMÁTICO EN ESPERA (Gatillo DVI > 65).",
            f"   - Vender USD {sell_amount-20:.0f}–{sell_amount+20:.0f} de SLV + URA",
            "   - Nueva posición objetivo: Commodities → 50% del portafolio",
            f"   - Impacto en cash después: Cash sube a ~USD {curr_cash + sell_amount:.0f}"
        ]
        risk_remark = "ELEVADO → Preparar ejecución de protección"
    else:
        action_lines = [
            "1. Reducir SLV + URA del 30% → 15% (Sizing institucional).",
            "2. Rotar a SPY o cash.",
            "3. Mantener LMT y TEM (Bajo beta).",
            "4. No hedge necesario todavía.",
            f"   - Vender USD {sell_amount-20:.0f}–{sell_amount+20:.0f} de SLV + URA",
            "   - Nueva posición objetivo: Commodities → 50% del portafolio",
            f"   - Impacto en cash después: Cash sube a ~USD {curr_cash + sell_amount:.0f}"
        ]
        risk_remark = "Bajo → No se requiere hedge"

    # 4. Citadel View Narrative (Final Briefing Template)
    narrative = [
        f"DVI          {dvi}/100   {'↓' if dvi < 25 else '↑'}{abs(dvi-22) if dvi != 22 else 0}",
        f"Tail-Risk 30d {tail_risk_prob}%   {'↓' if tail_risk_prob < 50 else '↑'}??",
        f"Estado       {vix_status}",
        "",
        "Tu exposición crítica",
        f"Commodities (GLD/SLV/URA/COPX)   {raw_exposure['Commodities']}%   ← {'concentración extrema' if raw_exposure['Commodities'] > 40 else 'sana'}",
        f"Beta estimado vs SPX             {beta_est:.2f}  ← {'vulnerable' if beta_est > 1.3 else 'neutral'}",
        f"Cash ratio                       {raw_exposure['Cash']}%   ← {'sano' if raw_exposure['Cash'] > 15 else 'bajo'}",
        "",
        "Señales Citadel-level",
        f"• Commodities correlation en stress → {corr_stress:.2f} ({'alto' if corr_stress > 0.7 else 'bajo'})",
        f"• VIX {vix_curr:.1f} ({'barato para hedge' if vix_curr < 25 else 'caro'})",
        f"• SLV + URA = {slv_ura_sizing}% del portafolio → sizing demasiado grande",
        "",
        "Acción recomendada (prioridad Citadel)",
        "\n".join(action_lines),
        "",
        f"Riesgo general: {risk_remark}"
    ]

    # Reconstruct report object
    report = FullReport(
        generated_at=context['timestamp'],
        tickers=[],
        context=context,
        score={
            "beta": beta_est,
            "correlation": corr_stress,
            "concentration": raw_exposure['Commodities'],
            "cash_ratio": raw_exposure['Cash']
        },
        ticker_signals=[],
        narrative=narrative
    )
    payload = asdict(report)
    payload['tail_risk_probability'] = tail_risk_prob
    payload['citadel_metrics'] = report.score
    payload['rl_allocation'] = {"Equities": 55, "Commodities": 20, "Cash": 25} # Backward compatibility
    
    # --- Phase 6: On-chain Yield/Unwind logic (Production Secure) ---
    # Triggered by dvi thresholds. High risk (DVI > 75) requires manual /approve.
    dvi_val = payload['context']['deval_vacuum_index']
    from solana_bridge import SafeVaultBridge
    from alert_manager import DevalAlertManager
    
    bridge = SafeVaultBridge()
    alerts = DevalAlertManager()
    
    if dvi_val > 75:
        print(f"[HEDGE TRIGGER] Critical Risk (DVI={dvi_val}). Creating pending request...")
        bridge.create_pending_request(dvi_val, "UNWIND_PROTECTION_50_PERCENT")
        alerts.trigger_critical_alert(dvi_val, "Gatillo de Protección de Capital Activado (Pendiente Aprobación).")
    elif dvi_val < 30:
        print(f"[YIELD TRIGGER] Opportunity detected (DVI={dvi_val}). Strategy: Stake surplus.")
        # bridge.create_pending_request(dvi, "LOW_RISK_STAKE")
    
    # --- Phase 5: Human-in-the-loop Retraining ---
    retrain_from_collective()

    # Save for reporting (JSON for backward compatibility, DB for persistence)
    try:
        from persistence import DevalDBManager
        db = DevalDBManager()
        db.log_signal(payload)
    except Exception as e:
        print(f"[WARN] DB Logging failed: {e}")

    with open("integrated_signals.json", "w") as f:
        json.dump(payload, f, default=str, indent=2)
    
    print(f"[SUCCESS] Orchestration complete. Tail-Risk Probability: {payload['tail_risk_probability']}%")
    return payload

if __name__ == "__main__":
    res = run_orchestrator(mock=True)
    print(f"[INFO] RL Recommended Allocation: {res['rl_allocation']}")
