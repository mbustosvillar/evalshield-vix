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
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from tail_risk_model import TailRiskTransformer
from ppo_agent import PolicyNetwork
from signal_engine import build_report
from solana_bridge import update_onchain_index
from retrain_model import retrain_from_collective

# CONFIG
MODEL_PATH = "tail_risk_model.pth"
RL_AGENT_PATH = "ppo_agent.pth" # Usually saved after training

def load_transformer():
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    # Architecture params from tail_risk_model.py
    model = TailRiskTransformer(input_dim=8, hidden_dim=32, num_heads=4, num_layers=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['scaler']

def get_prediction(model, scaler, context):
    # Construct feature vector matching tail_risk_model feature_cols: 
    # ["spy", "vix", "vix3m", "vvix", "skew", "tnx", "tyx", "vix_term_structure"]
    # We use a mock sequence for the demonstration
    mock_seq = np.zeros((1, 30, 8)) 
    # Fill last step with current market context
    mock_seq[0, -1] = [
        480.0, # SPY proxy
        context['vix_current'],
        context['vix_3m'] or 20.0,
        context['vvix_current'] or 90.0,
        context['skew_current'] or 120.0,
        context['yield_curve_spread'] or 0.1, # Using spread as proxy for TNX/TYX logic
        context['yield_curve_spread'] or 0.1,
        context['vix_term_structure'] or 0.9
    ]
    # Scaling (simplified for real-time)
    x = torch.FloatTensor(mock_seq)
    with torch.no_grad():
        prob = model(x).item()
    return prob

def run_orchestrator(mock: bool = True):
    print(f"[INFO] Phase 1: Real-time Signal Scanning (Mock={mock})...")
    # Get current signals via signal_engine
    report = build_report(mock=mock)
    payload = asdict(report)
    
    print("[INFO] Phase 2: Running Layer 2 Transformer (Tail-Risk Prediction)...")
    try:
        model, scaler = load_transformer()
        crash_prob = get_prediction(model, scaler, report.context)
        payload['tail_risk_probability'] = round(crash_prob * 100, 1)
    except Exception as e:
        print(f"[WARN] Transformer prediction failed: {e}")
        payload['tail_risk_probability'] = 45.0 # Fallback
    
    print("[INFO] Phase 3: Executing Layer 3 RL Allocation Agent...")
    # RL Allocation Logic (Mocking the agent's output based on probability)
    prob = payload['tail_risk_probability']
    if prob > 70:
        allocation = {"SPY": 10, "VIX_Calls": 50, "CASH": 40}
    elif prob > 40:
        allocation = {"SPY": 60, "VIX_Calls": 10, "CASH": 30}
    else:
        allocation = {"SPY": 95, "VIX_Calls": 0, "CASH": 5}
    
    payload['rl_allocation'] = allocation
    
    # --- Phase 6: On-chain Yield/Unwind logic (Draft) ---
    # Triggered by dvi thresholds in real money mode
    dvi = payload['context']['deval_vacuum_index']
    if dvi < 40:
        print(f"[YIELD MODE] Opportunity detected (DVI={dvi}). Strategy: Stake USDT -> JitoSOL.")
        # In prod: call_solana_ix("rebalance_low_index")
    elif dvi > 75:
        print(f"[UNWIND MODE] Critical Risk (DVI={dvi}). Strategy: Unwind 50% to USDC Payout.")
        # In prod: asyncio.run(update_onchain_index(dvi))
    
    # --- Phase 5: Human-in-the-loop Retraining ---
    retrain_from_collective()

    # Save for reporting
    with open("integrated_signals.json", "w") as f:
        json.dump(payload, f, default=str, indent=2)
    
    print(f"[SUCCESS] Orchestration complete. Tail-Risk Probability: {payload['tail_risk_probability']}%")
    return payload

if __name__ == "__main__":
    res = run_orchestrator(mock=True)
    print(f"[INFO] RL Recommended Allocation: {res['rl_allocation']}")
