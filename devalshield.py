import requests, json, torch, torch.nn as nn, os, sys
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
try:
    from solders.pubkey import Pubkey
    from solana.rpc.api import Client
    from solana.transaction import Transaction
    # anchorpy might be needed for high-level CPI, but we'll use a simple trigger for now
    SOLANA_SUPPORT = True
except ImportError:
    SOLANA_SUPPORT = False

# Load environment variables
load_dotenv(".env.monitoring")

# ==================== CONFIG ====================
APCA_API_KEY = os.getenv("APCA_API_KEY", "PKYOURKEYHERE")
APCA_API_SECRET = os.getenv("APCA_API_SECRET", "SKYOURSECRETHERE")
PAPER_MODE = os.getenv("APCA_PAPER_MODE", "True").lower() == "true"
HEDGE_NOTIONAL_USD = float(os.getenv("HEDGE_NOTIONAL_USD", "15.0"))
HEDGE_SYMBOL = os.getenv("HEDGE_SYMBOL", "SPY")

# Solana Config (Devnet for testing/shipping today)
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")
PROGRAM_ID_STR = os.getenv("DEVALSHIELD_PROGRAM_ID", "6hJ4z1REpTfJ9Q7uYy1R6pTfJ9Q7uYy1R6pTfJ9Q7uYy")
ORACLE_PDA_STR = os.getenv("INDEX_ORACLE_PDA", "") # Needs to be initialized
PRIVATE_KEY_BYTES = os.getenv("SOLANA_PRIVATE_KEY", "") # Base58 or bytes

# Initialize Alpaca Client
try:
    client = TradingClient(APCA_API_KEY, APCA_API_SECRET, paper=PAPER_MODE)
except Exception as e:
    print(f"[ERROR] Could not initialize Alpaca Client: {e}", file=sys.stderr)
    client = None

# ==================== TINY ML MODEL ====================
class DevalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 16), 
            nn.ReLU(), 
            nn.Linear(16, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

model = DevalNet()
MODEL_PATH = "deval_model.pt"
try: 
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        print(f"[INFO] Loaded existing model from {MODEL_PATH}")
except Exception as e:
    print(f"[WARN] Could not load model: {e}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def predict_deval_prob(features):
    """Predicts devaluation probability [global_comp, blue_gap, velocity, x_sent, hist_avg]"""
    # features: [global_comp, blue_gap, velocity, x_sent, hist_avg]
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        prob = model(x).item()
    return prob

def update_model(features, outcome):
    """Retrains the model based on human-in-the-loop feedback."""
    model.train()
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor([[float(outcome)]], dtype=torch.float32)
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[INFO] Model updated with outcome {outcome}. Loss: {loss.item():.4f}")

# ==================== MICRO HEDGE ====================
def place_micro_hedge(notional=HEDGE_NOTIONAL_USD, symbol=HEDGE_SYMBOL):
    if not client:
        print("[WARN] Alpaca client not initialized. Skipping hedge.")
        return
    
    try:
        order_details = MarketOrderRequest(
            symbol=symbol,
            notional=notional,          # fractional USD amount
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        order = client.submit_order(order_details)
        print(f"[AUTO-HEDGE] Bought ~${notional} {symbol} (paper={PAPER_MODE}) | Order ID: {order.id}")
    except Exception as e:
        print(f"[ERROR] Failed to place micro-hedge: {e}")

# ==================== SOLANA TRIGGER ====================
def trigger_unwind_if_needed(index: float):
    """Calls the Solana program to trigger the unwind (payout) mechanism."""
    if not SOLANA_SUPPORT:
        print("[WARN] Solana libraries not installed. Skipping on-chain trigger.")
        return
    
    if index <= 75:
        return
        
    print(f"[ON-CHAIN] Index {index} > 75. Triggering Solana Unwind...")
    # Placeholder for actual transaction construction
    # In a real scenario, we'd use anchorpy or solders to build the instruction
    try:
        # sol_client = Client(SOLANA_RPC_URL)
        # instruction = ... 
        # tx = Transaction().add(instruction)
        # result = sol_client.send_transaction(tx, ...)
        print(f"[SUCCESS] Solana Unwind Instruction Sent (MOCK for shipping)")
    except Exception as e:
        print(f"[ERROR] Solana trigger failed: {e}")

# ==================== CORE INTEGRATION ====================
def run_execution_logic(signals_path):
    """Reads the signal engine output and executes L3 logic."""
    if not os.path.exists(signals_path):
        print(f"[ERROR] Signals file not found at {signals_path}")
        return
    
    with open(signals_path, 'r') as f:
        data = json.load(f)
    
    ctx = data.get('context', {})
    
    # Extract features for ML
    # [global_comp, blue_gap, velocity, x_sent, hist_avg]
    # global_comp = 100 - vix_component (conceptual weight)
    # vix_component is in score, not context directly in signal_engine output usually, 
    # but let's check its structure.
    
    score_data = data.get('score', {})
    vix_comp = score_data.get('vix_component', 50)
    global_comp = max(0, 100 - vix_comp)
    
    blue_gap = ctx.get('blue_gap_pct', 0.0)
    velocity = ctx.get('gap_velocity', 0.0)
    x_sent = ctx.get('x_sentiment_score', 0.0)
    hist_avg = 35.0 # Placeholder for historical mean gap
    
    features = [global_comp, blue_gap, velocity, x_sent, hist_avg]
    
    # Predict
    prob_val = predict_deval_prob(features)
    prob_pct = prob_val * 100
    
    # Calculate intertwined index (DVI evolve)
    # User's formula: index = 0.5 * global_comp + 0.5 * deval_prob
    dvi_l3 = round(0.5 * global_comp + 0.5 * prob_pct, 1)
    
    print(f"[L3] Deval Vacuum Index (ML-Enhanced): {dvi_l3}")
    print(f"[L3] ML Predicted Prob: {prob_pct:.1f}%")
    
    # Execution decision
    if dvi_l3 > 75:
        place_micro_hedge()
        trigger_unwind_if_needed(dvi_l3)
    
    # Inject back into data
    data['tail_risk_probability'] = round(prob_pct, 1)
    data['rl_allocation'] = {
        "USD Proxy (SPY)": 20 if dvi_l3 > 75 else 5,
        "Local Assets": 80 if dvi_l3 <= 75 else 50,
        "Hedge Overlay": 30 if dvi_l3 > 75 else 0
    }
    
    # Save back
    with open(signals_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return features

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python devalshield.py <signals.json>")
        sys.exit(1)
        
    sig_file = sys.argv[1]
    feats = run_execution_logic(sig_file)
    
    # Feedback loop is handled by the user interactive session or a separate tool
    # For automated monitoring, we might skip the input() unless it's an interactive run.
    if os.isatty(sys.stdin.fileno()):
        try:
            feedback = input("\n[FEEDBACK] ¿Se materializó la presión? (SÍ/NO): ").strip().upper()
            if feedback:
                outcome = 1 if feedback in ("SÍ", "YES", "SI") else 0
                update_model(feats, outcome)
        except EOFError:
            pass
    else:
        print("[INFO] Non-interactive session, skipping manual feedback loop.")
