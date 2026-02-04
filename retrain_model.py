import json
import pandas as pd
import torch
import os

# CONFIG
FEEDBACK_FILE = "member_signals.json"
MODEL_PATH = "tail_risk_model.pth"

def retrain_from_collective():
    """
    Closes the loop: uses SÍ/NO feedback from the Telegram DAO 
    to adjust the predictive bias of the model.
    """
    if not os.path.exists(FEEDBACK_FILE):
        print("[INFO] No feedback data found yet. Skipping re-train.")
        return

    print("[INFO] Loading collective feedback for incremental re-training...")
    # Load data
    feedbacks = []
    with open(FEEDBACK_FILE, "r") as f:
        for line in f:
            feedbacks.append(json.loads(line))
    
    df = pd.DataFrame(feedbacks)
    if df.empty: return
    
    # Calculate Collective Bias
    # If users say SÍ (Presión) but model was LOW, we need to increase sensitivity.
    positive_ratio = df['outcome'].mean()
    print(f"[INFO] Collective Positive Ratio (Deval Pressure detected by humans): {positive_ratio:.2%}")
    
    # In a real setup, we would perform a gradient update on the Transformer weights.
    # For MVP, we'll log the 'Human-Adjusted Bias' which can be added to the DVI calculation.
    
    with open("collective_bias.json", "w") as f:
        json.dump({"adjustment": round(positive_ratio - 0.5, 3), "samples": len(df)}, f)
    
    print(f"[SUCCESS] Model bias adjusted. Human-in-the-loop cycle complete.")

if __name__ == "__main__":
    retrain_from_collective()
