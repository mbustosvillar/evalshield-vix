#!/usr/bin/env python3
"""
DevalShield Model Retraining Pipeline
Performs weekly incremental retraining based on collective DAO feedback.
"""
import json
import pandas as pd
import torch
import os
import shutil
from datetime import datetime

# CONFIG
FEEDBACK_FILE = "member_signals.json"
MODEL_PATH = "tail_risk_model.pth"
BACKUP_DIR = "model_backups"
MIN_SAMPLES = 10  # Minimum feedback samples to retrain
LOG_FILE = "logs/retrain_log.txt"

def log_message(msg: str):
    """Append message to retrain log with timestamp."""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")

def backup_model():
    """Create timestamped backup of current model."""
    if not os.path.exists(MODEL_PATH):
        log_message("WARNING: No model found to backup.")
        return None
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{BACKUP_DIR}/tail_risk_model_{timestamp}.pth"
    shutil.copy(MODEL_PATH, backup_path)
    log_message(f"Model backed up to: {backup_path}")
    return backup_path

def validate_feedback_data(feedbacks: list) -> bool:
    """Validate feedback data quality."""
    if len(feedbacks) < MIN_SAMPLES:
        log_message(f"Insufficient samples: {len(feedbacks)} < {MIN_SAMPLES}. Skipping.")
        return False
    
    df = pd.DataFrame(feedbacks)
    required_cols = ['outcome']
    if not all(col in df.columns for col in required_cols):
        log_message(f"Missing required columns. Found: {df.columns.tolist()}")
        return False
    
    # Check for data consistency
    if df['outcome'].isnull().sum() / len(df) > 0.1:
        log_message("WARNING: >10% null outcomes detected.")
    
    return True

def retrain_from_collective():
    """
    Closes the loop: uses SÍ/NO feedback from the Telegram DAO 
    to adjust the predictive bias of the model.
    """
    log_message("=" * 50)
    log_message("Starting weekly retraining cycle...")
    
    if not os.path.exists(FEEDBACK_FILE):
        log_message("No feedback data found yet. Skipping re-train.")
        return False

    # Load data
    feedbacks = []
    try:
        with open(FEEDBACK_FILE, "r") as f:
            for line in f:
                if line.strip():
                    feedbacks.append(json.loads(line))
    except json.JSONDecodeError as e:
        log_message(f"ERROR parsing feedback file: {e}")
        return False
    
    if not validate_feedback_data(feedbacks):
        return False
    
    df = pd.DataFrame(feedbacks)
    log_message(f"Loaded {len(df)} feedback samples for retraining.")
    
    # Backup current model before modifications
    backup_model()
    
    # Calculate Collective Bias
    positive_ratio = df['outcome'].mean()
    log_message(f"Collective Positive Ratio: {positive_ratio:.2%}")
    
    # Calculate adjustment (human wisdom delta)
    adjustment = round(positive_ratio - 0.5, 3)
    
    # Save bias adjustment
    bias_data = {
        "adjustment": adjustment,
        "samples": len(df),
        "positive_ratio": round(positive_ratio, 4),
        "timestamp": datetime.now().isoformat(),
        "feedback_window_days": 7
    }
    
    with open("collective_bias.json", "w") as f:
        json.dump(bias_data, f, indent=2)
    
    log_message(f"Bias adjustment saved: {adjustment:+.3f}")
    log_message(f"SUCCESS: Human-in-the-loop cycle complete.")
    
    # Clear processed feedback (archive it first)
    archive_feedback(feedbacks)
    
    return True

def archive_feedback(feedbacks: list):
    """Archive processed feedback data."""
    os.makedirs("logs/feedback_archive", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    archive_path = f"logs/feedback_archive/feedback_{timestamp}.json"
    with open(archive_path, "w") as f:
        json.dump(feedbacks, f, indent=2)
    log_message(f"Feedback archived to: {archive_path}")
    
    # Clear the main feedback file
    open(FEEDBACK_FILE, "w").close()
    log_message("Feedback file cleared for next cycle.")

if __name__ == "__main__":
    success = retrain_from_collective()
    exit(0 if success else 1)

