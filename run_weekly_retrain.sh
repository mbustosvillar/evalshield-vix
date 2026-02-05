#!/bin/bash
# =============================================================================
# DevalShield Weekly Retraining Pipeline
# Runs every Sunday at 3:00 AM (GMT-3)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="logs/weekly_retrain.log"
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "Starting Weekly Retraining Pipeline"
log "=========================================="

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    log "Activated virtual environment"
fi

# Step 1: Run retraining script
log "Running retrain_model.py..."
if python3 retrain_model.py; then
    log "✓ Retraining completed successfully"
else
    log "⚠ Retraining skipped (insufficient data or error)"
fi

# Step 2: Commit any model changes
if [ -f "collective_bias.json" ]; then
    log "Checking for model updates to commit..."
    if git diff --quiet collective_bias.json 2>/dev/null; then
        log "No changes to commit"
    else
        git add collective_bias.json
        git commit -m "chore: update collective bias $(date +%Y-%m-%d)" || true
        log "✓ Changes committed"
    fi
fi

# Step 3: Generate weekly summary
log "Generating weekly summary..."
echo "{
  \"last_retrain\": \"$(date -Iseconds)\",
  \"status\": \"completed\",
  \"next_run\": \"$(date -d '+7 days' -Iseconds 2>/dev/null || date -v+7d -Iseconds)\"
}" > logs/retrain_status.json

log "=========================================="
log "Weekly Retraining Pipeline Complete"
log "=========================================="
