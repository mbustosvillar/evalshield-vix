#!/bin/bash

# run_daily_monitoring.sh - Daily automated execution for VIX dashboard
# Scheduled to run via cron daily at 08:00 GMT-3

# --- Configuration ---
PROJECT_DIR="/Users/user/Downloads/Projects_Misc/VIX dashboard/files"
LOG_DIR="${PROJECT_DIR}/logs"
REPORT_DIR="${PROJECT_DIR}/reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DAILY_LOG="${LOG_DIR}/monitoring_${TIMESTAMP}.log"
SIGNAL_JSON="signals_${TIMESTAMP}.json"
REPORT_DOCX="tactical_report_${TIMESTAMP}.docx"

# --- Initialization ---
mkdir -p "${LOG_DIR}"
mkdir -p "${REPORT_DIR}"

exec > >(tee -a "${DAILY_LOG}") 2>&1

echo "=========================================================================="
echo "VIX MONITORING SESSION STARTED: $(date)"
echo "=========================================================================="

# --- Step 1: Signal Generation (Python) ---
echo "[1/4] Fetching tactical signals..."
# Using --mock to avoid rate limits during active development/testing
python3 "${PROJECT_DIR}/signal_engine.py" --output "${PROJECT_DIR}/signals_raw.json" --mock

echo "[2/3] Fetching macro regime data (Layer 1)..."
# Using --synthetic to avoid rate limits during active development/testing
python3 "${PROJECT_DIR}/macro_regime_engine.py" --output "${PROJECT_DIR}/macro_raw.json" --synthetic

echo "[3/4] Fusing data and generating tactical report..."
python3 "${PROJECT_DIR}/merge_reports.py" "${PROJECT_DIR}/signals_raw.json" "${PROJECT_DIR}/macro_raw.json" "${PROJECT_DIR}/${SIGNAL_JSON}"

echo "[4/4] Executing Layer 3: ML Prediction & Alpaca Hedge..."
# Note: feedback loop is skipped in non-interactive cron runs
python3 "${PROJECT_DIR}/devalshield.py" "${PROJECT_DIR}/${SIGNAL_JSON}"

node "${PROJECT_DIR}/generate_report.js" "${PROJECT_DIR}/${SIGNAL_JSON}" "${PROJECT_DIR}/${REPORT_DOCX}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Report generation failed."
    exit 1
fi
echo "[SUCCESS] Report generated: ${REPORT_DOCX}"

# --- Step 3: Archive and Cleanup ---
mv "${PROJECT_DIR}/${SIGNAL_JSON}" "${REPORT_DIR}/"
mv "${PROJECT_DIR}/${REPORT_DOCX}" "${REPORT_DIR}/"

# Keep the latest report as tactical_report.docx for quick access
cp "${REPORT_DIR}/${REPORT_DOCX}" "${PROJECT_DIR}/tactical_report.docx"

echo "=========================================================================="
echo "SESSION FINISHED SUCCESSFULLY: $(date)"
echo "Archive: ${REPORT_DIR}"
echo "=========================================================================="
