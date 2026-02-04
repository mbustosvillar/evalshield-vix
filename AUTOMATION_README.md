# Automation Documentation: Daily VIX Monitoring

This document describes the automated daily monitoring system for the VIX Dashboard.

## Overview

The system consists of a Bash wrapper script (`run_daily_monitoring.sh`) that orchestrates the execution of the Python signal engine and the Node.js report generator. It is scheduled to run daily via `cron`.

## Setup Instructions

1. **Permissions**: Ensure the wrapper script is executable:
   ```bash
   chmod +x "/Users/user/Downloads/Projects_Misc/VIX dashboard/files/run_daily_monitoring.sh"
   ```

2. **Cron Installation**: 
   Open your crontab for editing:
   ```bash
   crontab -e
   ```
   Add the following line (reference `crontab_entry.txt`):
   ```
   0 8 * * 1-5 /bin/bash "/Users/user/Downloads/Projects_Misc/VIX dashboard/files/run_daily_monitoring.sh" >> "/Users/user/Downloads/Projects_Misc/VIX dashboard/files/logs/cron_execution.log" 2>&1
   ```

## Directory Structure

- `logs/`: Contains execution logs with timestamps.
- `reports/`: Contains archived JSON signals and DOCX reports.
- `tactical_report.docx`: The most recent report (at project root).

## Manual Execution

You can run the monitoring manually at any time:
```bash
./run_daily_monitoring.sh
```

## Troubleshooting

- **Check logs**: If a run fails, check the latest log in the `logs/` directory.
- **Dependencies**: Ensure `python3` and `node` are available in your path.
- **YFinance/FRED**: The signal engine requires an internet connection to fetch market data.
