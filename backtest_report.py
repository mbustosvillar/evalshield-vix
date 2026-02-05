#!/usr/bin/env python3
"""
DevalShield Backtest Reporting
==============================
Generates visual reports from backtest results.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

def generate_report(json_path="backtest_results.json"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run backtester.py first.")
        sys.exit(1)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Prepare data for plotting
    dates = pd.to_datetime(pd.read_csv("historical_training_data.csv", index_col=0).index, utc=True)
    # Align dates with equity curve length (taking the end slice)
    dates = dates[-len(data['equity_curve']):]
    
    equity = pd.Series(data['equity_curve'], index=dates)
    drawdown = pd.Series(data['drawdown_series'], index=dates) * -100 # Convert to negative percentage

    # Create charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Equity Curve
    ax1.plot(equity.index, equity.values, label=data['strategy_name'], color='#00ff88', linewidth=1.5)
    ax1.set_title(f"Equity Curve: {data['strategy_name']} (CAGR: {data['cagr']}%)", fontsize=14, color='white')
    ax1.set_ylabel("Portfolio Value ($)", color='white')
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper left')
    
    # Styling for dark theme report
    fig.patch.set_facecolor('#1e1e1e')
    ax1.set_facecolor('#1e1e1e')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white') 
    ax1.spines['left'].set_color('white')
    ax1.spines['right'].set_color('white')

    # Plot 2: Drawdown
    ax2.fill_between(drawdown.index, drawdown.values, 0, color='#ff4444', alpha=0.6, label='Drawdown')
    ax2.set_title(f"Drawdown Profile (Max: {data['max_drawdown']}%)", fontsize=12, color='white')
    ax2.set_ylabel("Drawdown (%)", color='white')
    ax2.grid(True, alpha=0.2)
    ax2.set_facecolor('#1e1e1e')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    plt.savefig("backtest_chart.png", dpi=100, bbox_inches='tight')
    print("✅ Chart generated: backtest_chart.png")

    # Generate Markdown Summary
    md_report = f"""
# 📊 Backtest Report: {data['strategy_name']}

![Equity Curve](backtest_chart.png)

## Performance Metrics
| Metric | Value |
|--------|-------|
| **Total Return** | `{data['total_return']}%` |
| **CAGR** | `{data['cagr']}%` |
| **Sharpe Ratio** | `{data['sharpe_ratio']}` |
| **Sortino Ratio** | `{data['sortino_ratio']}` |
| **Max Drawdown** | `{data['max_drawdown']}%` |
| **Vol (Ann.)** | `{data['volatility']}%` |

## Risk Profile
- **Calmar Ratio**: {data['calmar_ratio']}
- **Win Rate**: {data['win_rate']}%
- **VaR (95%)**: {data['var_95']}%
- **Longest DD**: {data['max_drawdown_duration_days']} days

_Generated automatically by DevalShield Backtester_
    """
    
    with open("backtest_report.md", "w") as f:
        f.write(md_report)
    print("✅ Report generated: backtest_report.md")

if __name__ == "__main__":
    generate_report()
