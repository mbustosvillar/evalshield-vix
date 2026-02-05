import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def run_backtest(csv_path="historical_training_data.csv"):
    df = pd.read_csv(csv_path)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    date_col = df.columns[0]

    # Constants from signal_engine.py
    W_BREADTH = 0.25
    W_VIX = 0.35
    W_MOMENTUM = 0.25
    W_YIELD = 0.15
    
    # Strategy parameters
    HEDGE_THRESHOLD = 70.0
    
    spy_returns = []
    strategy_returns = []
    active_hedge = False
    
    portfolio_spy = 100.0
    portfolio_strat = 100.0
    
    results = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 1. Calculate basic VIX component
        # In signal_engine: vix_complacency (+40) if at 52w low, vix_panic_spike (+60) if >35% spike
        # We simplify for backtest using available columns
        vix_lvl = row['vix']
        vix_chg = (vix_lvl - prev_row['vix']) / prev_row['vix']
        
        vix_comp = 0.0
        if vix_chg > 0.30: vix_comp += 60 # Panic spike
        if vix_chg > 0.10: vix_comp += 20 # Rising
        
        # 2. VIX Term Structure and VVIX (Layer 2)
        vix_term = row['vix_term_structure'] # VIX/VIX3M
        vix_term_score = 100.0 if vix_term > 1.0 else 0.0 # Inversion
        
        vvix = row['vvix']
        vvix_score = 100.0 if vvix > 110 else 50.0 if vvix > 95 else 0.0
        
        composite_vix_score = (vix_comp * 0.5 + vix_term_score * 0.3 + vvix_score * 0.2)
        
        # 3. Yield Curve (Proxy using TNX level or change since we lack 2Y)
        # In real app: 10Y-2Y spread. Here we use a simpler logic for the demo
        y_score = 0.0 # Neutral in backtest unless we find 2Y data
        
        # 4. Momentum and Breadth Proxy (Using SPY distance from MA)
        # Calculate moving averages on the fly
        spy_prices = df['spy'].iloc[max(0, i-50):i+1]
        ma50 = spy_prices.mean()
        mom_score = 100.0 if row['spy'] < ma50 else 0.0 # Simple distance-based momentum score
        
        breadth_proxy = 0.0 # Unknown breadth in history CSV
        
        # Final Air Pocket Score (Simplified for backtest)
        # Using higher weight on VIX and Momentum as breadth/yield are proxies here
        total_score = (0.5 * composite_vix_score + 0.5 * mom_score)
        
        # Strategy Logic
        spy_ret = (row['spy'] - prev_row['spy']) / prev_row['spy']
        
        if total_score >= HEDGE_THRESHOLD:
            # HEDGE ACTIVE: Move to Cash (0 return) or minor cost
            strat_ret = 0.0 
            active_hedge = True
        else:
            strat_ret = spy_ret
            active_hedge = False
            
        portfolio_spy *= (1 + spy_ret)
        portfolio_strat *= (1 + strat_ret)
        
        results.append({
            'date': row[date_col],
            'spy_price': row['spy'],
            'score': total_score,
            'portfolio_spy': portfolio_spy,
            'portfolio_strat': portfolio_strat,
            'hedged': active_hedge
        })

    res_df = pd.DataFrame(results)
    
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(res_df['date'], res_df['portfolio_spy'], label='Benchmark (SPY Buy & Hold)', alpha=0.6)
    plt.plot(res_df['date'], res_df['portfolio_strat'], label='DevalShield Active Strategy', linewidth=2)
    plt.fill_between(res_df['date'], 0, res_df['portfolio_strat'].max(), where=res_df['hedged'], color='red', alpha=0.1, label='Hedge Active')
    
    plt.title("DevalShield Validation: 2008 & 2020 Stress-Test")
    plt.ylabel("Portfolio Value (Starting at 100)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = "backtest_validation.png"
    plt.savefig(plot_path)
    print(f"Backtest plot saved to {plot_path}")
    
    # Metrics
    final_spy = portfolio_spy
    final_strat = portfolio_strat
    outperformance = final_strat - final_spy
    
    print(f"Final SPY: {final_spy:.2f}")
    print(f"Final Strategy: {final_strat:.2f}")
    print(f"Total Outperformance: {outperformance:.2f}%")
    
    return res_df

if __name__ == "__main__":
    run_backtest()
