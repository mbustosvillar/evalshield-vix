"""
allocation_simulator.py
=======================
Standalone Portfolio Simulator for Tactical Asset Allocation.
No external dependencies (Gym/Gymnasium) required.
"""

import numpy as np
import pandas as pd

class TailRiskSimulator:
    def __init__(self, df):
        self.df = df
        self.current_step = 0
        self.max_steps = len(df) - 1
        self.portfolio_value = 1000.0
        
        # Ranges for normalization
        self.state_dim = 8
        self.action_dim = 3

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1000.0
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        # Normalized state
        obs = np.array([
            (row['vix'] - 15) / 10,
            (row['vix_term_structure'] - 0.9) / 0.1,
            (row['vvix'] - 90) / 20,
            (row['skew'] - 120) / 10,
            (row['tnx'] - 3) / 1,
            (row['tyx'] - 4) / 1,
            row['fwd_ret_21d'] * 10,
            0.0 
        ], dtype=np.float32)
        return obs

    def step(self, action):
        # Action normalization (Weights sum to 1.0)
        weights = action / (np.sum(action) + 1e-8)
        
        row = self.df.iloc[self.current_step]
        next_row = self.df.iloc[self.current_step + 1]

        # Returns
        ret_spy = next_row['spy'] / row['spy'] - 1
        # Vol hedge proxy: moves inversely to SPY and spikes on high VIX
        ret_vol = -2.5 * ret_spy + (0.01 if row['vix'] > 20 else 0.0)
        ret_safe = (row['tnx'] / 100) / 252 
        
        portfolio_return = weights[0]*ret_spy + weights[1]*ret_vol + weights[2]*ret_safe
        
        # Update portfolio
        self.portfolio_value *= (1 + portfolio_return)
        
        # Asymmetric Reward function
        penalty = 0
        if portfolio_return < 0:
            penalty = 10.0 * abs(portfolio_return)**2 # Heavy quadratic penalty
            
        reward = portfolio_return - penalty
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, done

def create_sim(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return TailRiskSimulator(df)
