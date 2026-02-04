"""
allocation_gym.py
=================
Custom Gymnasium environment for Tactical Asset Allocation.
Agent learns to manage a portfolio of Equities, Volatility Hedges, and Safe Havens.
Reward function is heavily asymmetric to punish tail-risk losses.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TailRiskEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df):
        super(TailRiskEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.max_steps = len(df) - 1

        # State: [vix, term_structure, vvix, skew, tnx, tyx, fwd_ret_21d (proxy for pred), cash_balance]
        # In practice, fwd_ret_21d is replaced by the Transformer's prediction.
        self.observation_space = spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32)

        # Action: [weight_spy, weight_vol_hedge, weight_safe_haven] - normalized to 1.0
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = 1000.0
        self.history = []
        return self._get_obs(), {}

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
            row['fwd_ret_21d'] * 10, # Proxy for prediction
            0.0 # Normalized portfolio delta placeholder
        ], dtype=np.float32)
        return obs

    def step(self, action):
        # Action normalization
        weights = action / (np.sum(action) + 1e-8)
        
        row = self.df.iloc[self.current_step]
        next_row = self.df.iloc[self.current_step + 1]

        # Returns
        ret_spy = next_row['spy'] / row['spy'] - 1
        # Vol hedge proxy: moves inversely to SPY and spikes on high VIX
        ret_vol = -2.0 * ret_spy + (0.1 if row['vix'] > 25 else 0.0) 
        # Safe haven proxy (rates)
        ret_safe = (row['tnx'] / 100) / 252 # Daily yield approx
        
        portfolio_return = weights[0]*ret_spy + weights[1]*ret_vol + weights[2]*ret_safe
        
        # Update portfolio
        prev_value = self.portfolio_value
        self.portfolio_value *= (1 + portfolio_return)
        
        # Reward function: Asymmetric with Max Drawdown Penalty
        # Sharpe/Sortino proxy: return - constant * losses
        penalty = 0
        if portfolio_return < 0:
            penalty = 5.0 * abs(portfolio_return)**2 # Squared penalty for downside
            
        reward = portfolio_return - penalty
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, done, False, {}

def create_env(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return TailRiskEnv(df)
