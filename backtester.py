#!/usr/bin/env python3
"""
DevalShield Backtesting Engine
==============================
Institutional-grade portfolio backtesting with CFA-standard risk metrics.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

# Constants
TRADING_DAYS_YEAR = 252
RISK_FREE_RATE = 0.04  # 4% annual, adjustable

@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    volatility: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: List[float]
    drawdown_series: List[float]
    
    def to_dict(self) -> Dict:
        return {
            "strategy_name": self.strategy_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_value": round(self.final_value, 2),
            "total_return": round(self.total_return * 100, 2),
            "cagr": round(self.cagr * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "max_drawdown": round(self.max_drawdown * 100, 2),
            "max_drawdown_duration_days": self.max_drawdown_duration,
            "var_95": round(self.var_95 * 100, 2),
            "cvar_95": round(self.cvar_95 * 100, 2),
            "volatility": round(self.volatility * 100, 2),
            "win_rate": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 2),
            "total_trades": self.total_trades,
            "equity_curve": self.equity_curve,
            "drawdown_series": self.drawdown_series
        }
    
    def summary(self) -> str:
        d = self.to_dict()
        return f"""
╔══════════════════════════════════════════════════════════╗
║  BACKTEST RESULTS: {d['strategy_name']:^36}  ║
╠══════════════════════════════════════════════════════════╣
║  Period: {d['start_date']} → {d['end_date']}
║  Initial: ${d['initial_capital']:,.0f}  →  Final: ${d['final_value']:,.2f}
╠══════════════════════════════════════════════════════════╣
║  RETURNS                        │  RISK METRICS
║  ────────────────────────────── │ ──────────────────────
║  Total Return:    {d['total_return']:>7.2f}%      │  Volatility:   {d['volatility']:>6.2f}%
║  CAGR:            {d['cagr']:>7.2f}%      │  Max Drawdown: {d['max_drawdown']:>6.2f}%
║  Win Rate:        {d['win_rate']:>7.2f}%      │  VaR (95%):    {d['var_95']:>6.2f}%
╠══════════════════════════════════════════════════════════╣
║  RISK-ADJUSTED METRICS
║  Sharpe Ratio:    {d['sharpe_ratio']:>6.3f}  │  Sortino Ratio: {d['sortino_ratio']:>6.3f}
║  Calmar Ratio:    {d['calmar_ratio']:>6.3f}  │  Profit Factor: {d['profit_factor']:>6.2f}
╚══════════════════════════════════════════════════════════╝
"""


class PortfolioBacktester:
    """
    Main backtesting engine for DevalShield strategies.
    
    Supports:
    - Multiple allocation strategies
    - Transaction cost modeling
    - Rolling window analysis
    - Monte Carlo simulation
    """
    
    def __init__(
        self,
        data_path: str = "historical_training_data.csv",
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,  # 10 bps
        risk_free_rate: float = RISK_FREE_RATE
    ):
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess historical data."""
        self.df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        self.df.index = pd.to_datetime(self.df.index, utc=True)
        
        # Calculate daily returns
        self.df['spy_ret'] = self.df['spy'].pct_change()
        self.df['vol_proxy_ret'] = -2.5 * self.df['spy_ret'] + np.where(self.df['vix'] > 20, 0.01, 0)
        self.df['safe_ret'] = (self.df['tnx'] / 100) / TRADING_DAYS_YEAR
        
        self.df = self.df.dropna()
    
    def run_backtest(
        self,
        strategy: str = "static_60_20_20",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResult:
        """
        Run backtest with specified strategy.
        
        Strategies:
        - static_60_20_20: 60% SPY, 20% Vol Hedge, 20% Safe
        - risk_parity: Equal risk contribution
        - dvi_adaptive: Adjust based on DVI signals
        - vix_regime: Switch based on VIX levels
        """
        # Filter date range
        df = self.df.copy()
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Get allocation weights based on strategy
        weights_history = self._get_strategy_weights(df, strategy)
        
        # Calculate portfolio returns
        portfolio_returns = []
        equity = [self.initial_capital]
        prev_weights = None
        
        for i, (idx, row) in enumerate(df.iterrows()):
            weights = weights_history[i]
            
            # Transaction costs on rebalance
            tc = 0
            if prev_weights is not None:
                tc = self.transaction_cost * np.sum(np.abs(weights - prev_weights))
            
            # Portfolio return
            ret = (
                weights[0] * row['spy_ret'] +
                weights[1] * row['vol_proxy_ret'] +
                weights[2] * row['safe_ret'] -
                tc
            )
            portfolio_returns.append(ret)
            equity.append(equity[-1] * (1 + ret))
            prev_weights = weights
        
        returns = np.array(portfolio_returns)
        equity = np.array(equity[1:])  # Remove initial
        
        # Calculate all metrics
        result = self._calculate_metrics(
            returns=returns,
            equity=equity,
            strategy_name=strategy,
            start_date=str(df.index[0].date()),
            end_date=str(df.index[-1].date())
        )
        
        return result
    
    def _get_strategy_weights(self, df: pd.DataFrame, strategy: str) -> np.ndarray:
        """Generate allocation weights based on strategy."""
        n = len(df)
        
        if strategy == "static_60_20_20":
            # Classic 60/20/20 allocation
            return np.tile([0.60, 0.20, 0.20], (n, 1))
        
        elif strategy == "static_40_30_30":
            # More defensive allocation
            return np.tile([0.40, 0.30, 0.30], (n, 1))
        
        elif strategy == "vix_regime":
            # Adaptive based on VIX
            weights = []
            for _, row in df.iterrows():
                vix = row['vix']
                if vix < 15:  # Low vol → Risk on
                    w = [0.80, 0.10, 0.10]
                elif vix < 25:  # Normal → Balanced
                    w = [0.60, 0.20, 0.20]
                elif vix < 35:  # High → Defensive
                    w = [0.40, 0.30, 0.30]
                else:  # Crisis → Max hedge
                    w = [0.20, 0.40, 0.40]
                weights.append(w)
            return np.array(weights)
        
        elif strategy == "dvi_adaptive":
            # Use our custom DVI-like signal
            weights = []
            for _, row in df.iterrows():
                # Simplified DVI: term structure + VVIX
                term_stress = 1 - row['vix_term_structure']  # Inverted structure = stress
                vvix_stress = (row['vvix'] - 80) / 40  # Normalized VVIX
                dvi = (term_stress + vvix_stress) / 2
                
                if dvi < 0.2:  # Low stress
                    w = [0.75, 0.15, 0.10]
                elif dvi < 0.5:  # Moderate
                    w = [0.55, 0.25, 0.20]
                else:  # High stress
                    w = [0.30, 0.35, 0.35]
                weights.append(w)
            return np.array(weights)
        
        elif strategy == "risk_parity":
            # Simplified risk parity (inverse vol weighting)
            weights = []
            lookback = 20
            spy_vol = df['spy_ret'].rolling(lookback).std()
            vol_vol = df['vol_proxy_ret'].rolling(lookback).std()
            safe_vol = df['safe_ret'].rolling(lookback).std().replace(0, 0.001)
            
            for i in range(len(df)):
                if i < lookback:
                    w = [0.33, 0.33, 0.34]
                else:
                    inv_vols = 1 / np.array([spy_vol.iloc[i], vol_vol.iloc[i], safe_vol.iloc[i]])
                    w = inv_vols / inv_vols.sum()
                weights.append(w)
            return np.array(weights)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _calculate_metrics(
        self,
        returns: np.ndarray,
        equity: np.ndarray,
        strategy_name: str,
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """Calculate comprehensive risk metrics."""
        
        # Basic returns
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        n_years = len(returns) / TRADING_DAYS_YEAR
        cagr = (equity[-1] / self.initial_capital) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(TRADING_DAYS_YEAR)
        
        # Sharpe Ratio
        excess_returns = returns - self.risk_free_rate / TRADING_DAYS_YEAR
        sharpe = np.mean(excess_returns) / (returns.std() + 1e-8) * np.sqrt(TRADING_DAYS_YEAR)
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-8
        sortino = np.mean(excess_returns) / (downside_std + 1e-8) * np.sqrt(TRADING_DAYS_YEAR)
        
        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()
        
        # Max Drawdown Duration
        dd_duration = self._calc_max_dd_duration(drawdown)
        
        # Calmar Ratio
        calmar = cagr / max_drawdown if max_drawdown > 0 else 0
        
        # VaR and CVaR (95%)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # Win rate and profit factor
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float('inf')
        
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=equity[-1],
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            max_drawdown_duration=dd_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor if profit_factor != float('inf') else 99.99,
            total_trades=len(returns),
            equity_curve=equity.tolist(),
            drawdown_series=drawdown.tolist()
        )
    
    def _calc_max_dd_duration(self, drawdown: np.ndarray) -> int:
        """Calculate maximum drawdown duration in trading days."""
        in_dd = drawdown > 0
        max_duration = 0
        current = 0
        for dd in in_dd:
            if dd:
                current += 1
                max_duration = max(max_duration, current)
            else:
                current = 0
        return max_duration
    
    def compare_strategies(
        self,
        strategies: List[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Compare multiple strategies side by side."""
        if strategies is None:
            strategies = ["static_60_20_20", "static_40_30_30", "vix_regime", "dvi_adaptive"]
        
        results = []
        for strat in strategies:
            result = self.run_backtest(strat, start_date, end_date)
            results.append(result.to_dict())
        
        df = pd.DataFrame(results)
        df = df.set_index('strategy_name')
        return df
    
    def rolling_analysis(
        self,
        strategy: str = "dvi_adaptive",
        window_years: int = 3
    ) -> pd.DataFrame:
        """Calculate rolling window performance."""
        window = window_years * TRADING_DAYS_YEAR
        df = self.df.copy()
        
        results = []
        for i in range(window, len(df), 63):  # Quarterly steps
            end_idx = i
            start_idx = i - window
            
            subset = df.iloc[start_idx:end_idx]
            result = self.run_backtest(
                strategy=strategy,
                start_date=str(subset.index[0].date()),
                end_date=str(subset.index[-1].date())
            )
            results.append({
                'end_date': str(subset.index[-1].date()),
                'sharpe': result.sharpe_ratio,
                'cagr': result.cagr,
                'max_dd': result.max_drawdown,
                'volatility': result.volatility
            })
        
        return pd.DataFrame(results)


def main():
    """Example usage."""
    bt = PortfolioBacktester()
    
    # Single strategy backtest
    print("Running backtest: dvi_adaptive")
    result = bt.run_backtest("dvi_adaptive")
    print(result.summary())
    
    # Compare all strategies
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    comparison = bt.compare_strategies()
    print(comparison[['total_return', 'cagr', 'sharpe_ratio', 'max_drawdown']].to_string())
    
    # Save results
    with open("backtest_results.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print("\n✅ Results saved to backtest_results.json")


if __name__ == "__main__":
    main()
