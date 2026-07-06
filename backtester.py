import numpy as np
import pandas as pd

class Backtester:
    def __init__(self, initial_capital=100000.0, risk_free_rate=0.02):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

    def run(self, df: pd.DataFrame, signals_series: pd.Series):
        """
        Runs a vectorized backtest given price data and binary signals (1 for long, 0 for cash).
        Assumes we buy at the open of the day after the signal is generated.
        Returns a dict of metrics and a DataFrame with equity curve.
        """
        # Align signals with prices. Shift signals by 1 to prevent look-ahead bias
        # (signal calculated on day t applies to day t+1 return)
        signals = signals_series.shift(1).fillna(0)
        
        # Calculate daily returns of the asset (Buy & Hold return)
        daily_asset_returns = df['close'].pct_change().fillna(0)
        
        # Calculate strategy returns
        # When signal is 1, we earn the daily_asset_returns
        strategy_returns = daily_asset_returns * signals
        
        # Calculate equity curves
        strategy_equity = self.initial_capital * (1 + strategy_returns).cumprod()
        buy_hold_equity = self.initial_capital * (1 + daily_asset_returns).cumprod()
        
        # Track trades
        # Diff of signals: 1 means buy, -1 means sell
        trades = signals.diff().fillna(0)
        
        # Log discrete trades
        trades_log = []
        entry_date = None
        entry_price = 0
        
        for date, row in trades.items():
            if row == 1.0:
                entry_date = date
                entry_price = df.loc[date, 'close']
            elif row == -1.0 and entry_date is not None:
                exit_date = date
                exit_price = df.loc[date, 'close']
                trade_return = (exit_price - entry_price) / entry_price
                trades_log.append({
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'exit_date': exit_date.strftime('%Y-%m-%d'),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return
                })
                entry_date = None
        
        strategy_metrics = self._calculate_metrics(strategy_returns, strategy_equity)
        buy_hold_metrics = self._calculate_metrics(daily_asset_returns, buy_hold_equity)
        
        result_df = pd.DataFrame({
            'asset_return': daily_asset_returns,
            'strategy_return': strategy_returns,
            'strategy_equity': strategy_equity,
            'buy_hold_equity': buy_hold_equity,
            'signal': signals,
            'trades': trades
        })
        result_df.attrs['trades_log'] = trades_log
        
        return strategy_metrics, buy_hold_metrics, result_df

    def _calculate_metrics(self, returns: pd.Series, equity_curve: pd.Series):
        """Calculates 7 core performance metrics."""
        # Total Return
        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1
        
        # CAGR (Compound Annual Growth Rate)
        years = len(returns) / 252
        cagr = ((1 + total_return) ** (1 / years) - 1) if years > 0 and (1 + total_return) > 0 else 0
        
        # Annualized Volatility
        ann_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        if excess_returns.std() == 0:
            sharpe = 0
        else:
            sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
        # Sortino Ratio
        downside_returns = excess_returns[excess_returns < 0]
        if downside_returns.std() == 0:
            sortino = 0
        else:
            sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
            
        # Maximum Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win Rate (Percentage of days with positive return when invested)
        invested_days = returns[returns != 0]
        if len(invested_days) > 0:
            win_rate = len(invested_days[invested_days > 0]) / len(invested_days)
        else:
            win_rate = 0
            
        return {
            'Total Return (%)': round(total_return * 100, 2),
            'CAGR (%)': round(cagr * 100, 2),
            'Annualized Volatility (%)': round(ann_volatility * 100, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Sortino Ratio': round(sortino, 2),
            'Max Drawdown (%)': round(max_drawdown * 100, 2),
            'Win Rate (%)': round(win_rate * 100, 2)
        }
