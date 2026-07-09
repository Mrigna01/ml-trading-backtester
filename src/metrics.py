import numpy as np

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

def max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()

def win_rate(returns):
    return (returns > 0).mean()