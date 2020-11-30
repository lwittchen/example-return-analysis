import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 


def simulate_random_walk(
    start_price: float, start_date: pd.Timestamp, end_date: pd.Timestamp, trend=0.1
) -> pd.Series:
    """
    Simulate Random Walk (with i.i.d. increments) from start_date to end_date
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="d")
    returns = np.random.standard_normal(size=len(dates)) + trend
    prices = np.cumsum(returns) + start_price
    return pd.Series(prices, index=dates)

def calc_log_returns(prices: pd.Series) -> pd.Series():
    """
    Logarithmic returns
    """
    return pd.Series(np.diff(np.log(prices)), index=prices.index[1:])

def calc_max_drawdown(prices: pd.Series) -> float:
    """
    Total maximum drawdown
    """
    return (prices - prices.cummax()).min() 

def calc_total_return(prices: pd.Series) -> float: 
    """
    Percentage return from start to end
    """
    return (prices[-1] - prices[0]) / prices[0]

def calc_value_at_risk(returns: pd.Series, alpha=0.95, hist=True) -> float:
    """
    Value at risk in %
    """
    if hist: 
        return returns.quantile(q=1-alpha)
    else: 
        mu = returns.mean()
        sigma = returns.std()
        return norm.ppf(1-alpha, mu, sigma)

def calc_sharpe_ratio(returns: pd.Series) -> float:
    """
    Calc sharpe ratio
    """
    return returns.mean() / returns.std()


_start_price = 100
_start_date = pd.Timestamp(2020, 1, 1)
_end_date = pd.Timestamp(2020, 11, 30)

prices = simulate_random_walk(
    start_price=_start_price, start_date=_start_date, end_date=_end_date
)
log_returns = calc_log_returns(prices=prices)
normal_returns = prices.pct_change()

"""
Naive Stategy: go long first day of month, close position 15th day
"""
position = pd.Series(
    np.where(prices.index.day == 1, 1, 0) + np.where(prices.index.day == 15, -1, 0), 
    index=prices.index
)

# strategy metrics
strat_returns = position.cumsum() * log_returns
var = calc_value_at_risk(strat_returns)
mdd = calc_max_drawdown(strat_returns.cumsum())
sharpe = calc_sharpe_ratio(strat_returns)

# results
print(f'Total returns: {strat_returns.sum():.2f}')
print(f'Sharpe Ratio: {sharpe:.2f}')
print(f'Value at Risk: {var:.2f}')
print(f'Maximum Drawdown: {mdd:.2f}')