# utils.py
from decimal import Decimal, getcontext
import numpy as np, pandas as pd
getcontext().prec = 12

def compute_metrics_from_pv(pv_series):
    """
    pv_series: pandas Series of portfolio values indexed by datetime
    Returns dict with mean_r, sharpe, sortino, calmar, max_drawdown
    """
    returns = pv_series.pct_change().dropna()
    mean_r = returns.mean()
    sigma_p = returns.std(ddof=0)
    sigma_d = returns[returns < 0].std(ddof=0) if any(returns < 0) else 0.0
    cummax = pv_series.cummax()
    draw = (pv_series - cummax) / cummax
    max_dd = draw.min()
    sharpe = mean_r / sigma_p if sigma_p > 0 else float('nan')
    sortino = mean_r / sigma_d if sigma_d > 0 else float('nan')
    calmar = mean_r / abs(max_dd) if max_dd < 0 else float('nan')
    return {
        "mean_r": float(mean_r),
        "sharpe": float(sharpe) if not pd.isna(sharpe) else None,
        "sortino": float(sortino) if not pd.isna(sortino) else None,
        "calmar": float(calmar) if not pd.isna(calmar) else None,
        "max_dd": float(max_dd)
    }
