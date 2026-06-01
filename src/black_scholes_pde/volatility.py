"""Helpers for estimating volatility from historical prices."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True


def log_returns(prices: pd.Series | np.ndarray) -> pd.Series:
    """Compute logarithmic returns from a price series."""

    series = pd.Series(prices, dtype=float).dropna()
    return np.log(series / series.shift(1)).dropna()


def annualized_volatility(
    prices: pd.Series | np.ndarray,
    periods_per_year: int = 365,
) -> float:
    """Estimate annualized volatility from historical prices."""

    returns = log_returns(prices)
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))
