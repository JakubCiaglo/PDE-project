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


def close_to_close_volatility(
    close: pd.Series | np.ndarray,
    periods_per_year: int = 365,
) -> float:
    """Estimate annualized volatility from close-to-close log returns."""

    return annualized_volatility(close, periods_per_year=periods_per_year)


def _ohlc_frame(
    open_prices: pd.Series | np.ndarray,
    high_prices: pd.Series | np.ndarray,
    low_prices: pd.Series | np.ndarray,
    close_prices: pd.Series | np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "open": pd.Series(open_prices, dtype=float),
            "high": pd.Series(high_prices, dtype=float),
            "low": pd.Series(low_prices, dtype=float),
            "close": pd.Series(close_prices, dtype=float),
        }
    ).dropna()
    if frame.empty:
        raise ValueError("OHLC data cannot be empty")
    if (frame <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("High price must be at least open, low, and close")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Low price must be at most open, high, and close")
    return frame


def garman_klass_volatility(
    open_prices: pd.Series | np.ndarray,
    high_prices: pd.Series | np.ndarray,
    low_prices: pd.Series | np.ndarray,
    close_prices: pd.Series | np.ndarray,
    periods_per_year: int = 365,
) -> float:
    """Estimate annualized volatility with the Garman-Klass OHLC estimator."""

    frame = _ohlc_frame(open_prices, high_prices, low_prices, close_prices)
    high_low = np.log(frame["high"] / frame["low"])
    close_open = np.log(frame["close"] / frame["open"])
    variance = 0.5 * high_low**2 - (2.0 * np.log(2.0) - 1.0) * close_open**2
    return float(np.sqrt(max(variance.mean() * periods_per_year, 0.0)))


def rogers_satchell_volatility(
    open_prices: pd.Series | np.ndarray,
    high_prices: pd.Series | np.ndarray,
    low_prices: pd.Series | np.ndarray,
    close_prices: pd.Series | np.ndarray,
    periods_per_year: int = 365,
) -> float:
    """Estimate annualized volatility with the Rogers-Satchell OHLC estimator."""

    frame = _ohlc_frame(open_prices, high_prices, low_prices, close_prices)
    high_open = np.log(frame["high"] / frame["open"])
    high_close = np.log(frame["high"] / frame["close"])
    low_open = np.log(frame["low"] / frame["open"])
    low_close = np.log(frame["low"] / frame["close"])
    variance = high_open * high_close + low_open * low_close
    return float(np.sqrt(max(variance.mean() * periods_per_year, 0.0)))
