"""Analytical Black-Scholes formulas."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .config import BlackScholesParams


def payoff(asset_prices: np.ndarray | float, strike: float, option_type: str = "call") -> np.ndarray:
    """Return the European option payoff at maturity."""

    prices = np.asarray(asset_prices, dtype=float)
    if option_type == "call":
        return np.maximum(prices - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - prices, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_price(
    asset_price: np.ndarray | float,
    params: BlackScholesParams,
    time: float = 0.0,
) -> np.ndarray:
    """Compute the exact Black-Scholes price for a European call or put."""

    params.validate()
    s = np.asarray(asset_price, dtype=float)
    tau = params.maturity - time

    if tau <= 0 or params.volatility == 0:
        intrinsic = payoff(s, params.strike, params.option_type)
        return intrinsic.astype(float)

    positive_s = np.maximum(s, np.finfo(float).tiny)
    sqrt_tau = np.sqrt(tau)
    d1 = (
        np.log(positive_s / params.strike)
        + (params.rate + 0.5 * params.volatility**2) * tau
    ) / (params.volatility * sqrt_tau)
    d2 = d1 - params.volatility * sqrt_tau
    discount = np.exp(-params.rate * tau)

    if params.option_type == "call":
        price = s * norm.cdf(d1) - params.strike * discount * norm.cdf(d2)
    elif params.option_type == "put":
        price = params.strike * discount * norm.cdf(-d2) - s * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return np.maximum(price, 0.0)
