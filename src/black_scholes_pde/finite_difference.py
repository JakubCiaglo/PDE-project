"""Finite difference solver for the Black-Scholes PDE."""

from __future__ import annotations

import numpy as np

from .black_scholes import payoff
from .config import BlackScholesParams, GridParams


def _boundary_values(tau: float, params: BlackScholesParams, s_max: float) -> tuple[float, float]:
    discount = np.exp(-params.rate * tau)
    if params.option_type == "call":
        return 0.0, max(s_max - params.strike * discount, 0.0)
    if params.option_type == "put":
        return params.strike * discount, 0.0
    raise ValueError("option_type must be 'call' or 'put'")


def solve_explicit_fd(
    params: BlackScholesParams,
    grid: GridParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the Black-Scholes PDE with an explicit finite difference scheme."""

    params.validate()
    grid.validate()

    d_tau = params.maturity / grid.time_steps
    asset_grid = np.linspace(0.0, grid.s_max, grid.asset_steps + 1)
    values = payoff(asset_grid, params.strike, params.option_type)

    i = np.arange(1, grid.asset_steps)
    sigma2_i2 = (params.volatility**2) * (i**2)
    a = 0.5 * d_tau * (sigma2_i2 - params.rate * i)
    b = 1.0 - d_tau * (sigma2_i2 + params.rate)
    c = 0.5 * d_tau * (sigma2_i2 + params.rate * i)

    for step in range(1, grid.time_steps + 1):
        previous = values.copy()
        values[1:-1] = a * previous[:-2] + b * previous[1:-1] + c * previous[2:]
        values[0], values[-1] = _boundary_values(step * d_tau, params, grid.s_max)

    return asset_grid, values


def finite_difference_price(
    asset_price: float,
    params: BlackScholesParams,
    grid: GridParams,
) -> float:
    """Interpolate a finite difference price for one asset price."""

    asset_grid, values = solve_explicit_fd(params, grid)
    return float(np.interp(asset_price, asset_grid, values))
