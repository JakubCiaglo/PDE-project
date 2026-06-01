"""Finite difference solver for the Black-Scholes PDE."""

from __future__ import annotations

import sys

import numpy as np

from .black_scholes import payoff
from .config import BlackScholesParams, GridParams

sys.dont_write_bytecode = True


def _discount(rate: float, tau: float) -> float:
    return float(np.exp(np.clip(-rate * tau, -700.0, 700.0)))


def _boundary_values(tau: float, params: BlackScholesParams, s_max: float) -> tuple[float, float]:
    discount = _discount(params.rate, tau)
    if params.option_type == "call":
        return 0.0, max(s_max - params.strike * discount, 0.0)
    if params.option_type == "put":
        return params.strike * discount, 0.0
    raise ValueError("option_type must be 'call' or 'put'")


def solve_explicit_fd(
    params: BlackScholesParams,
    grid: GridParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the Black-Scholes PDE on a finite difference grid.

    The PDE is solved backward in calendar time by introducing
    ``tau = T - t``. The returned values approximate ``V(S, 0)`` on the
    asset-price grid.
    """

    params = params.normalized()
    grid = grid.normalized(minimum_s_max=2.0 * params.strike)

    d_tau = params.maturity / grid.time_steps
    asset_grid = np.linspace(0.0, grid.s_max, grid.asset_steps + 1)
    values = payoff(asset_grid, params.strike, params.option_type)

    if params.maturity == 0 or params.volatility == 0:
        from .black_scholes import black_scholes_price

        return asset_grid, black_scholes_price(asset_grid, params)

    i = np.arange(1, grid.asset_steps)
    sigma2_i2 = (params.volatility**2) * (i**2)
    lower = -0.5 * d_tau * (sigma2_i2 - params.rate * i)
    diagonal = 1.0 + d_tau * (sigma2_i2 + params.rate)
    upper = -0.5 * d_tau * (sigma2_i2 + params.rate * i)

    for step in range(1, grid.time_steps + 1):
        left_boundary, right_boundary = _boundary_values(step * d_tau, params, grid.s_max)
        rhs = values[1:-1].copy()
        rhs[0] -= lower[0] * left_boundary
        rhs[-1] -= upper[-1] * right_boundary

        values[1:-1] = _solve_tridiagonal(lower[1:], diagonal, upper[:-1], rhs)
        values[0], values[-1] = left_boundary, right_boundary
        values = np.maximum(np.nan_to_num(values, nan=0.0, posinf=1e308, neginf=0.0), 0.0)

    return asset_grid, values


def _solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system with the Thomas algorithm."""

    diag = diagonal.astype(float).copy()
    upper_work = upper.astype(float).copy()
    solution = rhs.astype(float).copy()

    for idx in range(1, diag.size):
        pivot = diag[idx - 1]
        if abs(pivot) < np.finfo(float).eps:
            pivot = np.copysign(np.finfo(float).eps, pivot or 1.0)
        factor = lower[idx - 1] / pivot
        diag[idx] -= factor * upper_work[idx - 1]
        solution[idx] -= factor * solution[idx - 1]

    if abs(diag[-1]) < np.finfo(float).eps:
        diag[-1] = np.copysign(np.finfo(float).eps, diag[-1] or 1.0)
    solution[-1] /= diag[-1]

    for idx in range(diag.size - 2, -1, -1):
        pivot = diag[idx]
        if abs(pivot) < np.finfo(float).eps:
            pivot = np.copysign(np.finfo(float).eps, pivot or 1.0)
        solution[idx] = (solution[idx] - upper_work[idx] * solution[idx + 1]) / pivot

    return solution


def finite_difference_price(
    asset_price: float,
    params: BlackScholesParams,
    grid: GridParams,
) -> float:
    """Interpolate a finite difference price for one asset price."""

    params = params.normalized()
    asset_price = float(asset_price)
    if not np.isfinite(asset_price):
        asset_price = max(grid.s_max, 2.0 * params.strike) if asset_price > 0 else 0.0
    asset_price = max(asset_price, 0.0)
    grid = grid.normalized(minimum_s_max=max(2.0 * params.strike, 1.25 * asset_price))

    asset_grid, values = solve_explicit_fd(params, grid)
    return float(np.interp(asset_price, asset_grid, values))
