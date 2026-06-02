"""Finite difference solvers for the Black-Scholes PDE."""

from __future__ import annotations

import sys

import numpy as np

from .black_scholes import payoff
from .config import BlackScholesParams, GridParams

sys.dont_write_bytecode = True


def _discount(rate: float, theta: float) -> float:
    return float(np.exp(np.clip(-rate * theta, -700.0, 700.0)))


def _boundary_values(
    theta: float,
    params: BlackScholesParams,
    s_max: float,
) -> tuple[float, float]:
    discount = _discount(params.rate, theta)
    if params.option_type == "call":
        return 0.0, s_max - params.strike * discount
    if params.option_type == "put":
        return params.strike * discount, 0.0
    raise ValueError("option_type must be 'call' or 'put'")


def _prepare_grid(
    params: BlackScholesParams,
    grid: GridParams,
) -> tuple[BlackScholesParams, GridParams, np.ndarray, np.ndarray, np.ndarray]:
    params = params.normalized()
    grid = grid.normalized(minimum_s_max=2.0 * params.strike)
    asset_grid = np.linspace(0.0, grid.s_max, grid.asset_steps + 1)
    theta_grid = np.linspace(0.0, params.maturity, grid.time_steps + 1)
    values = payoff(asset_grid, params.strike, params.option_type)
    values[0], values[-1] = _boundary_values(0.0, params, grid.s_max)
    return params, grid, asset_grid, theta_grid, values


def _spatial_coefficients(
    asset_grid: np.ndarray,
    d_s: float,
    params: BlackScholesParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = asset_grid[1:-1]
    diffusion = 0.5 * params.volatility**2 * s**2 / d_s**2
    convection = params.rate * s / (2.0 * d_s)
    a = diffusion - convection
    b = -2.0 * diffusion - params.rate
    c = diffusion + convection
    return a, b, c


def _solution_result(
    asset_grid: np.ndarray,
    theta_grid: np.ndarray,
    values: np.ndarray,
    solution: np.ndarray | None,
):
    if solution is None:
        return asset_grid, values
    return asset_grid, theta_grid, solution


def solve_explicit_fd(
    params: BlackScholesParams,
    grid: GridParams,
    return_full_grid: bool = False,
):
    """Solve the Black-Scholes PDE with explicit Euler in ``theta = T - t``.

    The explicit scheme uses central differences in the asset price ``s`` and
    is conditionally stable. It is included as a simple educational method;
    Crank-Nicolson is used for the main numerical results.
    """

    params, grid, asset_grid, theta_grid, values = _prepare_grid(params, grid)
    solution = np.empty((grid.time_steps + 1, grid.asset_steps + 1)) if return_full_grid else None
    if solution is not None:
        solution[0] = values

    d_theta = params.maturity / grid.time_steps
    d_s = grid.s_max / grid.asset_steps
    a, b, c = _spatial_coefficients(asset_grid, d_s, params)

    for step in range(1, grid.time_steps + 1):
        previous = values.copy()
        values[1:-1] = previous[1:-1] + d_theta * (
            a * previous[:-2] + b * previous[1:-1] + c * previous[2:]
        )
        values[0], values[-1] = _boundary_values(theta_grid[step], params, grid.s_max)
        if solution is not None:
            solution[step] = values

    return _solution_result(asset_grid, theta_grid, values, solution)


def solve_implicit_fd(
    params: BlackScholesParams,
    grid: GridParams,
    return_full_grid: bool = False,
):
    """Solve the Black-Scholes PDE with implicit (backward) Euler in ``theta``."""

    params, grid, asset_grid, theta_grid, values = _prepare_grid(params, grid)
    solution = np.empty((grid.time_steps + 1, grid.asset_steps + 1)) if return_full_grid else None
    if solution is not None:
        solution[0] = values

    d_theta = params.maturity / grid.time_steps
    d_s = grid.s_max / grid.asset_steps
    a, b, c = _spatial_coefficients(asset_grid, d_s, params)
    lower = -d_theta * a
    diagonal = 1.0 - d_theta * b
    upper = -d_theta * c

    for step in range(1, grid.time_steps + 1):
        left_boundary, right_boundary = _boundary_values(theta_grid[step], params, grid.s_max)
        rhs = values[1:-1].copy()
        rhs[0] -= lower[0] * left_boundary
        rhs[-1] -= upper[-1] * right_boundary

        values[1:-1] = _solve_tridiagonal(lower[1:], diagonal, upper[:-1], rhs)
        values[0], values[-1] = left_boundary, right_boundary
        if solution is not None:
            solution[step] = values

    return _solution_result(asset_grid, theta_grid, values, solution)


def solve_crank_nicolson_fd(
    params: BlackScholesParams,
    grid: GridParams,
    return_full_grid: bool = False,
):
    """Solve the Black-Scholes PDE with the Crank-Nicolson scheme."""

    params, grid, asset_grid, theta_grid, values = _prepare_grid(params, grid)
    solution = np.empty((grid.time_steps + 1, grid.asset_steps + 1)) if return_full_grid else None
    if solution is not None:
        solution[0] = values

    d_theta = params.maturity / grid.time_steps
    d_s = grid.s_max / grid.asset_steps
    a, b, c = _spatial_coefficients(asset_grid, d_s, params)
    half_step = 0.5 * d_theta
    lower = -half_step * a
    diagonal = 1.0 - half_step * b
    upper = -half_step * c

    for step in range(1, grid.time_steps + 1):
        left_boundary, right_boundary = _boundary_values(theta_grid[step], params, grid.s_max)
        rhs = (
            half_step * a * values[:-2]
            + (1.0 + half_step * b) * values[1:-1]
            + half_step * c * values[2:]
        )
        rhs[0] -= lower[0] * left_boundary
        rhs[-1] -= upper[-1] * right_boundary

        values[1:-1] = _solve_tridiagonal(lower[1:], diagonal, upper[:-1], rhs)
        values[0], values[-1] = left_boundary, right_boundary
        if solution is not None:
            solution[step] = values

    return _solution_result(asset_grid, theta_grid, values, solution)


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
    """Interpolate a Crank-Nicolson finite difference price for one asset price."""

    params = params.normalized()
    asset_price = float(asset_price)
    if not np.isfinite(asset_price):
        asset_price = max(grid.s_max, 2.0 * params.strike) if asset_price > 0 else 0.0
    asset_price = max(asset_price, 0.0)
    grid = grid.normalized(minimum_s_max=max(2.0 * params.strike, 1.25 * asset_price))

    asset_grid, values = solve_crank_nicolson_fd(params, grid)
    return float(np.interp(asset_price, asset_grid, values))
