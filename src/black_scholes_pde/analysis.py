"""Numerical diagnostics for project experiments."""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from .black_scholes import black_scholes_price
from .config import BlackScholesParams, GridParams
from .finite_difference import solve_crank_nicolson_fd

sys.dont_write_bytecode = True

FiniteDifferenceSolver = Callable[
    [BlackScholesParams, GridParams],
    tuple[np.ndarray, np.ndarray],
]


def error_metrics(numerical: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    """Return maximum absolute error, MAE, and RMSE."""

    error = np.asarray(numerical, dtype=float) - np.asarray(exact, dtype=float)
    absolute = np.abs(error)
    return {
        "max_abs_error": float(np.max(absolute)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def comparison_table(
    asset_points: np.ndarray | list[float],
    params: BlackScholesParams,
    grid: GridParams,
    solver: FiniteDifferenceSolver = solve_crank_nicolson_fd,
) -> pd.DataFrame:
    """Compare exact and interpolated numerical prices at selected asset prices."""

    points = np.asarray(asset_points, dtype=float)
    asset_grid, numerical = solver(params, grid)
    exact = black_scholes_price(points, params)
    interpolated = np.interp(points, asset_grid, numerical)
    return pd.DataFrame(
        {
            "asset_price": points,
            "exact_price": exact,
            "numerical_price": interpolated,
            "absolute_error": np.abs(interpolated - exact),
        }
    )


def convergence_table(
    params: BlackScholesParams,
    grid_sizes: list[tuple[int, int]],
    s_max: float = 200.0,
    solver: FiniteDifferenceSolver = solve_crank_nicolson_fd,
) -> pd.DataFrame:
    """Compute finite difference errors for several grid sizes."""

    rows = []
    for asset_steps, time_steps in grid_sizes:
        grid = GridParams(s_max=s_max, asset_steps=asset_steps, time_steps=time_steps)
        asset_grid, numerical = solver(params, grid)
        exact = black_scholes_price(asset_grid, params)
        rows.append(
            {
                "asset_steps": asset_steps,
                "time_steps": time_steps,
                **error_metrics(numerical, exact),
            }
        )

    return pd.DataFrame(rows)


def parameter_sensitivity(
    asset_grid: np.ndarray,
    params: BlackScholesParams,
    parameter_name: str,
    parameter_values: list[float],
) -> dict[str, np.ndarray]:
    """Return exact Black-Scholes curves for selected model parameter values."""

    if parameter_name not in {"volatility", "rate", "strike", "maturity"}:
        raise ValueError("parameter_name must be volatility, rate, strike, or maturity")

    return {
        f"{parameter_name}={value:g}": black_scholes_price(
            asset_grid,
            replace(params, **{parameter_name: value}),
        )
        for value in parameter_values
    }
