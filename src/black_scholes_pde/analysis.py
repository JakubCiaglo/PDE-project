"""Numerical diagnostics for project experiments."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .black_scholes import black_scholes_price
from .config import BlackScholesParams, GridParams
from .finite_difference import solve_explicit_fd

sys.dont_write_bytecode = True


def error_metrics(numerical: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    """Return max, MAE and RMSE errors."""

    error = np.asarray(numerical, dtype=float) - np.asarray(exact, dtype=float)
    absolute = np.abs(error)
    return {
        "max_abs_error": float(np.max(absolute)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def convergence_table(
    params: BlackScholesParams,
    grid_sizes: list[tuple[int, int]],
    s_max: float = 200.0,
) -> pd.DataFrame:
    """Compute finite difference errors for several grid sizes."""

    rows = []
    for asset_steps, time_steps in grid_sizes:
        grid = GridParams(s_max=s_max, asset_steps=asset_steps, time_steps=time_steps)
        asset_grid, numerical = solve_explicit_fd(params, grid)
        exact = black_scholes_price(asset_grid, params)
        metrics = error_metrics(numerical, exact)
        rows.append(
            {
                "asset_steps": asset_steps,
                "time_steps": time_steps,
                **metrics,
            }
        )

    return pd.DataFrame(rows)
