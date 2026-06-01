"""Plotting helpers used by notebooks."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.dont_write_bytecode = True


def plot_price_comparison(
    asset_grid: np.ndarray,
    numerical: np.ndarray,
    exact: np.ndarray,
    title: str = "Black-Scholes: exact vs finite difference",
):
    """Plot analytical and finite difference option prices."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(asset_grid, exact, label="Exact Black-Scholes", linewidth=2)
    ax.plot(asset_grid, numerical, "--", label="Finite difference", linewidth=2)
    ax.set_xlabel("Underlying asset price S")
    ax.set_ylabel("Option value V(S, 0)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_error(asset_grid: np.ndarray, numerical: np.ndarray, exact: np.ndarray):
    """Plot absolute pricing error on the asset grid."""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(asset_grid, np.abs(numerical - exact), color="tab:red")
    ax.set_xlabel("Underlying asset price S")
    ax.set_ylabel("Absolute error")
    ax.set_title("Finite difference absolute error")
    ax.grid(True, alpha=0.3)
    return fig, ax
