"""Plotting helpers used by notebooks."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True


def plot_payoff_vs_exact(
    asset_grid: np.ndarray,
    exact: np.ndarray,
    payoff_values: np.ndarray,
    title: str | None = None,
):
    """Plot the payoff at maturity and the exact price at time zero."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(asset_grid, payoff_values, "--", label="Payoff at maturity", linewidth=2)
    ax.plot(asset_grid, exact, label="Exact Black-Scholes price", linewidth=2)
    ax.set_xlabel("Underlying asset price s")
    ax.set_ylabel("Payoff $\\Phi(s)$ / option value $V(0,s)$")
    ax.set_title(title or "Payoff and exact Black-Scholes price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_price_comparison(
    asset_grid: np.ndarray,
    numerical: np.ndarray,
    exact: np.ndarray,
    title: str | None = None,
):
    """Plot analytical and finite difference option prices."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(asset_grid, exact, label="Exact Black-Scholes", linewidth=2)
    ax.plot(asset_grid, numerical, "--", label="Crank-Nicolson", linewidth=2)
    ax.set_xlabel("Underlying asset price s")
    ax.set_ylabel("Option value V(0,s)")
    ax.set_title(title or "Black-Scholes: exact vs Crank-Nicolson")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_error(
    asset_grid: np.ndarray,
    numerical: np.ndarray,
    exact: np.ndarray,
    title: str | None = None,
):
    """Plot absolute pricing error on the asset grid."""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(asset_grid, np.abs(numerical - exact))
    ax.set_xlabel("Underlying asset price s")
    ax.set_ylabel("Absolute error")
    ax.set_title(title or "Crank-Nicolson absolute error")
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_convergence(
    convergence_df: pd.DataFrame,
    metric: str = "max_abs_error",
    title: str | None = None,
):
    """Plot a selected error metric against the number of asset grid steps."""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(convergence_df["asset_steps"], convergence_df[metric], marker="o")
    ax.set_xlabel("Number of asset price steps")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or "Grid convergence")
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_parameter_sensitivity(
    asset_grid: np.ndarray,
    price_curves: dict[str, np.ndarray],
    parameter_name: str,
    title: str | None = None,
):
    """Plot option price curves for several values of one model parameter."""

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in price_curves.items():
        ax.plot(asset_grid, values, label=label)
    ax.set_xlabel("Underlying asset price s")
    ax.set_ylabel("Option value V(0,s)")
    ax.set_title(title or f"Sensitivity to {parameter_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax
