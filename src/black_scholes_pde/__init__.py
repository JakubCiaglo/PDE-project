"""Utilities for the Black-Scholes PDE project."""

import sys

sys.dont_write_bytecode = True

from .analysis import comparison_table, convergence_table, error_metrics, parameter_sensitivity
from .black_scholes import black_scholes_price, payoff
from .config import BlackScholesParams, GridParams
from .finite_difference import (
    finite_difference_price,
    solve_crank_nicolson_fd,
    solve_explicit_fd,
    solve_implicit_fd,
)
from .market import MarketOptionQuote
from .volatility import annualized_volatility, log_returns

__all__ = [
    "BlackScholesParams",
    "GridParams",
    "MarketOptionQuote",
    "annualized_volatility",
    "black_scholes_price",
    "comparison_table",
    "convergence_table",
    "error_metrics",
    "finite_difference_price",
    "log_returns",
    "payoff",
    "parameter_sensitivity",
    "solve_crank_nicolson_fd",
    "solve_explicit_fd",
    "solve_implicit_fd",
]
