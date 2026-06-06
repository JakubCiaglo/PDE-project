"""Numerical diagnostics for project experiments."""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from .black_scholes import black_scholes_price
from .config import BlackScholesParams, GridParams
from .finite_difference import finite_difference_price, solve_crank_nicolson_fd
from .market import MarketOptionQuote

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


def market_comparison_table(
    quote: MarketOptionQuote,
    volatility_cases: dict[str, float],
    rate: float = 0.0,
    grid: GridParams | None = None,
) -> pd.DataFrame:
    """Compare a market option quote with exact and finite difference prices."""

    quote.validate()
    if not volatility_cases:
        raise ValueError("volatility_cases cannot be empty")

    grid = grid or GridParams(
        s_max=max(2.0 * quote.strike, 1.5 * quote.underlying_price),
        asset_steps=400,
        time_steps=400,
    )

    rows = []
    for source, volatility in volatility_cases.items():
        volatility = float(volatility)
        if not np.isfinite(volatility) or volatility < 0:
            raise ValueError("volatility values must be finite and non-negative")

        params = BlackScholesParams(
            strike=quote.strike,
            maturity=quote.maturity_years,
            rate=rate,
            volatility=volatility,
            option_type=quote.option_type,
        ).normalized()

        bs_price = float(np.asarray(black_scholes_price(quote.underlying_price, params)).item())
        fd_price = finite_difference_price(quote.underlying_price, params, grid)

        rows.append(
            {
                "symbol": quote.symbol,
                "option_type": params.option_type,
                "underlying_price": quote.underlying_price,
                "strike": params.strike,
                "maturity_years": params.maturity,
                "rate": params.rate,
                "volatility_source": str(source),
                "volatility": params.volatility,
                "market_price": quote.market_price,
                "bs_price": bs_price,
                "fd_price": fd_price,
                "model_minus_market": bs_price - quote.market_price,
                "fd_minus_bs": fd_price - bs_price,
            }
        )

    return pd.DataFrame(rows)


def market_chain_comparison_table(
    snapshot: pd.DataFrame,
    volatility_columns: tuple[str, ...] = ("mark_iv", "historical_volatility"),
    fd_symbols: set[str] | list[str] | tuple[str, ...] | None = None,
    grid: GridParams | None = None,
) -> pd.DataFrame:
    """Compare many option quotes with Black-Scholes model prices."""

    required = {
        "symbol",
        "option_type",
        "underlying_price",
        "strike",
        "maturity_years",
        "market_price",
    }
    missing = required.difference(snapshot.columns)
    if missing:
        raise ValueError(f"snapshot is missing required columns: {sorted(missing)}")

    for column in volatility_columns:
        if column not in snapshot.columns:
            raise ValueError(f"snapshot is missing volatility column: {column}")

    fd_symbols = set(fd_symbols or [])
    rows = []
    for _, quote_row in snapshot.iterrows():
        quote = MarketOptionQuote(
            symbol=quote_row["symbol"],
            underlying_price=float(quote_row["underlying_price"]),
            strike=float(quote_row["strike"]),
            maturity_years=float(quote_row["maturity_years"]),
            market_price=float(quote_row["market_price"]),
            option_type=quote_row["option_type"],
        )
        quote.validate()
        rate_value = quote_row.get("risk_free_rate", 0.0)
        rate = 0.0 if pd.isna(rate_value) else float(rate_value)
        days_to_maturity = float(quote_row.get("maturity_days", quote.maturity_years * 365.0))
        moneyness = float(quote_row.get("moneyness", quote.underlying_price / quote.strike))
        bid_price = quote_row.get("bid_price", np.nan)
        ask_price = quote_row.get("ask_price", np.nan)
        bid_ask_spread = np.nan
        if not pd.isna(bid_price) and not pd.isna(ask_price):
            bid_ask_spread = float(ask_price) - float(bid_price)

        for volatility_column in volatility_columns:
            volatility = float(quote_row[volatility_column])
            if not np.isfinite(volatility) or volatility < 0:
                continue

            params = BlackScholesParams(
                strike=quote.strike,
                maturity=quote.maturity_years,
                rate=rate,
                volatility=volatility,
                option_type=quote.option_type,
            ).normalized()
            bs_price = float(np.asarray(black_scholes_price(quote.underlying_price, params)).item())
            fd_price = np.nan
            if quote.symbol in fd_symbols:
                fd_grid = grid or GridParams(
                    s_max=max(2.0 * quote.strike, 1.5 * quote.underlying_price),
                    asset_steps=800,
                    time_steps=400,
                )
                fd_price = finite_difference_price(quote.underlying_price, params, fd_grid)

            model_minus_market = bs_price - quote.market_price
            relative_model_error = np.nan
            if quote.market_price != 0:
                relative_model_error = model_minus_market / quote.market_price

            rows.append(
                {
                    "symbol": quote.symbol,
                    "option_type": params.option_type,
                    "expiry_utc": quote_row.get("expiry_utc", ""),
                    "underlying_price": quote.underlying_price,
                    "strike": params.strike,
                    "maturity_days": days_to_maturity,
                    "maturity_years": params.maturity,
                    "moneyness": moneyness,
                    "rate": params.rate,
                    "volatility_source": volatility_column,
                    "volatility": params.volatility,
                    "market_price": quote.market_price,
                    "bid_price": bid_price,
                    "ask_price": ask_price,
                    "bid_ask_spread": bid_ask_spread,
                    "bs_price": bs_price,
                    "fd_price": fd_price,
                    "model_minus_market": model_minus_market,
                    "absolute_model_error": abs(model_minus_market),
                    "relative_model_error": relative_model_error,
                    "fd_minus_bs": fd_price - bs_price,
                }
            )

    return pd.DataFrame(rows)
