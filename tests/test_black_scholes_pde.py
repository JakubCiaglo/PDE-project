"""Sanity checks for analytical and finite difference pricing."""

import numpy as np
import pandas as pd

from black_scholes_pde import (
    BlackScholesParams,
    GridParams,
    MarketOptionQuote,
    black_scholes_price,
    close_to_close_volatility,
    convergence_table,
    garman_klass_volatility,
    market_chain_comparison_table,
    market_comparison_table,
    rogers_satchell_volatility,
    solve_crank_nicolson_fd,
)


def test_exact_formula_is_non_negative_and_satisfies_put_call_parity():
    asset_prices = np.array([80.0, 100.0, 120.0])
    call_params = BlackScholesParams(option_type="call")
    put_params = BlackScholesParams(option_type="put")

    calls = black_scholes_price(asset_prices, call_params)
    puts = black_scholes_price(asset_prices, put_params)
    parity = asset_prices - call_params.strike * np.exp(-call_params.rate * call_params.maturity)

    assert np.all(calls >= 0.0)
    assert np.all(puts >= 0.0)
    np.testing.assert_allclose(calls - puts, parity, atol=1e-10)


def test_crank_nicolson_solver_returns_accurate_finite_non_negative_prices():
    params = BlackScholesParams(option_type="call")
    grid = GridParams(s_max=200.0, asset_steps=200, time_steps=200)

    asset_grid, numerical = solve_crank_nicolson_fd(params, grid)
    exact = black_scholes_price(asset_grid, params)

    assert asset_grid.size == grid.asset_steps + 1
    assert np.all(np.isfinite(numerical))
    assert np.all(numerical >= -1e-12)
    assert np.max(np.abs(numerical - exact)) < 0.02


def test_crank_nicolson_full_grid_shape_and_initial_condition():
    params = BlackScholesParams(option_type="put")
    grid = GridParams(s_max=200.0, asset_steps=50, time_steps=25)

    asset_grid, theta_grid, solution = solve_crank_nicolson_fd(
        params,
        grid,
        return_full_grid=True,
    )

    assert asset_grid.size == grid.asset_steps + 1
    assert theta_grid.size == grid.time_steps + 1
    assert solution.shape == (grid.time_steps + 1, grid.asset_steps + 1)
    np.testing.assert_allclose(solution[0], np.maximum(params.strike - asset_grid, 0.0))


def test_crank_nicolson_error_decreases_when_grid_is_refined():
    params = BlackScholesParams(option_type="call")
    table = convergence_table(params, [(50, 50), (100, 100), (200, 200)])

    assert table.iloc[-1]["max_abs_error"] < table.iloc[0]["max_abs_error"]


def test_ohlc_volatility_estimators_return_finite_positive_values():
    open_prices = pd.Series([100.0, 102.0, 101.0, 105.0, 104.0])
    high_prices = pd.Series([103.0, 104.0, 106.0, 107.0, 108.0])
    low_prices = pd.Series([99.0, 100.0, 100.0, 103.0, 102.0])
    close_prices = pd.Series([102.0, 101.0, 105.0, 104.0, 107.0])

    estimates = [
        close_to_close_volatility(close_prices),
        garman_klass_volatility(open_prices, high_prices, low_prices, close_prices),
        rogers_satchell_volatility(open_prices, high_prices, low_prices, close_prices),
    ]

    assert all(np.isfinite(value) and value > 0.0 for value in estimates)


def test_market_comparison_table_compares_market_exact_and_fd_prices():
    quote = MarketOptionQuote(
        symbol="BTC-260626-100000-C",
        underlying_price=100.0,
        strike=100.0,
        maturity_years=0.25,
        market_price=8.0,
        option_type="call",
        implied_volatility=0.30,
    )
    grid = GridParams(s_max=200.0, asset_steps=120, time_steps=120)

    table = market_comparison_table(
        quote,
        volatility_cases={"market_iv": 0.30, "historical_volatility": 0.25},
        rate=0.05,
        grid=grid,
    )

    assert list(table["volatility_source"]) == ["market_iv", "historical_volatility"]
    assert np.all(np.isfinite(table[["market_price", "bs_price", "fd_price"]]))
    np.testing.assert_allclose(
        table["model_minus_market"],
        table["bs_price"] - table["market_price"],
    )
    assert np.max(np.abs(table["fd_minus_bs"])) < 0.2


def test_market_chain_comparison_table_handles_many_quotes_and_optional_fd():
    snapshot = pd.DataFrame(
        [
            {
                "symbol": "BTC-260626-100000-C",
                "option_type": "call",
                "underlying_price": 100.0,
                "strike": 100.0,
                "maturity_days": 30.0,
                "maturity_years": 30.0 / 365.0,
                "market_price": 5.0,
                "risk_free_rate": 0.05,
                "mark_iv": 0.30,
                "historical_volatility": 0.25,
            },
            {
                "symbol": "BTC-260626-100000-P",
                "option_type": "put",
                "underlying_price": 100.0,
                "strike": 100.0,
                "maturity_days": 30.0,
                "maturity_years": 30.0 / 365.0,
                "market_price": 4.0,
                "risk_free_rate": 0.05,
                "mark_iv": 0.32,
                "historical_volatility": 0.25,
            },
        ]
    )
    grid = GridParams(s_max=200.0, asset_steps=120, time_steps=120)

    table = market_chain_comparison_table(
        snapshot,
        fd_symbols={"BTC-260626-100000-C"},
        grid=grid,
    )

    assert table.shape[0] == 4
    assert set(table["volatility_source"]) == {"mark_iv", "historical_volatility"}
    assert table["bs_price"].notna().all()
    assert table.loc[table["symbol"] == "BTC-260626-100000-C", "fd_price"].notna().all()
    assert table.loc[table["symbol"] == "BTC-260626-100000-P", "fd_price"].isna().all()
