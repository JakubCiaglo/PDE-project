"""Sanity checks for analytical and finite difference pricing."""

import numpy as np

from black_scholes_pde import (
    BlackScholesParams,
    GridParams,
    black_scholes_price,
    convergence_table,
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
