# Analytical and Numerical Methods for the Black-Scholes Equation in European Option Pricing

Project for **Partial Differential Equations with Applications in Physics and
Industry**.

The PDF report is the main project document. The notebook reproduces the
numerical experiments and generates the figures used in the report. Reusable
Python code lives in the `src/` package so the numerical methods remain easy to
test and reuse.

## Project Scope

The project:

1. derives and discusses the Black-Scholes PDE,
2. explains terminal and boundary conditions for European call and put options,
3. connects the Black-Scholes equation with the heat equation,
4. computes exact analytical Black-Scholes prices,
5. solves the original PDE in the time-to-maturity variable with finite
   differences,
6. validates numerical prices against the exact Black-Scholes formula,
7. performs grid convergence and parameter sensitivity analysis,
8. compares the model with a BTCUSDT option-chain market snapshot.

The main numerical method is the Crank-Nicolson scheme. Explicit Euler is kept
as a simple educational example and implicit Euler is available as an
additional solver. The BTCUSDT comparison uses saved CSV snapshots so the
notebook remains reproducible.

## Repository Structure

```text
.
|-- data/
|   |-- btcusdt_option_chain_snapshot.csv
|   |-- btcusdt_option_snapshot.csv
|   `-- btcusdt_spot_history.csv
|-- notebooks/
|   |-- main.ipynb           # reproduces experiments and report figures
|-- reports/
|   `-- figures/             # generated plots
|-- scripts/
|   `-- fetch_btcusdt_option_snapshot.py
|-- src/
|   `-- black_scholes_pde/
|       |-- analysis.py      # errors, comparisons, and convergence tables
|       |-- black_scholes.py # analytical formulas and payoffs
|       |-- config.py        # dataclasses for parameters and grids
|       |-- finite_difference.py
|       |-- market.py        # optional market quote container
|       |-- visualization.py # Matplotlib plotting helpers
|       `-- volatility.py    # optional historical volatility estimation
|-- tests/
|   `-- test_black_scholes_pde.py
|-- requirements.txt
`-- README.md
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Open the main notebook:

```powershell
jupyter notebook notebooks/main.ipynb
```

The notebook adds `src/` to `sys.path`, so imports work without an editable
install. For example:

```python
from black_scholes_pde import (
    BlackScholesParams,
    GridParams,
    solve_crank_nicolson_fd,
)
```

## BTCUSDT Market Snapshot

The BTCUSDT option comparison uses public Binance market-data endpoints only.
No Binance account, API key, API secret, `.env` file, deposit, or paid plan is
required. The script does not trade and does not access account data.

To refresh the data snapshot:

```bash
PYENV_VERSION=ml python scripts/fetch_btcusdt_option_snapshot.py
```

This writes:

```text
data/btcusdt_option_snapshot.csv
data/btcusdt_option_chain_snapshot.csv
data/btcusdt_spot_history.csv
```

By default, the script saves all complete BTCUSDT option quotes with maturities
up to 30 days. It also saves one nearest at-the-money contract as a small
single-option example. The saved option rows include Binance mark IV
(`mark_iv_binance`) and three historical volatility estimates:
`close_to_close`, `german_klass`, and `rogers_satchell`.

To change the chain horizon:

```bash
PYENV_VERSION=ml python scripts/fetch_btcusdt_option_snapshot.py --max-maturity-days 45
```

To choose the single-option example manually:

```bash
PYENV_VERSION=ml python scripts/fetch_btcusdt_option_snapshot.py --symbol BTC-260626-100000-C
```

If Binance public endpoints are unavailable from your network, manually export
the same fields to `data/btcusdt_option_chain_snapshot.csv` from the Binance
options page and rerun the notebook.

## Numerical Validation

`notebooks/main.ipynb` uses a European call option as the baseline example,
compares the Crank-Nicolson finite difference solution with the exact
Black-Scholes formula, reports error metrics, creates a comparison table, and
checks grid convergence. It also generates exact-formula sensitivity plots for
volatility, interest rate, strike, and maturity. The final BTCUSDT section then
compares a saved option-chain snapshot with exact Black-Scholes prices and uses
a small near-the-money subset to check the finite difference PDE price.
