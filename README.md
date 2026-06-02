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
7. performs grid convergence and parameter sensitivity analysis.

The main numerical method is the Crank-Nicolson scheme. Explicit Euler is kept
as a simple educational example and implicit Euler is available as an
additional solver. Real market data comparison is optional and is not part of
this update.

## Repository Structure

```text
.
|-- data/
|   |-- raw/                 # optional manually downloaded data
|   `-- processed/           # optional cleaned data
|-- notebooks/
|   |-- main.ipynb           # reproduces experiments and report figures
|-- reports/
|   `-- figures/             # generated plots
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

## Numerical Validation

`notebooks/main.ipynb` uses a European call option as the baseline example,
compares the Crank-Nicolson finite difference solution with the exact
Black-Scholes formula, reports error metrics, creates a comparison table, and
checks grid convergence. It also generates exact-formula sensitivity plots for
volatility, interest rate, strike, and maturity.
