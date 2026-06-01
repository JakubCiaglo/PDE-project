"""Shared configuration objects for pricing experiments."""

import math
import sys
from dataclasses import dataclass

sys.dont_write_bytecode = True


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class BlackScholesParams:
    """Parameters of the Black-Scholes model."""

    strike: float = 100.0
    maturity: float = 1.0
    rate: float = 0.05
    volatility: float = 0.20
    option_type: str = "call"

    def normalized(self) -> "BlackScholesParams":
        """Return numerically safe parameters for pricing functions."""

        option_type = str(self.option_type).strip().lower()
        if option_type in {"c", "call"}:
            option_type = "call"
        elif option_type in {"p", "put"}:
            option_type = "put"
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return BlackScholesParams(
            strike=max(_finite_float(self.strike, "strike"), 1e-12),
            maturity=max(_finite_float(self.maturity, "maturity"), 0.0),
            rate=_finite_float(self.rate, "rate"),
            volatility=max(_finite_float(self.volatility, "volatility"), 0.0),
            option_type=option_type,
        )

    def validate(self) -> None:
        strike = _finite_float(self.strike, "strike")
        maturity = _finite_float(self.maturity, "maturity")
        _finite_float(self.rate, "rate")
        volatility = _finite_float(self.volatility, "volatility")

        if strike <= 0:
            raise ValueError("strike must be positive")
        if maturity < 0:
            raise ValueError("maturity cannot be negative")
        if volatility < 0:
            raise ValueError("volatility cannot be negative")
        if str(self.option_type).strip().lower() not in {"c", "call", "p", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")


@dataclass(frozen=True)
class GridParams:
    """Finite difference grid settings."""

    s_max: float = 200.0
    asset_steps: int = 200
    time_steps: int = 2000

    def normalized(self, minimum_s_max: float = 0.0) -> "GridParams":
        """Return a grid that is large enough and has valid step counts."""

        s_max = max(
            _finite_float(self.s_max, "s_max"),
            _finite_float(minimum_s_max, "minimum_s_max"),
            1e-12,
        )
        return GridParams(
            s_max=s_max,
            asset_steps=max(int(self.asset_steps), 3),
            time_steps=max(int(self.time_steps), 1),
        )

    def validate(self) -> None:
        s_max = _finite_float(self.s_max, "s_max")
        if s_max <= 0:
            raise ValueError("s_max must be positive")
        if self.asset_steps < 3:
            raise ValueError("asset_steps must be at least 3")
        if self.time_steps < 1:
            raise ValueError("time_steps must be positive")
