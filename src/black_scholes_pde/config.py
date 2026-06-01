"""Shared configuration objects for pricing experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BlackScholesParams:
    """Parameters of the Black-Scholes model."""

    strike: float = 100.0
    maturity: float = 1.0
    rate: float = 0.05
    volatility: float = 0.20
    option_type: str = "call"

    def validate(self) -> None:
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity < 0:
            raise ValueError("maturity cannot be negative")
        if self.volatility < 0:
            raise ValueError("volatility cannot be negative")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")


@dataclass(frozen=True)
class GridParams:
    """Finite difference grid settings."""

    s_max: float = 200.0
    asset_steps: int = 200
    time_steps: int = 2000

    def validate(self) -> None:
        if self.s_max <= 0:
            raise ValueError("s_max must be positive")
        if self.asset_steps < 3:
            raise ValueError("asset_steps must be at least 3")
        if self.time_steps < 1:
            raise ValueError("time_steps must be positive")
