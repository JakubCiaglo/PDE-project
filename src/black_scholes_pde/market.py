"""Small helpers for manual market-price comparisons."""

from __future__ import annotations

import sys
from dataclasses import dataclass

sys.dont_write_bytecode = True


@dataclass(frozen=True)
class MarketOptionQuote:
    """A manually collected option quote used in the BTCUSDT comparison."""

    symbol: str
    underlying_price: float
    strike: float
    maturity_years: float
    market_price: float
    option_type: str = "call"
    implied_volatility: float | None = None

    def validate(self) -> None:
        if self.underlying_price <= 0:
            raise ValueError("underlying_price must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity_years < 0:
            raise ValueError("maturity_years cannot be negative")
        if self.market_price < 0:
            raise ValueError("market_price cannot be negative")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
