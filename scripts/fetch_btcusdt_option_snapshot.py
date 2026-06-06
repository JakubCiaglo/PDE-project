"""Fetch reproducible BTCUSDT option snapshots from public Binance APIs.

The script uses only public market-data endpoints. It does not need a Binance
account, API key, API secret, or a local .env file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from black_scholes_pde import (
    close_to_close_volatility,
    garman_klass_volatility,
    rogers_satchell_volatility,
)

EAPI_BASE_URL = "https://eapi.binance.com"
SPOT_BASE_URL = "https://api.binance.com"
USER_AGENT = "black-scholes-pde-student-project/0.1"
DAY_MS = 24 * 60 * 60 * 1000


def fetch_json(base_url: str, path: str, params: dict[str, str | int] | None = None):
    query = f"?{urlencode(params)}" if params else ""
    request = Request(
        f"{base_url}{path}{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_float(value, default: float | None = None) -> float | None:
    if value in {None, ""}:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_timestamp_ms(value) -> int:
    return int(parse_float(value, 0.0) or 0.0)


def utc_iso_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat()


def normalize_option_type(side: str) -> str:
    side = str(side).strip().upper()
    if side in {"CALL", "C"}:
        return "call"
    if side in {"PUT", "P"}:
        return "put"
    raise ValueError(f"Unsupported option side: {side!r}")


def active_btcusdt_symbols(exchange_info: dict, underlying: str, now_ms: int) -> list[dict]:
    symbols = []
    for item in exchange_info.get("optionSymbols", []):
        if item.get("underlying") != underlying:
            continue
        if item.get("status") != "TRADING":
            continue
        expiry_ms = parse_timestamp_ms(item.get("expiryDate"))
        if expiry_ms <= now_ms:
            continue
        symbols.append(item)
    return symbols


def select_contract(
    option_symbols: list[dict],
    underlying_price: float,
    requested_symbol: str | None,
) -> dict:
    if requested_symbol:
        requested_symbol = requested_symbol.strip().upper()
        for item in option_symbols:
            if item.get("symbol") == requested_symbol:
                return item
        raise ValueError(f"Requested symbol {requested_symbol!r} is not an active BTCUSDT option")

    if not option_symbols:
        raise ValueError("No active BTCUSDT option symbols were returned by Binance")

    nearest_expiry = min(parse_timestamp_ms(item.get("expiryDate")) for item in option_symbols)
    candidates = [
        item
        for item in option_symbols
        if parse_timestamp_ms(item.get("expiryDate")) == nearest_expiry
    ]
    return min(
        candidates,
        key=lambda item: (
            abs((parse_float(item.get("strikePrice"), 0.0) or 0.0) - underlying_price),
            0 if normalize_option_type(item.get("side", "")) == "call" else 1,
            item.get("symbol", ""),
        ),
    )


def records_by_symbol(records) -> dict[str, dict]:
    if isinstance(records, dict):
        if "symbol" not in records:
            return {}
        return {records["symbol"]: records}
    return {
        item["symbol"]: item
        for item in records
        if isinstance(item, dict) and item.get("symbol")
    }


def fetch_spot_history(underlying: str, lookback_days: int) -> pd.DataFrame:
    lookback_days = max(2, min(int(lookback_days), 1000))
    rows = fetch_json(
        SPOT_BASE_URL,
        "/api/v3/klines",
        {"symbol": underlying, "interval": "1d", "limit": lookback_days},
    )
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    history = pd.DataFrame(rows, columns=columns)
    for column in ["open", "high", "low", "close", "volume", "quote_volume"]:
        history[column] = history[column].astype(float)
    history["open_time_utc"] = pd.to_datetime(history["open_time"], unit="ms", utc=True)
    history["close_time_utc"] = pd.to_datetime(history["close_time"], unit="ms", utc=True)
    return history[
        [
            "open_time_utc",
            "close_time_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "number_of_trades",
        ]
    ]


def build_option_row(
    option_symbol: dict,
    mark_record: dict,
    ticker_record: dict,
    underlying: str,
    underlying_price: float,
    now_ms: int,
    close_to_close: float,
    german_klass: float,
    rogers_satchell: float,
    historical_volatility_days: int,
) -> dict | None:
    symbol = option_symbol.get("symbol")
    expiry_ms = parse_timestamp_ms(option_symbol.get("expiryDate"))
    strike = parse_float(option_symbol.get("strikePrice"))
    market_price = parse_float(mark_record.get("markPrice"))
    mark_iv = parse_float(mark_record.get("markIV"))
    if not symbol or expiry_ms <= now_ms or strike is None or market_price is None or mark_iv is None:
        return None
    if market_price <= 0:
        return None

    maturity_days = max((expiry_ms - now_ms) / DAY_MS, 0.0)
    return {
        "observed_at_utc": utc_iso_from_ms(now_ms),
        "source": "Binance public market-data API",
        "underlying": underlying,
        "symbol": symbol,
        "option_type": normalize_option_type(option_symbol.get("side", "")),
        "expiry_utc": utc_iso_from_ms(expiry_ms),
        "strike": strike,
        "maturity_days": maturity_days,
        "maturity_years": maturity_days / 365.0,
        "underlying_price": underlying_price,
        "market_price": market_price,
        "bid_price": parse_float(ticker_record.get("bidPrice")),
        "ask_price": parse_float(ticker_record.get("askPrice")),
        "last_price": parse_float(ticker_record.get("lastPrice")),
        "mark_iv": mark_iv,
        "bid_iv": parse_float(mark_record.get("bidIV")),
        "ask_iv": parse_float(mark_record.get("askIV")),
        "risk_free_rate": parse_float(mark_record.get("riskFreeInterest"), 0.0),
        "mark_iv_binance": mark_iv,
        "close_to_close": close_to_close,
        "german_klass": german_klass,
        "rogers_satchell": rogers_satchell,
        "historical_volatility": close_to_close,
        "historical_volatility_days": historical_volatility_days,
        "moneyness": underlying_price / strike,
    }


def build_snapshot(
    underlying: str,
    requested_symbol: str | None,
    lookback_days: int,
    max_maturity_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exchange_info = fetch_json(EAPI_BASE_URL, "/eapi/v1/exchangeInfo")
    index_record = fetch_json(EAPI_BASE_URL, "/eapi/v1/index", {"underlying": underlying})

    now_ms = parse_timestamp_ms(index_record.get("time")) or parse_timestamp_ms(
        exchange_info.get("serverTime")
    )
    underlying_price = parse_float(index_record.get("indexPrice"))
    if underlying_price is None:
        raise ValueError("Binance did not return indexPrice")

    option_symbols = active_btcusdt_symbols(exchange_info, underlying, now_ms)
    selected = select_contract(option_symbols, underlying_price, requested_symbol)
    symbol = selected["symbol"]

    mark_records = records_by_symbol(fetch_json(EAPI_BASE_URL, "/eapi/v1/mark"))
    ticker_records = records_by_symbol(fetch_json(EAPI_BASE_URL, "/eapi/v1/ticker"))
    spot_history = fetch_spot_history(underlying, lookback_days)
    close_to_close = close_to_close_volatility(spot_history["close"], periods_per_year=365)
    german_klass = garman_klass_volatility(
        spot_history["open"],
        spot_history["high"],
        spot_history["low"],
        spot_history["close"],
        periods_per_year=365,
    )
    rogers_satchell = rogers_satchell_volatility(
        spot_history["open"],
        spot_history["high"],
        spot_history["low"],
        spot_history["close"],
        periods_per_year=365,
    )
    historical_volatility_days = max(len(spot_history) - 1, 0)

    selected_row = build_option_row(
        selected,
        mark_records.get(symbol, {}),
        ticker_records.get(symbol, {}),
        underlying,
        underlying_price,
        now_ms,
        close_to_close,
        german_klass,
        rogers_satchell,
        historical_volatility_days,
    )
    if selected_row is None:
        raise ValueError(f"Binance did not return complete quote data for {symbol}")

    rows = []
    max_maturity_days = max(int(max_maturity_days), 1)
    for option_symbol in option_symbols:
        row = build_option_row(
            option_symbol,
            mark_records.get(option_symbol.get("symbol"), {}),
            ticker_records.get(option_symbol.get("symbol"), {}),
            underlying,
            underlying_price,
            now_ms,
            close_to_close,
            german_klass,
            rogers_satchell,
            historical_volatility_days,
        )
        if row is None or row["maturity_days"] > max_maturity_days:
            continue
        rows.append(row)

    if not rows:
        raise ValueError(f"No complete BTCUSDT option quotes found within {max_maturity_days} days")

    snapshot = pd.DataFrame([selected_row])
    chain_snapshot = (
        pd.DataFrame(rows)
        .sort_values(["expiry_utc", "option_type", "strike", "symbol"])
        .reset_index(drop=True)
    )
    return snapshot, chain_snapshot, spot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch BTCUSDT option snapshots for the Black-Scholes project.",
    )
    parser.add_argument(
        "--symbol",
        help="Optional exact Binance option symbol, for example BTC-260626-100000-C.",
    )
    parser.add_argument("--underlying", default="BTCUSDT")
    parser.add_argument("--lookback-days", type=int, default=366)
    parser.add_argument("--max-maturity-days", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "btcusdt_option_snapshot.csv",
    )
    parser.add_argument(
        "--chain-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "btcusdt_option_chain_snapshot.csv",
    )
    parser.add_argument(
        "--spot-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "btcusdt_spot_history.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        snapshot, chain_snapshot, spot_history = build_snapshot(
            underlying=args.underlying,
            requested_symbol=args.symbol,
            lookback_days=args.lookback_days,
            max_maturity_days=args.max_maturity_days,
        )
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        print(f"Could not fetch Binance snapshot: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.chain_output.parent.mkdir(parents=True, exist_ok=True)
    args.spot_output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(args.output, index=False)
    chain_snapshot.to_csv(args.chain_output, index=False)
    spot_history.to_csv(args.spot_output, index=False)

    row = snapshot.iloc[0]
    print(f"Saved option snapshot: {args.output}")
    print(f"Saved option chain snapshot: {args.chain_output} ({len(chain_snapshot)} rows)")
    print(f"Saved BTCUSDT history: {args.spot_output}")
    print(
        f"Selected {row['symbol']} with market price {row['market_price']:.4f} "
        f"and Binance mark IV {row['mark_iv_binance']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
