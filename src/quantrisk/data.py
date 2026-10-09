"""Market data loading, cleaning, and local caching."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill short gaps and validate a clean price panel."""
    clean = prices.sort_index().loc[~prices.index.duplicated(keep="last")]
    clean = clean.ffill(limit=3).dropna(how="all")
    clean = clean.dropna(axis=1, how="all")
    clean = clean.loc[clean.notna().any(axis=1)]
    if not clean.index.is_monotonic_increasing or not clean.index.is_unique:
        raise ValueError("Price index must be monotonic and unique")
    first_valid = clean.notna().any(axis=1).idxmax()
    if clean.loc[first_valid:].isna().any().any():
        raise ValueError("Missing values remain after first valid price date")
    return clean.astype(float)


def download_prices(
    tickers: list[str], start: str, end: str | None = None, cache_path: str | Path = "data/raw/prices.parquet"
) -> pd.DataFrame:
    """Download adjusted close data with retries and a Stooq fallback."""
    target = Path(cache_path)
    if target.exists():
        return pd.read_parquet(target)
    import yfinance as yf

    prices = None
    for attempt in range(3):
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
            prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            break
        except Exception as exc:  # noqa: BLE001  # Provider exceptions are not consistent.
            LOGGER.warning("Yahoo download attempt %d failed: %s", attempt + 1, exc)
            if attempt < 2:
                time.sleep(2**attempt)
    if prices is None or prices.empty:
        try:
            from pandas_datareader import data as web

            prices = pd.concat({ticker: web.DataReader(ticker, "stooq", start, end)["Close"] for ticker in tickers}, axis=1)
            prices = prices.sort_index()
        except Exception as exc:
            raise RuntimeError("Yahoo and Stooq data downloads both failed") from exc
    if len(tickers) == 1 and isinstance(prices, pd.DataFrame) and len(prices.columns) == 1:
        prices.columns = tickers
    clean = clean_prices(prices.reindex(columns=tickers))
    target.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(target)
    return clean


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily returns."""
    return prices.pct_change(fill_method=None).dropna(how="all")


def load_risk_free(start: str, end: str | None = None, cache_path: str | Path = "data/raw/rf.parquet") -> pd.Series:
    """Fetch daily risk-free rates; return zeros when the provider is unavailable."""
    target = Path(cache_path)
    if target.exists():
        return pd.read_parquet(target).iloc[:, 0]
    try:
        import yfinance as yf

        values = yf.download("^IRX", start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        result = values / 100.0 / 252.0
        result.name = "rf"
    except Exception as exc:  # noqa: BLE001  # Missing provider data is an expected fallback case.
        LOGGER.warning("Risk-free series unavailable; using zero: %s", exc)
        result = pd.Series(0.0, index=pd.bdate_range(start=start, end=end), name="rf")
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_frame().to_parquet(target)
    return result
