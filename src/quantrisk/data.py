"""Market data loading, cleaning, and local caching."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from urllib.error import URLError

import pandas as pd

LOGGER = logging.getLogger(__name__)


class MissingTickersError(RuntimeError):
    """Raised when one or more requested symbols have no provider data."""

    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers
        super().__init__(f"No real market data available for: {', '.join(tickers)}")


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
    cached = pd.read_parquet(target) if target.exists() else pd.DataFrame()
    missing = [ticker for ticker in tickers if ticker not in cached.columns]
    if not missing:
        return cached.reindex(columns=tickers)
    import yfinance as yf

    fetched: dict[str, pd.Series] = {}
    # IWM has a specific two-attempt Yahoo retry cap before the Stooq fallback.
    yahoo_tickers = [ticker for ticker in missing if ticker != "IWM"]
    if yahoo_tickers:
        for attempt in range(3):
            try:
                raw = yf.download(yahoo_tickers, start=start, end=end, auto_adjust=True, progress=False)
                closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
                if isinstance(closes, pd.Series):
                    closes = closes.to_frame(name=yahoo_tickers[0])
                fetched.update({ticker: closes[ticker].dropna() for ticker in yahoo_tickers if ticker in closes})
                break
            except Exception as exc:  # noqa: BLE001  # Provider exceptions are not consistent.
                LOGGER.warning("Yahoo batch attempt %d failed: %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(2**attempt)
    if "IWM" in missing:
        for attempt in range(2):
            try:
                raw = yf.download("IWM", start=start, end=end, auto_adjust=True, progress=False)
                close = raw["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                fetched["IWM"] = close.dropna()
                if not fetched["IWM"].empty:
                    break
            except Exception as exc:  # noqa: BLE001  # Provider exceptions are not consistent.
                LOGGER.warning("Yahoo IWM attempt %d failed: %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(2)
        if "IWM" not in fetched or fetched["IWM"].empty:
            try:
                url = f"https://stooq.com/q/d/l/?s=iwm.us&i=d&d1={start.replace('-', '')}&d2={(end or pd.Timestamp.today().date().isoformat()).replace('-', '')}"
                stooq = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
                fetched["IWM"] = stooq["Close"].dropna()
                LOGGER.info("Loaded IWM from Stooq fallback")
            except (OSError, URLError, ValueError, KeyError) as exc:
                LOGGER.warning("Stooq fallback failed for IWM: %s", exc)
    if "IWM" in fetched and fetched["IWM"].empty:
        del fetched["IWM"]
    downloaded = pd.DataFrame(fetched)
    prices = pd.concat([cached, downloaded], axis=1).loc[:, lambda frame: ~frame.columns.duplicated(keep="last")]
    unavailable = [ticker for ticker in tickers if ticker not in prices.columns or prices[ticker].dropna().empty]
    if unavailable:
        raise MissingTickersError(unavailable)
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
