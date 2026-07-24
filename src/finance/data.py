"""Price data fetching and splice logic.

This module is the I/O boundary for all market data. Everything in here touches
the network; all downstream modules receive pure DataFrames.
"""

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from finance.consts import SPLICE_MAP, TBILL_TICKER, TICKERS, VXUS_VOL_BLEND


@dataclass(frozen=True)
class PriceData:
    """Adjusted close prices and dividends for the asset universe.

    Attributes:
        prices: DatetimeIndex x asset columns, adjusted close prices (investable only).
        dividends: DatetimeIndex x asset columns, per-share dividends (NaN where none).
        vol_prices: DatetimeIndex x vol-index columns (VIX, GVZ, etc.). Empty DataFrame
            if fetch_vol_indices was not requested.
        tickers: Ordered tuple of ticker symbols in prices columns.
        start_date: Inclusive start date of the price history.
        end_date: Inclusive end date of the price history.
        spliced: True if any proxy series was prepended via SPLICE_MAP.
    """

    prices: pd.DataFrame
    dividends: pd.DataFrame
    vol_prices: pd.DataFrame
    tickers: tuple[str, ...]
    start_date: str
    end_date: str
    spliced: bool


def fetch_prices(  # pragma: no cover
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download adjusted close prices from yfinance.

    Arguments:
        tickers: Ticker symbols to download.
        start_date: Start date string in YYYY-MM-DD format (inclusive).
        end_date: End date string in YYYY-MM-DD format (inclusive).

    Returns:
        DataFrame with DatetimeIndex and one column per ticker (adjusted close).

    Raises:
        ValueError: If any ticker returns an entirely empty price series.
    """
    raw = yf.download(
        list(tickers),
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    raw_df: pd.DataFrame = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices: pd.DataFrame = raw_df[[t for t in tickers if t in raw_df.columns]]
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"No price data returned for tickers: {missing}")
    return prices


def fetch_dividends(ticker: str, start_date: str, end_date: str) -> pd.Series:  # pragma: no cover
    """Download per-share dividend history for a single ticker.

    Arguments:
        ticker: Ticker symbol.
        start_date: Start date string in YYYY-MM-DD format.
        end_date: End date string in YYYY-MM-DD format.

    Returns:
        Series with DatetimeIndex of ex-dividend dates and dividend amounts.
        Empty Series if no dividends are found.
    """
    t = yf.Ticker(ticker)
    divs: pd.Series = t.dividends
    if divs.empty:
        return pd.Series(dtype=float, name=ticker)
    # Normalize timezone to UTC-naive for consistent joining
    dt_index = pd.DatetimeIndex(divs.index)
    if dt_index.tz is not None:
        divs.index = dt_index.tz_localize(None)
    result: pd.Series = divs.loc[start_date:end_date].rename(ticker)
    return result


def fetch_risk_free_rate(  # pragma: no cover
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Download the 3-month T-bill annualized yield and convert to a daily decimal rate.

    Uses ^IRX from yfinance, which reports the annualized yield as a percentage
    (e.g. 5.25 for 5.25%). Missing days are forward-filled so the series is
    contiguous over all trading days in the range.

    Arguments:
        start_date: Start date string in YYYY-MM-DD format (inclusive).
        end_date: End date string in YYYY-MM-DD format (inclusive).

    Returns:
        Series with DatetimeIndex of daily annualized rates as decimals
        (e.g. 0.0525). Returns a zero-filled Series if no data is available.
    """
    raw = yf.download(TBILL_TICKER, start=start_date, end=end_date, progress=False)
    if raw.empty:
        return pd.Series(dtype=float, name="risk_free_rate")

    close: pd.Series = raw["Close"].squeeze()
    close = close / 100.0  # percent → decimal annualized rate
    close = close.ffill()
    close.name = "risk_free_rate"
    return close


def fetch_volatility_index(  # pragma: no cover
    asset_ticker: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Fetch the volatility index series for a given asset.

    For VXUS, blends V2TX.DE (developed) and VXEEM (emerging) using VXUS_VOL_BLEND
    weights. For all other assets, fetches the single index defined in ASSET_VOL_INDEX.
    Returns an empty Series if no vol index is mapped for the asset.

    Arguments:
        asset_ticker: Asset ticker (e.g. "VTI", "VXUS").
        start_date: Start date string in YYYY-MM-DD format.
        end_date: End date string in YYYY-MM-DD format.

    Returns:
        Series of annualized implied volatility (decimal), named "<asset>_IV".
        Empty Series if no vol index is mapped.
    """
    from finance.consts import ASSET_VOL_INDEX

    vol_key = ASSET_VOL_INDEX.get(asset_ticker)
    if vol_key is None:
        return pd.Series(dtype=float, name=f"{asset_ticker}_IV")

    if vol_key == "VXUS_COMPOSITE":
        raw_dev = yf.download(
            "V2TX.DE", start=start_date, end=end_date, auto_adjust=True, progress=False
        )
        raw_em = yf.download(
            "VXEEM", start=start_date, end=end_date, auto_adjust=True, progress=False
        )
        dev: pd.Series = (raw_dev["Close"].squeeze() / 100.0).ffill()
        em: pd.Series = (raw_em["Close"].squeeze() / 100.0).ffill()
        w_dev = VXUS_VOL_BLEND["V2TX.DE"]
        w_em = VXUS_VOL_BLEND["VXEEM"]
        composite: pd.Series = (w_dev * dev + w_em * em).rename(f"{asset_ticker}_IV")
        return composite

    raw = yf.download(vol_key, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float, name=f"{asset_ticker}_IV")
    result: pd.Series = (raw["Close"].squeeze() / 100.0).ffill().rename(f"{asset_ticker}_IV")
    return result


def splice(
    primary_prices: pd.Series,
    proxy_prices: pd.Series,
    splice_date: str,
) -> pd.Series:
    """Concatenate a proxy and primary price series at splice_date.

    Proxy prices are level-adjusted so that the last pre-splice price equals
    the first primary price, preventing a spurious return discontinuity at
    the join.  Primary prices are unchanged.

    Arguments:
        primary_prices: Price series for the primary ticker (DatetimeIndex).
            Must have data on or after splice_date.
        proxy_prices: Price series for the proxy ticker (DatetimeIndex).
            Must have data before splice_date.
        splice_date: Date from which primary data begins (YYYY-MM-DD).

    Returns:
        Spliced price Series with the same name as primary_prices.

    Raises:
        ValueError: If primary_prices has no data on or after splice_date.
        ValueError: If proxy_prices has no data before splice_date.
    """
    pre = proxy_prices.loc[:splice_date].iloc[:-1]  # exclude splice_date itself
    post = primary_prices.loc[splice_date:]

    if post.empty:
        raise ValueError(
            f"{primary_prices.name} has no data on or after {splice_date}"
        )
    if pre.empty:
        raise ValueError(
            f"{proxy_prices.name} has no data before {splice_date}"
        )

    # Scale proxy so the last pre-splice value equals the first primary value.
    pre = pre * (post.iloc[0] / pre.iloc[-1])

    spliced: pd.Series = pd.concat([pre, post])
    spliced.name = primary_prices.name
    return spliced


def _forward_fill_prices(prices: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Forward-fill price gaps up to max_gap days; raise if a longer gap exists.

    Arguments:
        prices: Raw price DataFrame (may contain NaNs for holidays/halts).
        max_gap: Maximum consecutive NaN days to fill silently.

    Returns:
        Forward-filled DataFrame.

    Raises:
        ValueError: If any asset has a gap longer than max_gap consecutive days.
    """
    filled = prices.ffill(limit=max_gap)
    remaining_nans = filled.isna().sum()
    bad = remaining_nans[remaining_nans > 0]
    if not bad.empty:
        raise ValueError(
            f"Price gaps exceeding {max_gap} days found for: {bad.to_dict()}"
        )
    return filled


def build_price_data(
    start_date: str,
    end_date: str,
    tickers: list[str] | None = None,
    use_splice: bool = True,
    fetch_vol_indices: bool = False,
) -> PriceData:
    """Fetch prices for the asset universe and apply proxy splices where needed.

    This is the top-level I/O entry point for all market data. Call once per
    backtest; pass the resulting PriceData into downstream pure functions.

    Arguments:
        start_date: Start date string in YYYY-MM-DD format.
        end_date: End date string in YYYY-MM-DD format.
        tickers: Asset tickers to fetch. Defaults to all TICKERS from consts.
        use_splice: If True, prepend proxy series for any ticker in SPLICE_MAP
            whose start_date predates the splice date. If False, tickers are
            fetched as-is with no prepend.
        fetch_vol_indices: If True, also fetch volatility index series (VIX,
            GVZ, etc.) and store in PriceData.vol_prices. Defaults to False.

    Returns:
        PriceData with prices, dividends, vol_prices, tickers, dates, and
        splice flag.

    Notes:
        Dividends are fetched only for MUB (required for TEY adjustment).
        All other dividend columns will be zero-filled.
    """
    asset_tickers: tuple[str, ...] = tuple(tickers) if tickers is not None else TICKERS

    # Determine which proxy tickers need to be fetched
    splice_needed: dict[str, tuple[str, str]] = {}
    if use_splice:
        for ticker, (proxy, splice_date) in SPLICE_MAP.items():
            if ticker in asset_tickers and start_date < splice_date:
                splice_needed[ticker] = (proxy, splice_date)

    proxy_tickers = tuple(proxy for proxy, _ in splice_needed.values())
    fetch_tickers = (*asset_tickers, *proxy_tickers)

    raw_prices = fetch_prices(fetch_tickers, start_date, end_date)

    # Apply splice for each ticker that needs it
    prices = raw_prices[list(asset_tickers)].copy()
    for ticker, (proxy, splice_date) in splice_needed.items():
        spliced_col = splice(raw_prices[ticker], raw_prices[proxy], splice_date)
        prices[ticker] = spliced_col
        prices = prices.loc[spliced_col.index[0]:]

    # Trim leading NaNs (tickers with later inception), then fill small holiday gaps
    prices = prices.loc[prices.dropna().index[0]:]
    prices = _forward_fill_prices(prices)

    # Fetch MUB dividends for TEY adjustment
    dividends = pd.DataFrame(0.0, index=prices.index, columns=list(asset_tickers))
    if "MUB" in asset_tickers:
        mub_divs = fetch_dividends("MUB", start_date, end_date)
        if not mub_divs.empty:
            mub_divs = mub_divs.reindex(prices.index, fill_value=0.0)
            dividends["MUB"] = mub_divs

    # Optionally fetch volatility index series
    vol_prices: pd.DataFrame
    if fetch_vol_indices:
        vol_cols = {
            ticker: fetch_volatility_index(ticker, start_date, end_date)
            for ticker in asset_tickers
        }
        non_empty = {k: v for k, v in vol_cols.items() if not v.empty}
        vol_prices = pd.DataFrame(non_empty) if non_empty else pd.DataFrame()
    else:
        vol_prices = pd.DataFrame()

    return PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=vol_prices,
        tickers=tuple(asset_tickers),
        start_date=str(prices.index[0].date()),
        end_date=str(prices.index[-1].date()),
        spliced=bool(splice_needed),
    )
