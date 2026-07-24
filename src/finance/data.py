"""Price data fetching and AQMIX/KMLM splice logic.

This module is the I/O boundary for all market data. Everything in here touches
the network; all downstream modules receive pure DataFrames.
"""

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

TICKERS: tuple[str, ...] = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
KMLM_START: str = "2021-01-01"
AQMIX_PROXY_TICKER: str = "AQMIX"
TBILL_TICKER: str = "^IRX"  # 13-week T-bill annualized yield (percent)


@dataclass(frozen=True)
class PriceData:
    """Adjusted close prices and dividends for the asset universe.

    Attributes:
        prices: DatetimeIndex x asset columns, adjusted close prices.
        dividends: DatetimeIndex x asset columns, per-share dividends (NaN where none).
        tickers: Ordered tuple of ticker symbols in prices columns.
        start_date: Inclusive start date of the price history.
        end_date: Inclusive end date of the price history.
        spliced: True if AQMIX was prepended as a KMLM proxy.
    """

    prices: pd.DataFrame
    dividends: pd.DataFrame
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


def splice_kmlm(
    kmlm_prices: pd.Series,
    aqmix_prices: pd.Series,
    splice_date: str = KMLM_START,
) -> pd.Series:
    """Concatenate AQMIX (proxy) and KMLM price series at the splice date.

    Uses raw AQMIX returns without vol-scaling. AQMIX prices are level-adjusted
    so that the last pre-splice price equals the first KMLM price, preventing a
    spurious return discontinuity at the join.  KMLM prices are unchanged.

    Arguments:
        kmlm_prices: KMLM price series (DatetimeIndex).
        aqmix_prices: AQMIX price series (DatetimeIndex).
        splice_date: Date from which KMLM data begins. Defaults to KMLM_START.

    Returns:
        Spliced price Series named "KMLM".

    Raises:
        ValueError: If kmlm_prices has no data on or after splice_date.
        ValueError: If aqmix_prices has no data before splice_date.
    """
    pre = aqmix_prices.loc[:splice_date].iloc[:-1]  # exclude splice_date itself
    post = kmlm_prices.loc[splice_date:]

    if post.empty:
        raise ValueError(f"KMLM has no data on or after {splice_date}")
    if pre.empty:
        raise ValueError(f"AQMIX has no data before {splice_date}")

    # Scale AQMIX prices so the last pre-splice value equals the first KMLM value.
    # This makes pct_change() at the seam exactly 0%, preserving all daily returns.
    pre = pre * (post.iloc[0] / pre.iloc[-1])

    spliced: pd.Series = pd.concat([pre, post])
    spliced.name = "KMLM"
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
    use_aqmix_splice: bool = True,
) -> PriceData:
    """Fetch prices for the full asset universe and apply AQMIX/KMLM splice.

    This is the top-level I/O entry point for all market data. Call once per
    backtest; pass the resulting PriceData into downstream pure functions.

    Arguments:
        start_date: Start date string in YYYY-MM-DD format.
        end_date: End date string in YYYY-MM-DD format.
        use_aqmix_splice: If True and start_date < KMLM_START, prepend AQMIX
            as a raw proxy for KMLM. If False, KMLM data begins at KMLM_START.

    Returns:
        PriceData with prices, dividends, tickers, dates, and splice flag.

    Notes:
        Dividends are fetched only for MUB (required for TEY adjustment).
        All other dividend columns will be zero-filled.
    """
    needs_splice = use_aqmix_splice and start_date < KMLM_START

    # Determine which tickers to fetch
    fetch_tickers: tuple[str, ...]
    if needs_splice:
        fetch_tickers = (*TICKERS, AQMIX_PROXY_TICKER)
    else:
        fetch_tickers = TICKERS

    raw_prices = fetch_prices(fetch_tickers, start_date, end_date)

    if needs_splice:
        kmlm_col = raw_prices["KMLM"]
        aqmix_col = raw_prices[AQMIX_PROXY_TICKER]
        spliced_kmlm = splice_kmlm(kmlm_col, aqmix_col, KMLM_START)
        prices = raw_prices[list(TICKERS)].copy()
        prices["KMLM"] = spliced_kmlm
        prices = prices.loc[spliced_kmlm.index[0]:]
    else:
        prices = raw_prices[list(TICKERS)].copy()

    # Trim rows before any ticker's inception (leading NaNs); then fill small holiday gaps.
    prices = prices.loc[prices.dropna().index[0]:]
    prices = _forward_fill_prices(prices)

    # Fetch MUB dividends for TEY adjustment
    mub_divs = fetch_dividends("MUB", start_date, end_date)
    dividends = pd.DataFrame(0.0, index=prices.index, columns=list(TICKERS))
    if not mub_divs.empty:
        mub_divs = mub_divs.reindex(prices.index, fill_value=0.0)
        dividends["MUB"] = mub_divs

    return PriceData(
        prices=prices,
        dividends=dividends,
        tickers=TICKERS,
        start_date=str(prices.index[0].date()),
        end_date=str(prices.index[-1].date()),
        spliced=needs_splice,
    )
