"""Return computation and tax-equivalent yield adjustment.

All functions here are pure; no I/O. Receives PriceData from data.py and
produces ReturnData consumed by volatility.py, metrics.py, and portfolio.py.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance.consts import NIIT_RATE
from finance.data import PriceData


@dataclass(frozen=True)
class ReturnData:
    """Simple and log return series for all assets, with optional TEY adjustment.

    Attributes:
        returns: DatetimeIndex x asset columns, daily simple returns.
        log_returns: DatetimeIndex x asset columns, daily log returns.
        tey_adjusted: True if MUB returns include the TEY adjustment.
        marginal_rate: Marginal tax rate used for TEY (e.g. 0.408).
        risk_free_rate: Daily annualized risk-free rate (decimal) aligned to
            the returns index. Defaults to a zero Series when not supplied.
    """

    returns: pd.DataFrame
    log_returns: pd.DataFrame
    tey_adjusted: bool
    marginal_rate: float
    risk_free_rate: pd.Series


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple (arithmetic) returns.

    Arguments:
        prices: DataFrame of adjusted close prices (DatetimeIndex x assets).

    Returns:
        DataFrame of simple returns; first row is dropped (NaN from pct_change).
    """
    return prices.pct_change().dropna()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns.

    Arguments:
        prices: DataFrame of adjusted close prices (DatetimeIndex x assets).

    Returns:
        DataFrame of log returns; first row is dropped.
    """
    log_df: pd.DataFrame = pd.DataFrame(
        np.log(prices.values / prices.shift(1).values),
        index=prices.index,
        columns=prices.columns,
    )
    return log_df.dropna()


def _decompose_tax_exempt_return(
    prices: pd.Series,
    dividends: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Split a tax-exempt bond fund's total return into price and income components.

    Arguments:
        prices: Adjusted close price series for the fund.
        dividends: Per-share dividend series aligned to price index.

    Returns:
        Tuple of (price_return, income_return) as daily simple return Series.
        income_return is non-zero only on ex-dividend dates.
    """
    price_return = prices.pct_change().fillna(0.0)
    prior_prices = prices.shift(1)
    income_return = (dividends / prior_prices).fillna(0.0)
    return price_return, income_return


# Backward-compatible alias used by existing tests
_decompose_mub_return = _decompose_tax_exempt_return


def adjust_tey(
    prices: pd.Series,
    dividends: pd.Series,
    marginal_rate: float = NIIT_RATE,
) -> pd.Series:
    """Adjust a tax-exempt bond fund's return series for tax-equivalent yield.

    The price-appreciation component is unchanged. The income (yield) component
    is scaled by 1 / (1 - marginal_rate) to reflect the pre-tax equivalent
    yield a taxable investor would require.

    Arguments:
        prices: Adjusted close price series for the fund.
        dividends: Per-share dividends aligned to the price index.
        marginal_rate: Combined marginal tax rate (default 0.408 for NIIT).

    Returns:
        Daily simple return Series with TEY-adjusted income component.
        Series name matches prices.name.

    Raises:
        ValueError: If marginal_rate is not in (0, 1).
    """
    if not (0.0 < marginal_rate < 1.0):
        raise ValueError(f"marginal_rate must be in (0, 1), got {marginal_rate}")

    price_ret, income_ret = _decompose_tax_exempt_return(prices, dividends)
    tey_factor = 1.0 / (1.0 - marginal_rate)
    adjusted = price_ret + income_ret * tey_factor
    adjusted.name = prices.name
    return adjusted.iloc[1:]  # drop first row to match pct_change behaviour


def build_return_data(
    price_data: PriceData,
    marginal_rate: float = NIIT_RATE,
    apply_tey: bool = True,
    tey_tickers: list[str] | None = None,
    risk_free_series: pd.Series | None = None,
) -> ReturnData:
    """Compute full return dataset from PriceData.

    Applies TEY adjustment to each ticker in tey_tickers if apply_tey is True.
    All other assets use raw adjusted-close simple returns.

    Arguments:
        price_data: PriceData from data.build_price_data().
        marginal_rate: Marginal tax rate for TEY adjustment.
        apply_tey: Whether to apply the TEY adjustment.
        tey_tickers: Tickers to apply TEY to. Each must have a dividends column
            in price_data.dividends. Defaults to ["MUB"].
        risk_free_series: Optional daily annualized risk-free rate Series
            (decimal, e.g. 0.05 for 5%) from data.fetch_risk_free_rate().
            If None, defaults to a zero Series aligned to the returns index.

    Returns:
        ReturnData with aligned simple and log return DataFrames and
        risk_free_rate Series.
    """
    if tey_tickers is None:
        tey_tickers = ["MUB"]

    simple = compute_simple_returns(price_data.prices)
    log_ret = compute_log_returns(price_data.prices)

    if apply_tey:
        simple = simple.copy()
        for ticker in tey_tickers:
            if ticker not in price_data.prices.columns:
                continue
            divs = price_data.dividends[ticker].reindex(
                price_data.prices.index, fill_value=0.0
            )
            tey_col = adjust_tey(price_data.prices[ticker], divs, marginal_rate)
            simple[ticker] = tey_col.reindex(simple.index, fill_value=0.0)

    if risk_free_series is not None:
        rfr = risk_free_series.reindex(simple.index, method="ffill").fillna(0.0)
    else:
        rfr = pd.Series(0.0, index=simple.index, name="risk_free_rate")

    return ReturnData(
        returns=simple,
        log_returns=log_ret,
        tey_adjusted=apply_tey,
        marginal_rate=marginal_rate,
        risk_free_rate=rfr,
    )
