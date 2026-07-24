"""Return computation and tax-equivalent yield adjustment.

All functions here are pure; no I/O. Receives PriceData from data.py and
produces ReturnData consumed by volatility.py, metrics.py, and portfolio.py.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance.data import PriceData

NIIT_RATE: float = 0.408  # Federal NIIT + no state tax


@dataclass(frozen=True)
class ReturnData:
    """Simple and log return series for all assets, with optional TEY adjustment.

    Attributes:
        returns: DatetimeIndex x asset columns, daily simple returns.
        log_returns: DatetimeIndex x asset columns, daily log returns.
        tey_adjusted: True if MUB returns include the TEY adjustment.
        marginal_rate: Marginal tax rate used for TEY (e.g. 0.408).
    """

    returns: pd.DataFrame
    log_returns: pd.DataFrame
    tey_adjusted: bool
    marginal_rate: float


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


def _decompose_mub_return(
    mub_prices: pd.Series,
    mub_dividends: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Split MUB total return into price-appreciation and income components.

    Arguments:
        mub_prices: Adjusted close price series for MUB.
        mub_dividends: Per-share dividend series aligned to price index.

    Returns:
        Tuple of (price_return, income_return) as daily simple return Series.
        income_return is non-zero only on ex-dividend dates.
    """
    price_return = mub_prices.pct_change().fillna(0.0)

    # Income return: dividend / prior price
    prior_prices = mub_prices.shift(1)
    income_return = (mub_dividends / prior_prices).fillna(0.0)

    return price_return, income_return


def adjust_tey(
    mub_prices: pd.Series,
    mub_dividends: pd.Series,
    marginal_rate: float = NIIT_RATE,
) -> pd.Series:
    """Adjust MUB return series for tax-equivalent yield.

    The price-appreciation component is unchanged. The income (yield) component
    is scaled by 1 / (1 - marginal_rate) to reflect the pre-tax equivalent
    yield a taxable investor would require.

    Arguments:
        mub_prices: Adjusted close price series for MUB.
        mub_dividends: Per-share dividends aligned to the price index.
        marginal_rate: Combined marginal tax rate (default 0.408 for NIIT).

    Returns:
        Daily simple return Series for MUB with TEY-adjusted income component.

    Raises:
        ValueError: If marginal_rate is not in (0, 1).
    """
    if not (0.0 < marginal_rate < 1.0):
        raise ValueError(f"marginal_rate must be in (0, 1), got {marginal_rate}")

    price_ret, income_ret = _decompose_mub_return(mub_prices, mub_dividends)
    tey_factor = 1.0 / (1.0 - marginal_rate)
    adjusted = price_ret + income_ret * tey_factor
    adjusted.name = "MUB"
    return adjusted.iloc[1:]  # drop first row to match pct_change behaviour


def build_return_data(
    price_data: PriceData,
    marginal_rate: float = NIIT_RATE,
    apply_tey: bool = True,
) -> ReturnData:
    """Compute full return dataset from PriceData.

    Applies TEY adjustment to MUB if apply_tey is True.  All other assets
    use raw adjusted-close simple returns.

    Arguments:
        price_data: PriceData from data.build_price_data().
        marginal_rate: Marginal tax rate for MUB TEY adjustment.
        apply_tey: Whether to apply the TEY adjustment to MUB.

    Returns:
        ReturnData with aligned simple and log return DataFrames.
    """
    simple = compute_simple_returns(price_data.prices)
    log_ret = compute_log_returns(price_data.prices)

    if apply_tey and "MUB" in price_data.prices.columns:
        mub_divs = price_data.dividends["MUB"].reindex(
            price_data.prices.index, fill_value=0.0
        )
        tey_mub = adjust_tey(price_data.prices["MUB"], mub_divs, marginal_rate)
        simple = simple.copy()
        simple["MUB"] = tey_mub.reindex(simple.index, fill_value=0.0)

    return ReturnData(
        returns=simple,
        log_returns=log_ret,
        tey_adjusted=apply_tey,
        marginal_rate=marginal_rate,
    )
