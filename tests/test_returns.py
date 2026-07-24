"""Tests for returns.py — return computation and TEY adjustment."""

import numpy as np
import pandas as pd
import pytest

from finance.data import PriceData
from finance.returns import (
    NIIT_RATE,
    _decompose_mub_return,
    _decompose_tax_exempt_return,
    adjust_tey,
    build_return_data,
    compute_log_returns,
    compute_simple_returns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _price_series(values: list[float], name: str = "X") -> pd.Series:
    idx = pd.bdate_range("2022-01-03", periods=len(values))
    return pd.Series(values, index=idx, name=name)


def _price_df(n: int = 5) -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "VTI": 200.0 * np.cumprod(1 + rng.normal(0, 0.01, n)),
            "MUB": 55.0 * np.cumprod(1 + rng.normal(0, 0.002, n)),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# compute_simple_returns
# ---------------------------------------------------------------------------


def test_simple_returns_known_values() -> None:
    """Simple return = (P_t / P_{t-1}) - 1."""
    prices = _price_series([100.0, 110.0, 99.0])
    df = compute_simple_returns(prices.to_frame())
    col = prices.name
    assert df[col].iloc[0] == pytest.approx(0.10)
    assert df[col].iloc[1] == pytest.approx(-0.10, rel=1e-6)


def test_simple_returns_drops_first_row() -> None:
    """Output has one fewer row than input."""
    prices = _price_df(10)
    ret = compute_simple_returns(prices)
    assert len(ret) == len(prices) - 1


def test_simple_returns_no_nans() -> None:
    """No NaN in output after dropna."""
    prices = _price_df(20)
    ret = compute_simple_returns(prices)
    assert ret.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# compute_log_returns
# ---------------------------------------------------------------------------


def test_log_returns_known_values() -> None:
    """Log return = log(P_t / P_{t-1})."""
    prices = _price_series([100.0, np.e * 100.0])  # ratio = e → log return = 1
    df = compute_log_returns(prices.to_frame())
    assert df[prices.name].iloc[0] == pytest.approx(1.0)


def test_log_returns_approx_simple_for_small() -> None:
    """For small returns, log ≈ simple."""
    rng = np.random.default_rng(1)
    vals = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, 100))
    prices = pd.DataFrame({"A": vals}, index=pd.bdate_range("2022-01-03", periods=100))
    simple = compute_simple_returns(prices)
    log_ = compute_log_returns(prices)
    np.testing.assert_allclose(log_["A"].values, simple["A"].values, atol=1e-4)


# ---------------------------------------------------------------------------
# _decompose_tax_exempt_return (+ backward-compat alias _decompose_mub_return)
# ---------------------------------------------------------------------------


def test_decompose_income_only_on_ex_date() -> None:
    """Income return is zero on non-dividend days."""
    prices = _price_series([55.0, 55.0, 55.0, 55.0, 55.0], name="MUB")
    divs = pd.Series(0.0, index=prices.index)
    divs.iloc[2] = 0.10  # dividend on day 3
    _price_ret, income_ret = _decompose_tax_exempt_return(prices, divs)
    assert income_ret.iloc[0] == pytest.approx(0.0)
    assert income_ret.iloc[1] == pytest.approx(0.0)
    assert income_ret.iloc[2] == pytest.approx(0.10 / 55.0)
    assert income_ret.iloc[3] == pytest.approx(0.0)


def test_decompose_mub_alias_matches_generic() -> None:
    """_decompose_mub_return is the same function as _decompose_tax_exempt_return."""
    prices = _price_series([55.0, 56.0, 57.0], name="MUB")
    divs = pd.Series([0.0, 0.05, 0.0], index=prices.index)
    r1 = _decompose_mub_return(prices, divs)
    r2 = _decompose_tax_exempt_return(prices, divs)
    np.testing.assert_array_equal(r1[0].values, r2[0].values)
    np.testing.assert_array_equal(r1[1].values, r2[1].values)


# ---------------------------------------------------------------------------
# adjust_tey
# ---------------------------------------------------------------------------


def test_tey_amplifies_income() -> None:
    """TEY scales up income return by 1/(1-rate)."""
    prices = _price_series([55.0, 55.0, 55.0], name="MUB")
    divs = pd.Series(0.0, index=prices.index)
    divs.iloc[1] = 0.055  # yield = 0.1% of price

    adjusted = adjust_tey(prices, divs, marginal_rate=0.408)
    tey_factor = 1.0 / (1.0 - 0.408)
    raw_income = 0.055 / 55.0
    # Day index 0 in adjusted corresponds to iloc[1] of divs (after the iloc[1:] drop)
    assert adjusted.iloc[0] == pytest.approx(raw_income * tey_factor, rel=1e-6)


def test_tey_no_dividend_unchanged() -> None:
    """Days with no dividend are unaffected by TEY."""
    prices = _price_series([55.0, 56.0, 57.0], name="MUB")
    divs = pd.Series(0.0, index=prices.index)
    adjusted = adjust_tey(prices, divs, marginal_rate=0.408)
    simple = prices.pct_change().dropna()
    np.testing.assert_allclose(adjusted.values, simple.values, rtol=1e-6)


def test_tey_raises_invalid_rate() -> None:
    """marginal_rate outside (0, 1) raises ValueError — tests all boundary cases."""
    prices = _price_series([55.0, 55.0], name="MUB")
    divs = pd.Series(0.0, index=prices.index)
    with pytest.raises(ValueError, match="marginal_rate"):
        adjust_tey(prices, divs, marginal_rate=1.5)
    with pytest.raises(ValueError, match="marginal_rate"):
        adjust_tey(prices, divs, marginal_rate=0.0)
    with pytest.raises(ValueError, match="marginal_rate"):
        adjust_tey(prices, divs, marginal_rate=1.0)
    with pytest.raises(ValueError, match="marginal_rate"):
        adjust_tey(prices, divs, marginal_rate=-0.1)


def test_tey_default_rate_is_niit() -> None:
    """Default marginal rate matches NIIT_RATE constant."""
    prices = _price_series([55.0, 55.0, 55.0], name="MUB")
    divs = pd.Series(0.0, index=prices.index)
    divs.iloc[1] = 0.055
    r1 = adjust_tey(prices, divs)
    r2 = adjust_tey(prices, divs, marginal_rate=NIIT_RATE)
    np.testing.assert_array_equal(r1.values, r2.values)


# ---------------------------------------------------------------------------
# build_return_data
# ---------------------------------------------------------------------------


def _make_price_data(n: int = 20) -> PriceData:
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(42)
    tickers = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
    starts = [200.0, 60.0, 170.0, 55.0, 25.0, 65.0]
    prices_data = {}
    for t, s in zip(tickers, starts, strict=False):
        prices_data[t] = s * np.cumprod(1 + rng.normal(0, 0.01, n))
    prices = pd.DataFrame(prices_data, index=idx)
    dividends = pd.DataFrame(0.0, index=idx, columns=list(tickers))
    # Inject a small MUB dividend mid-series
    dividends.loc[idx[n // 2], "MUB"] = 0.05
    return PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=pd.DataFrame(),
        tickers=tickers,
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
    )


def test_build_return_data_shape() -> None:
    """ReturnData returns and log_returns have same shape and correct log values."""
    pd_obj = _make_price_data(30)
    rd = build_return_data(pd_obj)
    assert rd.returns.shape == rd.log_returns.shape
    # Verify log_returns content is actually log returns, not simple returns
    vti_prices = pd_obj.prices["VTI"]
    expected_log = np.log(vti_prices.iloc[1] / vti_prices.iloc[0])
    assert rd.log_returns["VTI"].iloc[0] == pytest.approx(expected_log)


def test_build_return_data_tey_flag() -> None:
    """tey_adjusted flag reflects apply_tey argument."""
    pd_obj = _make_price_data()
    assert build_return_data(pd_obj, apply_tey=True).tey_adjusted is True
    assert build_return_data(pd_obj, apply_tey=False).tey_adjusted is False


def test_build_return_data_mub_differs_with_tey() -> None:
    """MUB returns differ when TEY is applied vs not."""
    pd_obj = _make_price_data(30)
    rd_tey = build_return_data(pd_obj, apply_tey=True)
    rd_raw = build_return_data(pd_obj, apply_tey=False)
    # With at least one dividend, TEY-adjusted != raw on the dividend date
    assert not rd_tey.returns["MUB"].equals(rd_raw.returns["MUB"])


def test_return_data_immutable() -> None:
    """ReturnData is frozen."""
    pd_obj = _make_price_data()
    rd = build_return_data(pd_obj)
    with pytest.raises((AttributeError, TypeError)):
        rd.tey_adjusted = False  # type: ignore[misc]


def test_build_return_data_tey_tickers_custom() -> None:
    """tey_tickers controls which columns receive TEY; others are unaffected."""
    pd_obj = _make_price_data(30)
    # Apply TEY only to MUB (explicit)
    rd_mub = build_return_data(pd_obj, apply_tey=True, tey_tickers=["MUB"])
    # Apply TEY to no tickers
    rd_none = build_return_data(pd_obj, apply_tey=True, tey_tickers=[])

    # MUB differs between the two
    assert not rd_mub.returns["MUB"].equals(rd_none.returns["MUB"])
    # VTI is identical in both
    np.testing.assert_array_equal(
        rd_mub.returns["VTI"].values, rd_none.returns["VTI"].values
    )


def test_build_return_data_tey_ticker_not_in_prices_is_skipped() -> None:
    """Requesting TEY for a ticker absent from prices silently skips it."""
    pd_obj = _make_price_data(20)
    # "NONEXISTENT" is not in prices — should not raise
    rd = build_return_data(pd_obj, apply_tey=True, tey_tickers=["MUB", "NONEXISTENT"])
    assert rd.tey_adjusted is True


def test_adjust_tey_result_named_after_input() -> None:
    """adjust_tey result Series name matches prices.name (generic, not 'MUB')."""
    prices = _price_series([100.0, 100.0, 100.0], name="VWITX")
    divs = pd.Series(0.0, index=prices.index)
    divs.iloc[1] = 0.10
    result = adjust_tey(prices, divs, marginal_rate=0.30)
    assert result.name == "VWITX"
