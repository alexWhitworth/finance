"""Tests for data.py — splice logic and price validation."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from finance.data import (
    KMLM_START,
    TICKERS,
    PriceData,
    _forward_fill_prices,
    build_price_data,
    splice_kmlm,
)

# ---------------------------------------------------------------------------
# splice_kmlm
# ---------------------------------------------------------------------------


def _make_series(start: str, end: str, name: str, seed: int = 0) -> pd.Series:
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx)))
    return pd.Series(prices, index=idx, name=name)


def test_splice_kmlm_no_overlap() -> None:
    """Pre-splice segment ends the day before splice_date; post starts on it."""
    aqmix = _make_series("2015-01-02", "2021-06-30", "AQMIX")
    kmlm = _make_series("2021-01-04", "2023-12-31", "KMLM")
    spliced = splice_kmlm(kmlm, aqmix, KMLM_START)

    assert spliced.name == "KMLM"
    # All dates before KMLM_START come from AQMIX
    pre_dates = spliced.index[spliced.index < KMLM_START]
    assert not pre_dates.empty
    # All dates from KMLM_START onward come from KMLM
    post_dates = spliced.index[spliced.index >= KMLM_START]
    assert not post_dates.empty
    # No duplicate dates at the boundary
    assert spliced.index.is_unique


def test_splice_kmlm_values_match_sources() -> None:
    """Values in spliced series match the source series on either side."""
    aqmix = _make_series("2018-01-02", "2021-06-30", "AQMIX", seed=1)
    kmlm = _make_series("2021-01-04", "2022-12-31", "KMLM", seed=2)
    spliced = splice_kmlm(kmlm, aqmix, KMLM_START)

    # Use guaranteed index positions rather than calendar dates to avoid
    # silent skips if market holidays shift the expected dates.
    pre_date = spliced.index[10]   # well before splice
    assert pre_date in aqmix.index
    assert spliced[pre_date] == pytest.approx(aqmix[pre_date])

    post_date = spliced.index[-10]  # well after splice
    assert post_date in kmlm.index
    assert spliced[post_date] == pytest.approx(kmlm[post_date])


def test_splice_kmlm_raises_empty_post() -> None:
    """Raises if KMLM has no data on or after splice_date."""
    aqmix = _make_series("2018-01-02", "2020-12-31", "AQMIX")
    kmlm = _make_series("2018-01-02", "2020-06-30", "KMLM")  # ends before splice
    with pytest.raises(ValueError, match="KMLM has no data"):
        splice_kmlm(kmlm, aqmix, KMLM_START)


def test_splice_kmlm_raises_empty_pre() -> None:
    """Raises if AQMIX has no data before splice_date."""
    aqmix = _make_series("2021-06-01", "2023-12-31", "AQMIX")  # starts after splice
    kmlm = _make_series("2021-01-04", "2023-12-31", "KMLM")
    with pytest.raises(ValueError, match="AQMIX has no data before"):
        splice_kmlm(kmlm, aqmix, KMLM_START)


# ---------------------------------------------------------------------------
# _forward_fill_prices
# ---------------------------------------------------------------------------


def test_forward_fill_within_limit() -> None:
    """Gaps <= max_gap are filled without raising."""
    idx = pd.bdate_range("2022-01-03", periods=10)
    data = {"A": [100.0] * 10, "B": [50.0] * 10}
    df = pd.DataFrame(data, index=idx)
    df.loc[idx[3], "B"] = float("nan")  # 1-day gap
    df.loc[idx[4], "B"] = float("nan")  # 2-day gap

    filled = _forward_fill_prices(df, max_gap=5)
    assert filled.isna().sum().sum() == 0
    assert filled.loc[idx[3], "B"] == pytest.approx(50.0)


def test_forward_fill_exceeds_limit_raises() -> None:
    """Gaps > max_gap raise ValueError."""
    idx = pd.bdate_range("2022-01-03", periods=15)
    data = {"A": [100.0] * 15}
    df = pd.DataFrame(data, index=idx)
    # Insert a 6-day gap (indices 2-7)
    for i in range(2, 8):
        df.loc[idx[i], "A"] = float("nan")

    with pytest.raises(ValueError, match="gaps exceeding"):
        _forward_fill_prices(df, max_gap=5)


# ---------------------------------------------------------------------------
# PriceData dataclass
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_price_data (via mocks — no network)
# ---------------------------------------------------------------------------


def _fake_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Synthetic price DataFrame matching the requested tickers and date range."""
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(99)
    data = {t: 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx))) for t in tickers}
    return pd.DataFrame(data, index=idx)


def test_build_price_data_no_splice() -> None:
    """Without splice, PriceData contains only TICKERS and spliced=False."""
    start, end = "2021-06-01", "2022-12-31"

    def fake_fetch(tickers: tuple[str, ...], s: str, e: str) -> pd.DataFrame:
        return _fake_prices(tickers, s, e)

    with (
        patch("finance.data.fetch_prices", side_effect=fake_fetch),
        patch("finance.data.fetch_dividends", return_value=pd.Series(dtype=float)),
    ):
        pd_obj = build_price_data(start, end, use_aqmix_splice=False)

    assert pd_obj.spliced is False
    assert set(pd_obj.prices.columns) == set(TICKERS)
    assert pd_obj.prices.isna().sum().sum() == 0


def test_build_price_data_with_splice() -> None:
    """With splice, AQMIX prepends KMLM and spliced=True."""
    start, end = "2018-01-02", "2022-12-31"

    def fake_fetch(tickers: tuple[str, ...], s: str, e: str) -> pd.DataFrame:
        return _fake_prices(tickers, s, e)

    with (
        patch("finance.data.fetch_prices", side_effect=fake_fetch),
        patch("finance.data.fetch_dividends", return_value=pd.Series(dtype=float)),
    ):
        pd_obj = build_price_data(start, end, use_aqmix_splice=True)

    assert pd_obj.spliced is True
    assert set(pd_obj.prices.columns) == set(TICKERS)
    # KMLM column should have data before KMLM_START
    pre_kmlm = pd_obj.prices["KMLM"].loc[:KMLM_START]
    assert not pre_kmlm.empty


def test_build_price_data_missing_aqmix_raises() -> None:
    """If AQMIX is unavailable, fetch_prices raises and it propagates."""
    start, end = "2018-01-02", "2022-12-31"

    def fake_fetch_no_aqmix(tickers: tuple[str, ...], s: str, e: str) -> pd.DataFrame:
        if "AQMIX" in tickers:
            raise ValueError("No price data returned for tickers: ['AQMIX']")
        return _fake_prices(tickers, s, e)

    with (
        patch("finance.data.fetch_prices", side_effect=fake_fetch_no_aqmix),
        patch("finance.data.fetch_dividends", return_value=pd.Series(dtype=float)),
    ):
        with pytest.raises(ValueError, match="AQMIX"):
            build_price_data(start, end, use_aqmix_splice=True)


def test_price_data_immutable() -> None:
    """PriceData is frozen — attribute assignment raises."""
    idx = pd.bdate_range("2022-01-03", periods=5)
    prices = pd.DataFrame({"VTI": [100.0] * 5}, index=idx)
    dividends = pd.DataFrame({"VTI": [0.0] * 5}, index=idx)
    pd_obj = PriceData(
        prices=prices,
        dividends=dividends,
        tickers=("VTI",),
        start_date="2022-01-03",
        end_date="2022-01-07",
        spliced=False,
    )
    with pytest.raises((AttributeError, TypeError)):
        pd_obj.spliced = True  # type: ignore[misc]
