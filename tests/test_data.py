"""Tests for data.py — splice logic and price validation."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from finance.consts import SPLICE_MAP, TICKERS
from finance.data import (
    PriceData,
    _forward_fill_prices,
    build_price_data,
    splice,
)

# KMLM splice date from SPLICE_MAP for convenience
_KMLM_START: str = SPLICE_MAP["KMLM"][1]

# ---------------------------------------------------------------------------
# splice
# ---------------------------------------------------------------------------


def _make_series(start: str, end: str, name: str, seed: int = 0) -> pd.Series:
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx)))
    return pd.Series(prices, index=idx, name=name)


def test_splice_no_overlap() -> None:
    """Pre-splice segment ends the day before splice_date; post starts on it."""
    aqmix = _make_series("2015-01-02", "2021-06-30", "AQMIX")
    kmlm = _make_series("2021-01-04", "2023-12-31", "KMLM")
    spliced = splice(kmlm, aqmix, _KMLM_START)

    assert spliced.name == "KMLM"
    # All dates before _KMLM_START come from AQMIX
    pre_dates = spliced.index[spliced.index < _KMLM_START]
    assert not pre_dates.empty
    # All dates from _KMLM_START onward come from KMLM
    post_dates = spliced.index[spliced.index >= _KMLM_START]
    assert not post_dates.empty
    # No duplicate dates at the boundary
    assert spliced.index.is_unique


def test_splice_values_match_sources() -> None:
    """KMLM values are unchanged post-splice; proxy returns are preserved pre-splice."""
    aqmix = _make_series("2018-01-02", "2021-06-30", "AQMIX", seed=1)
    kmlm = _make_series("2021-01-04", "2022-12-31", "KMLM", seed=2)
    spliced = splice(kmlm, aqmix, _KMLM_START)

    # Post-splice values are unchanged from KMLM
    post_date = spliced.index[-10]
    assert post_date in kmlm.index
    assert spliced[post_date] == pytest.approx(kmlm[post_date])

    # Pre-splice daily returns match AQMIX (level-scaling preserves return ratios)
    pre_idx = spliced.index[spliced.index < _KMLM_START]
    spliced_pre_rets = spliced.loc[pre_idx].pct_change().dropna()
    aqmix_pre_rets = aqmix.loc[pre_idx].pct_change().dropna()
    common = spliced_pre_rets.index.intersection(aqmix_pre_rets.index)
    assert (spliced_pre_rets.loc[common].values == pytest.approx(
        aqmix_pre_rets.loc[common].values, rel=1e-9
    ))


def test_splice_no_seam_jump() -> None:
    """pct_change() at the splice boundary is exactly 0% (no level discontinuity)."""
    aqmix = _make_series("2018-01-02", "2021-06-30", "AQMIX", seed=3)
    kmlm = _make_series("2021-01-04", "2022-12-31", "KMLM", seed=4)
    spliced = splice(kmlm, aqmix, _KMLM_START)

    returns = spliced.pct_change()
    # The first KMLM date (splice boundary) should have a ~0 return
    seam_date = kmlm.index[0]
    assert returns.loc[seam_date] == pytest.approx(0.0, abs=1e-9)


def test_splice_raises_empty_post() -> None:
    """Raises if primary has no data on or after splice_date."""
    aqmix = _make_series("2018-01-02", "2020-12-31", "AQMIX")
    kmlm = _make_series("2018-01-02", "2020-06-30", "KMLM")  # ends before splice
    with pytest.raises(ValueError, match="KMLM has no data"):
        splice(kmlm, aqmix, _KMLM_START)


def test_splice_raises_empty_pre() -> None:
    """Raises if proxy has no data before splice_date."""
    aqmix = _make_series("2021-06-01", "2023-12-31", "AQMIX")  # starts after splice
    kmlm = _make_series("2021-01-04", "2023-12-31", "KMLM")
    with pytest.raises(ValueError, match="AQMIX has no data before"):
        splice(kmlm, aqmix, _KMLM_START)


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
        pd_obj = build_price_data(start, end, use_splice=False)

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
        pd_obj = build_price_data(start, end, use_splice=True)

    assert pd_obj.spliced is True
    assert set(pd_obj.prices.columns) == set(TICKERS)
    # KMLM column should have data before _KMLM_START
    pre_kmlm = pd_obj.prices["KMLM"].loc[:_KMLM_START]
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
            build_price_data(start, end, use_splice=True)


def test_price_data_immutable() -> None:
    """PriceData is frozen — attribute assignment raises."""
    idx = pd.bdate_range("2022-01-03", periods=5)
    prices = pd.DataFrame({"VTI": [100.0] * 5}, index=idx)
    dividends = pd.DataFrame({"VTI": [0.0] * 5}, index=idx)
    pd_obj = PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=pd.DataFrame(),
        tickers=("VTI",),
        start_date="2022-01-03",
        end_date="2022-01-07",
        spliced=False,
    )
    with pytest.raises((AttributeError, TypeError)):
        pd_obj.spliced = True  # type: ignore[misc]


def test_build_price_data_vol_prices_empty_when_not_requested() -> None:
    """vol_prices is an empty DataFrame when fetch_vol_indices=False."""
    start, end = "2021-06-01", "2022-12-31"

    with (
        patch("finance.data.fetch_prices", side_effect=lambda t, s, e: _fake_prices(t, s, e)),
        patch("finance.data.fetch_dividends", return_value=pd.Series(dtype=float)),
    ):
        pd_obj = build_price_data(start, end, use_splice=False, fetch_vol_indices=False)

    assert pd_obj.vol_prices.empty


def test_build_price_data_custom_tickers() -> None:
    """Custom ticker list is respected."""
    start, end = "2021-06-01", "2022-12-31"
    custom = ["VTI", "GLD"]

    with (
        patch("finance.data.fetch_prices", side_effect=lambda t, s, e: _fake_prices(t, s, e)),
        patch("finance.data.fetch_dividends", return_value=pd.Series(dtype=float)),
    ):
        pd_obj = build_price_data(start, end, tickers=custom, use_splice=False)

    assert set(pd_obj.prices.columns) == {"VTI", "GLD"}
    assert pd_obj.tickers == ("VTI", "GLD")
