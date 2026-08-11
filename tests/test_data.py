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
    proxy = _make_series("2015-01-02", "2021-06-30", "kmlm_mlmi_pre")
    kmlm = _make_series("2020-12-02", "2023-12-31", "KMLM")
    spliced = splice(kmlm, proxy, _KMLM_START)

    assert spliced.name == "KMLM"
    # All dates before _KMLM_START come from proxy
    pre_dates = spliced.index[spliced.index < _KMLM_START]
    assert not pre_dates.empty
    # All dates from _KMLM_START onward come from KMLM
    post_dates = spliced.index[spliced.index >= _KMLM_START]
    assert not post_dates.empty
    # No duplicate dates at the boundary
    assert spliced.index.is_unique


def test_splice_values_match_sources() -> None:
    """KMLM values are unchanged post-splice; proxy returns are preserved pre-splice."""
    proxy = _make_series("2018-01-02", "2021-06-30", "kmlm_mlmi_pre", seed=1)
    kmlm = _make_series("2020-12-02", "2022-12-31", "KMLM", seed=2)
    spliced = splice(kmlm, proxy, _KMLM_START)

    # Post-splice values are unchanged from KMLM
    post_date = spliced.index[-10]
    assert post_date in kmlm.index
    assert spliced[post_date] == pytest.approx(kmlm[post_date])

    # Pre-splice daily returns match proxy (level-scaling preserves return ratios)
    pre_idx = spliced.index[spliced.index < _KMLM_START]
    spliced_pre_rets = spliced.loc[pre_idx].pct_change().dropna()
    proxy_pre_rets = proxy.loc[pre_idx].pct_change().dropna()
    common = spliced_pre_rets.index.intersection(proxy_pre_rets.index)
    assert (spliced_pre_rets.loc[common].values == pytest.approx(
        proxy_pre_rets.loc[common].values, rel=1e-9
    ))


def test_splice_no_seam_jump() -> None:
    """pct_change() at the splice boundary is exactly 0% (no level discontinuity)."""
    proxy = _make_series("2018-01-02", "2021-06-30", "kmlm_mlmi_pre", seed=3)
    kmlm = _make_series("2020-12-02", "2022-12-31", "KMLM", seed=4)
    spliced = splice(kmlm, proxy, _KMLM_START)

    returns = spliced.pct_change()
    # The first KMLM date (splice boundary) should have a ~0 return
    seam_date = kmlm.index[0]
    assert returns.loc[seam_date] == pytest.approx(0.0, abs=1e-9)


def test_splice_raises_empty_post() -> None:
    """Raises if primary has no data on or after splice_date."""
    proxy = _make_series("2018-01-02", "2020-11-30", "kmlm_mlmi_pre")
    kmlm = _make_series("2018-01-02", "2020-06-30", "KMLM")  # ends before splice
    with pytest.raises(ValueError, match="KMLM has no data"):
        splice(kmlm, proxy, _KMLM_START)


def test_splice_raises_empty_pre() -> None:
    """Raises if proxy has no data before splice_date."""
    proxy = _make_series("2021-06-01", "2023-12-31", "kmlm_mlmi_pre")  # starts after splice
    kmlm = _make_series("2020-12-02", "2023-12-31", "KMLM")
    with pytest.raises(ValueError, match="kmlm_mlmi_pre has no data before"):
        splice(kmlm, proxy, _KMLM_START)


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
    """With splice, MLMI parquet prepends KMLM and spliced=True."""
    start, end = "2018-01-02", "2022-12-31"

    def fake_fetch(tickers: tuple[str, ...], s: str, e: str) -> pd.DataFrame:
        return _fake_prices(tickers, s, e)

    fake_proxy = _make_series("2018-01-02", "2020-12-01", "kmlm_mlmi_pre", seed=7)

    with (
        patch("finance.data.fetch_prices", side_effect=fake_fetch),
        patch("finance.data.fetch_file_proxy", return_value=fake_proxy),
        patch("finance.data.fetch_dividends", return_value=pd.Series(dtype=float)),
    ):
        pd_obj = build_price_data(start, end, use_splice=True)

    assert pd_obj.spliced is True
    assert set(pd_obj.prices.columns) == set(TICKERS)
    # KMLM column should have data before _KMLM_START
    pre_kmlm = pd_obj.prices["KMLM"].loc[:_KMLM_START]
    assert not pre_kmlm.empty


def test_build_price_data_missing_proxy_file_raises() -> None:
    """If the MLMI parquet proxy is missing, FileNotFoundError propagates."""
    start, end = "2018-01-02", "2022-12-31"

    with (
        patch("finance.data.fetch_prices", side_effect=lambda t, s, e: _fake_prices(t, s, e)),
        patch(
            "finance.data.fetch_file_proxy",
            side_effect=FileNotFoundError("Proxy parquet not found"),
        ),
        patch("finance.data.fetch_dividends", return_value=pd.Series(dtype=float)),
    ):
        with pytest.raises(FileNotFoundError, match="Proxy parquet not found"):
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


def test_build_price_data_all_nan_raises() -> None:
    """Raises ValueError if all prices are NaN (e.g. yfinance API outage)."""
    start, end = "2021-06-01", "2022-12-31"

    def fake_fetch_all_nan(tickers: tuple[str, ...], s: str, e: str) -> pd.DataFrame:
        idx = pd.bdate_range(s, e)
        return pd.DataFrame(float("nan"), index=idx, columns=list(tickers))

    with (
        patch("finance.data.fetch_prices", side_effect=fake_fetch_all_nan),
        patch("finance.data.fetch_dividends", return_value=pd.Series(dtype=float)),
    ):
        with pytest.raises(ValueError, match="entirely NaN"):
            build_price_data(start, end, use_splice=False)


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


# ---------------------------------------------------------------------------
# F-3: splice-eligibility guard (start_date < splice_date <= end_date)
# ---------------------------------------------------------------------------

# VXUS splice_date = "2011-01-28" per SPLICE_MAP
_VXUS_SPLICE_DATE: str = SPLICE_MAP["VXUS"][1]  # "2011-01-28"


def test_splice_skipped_when_window_before_splice_date() -> None:
    """Window ending before splice_date fetches primary-only with no ValueError.

    Before F-3: start_date < splice_date was True even when end_date < splice_date,
    so splice() was called with an empty post slice and raised. After F-3 the guard
    adds `<= end_date` and the splice path is never entered.
    """
    # Window entirely before VXUS splice_date — no splice should be attempted.
    start = "2009-01-02"
    end = "2010-12-31"  # ends the day before splice_date

    called_with_proxy: list[tuple[str, ...]] = []

    def fake_fetch(tickers: tuple[str, ...], s: str, e: str) -> pd.DataFrame:
        called_with_proxy.append(tickers)
        return _fake_prices(tickers, s, e)

    with (
        patch("finance.data.fetch_prices", side_effect=fake_fetch),
        patch("finance.data.fetch_dividends", return_value=pd.Series(dtype=float)),
    ):
        pd_obj = build_price_data(start, end, tickers=["VXUS"], use_splice=True)

    # No ValueError means splice() was not called with empty post.
    # spliced=False because the guard prevented the splice-needed entry.
    assert pd_obj.spliced is False
    # The proxy ticker VGTSX must NOT have been fetched.
    assert all("VGTSX" not in tickers for tickers in called_with_proxy)
    # Primary VXUS prices are present.
    assert "VXUS" in pd_obj.prices.columns


def test_splice_still_applied_when_window_straddles_splice_date() -> None:
    """A window straddling splice_date (start < splice_date <= end) still splices.

    Verifies that the new `<= end_date` bound does NOT break the normal case:
    the splice must engage when the window covers the splice date.
    """
    start = "2009-01-02"
    end = "2012-12-31"  # end_date > splice_date, window straddles

    def fake_fetch(tickers: tuple[str, ...], s: str, e: str) -> pd.DataFrame:
        return _fake_prices(tickers, s, e)

    with (
        patch("finance.data.fetch_prices", side_effect=fake_fetch),
        patch("finance.data.fetch_dividends", return_value=pd.Series(dtype=float)),
    ):
        pd_obj = build_price_data(start, end, tickers=["VXUS"], use_splice=True)

    assert pd_obj.spliced is True
    # VXUS should have pre-splice history
    pre = pd_obj.prices["VXUS"].loc[:_VXUS_SPLICE_DATE]
    assert not pre.empty


def test_splice_applied_when_end_date_equals_splice_date() -> None:
    """When end_date exactly equals splice_date the splice IS applied (inclusive <=).

    The boundary case: `start_date < splice_date <= end_date` with equality on the
    right side must still trigger the splice path, not skip it.
    """
    start = "2009-01-02"
    end = _VXUS_SPLICE_DATE  # end_date == splice_date exactly

    def fake_fetch(tickers: tuple[str, ...], s: str, e: str) -> pd.DataFrame:
        return _fake_prices(tickers, s, e)

    with (
        patch("finance.data.fetch_prices", side_effect=fake_fetch),
        patch("finance.data.fetch_dividends", return_value=pd.Series(dtype=float)),
    ):
        pd_obj = build_price_data(start, end, tickers=["VXUS"], use_splice=True)

    assert pd_obj.spliced is True
