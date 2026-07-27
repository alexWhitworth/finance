"""Tests for fetch_gtt_signal_data (F-07 — I/O boundary).

Unit tests use mocks; the integration test is marked slow and requires
live network access + a FRED_API_KEY environment variable.

EMPIRICAL FRED-INDEXING CHECK (Phase 3 blocking requirement) is embedded
in the integration test: it asserts that Fred.get_series('UNRATE') indexes
each observation at the reference-month start (day == 1) and that the
Option-B first-Friday-of-following-month re-stamp lands on/after the known
BLS Employment Situation release dates for selected months.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from finance.gtt import (
    GttSignalData,
    compute_position_mask,
    fetch_gtt_signal_data,
)

# ---------------------------------------------------------------------------
# Shared synthetic data builders
# ---------------------------------------------------------------------------


def _monthly_unrate(start: str = "2010-01-01", n: int = 36) -> pd.Series:
    """Synthetic monthly UNRATE indexed at reference-month start (FRED convention)."""
    idx = pd.date_range(start, periods=n, freq="MS")
    values = [6.0 - 0.02 * i for i in range(n)]  # gently declining -> UE signal stays 0
    return pd.Series(values, index=idx, name="UNRATE")


def _daily_vix(start: str = "2010-01-04", n: int = 800) -> pd.DataFrame:
    """Synthetic daily ^VIX DataFrame (as returned by yf.download, Close column, pct x 100)."""
    idx = pd.bdate_range(start, periods=n)
    # VIX expressed as 0-100 scale for the raw download (÷100 applied inside fetch_gtt_signal_data)
    values = [15.0] * n  # 0.15 in decimal, well below any P90 threshold -> VIX signal stays 0
    df = pd.DataFrame({"Close": values}, index=idx)
    return df


def _daily_vti(start: str = "2010-01-04", n: int = 800) -> pd.DataFrame:
    """Synthetic VTI DataFrame for yf.download (ascending prices -> price > SMA200)."""
    idx = pd.bdate_range(start, periods=n)
    prices = 100.0 * np.cumprod(1 + np.full(n, 0.0005))  # gently rising
    df = pd.DataFrame({"Close": prices}, index=idx)
    return df


# ---------------------------------------------------------------------------
# Helper: build a mocked fetch_gtt_signal_data call
# ---------------------------------------------------------------------------


def _mock_call(
    unrate: pd.Series | None = None,
    vix_df: pd.DataFrame | None = None,
    vti_df: pd.DataFrame | None = None,
    start: str = "2010-01-04",
    end: str = "2012-12-31",
    threshold: float = 0.272,
    equity_prices: pd.Series | None = None,
) -> GttSignalData:
    """Invoke fetch_gtt_signal_data with mocked FRED and yfinance.

    fred and yf are imported *inside* fetch_gtt_signal_data, so we patch at
    the source package level: 'fredapi.Fred' and 'yfinance.download'.
    """
    _unrate = unrate if unrate is not None else _monthly_unrate()
    _vix = vix_df if vix_df is not None else _daily_vix()
    _vti = vti_df if vti_df is not None else _daily_vti()

    mock_fred_instance = MagicMock()
    mock_fred_instance.get_series.return_value = _unrate

    def _dl(t: str, **kw: object) -> pd.DataFrame:
        return _vti if t == "VTI" else _vix

    with (
        patch("fredapi.Fred", return_value=mock_fred_instance),
        patch("yfinance.download", side_effect=_dl),
    ):
        return fetch_gtt_signal_data(
            start_date=start,
            end_date=end,
            vix_p90_threshold=threshold,
            equity_prices=equity_prices,
        )


# ---------------------------------------------------------------------------
# F-07: ValueError — pre-1993 start (checked before any network call)
# ---------------------------------------------------------------------------


def test_fetch_gtt_pre_1993_raises_before_network() -> None:
    """start_date < 1993-01-01 raises ValueError without touching FRED or yfinance."""
    with (
        patch("fredapi.Fred") as mock_fred_cls,
        patch("yfinance.download") as mock_yf,
    ):
        with pytest.raises(ValueError, match="1993-01-01"):
            fetch_gtt_signal_data(
                start_date="1992-12-31",
                end_date="2000-01-01",
                vix_p90_threshold=0.272,
            )
        mock_fred_cls.assert_not_called()
        mock_yf.assert_not_called()


def test_fetch_gtt_exactly_1993_does_not_raise_pre1993_guard() -> None:
    """start_date == 1993-01-01 passes the pre-1993 guard (may raise for other reasons)."""
    mock_fred_instance = MagicMock()
    mock_fred_instance.get_series.return_value = _monthly_unrate(start="1993-01-01")

    def _dl_1993(t: str, **kw: object) -> pd.DataFrame:
        return _daily_vti() if t == "VTI" else _daily_vix()

    with (
        patch("fredapi.Fred", return_value=mock_fred_instance),
        patch("yfinance.download", side_effect=_dl_1993),
    ):
        result = fetch_gtt_signal_data(
            start_date="1993-01-01",
            end_date="1995-12-31",
            vix_p90_threshold=0.272,
        )
    assert isinstance(result, GttSignalData)


# ---------------------------------------------------------------------------
# F-07: ValueError — empty UNRATE fetch
# ---------------------------------------------------------------------------


def test_fetch_gtt_empty_unrate_raises() -> None:
    """An empty UNRATE response raises ValueError with an actionable message."""
    mock_fred_instance = MagicMock()
    mock_fred_instance.get_series.return_value = pd.Series(dtype=float)

    with (
        patch("fredapi.Fred", return_value=mock_fred_instance),
        patch("yfinance.download"),
    ):
        with pytest.raises(ValueError, match="empty UNRATE"):
            fetch_gtt_signal_data("2010-01-04", "2012-12-31", 0.272)


# ---------------------------------------------------------------------------
# F-07: ValueError — empty VIX fetch
# ---------------------------------------------------------------------------


def test_fetch_gtt_empty_vix_raises() -> None:
    """An empty ^VIX response raises ValueError with an actionable message."""
    mock_fred_instance = MagicMock()
    mock_fred_instance.get_series.return_value = _monthly_unrate()

    empty_vix = pd.DataFrame()

    def _download(t: str, **kw: object) -> pd.DataFrame:
        if t == "VTI":
            return _daily_vti()
        return empty_vix

    with (
        patch("fredapi.Fred", return_value=mock_fred_instance),
        patch("yfinance.download", side_effect=_download),
    ):
        with pytest.raises(ValueError, match=r"empty.*VIX|VIX.*empty"):
            fetch_gtt_signal_data("2010-01-04", "2012-12-31", 0.272)


# ---------------------------------------------------------------------------
# F-07: Successful mock fetch — delegation to pure functions
# ---------------------------------------------------------------------------


def test_fetch_gtt_returns_gttsignaldata() -> None:
    """On a successful mock fetch, the return type is GttSignalData."""
    result = _mock_call()
    assert isinstance(result, GttSignalData)


def test_fetch_gtt_position_mask_equals_pure_functions() -> None:
    """position_mask equals compute_position_mask applied to the mocked inputs."""
    unrate = _monthly_unrate()
    vix_df = _daily_vix()
    vti_df = _daily_vti()

    result = _mock_call(unrate=unrate, vix_df=vix_df, vti_df=vti_df)

    from finance.gtt import compute_ue_signal, compute_vix_signal

    vix_dec = (vix_df["Close"] / 100.0).rename("VIX")
    vti_prices = vti_df["Close"].squeeze().rename("VTI")
    ue = compute_ue_signal(unrate)
    vix_sig = compute_vix_signal(vix_dec, threshold=0.272)
    expected_mask = compute_position_mask(ue, vix_sig, vti_prices)

    # Align on the common index before comparing (the two computations may start at
    # slightly different calendar dates due to ffill/resampling).
    common = result.position_mask.index.intersection(expected_mask.index)
    assert not common.empty
    pd.testing.assert_series_equal(
        result.position_mask.loc[common].rename(None),
        expected_mask.loc[common].rename(None),
        check_names=False,
    )


def test_fetch_gtt_threshold_stored_for_reproducibility() -> None:
    """vix_p90_threshold on GttSignalData equals the value passed in."""
    result = _mock_call(threshold=0.30)
    assert result.vix_p90_threshold == 0.30


def test_fetch_gtt_unrate_start_reflects_series_start() -> None:
    """unrate_start equals the first index date of the fetched UNRATE series."""
    unrate = _monthly_unrate(start="2008-01-01")
    result = _mock_call(unrate=unrate)
    assert result.unrate_start == pd.Timestamp("2008-01-01")


def test_fetch_gtt_vix_start_reflects_series_start() -> None:
    """vix_start equals the first index date of the fetched VIX series."""
    result = _mock_call()
    assert result.vix_start == pd.Timestamp(_daily_vix().index[0])


def test_fetch_gtt_caller_supplied_equity_prices_skips_vti_download() -> None:
    """When equity_prices is supplied, VTI is NOT fetched from yfinance."""
    mock_fred_instance = MagicMock()
    mock_fred_instance.get_series.return_value = _monthly_unrate()

    vti_prices = _daily_vti()["Close"].squeeze().rename("VTI")
    downloads: list[str] = []

    def _recording_download(t: str, **kw: object) -> pd.DataFrame:
        downloads.append(t)
        return _daily_vix()  # Only ^VIX will be called

    with (
        patch("fredapi.Fred", return_value=mock_fred_instance),
        patch("yfinance.download", side_effect=_recording_download),
    ):
        fetch_gtt_signal_data(
            start_date="2010-01-04",
            end_date="2012-12-31",
            vix_p90_threshold=0.272,
            equity_prices=vti_prices,
        )

    assert "VTI" not in downloads, "VTI fetch should be skipped when equity_prices is supplied"
    assert "^VIX" in downloads


def test_fetch_gtt_position_mask_domain_01() -> None:
    """position_mask contains only 0 and 1 values."""
    result = _mock_call()
    unique = set(result.position_mask.unique().tolist())
    assert unique.issubset({0, 1})


# ---------------------------------------------------------------------------
# F-07: Integration test — live FRED data (slow, requires FRED_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fetch_gtt_live_fred_indexing_check() -> None:
    """EMPIRICAL FRED-INDEXING CHECK (Phase 3, blocking).

    Confirms that Fred.get_series('UNRATE') indexes each observation at the
    reference-month start (day == 1 for all entries) and that the Option-B
    first-Friday-of-following-month re-stamp lands on/after the known BLS
    Employment Situation release dates for selected historical months.

    Known BLS release dates used as ground truth:
        Jan-2021 rate -> published 2021-02-05 (first Friday of Feb 2021)
        Sep-2021 rate -> published 2021-10-01 (first Friday of Oct 2021 = Oct 1 itself)
        Jan-2008 rate -> published 2008-02-01 (first Friday of Feb 2008 = Feb 1 itself)

    This test is marked 'slow' and requires a valid FRED_API_KEY environment variable
    (or a .env file in the project root readable by python-dotenv).
    """
    import os

    from dotenv import load_dotenv
    from fredapi import Fred

    load_dotenv()

    from finance.gtt import _first_friday_of_following_month

    api_key = os.environ.get("FRED_API_KEY", "")
    fred = Fred(api_key=api_key) if api_key else Fred()

    unrate: pd.Series = fred.get_series(
        "UNRATE", observation_start="2007-01-01", observation_end="2022-01-01"
    )

    # 1. Every observation must be indexed at the reference-month start (day == 1).
    assert (unrate.index.day == 1).all(), (
        "FRED UNRATE index does not consist entirely of month-start dates. "
        "The Option-B re-stamp offset in compute_ue_signal must be re-verified."
    )

    # 2. Print last 6 index dates vs their re-stamped publication dates (human-readable).
    last6 = unrate.tail(6)
    print("\n--- FRED UNRATE: last 6 reference-month dates vs. publication dates ---")
    for ref_date in last6.index:
        pub_date = _first_friday_of_following_month(pd.Timestamp(ref_date))
        print(f"  ref {ref_date.date()}  ->  pub {pub_date.date()}")

    # 3. Spot-check known BLS release dates: re-stamp must land on or after the true date.
    # Ground truth from BLS Employment Situation release calendar.
    known_releases: dict[str, str] = {
        "2021-01-01": "2021-02-05",  # Jan-2021 rate published Feb 5, 2021
        "2021-09-01": "2021-10-01",  # Sep-2021 rate published Oct 1, 2021
        "2008-01-01": "2008-02-01",  # Jan-2008 rate published Feb 1, 2008
    }
    for ref_str, true_pub_str in known_releases.items():
        ref_ts = pd.Timestamp(ref_str)
        computed_pub = _first_friday_of_following_month(ref_ts)
        true_pub = pd.Timestamp(true_pub_str)
        assert computed_pub >= true_pub, (
            f"Option-B re-stamp {computed_pub.date()} for reference month {ref_ts.date()} "
            f"is BEFORE the true BLS release date {true_pub.date()}. "
            "This would introduce a look-ahead leak."
        )
        print(
            f"  [OK] ref {ref_ts.date()} -> computed_pub {computed_pub.date()} "
            f">= true_pub {true_pub.date()}"
        )

    # 4. Smoke-test the full pipeline for a short recent window.
    result = fetch_gtt_signal_data(
        start_date="2020-01-02",
        end_date="2021-12-31",
        vix_p90_threshold=0.272,
    )
    assert isinstance(result, GttSignalData)
    assert not result.position_mask.empty
    assert set(result.position_mask.unique().tolist()).issubset({0, 1})
    print(f"\n  [OK] Live smoke test: {len(result.position_mask)} daily observations returned.")
