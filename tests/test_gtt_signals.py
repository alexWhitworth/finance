"""Tests for gtt.py pure signal functions and GttSignalData (F-03..F-06)."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from finance.gtt import (
    GttSignalData,
    _first_friday_of_following_month,
    compute_position_mask,
    compute_ue_signal,
    compute_vix_signal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _monthly(values: list[float], start: str = "2019-06-01") -> pd.Series:
    """Monthly UNRATE series indexed at reference-month start (FRED convention)."""
    idx = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series(values, index=idx, name="UNRATE")


# ---------------------------------------------------------------------------
# F-03: _first_friday_of_following_month (publication-date mapping)
# ---------------------------------------------------------------------------


def test_first_friday_hand_verified() -> None:
    # Jan-2021 reference month -> published 1st Friday of Feb 2021.
    # Feb 1 2021 is a Monday, so the first Friday is Feb 5 2021.
    assert _first_friday_of_following_month(pd.Timestamp("2021-01-01")) == pd.Timestamp(
        "2021-02-05"
    )
    # Following month begins ON a Friday -> that day is the first Friday.
    # Oct 1 2021 is a Friday, so Sep-2021 reference -> 2021-10-01.
    assert _first_friday_of_following_month(pd.Timestamp("2021-09-01")) == pd.Timestamp(
        "2021-10-01"
    )


# ---------------------------------------------------------------------------
# F-03: compute_ue_signal
# ---------------------------------------------------------------------------


def _step_up_unrate() -> pd.Series:
    """20 monthly obs: strictly declining, then a jump at the final month (2021-01)."""
    values = [6.0 - 0.05 * i for i in range(19)] + [10.0]
    return _monthly(values, start="2019-06-01")  # 2019-06 .. 2021-01


def test_ue_signal_anti_lookahead_publication_dated() -> None:
    """CRITICAL: the Jan-2021 step-up must not be visible before its Feb-05 publish."""
    unrate = _step_up_unrate()
    sig = compute_ue_signal(unrate)

    pub_date = pd.Timestamp("2021-02-05")  # first Friday of Feb 2021 (hand-verified)

    # 0 on every date strictly before the publication date...
    before = sig.loc[: pub_date - pd.Timedelta(days=1)]
    assert (before == 0).all()
    # ...and 1 from the publication date onward (last obs is the step-up).
    on_and_after = sig.loc[pub_date:]
    assert (on_and_after == 1).all()
    # Boundary: the day before publication is still 0.
    assert int(sig.loc[pd.Timestamp("2021-02-04")]) == 0
    assert int(sig.loc[pub_date]) == 1


def test_ue_signal_does_not_fire_within_reference_month() -> None:
    """The Jan-2021 rate must not fire anywhere inside January 2021 itself."""
    sig = compute_ue_signal(_step_up_unrate())
    jan_2021 = sig.loc["2021-01-01":"2021-01-31"]
    assert (jan_2021 == 0).all()


def test_ue_signal_domain_and_no_nan_after_warmup() -> None:
    sig = compute_ue_signal(_step_up_unrate())
    assert set(np.unique(sig.to_numpy())).issubset({0, 1})
    assert not sig.isna().any()
    assert sig.dtype == int or np.issubdtype(sig.dtype, np.integer)


def test_ue_signal_strictly_declining_never_fires() -> None:
    unrate = _monthly([8.0 - 0.1 * i for i in range(24)])
    sig = compute_ue_signal(unrate)
    assert (sig == 0).all()


def test_ue_signal_equality_counts_as_fire() -> None:
    # Constant series: once the MA is defined, UNRATE == MA -> flag 1 (>=).
    unrate = _monthly([5.0] * 24)
    sig = compute_ue_signal(unrate)
    # The final (constant) observation publishes on the first Friday after its month.
    assert int(sig.iloc[-1]) == 1


def test_ue_signal_empty_raises() -> None:
    with pytest.raises(ValueError, match="unrate is empty"):
        compute_ue_signal(pd.Series(dtype=float))


def test_ue_signal_too_few_observations_raises() -> None:
    unrate = _monthly([5.0] * 6)  # fewer than 12
    with pytest.raises(ValueError, match="need at least rolling_window_months"):
        compute_ue_signal(unrate)


# ---------------------------------------------------------------------------
# F-04: compute_vix_signal
# ---------------------------------------------------------------------------


def _daily_vix(values: list[float], start: str = "2021-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, name="VIX")


def test_vix_signal_fires_on_fifth_consecutive_day_not_fourth() -> None:
    # 4 days below, then 5 days above threshold.
    vix = _daily_vix([0.10] * 4 + [0.30] * 5)
    sig = compute_vix_signal(vix, threshold=0.272, consecutive_days=5)
    # Days 0..7 are 0; the fifth consecutive above-day (index 8) fires.
    assert int(sig.iloc[7]) == 0  # only 4 consecutive above by index 7
    assert int(sig.iloc[8]) == 1  # fifth consecutive above


def test_vix_signal_four_consecutive_never_fires() -> None:
    vix = _daily_vix([0.30] * 4 + [0.10] * 5)
    sig = compute_vix_signal(vix, threshold=0.272, consecutive_days=5)
    assert (sig == 0).all()


def test_vix_signal_equality_counts_as_above() -> None:
    # Exactly at threshold for 5 days -> fires (>=).
    vix = _daily_vix([0.272] * 5)
    sig = compute_vix_signal(vix, threshold=0.272, consecutive_days=5)
    assert int(sig.iloc[4]) == 1


def test_vix_signal_shorter_than_window_all_zero() -> None:
    vix = _daily_vix([0.30] * 3)
    sig = compute_vix_signal(vix, threshold=0.272, consecutive_days=5)
    assert (sig == 0).all()


def test_vix_signal_consecutive_days_below_one_raises() -> None:
    vix = _daily_vix([0.30] * 5)
    with pytest.raises(ValueError, match="consecutive_days must be >= 1"):
        compute_vix_signal(vix, threshold=0.272, consecutive_days=0)


def test_vix_signal_empty_raises() -> None:
    with pytest.raises(ValueError, match="vix is empty"):
        compute_vix_signal(pd.Series(dtype=float), threshold=0.272)


def test_vix_signal_lower_threshold_never_reduces_fire_count() -> None:
    vix = _daily_vix([0.10, 0.25, 0.30, 0.28, 0.35, 0.40, 0.20, 0.31, 0.33, 0.34])
    high = compute_vix_signal(vix, threshold=0.30, consecutive_days=3).sum()
    low = compute_vix_signal(vix, threshold=0.20, consecutive_days=3).sum()
    assert low >= high


def test_vix_signal_domain() -> None:
    vix = _daily_vix([0.10, 0.30, 0.30, 0.30, 0.30, 0.30, 0.10])
    sig = compute_vix_signal(vix, threshold=0.272, consecutive_days=5)
    assert set(np.unique(sig.to_numpy())).issubset({0, 1})


# ---------------------------------------------------------------------------
# F-05: compute_position_mask
# ---------------------------------------------------------------------------


def _biz(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


def test_position_mask_defensive_branch_with_lag() -> None:
    """recession_risk AND price < SMA -> Defensive, observed one trading day later."""
    idx = _biz("2021-01-04", 8)
    prices = pd.Series([100.0, 99, 98, 97, 96, 95, 94, 93], index=idx)  # strictly down
    ue = pd.Series([1] * 8, index=idx)  # recession risk every day
    vix = pd.Series([0] * 8, index=idx)

    mask = compute_position_mask(ue, vix, prices, sma_window=3)

    # position_today = [1,1,0,0,0,0,0,0] (idx 0,1 warm-up -> Long; idx>=2 below SMA).
    # After the shared 1-day execution lag (shift 1, first day defaults Long):
    expected = [1, 1, 1, 0, 0, 0, 0, 0]
    assert mask.tolist() == expected


def test_position_mask_no_recession_all_long() -> None:
    idx = _biz("2021-01-04", 8)
    prices = pd.Series([100.0, 99, 98, 97, 96, 95, 94, 93], index=idx)
    ue = pd.Series([0] * 8, index=idx)
    vix = pd.Series([0] * 8, index=idx)
    mask = compute_position_mask(ue, vix, prices, sma_window=3)
    assert (mask == 1).all()


def test_position_mask_price_above_sma_all_long() -> None:
    idx = _biz("2021-01-04", 8)
    prices = pd.Series([90.0, 91, 92, 93, 94, 95, 96, 97], index=idx)  # strictly up
    ue = pd.Series([1] * 8, index=idx)
    vix = pd.Series([1] * 8, index=idx)
    mask = compute_position_mask(ue, vix, prices, sma_window=3)
    assert (mask == 1).all()


def test_position_mask_sma_warmup_stays_long_even_when_signal_active() -> None:
    idx = _biz("2021-01-04", 5)
    prices = pd.Series([100.0, 99, 98, 97, 96], index=idx)
    ue = pd.Series([1] * 5, index=idx)
    vix = pd.Series([1] * 5, index=idx)
    # sma_window larger than the series -> SMA never computable -> always Long.
    mask = compute_position_mask(ue, vix, prices, sma_window=10)
    assert (mask == 1).all()


def test_position_mask_vix_branch_fires_defensive() -> None:
    """VIX alone (UE off) also drives the defensive branch (OR logic)."""
    idx = _biz("2021-01-04", 6)
    prices = pd.Series([100.0, 99, 98, 97, 96, 95], index=idx)
    ue = pd.Series([0] * 6, index=idx)
    vix = pd.Series([0, 0, 1, 1, 1, 1], index=idx)
    mask = compute_position_mask(ue, vix, prices, sma_window=3)
    # position_today: idx0,1 warm-up->Long; idx2 vix=1 & below SMA->0; idx3,4,5->0.
    # today = [1,1,0,0,0,0]; lag+fill -> [1,1,1,0,0,0]
    assert mask.tolist() == [1, 1, 1, 0, 0, 0]


def test_position_mask_domain() -> None:
    idx = _biz("2021-01-04", 8)
    prices = pd.Series([100.0, 99, 98, 97, 96, 95, 94, 93], index=idx)
    ue = pd.Series([1, 0, 1, 0, 1, 0, 1, 0], index=idx)
    vix = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], index=idx)
    mask = compute_position_mask(ue, vix, prices, sma_window=3)
    assert set(np.unique(mask.to_numpy())).issubset({0, 1})
    assert mask.index.equals(prices.index)


def test_position_mask_empty_prices_raises() -> None:
    idx = _biz("2021-01-04", 3)
    ue = pd.Series([1, 1, 1], index=idx)
    vix = pd.Series([0, 0, 0], index=idx)
    with pytest.raises(ValueError, match="equity_prices is empty"):
        compute_position_mask(ue, vix, pd.Series(dtype=float), sma_window=3)


def test_position_mask_misaligned_indexes_raise() -> None:
    ue = pd.Series([1, 1, 1], index=_biz("2019-01-02", 3))
    vix = pd.Series([0, 0, 0], index=_biz("2019-01-02", 3))
    prices = pd.Series([100.0, 99, 98], index=_biz("2021-01-04", 3))
    with pytest.raises(ValueError, match="indexes do not overlap"):
        compute_position_mask(ue, vix, prices, sma_window=2)


# ---------------------------------------------------------------------------
# F-06: GttSignalData
# ---------------------------------------------------------------------------


def _sig_data() -> GttSignalData:
    idx = _biz("2021-01-04", 3)
    mask = pd.Series([1, 0, 1], index=idx)
    ue = pd.Series([0, 1, 1], index=idx)
    vix = pd.Series([1, 0, 0], index=idx)
    return GttSignalData(
        position_mask=mask,
        ue_signal=ue,
        vix_signal=vix,
        vix_p90_threshold=0.272,
        unrate_start=pd.Timestamp("2019-06-01"),
        vix_start=pd.Timestamp("1993-01-01"),
    )


def test_gttsignaldata_field_roundtrip() -> None:
    d = _sig_data()
    assert d.vix_p90_threshold == 0.272
    assert d.unrate_start == pd.Timestamp("2019-06-01")
    assert d.vix_start == pd.Timestamp("1993-01-01")
    assert d.position_mask.tolist() == [1, 0, 1]
    assert d.ue_signal.tolist() == [0, 1, 1]
    assert d.vix_signal.tolist() == [1, 0, 0]


def test_gttsignaldata_is_frozen() -> None:
    d = _sig_data()
    with pytest.raises(FrozenInstanceError):
        d.vix_p90_threshold = 0.30  # type: ignore[misc]


def test_gttsignaldata_constructs_with_empty_series() -> None:
    empty = pd.Series(dtype=int)
    d = GttSignalData(
        position_mask=empty,
        ue_signal=empty,
        vix_signal=empty,
        vix_p90_threshold=0.272,
        unrate_start=pd.Timestamp("2019-06-01"),
        vix_start=pd.Timestamp("1993-01-01"),
    )
    assert d.position_mask.empty
