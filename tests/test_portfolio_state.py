"""Tests for DayInputs frozen dataclass (F-002)."""

import dataclasses

import pandas as pd
import pytest

from finance.portfolio import DayInputs


@pytest.fixture
def day_inputs_full() -> DayInputs:
    """DayInputs with all optional fields populated."""
    return DayInputs(
        date_ts=pd.Timestamp("2023-03-31"),
        day_ret=pd.Series({"VTI": 0.01, "VXUS": -0.005}),
        regime_t=1,
        def_gross_return=0.002,
        spot=205.50,
        raw_vix_value=0.185,
        mtm_iv_value=0.192,
        rfr=0.05,
        is_month_end=True,
        is_rebal_date=True,
    )


def test_day_inputs_fields(day_inputs_full: DayInputs) -> None:
    """All 10 fields are accessible and equal their injected values."""
    d = day_inputs_full
    assert d.date_ts == pd.Timestamp("2023-03-31")
    assert float(d.day_ret["VTI"]) == pytest.approx(0.01)
    assert float(d.day_ret["VXUS"]) == pytest.approx(-0.005)
    assert d.regime_t == 1
    assert d.def_gross_return == pytest.approx(0.002)
    assert d.spot == pytest.approx(205.50)
    assert d.raw_vix_value == pytest.approx(0.185)
    assert d.mtm_iv_value == pytest.approx(0.192)
    assert d.rfr == pytest.approx(0.05)
    assert d.is_month_end is True
    assert d.is_rebal_date is True


def test_day_inputs_frozen(day_inputs_full: DayInputs) -> None:
    """Assignment to any field raises FrozenInstanceError."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        day_inputs_full.regime_t = 0  # type: ignore[misc]


def test_day_inputs_none_optional_fields() -> None:
    """Optional fields accept None (no LEAPS, no vol_prices, warmup period)."""
    d = DayInputs(
        date_ts=pd.Timestamp("2023-01-03"),
        day_ret=pd.Series({"VTI": 0.0}),
        regime_t=1,
        def_gross_return=0.0,
        spot=None,
        raw_vix_value=None,
        mtm_iv_value=None,
        rfr=0.04,
        is_month_end=False,
        is_rebal_date=False,
    )
    assert d.spot is None
    assert d.raw_vix_value is None
    assert d.mtm_iv_value is None
    assert d.is_month_end is False
    assert d.is_rebal_date is False
