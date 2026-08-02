"""Tests for PortfolioState (F-001), DayInputs (F-002), and BacktestContext (F-003) frozen dataclasses."""

import dataclasses

import pandas as pd
import pytest

from finance.leverage import AccountType, LeapsContract, LeapsGttCloseEvent, LeapsLedger
from finance.portfolio import DayInputs, PortfolioState


# ---------------------------------------------------------------------------
# PortfolioState (F-001)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_contract() -> LeapsContract:
    """Minimal LeapsContract for use in PortfolioState fixtures."""
    return LeapsContract(
        purchase_date=pd.Timestamp("2023-01-03"),
        expiry_date=pd.Timestamp("2025-01-17"),
        strike=160.0,
        spot_at_purchase=200.0,
        premium_paid=45.0,
        notional=20000.0,
        n_contracts=1.0,
        account_type=AccountType.TAXABLE,
    )


@pytest.fixture
def sample_ledger(sample_contract: LeapsContract) -> LeapsLedger:
    """Minimal LeapsLedger with one contract."""
    return LeapsLedger(
        contracts=(sample_contract,),
        roll_events=(),
        account_type=AccountType.TAXABLE,
    )


@pytest.fixture
def sample_gtt_close(sample_contract: LeapsContract) -> LeapsGttCloseEvent:
    """Minimal LeapsGttCloseEvent for use in PortfolioState fixtures."""
    return LeapsGttCloseEvent(
        close_date=pd.Timestamp("2023-06-01"),
        contract=sample_contract,
        mtm_value=5000.0,
        gain_realized=500.0,
        tax_paid=100.0,
        net_proceeds=4900.0,
    )


@pytest.fixture
def full_state(
    sample_contract: LeapsContract,
    sample_ledger: LeapsLedger,
    sample_gtt_close: LeapsGttCloseEvent,
) -> PortfolioState:
    """PortfolioState with non-trivial values for all 11 fields."""
    return PortfolioState(
        holdings={"VTI": 50000.0, "VXUS": 20000.0},
        defensive_sleeve=10000.0,
        leaps_pool=5000.0,
        leaps_value=7500.0,
        prev_total_nav=92500.0,
        prev_regime=1,
        prev_date_ts=pd.Timestamp("2023-05-31"),
        leaps_ledger=sample_ledger,
        leaps_scale={sample_contract: 0.8},
        all_window_ledgers=(sample_ledger,),
        all_gtt_closes=(sample_gtt_close,),
    )


def test_all_fields_accessible(
    full_state: PortfolioState,
    sample_contract: LeapsContract,
    sample_ledger: LeapsLedger,
    sample_gtt_close: LeapsGttCloseEvent,
) -> None:
    """All 11 fields are accessible and equal injected values."""
    assert full_state.holdings == {"VTI": 50000.0, "VXUS": 20000.0}
    assert full_state.defensive_sleeve == 10000.0
    assert full_state.leaps_pool == 5000.0
    assert full_state.leaps_value == 7500.0
    assert full_state.prev_total_nav == 92500.0
    assert full_state.prev_regime == 1
    assert full_state.prev_date_ts == pd.Timestamp("2023-05-31")
    assert full_state.leaps_ledger is sample_ledger
    assert full_state.leaps_scale == {sample_contract: 0.8}
    assert full_state.all_window_ledgers == (sample_ledger,)
    assert full_state.all_gtt_closes == (sample_gtt_close,)


def test_none_optional_fields() -> None:
    """prev_date_ts and leaps_ledger accept None."""
    state = PortfolioState(
        holdings={"VTI": 100000.0},
        defensive_sleeve=0.0,
        leaps_pool=0.0,
        leaps_value=0.0,
        prev_total_nav=100000.0,
        prev_regime=1,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )
    assert state.prev_date_ts is None
    assert state.leaps_ledger is None


def test_frozen_assignment_raises(full_state: PortfolioState) -> None:
    """Assigning to any field raises FrozenInstanceError."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        full_state.holdings = {}  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        full_state.prev_regime = 0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        full_state.prev_total_nav = 0.0  # type: ignore[misc]


def test_empty_holdings() -> None:
    """holdings={} is a valid edge case (no base assets allocated)."""
    state = PortfolioState(
        holdings={},
        defensive_sleeve=0.0,
        leaps_pool=0.0,
        leaps_value=0.0,
        prev_total_nav=0.0,
        prev_regime=1,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )
    assert state.holdings == {}


def test_empty_accumulators() -> None:
    """all_window_ledgers=() and all_gtt_closes=() are valid (no GTT activity)."""
    state = PortfolioState(
        holdings={"VTI": 100000.0},
        defensive_sleeve=0.0,
        leaps_pool=0.0,
        leaps_value=0.0,
        prev_total_nav=100000.0,
        prev_regime=1,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )
    assert state.all_window_ledgers == ()
    assert state.all_gtt_closes == ()


# ---------------------------------------------------------------------------
# DayInputs (F-002)
# ---------------------------------------------------------------------------


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
