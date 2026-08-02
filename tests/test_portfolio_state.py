"""Tests for PortfolioState frozen dataclass (F-001).

Verifies:
- All 11 fields are accessible after construction and match injected values.
- The dataclass is frozen (assignment raises FrozenInstanceError).
- Edge cases: empty holdings dict, empty all_window_ledgers/all_gtt_closes tuples.
"""

import dataclasses

import pandas as pd
import pytest

from finance.leverage import AccountType, LeapsContract, LeapsGttCloseEvent, LeapsLedger
from finance.portfolio import PortfolioState


# ---------------------------------------------------------------------------
# Fixtures
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


# ---------------------------------------------------------------------------
# Construction + field access
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_frozen_assignment_raises(full_state: PortfolioState) -> None:
    """Assigning to any field raises FrozenInstanceError."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        full_state.holdings = {}  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        full_state.prev_regime = 0  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        full_state.prev_total_nav = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


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
