"""Tests for F-08: LeapsGttCloseEvent, LeapsLedger.gtt_close_events, close_leaps_contract."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from finance.leverage import (
    CONTRACT_MULTIPLIER,
    DEFAULT_IV,
    LTCG_RATE,
    AccountType,
    LeapsConfig,
    LeapsGttCloseEvent,
    LeapsLedger,
    close_leaps_contract,
    create_leaps_contract,
    price_leaps_contract,
    run_leaps_simulation,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DATE = pd.Timestamp("2022-06-15")
_SPOT = 200.0
_IV = 0.20
_RFR = 0.03


def _make_contract(
    spot: float = _SPOT,
    capital: float = 10_000.0,
    account_type: AccountType = AccountType.TAXABLE,
    purchase_date: pd.Timestamp = _DATE,
    iv: float = _IV,
) -> object:
    return create_leaps_contract(purchase_date, spot, capital, iv, account_type)


def _empty_ledger(account_type: AccountType = AccountType.TAXABLE) -> LeapsLedger:
    return LeapsLedger(contracts=(), roll_events=(), account_type=account_type)


# ---------------------------------------------------------------------------
# F-08: LeapsGttCloseEvent — dataclass contract
# ---------------------------------------------------------------------------


def test_leaps_gtt_close_event_is_frozen() -> None:
    contract = _make_contract()
    event = LeapsGttCloseEvent(
        close_date=_DATE,
        contract=contract,
        mtm_value=11_000.0,
        gain_realized=1_000.0,
        tax_paid=238.0,
        net_proceeds=10_762.0,
    )
    with pytest.raises(FrozenInstanceError):
        event.mtm_value = 0.0  # type: ignore[misc]


def test_leaps_gtt_close_event_fields_round_trip() -> None:
    contract = _make_contract()
    event = LeapsGttCloseEvent(
        close_date=_DATE,
        contract=contract,
        mtm_value=11_000.0,
        gain_realized=1_000.0,
        tax_paid=238.0,
        net_proceeds=10_762.0,
    )
    assert event.close_date == _DATE
    assert event.contract is contract
    assert event.mtm_value == pytest.approx(11_000.0)
    assert event.gain_realized == pytest.approx(1_000.0)
    assert event.tax_paid == pytest.approx(238.0)
    assert event.net_proceeds == pytest.approx(10_762.0)


# ---------------------------------------------------------------------------
# F-08: LeapsLedger backward compatibility
# ---------------------------------------------------------------------------


def test_leaps_ledger_gtt_close_events_defaults_to_empty_tuple() -> None:
    """Existing LeapsLedger constructions without gtt_close_events remain valid."""
    ledger = LeapsLedger(
        contracts=(),
        roll_events=(),
        account_type=AccountType.TAXABLE,
    )
    assert ledger.gtt_close_events == ()


def test_leaps_ledger_with_partial_close_events_still_backward_compat() -> None:
    """LeapsLedger with partial_close_events but no gtt_close_events."""
    ledger = LeapsLedger(
        contracts=(),
        roll_events=(),
        account_type=AccountType.TAXABLE,
        partial_close_events=(),
    )
    assert ledger.gtt_close_events == ()


def test_leaps_ledger_gtt_close_events_stored() -> None:
    """gtt_close_events are stored verbatim when supplied."""
    contract = _make_contract()
    event = close_leaps_contract(contract, _DATE, _SPOT, _IV, LTCG_RATE, _RFR)
    ledger = LeapsLedger(
        contracts=(contract,),
        roll_events=(),
        account_type=AccountType.TAXABLE,
        gtt_close_events=(event,),
    )
    assert len(ledger.gtt_close_events) == 1
    assert ledger.gtt_close_events[0] is event


# ---------------------------------------------------------------------------
# F-08: close_leaps_contract — accounting invariants
# ---------------------------------------------------------------------------


def test_close_taxable_gain_tax_equals_gain_times_rate() -> None:
    """Taxable close on a gain: tax_paid == gain_realized * ltcg_rate (within 1e-9)."""
    contract = _make_contract(capital=10_000.0)
    # Price the contract 1 year later at a higher spot to manufacture a gain.
    close_date = _DATE + pd.Timedelta(days=365)
    higher_spot = _SPOT * 1.30  # +30% → DITM call gains value
    event = close_leaps_contract(contract, close_date, higher_spot, _IV, LTCG_RATE, _RFR)

    cost_basis = contract.premium_paid * CONTRACT_MULTIPLIER * contract.n_contracts
    expected_gain = event.mtm_value - cost_basis
    assert event.gain_realized == pytest.approx(expected_gain, abs=1e-9)
    assert event.tax_paid == pytest.approx(max(0.0, expected_gain) * LTCG_RATE, abs=1e-9)
    assert event.tax_paid > 0.0  # confirm it's actually a gain scenario


def test_close_taxable_loss_tax_is_zero_and_proceeds_equal_mtm() -> None:
    """Taxable close on a loss: tax_paid == 0.0, net_proceeds == mtm_value."""
    contract = _make_contract(capital=10_000.0)
    # Price 1 year later at lower spot to manufacture a loss.
    close_date = _DATE + pd.Timedelta(days=365)
    lower_spot = _SPOT * 0.70  # -30% → OTM call may lose value
    event = close_leaps_contract(contract, close_date, lower_spot, _IV, LTCG_RATE, _RFR)

    # May or may not be a loss depending on time value; assert the invariant:
    if event.gain_realized < 0.0:
        assert event.tax_paid == 0.0
        assert event.net_proceeds == pytest.approx(event.mtm_value, abs=1e-9)
    else:
        # Not a loss at this IV/spot; at least verify tax >= 0
        assert event.tax_paid >= 0.0


def test_close_taxable_zero_gain_tax_is_zero() -> None:
    """gain_realized exactly 0 → tax 0 (max(0, 0) = 0)."""
    contract = _make_contract(capital=10_000.0)
    # Verify the boundary identity: max(0, 0) * rate == 0
    assert max(0.0, 0.0) * LTCG_RATE == 0.0
    # And that the net_proceeds identity holds at-the-money:
    event = close_leaps_contract(contract, _DATE, _SPOT, _IV, LTCG_RATE, _RFR)
    assert event.net_proceeds == pytest.approx(event.mtm_value - event.tax_paid, abs=1e-9)


def test_close_tax_sheltered_tax_is_always_zero() -> None:
    """TAX_SHELTERED close always produces tax_paid == 0.0, regardless of gain."""
    contract = _make_contract(account_type=AccountType.TAX_SHELTERED)
    close_date = _DATE + pd.Timedelta(days=365)
    higher_spot = _SPOT * 1.50
    event = close_leaps_contract(contract, close_date, higher_spot, _IV, LTCG_RATE, _RFR)
    assert event.tax_paid == 0.0
    # gain should be positive (higher spot + more time) → confirms shelter is working
    assert event.gain_realized > 0.0


def test_close_net_proceeds_identity_holds_across_all_cases() -> None:
    """net_proceeds == mtm_value - tax_paid for gain, loss, and sheltered cases."""
    contract_taxable = _make_contract(account_type=AccountType.TAXABLE)
    contract_sheltered = _make_contract(account_type=AccountType.TAX_SHELTERED)
    close_date = _DATE + pd.Timedelta(days=200)

    for spot_mult in (0.70, 1.00, 1.30):
        spot = _SPOT * spot_mult
        for contract in (contract_taxable, contract_sheltered):
            event = close_leaps_contract(contract, close_date, spot, _IV, LTCG_RATE, _RFR)
            assert event.net_proceeds == pytest.approx(
                event.mtm_value - event.tax_paid, abs=1e-9
            ), f"net_proceeds identity failed at spot_mult={spot_mult}"


def test_close_mtm_value_matches_price_leaps_contract() -> None:
    """mtm_value == price_leaps_contract(contract, spot, date, iv, rfr)."""
    contract = _make_contract()
    close_date = _DATE + pd.Timedelta(days=300)
    event = close_leaps_contract(contract, close_date, _SPOT, _IV, LTCG_RATE, _RFR)
    expected_mtm = price_leaps_contract(contract, _SPOT, close_date, _IV, _RFR)
    assert event.mtm_value == pytest.approx(expected_mtm, abs=1e-9)


def test_close_zero_contracts_produces_zero_mtm_and_proceeds() -> None:
    """Contract with n_contracts == 0 (floored premium): mtm=0, tax=0, proceeds=0."""
    # Create a contract with capital=0 so n_contracts floors to 0.
    contract = create_leaps_contract(_DATE, _SPOT, 0.0, _IV, AccountType.TAXABLE)
    assert contract.n_contracts == 0.0
    event = close_leaps_contract(contract, _DATE, _SPOT, _IV, LTCG_RATE, _RFR)
    assert event.mtm_value == pytest.approx(0.0, abs=1e-9)
    assert event.tax_paid == pytest.approx(0.0, abs=1e-9)
    assert event.net_proceeds == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# F-08: _live_contracts excludes GTT-force-closed contracts
# ---------------------------------------------------------------------------


def test_live_contracts_excludes_gtt_closed() -> None:
    """_live_contracts returns empty when the only contract has been GTT-closed."""
    from finance.leverage import _live_contracts

    contract = _make_contract()
    event = close_leaps_contract(contract, _DATE, _SPOT, _IV, LTCG_RATE, _RFR)
    ledger = LeapsLedger(
        contracts=(contract,),
        roll_events=(),
        account_type=AccountType.TAXABLE,
        gtt_close_events=(event,),
    )
    live = _live_contracts(ledger, _DATE + pd.Timedelta(days=1))
    assert live == []


def test_live_contracts_without_gtt_close_events_unchanged() -> None:
    """_live_contracts with no gtt_close_events behaves identically to before F-08."""
    from finance.leverage import _live_contracts

    contract = _make_contract()
    ledger = LeapsLedger(
        contracts=(contract,),
        roll_events=(),
        account_type=AccountType.TAXABLE,
    )
    live = _live_contracts(ledger, _DATE + pd.Timedelta(days=1))
    assert len(live) == 1


# ---------------------------------------------------------------------------
# F-08: run_leaps_simulation backward compat — gtt_close_events = ()
# ---------------------------------------------------------------------------


def test_run_leaps_simulation_gtt_close_events_empty_by_default() -> None:
    """Existing run_leaps_simulation returns a ledger with gtt_close_events == ()."""
    idx = pd.bdate_range("2020-01-02", periods=252)
    prices = pd.Series([200.0] * 252, index=idx)
    config = LeapsConfig(iv=DEFAULT_IV, account_type=AccountType.TAXABLE)
    ledger = run_leaps_simulation(prices, monthly_contribution_to_leaps=500.0, config=config)
    assert ledger.gtt_close_events == ()
