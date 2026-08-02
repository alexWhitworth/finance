"""Tests for F-008: _apply_gtt_force_close.

Covers:
  - No-op: not a transition day (same regime)
  - No-op: leaps_ledger is None
  - No-op: GTT inactive (gtt_active=False)
  - Force-close fires and Invariant A3 holds
  - Scale (leaps_scale fraction) is applied to n_contracts
  - TAX_SHELTERED account: tax_paid == 0.0
  - Hypothesis property: A3 holds over varied spot / n_contracts
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from finance._step_f008 import _apply_gtt_force_close
from finance.leverage import (
    AccountType,
    LeapsConfig,
    LeapsContract,
    LeapsLedger,
    RebalanceRule,
    WeightStrategy,
)
from finance.portfolio import (
    BacktestContext,
    DayInputs,
    PortfolioConfig,
    PortfolioState,
)
from finance.returns import ReturnData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = pd.Timestamp("2020-01-02")  # prev_date_ts (last Long day)
T1 = pd.Timestamp("2020-01-03")  # current day (first Defensive day)

DATES = pd.bdate_range("2019-06-03", periods=400)


def make_contract(
    *,
    purchase_date: pd.Timestamp = pd.Timestamp("2020-01-02"),
    expiry: pd.Timestamp = pd.Timestamp("2022-01-21"),
    strike: float = 160.0,
    spot: float = 200.0,
    premium: float = 45.0,
    notional: float = 20000.0,
    n: float = 1.0,
    account_type: AccountType = AccountType.TAXABLE,
) -> LeapsContract:
    """Construct a minimal LeapsContract."""
    return LeapsContract(
        purchase_date=purchase_date,
        expiry_date=expiry,
        strike=strike,
        spot_at_purchase=spot,
        premium_paid=premium,
        notional=notional,
        n_contracts=n,
        account_type=account_type,
    )


def make_ledger(contract: LeapsContract) -> LeapsLedger:
    """Construct a minimal LeapsLedger with one contract."""
    return LeapsLedger(
        contracts=(contract,),
        roll_events=(),
        account_type=contract.account_type,
    )


def _make_config(*, with_leaps: bool = True) -> PortfolioConfig:
    """Build a minimal PortfolioConfig."""
    weights = {"VTI": 0.85, "VTI_LEAPS": 0.15} if with_leaps else {"VTI": 1.0}
    return PortfolioConfig(
        target_weights=weights,
        initial_nav=100_000.0,
        monthly_contribution=500.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(iv=0.20, ltcg_rate=0.238) if with_leaps else None,
    )


def _make_return_data() -> ReturnData:
    """Build minimal ReturnData over DATES."""
    rng = np.random.default_rng(42)
    simple = rng.normal(0.0003, 0.01, len(DATES))
    returns = pd.DataFrame({"VTI": simple}, index=DATES)
    log_returns = pd.DataFrame({"VTI": np.log1p(simple)}, index=DATES)
    rfr = pd.Series(0.04, index=DATES, name="risk_free_rate")
    return ReturnData(
        returns=returns,
        log_returns=log_returns,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr,
    )


def minimal_ctx(
    *,
    gtt_active: bool = True,
    with_leaps: bool = True,
    spot: float = 210.0,
    raw_vix_value: float = 0.25,
) -> BacktestContext:
    """Construct a minimal BacktestContext with underlying_prices and raw_vix populated."""
    config = _make_config(with_leaps=with_leaps)
    return_data = _make_return_data()
    underlying_prices = pd.Series(spot, index=DATES, name="VTI")
    raw_vix = pd.Series(raw_vix_value, index=DATES, name="VIX")
    rfr_series = pd.Series(0.04, index=DATES, name="rfr")
    return BacktestContext(
        base_assets=("VTI",),
        leaps_keys=("VTI_LEAPS",) if with_leaps else (),
        leaps_fraction=0.15 if with_leaps else 0.0,
        base_target_w=pd.Series({"VTI": 1.0}),
        governed_base=("VTI",),
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0},
        use_leaps=with_leaps,
        iv=0.20,
        leaps_monthly=500.0 * 0.15 if with_leaps else 0.0,
        base_contribution=500.0 * (0.85 if with_leaps else 1.0),
        config=config,
        return_data=return_data,
        underlying_prices=underlying_prices if with_leaps else None,
        raw_vix=raw_vix,
        mtm_iv_series=None,
        rfr_series=rfr_series,
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end={},
        w=pd.Series({"VTI": 0.85, "VTI_LEAPS": 0.15} if with_leaps else {"VTI": 1.0}),
    )


def minimal_state(
    *,
    prev_regime: int = 1,
    leaps_ledger: LeapsLedger | None = None,
    leaps_scale: dict[LeapsContract, float] | None = None,
    leaps_pool: float = 0.0,
) -> PortfolioState:
    """Construct a minimal PortfolioState suitable for F-008 tests."""
    return PortfolioState(
        holdings={"VTI": 85_000.0},
        defensive_sleeve=0.0,
        leaps_pool=leaps_pool,
        leaps_value=0.0,
        prev_total_nav=100_000.0,
        prev_regime=prev_regime,
        prev_date_ts=T0,
        leaps_ledger=leaps_ledger,
        leaps_scale=leaps_scale or {},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )


def transition_inputs(*, regime_t: int = 0) -> DayInputs:
    """Build DayInputs for a Long->Defensive transition day (regime_t=0 by default)."""
    return DayInputs(
        date_ts=T1,
        day_ret=pd.Series({"VTI": 0.0}),
        regime_t=regime_t,
        def_gross_return=0.001,
        spot=210.0,
        raw_vix_value=0.25,
        mtm_iv_value=0.22,
        rfr=0.04,
        is_month_end=False,
        is_rebal_date=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_op_not_a_transition_day() -> None:
    """When prev_regime==1 and regime_t==1, state is returned unchanged."""
    contract = make_contract()
    ledger = make_ledger(contract)
    state = minimal_state(prev_regime=1, leaps_ledger=ledger)
    ctx = minimal_ctx()
    inputs = transition_inputs(regime_t=1)  # Long day, NOT a transition

    new_state = _apply_gtt_force_close(state, inputs, ctx)

    assert new_state is state


def test_no_op_leaps_ledger_none() -> None:
    """When leaps_ledger is None on a transition day, state is returned unchanged."""
    state = minimal_state(prev_regime=1, leaps_ledger=None)
    ctx = minimal_ctx()
    inputs = transition_inputs(regime_t=0)

    new_state = _apply_gtt_force_close(state, inputs, ctx)

    assert new_state is state


def test_no_op_gtt_inactive() -> None:
    """When gtt_active=False, state is returned unchanged regardless of regime."""
    contract = make_contract()
    ledger = make_ledger(contract)
    state = minimal_state(prev_regime=1, leaps_ledger=ledger)
    ctx = minimal_ctx(gtt_active=False)
    inputs = transition_inputs(regime_t=0)

    new_state = _apply_gtt_force_close(state, inputs, ctx)

    assert new_state is state


def test_force_close_fires_a3_invariant() -> None:
    """Invariant A3: new_leaps_pool - old_leaps_pool == new_event.net_proceeds."""
    contract = make_contract()
    ledger = make_ledger(contract)
    state = minimal_state(prev_regime=1, leaps_ledger=ledger, leaps_pool=1000.0)
    ctx = minimal_ctx()
    inputs = transition_inputs(regime_t=0)

    new_state = _apply_gtt_force_close(state, inputs, ctx)

    assert len(new_state.all_gtt_closes) == len(state.all_gtt_closes) + 1
    evt = new_state.all_gtt_closes[-1]
    pool_delta = new_state.leaps_pool - state.leaps_pool
    assert abs(pool_delta - evt.net_proceeds) < 1e-9


def test_scale_applied_to_n_contracts() -> None:
    """leaps_scale={contract: 0.5} halves n_contracts in the close event."""
    contract = make_contract(n=2.0)
    ledger = make_ledger(contract)
    state = minimal_state(
        prev_regime=1,
        leaps_ledger=ledger,
        leaps_scale={contract: 0.5},
    )
    ctx = minimal_ctx()
    inputs = transition_inputs(regime_t=0)

    new_state = _apply_gtt_force_close(state, inputs, ctx)

    assert len(new_state.all_gtt_closes) == 1
    closed_contract = new_state.all_gtt_closes[0].contract
    assert abs(closed_contract.n_contracts - 2.0 * 0.5) < 1e-12


def test_tax_sheltered_tax_paid_zero() -> None:
    """TAX_SHELTERED account: tax_paid == 0.0 for the GTT close event."""
    contract = make_contract(account_type=AccountType.TAX_SHELTERED)
    ledger = LeapsLedger(
        contracts=(contract,),
        roll_events=(),
        account_type=AccountType.TAX_SHELTERED,
    )
    state = minimal_state(prev_regime=1, leaps_ledger=ledger)
    ctx = minimal_ctx()
    inputs = transition_inputs(regime_t=0)

    new_state = _apply_gtt_force_close(state, inputs, ctx)

    assert len(new_state.all_gtt_closes) == 1
    evt = new_state.all_gtt_closes[0]
    assert evt.tax_paid == 0.0


@given(
    spot=st.floats(min_value=50.0, max_value=600.0, allow_nan=False, allow_infinity=False),
    n_contracts=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_hypothesis_a3_invariant(spot: float, n_contracts: float) -> None:
    """Invariant A3 holds over varied spot prices and contract counts."""
    contract = make_contract(spot=spot, n=n_contracts)
    ledger = make_ledger(contract)
    initial_pool = 500.0
    state = minimal_state(
        prev_regime=1,
        leaps_ledger=ledger,
        leaps_pool=initial_pool,
    )
    ctx = minimal_ctx(spot=spot)
    inputs = transition_inputs(regime_t=0)

    new_state = _apply_gtt_force_close(state, inputs, ctx)

    new_events = new_state.all_gtt_closes[len(state.all_gtt_closes):]
    expected_pool = initial_pool + sum(e.net_proceeds for e in new_events)
    assert abs(new_state.leaps_pool - expected_pool) < 1e-9
