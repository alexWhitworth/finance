"""Tests for F-010: _compute_leaps_mtm step function (Bug 1 fix).

Covers all suppression conditions, normal MTM computation, IV fallback,
leaps_scale application, and a Hypothesis property test.
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from finance.leverage import (
    AccountType,
    LeapsConfig,
    LeapsContract,
    LeapsLedger,
    RebalanceRule,
    WeightStrategy,
    price_leaps_contract,
)
from finance.portfolio import BacktestContext, DayInputs, PortfolioConfig, PortfolioState
from finance.returns import ReturnData
from finance._step_f010 import _compute_leaps_mtm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATE_PURCHASE = pd.Timestamp("2023-01-03")
_DATE_EXPIRY = pd.Timestamp("2025-01-17")
_DATE_MTM = pd.Timestamp("2023-06-01")


def make_contract(
    purchase_date: pd.Timestamp = _DATE_PURCHASE,
    expiry: pd.Timestamp = _DATE_EXPIRY,
    strike: float = 160.0,
    spot: float = 200.0,
    premium: float = 45.0,
    notional: float = 20000.0,
    n: float = 1.0,
) -> LeapsContract:
    """Build a minimal LeapsContract for testing.

    Arguments:
        purchase_date: Trade date.
        expiry: Option expiry date.
        strike: Strike price.
        spot: Spot price at purchase.
        premium: Premium paid.
        notional: Notional value.
        n: Number of contracts.

    Returns:
        Fully populated LeapsContract.
    """
    return LeapsContract(
        purchase_date=purchase_date,
        expiry_date=expiry,
        strike=strike,
        spot_at_purchase=spot,
        premium_paid=premium,
        notional=notional,
        n_contracts=n,
        account_type=AccountType.TAXABLE,
    )


def make_ledger(contract: LeapsContract) -> LeapsLedger:
    """Build a minimal LeapsLedger containing a single contract.

    Arguments:
        contract: The LeapsContract to include.

    Returns:
        LeapsLedger with one contract and no events.
    """
    return LeapsLedger(
        contracts=(contract,),
        roll_events=(),
        account_type=AccountType.TAXABLE,
    )


def _make_return_data(dates: pd.DatetimeIndex) -> ReturnData:
    """Build a minimal ReturnData for BacktestContext fixtures.

    Arguments:
        dates: DatetimeIndex to use for all series.

    Returns:
        ReturnData with flat returns.
    """
    rng = np.random.default_rng(42)
    simple = rng.normal(0.0003, 0.01, len(dates))
    returns = pd.DataFrame({"VTI": simple}, index=dates)
    log_returns = pd.DataFrame({"VTI": np.log1p(simple)}, index=dates)
    rfr = pd.Series(0.04, index=dates, name="risk_free_rate")
    return ReturnData(
        returns=returns,
        log_returns=log_returns,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr,
    )


def _make_config(use_leaps: bool = True) -> PortfolioConfig:
    """Build a minimal PortfolioConfig.

    Arguments:
        use_leaps: If True, includes VTI_LEAPS key in target_weights.

    Returns:
        PortfolioConfig appropriate for the test.
    """
    if use_leaps:
        return PortfolioConfig(
            target_weights={"VTI": 0.85, "VTI_LEAPS": 0.15},
            initial_nav=10_000.0,
            monthly_contribution=500.0,
            rebalance_rule=RebalanceRule.QUARTERLY,
            weight_strategy=WeightStrategy.USER_SPECIFIED,
            leaps_config=LeapsConfig(iv=0.20),
        )
    return PortfolioConfig(
        target_weights={"VTI": 1.0},
        initial_nav=10_000.0,
        monthly_contribution=500.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )


def _make_ctx(
    *,
    use_leaps: bool = True,
    gtt_active: bool = True,
    underlying_prices: pd.Series | None = None,
    iv: float = 0.20,
) -> BacktestContext:
    """Build a BacktestContext with controllable fields for _compute_leaps_mtm.

    Arguments:
        use_leaps: Whether LEAPS overlay is active.
        gtt_active: Whether GTT signal is active.
        underlying_prices: Spot price series; defaults to a constant 200.0 series.
        iv: IV floor.

    Returns:
        BacktestContext with sensible defaults for F-010 tests.
    """
    dates = pd.bdate_range("2023-01-03", periods=130)
    config = _make_config(use_leaps=use_leaps)
    return_data = _make_return_data(dates)
    if underlying_prices is None and use_leaps:
        underlying_prices = pd.Series(200.0, index=dates, name="VTI")
    return BacktestContext(
        base_assets=("VTI",),
        leaps_keys=("VTI_LEAPS",) if use_leaps else (),
        leaps_fraction=0.15 if use_leaps else 0.0,
        base_target_w=pd.Series({"VTI": 1.0}),
        governed_base=("VTI",) if gtt_active else (),
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0},
        use_leaps=use_leaps,
        iv=iv,
        leaps_monthly=500.0 * 0.15 if use_leaps else 0.0,
        base_contribution=500.0 * (0.85 if use_leaps else 1.0),
        config=config,
        return_data=return_data,
        underlying_prices=underlying_prices,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=None,
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end={},
        w=pd.Series({"VTI": 0.85, "VTI_LEAPS": 0.15}) if use_leaps else pd.Series({"VTI": 1.0}),
    )


def _make_state(
    *,
    leaps_ledger: LeapsLedger | None = None,
    leaps_scale: dict[LeapsContract, float] | None = None,
    prev_regime: int = 1,
    leaps_value: float = 0.0,
) -> PortfolioState:
    """Build a minimal PortfolioState for F-010 tests.

    Arguments:
        leaps_ledger: Active LEAPS ledger (or None).
        leaps_scale: Per-contract scale factors (defaults to empty dict).
        prev_regime: GTT regime from the previous day.
        leaps_value: Starting leaps_value.

    Returns:
        PortfolioState with sensible defaults.
    """
    return PortfolioState(
        holdings={"VTI": 50000.0},
        defensive_sleeve=0.0,
        leaps_pool=0.0,
        leaps_value=leaps_value,
        prev_total_nav=50000.0,
        prev_regime=prev_regime,
        prev_date_ts=None,
        leaps_ledger=leaps_ledger,
        leaps_scale=leaps_scale if leaps_scale is not None else {},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )


def _make_inputs(
    *,
    date_ts: pd.Timestamp = _DATE_MTM,
    regime_t: int = 1,
    spot: float | None = 200.0,
    mtm_iv_value: float | None = None,
    rfr: float = 0.04,
) -> DayInputs:
    """Build a minimal DayInputs for F-010 tests.

    Arguments:
        date_ts: Current trading day.
        regime_t: GTT regime today (0=Defensive, 1=Long).
        spot: Underlying spot price.
        mtm_iv_value: Smoothed MTM IV (or None/NaN).
        rfr: Risk-free rate.

    Returns:
        DayInputs with sensible defaults.
    """
    return DayInputs(
        date_ts=date_ts,
        day_ret=pd.Series({"VTI": 0.0}),
        regime_t=regime_t,
        def_gross_return=0.0,
        spot=spot,
        raw_vix_value=None,
        mtm_iv_value=mtm_iv_value,
        rfr=rfr,
        is_month_end=False,
        is_rebal_date=False,
    )


# ---------------------------------------------------------------------------
# Suppression: Bug 1 regression (re-entry day)
# ---------------------------------------------------------------------------


def test_bug1_reentry_suppression() -> None:
    """Bug 1 regression: MTM is suppressed exactly on re-entry days.

    When prev_regime=0 and regime_t=1 with gtt_active=True and use_leaps=True,
    the stale old-window ledger must NOT be marked to market. This test will fail
    if the suppression condition is removed.
    """
    contract = make_contract()
    ledger = make_ledger(contract)
    state = _make_state(leaps_ledger=ledger, prev_regime=0)
    inputs = _make_inputs(regime_t=1)
    ctx = _make_ctx(use_leaps=True, gtt_active=True)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == 0.0


def test_bug1_non_reentry_long_day_is_not_suppressed() -> None:
    """On a normal Long day (prev_regime=1, regime_t=1), MTM fires normally.

    This confirms the Bug 1 condition is scoped only to the transition day.
    """
    contract = make_contract()
    ledger = make_ledger(contract)
    state = _make_state(leaps_ledger=ledger, prev_regime=1)
    inputs = _make_inputs(regime_t=1)
    ctx = _make_ctx(use_leaps=True, gtt_active=True, iv=0.20)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    # Should have a positive leaps_value (not suppressed)
    assert new_state.leaps_value > 0.0


# ---------------------------------------------------------------------------
# Suppression: defensive day
# ---------------------------------------------------------------------------


def test_defensive_day_suppression() -> None:
    """MTM is suppressed on defensive days (regime_t=0) when gtt_active=True."""
    contract = make_contract()
    ledger = make_ledger(contract)
    state = _make_state(leaps_ledger=ledger, prev_regime=1)
    inputs = _make_inputs(regime_t=0)
    ctx = _make_ctx(use_leaps=True, gtt_active=True)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == 0.0


# ---------------------------------------------------------------------------
# Suppression: use_leaps=False
# ---------------------------------------------------------------------------


def test_no_op_when_use_leaps_false() -> None:
    """MTM returns leaps_value=0.0 when use_leaps is False."""
    contract = make_contract()
    ledger = make_ledger(contract)
    state = _make_state(leaps_ledger=ledger, prev_regime=1)
    inputs = _make_inputs(regime_t=1)
    ctx = _make_ctx(use_leaps=False, gtt_active=False)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == 0.0


# ---------------------------------------------------------------------------
# Suppression: leaps_ledger=None
# ---------------------------------------------------------------------------


def test_no_op_when_ledger_is_none() -> None:
    """MTM returns leaps_value=0.0 when leaps_ledger is None."""
    state = _make_state(leaps_ledger=None, prev_regime=1)
    inputs = _make_inputs(regime_t=1)
    ctx = _make_ctx(use_leaps=True, gtt_active=False)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == 0.0


# ---------------------------------------------------------------------------
# Suppression: underlying_prices=None
# ---------------------------------------------------------------------------


def test_no_op_when_underlying_prices_is_none() -> None:
    """MTM returns leaps_value=0.0 when ctx.underlying_prices is None."""
    contract = make_contract()
    ledger = make_ledger(contract)
    state = _make_state(leaps_ledger=ledger, prev_regime=1)
    inputs = _make_inputs(regime_t=1)
    ctx = _make_ctx(use_leaps=True, gtt_active=False, underlying_prices=pd.Series([], dtype=float))
    # Override underlying_prices to None via a new context
    import dataclasses
    ctx_none = dataclasses.replace(ctx, underlying_prices=None)

    new_state = _compute_leaps_mtm(state, inputs, ctx_none)

    assert new_state.leaps_value == 0.0


# ---------------------------------------------------------------------------
# Normal Long day MTM
# ---------------------------------------------------------------------------


def test_normal_long_day_mtm() -> None:
    """MTM is computed correctly on a normal Long day (prev_regime=1, regime_t=1).

    Expected value is computed directly via price_leaps_contract so the test
    stays independent of the BS implementation detail.
    """
    contract = make_contract()
    ledger = make_ledger(contract)
    state = _make_state(leaps_ledger=ledger, prev_regime=1)

    spot = 205.0
    rfr = 0.04
    iv = 0.20
    mtm_date = _DATE_MTM

    inputs = _make_inputs(date_ts=mtm_date, regime_t=1, spot=spot, rfr=rfr)
    ctx = _make_ctx(use_leaps=True, gtt_active=False, iv=iv)

    expected = price_leaps_contract(contract, spot, mtm_date, iv, rfr)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# MTM IV: NaN fallback to ctx.iv
# ---------------------------------------------------------------------------


def test_nan_mtm_iv_falls_back_to_ctx_iv() -> None:
    """When inputs.mtm_iv_value is NaN, day_iv should equal ctx.iv.

    The resulting leaps_value must match a manual computation using ctx.iv.
    """
    contract = make_contract()
    ledger = make_ledger(contract)
    state = _make_state(leaps_ledger=ledger, prev_regime=1)

    spot = 200.0
    rfr = 0.04
    iv = 0.20
    mtm_date = _DATE_MTM

    inputs = _make_inputs(
        date_ts=mtm_date, regime_t=1, spot=spot, rfr=rfr, mtm_iv_value=float("nan")
    )
    ctx = _make_ctx(use_leaps=True, gtt_active=False, iv=iv)

    expected = price_leaps_contract(contract, spot, mtm_date, iv, rfr)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# MTM IV: None fallback to ctx.iv
# ---------------------------------------------------------------------------


def test_none_mtm_iv_falls_back_to_ctx_iv() -> None:
    """When inputs.mtm_iv_value is None, day_iv should equal ctx.iv.

    The resulting leaps_value must match a manual computation using ctx.iv.
    """
    contract = make_contract()
    ledger = make_ledger(contract)
    state = _make_state(leaps_ledger=ledger, prev_regime=1)

    spot = 200.0
    rfr = 0.04
    iv = 0.20
    mtm_date = _DATE_MTM

    inputs = _make_inputs(date_ts=mtm_date, regime_t=1, spot=spot, rfr=rfr, mtm_iv_value=None)
    ctx = _make_ctx(use_leaps=True, gtt_active=False, iv=iv)

    expected = price_leaps_contract(contract, spot, mtm_date, iv, rfr)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# MTM IV: smoothed IV when larger than ctx.iv
# ---------------------------------------------------------------------------


def test_smoothed_iv_used_when_larger_than_floor() -> None:
    """When mtm_iv_value > ctx.iv, day_iv = mtm_iv_value (the larger value)."""
    contract = make_contract()
    ledger = make_ledger(contract)
    state = _make_state(leaps_ledger=ledger, prev_regime=1)

    spot = 200.0
    rfr = 0.04
    iv_floor = 0.20
    mtm_iv = 0.35  # higher than floor
    mtm_date = _DATE_MTM

    inputs = _make_inputs(
        date_ts=mtm_date, regime_t=1, spot=spot, rfr=rfr, mtm_iv_value=mtm_iv
    )
    ctx = _make_ctx(use_leaps=True, gtt_active=False, iv=iv_floor)

    expected = price_leaps_contract(contract, spot, mtm_date, mtm_iv, rfr)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# leaps_scale applied
# ---------------------------------------------------------------------------


def test_leaps_scale_applied() -> None:
    """leaps_scale={contract: 0.5} halves the contract's MTM contribution."""
    contract = make_contract()
    ledger = make_ledger(contract)
    scale = {contract: 0.5}
    state = _make_state(leaps_ledger=ledger, leaps_scale=scale, prev_regime=1)

    spot = 200.0
    rfr = 0.04
    iv = 0.20
    mtm_date = _DATE_MTM

    inputs = _make_inputs(date_ts=mtm_date, regime_t=1, spot=spot, rfr=rfr)
    ctx = _make_ctx(use_leaps=True, gtt_active=False, iv=iv)

    full_price = price_leaps_contract(contract, spot, mtm_date, iv, rfr)
    expected = full_price * 0.5

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Hypothesis: leaps_value >= 0.0 for any valid LEAPS contract
# ---------------------------------------------------------------------------


@given(
    spot=st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    iv=st.floats(min_value=0.10, max_value=0.80, allow_nan=False, allow_infinity=False),
    strike_ratio=st.floats(min_value=0.50, max_value=0.90, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_hypothesis_leaps_value_nonneg(
    spot: float, iv: float, strike_ratio: float
) -> None:
    """For any valid spot/iv/strike combination, leaps_value is always >= 0.0.

    Arguments:
        spot: Underlying spot price.
        iv: Implied volatility.
        strike_ratio: Strike as fraction of spot (generates DITM contracts).
    """
    strike = spot * strike_ratio
    purchase_date = pd.Timestamp("2022-01-03")
    expiry = pd.Timestamp("2024-01-17")
    mtm_date = pd.Timestamp("2023-06-01")

    contract = LeapsContract(
        purchase_date=purchase_date,
        expiry_date=expiry,
        strike=strike,
        spot_at_purchase=spot,
        premium_paid=max(spot * 0.20, 1.0),
        notional=spot * 100.0,
        n_contracts=1.0,
        account_type=AccountType.TAXABLE,
    )
    ledger = LeapsLedger(
        contracts=(contract,),
        roll_events=(),
        account_type=AccountType.TAXABLE,
    )
    state = _make_state(leaps_ledger=ledger, prev_regime=1)
    inputs = _make_inputs(date_ts=mtm_date, regime_t=1, spot=spot, rfr=0.04)
    ctx = _make_ctx(use_leaps=True, gtt_active=False, iv=iv)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value >= 0.0
