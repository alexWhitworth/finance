"""Tests for F-014: _apply_gtt_reentry (Bug 2 fix).

Verifies the Defensive->Long re-entry step function:
- No-op conditions (tests 1-3)
- NAV-neutral invariant A2 (test 4)
- Creation IV invariant A4 (test 5) — primary Bug 2 regression test
- Elevated VIX at re-entry priced correctly (test 6)
- leaps_scale reset on re-entry (test 7)
- Hypothesis: A2 holds across leaps_fraction and total NAV ranges (test 8)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from finance.consts import DEFAULT_IV
from finance.leverage import (
    AccountType,
    LeapsConfig,
    LeapsContract,
    _live_contracts,
    price_leaps_contract,
    run_leaps_simulation,
)
from finance.portfolio import PortfolioConfig, RebalanceRule, WeightStrategy
from finance.returns import ReturnData

from finance._step_f014 import _apply_gtt_reentry
from finance.portfolio import BacktestContext, DayInputs, PortfolioState

# ---------------------------------------------------------------------------
# Shared test corpus
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_DATES = pd.bdate_range("2020-01-02", periods=252)
_PRICES = pd.Series(
    200.0 * np.cumprod(1 + _RNG.normal(0, 0.01, len(_DATES))),
    index=_DATES,
)

# A small returns DataFrame for ReturnData (only VTI needed for LEAPS)
_RETURNS_DF = pd.DataFrame(
    {"VTI": _PRICES.pct_change().dropna()},
)
_RETURN_DATA = ReturnData(
    returns=_RETURNS_DF,
    log_returns=np.log(1 + _RETURNS_DF),
    tey_adjusted=False,
    marginal_rate=0.0,
    risk_free_rate=pd.Series(0.04, index=_DATES),
)

_LEAPS_CONFIG = LeapsConfig(iv=DEFAULT_IV, ltcg_rate=0.238, account_type=AccountType.TAXABLE)

_PORTFOLIO_CONFIG = PortfolioConfig(
    target_weights={"VTI": 0.85, "VTI_LEAPS": 0.15},
    initial_nav=100_000.0,
    monthly_contribution=500.0,
    rebalance_rule=RebalanceRule.QUARTERLY,
    weight_strategy=WeightStrategy.USER_SPECIFIED,
    leaps_config=_LEAPS_CONFIG,
)

# base_target_w: only VTI in base_assets, so weight is 1.0
_BASE_TARGET_W = pd.Series({"VTI": 1.0})

_RE_ENTRY_DATE = _DATES[10]  # a date well within the window
_SPOT = float(_PRICES.loc[_RE_ENTRY_DATE])
_RFR = 0.04

# long_window_end: map re-entry date -> last date in corpus
_LONG_WINDOW_END: dict[pd.Timestamp, pd.Timestamp] = {
    _RE_ENTRY_DATE: _DATES[-1],
}


def _make_ctx(
    leaps_fraction: float = 0.15,
    use_leaps: bool = True,
    gtt_active: bool = True,
    iv: float = DEFAULT_IV,
    raw_vix: pd.Series | None = None,
) -> BacktestContext:
    """Build a minimal BacktestContext for re-entry tests."""
    base_target_w = _BASE_TARGET_W.copy()
    return BacktestContext(
        base_assets=("VTI",),
        leaps_keys=("VTI_LEAPS",),
        leaps_fraction=leaps_fraction,
        base_target_w=base_target_w,
        governed_base=("VTI",),
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0},
        use_leaps=use_leaps,
        iv=iv,
        leaps_monthly=500.0 * leaps_fraction,
        base_contribution=500.0 * (1.0 - leaps_fraction),
        config=_PORTFOLIO_CONFIG,
        return_data=_RETURN_DATA,
        underlying_prices=_PRICES,
        raw_vix=raw_vix,
        mtm_iv_series=None,
        rfr_series=pd.Series(0.04, index=_DATES),
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end=_LONG_WINDOW_END,
        w=pd.Series({"VTI": 0.85, "VTI_LEAPS": 0.15}),
    )


def _make_inputs(
    regime_t: int = 1,
    raw_vix_value: float | None = None,
    spot: float | None = None,
) -> DayInputs:
    """Build minimal DayInputs."""
    return DayInputs(
        date_ts=_RE_ENTRY_DATE,
        day_ret=pd.Series({"VTI": 0.005}),
        regime_t=regime_t,
        def_gross_return=0.0,
        spot=spot if spot is not None else _SPOT,
        raw_vix_value=raw_vix_value,
        mtm_iv_value=None,
        rfr=_RFR,
        is_month_end=False,
        is_rebal_date=False,
    )


def _make_state(
    holdings: dict[str, float] | None = None,
    defensive_sleeve: float = 0.0,
    leaps_pool: float = 0.0,
    leaps_value: float = 0.0,
    prev_regime: int = 0,
    leaps_scale: dict[LeapsContract, float] | None = None,
) -> PortfolioState:
    """Build minimal PortfolioState."""
    if holdings is None:
        holdings = {"VTI": 75_000.0}
    return PortfolioState(
        holdings=holdings,
        defensive_sleeve=defensive_sleeve,
        leaps_pool=leaps_pool,
        leaps_value=leaps_value,
        prev_total_nav=100_000.0,
        prev_regime=prev_regime,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale=leaps_scale if leaps_scale is not None else {},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )


# ---------------------------------------------------------------------------
# Test 1: No-op — Long day (prev_regime=1, regime_t=1)
# ---------------------------------------------------------------------------


def test_noop_long_day_unchanged() -> None:
    """State is returned unchanged on a normal Long day (prev_regime=1, regime_t=1)."""
    ctx = _make_ctx()
    state = _make_state(prev_regime=1)
    inputs = _make_inputs(regime_t=1)

    result = _apply_gtt_reentry(state, inputs, ctx)

    assert result is state


# ---------------------------------------------------------------------------
# Test 2: No-op — prev_regime=1, regime_t=0 (defensive day)
# ---------------------------------------------------------------------------


def test_noop_defensive_day_unchanged() -> None:
    """State is returned unchanged on a defensive day (prev_regime=1, regime_t=0)."""
    ctx = _make_ctx()
    state = _make_state(prev_regime=1)
    inputs = _make_inputs(regime_t=0)

    result = _apply_gtt_reentry(state, inputs, ctx)

    assert result is state


# ---------------------------------------------------------------------------
# Test 3: No-op — GTT inactive
# ---------------------------------------------------------------------------


def test_noop_gtt_inactive() -> None:
    """State is returned unchanged when gtt_active=False."""
    ctx = _make_ctx(gtt_active=False)
    state = _make_state(prev_regime=0)
    inputs = _make_inputs(regime_t=1)

    result = _apply_gtt_reentry(state, inputs, ctx)

    assert result is state


# ---------------------------------------------------------------------------
# Test 4: Re-entry — A2 invariant (NAV-neutral)
# ---------------------------------------------------------------------------


def test_reentry_a2_nav_neutral() -> None:
    """After re-entry, sum(holdings) + leaps_value == total within 1e-9 (A2)."""
    ctx = _make_ctx(leaps_fraction=0.15)
    state = _make_state(
        holdings={"VTI": 75_000.0},
        defensive_sleeve=20_000.0,
        leaps_pool=5_000.0,
        prev_regime=0,
    )
    inputs = _make_inputs(regime_t=1)

    total_before = (
        sum(state.holdings.values()) + state.defensive_sleeve + state.leaps_pool
    )  # 100_000.0
    result = _apply_gtt_reentry(state, inputs, ctx)

    reconstructed_nav = sum(result.holdings.values()) + result.leaps_value
    assert reconstructed_nav == pytest.approx(total_before, abs=1e-9), (
        f"A2 violated: sum(holdings)+leaps_value={reconstructed_nav} "
        f"!= total={total_before}"
    )
    assert result.defensive_sleeve == pytest.approx(0.0)
    assert result.leaps_pool == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 5: Re-entry — A4 invariant (creation IV used, Bug 2 regression)
# ---------------------------------------------------------------------------


def test_reentry_a4_leaps_value_equals_capital_deployed() -> None:
    """leaps_value == total * leaps_fraction within 1e-6 (A4, Bug 2 regression test)."""
    leaps_fraction = 0.15
    ctx = _make_ctx(leaps_fraction=leaps_fraction)
    state = _make_state(
        holdings={"VTI": 75_000.0},
        defensive_sleeve=20_000.0,
        leaps_pool=5_000.0,
        prev_regime=0,
    )
    inputs = _make_inputs(regime_t=1, raw_vix_value=None)

    total_before = (
        sum(state.holdings.values()) + state.defensive_sleeve + state.leaps_pool
    )
    result = _apply_gtt_reentry(state, inputs, ctx)

    expected_leaps_capital = total_before * leaps_fraction
    assert result.leaps_value == pytest.approx(expected_leaps_capital, rel=1e-6), (
        f"A4 violated: leaps_value={result.leaps_value} "
        f"!= expected={expected_leaps_capital}"
    )


# ---------------------------------------------------------------------------
# Test 6: Bug 2 regression — elevated-then-dropped VIX
# ---------------------------------------------------------------------------


def test_reentry_bug2_elevated_vix_priced_at_creation_iv() -> None:
    """Elevated raw_vix_value at re-entry: creation_iv=max(raw_vix_value, ctx.iv).

    Verifies that leaps_value matches manual re-computation using the elevated
    raw VIX as creation_iv, NOT the smoothed MTM IV.
    """
    raw_vix_value = 0.50  # elevated VIX at re-entry
    ctx = _make_ctx(leaps_fraction=0.15, iv=DEFAULT_IV)
    state = _make_state(
        holdings={"VTI": 75_000.0},
        defensive_sleeve=20_000.0,
        leaps_pool=5_000.0,
        prev_regime=0,
    )
    inputs = _make_inputs(regime_t=1, raw_vix_value=raw_vix_value)

    total_before = (
        sum(state.holdings.values()) + state.defensive_sleeve + state.leaps_pool
    )
    result = _apply_gtt_reentry(state, inputs, ctx)

    # Recompute what creation_iv should be
    creation_iv = max(raw_vix_value, DEFAULT_IV)

    # Replay the ledger manually to get the contracts
    assert result.leaps_ledger is not None
    win_prices = _PRICES.loc[_RE_ENTRY_DATE:_DATES[-1]]
    manual_ledger = run_leaps_simulation(
        win_prices,
        ctx.leaps_monthly,
        ctx.config.leaps_config,
        risk_free_series=_RETURN_DATA.risk_free_rate,
        iv_series=None,  # no raw_vix in ctx for this sub-test
        initial_capital=total_before * 0.15,
    )
    # The step function's ledger uses ctx.raw_vix (None here), so both ledgers
    # should have identical contracts; compute expected value using creation_iv
    expected_leaps_value = sum(
        price_leaps_contract(c, _SPOT, _RE_ENTRY_DATE, creation_iv, _RFR)
        for c in _live_contracts(manual_ledger, _RE_ENTRY_DATE)
    )

    assert result.leaps_value == pytest.approx(expected_leaps_value, rel=1e-6), (
        f"Bug 2: leaps_value={result.leaps_value} != expected={expected_leaps_value} "
        f"(creation_iv={creation_iv}, raw_vix_value={raw_vix_value})"
    )


# ---------------------------------------------------------------------------
# Test 7: leaps_scale reset on re-entry
# ---------------------------------------------------------------------------


def test_reentry_leaps_scale_reset() -> None:
    """leaps_scale is reset to {} on re-entry (clears orphaned keys from old window)."""
    # Create a fake old contract to populate leaps_scale
    old_contract = LeapsContract(
        purchase_date=pd.Timestamp("2019-01-02"),
        expiry_date=pd.Timestamp("2021-01-15"),
        strike=180.0,
        spot_at_purchase=200.0,
        premium_paid=25.0,
        notional=20_000.0,
        n_contracts=1.0,
        account_type=AccountType.TAXABLE,
    )
    ctx = _make_ctx(leaps_fraction=0.15)
    state = _make_state(
        holdings={"VTI": 75_000.0},
        defensive_sleeve=20_000.0,
        leaps_pool=5_000.0,
        prev_regime=0,
        leaps_scale={old_contract: 0.5},
    )
    inputs = _make_inputs(regime_t=1)

    result = _apply_gtt_reentry(state, inputs, ctx)

    assert result.leaps_scale == {}, (
        f"Expected empty leaps_scale after re-entry, got: {result.leaps_scale}"
    )


# ---------------------------------------------------------------------------
# Test 8: Hypothesis — A2 holds across leaps_fraction and total NAV ranges
# ---------------------------------------------------------------------------


@settings(max_examples=30, deadline=10_000)
@given(
    leaps_fraction=st.floats(min_value=0.0, max_value=0.30, allow_nan=False),
    total_nav=st.floats(min_value=10_000.0, max_value=100_000.0, allow_nan=False),
)
def test_hypothesis_a2_nav_conservation(leaps_fraction: float, total_nav: float) -> None:
    """A2: sum(holdings) + leaps_value == total within 1e-9 for any valid inputs."""
    # Distribute total across holdings + sleeve + pool
    holdings_val = total_nav * 0.75
    sleeve_val = total_nav * 0.15
    pool_val = total_nav * 0.10

    ctx = _make_ctx(leaps_fraction=leaps_fraction)
    state = _make_state(
        holdings={"VTI": holdings_val},
        defensive_sleeve=sleeve_val,
        leaps_pool=pool_val,
        prev_regime=0,
    )
    inputs = _make_inputs(regime_t=1, raw_vix_value=None)

    result = _apply_gtt_reentry(state, inputs, ctx)

    reconstructed_nav = sum(result.holdings.values()) + result.leaps_value
    expected_total = holdings_val + sleeve_val + pool_val

    assert abs(reconstructed_nav - expected_total) < 1e-9, (
        f"A2 violated: reconstructed={reconstructed_nav}, expected={expected_total}, "
        f"leaps_fraction={leaps_fraction}, total_nav={total_nav}"
    )
