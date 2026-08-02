"""Tests for F-007 — _apply_gtt_open.

Covers all six required test cases:
  1. No-op on Long day (regime_t=1)
  2. Sweep on defensive day (regime_t=0)
  3. No-op when GTT inactive (gtt_active=False)
  4. Double-sweep safety (holdings already 0.0)
  5. Accounting invariant (total conserved)
  6. Hypothesis property test (conservation for arbitrary positive floats)
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from finance._step_f007 import _apply_gtt_open
from finance.leverage import RebalanceRule, WeightStrategy
from finance.portfolio import BacktestContext, DayInputs, PortfolioConfig, PortfolioState
from finance.returns import ReturnData

# ---------------------------------------------------------------------------
# Minimal fixture helpers
# ---------------------------------------------------------------------------

_DATES = pd.bdate_range("2020-01-02", periods=10)
_RETURNS = pd.DataFrame({"VTI": [0.001] * 10, "VXUS": [0.001] * 10}, index=_DATES)
_RFR = pd.Series(0.04, index=_DATES)


def _make_config() -> PortfolioConfig:
    """Return a minimal PortfolioConfig for VTI/VXUS."""
    return PortfolioConfig(
        target_weights={"VTI": 0.7, "VXUS": 0.3},
        initial_nav=10_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )


def _make_return_data() -> ReturnData:
    """Return a minimal ReturnData aligned to _DATES."""
    return ReturnData(
        returns=_RETURNS,
        log_returns=pd.DataFrame(
            {"VTI": np.log1p(0.001), "VXUS": np.log1p(0.001)},
            index=_DATES,
        ),
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=_RFR,
    )


def minimal_ctx(
    gtt_active: bool = True,
    governed_base: tuple[str, ...] = ("VTI",),
) -> BacktestContext:
    """Construct a BacktestContext with the minimal fields required by _apply_gtt_open."""
    config = _make_config()
    rd = _make_return_data()
    return BacktestContext(
        base_assets=("VTI", "VXUS"),
        leaps_keys=(),
        leaps_fraction=0.0,
        base_target_w=pd.Series({"VTI": 0.7, "VXUS": 0.3}),
        governed_base=governed_base,
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0},
        use_leaps=False,
        iv=0.20,
        leaps_monthly=0.0,
        base_contribution=0.0,
        config=config,
        return_data=rd,
        underlying_prices=None,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=_RFR,
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end={},
        w=pd.Series({"VTI": 0.7, "VXUS": 0.3}),
    )


def minimal_state(
    holdings: dict[str, float],
    sleeve: float = 0.0,
) -> PortfolioState:
    """Construct a PortfolioState with the given holdings and defensive sleeve."""
    return PortfolioState(
        holdings=holdings,
        defensive_sleeve=sleeve,
        leaps_pool=0.0,
        leaps_value=0.0,
        prev_total_nav=sum(holdings.values()),
        prev_regime=1,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )


def minimal_inputs(
    regime_t: int = 0,
    date_ts: pd.Timestamp = pd.Timestamp("2020-01-02"),
) -> DayInputs:
    """Construct a DayInputs with the given regime."""
    return DayInputs(
        date_ts=date_ts,
        day_ret=pd.Series({"VTI": 0.001, "VXUS": 0.001}),
        regime_t=regime_t,
        def_gross_return=0.0,
        spot=None,
        raw_vix_value=None,
        mtm_iv_value=None,
        rfr=0.04,
        is_month_end=False,
        is_rebal_date=False,
    )


# ---------------------------------------------------------------------------
# Test 1 — No-op on Long day (regime_t=1)
# ---------------------------------------------------------------------------


def test_noop_on_long_day() -> None:
    """State is returned unchanged (same object) when regime_t == 1."""
    state = minimal_state({"VTI": 50_000.0, "VXUS": 20_000.0})
    ctx = minimal_ctx(gtt_active=True, governed_base=("VTI",))
    inputs = minimal_inputs(regime_t=1)

    result = _apply_gtt_open(state, inputs, ctx)

    assert result is state, "No-op path must return the same PortfolioState object"


# ---------------------------------------------------------------------------
# Test 2 — Sweep on defensive day (regime_t=0)
# ---------------------------------------------------------------------------


def test_sweep_on_defensive_day() -> None:
    """Governed VTI is zeroed and moved into defensive_sleeve; VXUS untouched."""
    initial_sleeve = 5_000.0
    state = minimal_state({"VTI": 50_000.0, "VXUS": 20_000.0}, sleeve=initial_sleeve)
    ctx = minimal_ctx(gtt_active=True, governed_base=("VTI",))
    inputs = minimal_inputs(regime_t=0)

    new_state = _apply_gtt_open(state, inputs, ctx)

    assert new_state.holdings["VTI"] == 0.0
    assert new_state.holdings["VXUS"] == 20_000.0
    assert new_state.defensive_sleeve == pytest.approx(initial_sleeve + 50_000.0)


# ---------------------------------------------------------------------------
# Test 3 — No-op when GTT inactive
# ---------------------------------------------------------------------------


def test_noop_when_gtt_inactive() -> None:
    """State returned unchanged when gtt_active=False, even on a defensive day."""
    state = minimal_state({"VTI": 50_000.0, "VXUS": 20_000.0})
    ctx = minimal_ctx(gtt_active=False)
    inputs = minimal_inputs(regime_t=0)

    result = _apply_gtt_open(state, inputs, ctx)

    assert result is state


# ---------------------------------------------------------------------------
# Test 4 — Double-sweep safety (holdings already 0.0)
# ---------------------------------------------------------------------------


def test_double_sweep_safety() -> None:
    """Sleeve never goes negative when governed holding is already 0.0."""
    initial_sleeve = 50_000.0
    state = minimal_state({"VTI": 0.0, "VXUS": 20_000.0}, sleeve=initial_sleeve)
    ctx = minimal_ctx(gtt_active=True, governed_base=("VTI",))
    inputs = minimal_inputs(regime_t=0)

    new_state = _apply_gtt_open(state, inputs, ctx)

    assert new_state.holdings["VTI"] == 0.0
    assert new_state.defensive_sleeve == pytest.approx(initial_sleeve)


# ---------------------------------------------------------------------------
# Test 5 — Accounting invariant (explicit)
# ---------------------------------------------------------------------------


def test_accounting_invariant_explicit() -> None:
    """Total (holdings_sum + sleeve) is conserved within 1e-12."""
    initial_sleeve = 3_000.0
    state = minimal_state({"VTI": 50_000.0, "VXUS": 20_000.0}, sleeve=initial_sleeve)
    ctx = minimal_ctx(gtt_active=True, governed_base=("VTI",))
    inputs = minimal_inputs(regime_t=0)

    before = sum(state.holdings.values()) + state.defensive_sleeve
    new_state = _apply_gtt_open(state, inputs, ctx)
    after = sum(new_state.holdings.values()) + new_state.defensive_sleeve

    assert abs(after - before) < 1e-12, (
        f"Accounting invariant violated: before={before}, after={after}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Hypothesis property test (conservation for arbitrary floats)
# ---------------------------------------------------------------------------


@given(
    vti_holding=st.floats(0, 1e6),
    vxus_holding=st.floats(0, 1e6),
    initial_sleeve=st.floats(0, 1e4),
)
@settings(max_examples=500)
def test_accounting_invariant_hypothesis(
    vti_holding: float,
    vxus_holding: float,
    initial_sleeve: float,
) -> None:
    """Total capital is conserved for all non-negative holding/sleeve combinations."""
    state = minimal_state(
        {"VTI": vti_holding, "VXUS": vxus_holding},
        sleeve=initial_sleeve,
    )
    ctx = minimal_ctx(gtt_active=True, governed_base=("VTI",))
    inputs = minimal_inputs(regime_t=0)

    before = sum(state.holdings.values()) + state.defensive_sleeve
    new_state = _apply_gtt_open(state, inputs, ctx)
    after = sum(new_state.holdings.values()) + new_state.defensive_sleeve

    assert abs(after - before) < 1e-9, (
        f"Invariant violated: before={before:.6f}, after={after:.6f}"
    )
