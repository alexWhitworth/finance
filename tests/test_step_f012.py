"""Tests for F-012: _apply_contribution step function.

Covers all seven required test cases:
1. No-op on non-month-end day
2. No-op when base_assets is empty
3. Month-end Long day — full contribution flows to holdings
4. Month-end Defensive day — governed portion to sleeve
5. Month-end Defensive day — LEAPS pool receives leaps_monthly
6. Month-end re-entry day (regime_t=1) — governed to holdings, not sleeve
7. Accounting invariant on Long month-end
"""

import pandas as pd
import pytest

from finance._step_f012 import _apply_contribution
from finance.leverage import RebalanceRule, WeightStrategy
from finance.portfolio import BacktestContext, DayInputs, PortfolioConfig, PortfolioState
from finance.returns import ReturnData

# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------

_DATE = pd.Timestamp("2024-01-31")
_DAY_RET = pd.Series({"VTI": 0.0, "VXUS": 0.0})


def _make_state(
    holdings: dict[str, float] | None = None,
    defensive_sleeve: float = 0.0,
    leaps_pool: float = 0.0,
) -> PortfolioState:
    """Build a minimal PortfolioState for contribution tests.

    Arguments:
        holdings: Dollar-value holdings dict. Defaults to {"VTI": 1000.0}.
        defensive_sleeve: Starting defensive sleeve value.
        leaps_pool: Starting LEAPS pool value.

    Returns:
        PortfolioState with all accumulator/scalar fields set to sensible
        defaults.
    """
    return PortfolioState(
        holdings=holdings if holdings is not None else {"VTI": 1000.0, "VXUS": 500.0},
        defensive_sleeve=defensive_sleeve,
        leaps_pool=leaps_pool,
        leaps_value=0.0,
        prev_total_nav=1500.0,
        prev_regime=1,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )


def _make_inputs(
    is_month_end: bool = True,
    regime_t: int = 1,
    date_ts: pd.Timestamp = _DATE,
) -> DayInputs:
    """Build a minimal DayInputs for contribution tests.

    Arguments:
        is_month_end: Whether this is a month-end day.
        regime_t: GTT regime (0=Defensive, 1=Long).
        date_ts: Trading date.

    Returns:
        DayInputs with all fields populated.
    """
    return DayInputs(
        date_ts=date_ts,
        day_ret=_DAY_RET,
        regime_t=regime_t,
        def_gross_return=0.0,
        spot=None,
        raw_vix_value=None,
        mtm_iv_value=None,
        rfr=0.0,
        is_month_end=is_month_end,
        is_rebal_date=False,
    )


def _make_return_data() -> ReturnData:
    """Build a minimal ReturnData stub for BacktestContext.

    Returns:
        ReturnData with a single-row returns DataFrame and zero risk-free rate.
    """
    idx = pd.DatetimeIndex([_DATE])
    returns = pd.DataFrame({"VTI": [0.0], "VXUS": [0.0]}, index=idx)
    log_returns = pd.DataFrame({"VTI": [0.0], "VXUS": [0.0]}, index=idx)
    risk_free_rate = pd.Series([0.0], index=idx)
    return ReturnData(
        returns=returns,
        log_returns=log_returns,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=risk_free_rate,
    )


def _make_ctx(
    base_assets: tuple[str, ...] = ("VTI", "VXUS"),
    base_target_w: dict[str, float] | None = None,
    governed_base: tuple[str, ...] = (),
    gtt_active: bool = False,
    base_contribution: float = 500.0,
    leaps_monthly: float = 0.0,
) -> BacktestContext:
    """Build a minimal BacktestContext for contribution tests.

    Arguments:
        base_assets: Tuple of base asset tickers.
        base_target_w: Weight dict over base_assets (sums to 1.0). Defaults
            to equal-weight over base_assets.
        governed_base: GTT-governed subset of base_assets.
        gtt_active: Whether the GTT overlay is active.
        base_contribution: Monthly dollar contribution for base holdings.
        leaps_monthly: Monthly dollar contribution for LEAPS pool.

    Returns:
        BacktestContext with minimal fields needed for _apply_contribution.
    """
    if base_target_w is None:
        n = len(base_assets)
        base_target_w = {a: 1.0 / n for a in base_assets} if n > 0 else {}

    w_series = pd.Series(base_target_w)
    # PortfolioConfig requires target_weights to sum to 1.0; use a fallback
    # when base_assets is empty (the test exercises ctx.base_assets, not config).
    config_weights = (
        {a: 1.0 / len(base_assets) for a in base_assets}
        if base_assets
        else {"VTI": 1.0}
    )
    config = PortfolioConfig(
        target_weights=config_weights,
        initial_nav=10_000.0,
        monthly_contribution=base_contribution,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )

    return BacktestContext(
        base_assets=base_assets,
        leaps_keys=(),
        leaps_fraction=0.0,
        base_target_w=w_series,
        governed_base=governed_base,
        gtt_active=gtt_active,
        defensive_weights={},
        use_leaps=False,
        iv=0.25,
        leaps_monthly=leaps_monthly,
        base_contribution=base_contribution,
        config=config,
        return_data=_make_return_data(),
        underlying_prices=None,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=None,
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end={},
        w=w_series,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_op_non_month_end() -> None:
    """Non-month-end day returns the original state unchanged."""
    state = _make_state()
    inputs = _make_inputs(is_month_end=False)
    ctx = _make_ctx()
    result = _apply_contribution(state, inputs, ctx, nav_before=1500.0)
    assert result is state


def test_no_op_empty_base_assets() -> None:
    """Empty base_assets tuple returns the original state unchanged."""
    state = _make_state()
    inputs = _make_inputs(is_month_end=True)
    ctx = _make_ctx(base_assets=())
    result = _apply_contribution(state, inputs, ctx, nav_before=1500.0)
    assert result is state


def test_month_end_long_day_holdings_increase() -> None:
    """Long month-end: full contribution flows to holdings proportional to weights."""
    state = _make_state(holdings={"VTI": 1000.0, "VXUS": 0.0})
    inputs = _make_inputs(is_month_end=True, regime_t=1)
    ctx = _make_ctx(
        base_assets=("VTI", "VXUS"),
        base_target_w={"VTI": 1.0, "VXUS": 0.0},
        base_contribution=500.0,
    )
    result = _apply_contribution(state, inputs, ctx, nav_before=1000.0)

    assert abs(result.holdings["VTI"] - (1000.0 + 500.0)) < 1e-9
    assert result.defensive_sleeve == state.defensive_sleeve
    assert result.leaps_pool == state.leaps_pool


def test_month_end_defensive_governed_to_sleeve() -> None:
    """Defensive month-end: governed asset allocation goes to sleeve, not holdings."""
    state = _make_state(
        holdings={"VTI": 1000.0, "VXUS": 500.0},
        defensive_sleeve=200.0,
    )
    inputs = _make_inputs(is_month_end=True, regime_t=0)
    ctx = _make_ctx(
        base_assets=("VTI", "VXUS"),
        base_target_w={"VTI": 0.7, "VXUS": 0.3},
        governed_base=("VTI",),
        gtt_active=True,
        base_contribution=500.0,
        leaps_monthly=0.0,
    )
    result = _apply_contribution(state, inputs, ctx, nav_before=1500.0)

    # VXUS (non-governed) gets its share: 0.3 * 500 = 150
    assert abs(result.holdings["VXUS"] - (500.0 + 150.0)) < 1e-9
    # VTI (governed) diverted to sleeve: 0.7 * 500 = 350
    assert abs(result.defensive_sleeve - (200.0 + 350.0)) < 1e-9
    # VTI holdings unchanged (governed, zero during defensive window)
    assert result.holdings.get("VTI", 0.0) == pytest.approx(1000.0)


def test_month_end_defensive_leaps_pool_receives_monthly() -> None:
    """Defensive month-end with gtt_active: leaps_pool receives leaps_monthly."""
    state = _make_state(leaps_pool=50.0)
    inputs = _make_inputs(is_month_end=True, regime_t=0)
    ctx = _make_ctx(
        base_assets=("VTI", "VXUS"),
        base_target_w={"VTI": 1.0, "VXUS": 0.0},
        governed_base=("VTI",),
        gtt_active=True,
        base_contribution=500.0,
        leaps_monthly=100.0,
    )
    result = _apply_contribution(state, inputs, ctx, nav_before=1500.0)

    assert abs(result.leaps_pool - (50.0 + 100.0)) < 1e-9


def test_month_end_reentry_governed_to_holdings() -> None:
    """Re-entry day (regime_t=1): governed allocation goes to holdings, not sleeve."""
    state = _make_state(
        holdings={"VTI": 1000.0, "VXUS": 500.0},
        defensive_sleeve=300.0,
    )
    inputs = _make_inputs(is_month_end=True, regime_t=1)
    ctx = _make_ctx(
        base_assets=("VTI", "VXUS"),
        base_target_w={"VTI": 0.7, "VXUS": 0.3},
        governed_base=("VTI",),
        gtt_active=True,
        base_contribution=500.0,
        leaps_monthly=100.0,
    )
    result = _apply_contribution(state, inputs, ctx, nav_before=1500.0)

    # regime_t=1, so governed goes to holdings (not sleeve)
    assert abs(result.holdings["VTI"] - (1000.0 + 0.7 * 500.0)) < 1e-9
    assert abs(result.holdings["VXUS"] - (500.0 + 0.3 * 500.0)) < 1e-9
    # Sleeve unchanged on Long day
    assert result.defensive_sleeve == state.defensive_sleeve
    # LEAPS pool unchanged (regime_t=1)
    assert result.leaps_pool == state.leaps_pool


def test_accounting_invariant_long_month_end() -> None:
    """Long month-end: sum(holdings) increases by exactly base_contribution."""
    holdings_start = {"VTI": 600.0, "VXUS": 400.0}
    state = _make_state(holdings=dict(holdings_start))
    inputs = _make_inputs(is_month_end=True, regime_t=1)
    contribution = 500.0
    ctx = _make_ctx(
        base_assets=("VTI", "VXUS"),
        base_target_w={"VTI": 0.6, "VXUS": 0.4},
        base_contribution=contribution,
    )
    result = _apply_contribution(state, inputs, ctx, nav_before=1000.0)

    old_total = sum(holdings_start.values())
    new_total = sum(result.holdings.values())
    assert abs(new_total - (old_total + contribution)) < 1e-9
