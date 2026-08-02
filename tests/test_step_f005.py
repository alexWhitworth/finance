"""Tests for F-005: _build_initial_state.

Verifies:
- No-LEAPS path: all accumulators empty, all NAV in base holdings.
- LEAPS path: ledger populated, all_window_ledgers has one entry.
- GTT + LEAPS path: price slice starts at the first Long day.
- Invariants: prev_total_nav == initial_nav, prev_regime == 1, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance.leverage import AccountType, LeapsConfig, LeapsContract, RebalanceRule, WeightStrategy
from finance.portfolio import BacktestContext, PortfolioConfig, PortfolioState
from finance.returns import ReturnData
from finance._step_f005 import _build_initial_state


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_return_data(dates: pd.DatetimeIndex) -> ReturnData:
    """Build a minimal ReturnData over the given dates.

    Arguments:
        dates: Business-day index for the return series.

    Returns:
        ReturnData with a single VTI asset and constant risk-free rate.
    """
    rng = np.random.default_rng(0)
    simple = rng.normal(0.0003, 0.01, len(dates))
    returns = pd.DataFrame({"VTI": simple}, index=dates)
    log_returns = pd.DataFrame({"VTI": np.log1p(simple)}, index=dates)
    rfr = pd.Series(0.04 / 252, index=dates, name="risk_free_rate")
    return ReturnData(
        returns=returns,
        log_returns=log_returns,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr,
    )


def _make_no_leaps_ctx(
    initial_nav: float = 100_000.0,
    weights: dict[str, float] | None = None,
) -> BacktestContext:
    """Build a BacktestContext with LEAPS disabled.

    Arguments:
        initial_nav: Starting portfolio value.
        weights: Target weights dict; defaults to {"VTI": 1.0}.

    Returns:
        BacktestContext with use_leaps=False and all optional series set to None.
    """
    if weights is None:
        weights = {"VTI": 1.0}
    dates = pd.bdate_range("2020-01-02", periods=60)
    config = PortfolioConfig(
        target_weights=weights,
        initial_nav=initial_nav,
        monthly_contribution=1000.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )
    return_data = _make_return_data(dates)
    base_assets = tuple(weights.keys())
    w = pd.Series(weights)
    base_target_w = w / w.sum()
    rebal_dates: frozenset[pd.Timestamp] = frozenset(
        {pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30")}
    )
    month_end_dates: frozenset[pd.Timestamp] = frozenset(
        {pd.Timestamp(d) for d in dates[::20]}
    )
    return BacktestContext(
        base_assets=base_assets,
        leaps_keys=(),
        leaps_fraction=0.0,
        base_target_w=base_target_w,
        governed_base=(),
        gtt_active=False,
        defensive_weights={},
        use_leaps=False,
        iv=0.18,
        leaps_monthly=0.0,
        base_contribution=1000.0,
        config=config,
        return_data=return_data,
        underlying_prices=None,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=None,
        mask_aligned=None,
        def_gross=None,
        rebal_dates=rebal_dates,
        month_end_dates=month_end_dates,
        long_window_end={pd.Timestamp(dates[0]): pd.Timestamp(dates[-1])},
        w=w,
    )


def _make_leaps_ctx(
    *,
    gtt_active: bool = False,
    mask_aligned: pd.Series | None = None,
    initial_nav: float = 100_000.0,
    n_periods: int = 126,
) -> BacktestContext:
    """Build a BacktestContext with LEAPS enabled.

    Arguments:
        gtt_active: Whether the GTT overlay is active.
        mask_aligned: GTT position mask (1=Long, 0=Defensive); required when
            gtt_active=True.
        initial_nav: Starting portfolio value.
        n_periods: Number of business days in the simulated period.

    Returns:
        BacktestContext with use_leaps=True, underlying_prices set, and leaps_config set.
    """
    dates = pd.bdate_range("2020-01-02", periods=n_periods)
    leaps_config = LeapsConfig(iv=0.22, ltcg_rate=0.20)
    config = PortfolioConfig(
        target_weights={"VTI": 0.85, "VTI_LEAPS": 0.15},
        initial_nav=initial_nav,
        monthly_contribution=1000.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=leaps_config,
    )
    return_data = _make_return_data(dates)
    rng = np.random.default_rng(42)
    prices = pd.Series(
        200.0 * np.cumprod(1 + rng.normal(0, 0.01, n_periods)),
        index=dates,
    )
    rfr = pd.Series(0.04 / 252, index=dates, name="risk_free_rate")
    w = pd.Series({"VTI": 0.85, "VTI_LEAPS": 0.15})
    base_target_w = pd.Series({"VTI": 1.0})
    long_window_end: dict[pd.Timestamp, pd.Timestamp] = {}
    if mask_aligned is not None:
        from finance.portfolio import _long_windows
        for start, end in _long_windows(mask_aligned):
            long_window_end[start] = end
    else:
        long_window_end = {pd.Timestamp(dates[0]): pd.Timestamp(dates[-1])}

    rebal_dates: frozenset[pd.Timestamp] = frozenset(
        {pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30")}
    )
    month_end_dates: frozenset[pd.Timestamp] = frozenset(
        {pd.Timestamp(d) for d in dates[::20]}
    )
    return BacktestContext(
        base_assets=("VTI",),
        leaps_keys=("VTI_LEAPS",),
        leaps_fraction=0.15,
        base_target_w=base_target_w,
        governed_base=("VTI",) if gtt_active else (),
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0} if gtt_active else {},
        use_leaps=True,
        iv=0.22,
        leaps_monthly=1000.0 * 0.15,
        base_contribution=1000.0 * 0.85,
        config=config,
        return_data=return_data,
        underlying_prices=prices,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=rfr,
        mask_aligned=mask_aligned,
        def_gross=None,
        rebal_dates=rebal_dates,
        month_end_dates=month_end_dates,
        long_window_end=long_window_end,
        w=w,
    )


# ---------------------------------------------------------------------------
# No-LEAPS tests
# ---------------------------------------------------------------------------


def test_no_leaps_ledger_is_none() -> None:
    """When use_leaps=False, leaps_ledger must be None."""
    ctx = _make_no_leaps_ctx()
    state = _build_initial_state(ctx)
    assert state.leaps_ledger is None


def test_no_leaps_leaps_value_zero() -> None:
    """When use_leaps=False, leaps_value must be 0.0."""
    ctx = _make_no_leaps_ctx()
    state = _build_initial_state(ctx)
    assert state.leaps_value == 0.0


def test_no_leaps_all_window_ledgers_empty() -> None:
    """When use_leaps=False, all_window_ledgers must be an empty tuple."""
    ctx = _make_no_leaps_ctx()
    state = _build_initial_state(ctx)
    assert state.all_window_ledgers == ()


def test_no_leaps_holdings_sum_equals_initial_nav() -> None:
    """Without LEAPS (leaps_fraction=0), all NAV goes into base holdings."""
    ctx = _make_no_leaps_ctx(initial_nav=100_000.0)
    state = _build_initial_state(ctx)
    assert sum(state.holdings.values()) == pytest.approx(ctx.config.initial_nav, rel=1e-9)


# ---------------------------------------------------------------------------
# Holdings arithmetic tests
# ---------------------------------------------------------------------------


def test_holdings_sum_equals_base_nav_slice() -> None:
    """Holdings sum == initial_nav * (1 - leaps_fraction) for any leaps_fraction."""
    ctx = _make_leaps_ctx()
    state = _build_initial_state(ctx)
    expected_base = ctx.config.initial_nav * (1.0 - ctx.leaps_fraction)
    assert sum(state.holdings.values()) == pytest.approx(expected_base, rel=1e-9)


def test_holdings_weights_match_base_target_w() -> None:
    """Holdings ratios must equal base_target_w ratios for a 60/40 base portfolio."""
    ctx = _make_no_leaps_ctx(weights={"VTI": 0.6, "VXUS": 0.4})
    # Rebuild ctx so base_target_w reflects the two-asset split.
    state = _build_initial_state(ctx)
    ratio = state.holdings["VTI"] / state.holdings["VXUS"]
    assert ratio == pytest.approx(0.6 / 0.4, rel=1e-9)


# ---------------------------------------------------------------------------
# prev_total_nav / sentinel field tests
# ---------------------------------------------------------------------------


def test_prev_total_nav_equals_initial_nav() -> None:
    """prev_total_nav must equal initial_nav regardless of LEAPS fraction."""
    ctx_base = _make_no_leaps_ctx(initial_nav=50_000.0)
    ctx_leaps = _make_leaps_ctx(initial_nav=50_000.0)
    for ctx in (ctx_base, ctx_leaps):
        state = _build_initial_state(ctx)
        assert state.prev_total_nav == ctx.config.initial_nav


def test_sentinel_fields_always_at_defaults() -> None:
    """prev_regime=1, prev_date_ts=None, all_gtt_closes=() on first iteration."""
    ctx = _make_no_leaps_ctx()
    state = _build_initial_state(ctx)
    assert state.prev_regime == 1
    assert state.prev_date_ts is None
    assert state.all_gtt_closes == ()


# ---------------------------------------------------------------------------
# LEAPS-active tests
# ---------------------------------------------------------------------------


def test_leaps_active_ledger_not_none() -> None:
    """When use_leaps=True and underlying_prices is set, leaps_ledger is not None."""
    ctx = _make_leaps_ctx()
    state = _build_initial_state(ctx)
    assert state.leaps_ledger is not None


def test_leaps_active_all_window_ledgers_has_one_entry() -> None:
    """When LEAPS are active, all_window_ledgers must contain exactly one ledger."""
    ctx = _make_leaps_ctx()
    state = _build_initial_state(ctx)
    assert len(state.all_window_ledgers) == 1
    assert state.all_window_ledgers[0] is state.leaps_ledger


# ---------------------------------------------------------------------------
# GTT + LEAPS: first Long window slicing
# ---------------------------------------------------------------------------


def test_gtt_leaps_first_long_window_used() -> None:
    """With GTT active, LEAPS simulation uses prices from the first Long day onward.

    Mask: 4 Defensive days, then all Long. We verify that no contract was
    purchased before the first Long day (index 4).
    """
    dates = pd.bdate_range("2020-01-02", periods=126)
    # First 4 days Defensive (0), rest Long (1)
    values = [0] * 4 + [1] * (len(dates) - 4)
    mask = pd.Series(values, index=dates, dtype=int)
    first_long_date = dates[4]

    ctx = _make_leaps_ctx(gtt_active=True, mask_aligned=mask)
    state = _build_initial_state(ctx)

    assert state.leaps_ledger is not None
    # Every contract in the ledger must have a purchase_date >= first_long_date.
    for contract in state.leaps_ledger.contracts:
        assert contract.purchase_date >= pd.Timestamp(first_long_date), (
            f"Contract purchased on {contract.purchase_date} before first Long day "
            f"{first_long_date}"
        )


def test_gtt_leaps_all_defensive_mask_gives_empty_ledger() -> None:
    """When the mask is all Defensive, run_leaps_simulation returns an empty ledger."""
    dates = pd.bdate_range("2020-01-02", periods=126)
    mask = pd.Series([0] * len(dates), index=dates, dtype=int)

    ctx = _make_leaps_ctx(gtt_active=True, mask_aligned=mask)
    state = _build_initial_state(ctx)

    assert state.leaps_ledger is not None
    # Empty price series → no contracts created.
    assert len(state.leaps_ledger.contracts) == 0
