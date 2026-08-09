"""Tests for F-GP-08: _apply_gtt_reentry uses compute_glide_target_weights when glide_path_config is present."""

import numpy as np
import pandas as pd
import pytest

from dataclasses import replace

from finance._backtest_steps import (
    _apply_gtt_reentry,
    compute_glide_target_weights,
)
from finance._portfolio_types import (
    BacktestContext,
    DayInputs,
    GlidepathConfig,
    GttConfig,
    PortfolioConfig,
    PortfolioState,
)
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance.returns import ReturnData

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_INITIAL_NAV = 100_000.0
_MONTHLY = 500.0
_RFR = 0.04

# Glide-path portfolio with LEAPS + VTI (base target = 0.0)
_WEIGHTS_GP: dict[str, float] = {
    "VTI": 0.0,
    "VTI_LEAPS": 0.40,
    "VXUS": 0.20,
    "GLD": 0.15,
    "MUB": 0.15,
    "KMLM": 0.05,
    "VGIT": 0.05,
}
_GP_CONFIG = GlidepathConfig(half_life_multiple=2.0, floor=0.05, vti_alpha=0.65)
# Use R_f-only defensive weights to avoid validation errors on small target_weights dicts.
_GTT_CONFIG = GttConfig(vix_p90_threshold=0.272, defensive_weights={"R_f": 1.0})
_LEAPS_CONFIG = LeapsConfig(iv=0.18, account_type=AccountType.TAX_SHELTERED)

_DATES = pd.bdate_range("2020-01-02", periods=252)
_RFR_SERIES = pd.Series(_RFR, index=_DATES, name="risk_free_rate")

# Non-LEAPS portfolio for regression tests
_WEIGHTS_NO_GP: dict[str, float] = {
    "VTI": 0.85,
    "VTI_LEAPS": 0.15,
}

# ---------------------------------------------------------------------------
# BacktestContext builder helpers (manual, so we control use_leaps independently)
# ---------------------------------------------------------------------------


def _make_return_data_for(tickers: tuple[str, ...]) -> ReturnData:
    rng = np.random.default_rng(42)
    rets = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(len(_DATES), len(tickers))),
        index=_DATES,
        columns=list(tickers),
    )
    log_rets = pd.DataFrame(np.log1p(rets.values), index=_DATES, columns=rets.columns)
    return ReturnData(
        returns=rets,
        log_returns=log_rets,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=_RFR_SERIES,
    )


def _make_gp_ctx(
    *,
    use_leaps: bool = False,
    gtt_active: bool = True,
) -> BacktestContext:
    """Build a BacktestContext for the glide-path portfolio.

    Arguments:
        use_leaps: Set ctx.use_leaps (controls whether LEAPS simulation fires in re-entry).
        gtt_active: Whether GTT overlay is active (required for re-entry to trigger).

    Returns:
        BacktestContext with glide_path_config set and all fields needed by _apply_gtt_reentry.
    """
    config = PortfolioConfig(
        target_weights=_WEIGHTS_GP,
        initial_nav=_INITIAL_NAV,
        monthly_contribution=_MONTHLY,
        rebalance_rule=RebalanceRule.DRIFT,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=_LEAPS_CONFIG,
        gtt_config=_GTT_CONFIG,
        glide_path_config=_GP_CONFIG,
    )
    base_assets = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
    leaps_keys = ("VTI_LEAPS",)
    leaps_fraction = 0.40
    w = pd.Series(_WEIGHTS_GP, dtype=float)
    # base_target_w: weights over base_assets normalized (excluding LEAPS)
    base_vals = {a: _WEIGHTS_GP[a] for a in base_assets}
    base_sum = sum(base_vals.values())  # 0.60
    base_target_w = pd.Series({a: v / base_sum for a, v in base_vals.items()})

    ret_data = _make_return_data_for(tuple(_WEIGHTS_GP.keys()))

    # GTT mask: first half defensive (0), second half long (1)
    n = len(_DATES)
    mask = pd.Series([0] * (n // 2) + [1] * (n - n // 2), index=_DATES, dtype=int)

    long_window_end: dict[pd.Timestamp, pd.Timestamp] = {}
    if gtt_active:
        # Re-entry date is the first Long day
        reentry_date = _DATES[n // 2]
        long_window_end[reentry_date] = _DATES[-1]

    return BacktestContext(
        base_assets=base_assets,
        leaps_keys=leaps_keys,
        leaps_fraction=leaps_fraction,
        base_target_w=base_target_w,
        governed_base=("VTI",),
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0},
        use_leaps=use_leaps,
        iv=0.18,
        leaps_monthly=_MONTHLY * leaps_fraction,
        base_contribution=_MONTHLY * (1.0 - leaps_fraction),
        config=config,
        return_data=ret_data,
        underlying_prices=None,  # no LEAPS simulation
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=_RFR_SERIES,
        mask_aligned=mask if gtt_active else None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end=long_window_end,
        w=w,
    )


def _make_no_gp_ctx(*, gtt_active: bool = True) -> BacktestContext:
    """Build a BacktestContext for a LEAPS portfolio WITHOUT glide_path_config."""
    config = PortfolioConfig(
        target_weights=_WEIGHTS_NO_GP,
        initial_nav=_INITIAL_NAV,
        monthly_contribution=_MONTHLY,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=_LEAPS_CONFIG,
        gtt_config=_GTT_CONFIG,
        glide_path_config=None,
    )
    base_assets = ("VTI",)
    leaps_keys = ("VTI_LEAPS",)
    leaps_fraction = 0.15
    w = pd.Series(_WEIGHTS_NO_GP, dtype=float)
    base_target_w = pd.Series({"VTI": 1.0})

    ret_data = _make_return_data_for(("VTI", "VTI_LEAPS"))

    n = len(_DATES)
    mask = pd.Series([0] * (n // 2) + [1] * (n - n // 2), index=_DATES, dtype=int)
    reentry_date = _DATES[n // 2]

    return BacktestContext(
        base_assets=base_assets,
        leaps_keys=leaps_keys,
        leaps_fraction=leaps_fraction,
        base_target_w=base_target_w,
        governed_base=("VTI",),
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0},
        use_leaps=False,  # no LEAPS simulation in tests
        iv=0.18,
        leaps_monthly=_MONTHLY * leaps_fraction,
        base_contribution=_MONTHLY * (1.0 - leaps_fraction),
        config=config,
        return_data=ret_data,
        underlying_prices=None,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=_RFR_SERIES,
        mask_aligned=mask if gtt_active else None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end={reentry_date: _DATES[-1]} if gtt_active else {},
        w=w,
    )


def _make_state(
    ctx: BacktestContext,
    total: float,
    hurdle: float,
    prev_regime: int = 0,
    use_sleeve: bool = False,
) -> PortfolioState:
    """Build a PortfolioState with controlled total NAV and hurdle_contributed.

    _apply_gtt_reentry computes: total = sum(holdings) + sleeve + pool  (leaps_value excluded).
    For a defensive state, all capital sits in sleeve + pool; holdings and leaps_value are 0.

    Arguments:
        ctx: BacktestContext providing base_assets and leaps_fraction.
        total: Total NAV to distribute.
        hurdle: hurdle_contributed (determines m = total / hurdle).
        prev_regime: GTT regime from the previous day.
        use_sleeve: If True, park all capital in sleeve+pool (like a mid-defensive-window state).

    Returns:
        PortfolioState ready for _apply_gtt_reentry.
    """
    if use_sleeve:
        # Defensive state: all capital parked in sleeve + pool
        holdings = {a: 0.0 for a in ctx.base_assets}
        sleeve = total * 0.7
        pool = total * 0.3
        leaps_val = 0.0
    else:
        # Just-entered-defensive on the same day (holdings still non-zero, leaps_value=0)
        holdings = {a: total * float(ctx.base_target_w[a]) * (1.0 - ctx.leaps_fraction) for a in ctx.base_assets}
        sleeve = 0.0
        pool = total * ctx.leaps_fraction  # LEAPS capital in pool
        leaps_val = 0.0

    return PortfolioState(
        holdings=holdings,
        defensive_sleeve=sleeve,
        leaps_pool=pool,
        leaps_value=leaps_val,
        prev_total_nav=total,
        prev_regime=prev_regime,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
        hurdle_contributed=hurdle,
        dynamic_target_weights=compute_glide_target_weights(
            total / hurdle, ctx.config, _GP_CONFIG
        ) if ctx.config.glide_path_config is not None else None,
    )


def _reentry_inputs() -> DayInputs:
    """Build DayInputs for the first Long day after a Defensive window."""
    return DayInputs(
        date_ts=_DATES[len(_DATES) // 2],
        day_ret=pd.Series(dtype=float),
        regime_t=1,
        def_gross_return=0.0,
        spot=200.0,
        raw_vix_value=None,
        mtm_iv_value=None,
        rfr=_RFR,
        is_month_end=False,
        is_rebal_date=False,
    )


# ---------------------------------------------------------------------------
# Unit tests for F-GP-08
# ---------------------------------------------------------------------------


def test_gtt_reentry_glide_path_base_allocs_at_m2() -> None:
    """Under glide_path_config with m_current=2.0, each base asset (including VTI) is
    allocated at dynamic_targets[a] * total within 1e-6.
    """
    ctx = _make_gp_ctx()
    total = _INITIAL_NAV
    hurdle = total / 2.0  # m=2.0
    state = _make_state(ctx, total=total, hurdle=hurdle, prev_regime=0)

    expected_targets = compute_glide_target_weights(2.0, ctx.config, _GP_CONFIG)

    inputs = _reentry_inputs()
    new_state = _apply_gtt_reentry(state, inputs, ctx)

    for a in ctx.base_assets:
        expected = float(expected_targets[a]) * total
        np.testing.assert_allclose(
            new_state.holdings[a],
            expected,
            atol=1e-6,
            err_msg=f"{a}: expected {expected:.6f}, got {new_state.holdings[a]:.6f}",
        )


def test_gtt_reentry_glide_path_vti_allocation_positive_at_m2() -> None:
    """VTI allocation at re-entry is positive at m=2.0 (dynamic target > 0.0)."""
    ctx = _make_gp_ctx()
    total = _INITIAL_NAV
    hurdle = total / 2.0  # m=2.0
    state = _make_state(ctx, total=total, hurdle=hurdle, prev_regime=0)

    expected_targets = compute_glide_target_weights(2.0, ctx.config, _GP_CONFIG)
    expected_vti = float(expected_targets["VTI"]) * total

    assert expected_vti > 0.0, "VTI target at m=2.0 must be positive (glide path allocates freed weight)"

    inputs = _reentry_inputs()
    new_state = _apply_gtt_reentry(state, inputs, ctx)

    np.testing.assert_allclose(new_state.holdings["VTI"], expected_vti, atol=1e-6)


def test_gtt_reentry_glide_path_nav_conserved() -> None:
    """Total NAV (holdings + leaps_value) post-reentry equals pre-reentry total within 1e-6.

    With use_leaps=False, leaps_value stays 0 so we verify sum(holdings) equals the
    base fraction of NAV (dynamic_targets base weight sum * total).
    """
    ctx = _make_gp_ctx(use_leaps=False)
    total = _INITIAL_NAV
    hurdle = total / 2.0
    state = _make_state(ctx, total=total, hurdle=hurdle, prev_regime=0, use_sleeve=True)

    expected_targets = compute_glide_target_weights(2.0, ctx.config, _GP_CONFIG)
    # With use_leaps=False: new_leaps_value=0, so only base allocations consume NAV
    base_frac = float(sum(expected_targets[a] for a in ctx.base_assets))
    expected_holdings_total = base_frac * total

    inputs = _reentry_inputs()
    new_state = _apply_gtt_reentry(state, inputs, ctx)

    holdings_total = sum(new_state.holdings.values())
    np.testing.assert_allclose(holdings_total, expected_holdings_total, atol=1e-6)

    # Sleeve and pool must be zeroed
    assert new_state.defensive_sleeve == 0.0
    assert new_state.leaps_pool == 0.0


def test_gtt_reentry_not_fired_when_prev_regime_1() -> None:
    """_apply_gtt_reentry is a no-op when prev_regime == 1 (not coming from Defensive)."""
    ctx = _make_gp_ctx()
    total = _INITIAL_NAV
    hurdle = total / 2.0
    # prev_regime=1 means no Defensive->Long transition
    state = _make_state(ctx, total=total, hurdle=hurdle, prev_regime=1)

    inputs = _reentry_inputs()
    new_state = _apply_gtt_reentry(state, inputs, ctx)

    assert new_state is state, "Expected no-op (state returned unchanged)"


def test_gtt_reentry_non_glide_path_uses_fixed_leaps_fraction() -> None:
    """Non-glide-path re-entry allocates base at (1 - ctx.leaps_fraction) * base_target_w.

    This verifies no regression to the existing behavior.
    """
    ctx = _make_no_gp_ctx()
    total = _INITIAL_NAV
    state = _make_state(ctx, total=total, hurdle=total, prev_regime=0)

    inputs = _reentry_inputs()
    new_state = _apply_gtt_reentry(state, inputs, ctx)

    expected_base_total = total * (1.0 - ctx.leaps_fraction)
    for a in ctx.base_assets:
        expected = expected_base_total * float(ctx.base_target_w[a])
        np.testing.assert_allclose(
            new_state.holdings[a],
            expected,
            atol=1e-6,
            err_msg=f"{a}: expected {expected:.6f}, got {new_state.holdings[a]:.6f}",
        )


def test_gtt_reentry_glide_path_at_m_below_1_restores_full_leaps() -> None:
    """At re-entry with m < 1.0, dynamic_targets == original weights: VTI=0.0, LEAPS=w0.

    The base assets (excluding VTI) receive their original target_weights * total.
    VTI holding is 0.0 since full LEAPS leverage is restored.
    """
    ctx = _make_gp_ctx()
    total = _INITIAL_NAV
    hurdle = total * 2.0  # m = 0.5 (underwater)
    state = _make_state(ctx, total=total, hurdle=hurdle, prev_regime=0)

    expected_targets = compute_glide_target_weights(0.5, ctx.config, _GP_CONFIG)
    # At m <= 1.0, VTI == 0 exactly
    assert float(expected_targets["VTI"]) == pytest.approx(0.0, abs=1e-12)

    inputs = _reentry_inputs()
    new_state = _apply_gtt_reentry(state, inputs, ctx)

    np.testing.assert_allclose(new_state.holdings["VTI"], 0.0, atol=1e-12)

    # Other base assets receive their original target_weights allocations
    for a in ctx.base_assets:
        if a == "VTI":
            continue
        expected = float(expected_targets[a]) * total
        np.testing.assert_allclose(
            new_state.holdings[a],
            expected,
            atol=1e-6,
            err_msg=f"{a}: expected {expected:.6f} at m<1, got {new_state.holdings[a]:.6f}",
        )


def test_gtt_reentry_glide_path_weight_sum_at_m3() -> None:
    """Base allocation fractions (holdings[a] / total) sum to the correct non-LEAPS fraction.

    At m=3.0, sum(dynamic_targets[a] for a in base_assets) == 1 - LEAPS_weight_at_m3.
    """
    ctx = _make_gp_ctx()
    total = _INITIAL_NAV
    hurdle = total / 3.0  # m=3.0
    state = _make_state(ctx, total=total, hurdle=hurdle, prev_regime=0)

    expected_targets = compute_glide_target_weights(3.0, ctx.config, _GP_CONFIG)
    expected_base_frac = float(sum(expected_targets[a] for a in ctx.base_assets))

    inputs = _reentry_inputs()
    new_state = _apply_gtt_reentry(state, inputs, ctx)

    actual_base_frac = sum(new_state.holdings.values()) / total
    np.testing.assert_allclose(actual_base_frac, expected_base_frac, atol=1e-12)
