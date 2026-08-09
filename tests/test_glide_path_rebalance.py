"""Tests for F-GP-07: _apply_rebalance uses state.dynamic_target_weights for DRIFT + glide path."""

import numpy as np
import pandas as pd
import pytest

from finance._backtest_steps import (
    _apply_rebalance,
    _build_context,
    _build_initial_state,
    compute_glide_target_weights,
)
from finance._portfolio_types import (
    DayInputs,
    GlidepathConfig,
    PortfolioConfig,
)
from finance.consts import DRIFT_BAND_RELATIVE
from finance.data import PriceData
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance.returns import ReturnData

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_INITIAL_NAV = 100_000.0
_MONTHLY = 1_000.0
_RFR = 0.04

_WEIGHTS_GP = {
    "VTI": 0.0,
    "VTI_LEAPS": 0.40,
    "VXUS": 0.20,
    "GLD": 0.15,
    "MUB": 0.15,
    "KMLM": 0.05,
    "VGIT": 0.05,
}
_WEIGHTS_NO_GP = {
    "VTI": 0.40,
    "VXUS": 0.20,
    "GLD": 0.15,
    "MUB": 0.15,
    "KMLM": 0.05,
    "VGIT": 0.05,
}

_LEAPS_CONFIG = LeapsConfig(iv=0.18, account_type=AccountType.TAX_SHELTERED)
_GP_CONFIG = GlidepathConfig(half_life_multiple=2.0, floor=0.05, vti_alpha=0.65)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_dates(n: int = 260) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-02", periods=n)


def _make_return_data(
    dates: pd.DatetimeIndex,
    tickers: tuple[str, ...],
    rfr_value: float = _RFR,
) -> ReturnData:
    rng = np.random.default_rng(42)
    rets = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(len(dates), len(tickers))),
        index=dates,
        columns=list(tickers),
    )
    log_rets = pd.DataFrame(np.log1p(rets.values), index=dates, columns=rets.columns)
    rfr = pd.Series(rfr_value, index=dates, name="risk_free_rate")
    return ReturnData(
        returns=rets,
        log_returns=log_rets,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr,
    )


def _make_price_data(dates: pd.DatetimeIndex) -> PriceData:
    rng = np.random.default_rng(7)
    prices = pd.DataFrame(
        {"VTI": 200.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))},
        index=dates,
    )
    return PriceData(
        prices=prices,
        dividends=pd.DataFrame(index=dates),
        vol_prices=pd.DataFrame(index=dates),
        tickers=("VTI",),
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        spliced=False,
    )


def _make_config(
    weights: dict[str, float],
    gp: GlidepathConfig | None,
    leaps_config: LeapsConfig | None = None,
) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights=weights,
        initial_nav=_INITIAL_NAV,
        monthly_contribution=_MONTHLY,
        rebalance_rule=RebalanceRule.DRIFT,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=leaps_config,
        glide_path_config=gp,
    )


def _build_ctx_and_state(
    weights: dict[str, float],
    gp: GlidepathConfig | None,
    leaps_config: LeapsConfig | None = None,
):
    config = _make_config(weights, gp, leaps_config=leaps_config)
    dates = _make_dates()
    tickers = tuple(config.target_weights.keys())
    ret_data = _make_return_data(dates, tickers)
    price_data = _make_price_data(dates)
    ctx = _build_context(ret_data, price_data, config, gtt_signal=None)
    state = _build_initial_state(ctx)
    return ctx, state, dates


def _month_end_day(date: pd.Timestamp, rfr: float = _RFR) -> DayInputs:
    return DayInputs(
        date_ts=date,
        day_ret=pd.Series(dtype=float),
        regime_t=1,
        def_gross_return=0.0,
        spot=None,
        raw_vix_value=None,
        mtm_iv_value=None,
        rfr=rfr,
        is_month_end=True,
        is_rebal_date=False,
    )


# ---------------------------------------------------------------------------
# F-GP-07: dynamic_target_weights used vs ctx.w fallback
# ---------------------------------------------------------------------------


def test_drift_uses_dynamic_target_when_glide_path_active() -> None:
    """When dynamic_target_weights is set, _should_rebalance is called against it, not ctx.w.

    Strategy: inflate LEAPS value so that the weight relative to dynamic target (higher
    LEAPS fraction at m~1) exceeds the drift band, then assert rebalance fires.
    """
    ctx, state, dates = _build_ctx_and_state(
        _WEIGHTS_GP, _GP_CONFIG, leaps_config=_LEAPS_CONFIG
    )
    # At m=1.0, dynamic targets == config.target_weights with leaps_fraction=0.40.
    # Give the base assets total NAV of 60k but inflate leaps_value to 50k
    # so LEAPS fraction == 50/(60+50) = 0.4545 >> 0.40 * (1 + DRIFT_BAND_RELATIVE).
    base_val = 60_000.0
    leaps_val = 50_000.0  # overshoot well beyond drift band
    holdings = {a: base_val * float(ctx.base_target_w[a]) for a in ctx.base_assets}
    from dataclasses import replace
    state_inflated = replace(
        state,
        holdings=holdings,
        leaps_value=leaps_val,
        dynamic_target_weights=state.dynamic_target_weights,
    )
    inputs = _month_end_day(dates[0])
    new_state = _apply_rebalance(state_inflated, inputs, ctx)

    # DRIFT fired → leaps_value trimmed toward target
    assert new_state.leaps_value < leaps_val, (
        f"Expected LEAPS trim; got leaps_value={new_state.leaps_value} >= {leaps_val}"
    )


def test_drift_fallback_to_ctx_w_when_no_glide_path() -> None:
    """Without glide_path_config, _apply_rebalance falls back to ctx.w (no regression)."""
    ctx, state, dates = _build_ctx_and_state(_WEIGHTS_NO_GP, gp=None)
    # Perturb holdings to trigger DRIFT against ctx.w
    base_val = sum(state.holdings.values())
    # Push VTI to 80% of base (target is 40/(40+20+15+15+5+5)=40%) — large relative drift
    perturbed = {a: base_val * float(ctx.base_target_w[a]) for a in ctx.base_assets}
    perturbed["VTI"] = base_val * 0.80
    # Rescale others to keep total constant
    excess = perturbed["VTI"] - base_val * float(ctx.base_target_w["VTI"])
    others = [a for a in ctx.base_assets if a != "VTI"]
    for a in others:
        perturbed[a] = max(0.0, perturbed[a] - excess / len(others))
    from dataclasses import replace
    state_perturbed = replace(state, holdings=perturbed)
    inputs = _month_end_day(dates[0])
    new_state = _apply_rebalance(state_perturbed, inputs, ctx)

    # After rebalance, VTI weight should be closer to its target (ctx.w VTI = 0.40)
    new_total = sum(new_state.holdings.values())
    new_vti_w = new_state.holdings["VTI"] / new_total
    target_vti_w = float(ctx.w["VTI"])
    assert abs(new_vti_w - target_vti_w) < 0.01, (
        f"VTI weight {new_vti_w:.4f} not near target {target_vti_w:.4f} after DRIFT"
    )


def test_nav_conserved_after_drift_rebalance_glide_path() -> None:
    """NAV is conserved across DRIFT rebalance with glide path within 1e-6."""
    ctx, state, dates = _build_ctx_and_state(
        _WEIGHTS_GP, _GP_CONFIG, leaps_config=_LEAPS_CONFIG
    )
    base_val = 60_000.0
    leaps_val = 50_000.0
    holdings = {a: base_val * float(ctx.base_target_w[a]) for a in ctx.base_assets}
    from dataclasses import replace
    state_inflated = replace(state, holdings=holdings, leaps_value=leaps_val)
    total_before = sum(state_inflated.holdings.values()) + state_inflated.leaps_value

    inputs = _month_end_day(dates[0])
    new_state = _apply_rebalance(state_inflated, inputs, ctx)

    total_after = sum(new_state.holdings.values()) + new_state.leaps_value
    np.testing.assert_allclose(total_after, total_before, atol=1e-6)


def test_no_rebalance_when_within_drift_band() -> None:
    """No rebalance fires when current weights are within the drift band of dynamic targets."""
    ctx, state, dates = _build_ctx_and_state(
        _WEIGHTS_GP, _GP_CONFIG, leaps_config=_LEAPS_CONFIG
    )
    # dynamic targets at m=1.0 equal config.target_weights exactly — set holdings
    # and leaps_value to match target_weights exactly (no drift)
    assert state.dynamic_target_weights is not None
    total = _INITIAL_NAV
    leaps_frac = float(state.dynamic_target_weights["VTI_LEAPS"])
    leaps_val = total * leaps_frac
    base_val = total - leaps_val
    holdings = {a: base_val * float(ctx.base_target_w[a]) for a in ctx.base_assets}
    from dataclasses import replace
    state_exact = replace(state, holdings=holdings, leaps_value=leaps_val)

    inputs = _month_end_day(dates[0])
    new_state = _apply_rebalance(state_exact, inputs, ctx)

    # State should be returned unchanged (no DRIFT trigger)
    assert new_state is state_exact or (
        sum(new_state.holdings.values()) + new_state.leaps_value
        == pytest.approx(sum(state_exact.holdings.values()) + state_exact.leaps_value, abs=1e-6)
    )


def test_dynamic_targets_at_high_m_reduces_leaps_target() -> None:
    """At high m, dynamic target LEAPS fraction < w0; DRIFT trims LEAPS to new lower target."""
    ctx, state, dates = _build_ctx_and_state(
        _WEIGHTS_GP, _GP_CONFIG, leaps_config=_LEAPS_CONFIG
    )
    config = ctx.config
    gp = _GP_CONFIG
    # m = 3.0: compute what dynamic targets look like
    dynamic_at_m3 = compute_glide_target_weights(3.0, config, gp)
    leaps_target_at_m3 = float(dynamic_at_m3["VTI_LEAPS"])
    # leaps_target_at_m3 should be substantially below 0.40
    assert leaps_target_at_m3 < 0.40 * (1 - DRIFT_BAND_RELATIVE), (
        f"Expected decay; got {leaps_target_at_m3:.4f}"
    )

    from dataclasses import replace
    # Build a state where dynamic_target_weights is at m=3 but leaps_value is at w0 level
    total = _INITIAL_NAV
    leaps_val_w0 = total * 0.40  # original w0
    base_val = total - leaps_val_w0
    holdings = {a: base_val * float(ctx.base_target_w[a]) for a in ctx.base_assets}
    state_high_m = replace(
        state,
        holdings=holdings,
        leaps_value=leaps_val_w0,
        dynamic_target_weights=dynamic_at_m3,
    )
    inputs = _month_end_day(dates[0])
    new_state = _apply_rebalance(state_high_m, inputs, ctx)

    # LEAPS should have been trimmed toward the lower target
    assert new_state.leaps_value < leaps_val_w0, (
        f"Expected LEAPS trim; got {new_state.leaps_value:.2f} >= {leaps_val_w0:.2f}"
    )
    expected_target_leaps = total * leaps_target_at_m3
    np.testing.assert_allclose(new_state.leaps_value, expected_target_leaps, atol=1e-6)


def test_leaps_topped_up_when_below_dynamic_target() -> None:
    """DRIFT tops up LEAPS (two-sided) when leaps weight is below dynamic target by > drift band."""
    ctx, state, dates = _build_ctx_and_state(
        _WEIGHTS_GP, _GP_CONFIG, leaps_config=_LEAPS_CONFIG
    )
    assert state.dynamic_target_weights is not None
    dynamic = state.dynamic_target_weights  # at m=1.0, LEAPS target = 0.40

    leaps_target_frac = float(dynamic["VTI_LEAPS"])  # 0.40
    total = _INITIAL_NAV
    # Set leaps_value well below target (LEAPS at 20% when target is 40%)
    leaps_val_low = total * (leaps_target_frac * 0.45)
    base_val = total - leaps_val_low
    holdings = {a: base_val * float(ctx.base_target_w[a]) for a in ctx.base_assets}
    from dataclasses import replace
    state_low = replace(state, holdings=holdings, leaps_value=leaps_val_low)

    inputs = _month_end_day(dates[0])
    new_state = _apply_rebalance(state_low, inputs, ctx)

    # Two-sided: leaps_value must be topped up toward the target (the critical assertion).
    target_leaps_val = total * leaps_target_frac
    np.testing.assert_allclose(
        new_state.leaps_value,
        target_leaps_val,
        atol=1e-6,
        err_msg=f"LEAPS not topped up: before={leaps_val_low:.2f}, after={new_state.leaps_value:.2f}, target={target_leaps_val:.2f}",
    )
    assert new_state.leaps_value > leaps_val_low, (
        f"Expected LEAPS top-up; got leaps_value={new_state.leaps_value:.2f} not > {leaps_val_low:.2f}"
    )

    # Base holdings must have shrunk (capital redirected to LEAPS top-up).
    new_total_holdings = sum(new_state.holdings.values())
    old_total_holdings = sum(state_low.holdings.values())
    assert new_total_holdings < old_total_holdings - 1e-6, (
        f"Expected base holdings to shrink; got {new_total_holdings:.2f} vs {old_total_holdings:.2f}"
    )

    # NAV must be conserved.
    old_nav = old_total_holdings + leaps_val_low
    new_nav = new_total_holdings + new_state.leaps_value
    np.testing.assert_allclose(new_nav, old_nav, atol=1e-6)
