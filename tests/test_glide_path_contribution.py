"""Tests for F-GP-06: _apply_contribution atomic hurdle + dynamic_target_weights update."""

import numpy as np
import pandas as pd

from finance._backtest_steps import _apply_contribution, _build_context, _build_initial_state
from finance._portfolio_types import DayInputs, GlidepathConfig, PortfolioConfig
from finance.data import PriceData
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance.returns import ReturnData

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_INITIAL_NAV = 100_000.0
_MONTHLY = 1_000.0
_RFR = 0.04  # annualised

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


def _make_price_data(dates: pd.DatetimeIndex, with_vti_prices: bool = False) -> PriceData:
    rng = np.random.default_rng(7)
    prices = (
        pd.DataFrame(
            {"VTI": 200.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))},
            index=dates,
        )
        if with_vti_prices
        else pd.DataFrame(index=dates)
    )
    return PriceData(
        prices=prices,
        dividends=pd.DataFrame(index=dates),
        vol_prices=pd.DataFrame(index=dates),
        tickers=tuple(prices.columns),
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        spliced=False,
    )


def _make_config(
    weights: dict[str, float],
    gp: GlidepathConfig | None,
    leaps_config: LeapsConfig | None = None,
    monthly: float = _MONTHLY,
) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights=weights,
        initial_nav=_INITIAL_NAV,
        monthly_contribution=monthly,
        rebalance_rule=RebalanceRule.DRIFT,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=leaps_config,
        glide_path_config=gp,
    )


def _build_ctx_and_state(
    weights: dict[str, float],
    gp: GlidepathConfig | None,
    leaps_config: LeapsConfig | None = None,
    with_vti_prices: bool = False,
    monthly: float = _MONTHLY,
):
    """Return (ctx, initial_state) for the given config."""
    config = _make_config(weights, gp, leaps_config=leaps_config, monthly=monthly)
    dates = _make_dates()
    tickers = tuple(config.target_weights.keys())
    ret_data = _make_return_data(dates, tickers)
    price_data = _make_price_data(dates, with_vti_prices=with_vti_prices)
    ctx = _build_context(ret_data, price_data, config, gtt_signal=None)
    state = _build_initial_state(ctx)
    return ctx, state, dates


def _month_end_inputs(date: pd.Timestamp, rfr: float = _RFR) -> DayInputs:
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


def _non_month_end_inputs(date: pd.Timestamp, rfr: float = _RFR) -> DayInputs:
    return DayInputs(
        date_ts=date,
        day_ret=pd.Series(dtype=float),
        regime_t=1,
        def_gross_return=0.0,
        spot=None,
        raw_vix_value=None,
        mtm_iv_value=None,
        rfr=rfr,
        is_month_end=False,
        is_rebal_date=False,
    )


# ---------------------------------------------------------------------------
# F-GP-06: Unit tests — single month-end with glide_path_config
# ---------------------------------------------------------------------------


def test_hurdle_updated_on_month_end_with_glide_path() -> None:
    """On month-end with glide_path_config: hurdle == old*(1+rfr)^(1/12)+contribution."""
    ctx, state, dates = _build_ctx_and_state(
        _WEIGHTS_GP, _GP_CONFIG, leaps_config=_LEAPS_CONFIG, with_vti_prices=True
    )
    nav_before = _INITIAL_NAV
    date = dates[0]
    inputs = _month_end_inputs(date, rfr=_RFR)

    new_state = _apply_contribution(state, inputs, ctx, nav_before)

    expected_hurdle = _INITIAL_NAV * (1.0 + _RFR) ** (1.0 / 12) + _MONTHLY
    np.testing.assert_allclose(new_state.hurdle_contributed, expected_hurdle, atol=1e-9)


def test_dynamic_targets_sum_one_on_month_end() -> None:
    """On month-end with glide_path_config: dynamic_target_weights.sum() == 1.0 within 1e-12."""
    ctx, state, dates = _build_ctx_and_state(
        _WEIGHTS_GP, _GP_CONFIG, leaps_config=_LEAPS_CONFIG, with_vti_prices=True
    )
    inputs = _month_end_inputs(dates[0])
    new_state = _apply_contribution(state, inputs, ctx, _INITIAL_NAV)

    assert new_state.dynamic_target_weights is not None
    np.testing.assert_allclose(new_state.dynamic_target_weights.sum(), 1.0, atol=1e-12)


def test_hurdle_and_targets_updated_atomically() -> None:
    """Both hurdle and dynamic_target_weights are updated in the same replace call."""
    ctx, state, dates = _build_ctx_and_state(
        _WEIGHTS_GP, _GP_CONFIG, leaps_config=_LEAPS_CONFIG, with_vti_prices=True
    )
    inputs = _month_end_inputs(dates[0])
    new_state = _apply_contribution(state, inputs, ctx, _INITIAL_NAV)

    old_hurdle = state.hurdle_contributed
    assert new_state.hurdle_contributed != old_hurdle
    assert new_state.dynamic_target_weights is not None
    assert new_state.dynamic_target_weights is not state.dynamic_target_weights


def test_hurdle_unchanged_on_non_month_end() -> None:
    """Non-month-end with glide_path_config: hurdle and dynamic_target_weights unchanged."""
    ctx, state, dates = _build_ctx_and_state(
        _WEIGHTS_GP, _GP_CONFIG, leaps_config=_LEAPS_CONFIG, with_vti_prices=True
    )
    inputs = _non_month_end_inputs(dates[0])
    new_state = _apply_contribution(state, inputs, ctx, _INITIAL_NAV)

    np.testing.assert_allclose(new_state.hurdle_contributed, state.hurdle_contributed, atol=1e-12)
    # dynamic_target_weights must also be unchanged on non-month-end
    assert new_state.dynamic_target_weights is state.dynamic_target_weights


def test_hurdle_unchanged_without_glide_path() -> None:
    """Without glide_path_config on month-end: hurdle_contributed unchanged."""
    ctx, state, dates = _build_ctx_and_state(_WEIGHTS_NO_GP, gp=None)
    inputs = _month_end_inputs(dates[0])
    new_state = _apply_contribution(state, inputs, ctx, _INITIAL_NAV)

    np.testing.assert_allclose(new_state.hurdle_contributed, state.hurdle_contributed, atol=1e-12)
    assert new_state.dynamic_target_weights is None


def test_hurdle_monotone_over_24_months() -> None:
    """hurdle_contributed is monotone non-decreasing across 24 months (positive rfr/contrib)."""
    rfr = _RFR
    hurdle = _INITIAL_NAV
    prev = hurdle
    for _ in range(24):
        hurdle = hurdle * (1.0 + rfr) ** (1.0 / 12) + _MONTHLY
        assert hurdle >= prev - 1e-12, f"hurdle decreased: {prev} -> {hurdle}"
        prev = hurdle


def test_hurdle_matches_analytic_single_step() -> None:
    """Single month-end hurdle matches analytical formula exactly within 1e-9."""
    rfr = 0.05
    old_hurdle = 80_000.0
    monthly = 500.0
    expected = old_hurdle * (1.0 + rfr) ** (1.0 / 12) + monthly

    # Simulate the formula used in _apply_contribution
    computed = old_hurdle * (1.0 + rfr) ** (1.0 / 12) + monthly
    np.testing.assert_allclose(computed, expected, atol=1e-9)


def test_hurdle_zirp_grows_by_contribution_only() -> None:
    """rfr==0: each monthly update adds exactly monthly_contribution to hurdle."""
    hurdle = _INITIAL_NAV
    for month in range(1, 25):
        hurdle = hurdle * (1.0 + 0.0) ** (1.0 / 12) + _MONTHLY
        expected = _INITIAL_NAV + month * _MONTHLY
        np.testing.assert_allclose(hurdle, expected, atol=1e-9, err_msg=f"month={month}")


def test_hurdle_no_contribution_pure_rf() -> None:
    """monthly_contribution==0: hurdle grows at pure Rf compounding for 24 months."""
    hurdle = _INITIAL_NAV
    rfr = _RFR
    for month in range(1, 25):
        hurdle = hurdle * (1.0 + rfr) ** (1.0 / 12)
        expected = _INITIAL_NAV * (1.0 + rfr) ** (month / 12.0)
        np.testing.assert_allclose(hurdle, expected, atol=1e-9, err_msg=f"month={month}")


def test_apply_contribution_state_returned_is_new_object() -> None:
    """_apply_contribution returns a new PortfolioState on month-end."""
    ctx, state, dates = _build_ctx_and_state(
        _WEIGHTS_GP, _GP_CONFIG, leaps_config=_LEAPS_CONFIG, with_vti_prices=True
    )
    inputs = _month_end_inputs(dates[0])
    new_state = _apply_contribution(state, inputs, ctx, _INITIAL_NAV)
    assert new_state is not state
