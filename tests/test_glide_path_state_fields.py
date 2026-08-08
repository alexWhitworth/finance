"""Tests for F-GP-04: PortfolioState hurdle_contributed and dynamic_target_weights fields."""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from finance._backtest_steps import _build_context, _build_initial_state
from finance._portfolio_types import GlidepathConfig, PortfolioConfig, PortfolioState
from finance.data import PriceData
from finance.gtt import GttSignalData
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance.returns import ReturnData

# ---------------------------------------------------------------------------
# Shared test fixtures / builders
# ---------------------------------------------------------------------------

_TICKERS = ("VTI", "VTI_LEAPS", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
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
_INITIAL_NAV = 100_000.0
_MONTHLY = 1_000.0
_RFR = 0.04  # 4% annualised


def _make_dates(n: int = 260) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-02", periods=n)


def _make_return_data(dates: pd.DatetimeIndex, tickers: tuple[str, ...]) -> ReturnData:
    rng = np.random.default_rng(42)
    rets = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(len(dates), len(tickers))),
        index=dates,
        columns=list(tickers),
    )
    log_rets = pd.DataFrame(np.log1p(rets.values), index=dates, columns=rets.columns)
    rfr = pd.Series(_RFR, index=dates, name="risk_free_rate")
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


_LEAPS_CONFIG = LeapsConfig(iv=0.18, account_type=AccountType.TAX_SHELTERED)


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


def _build(config: PortfolioConfig, with_vti_prices: bool = False) -> PortfolioState:
    dates = _make_dates()
    tickers = tuple(config.target_weights.keys())
    ret_data = _make_return_data(dates, tickers)
    price_data = _make_price_data(dates, with_vti_prices=with_vti_prices)
    ctx = _build_context(ret_data, price_data, config, gtt_signal=None)
    return _build_initial_state(ctx)


# ---------------------------------------------------------------------------
# PortfolioState construction: new fields present and frozen
# ---------------------------------------------------------------------------


def test_portfolio_state_new_fields_default() -> None:
    """PortfolioState with existing fields only uses defaults for new fields."""
    state = PortfolioState(
        holdings={"VTI": 100_000.0},
        defensive_sleeve=0.0,
        leaps_pool=0.0,
        leaps_value=0.0,
        prev_total_nav=100_000.0,
        prev_regime=1,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )
    assert state.hurdle_contributed == 0.0
    assert state.dynamic_target_weights is None


def test_portfolio_state_new_fields_explicit() -> None:
    """PortfolioState accepts explicit values for hurdle_contributed and dynamic_target_weights."""
    dw = pd.Series({"VTI": 0.5, "VXUS": 0.5})
    state = PortfolioState(
        holdings={"VTI": 100_000.0},
        defensive_sleeve=0.0,
        leaps_pool=0.0,
        leaps_value=0.0,
        prev_total_nav=100_000.0,
        prev_regime=1,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
        hurdle_contributed=50_000.0,
        dynamic_target_weights=dw,
    )
    assert state.hurdle_contributed == 50_000.0
    assert state.dynamic_target_weights is not None
    np.testing.assert_allclose(state.dynamic_target_weights["VTI"], 0.5, atol=1e-12)


def test_portfolio_state_hurdle_frozen() -> None:
    """hurdle_contributed assignment raises FrozenInstanceError."""
    state = PortfolioState(
        holdings={}, defensive_sleeve=0.0, leaps_pool=0.0, leaps_value=0.0,
        prev_total_nav=0.0, prev_regime=1, prev_date_ts=None,
        leaps_ledger=None, leaps_scale={}, all_window_ledgers=(), all_gtt_closes=(),
        hurdle_contributed=100_000.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.hurdle_contributed = 200_000.0  # type: ignore[misc]


def test_portfolio_state_dynamic_target_weights_frozen() -> None:
    """dynamic_target_weights assignment raises FrozenInstanceError."""
    state = PortfolioState(
        holdings={}, defensive_sleeve=0.0, leaps_pool=0.0, leaps_value=0.0,
        prev_total_nav=0.0, prev_regime=1, prev_date_ts=None,
        leaps_ledger=None, leaps_scale={}, all_window_ledgers=(), all_gtt_closes=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.dynamic_target_weights = pd.Series({"VTI": 1.0})  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _build_initial_state: no glide path
# ---------------------------------------------------------------------------


def test_build_initial_state_no_glide_path_hurdle() -> None:
    """Without glide_path_config, hurdle_contributed == initial_nav within 1e-9."""
    state = _build(_make_config(_WEIGHTS_NO_GP, gp=None))
    np.testing.assert_allclose(state.hurdle_contributed, _INITIAL_NAV, atol=1e-9)


def test_build_initial_state_no_glide_path_dynamic_weights_none() -> None:
    """Without glide_path_config, dynamic_target_weights is None."""
    state = _build(_make_config(_WEIGHTS_NO_GP, gp=None))
    assert state.dynamic_target_weights is None


# ---------------------------------------------------------------------------
# _build_initial_state: with glide path
# ---------------------------------------------------------------------------


def test_build_initial_state_with_glide_path_hurdle() -> None:
    """With glide_path_config, hurdle_contributed == initial_nav within 1e-9."""
    gp = GlidepathConfig()
    state = _build(_make_config(_WEIGHTS_GP, gp=gp, leaps_config=_LEAPS_CONFIG), with_vti_prices=True)
    np.testing.assert_allclose(state.hurdle_contributed, _INITIAL_NAV, atol=1e-9)


def test_build_initial_state_with_glide_path_dynamic_weights_identity() -> None:
    """With glide_path_config, dynamic_target_weights == config.target_weights at m=1.0."""
    gp = GlidepathConfig()
    config = _make_config(_WEIGHTS_GP, gp=gp, leaps_config=_LEAPS_CONFIG)
    state = _build(config, with_vti_prices=True)
    assert state.dynamic_target_weights is not None
    for k, v in _WEIGHTS_GP.items():
        np.testing.assert_allclose(
            float(state.dynamic_target_weights[k]), v, atol=1e-12,
            err_msg=f"key={k}"
        )


def test_build_initial_state_with_glide_path_dynamic_weights_sum() -> None:
    """With glide_path_config, dynamic_target_weights.sum() == 1.0 within 1e-12."""
    gp = GlidepathConfig()
    state = _build(_make_config(_WEIGHTS_GP, gp=gp, leaps_config=_LEAPS_CONFIG), with_vti_prices=True)
    assert state.dynamic_target_weights is not None
    np.testing.assert_allclose(state.dynamic_target_weights.sum(), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Integration: hurdle monotonicity across 12 monthly steps (analytic check)
# ---------------------------------------------------------------------------


def test_hurdle_never_decreases_across_12_months() -> None:
    """hurdle_contributed never decreases across 12 monthly steps (non-negative rfr/contribution).

    Simulates the hurdle update formula directly:
        new = old * (1 + rfr)^(1/12) + monthly_contribution
    and asserts monotone non-decrease.
    """
    rfr = _RFR  # annualised
    hurdle = _INITIAL_NAV
    prev = hurdle
    for _ in range(12):
        hurdle = hurdle * (1.0 + rfr) ** (1.0 / 12) + _MONTHLY
        assert hurdle >= prev - 1e-12, f"hurdle decreased: {prev} -> {hurdle}"
        prev = hurdle


def test_hurdle_zirp_grows_by_monthly_contribution_only() -> None:
    """rfr==0: hurdle grows by exactly monthly_contribution each month."""
    hurdle = _INITIAL_NAV
    for month in range(1, 13):
        hurdle = hurdle * 1.0 + _MONTHLY  # (1+0)^(1/12) == 1.0
        expected = _INITIAL_NAV + month * _MONTHLY
        np.testing.assert_allclose(hurdle, expected, atol=1e-9, err_msg=f"month={month}")


def test_hurdle_no_contribution_grows_at_rf_compounding() -> None:
    """monthly_contribution==0: hurdle grows at pure Rf compounding."""
    rfr = _RFR
    hurdle = _INITIAL_NAV
    for month in range(1, 13):
        hurdle = hurdle * (1.0 + rfr) ** (1.0 / 12)
        expected = _INITIAL_NAV * (1.0 + rfr) ** (month / 12.0)
        np.testing.assert_allclose(hurdle, expected, atol=1e-9, err_msg=f"month={month}")


def test_hurdle_analytic_12_month_closed_form() -> None:
    """After 12 months, hurdle matches geometric-series closed form within 1e-6."""
    rfr = _RFR
    c = _MONTHLY
    h0 = _INITIAL_NAV
    r = (1.0 + rfr) ** (1.0 / 12)  # monthly growth factor

    # Closed form: h_n = h0 * r^n + c * (r^n - 1) / (r - 1)
    n = 12
    expected = h0 * r**n + c * (r**n - 1.0) / (r - 1.0)

    hurdle = h0
    for _ in range(n):
        hurdle = hurdle * r + c
    np.testing.assert_allclose(hurdle, expected, atol=1e-6)
