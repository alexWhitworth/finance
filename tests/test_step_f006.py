"""Tests for F-006: _extract_day_inputs pure index-lookup function."""

import math

import numpy as np
import pandas as pd
import pytest

from finance._step_f006 import _extract_day_inputs
from finance.leverage import RebalanceRule, WeightStrategy
from finance.portfolio import BacktestContext, PortfolioConfig
from finance.returns import ReturnData


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_ctx(dates: pd.DatetimeIndex) -> BacktestContext:
    """Build a minimal BacktestContext with known series over ``dates``.

    Arguments:
        dates: DatetimeIndex of trading days to populate.

    Returns:
        BacktestContext with deterministic series for all optional fields.
    """
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({"VTI": rng.normal(0.001, 0.01, len(dates))}, index=dates)
    rfr = pd.Series(0.04, index=dates)
    vix = pd.Series(0.20 + rng.normal(0, 0.02, len(dates)), index=dates)
    mtm_iv = vix.rolling(30).mean().ffill()

    # mask: first half Long (1), second half Defensive (0)
    mask = pd.Series([1] * len(dates), index=dates, dtype=int)
    mask.iloc[len(dates) // 2 :] = 0

    config = PortfolioConfig(
        target_weights={"VTI": 1.0},
        initial_nav=10000.0,
        monthly_contribution=500.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )
    return_data = ReturnData(
        returns=returns,
        log_returns=pd.DataFrame({"VTI": np.log1p(returns["VTI"].values)}, index=dates),
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr,
    )

    month_ends = frozenset(
        pd.Timestamp(g.index[-1])
        for _, g in returns.groupby(pd.DatetimeIndex(dates).to_period("M"))
    )

    return BacktestContext(
        base_assets=("VTI",),
        leaps_keys=(),
        leaps_fraction=0.0,
        base_target_w=pd.Series({"VTI": 1.0}),
        governed_base=("VTI",),
        gtt_active=True,
        defensive_weights={"R_f": 1.0},
        use_leaps=False,
        iv=0.20,
        leaps_monthly=0.0,
        base_contribution=500.0,
        config=config,
        return_data=return_data,
        underlying_prices=None,
        raw_vix=vix,
        mtm_iv_series=mtm_iv,
        rfr_series=rfr,
        mask_aligned=mask,
        def_gross=rfr * 0,
        rebal_dates=frozenset(),
        month_end_dates=month_ends,
        long_window_end={},
        w=pd.Series({"VTI": 1.0}),
    )


# ---------------------------------------------------------------------------
# Test 1: all fields match series for a known date
# ---------------------------------------------------------------------------


def test_all_fields_match_series() -> None:
    """All DayInputs fields equal expected values from the underlying series."""
    dates = pd.bdate_range("2022-01-03", periods=100)
    ctx = _make_ctx(dates)

    # Use a date well into the series (past warmup so mtm_iv_value is finite)
    date = dates[50]

    inputs = _extract_day_inputs(date, ctx)

    assert inputs.date_ts == date
    assert float(inputs.day_ret["VTI"]) == pytest.approx(
        float(ctx.return_data.returns.loc[date, "VTI"]), abs=1e-12
    )
    assert inputs.regime_t == int(ctx.mask_aligned.loc[date])  # type: ignore[index]
    assert inputs.def_gross_return == pytest.approx(
        float(ctx.def_gross.loc[date]), abs=1e-12  # type: ignore[index]
    )
    assert inputs.spot is None  # underlying_prices is None
    assert inputs.raw_vix_value == pytest.approx(
        float(ctx.raw_vix.loc[date]), abs=1e-12  # type: ignore[index]
    )
    assert inputs.mtm_iv_value == pytest.approx(
        float(ctx.mtm_iv_series.loc[date]), abs=1e-12  # type: ignore[index]
    )
    assert inputs.rfr == pytest.approx(
        float(ctx.rfr_series.loc[date]), abs=1e-12  # type: ignore[index]
    )
    assert isinstance(inputs.is_month_end, bool)
    assert inputs.is_month_end == (date in ctx.month_end_dates)
    assert isinstance(inputs.is_rebal_date, bool)
    assert inputs.is_rebal_date == (date in ctx.rebal_dates)


# ---------------------------------------------------------------------------
# Test 2: Long day regime_t == 1
# ---------------------------------------------------------------------------


def test_long_day_regime_is_1() -> None:
    """A date in the first half of the mask returns regime_t == 1."""
    dates = pd.bdate_range("2022-01-03", periods=60)
    ctx = _make_ctx(dates)

    long_date = dates[10]  # well within first (Long) half
    inputs = _extract_day_inputs(long_date, ctx)

    assert inputs.regime_t == 1


# ---------------------------------------------------------------------------
# Test 3: Defensive day regime_t == 0
# ---------------------------------------------------------------------------


def test_defensive_day_regime_is_0() -> None:
    """A date in the second half of the mask returns regime_t == 0."""
    dates = pd.bdate_range("2022-01-03", periods=60)
    ctx = _make_ctx(dates)

    defensive_date = dates[50]  # well within second (Defensive) half
    inputs = _extract_day_inputs(defensive_date, ctx)

    assert inputs.regime_t == 0


# ---------------------------------------------------------------------------
# Test 4: mtm_iv_value is NaN during 29-day warmup
# ---------------------------------------------------------------------------


def test_mtm_iv_nan_during_warmup() -> None:
    """mtm_iv_value is NaN for dates in the rolling-mean warmup window."""
    dates = pd.bdate_range("2022-01-03", periods=100)

    # Build a ctx where mtm_iv_series uses rolling(30).mean() WITHOUT ffill,
    # so the first 29 dates are genuine NaN.
    rng = np.random.default_rng(42)
    returns = pd.DataFrame({"VTI": rng.normal(0.001, 0.01, len(dates))}, index=dates)
    rfr = pd.Series(0.04, index=dates)
    vix = pd.Series(0.20 + rng.normal(0, 0.02, len(dates)), index=dates)
    mtm_iv_no_fill = vix.rolling(30).mean()  # NaN for first 29 rows

    config = PortfolioConfig(
        target_weights={"VTI": 1.0},
        initial_nav=10000.0,
        monthly_contribution=500.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )
    return_data = ReturnData(
        returns=returns,
        log_returns=pd.DataFrame({"VTI": np.log1p(returns["VTI"].values)}, index=dates),
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr,
    )
    month_ends = frozenset(
        pd.Timestamp(g.index[-1])
        for _, g in returns.groupby(pd.DatetimeIndex(dates).to_period("M"))
    )
    ctx = BacktestContext(
        base_assets=("VTI",),
        leaps_keys=(),
        leaps_fraction=0.0,
        base_target_w=pd.Series({"VTI": 1.0}),
        governed_base=("VTI",),
        gtt_active=False,
        defensive_weights={"R_f": 1.0},
        use_leaps=False,
        iv=0.20,
        leaps_monthly=0.0,
        base_contribution=500.0,
        config=config,
        return_data=return_data,
        underlying_prices=None,
        raw_vix=vix,
        mtm_iv_series=mtm_iv_no_fill,
        rfr_series=rfr,
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=month_ends,
        long_window_end={},
        w=pd.Series({"VTI": 1.0}),
    )

    warmup_date = dates[5]  # day 5 — definitely within warmup (rolling(30) needs 30 points)
    inputs = _extract_day_inputs(warmup_date, ctx)

    assert inputs.mtm_iv_value is not None
    assert math.isnan(inputs.mtm_iv_value)


# ---------------------------------------------------------------------------
# Test 5: raw_vix=None → raw_vix_value is None
# ---------------------------------------------------------------------------


def test_raw_vix_none_returns_none() -> None:
    """When ctx.raw_vix is None, raw_vix_value is None."""
    dates = pd.bdate_range("2022-01-03", periods=30)
    ctx = _make_ctx(dates)

    # Rebuild ctx with raw_vix=None (and mtm_iv_series=None for consistency)
    ctx_no_vix = BacktestContext(
        base_assets=ctx.base_assets,
        leaps_keys=ctx.leaps_keys,
        leaps_fraction=ctx.leaps_fraction,
        base_target_w=ctx.base_target_w,
        governed_base=ctx.governed_base,
        gtt_active=ctx.gtt_active,
        defensive_weights=ctx.defensive_weights,
        use_leaps=ctx.use_leaps,
        iv=ctx.iv,
        leaps_monthly=ctx.leaps_monthly,
        base_contribution=ctx.base_contribution,
        config=ctx.config,
        return_data=ctx.return_data,
        underlying_prices=None,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=ctx.rfr_series,
        mask_aligned=ctx.mask_aligned,
        def_gross=ctx.def_gross,
        rebal_dates=ctx.rebal_dates,
        month_end_dates=ctx.month_end_dates,
        long_window_end=ctx.long_window_end,
        w=ctx.w,
    )

    inputs = _extract_day_inputs(dates[10], ctx_no_vix)

    assert inputs.raw_vix_value is None
    assert inputs.mtm_iv_value is None


# ---------------------------------------------------------------------------
# Test 6: is_month_end True and False
# ---------------------------------------------------------------------------


def test_is_month_end_true_and_false() -> None:
    """is_month_end is True for a date in month_end_dates and False otherwise."""
    dates = pd.bdate_range("2022-01-03", periods=60)
    ctx = _make_ctx(dates)

    # Find a month-end date and a non-month-end date
    month_end_date = next(iter(ctx.month_end_dates))
    non_month_end_date = next(d for d in dates if d not in ctx.month_end_dates)

    end_inputs = _extract_day_inputs(month_end_date, ctx)
    non_end_inputs = _extract_day_inputs(non_month_end_date, ctx)

    assert end_inputs.is_month_end is True
    assert non_end_inputs.is_month_end is False


# ---------------------------------------------------------------------------
# Test 7: is_rebal_date True
# ---------------------------------------------------------------------------


def test_is_rebal_date_true() -> None:
    """is_rebal_date is True when date is in ctx.rebal_dates."""
    dates = pd.bdate_range("2022-01-03", periods=100)
    ctx = _make_ctx(dates)

    # Inject a known rebal_date into the ctx
    rebal_date = dates[40]
    ctx_with_rebal = BacktestContext(
        base_assets=ctx.base_assets,
        leaps_keys=ctx.leaps_keys,
        leaps_fraction=ctx.leaps_fraction,
        base_target_w=ctx.base_target_w,
        governed_base=ctx.governed_base,
        gtt_active=ctx.gtt_active,
        defensive_weights=ctx.defensive_weights,
        use_leaps=ctx.use_leaps,
        iv=ctx.iv,
        leaps_monthly=ctx.leaps_monthly,
        base_contribution=ctx.base_contribution,
        config=ctx.config,
        return_data=ctx.return_data,
        underlying_prices=ctx.underlying_prices,
        raw_vix=ctx.raw_vix,
        mtm_iv_series=ctx.mtm_iv_series,
        rfr_series=ctx.rfr_series,
        mask_aligned=ctx.mask_aligned,
        def_gross=ctx.def_gross,
        rebal_dates=frozenset({rebal_date}),
        month_end_dates=ctx.month_end_dates,
        long_window_end=ctx.long_window_end,
        w=ctx.w,
    )

    inputs = _extract_day_inputs(rebal_date, ctx_with_rebal)
    non_rebal = _extract_day_inputs(dates[41], ctx_with_rebal)

    assert inputs.is_rebal_date is True
    assert non_rebal.is_rebal_date is False
