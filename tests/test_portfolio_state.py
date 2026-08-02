"""Tests for BacktestContext frozen dataclass (F-003).

Verifies construction, frozenset membership types, frozen enforcement, and
edge cases: empty governed_base and empty long_window_end.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from finance.leverage import LeapsConfig, RebalanceRule, WeightStrategy
from finance.portfolio import BacktestContext, PortfolioConfig
from finance.returns import ReturnData


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_return_data(dates: pd.DatetimeIndex) -> ReturnData:
    """Build a minimal ReturnData over the given dates.

    Arguments:
        dates: DatetimeIndex of trading days.

    Returns:
        ReturnData with one VTI column and a flat 4% risk-free rate.
    """
    rng = np.random.default_rng(0)
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


def _make_config() -> PortfolioConfig:
    """Build a minimal PortfolioConfig with one base asset.

    Returns:
        PortfolioConfig with VTI=1.0, quarterly rebalancing.
    """
    return PortfolioConfig(
        target_weights={"VTI": 1.0},
        initial_nav=10_000.0,
        monthly_contribution=500.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )


def _make_context(
    dates: pd.DatetimeIndex,
    *,
    governed_base: tuple[str, ...] = ("VTI",),
    long_window_end: dict[pd.Timestamp, pd.Timestamp] | None = None,
) -> BacktestContext:
    """Construct a fully-populated BacktestContext for the given dates.

    Arguments:
        dates: DatetimeIndex of trading days.
        governed_base: Tuple of base assets governed by GTT.
        long_window_end: Mapping from Long-window start to end date; defaults
            to a single window spanning the full date range.

    Returns:
        BacktestContext with all 22 fields populated.
    """
    if long_window_end is None:
        long_window_end = {
            pd.Timestamp(dates[0]): pd.Timestamp(dates[-1]),
        }
    config = _make_config()
    return_data = _make_return_data(dates)
    base_target_w = pd.Series({"VTI": 1.0})
    w = pd.Series({"VTI": 1.0})
    rebal_dates: frozenset[pd.Timestamp] = frozenset(
        {pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30")}
    )
    month_end_dates: frozenset[pd.Timestamp] = frozenset(
        {pd.Timestamp(d) for d in dates[::20]}
    )
    return BacktestContext(
        base_assets=("VTI",),
        leaps_keys=(),
        leaps_fraction=0.0,
        base_target_w=base_target_w,
        governed_base=governed_base,
        gtt_active=len(governed_base) > 0,
        defensive_weights={"R_f": 1.0},
        use_leaps=False,
        iv=0.20,
        leaps_monthly=0.0,
        base_contribution=500.0,
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
        long_window_end=long_window_end,
        w=w,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBacktestContextConstruction:
    """BacktestContext constructs correctly and fields have expected types."""

    def test_all_22_fields_accessible(self) -> None:
        """All 22 fields are accessible on a constructed instance."""
        dates = pd.bdate_range("2020-01-02", periods=60)
        ctx = _make_context(dates)

        assert ctx.base_assets == ("VTI",)
        assert ctx.leaps_keys == ()
        assert ctx.leaps_fraction == 0.0
        assert isinstance(ctx.base_target_w, pd.Series)
        assert ctx.governed_base == ("VTI",)
        assert ctx.gtt_active is True
        assert ctx.defensive_weights == {"R_f": 1.0}
        assert ctx.use_leaps is False
        assert ctx.iv == 0.20
        assert ctx.leaps_monthly == 0.0
        assert ctx.base_contribution == 500.0
        assert isinstance(ctx.config, PortfolioConfig)
        assert isinstance(ctx.return_data, ReturnData)
        assert ctx.underlying_prices is None
        assert ctx.raw_vix is None
        assert ctx.mtm_iv_series is None
        assert ctx.rfr_series is None
        assert ctx.mask_aligned is None
        assert ctx.def_gross is None
        assert isinstance(ctx.rebal_dates, frozenset)
        assert isinstance(ctx.month_end_dates, frozenset)
        assert isinstance(ctx.long_window_end, dict)
        assert isinstance(ctx.w, pd.Series)

    def test_rebal_dates_is_frozenset(self) -> None:
        """rebal_dates is a frozenset instance (O(1) membership)."""
        dates = pd.bdate_range("2020-01-02", periods=60)
        ctx = _make_context(dates)
        assert isinstance(ctx.rebal_dates, frozenset)

    def test_month_end_dates_is_frozenset(self) -> None:
        """month_end_dates is a frozenset instance (O(1) membership)."""
        dates = pd.bdate_range("2020-01-02", periods=60)
        ctx = _make_context(dates)
        assert isinstance(ctx.month_end_dates, frozenset)

    def test_rebal_dates_membership(self) -> None:
        """Known dates are found in rebal_dates via the 'in' operator."""
        dates = pd.bdate_range("2020-01-02", periods=60)
        ctx = _make_context(dates)
        assert pd.Timestamp("2020-03-31") in ctx.rebal_dates

    def test_month_end_dates_membership(self) -> None:
        """At least one trading date falls in month_end_dates."""
        dates = pd.bdate_range("2020-01-02", periods=60)
        ctx = _make_context(dates)
        assert any(d in ctx.month_end_dates for d in dates)


class TestBacktestContextFrozen:
    """BacktestContext is frozen — field reassignment raises FrozenInstanceError."""

    def test_cannot_reassign_scalar_field(self) -> None:
        """Assigning to a scalar field raises FrozenInstanceError."""
        dates = pd.bdate_range("2020-01-02", periods=60)
        ctx = _make_context(dates)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.iv = 0.30  # type: ignore[misc]

    def test_cannot_reassign_tuple_field(self) -> None:
        """Assigning to a tuple field raises FrozenInstanceError."""
        dates = pd.bdate_range("2020-01-02", periods=60)
        ctx = _make_context(dates)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.base_assets = ("VXUS",)  # type: ignore[misc]

    def test_cannot_reassign_frozenset_field(self) -> None:
        """Assigning to rebal_dates raises FrozenInstanceError."""
        dates = pd.bdate_range("2020-01-02", periods=60)
        ctx = _make_context(dates)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.rebal_dates = frozenset()  # type: ignore[misc]

    def test_cannot_reassign_bool_field(self) -> None:
        """Assigning to gtt_active raises FrozenInstanceError."""
        dates = pd.bdate_range("2020-01-02", periods=60)
        ctx = _make_context(dates)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.gtt_active = False  # type: ignore[misc]


class TestBacktestContextEdgeCases:
    """Edge cases: empty governed_base and empty long_window_end."""

    def test_empty_governed_base(self) -> None:
        """governed_base=() is valid — GTT overlay is inactive."""
        dates = pd.bdate_range("2020-01-02", periods=60)
        ctx = _make_context(dates, governed_base=())
        assert ctx.governed_base == ()
        assert ctx.gtt_active is False

    def test_empty_long_window_end(self) -> None:
        """long_window_end={} is valid — all-Long signal with no window transitions."""
        dates = pd.bdate_range("2020-01-02", periods=60)
        ctx = _make_context(dates, long_window_end={})
        assert ctx.long_window_end == {}
        assert isinstance(ctx.long_window_end, dict)

    def test_leaps_config_fields_with_leaps_config(self) -> None:
        """BacktestContext with a LEAPS-configured PortfolioConfig is constructable."""
        dates = pd.bdate_range("2020-01-02", periods=60)
        leaps_config = LeapsConfig(iv=0.22, ltcg_rate=0.20)
        config = PortfolioConfig(
            target_weights={"VTI": 0.85, "VTI_LEAPS": 0.15},
            initial_nav=10_000.0,
            monthly_contribution=500.0,
            rebalance_rule=RebalanceRule.QUARTERLY,
            weight_strategy=WeightStrategy.USER_SPECIFIED,
            leaps_config=leaps_config,
        )
        return_data = _make_return_data(dates)
        base_target_w = pd.Series({"VTI": 1.0})
        w = pd.Series({"VTI": 0.85, "VTI_LEAPS": 0.15})
        ctx = BacktestContext(
            base_assets=("VTI",),
            leaps_keys=("VTI_LEAPS",),
            leaps_fraction=0.15,
            base_target_w=base_target_w,
            governed_base=("VTI",),
            gtt_active=False,
            defensive_weights={},
            use_leaps=True,
            iv=0.22,
            leaps_monthly=500.0 * 0.15,
            base_contribution=500.0 * 0.85,
            config=config,
            return_data=return_data,
            underlying_prices=None,
            raw_vix=None,
            mtm_iv_series=None,
            rfr_series=None,
            mask_aligned=None,
            def_gross=None,
            rebal_dates=frozenset(),
            month_end_dates=frozenset(),
            long_window_end={},
            w=w,
        )
        assert ctx.use_leaps is True
        assert ctx.leaps_fraction == 0.15
        assert ctx.leaps_keys == ("VTI_LEAPS",)
