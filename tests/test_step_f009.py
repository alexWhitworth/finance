"""Tests for F-009: _apply_returns and _apply_defensive_compounding.

Verifies the exact compounding invariants for both step functions including
flat-return no-ops, known-return scaling, and defensive sleeve/pool compounding.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from finance._step_f009 import (
    BacktestContext,
    DayInputs,
    PortfolioState,
    _apply_defensive_compounding,
    _apply_returns,
)
from finance.leverage import (
    AccountType,
    LeapsConfig,
    RebalanceRule,
    WeightStrategy,
)
from finance.portfolio import GttConfig, PortfolioConfig
from finance.returns import ReturnData

# ---------------------------------------------------------------------------
# Minimal builder helpers
# ---------------------------------------------------------------------------

_DATE = pd.Timestamp("2020-01-02")
_TICKERS = ("VTI", "VXUS")


def _minimal_return_data() -> ReturnData:
    """Single-row ReturnData with zero returns for VTI and VXUS."""
    idx = pd.DatetimeIndex([_DATE])
    returns = pd.DataFrame({"VTI": [0.0], "VXUS": [0.0]}, index=idx)
    rfr = pd.Series([0.04], index=idx)
    return ReturnData(
        returns=returns,
        log_returns=returns,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr,
    )


def _minimal_config(weights: dict[str, float] | None = None) -> PortfolioConfig:
    """Minimal PortfolioConfig for VTI + VXUS with equal weights."""
    w = weights or {"VTI": 0.5, "VXUS": 0.5}
    return PortfolioConfig(
        target_weights=w,
        initial_nav=10_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=None,
        gtt_config=None,
    )


def minimal_state(
    holdings: dict[str, float],
    sleeve: float = 0.0,
    pool: float = 0.0,
    prev_regime: int = 1,
) -> PortfolioState:
    """Build a minimal PortfolioState for testing.

    Arguments:
        holdings: Dollar value per base asset.
        sleeve: defensive_sleeve value (default 0.0).
        pool: leaps_pool value (default 0.0).
        prev_regime: 1=Long (default), 0=Defensive.

    Returns:
        PortfolioState with sensible defaults for all non-varied fields.
    """
    return PortfolioState(
        holdings=holdings,
        defensive_sleeve=sleeve,
        leaps_pool=pool,
        leaps_value=0.0,
        prev_total_nav=sum(holdings.values()) + sleeve + pool,
        prev_regime=prev_regime,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )


def minimal_inputs(
    regime_t: int = 1,
    def_gross_return: float = 0.0,
    day_ret: pd.Series | None = None,
    date_ts: pd.Timestamp | None = None,
) -> DayInputs:
    """Build a minimal DayInputs for testing.

    Arguments:
        regime_t: GTT regime today; 1=Long (default), 0=Defensive.
        def_gross_return: Blended defensive return for today (default 0.0).
        day_ret: Asset return Series (default: VTI=0.0, VXUS=0.0).
        date_ts: Trading date (default: 2020-01-02).

    Returns:
        DayInputs with sensible defaults for all non-varied fields.
    """
    if day_ret is None:
        day_ret = pd.Series({"VTI": 0.0, "VXUS": 0.0})
    return DayInputs(
        date_ts=date_ts or _DATE,
        day_ret=day_ret,
        regime_t=regime_t,
        def_gross_return=def_gross_return,
        spot=None,
        raw_vix_value=None,
        mtm_iv_value=None,
        rfr=0.04,
        is_month_end=False,
        is_rebal_date=False,
    )


def minimal_ctx(
    gtt_active: bool = False,
    def_gross: pd.Series | None = None,
) -> BacktestContext:
    """Build a minimal BacktestContext for testing.

    Arguments:
        gtt_active: Whether GTT overlay is active (default False).
        def_gross: Precomputed defensive gross return series (default None).

    Returns:
        BacktestContext with VTI/VXUS base assets and sensible defaults.
    """
    rd = _minimal_return_data()
    cfg = _minimal_config()
    idx = pd.DatetimeIndex([_DATE])
    return BacktestContext(
        base_assets=("VTI", "VXUS"),
        leaps_keys=(),
        leaps_fraction=0.0,
        base_target_w=pd.Series({"VTI": 0.5, "VXUS": 0.5}),
        governed_base=("VTI",) if gtt_active else (),
        gtt_active=gtt_active,
        defensive_weights={"VTI": 1.0} if gtt_active else {},
        use_leaps=False,
        iv=0.18,
        leaps_monthly=0.0,
        base_contribution=0.0,
        config=cfg,
        return_data=rd,
        underlying_prices=None,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=pd.Series([0.04], index=idx),
        mask_aligned=None,
        def_gross=def_gross,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end={},
        w=pd.Series({"VTI": 0.5, "VXUS": 0.5}),
    )


# ---------------------------------------------------------------------------
# _apply_returns tests
# ---------------------------------------------------------------------------


class TestApplyReturns:
    """Unit tests for _apply_returns."""

    def test_flat_returns_holdings_unchanged(self) -> None:
        """Flat returns (day_ret=0.0) leave holdings numerically identical."""
        holdings = {"VTI": 5000.0, "VXUS": 3000.0}
        state = minimal_state(holdings)
        inputs = minimal_inputs(day_ret=pd.Series({"VTI": 0.0, "VXUS": 0.0}))
        ctx = minimal_ctx()

        new = _apply_returns(state, inputs, ctx)

        assert abs(new.holdings["VTI"] - 5000.0) < 1e-14
        assert abs(new.holdings["VXUS"] - 3000.0) < 1e-14

    def test_known_returns_scale_holdings(self) -> None:
        """Known returns scale holdings by (1 + r) for each asset."""
        holdings = {"VTI": 10_000.0, "VXUS": 5_000.0}
        state = minimal_state(holdings)
        inputs = minimal_inputs(day_ret=pd.Series({"VTI": 0.05, "VXUS": -0.02}))
        ctx = minimal_ctx()

        new = _apply_returns(state, inputs, ctx)

        assert new.holdings["VTI"] == pytest.approx(10_500.0)
        assert new.holdings["VXUS"] == pytest.approx(4_900.0)

    @given(
        ret_vti=st.floats(-0.5, 0.5),
        ret_vxus=st.floats(-0.5, 0.5),
        h_vti=st.floats(100.0, 10_000.0),
        h_vxus=st.floats(100.0, 10_000.0),
    )
    @settings(max_examples=500)
    def test_exact_formula_property(
        self,
        ret_vti: float,
        ret_vxus: float,
        h_vti: float,
        h_vxus: float,
    ) -> None:
        """Property: holdings_out[a] == holdings_in[a] * (1 + day_ret[a])."""
        state = minimal_state({"VTI": h_vti, "VXUS": h_vxus})
        inputs = minimal_inputs(day_ret=pd.Series({"VTI": ret_vti, "VXUS": ret_vxus}))
        ctx = minimal_ctx()

        new = _apply_returns(state, inputs, ctx)

        assert abs(new.holdings["VTI"] - h_vti * (1.0 + ret_vti)) < 1e-10
        assert abs(new.holdings["VXUS"] - h_vxus * (1.0 + ret_vxus)) < 1e-10

    def test_non_holdings_fields_unchanged(self) -> None:
        """Fields other than holdings are not mutated by _apply_returns."""
        state = minimal_state({"VTI": 5000.0, "VXUS": 3000.0}, sleeve=200.0, pool=100.0)
        inputs = minimal_inputs(day_ret=pd.Series({"VTI": 0.01, "VXUS": 0.02}))
        ctx = minimal_ctx()

        new = _apply_returns(state, inputs, ctx)

        assert new.defensive_sleeve == state.defensive_sleeve
        assert new.leaps_pool == state.leaps_pool
        assert new.leaps_value == state.leaps_value
        assert new.prev_total_nav == state.prev_total_nav
        assert new.prev_regime == state.prev_regime


# ---------------------------------------------------------------------------
# _apply_defensive_compounding tests
# ---------------------------------------------------------------------------


class TestApplyDefensiveCompounding:
    """Unit tests for _apply_defensive_compounding."""

    def _def_gross_series(self, value: float = 0.001) -> pd.Series:
        """Single-date defensive gross return Series."""
        return pd.Series([value], index=pd.DatetimeIndex([_DATE]))

    def test_noop_when_gtt_inactive(self) -> None:
        """No compounding when gtt_active=False; state returned unchanged."""
        state = minimal_state({"VTI": 0.0, "VXUS": 0.0}, sleeve=10_000.0, pool=2_000.0)
        inputs = minimal_inputs(regime_t=0, def_gross_return=0.001)
        ctx = minimal_ctx(gtt_active=False, def_gross=self._def_gross_series())

        new = _apply_defensive_compounding(state, inputs, ctx)

        assert new is state

    def test_noop_on_long_day(self) -> None:
        """No compounding on a pure Long day (prev_regime=1, regime_t=1)."""
        state = minimal_state({"VTI": 0.0, "VXUS": 0.0}, sleeve=10_000.0, pool=2_000.0, prev_regime=1)
        inputs = minimal_inputs(regime_t=1, def_gross_return=0.001)
        ctx = minimal_ctx(gtt_active=True, def_gross=self._def_gross_series())

        new = _apply_defensive_compounding(state, inputs, ctx)

        assert new.defensive_sleeve == state.defensive_sleeve
        assert new.leaps_pool == state.leaps_pool

    def test_defensive_day_compounds_sleeve_and_pool(self) -> None:
        """Defensive day: sleeve and pool both compounded by (1 + def_gross_return)."""
        state = minimal_state({"VTI": 0.0, "VXUS": 0.0}, sleeve=10_000.0, pool=2_000.0)
        inputs = minimal_inputs(regime_t=0, def_gross_return=0.001)
        ctx = minimal_ctx(gtt_active=True, def_gross=self._def_gross_series(0.001))

        new = _apply_defensive_compounding(state, inputs, ctx)

        assert new.defensive_sleeve == pytest.approx(10_010.0)
        assert new.leaps_pool == pytest.approx(2_002.0)

    def test_reentry_day_earns_one_final_defensive_return(self) -> None:
        """Re-entry day (prev_regime=0, regime_t=1) compounds once before redeployment."""
        state = minimal_state(
            {"VTI": 0.0, "VXUS": 0.0}, sleeve=8_000.0, pool=1_500.0, prev_regime=0
        )
        inputs = minimal_inputs(regime_t=1, def_gross_return=0.002)
        ctx = minimal_ctx(gtt_active=True, def_gross=self._def_gross_series(0.002))

        new = _apply_defensive_compounding(state, inputs, ctx)

        assert new.defensive_sleeve == pytest.approx(8_016.0)
        assert new.leaps_pool == pytest.approx(1_503.0)

    def test_zero_def_gross_return_no_change(self) -> None:
        """Zero def_gross_return leaves sleeve and pool exactly unchanged."""
        state = minimal_state({"VTI": 0.0, "VXUS": 0.0}, sleeve=10_000.0, pool=2_000.0)
        inputs = minimal_inputs(regime_t=0, def_gross_return=0.0)
        ctx = minimal_ctx(gtt_active=True, def_gross=self._def_gross_series(0.0))

        new = _apply_defensive_compounding(state, inputs, ctx)

        assert new.defensive_sleeve == state.defensive_sleeve
        assert new.leaps_pool == state.leaps_pool

    def test_noop_when_def_gross_is_none(self) -> None:
        """No compounding when def_gross series is None (GTT active but no series)."""
        state = minimal_state({"VTI": 0.0, "VXUS": 0.0}, sleeve=10_000.0, pool=2_000.0)
        inputs = minimal_inputs(regime_t=0, def_gross_return=0.001)
        ctx = minimal_ctx(gtt_active=True, def_gross=None)

        new = _apply_defensive_compounding(state, inputs, ctx)

        assert new is state

    def test_non_sleeve_fields_unchanged(self) -> None:
        """Fields other than sleeve and pool are not mutated."""
        holdings = {"VTI": 500.0, "VXUS": 300.0}
        state = minimal_state(holdings, sleeve=1_000.0, pool=200.0)
        inputs = minimal_inputs(regime_t=0, def_gross_return=0.005)
        ctx = minimal_ctx(gtt_active=True, def_gross=self._def_gross_series(0.005))

        new = _apply_defensive_compounding(state, inputs, ctx)

        assert new.holdings == state.holdings
        assert new.leaps_value == state.leaps_value
        assert new.prev_total_nav == state.prev_total_nav
        assert new.prev_regime == state.prev_regime
