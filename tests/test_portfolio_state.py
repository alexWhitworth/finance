"""Tests for PortfolioState (F-001), DayInputs (F-002), and BacktestContext (F-003) frozen dataclasses."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from finance.leverage import AccountType, LeapsConfig, LeapsContract, LeapsGttCloseEvent, LeapsLedger, RebalanceRule, WeightStrategy
from finance.portfolio import BacktestContext, DayInputs, PortfolioConfig, PortfolioState
from finance.returns import ReturnData


# ---------------------------------------------------------------------------
# PortfolioState (F-001)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_contract() -> LeapsContract:
    """Minimal LeapsContract for use in PortfolioState fixtures."""
    return LeapsContract(
        purchase_date=pd.Timestamp("2023-01-03"),
        expiry_date=pd.Timestamp("2025-01-17"),
        strike=160.0,
        spot_at_purchase=200.0,
        premium_paid=45.0,
        notional=20000.0,
        n_contracts=1.0,
        account_type=AccountType.TAXABLE,
    )


@pytest.fixture
def sample_ledger(sample_contract: LeapsContract) -> LeapsLedger:
    """Minimal LeapsLedger with one contract."""
    return LeapsLedger(
        contracts=(sample_contract,),
        roll_events=(),
        account_type=AccountType.TAXABLE,
    )


@pytest.fixture
def sample_gtt_close(sample_contract: LeapsContract) -> LeapsGttCloseEvent:
    """Minimal LeapsGttCloseEvent for use in PortfolioState fixtures."""
    return LeapsGttCloseEvent(
        close_date=pd.Timestamp("2023-06-01"),
        contract=sample_contract,
        mtm_value=5000.0,
        gain_realized=500.0,
        tax_paid=100.0,
        net_proceeds=4900.0,
    )


@pytest.fixture
def full_state(
    sample_contract: LeapsContract,
    sample_ledger: LeapsLedger,
    sample_gtt_close: LeapsGttCloseEvent,
) -> PortfolioState:
    """PortfolioState with non-trivial values for all 11 fields."""
    return PortfolioState(
        holdings={"VTI": 50000.0, "VXUS": 20000.0},
        defensive_sleeve=10000.0,
        leaps_pool=5000.0,
        leaps_value=7500.0,
        prev_total_nav=92500.0,
        prev_regime=1,
        prev_date_ts=pd.Timestamp("2023-05-31"),
        leaps_ledger=sample_ledger,
        leaps_scale={sample_contract: 0.8},
        all_window_ledgers=(sample_ledger,),
        all_gtt_closes=(sample_gtt_close,),
    )


def test_all_fields_accessible(
    full_state: PortfolioState,
    sample_contract: LeapsContract,
    sample_ledger: LeapsLedger,
    sample_gtt_close: LeapsGttCloseEvent,
) -> None:
    """All 11 fields are accessible and equal injected values."""
    assert full_state.holdings == {"VTI": 50000.0, "VXUS": 20000.0}
    assert full_state.defensive_sleeve == 10000.0
    assert full_state.leaps_pool == 5000.0
    assert full_state.leaps_value == 7500.0
    assert full_state.prev_total_nav == 92500.0
    assert full_state.prev_regime == 1
    assert full_state.prev_date_ts == pd.Timestamp("2023-05-31")
    assert full_state.leaps_ledger is sample_ledger
    assert full_state.leaps_scale == {sample_contract: 0.8}
    assert full_state.all_window_ledgers == (sample_ledger,)
    assert full_state.all_gtt_closes == (sample_gtt_close,)


def test_none_optional_fields() -> None:
    """prev_date_ts and leaps_ledger accept None."""
    state = PortfolioState(
        holdings={"VTI": 100000.0},
        defensive_sleeve=0.0,
        leaps_pool=0.0,
        leaps_value=0.0,
        prev_total_nav=100000.0,
        prev_regime=1,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )
    assert state.prev_date_ts is None
    assert state.leaps_ledger is None


def test_frozen_assignment_raises(full_state: PortfolioState) -> None:
    """Assigning to any field raises FrozenInstanceError."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        full_state.holdings = {}  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        full_state.prev_regime = 0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        full_state.prev_total_nav = 0.0  # type: ignore[misc]


def test_empty_holdings() -> None:
    """holdings={} is a valid edge case (no base assets allocated)."""
    state = PortfolioState(
        holdings={},
        defensive_sleeve=0.0,
        leaps_pool=0.0,
        leaps_value=0.0,
        prev_total_nav=0.0,
        prev_regime=1,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )
    assert state.holdings == {}


def test_empty_accumulators() -> None:
    """all_window_ledgers=() and all_gtt_closes=() are valid (no GTT activity)."""
    state = PortfolioState(
        holdings={"VTI": 100000.0},
        defensive_sleeve=0.0,
        leaps_pool=0.0,
        leaps_value=0.0,
        prev_total_nav=100000.0,
        prev_regime=1,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )
    assert state.all_window_ledgers == ()
    assert state.all_gtt_closes == ()


# ---------------------------------------------------------------------------
# DayInputs (F-002)
# ---------------------------------------------------------------------------


@pytest.fixture
def day_inputs_full() -> DayInputs:
    """DayInputs with all optional fields populated."""
    return DayInputs(
        date_ts=pd.Timestamp("2023-03-31"),
        day_ret=pd.Series({"VTI": 0.01, "VXUS": -0.005}),
        regime_t=1,
        def_gross_return=0.002,
        spot=205.50,
        raw_vix_value=0.185,
        mtm_iv_value=0.192,
        rfr=0.05,
        is_month_end=True,
        is_rebal_date=True,
    )


def test_day_inputs_fields(day_inputs_full: DayInputs) -> None:
    """All 10 fields are accessible and equal their injected values."""
    d = day_inputs_full
    assert d.date_ts == pd.Timestamp("2023-03-31")
    assert float(d.day_ret["VTI"]) == pytest.approx(0.01)
    assert float(d.day_ret["VXUS"]) == pytest.approx(-0.005)
    assert d.regime_t == 1
    assert d.def_gross_return == pytest.approx(0.002)
    assert d.spot == pytest.approx(205.50)
    assert d.raw_vix_value == pytest.approx(0.185)
    assert d.mtm_iv_value == pytest.approx(0.192)
    assert d.rfr == pytest.approx(0.05)
    assert d.is_month_end is True
    assert d.is_rebal_date is True


def test_day_inputs_frozen(day_inputs_full: DayInputs) -> None:
    """Assignment to any field raises FrozenInstanceError."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        day_inputs_full.regime_t = 0  # type: ignore[misc]


def test_day_inputs_none_optional_fields() -> None:
    """Optional fields accept None (no LEAPS, no vol_prices, warmup period)."""
    d = DayInputs(
        date_ts=pd.Timestamp("2023-01-03"),
        day_ret=pd.Series({"VTI": 0.0}),
        regime_t=1,
        def_gross_return=0.0,
        spot=None,
        raw_vix_value=None,
        mtm_iv_value=None,
        rfr=0.04,
        is_month_end=False,
        is_rebal_date=False,
    )
    assert d.spot is None
    assert d.raw_vix_value is None
    assert d.mtm_iv_value is None
    assert d.is_month_end is False
    assert d.is_rebal_date is False


# ---------------------------------------------------------------------------
# BacktestContext (F-003)
# ---------------------------------------------------------------------------


def _make_return_data(dates: pd.DatetimeIndex) -> ReturnData:
    """Build a minimal ReturnData over the given dates."""
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
    """Build a minimal PortfolioConfig with one base asset."""
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
    """Construct a fully-populated BacktestContext for the given dates."""
    if long_window_end is None:
        long_window_end = {pd.Timestamp(dates[0]): pd.Timestamp(dates[-1])}
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
        ctx = _make_context(pd.bdate_range("2020-01-02", periods=60))
        assert isinstance(ctx.rebal_dates, frozenset)

    def test_month_end_dates_is_frozenset(self) -> None:
        """month_end_dates is a frozenset instance (O(1) membership)."""
        ctx = _make_context(pd.bdate_range("2020-01-02", periods=60))
        assert isinstance(ctx.month_end_dates, frozenset)

    def test_rebal_dates_membership(self) -> None:
        """Known dates are found in rebal_dates via 'in' operator."""
        ctx = _make_context(pd.bdate_range("2020-01-02", periods=60))
        assert pd.Timestamp("2020-03-31") in ctx.rebal_dates

    def test_month_end_dates_membership(self) -> None:
        """At least one trading date falls in month_end_dates."""
        dates = pd.bdate_range("2020-01-02", periods=60)
        ctx = _make_context(dates)
        assert any(d in ctx.month_end_dates for d in dates)


class TestBacktestContextFrozen:
    """BacktestContext is frozen — field reassignment raises FrozenInstanceError."""

    def test_cannot_reassign_scalar_field(self) -> None:
        ctx = _make_context(pd.bdate_range("2020-01-02", periods=60))
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.iv = 0.30  # type: ignore[misc]

    def test_cannot_reassign_tuple_field(self) -> None:
        ctx = _make_context(pd.bdate_range("2020-01-02", periods=60))
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.base_assets = ("VXUS",)  # type: ignore[misc]

    def test_cannot_reassign_frozenset_field(self) -> None:
        ctx = _make_context(pd.bdate_range("2020-01-02", periods=60))
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.rebal_dates = frozenset()  # type: ignore[misc]

    def test_cannot_reassign_bool_field(self) -> None:
        ctx = _make_context(pd.bdate_range("2020-01-02", periods=60))
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.gtt_active = False  # type: ignore[misc]


class TestBacktestContextEdgeCases:
    """Edge cases: empty governed_base and empty long_window_end."""

    def test_empty_governed_base(self) -> None:
        """governed_base=() is valid — GTT overlay is inactive."""
        ctx = _make_context(pd.bdate_range("2020-01-02", periods=60), governed_base=())
        assert ctx.governed_base == ()
        assert ctx.gtt_active is False

    def test_empty_long_window_end(self) -> None:
        """long_window_end={} is valid — all-Long signal with no window transitions."""
        ctx = _make_context(pd.bdate_range("2020-01-02", periods=60), long_window_end={})
        assert ctx.long_window_end == {}

    def test_leaps_config_fields(self) -> None:
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
        ctx = BacktestContext(
            base_assets=("VTI",),
            leaps_keys=("VTI_LEAPS",),
            leaps_fraction=0.15,
            base_target_w=pd.Series({"VTI": 1.0}),
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
            w=pd.Series({"VTI": 0.85, "VTI_LEAPS": 0.15}),
        )
        assert ctx.use_leaps is True
        assert ctx.leaps_fraction == 0.15
        assert ctx.leaps_keys == ("VTI_LEAPS",)
