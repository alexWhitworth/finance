"""Tests for F-015: _compute_total_nav, _advance_state, _build_weight_row, _assemble_leaps_ledger."""

import numpy as np
import pandas as pd
import pytest

from finance._step_f015 import (
    _advance_state,
    _assemble_leaps_ledger,
    _build_weight_row,
    _compute_total_nav,
)
from finance.leverage import (
    AccountType,
    LeapsConfig,
    LeapsContract,
    LeapsGttCloseEvent,
    LeapsLedger,
    RebalanceRule,
    WeightStrategy,
)
from finance.portfolio import BacktestContext, DayInputs, PortfolioConfig, PortfolioState
from finance.returns import ReturnData


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_contract() -> LeapsContract:
    """Minimal LeapsContract for use in fixtures."""
    return LeapsContract(
        purchase_date=pd.Timestamp("2023-01-03"),
        expiry_date=pd.Timestamp("2025-01-17"),
        strike=160.0,
        spot_at_purchase=200.0,
        premium_paid=45.0,
        notional=20000.0,
        n_contracts=2.0,
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
    """Minimal LeapsGttCloseEvent."""
    return LeapsGttCloseEvent(
        close_date=pd.Timestamp("2023-06-01"),
        contract=sample_contract,
        mtm_value=5000.0,
        gain_realized=500.0,
        tax_paid=100.0,
        net_proceeds=4900.0,
    )


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


def _make_minimal_state(
    *,
    holdings: dict[str, float] | None = None,
    defensive_sleeve: float = 0.0,
    leaps_pool: float = 0.0,
    leaps_value: float = 0.0,
    prev_total_nav: float = 100_000.0,
    prev_regime: int = 1,
    prev_date_ts: pd.Timestamp | None = None,
    leaps_ledger: LeapsLedger | None = None,
    leaps_scale: dict[LeapsContract, float] | None = None,
    all_window_ledgers: tuple[LeapsLedger, ...] = (),
    all_gtt_closes: tuple[LeapsGttCloseEvent, ...] = (),
) -> PortfolioState:
    """Build a minimal PortfolioState with sensible defaults."""
    return PortfolioState(
        holdings=holdings if holdings is not None else {"VTI": 100_000.0},
        defensive_sleeve=defensive_sleeve,
        leaps_pool=leaps_pool,
        leaps_value=leaps_value,
        prev_total_nav=prev_total_nav,
        prev_regime=prev_regime,
        prev_date_ts=prev_date_ts,
        leaps_ledger=leaps_ledger,
        leaps_scale=leaps_scale if leaps_scale is not None else {},
        all_window_ledgers=all_window_ledgers,
        all_gtt_closes=all_gtt_closes,
    )


def _make_minimal_ctx(
    *,
    base_assets: tuple[str, ...] = ("VTI",),
    leaps_keys: tuple[str, ...] = (),
    leaps_fraction: float = 0.0,
    gtt_active: bool = False,
    governed_base: tuple[str, ...] = (),
    defensive_weights: dict[str, float] | None = None,
    use_leaps: bool = False,
    w: pd.Series | None = None,
    leaps_config: LeapsConfig | None = None,
) -> BacktestContext:
    """Build a minimal BacktestContext for step-function unit tests."""
    if defensive_weights is None:
        defensive_weights = {"R_f": 1.0}
    if w is None:
        w = pd.Series({a: 1.0 / len(base_assets) for a in base_assets})

    target_weights: dict[str, float] = {a: float(w[a]) for a in w.index}
    config = PortfolioConfig(
        target_weights=target_weights,
        initial_nav=100_000.0,
        monthly_contribution=500.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=leaps_config,
    )
    dates = pd.bdate_range("2023-01-03", periods=30)
    return_data = _make_return_data(dates)
    base_target_w = pd.Series({a: 1.0 / len(base_assets) for a in base_assets})

    return BacktestContext(
        base_assets=base_assets,
        leaps_keys=leaps_keys,
        leaps_fraction=leaps_fraction,
        base_target_w=base_target_w,
        governed_base=governed_base,
        gtt_active=gtt_active,
        defensive_weights=defensive_weights,
        use_leaps=use_leaps,
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
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end={},
        w=w,
    )


def _make_day_inputs(
    *,
    date_ts: pd.Timestamp = pd.Timestamp("2023-03-31"),
    regime_t: int = 1,
) -> DayInputs:
    """Build a minimal DayInputs."""
    return DayInputs(
        date_ts=date_ts,
        day_ret=pd.Series({"VTI": 0.01}),
        regime_t=regime_t,
        def_gross_return=0.0,
        spot=None,
        raw_vix_value=None,
        mtm_iv_value=None,
        rfr=0.04,
        is_month_end=False,
        is_rebal_date=False,
    )


# ---------------------------------------------------------------------------
# _compute_total_nav
# ---------------------------------------------------------------------------


class TestComputeTotalNav:
    """Tests for _compute_total_nav."""

    def test_known_state_exact_sum(self) -> None:
        """Sum of all four components equals expected value."""
        state = _make_minimal_state(
            holdings={"VTI": 50_000.0, "VXUS": 20_000.0},
            leaps_value=7_500.0,
            defensive_sleeve=10_000.0,
            leaps_pool=5_000.0,
        )
        assert _compute_total_nav(state) == pytest.approx(92_500.0)

    def test_all_zero_state(self) -> None:
        """All-zero state returns 0.0."""
        state = _make_minimal_state(
            holdings={"VTI": 0.0},
            leaps_value=0.0,
            defensive_sleeve=0.0,
            leaps_pool=0.0,
        )
        assert _compute_total_nav(state) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _advance_state
# ---------------------------------------------------------------------------


class TestAdvanceState:
    """Tests for _advance_state."""

    def test_prev_fields_updated(self) -> None:
        """prev_total_nav, prev_regime, prev_date_ts are updated from arguments."""
        state = _make_minimal_state(
            holdings={"VTI": 80_000.0},
            leaps_value=12_000.0,
            defensive_sleeve=5_000.0,
            leaps_pool=3_000.0,
            prev_total_nav=90_000.0,
            prev_regime=0,
            prev_date_ts=pd.Timestamp("2023-01-02"),
        )
        total_nav = 100_000.0
        inputs = _make_day_inputs(date_ts=pd.Timestamp("2023-03-31"), regime_t=1)

        result = _advance_state(state, total_nav, inputs)

        assert result.prev_total_nav == pytest.approx(100_000.0)
        assert result.prev_regime == 1
        assert result.prev_date_ts == pd.Timestamp("2023-03-31")

    def test_other_fields_unchanged(self) -> None:
        """All fields other than prev_* carry forward unchanged."""
        state = _make_minimal_state(
            holdings={"VTI": 80_000.0},
            leaps_value=12_000.0,
            defensive_sleeve=5_000.0,
            leaps_pool=3_000.0,
        )
        inputs = _make_day_inputs()
        result = _advance_state(state, 100_000.0, inputs)

        assert result.holdings == {"VTI": 80_000.0}
        assert result.leaps_value == pytest.approx(12_000.0)
        assert result.defensive_sleeve == pytest.approx(5_000.0)
        assert result.leaps_pool == pytest.approx(3_000.0)
        assert result.leaps_ledger is state.leaps_ledger
        assert result.leaps_scale is state.leaps_scale
        assert result.all_window_ledgers is state.all_window_ledgers
        assert result.all_gtt_closes is state.all_gtt_closes


# ---------------------------------------------------------------------------
# _build_weight_row
# ---------------------------------------------------------------------------


class TestBuildWeightRow:
    """Tests for _build_weight_row."""

    def test_base_only_sums_to_one(self) -> None:
        """Base-only portfolio: weight row sums to 1.0."""
        holdings = {"VTI": 60_000.0, "VXUS": 40_000.0}
        total_nav = 100_000.0
        state = _make_minimal_state(holdings=holdings)
        ctx = _make_minimal_ctx(
            base_assets=("VTI", "VXUS"),
            w=pd.Series({"VTI": 0.6, "VXUS": 0.4}),
        )
        row = _build_weight_row(state, total_nav, ctx)
        assert sum(row.values()) == pytest.approx(1.0, rel=1e-9)

    def test_leaps_key_weight(self) -> None:
        """LEAPS key weight computed as leaps_value * (w[k]/leaps_fraction) / total_nav."""
        total_nav = 100_000.0
        leaps_value = 15_000.0
        # Base: 85_000 in VTI; LEAPS: 15_000; leaps_fraction=0.15; w[VTI_LEAPS]=0.15
        state = _make_minimal_state(
            holdings={"VTI": 85_000.0},
            leaps_value=leaps_value,
        )
        w = pd.Series({"VTI": 0.85, "VTI_LEAPS": 0.15})
        ctx = _make_minimal_ctx(
            base_assets=("VTI",),
            leaps_keys=("VTI_LEAPS",),
            leaps_fraction=0.15,
            use_leaps=True,
            w=w,
        )
        row = _build_weight_row(state, total_nav, ctx)
        # share = 0.15 / 0.15 = 1.0; row["VTI_LEAPS"] = 15000 * 1.0 / 100000 = 0.15
        assert row["VTI_LEAPS"] == pytest.approx(0.15, rel=1e-9)

    def test_gtt_defensive_parked_decomposed(self) -> None:
        """GTT active: parked capital split across defensive_weights."""
        # holdings=40k base VTI (non-governed), defensive_sleeve=30k, leaps_pool=0
        # parked=30k; defensive_weights={"BIL":0.5,"R_f":0.5}; total_nav=100k
        # VTI weight = 40000/100000 = 0.40
        # BIL weight = 0.5 * 30000 / 100000 = 0.15
        # R_f weight  = 0.5 * 30000 / 100000 = 0.15
        # => sum = 0.40 + 0.15 + 0.15 = 0.70 ... need another 0.30 in holdings
        # Let's use: holdings={"VTI":70_000.0}, sleeve=30_000 => total_nav=100_000
        total_nav = 100_000.0
        state = _make_minimal_state(
            holdings={"VTI": 70_000.0},
            defensive_sleeve=30_000.0,
        )
        ctx = _make_minimal_ctx(
            base_assets=("VTI",),
            gtt_active=True,
            governed_base=("VTI",),
            defensive_weights={"BIL": 0.5, "R_f": 0.5},
            w=pd.Series({"VTI": 1.0}),
        )
        row = _build_weight_row(state, total_nav, ctx)
        assert row["BIL"] == pytest.approx(0.15, rel=1e-9)
        assert row["R_f"] == pytest.approx(0.15, rel=1e-9)

    def test_combined_all_components_sums_to_one(self) -> None:
        """Base + LEAPS + GTT defensive: weight row sums to 1.0.

        Layout (total_nav = 100_000):
          VTI holdings  = 55_000   => VTI weight = 0.55
          VTI_LEAPS     = 15_000   => VTI_LEAPS weight = 0.15 (leaps_fraction=0.15)
          defensive_sleeve = 20_000 \\
          leaps_pool       = 10_000  => parked = 30_000 => BIL=0.15, R_f=0.15
        Sum: 0.55 + 0.15 + 0.15 + 0.15 = 1.0
        """
        total_nav = 100_000.0
        state = _make_minimal_state(
            holdings={"VTI": 55_000.0},
            leaps_value=15_000.0,
            defensive_sleeve=20_000.0,
            leaps_pool=10_000.0,
        )
        # w for ctx.w — used in LEAPS share calculation (VTI_LEAPS key).
        # target_weights in PortfolioConfig must sum to 1.0; include BIL so the
        # config validation passes. The defensive allocation is not in w itself.
        w = pd.Series({"VTI": 0.55, "VTI_LEAPS": 0.15, "BIL": 0.30})
        ctx = _make_minimal_ctx(
            base_assets=("VTI",),
            leaps_keys=("VTI_LEAPS",),
            leaps_fraction=0.15,
            use_leaps=True,
            gtt_active=True,
            governed_base=("VTI",),
            defensive_weights={"BIL": 0.5, "R_f": 0.5},
            w=w,
        )
        row = _build_weight_row(state, total_nav, ctx)
        assert sum(row.values()) == pytest.approx(1.0, rel=1e-9)

    def test_zero_total_nav_returns_zero_weights(self) -> None:
        """total_nav <= 0 returns all-zero weights without division error."""
        state = _make_minimal_state(holdings={"VTI": 0.0})
        ctx = _make_minimal_ctx()
        row = _build_weight_row(state, 0.0, ctx)
        assert all(v == 0.0 for v in row.values())


# ---------------------------------------------------------------------------
# _assemble_leaps_ledger
# ---------------------------------------------------------------------------


class TestAssembleLeapsLedger:
    """Tests for _assemble_leaps_ledger."""

    def test_single_window_no_gtt(
        self, sample_ledger: LeapsLedger, sample_contract: LeapsContract
    ) -> None:
        """Non-GTT run: returns per-window ledger unchanged."""
        state = _make_minimal_state(leaps_ledger=sample_ledger)
        ctx = _make_minimal_ctx(gtt_active=False, use_leaps=True)
        final_date = pd.Timestamp("2025-01-17")

        result = _assemble_leaps_ledger(state, ctx, final_date)

        assert result is not None
        assert result.contracts == sample_ledger.contracts

    def test_two_window_gtt_assembly(self, sample_contract: LeapsContract) -> None:
        """GTT active: contracts from both windows are concatenated."""
        contract_a = sample_contract
        contract_b = LeapsContract(
            purchase_date=pd.Timestamp("2024-01-02"),
            expiry_date=pd.Timestamp("2026-01-16"),
            strike=170.0,
            spot_at_purchase=210.0,
            premium_paid=50.0,
            notional=21000.0,
            n_contracts=1.5,
            account_type=AccountType.TAXABLE,
        )
        ledger_w1 = LeapsLedger(
            contracts=(contract_a,), roll_events=(), account_type=AccountType.TAXABLE
        )
        ledger_w2 = LeapsLedger(
            contracts=(contract_b,), roll_events=(), account_type=AccountType.TAXABLE
        )
        leaps_config = LeapsConfig(iv=0.20, ltcg_rate=0.20)
        state = _make_minimal_state(
            leaps_ledger=ledger_w2,
            all_window_ledgers=(ledger_w1, ledger_w2),
        )
        ctx = _make_minimal_ctx(
            gtt_active=True,
            use_leaps=True,
            leaps_config=leaps_config,
        )
        final_date = pd.Timestamp("2025-12-31")

        result = _assemble_leaps_ledger(state, ctx, final_date)

        assert result is not None
        assert len(result.contracts) == len(ledger_w1.contracts) + len(ledger_w2.contracts)

    def test_gtt_close_events_attached(
        self,
        sample_contract: LeapsContract,
        sample_gtt_close: LeapsGttCloseEvent,
    ) -> None:
        """Assembled ledger has gtt_close_events == state.all_gtt_closes."""
        ledger = LeapsLedger(
            contracts=(sample_contract,), roll_events=(), account_type=AccountType.TAXABLE
        )
        leaps_config = LeapsConfig(iv=0.20, ltcg_rate=0.20)
        state = _make_minimal_state(
            leaps_ledger=ledger,
            all_window_ledgers=(ledger,),
            all_gtt_closes=(sample_gtt_close,),
        )
        ctx = _make_minimal_ctx(
            gtt_active=True, use_leaps=True, leaps_config=leaps_config
        )
        final_date = pd.Timestamp("2025-12-31")

        result = _assemble_leaps_ledger(state, ctx, final_date)

        assert result is not None
        assert result.gtt_close_events == state.all_gtt_closes

    def test_leaps_scale_frozen_into_partial_close_events(
        self, sample_contract: LeapsContract, sample_ledger: LeapsLedger
    ) -> None:
        """leaps_scale entry creates a LeapsPartialCloseEvent on the ledger."""
        surviving_fraction = 0.8
        original_n = sample_contract.n_contracts  # 2.0

        state = _make_minimal_state(
            leaps_ledger=sample_ledger,
            leaps_scale={sample_contract: surviving_fraction},
        )
        ctx = _make_minimal_ctx(gtt_active=False, use_leaps=True)
        final_date = pd.Timestamp("2025-01-17")

        result = _assemble_leaps_ledger(state, ctx, final_date)

        assert result is not None
        assert len(result.partial_close_events) == 1
        event = result.partial_close_events[0]
        assert event.continuation_contract.n_contracts == pytest.approx(
            original_n * surviving_fraction, rel=1e-9
        )
