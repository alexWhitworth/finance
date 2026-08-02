"""Tests for F-013: _apply_rebalance in src/finance/_step_f013.py.

Covers:
- No-op when neither rebalance date nor month-end.
- QUARTERLY: NAV-neutrality (invariant A5).
- QUARTERLY: correct weight allocation.
- QUARTERLY + GTT defensive: governed key re-swept to sleeve.
- DRIFT not triggered: state unchanged.
- DRIFT triggered: holdings realigned to target.
- Hypothesis property test for A5 (QUARTERLY NAV conservation).
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from finance._step_f013 import _apply_rebalance
from finance.leverage import (
    AccountType,
    LeapsConfig,
    LeapsContract,
    RebalanceRule,
    WeightStrategy,
)
from finance.portfolio import BacktestContext, DayInputs, PortfolioConfig, PortfolioState
from finance.returns import ReturnData


# ---------------------------------------------------------------------------
# Minimal fixture helpers
# ---------------------------------------------------------------------------


def _make_state(
    holdings: dict[str, float],
    *,
    defensive_sleeve: float = 0.0,
    leaps_value: float = 0.0,
    leaps_scale: dict[LeapsContract, float] | None = None,
    leaps_ledger: object | None = None,
) -> PortfolioState:
    """Build a minimal PortfolioState for use in tests."""
    return PortfolioState(
        holdings=holdings,
        defensive_sleeve=defensive_sleeve,
        leaps_pool=0.0,
        leaps_value=leaps_value,
        prev_total_nav=sum(holdings.values()) + leaps_value + defensive_sleeve,
        prev_regime=1,
        prev_date_ts=None,
        leaps_ledger=leaps_ledger,  # type: ignore[arg-type]
        leaps_scale=leaps_scale if leaps_scale is not None else {},
        all_window_ledgers=(),
        all_gtt_closes=(),
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


def _make_ctx(
    *,
    base_assets: tuple[str, ...] = ("VTI", "VXUS", "GLD"),
    base_target_w: pd.Series | None = None,
    governed_base: tuple[str, ...] = (),
    gtt_active: bool = False,
    rebalance_rule: RebalanceRule = RebalanceRule.QUARTERLY,
    leaps_keys: tuple[str, ...] = (),
    leaps_fraction: float = 0.0,
    w: pd.Series | None = None,
) -> BacktestContext:
    """Build a minimal BacktestContext with sensible defaults."""
    dates = pd.bdate_range("2023-01-02", periods=30)
    if base_target_w is None:
        n = len(base_assets)
        uniform = 1.0 / n
        base_target_w = pd.Series({a: uniform for a in base_assets})
    if w is None:
        w_dict = {a: float(base_target_w[a]) * (1.0 - leaps_fraction) for a in base_assets}
        for k in leaps_keys:
            w_dict[k] = leaps_fraction / max(len(leaps_keys), 1)
        w = pd.Series(w_dict)
    config = PortfolioConfig(
        target_weights={str(k): float(v) for k, v in w.items()},
        initial_nav=100_000.0,
        monthly_contribution=1_000.0,
        rebalance_rule=rebalance_rule,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(iv=0.20, ltcg_rate=0.20) if leaps_keys else None,
    )
    return BacktestContext(
        base_assets=base_assets,
        leaps_keys=leaps_keys,
        leaps_fraction=leaps_fraction,
        base_target_w=base_target_w,
        governed_base=governed_base,
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0},
        use_leaps=len(leaps_keys) > 0,
        iv=0.20,
        leaps_monthly=0.0,
        base_contribution=1_000.0,
        config=config,
        return_data=_make_return_data(dates),
        underlying_prices=None,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=None,
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset({pd.Timestamp("2023-03-31")}),
        month_end_dates=frozenset({pd.Timestamp("2023-01-31")}),
        long_window_end={},
        w=w,
    )


def _make_inputs(
    *,
    date_ts: pd.Timestamp = pd.Timestamp("2023-01-15"),
    is_rebal_date: bool = False,
    is_month_end: bool = False,
    regime_t: int = 1,
) -> DayInputs:
    """Build a minimal DayInputs."""
    return DayInputs(
        date_ts=date_ts,
        day_ret=pd.Series(dtype=float),
        regime_t=regime_t,
        def_gross_return=0.0,
        spot=None,
        raw_vix_value=None,
        mtm_iv_value=None,
        rfr=0.04,
        is_month_end=is_month_end,
        is_rebal_date=is_rebal_date,
    )


# ---------------------------------------------------------------------------
# Test 1: No-op — not rebalance date, not month-end
# ---------------------------------------------------------------------------


def test_noop_neither_rebal_nor_month_end() -> None:
    """State is unchanged when is_rebal_date=False and is_month_end=False."""
    state = _make_state({"VTI": 60_000.0, "VXUS": 25_000.0, "GLD": 15_000.0})
    ctx = _make_ctx()
    inputs = _make_inputs(is_rebal_date=False, is_month_end=False)
    result = _apply_rebalance(state, inputs, ctx)
    assert result is state


# ---------------------------------------------------------------------------
# Test 2: QUARTERLY — NAV-neutral (invariant A5)
# ---------------------------------------------------------------------------


def test_quarterly_nav_neutral() -> None:
    """QUARTERLY rebalance preserves sum(holdings) within 1e-9 (invariant A5)."""
    holdings_in = {"VTI": 60_000.0, "VXUS": 20_000.0, "GLD": 20_000.0}
    base_target_w = pd.Series({"VTI": 0.6, "VXUS": 0.2, "GLD": 0.2})
    state = _make_state(holdings_in)
    ctx = _make_ctx(
        base_assets=("VTI", "VXUS", "GLD"),
        base_target_w=base_target_w,
    )
    inputs = _make_inputs(is_rebal_date=True)
    result = _apply_rebalance(state, inputs, ctx)
    assert sum(result.holdings.values()) == pytest.approx(
        sum(holdings_in.values()), rel=1e-9
    )


# ---------------------------------------------------------------------------
# Test 3: QUARTERLY — weights correct after rebalance
# ---------------------------------------------------------------------------


def test_quarterly_weights_correct() -> None:
    """After QUARTERLY rebalance, each asset's weight matches base_target_w."""
    holdings_in = {"VTI": 50_000.0, "VXUS": 30_000.0, "GLD": 20_000.0}
    base_target_w = pd.Series({"VTI": 0.6, "VXUS": 0.2, "GLD": 0.2})
    state = _make_state(holdings_in)
    ctx = _make_ctx(
        base_assets=("VTI", "VXUS", "GLD"),
        base_target_w=base_target_w,
    )
    inputs = _make_inputs(is_rebal_date=True)
    result = _apply_rebalance(state, inputs, ctx)
    total = sum(result.holdings.values())
    assert result.holdings["VTI"] / total == pytest.approx(0.6, rel=1e-9)
    assert result.holdings["VXUS"] / total == pytest.approx(0.2, rel=1e-9)
    assert result.holdings["GLD"] / total == pytest.approx(0.2, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 4: QUARTERLY + GTT defensive — governed key re-swept into sleeve
# ---------------------------------------------------------------------------


def test_quarterly_gtt_defensive_governed_swept() -> None:
    """On a defensive QUARTERLY rebalance day, governed holdings are zeroed and added to sleeve."""
    holdings_in = {"VTI": 60_000.0, "VXUS": 20_000.0, "GLD": 20_000.0}
    base_target_w = pd.Series({"VTI": 0.6, "VXUS": 0.2, "GLD": 0.2})
    sleeve_in = 5_000.0
    state = _make_state(holdings_in, defensive_sleeve=sleeve_in)
    ctx = _make_ctx(
        base_assets=("VTI", "VXUS", "GLD"),
        base_target_w=base_target_w,
        governed_base=("VTI",),
        gtt_active=True,
    )
    inputs = _make_inputs(is_rebal_date=True, regime_t=0)
    result = _apply_rebalance(state, inputs, ctx)

    # VTI must be zeroed in holdings.
    assert result.holdings["VTI"] == 0.0

    # Total base NAV (before sweep) was 100_000. After rebalance, VTI target = 60_000.
    # That 60_000 should now be in the sleeve (plus the original sleeve).
    base_nav = sum(holdings_in.values())  # 100_000
    expected_vti_rebalanced = base_nav * 0.6  # 60_000
    assert result.defensive_sleeve == pytest.approx(sleeve_in + expected_vti_rebalanced, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 5: DRIFT not triggered — state unchanged
# ---------------------------------------------------------------------------


def test_drift_not_triggered_noop() -> None:
    """DRIFT rebalance is a no-op when current weights are within the band."""
    # Equal-weight allocation, equal-weight target → no drift.
    holdings_in = {"VTI": 50_000.0, "VXUS": 50_000.0}
    base_target_w = pd.Series({"VTI": 0.5, "VXUS": 0.5})
    w = pd.Series({"VTI": 0.5, "VXUS": 0.5})
    state = _make_state(holdings_in)
    ctx = _make_ctx(
        base_assets=("VTI", "VXUS"),
        base_target_w=base_target_w,
        rebalance_rule=RebalanceRule.DRIFT,
        w=w,
    )
    inputs = _make_inputs(is_month_end=True)
    result = _apply_rebalance(state, inputs, ctx)
    # Holdings unchanged; same object returned.
    assert result.holdings == holdings_in


# ---------------------------------------------------------------------------
# Test 6: DRIFT triggered — holdings realigned
# ---------------------------------------------------------------------------


def test_drift_triggered_realigns_holdings() -> None:
    """DRIFT rebalance realigns holdings when an asset has drifted beyond the band."""
    # VTI at 80% (target 60%) — well beyond the ±10% relative band.
    holdings_in = {"VTI": 80_000.0, "VXUS": 10_000.0, "GLD": 10_000.0}
    base_target_w = pd.Series({"VTI": 0.6, "VXUS": 0.2, "GLD": 0.2})
    w = pd.Series({"VTI": 0.6, "VXUS": 0.2, "GLD": 0.2})
    state = _make_state(holdings_in)
    ctx = _make_ctx(
        base_assets=("VTI", "VXUS", "GLD"),
        base_target_w=base_target_w,
        rebalance_rule=RebalanceRule.DRIFT,
        w=w,
    )
    inputs = _make_inputs(is_month_end=True)
    result = _apply_rebalance(state, inputs, ctx)

    total = sum(result.holdings.values())
    assert result.holdings["VTI"] / total == pytest.approx(0.6, rel=1e-9)
    assert result.holdings["VXUS"] / total == pytest.approx(0.2, rel=1e-9)
    assert result.holdings["GLD"] / total == pytest.approx(0.2, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 7: Hypothesis — QUARTERLY invariant A5 (sum conservation)
# ---------------------------------------------------------------------------


@given(
    vti=st.floats(min_value=1.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    vxus=st.floats(min_value=1.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    gld=st.floats(min_value=1.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    w_raw=st.lists(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=3,
    ),
)
@settings(max_examples=300)
def test_quarterly_a5_sum_conservation(
    vti: float, vxus: float, gld: float, w_raw: list[float]
) -> None:
    """Hypothesis: QUARTERLY rebalance conserves sum(holdings) within 1e-9 for random inputs."""
    total_w = sum(w_raw)
    assume(total_w > 1e-6)
    assets = ("VTI", "VXUS", "GLD")
    norm_w = [x / total_w for x in w_raw]
    base_target_w = pd.Series(dict(zip(assets, norm_w)))
    holdings_in = {"VTI": vti, "VXUS": vxus, "GLD": gld}
    state = _make_state(holdings_in)
    ctx = _make_ctx(
        base_assets=assets,
        base_target_w=base_target_w,
    )
    inputs = _make_inputs(is_rebal_date=True)
    result = _apply_rebalance(state, inputs, ctx)
    assert sum(result.holdings.values()) == pytest.approx(
        sum(holdings_in.values()), rel=1e-9
    )
