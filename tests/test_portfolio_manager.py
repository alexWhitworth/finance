"""Unit tests for portfolio_manager.py — LivePortfolio and as_live_portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from finance._portfolio_types import PortfolioConfig, PortfolioState
from finance.data import PriceData
from finance.leverage import (
    AccountType,
    LeapsConfig,
    LeapsContract,
    RebalanceRule,
    WeightStrategy,
)
from finance.portfolio import run_backtest
from finance.portfolio_manager import (
    HoldingView,
    LivePortfolio,
    NavBreakdown,
    as_live_portfolio,
    compute_holdings_view,
    compute_nav_breakdown,
)
from finance.returns import ReturnData, build_return_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TICKERS = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
_EQUAL_WEIGHTS = {t: 1.0 / len(_TICKERS) for t in _TICKERS}

_LEAPS_WEIGHTS = {
    "VTI_LEAPS": 0.30, "VTI": 0.10, "VXUS": 0.15, "GLD": 0.15,
    "MUB": 0.10, "KMLM": 0.10, "VGIT": 0.10,
}

_AS_OF = pd.Timestamp("2020-01-15")
_FUTURE = pd.Timestamp("2022-06-15")


def _config(
    weights: dict[str, float] | None = None,
    initial_nav: float = 1_000_000.0,
    contribution: float = 0.0,
    leaps_config: LeapsConfig | None = None,
    rebalance_rule: RebalanceRule = RebalanceRule.QUARTERLY,
) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights=weights or dict(_EQUAL_WEIGHTS),
        initial_nav=initial_nav,
        monthly_contribution=contribution,
        rebalance_rule=rebalance_rule,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=leaps_config,
    )


def _make_rd_and_pd(
    n: int = 504,
    daily_ret: float = 0.0003,
    daily_vol: float = 0.01,
    seed: int = 42,
    start: str = "2015-01-02",
) -> tuple[ReturnData, PriceData]:
    """Return matching (ReturnData, PriceData) pair from the same synthetic series."""
    idx = pd.bdate_range(start, periods=n + 1)
    rng = np.random.default_rng(seed)
    starts = {
        "VTI": 200.0, "VXUS": 60.0, "GLD": 170.0,
        "MUB": 55.0, "KMLM": 25.0, "VGIT": 65.0,
    }
    prices_data = {
        t: starts[t] * np.cumprod(1 + rng.normal(daily_ret, daily_vol, n + 1))
        for t in _TICKERS
    }
    prices = pd.DataFrame(prices_data, index=idx)
    dividends = pd.DataFrame(0.0, index=idx, columns=list(_TICKERS))
    pd_obj = PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=pd.DataFrame(),
        tickers=_TICKERS,
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
    )
    return build_return_data(pd_obj, apply_tey=False), pd_obj


def _minimal_state(
    as_of: pd.Timestamp,
    holdings: dict[str, float] | None = None,
) -> PortfolioState:
    """Build a minimal PortfolioState with the given date and holdings."""
    return PortfolioState(
        holdings=holdings or {},
        defensive_sleeve=0.0,
        leaps_pool=0.0,
        leaps_value=0.0,
        prev_total_nav=sum((holdings or {}).values()),
        prev_regime=1,
        prev_date_ts=as_of,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )


def _fake_contract(
    purchase: str = "2019-01-15",
    expiry: str = "2021-01-15",
    spot: float = 200.0,
) -> LeapsContract:
    """Create a minimal LeapsContract with given dates."""
    return LeapsContract(
        purchase_date=pd.Timestamp(purchase),
        expiry_date=pd.Timestamp(expiry),
        strike=spot * 0.8,
        spot_at_purchase=spot,
        premium_paid=20.0,
        notional=spot * 100,
        n_contracts=5.0,
        account_type=AccountType.TAXABLE,
    )


# ---------------------------------------------------------------------------
# LivePortfolio — validation
# ---------------------------------------------------------------------------


def test_live_portfolio_valid_construction() -> None:
    """LivePortfolio constructs without error when all invariants hold."""
    lp = LivePortfolio(
        as_of_date=_AS_OF,
        holdings={"VTI": 50_000.0},
        target_weights={"VTI": 1.0},
        leaps_contracts=(),
        gtt_regime=None,
    )
    assert lp.as_of_date == _AS_OF
    assert lp.defensive_sleeve == 0.0
    assert lp.leaps_pool == 0.0


def test_live_portfolio_bad_weights_raises() -> None:
    """ValueError raised when target_weights do not sum to 1.0 ± 1e-6."""
    with pytest.raises(ValueError, match=r"target_weights must sum to 1\.0"):
        LivePortfolio(
            as_of_date=_AS_OF,
            holdings={"VTI": 50_000.0},
            target_weights={"VTI": 0.5, "VXUS": 0.3},  # sums to 0.8
            leaps_contracts=(),
            gtt_regime=None,
        )


def test_live_portfolio_bad_weights_over_one_raises() -> None:
    """ValueError raised when target_weights sum to more than 1.0 + 1e-6."""
    with pytest.raises(ValueError, match=r"target_weights must sum to 1\.0"):
        LivePortfolio(
            as_of_date=_AS_OF,
            holdings={},
            target_weights={"VTI": 0.6, "VXUS": 0.6},  # sums to 1.2
            leaps_contracts=(),
            gtt_regime=None,
        )


def test_live_portfolio_leaps_scale_zero_raises() -> None:
    """ValueError raised when leaps_scale == 0.0 (must be > 0)."""
    contract = _fake_contract(expiry="2022-01-15")
    with pytest.raises(ValueError, match="leaps_scale must be in"):
        LivePortfolio(
            as_of_date=_AS_OF,
            holdings={},
            target_weights={"VTI": 1.0},
            leaps_contracts=((contract, 0.0),),
            gtt_regime=None,
        )


def test_live_portfolio_leaps_scale_above_one_raises() -> None:
    """ValueError raised when leaps_scale > 1.0."""
    contract = _fake_contract(expiry="2022-01-15")
    with pytest.raises(ValueError, match="leaps_scale must be in"):
        LivePortfolio(
            as_of_date=_AS_OF,
            holdings={},
            target_weights={"VTI": 1.0},
            leaps_contracts=((contract, 1.001),),
            gtt_regime=None,
        )


def test_live_portfolio_expired_contract_raises() -> None:
    """ValueError raised when contract.expiry_date <= as_of_date."""
    contract = _fake_contract(expiry="2020-01-01")  # before _AS_OF
    with pytest.raises(ValueError, match=r"expiry_date.*must be"):
        LivePortfolio(
            as_of_date=_AS_OF,
            holdings={},
            target_weights={"VTI": 1.0},
            leaps_contracts=((contract, 0.8),),
            gtt_regime=None,
        )


def test_live_portfolio_expired_same_day_raises() -> None:
    """ValueError raised when contract.expiry_date == as_of_date (not strictly after)."""
    contract = _fake_contract(expiry="2020-01-15")  # same as _AS_OF
    with pytest.raises(ValueError, match=r"expiry_date.*must be"):
        LivePortfolio(
            as_of_date=_AS_OF,
            holdings={},
            target_weights={"VTI": 1.0},
            leaps_contracts=((contract, 0.5),),
            gtt_regime=None,
        )


def test_live_portfolio_valid_contract_passes() -> None:
    """No error when contract.expiry_date is strictly after as_of_date."""
    contract = _fake_contract(expiry="2022-06-15")
    lp = LivePortfolio(
        as_of_date=_AS_OF,
        holdings={"VTI": 10_000.0},
        target_weights={"VTI": 1.0},
        leaps_contracts=((contract, 0.9),),
        gtt_regime=1,
    )
    assert len(lp.leaps_contracts) == 1


def test_live_portfolio_empty_leaps_no_expiry_check() -> None:
    """Empty leaps_contracts: no expiry validation runs."""
    lp = LivePortfolio(
        as_of_date=_AS_OF,
        holdings={"VTI": 1.0},
        target_weights={"VTI": 1.0},
        leaps_contracts=(),
        gtt_regime=None,
    )
    assert lp.leaps_contracts == ()


def test_live_portfolio_zero_weight_entry_valid() -> None:
    """A 0.0 weight entry is valid as long as the sum is 1.0."""
    lp = LivePortfolio(
        as_of_date=_AS_OF,
        holdings={"VTI": 1.0},
        target_weights={"VTI": 1.0, "VXUS": 0.0},
        leaps_contracts=(),
        gtt_regime=None,
    )
    assert abs(sum(lp.target_weights.values()) - 1.0) < 1e-9


def test_live_portfolio_is_frozen() -> None:
    """LivePortfolio is frozen — attribute assignment raises."""
    lp = LivePortfolio(
        as_of_date=_AS_OF,
        holdings={},
        target_weights={"VTI": 1.0},
        leaps_contracts=(),
        gtt_regime=None,
    )
    with pytest.raises((AttributeError, TypeError)):
        lp.holdings = {}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# as_live_portfolio — no LEAPS backtest
# ---------------------------------------------------------------------------


def test_as_live_portfolio_no_leaps_empty_contracts() -> None:
    """as_live_portfolio returns empty leaps_contracts when backtest had no LEAPS."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    lp = as_live_portfolio(result)
    assert lp.leaps_contracts == ()


def test_as_live_portfolio_no_leaps_holdings_dict() -> None:
    """LivePortfolio.holdings is a plain dict of base asset values."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    lp = as_live_portfolio(result)
    assert isinstance(lp.holdings, dict)
    assert len(lp.holdings) == len(_EQUAL_WEIGHTS)


def test_as_live_portfolio_target_weights_sum_to_one() -> None:
    """target_weights on the returned LivePortfolio sum to 1.0 within 1e-6."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    lp = as_live_portfolio(result)
    total = sum(lp.target_weights.values())
    assert abs(total - 1.0) < 1e-6


def test_as_live_portfolio_as_of_date_matches_final_nav_date() -> None:
    """LivePortfolio.as_of_date equals the last date in nav_series."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    lp = as_live_portfolio(result)
    assert lp.as_of_date == pd.Timestamp(result.nav_series.index[-1])


def test_as_live_portfolio_gtt_inactive_regime_none() -> None:
    """gtt_active=False → gtt_regime is None."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    lp = as_live_portfolio(result, gtt_active=False)
    assert lp.gtt_regime is None


def test_as_live_portfolio_gtt_active_regime_set() -> None:
    """gtt_active=True → gtt_regime is int from final_state.prev_regime."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    lp = as_live_portfolio(result, gtt_active=True)
    assert lp.gtt_regime in {0, 1}


# ---------------------------------------------------------------------------
# as_live_portfolio — with LEAPS backtest
# ---------------------------------------------------------------------------


def test_as_live_portfolio_with_leaps_has_contracts() -> None:
    """as_live_portfolio returns live contracts when backtest had LEAPS."""
    rd, pd_obj = _make_rd_and_pd(504)
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS),
        leaps_config=LeapsConfig(account_type=AccountType.TAXABLE),
        contribution=5_000.0,
    )
    result = run_backtest(rd, pd_obj, cfg)
    lp = as_live_portfolio(result)
    # A 2-year horizon backtest with LEAPS should have at least one live contract.
    assert len(lp.leaps_contracts) > 0


def test_as_live_portfolio_leaps_contracts_are_tuples() -> None:
    """Each element of leaps_contracts is a (LeapsContract, float) pair."""
    rd, pd_obj = _make_rd_and_pd(252)
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS),
        leaps_config=LeapsConfig(account_type=AccountType.TAXABLE),
        contribution=3_000.0,
    )
    result = run_backtest(rd, pd_obj, cfg)
    lp = as_live_portfolio(result)
    for pair in lp.leaps_contracts:
        contract, scale = pair
        assert isinstance(contract, LeapsContract)
        assert 0.0 < scale <= 1.0


def test_as_live_portfolio_with_leaps_target_weights_sum_to_one() -> None:
    """target_weights sum to 1.0 in a LEAPS backtest (LEAPS key included)."""
    rd, pd_obj = _make_rd_and_pd(252)
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS),
        leaps_config=LeapsConfig(account_type=AccountType.TAXABLE),
        contribution=3_000.0,
    )
    result = run_backtest(rd, pd_obj, cfg)
    lp = as_live_portfolio(result)
    total = sum(lp.target_weights.values())
    assert abs(total - 1.0) < 1e-6


def test_as_live_portfolio_leaps_none_ledger_returns_empty() -> None:
    """final_state.leaps_ledger=None → leaps_contracts is empty tuple."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    assert result.final_state.leaps_ledger is None
    lp = as_live_portfolio(result)
    assert lp.leaps_contracts == ()


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(
    w_a=st.floats(min_value=0.01, max_value=0.99),
)
@settings(max_examples=50)
def test_live_portfolio_target_weights_sum_invariant(w_a: float) -> None:
    """LivePortfolio with exactly-summing weights always constructs without error."""
    w_b = 1.0 - w_a
    lp = LivePortfolio(
        as_of_date=_AS_OF,
        holdings={},
        target_weights={"A": w_a, "B": w_b},
        leaps_contracts=(),
        gtt_regime=None,
    )
    assert abs(sum(lp.target_weights.values()) - 1.0) < 1e-9


@given(
    bad_sum=st.floats(min_value=0.0, max_value=0.5),
)
@settings(max_examples=30)
def test_live_portfolio_bad_sum_always_raises(bad_sum: float) -> None:
    """LivePortfolio always raises when target_weights sum is clearly not 1.0."""
    # bad_sum in [0.0, 0.5] guarantees |sum - 1.0| > 1e-6
    with pytest.raises(ValueError, match=r"target_weights must sum to 1\.0"):
        LivePortfolio(
            as_of_date=_AS_OF,
            holdings={},
            target_weights={"A": bad_sum},
            leaps_contracts=(),
            gtt_regime=None,
        )


# ---------------------------------------------------------------------------
# F-010: compute_nav_breakdown
# ---------------------------------------------------------------------------


def _make_portfolio(
    holdings: dict[str, float],
    defensive_sleeve: float = 0.0,
    leaps_pool: float = 0.0,
) -> LivePortfolio:
    """Build a minimal LivePortfolio with the given holdings."""
    total = sum(holdings.values()) or 1.0
    tw = {k: v / total for k, v in holdings.items()} if holdings else {"VTI": 1.0}
    return LivePortfolio(
        as_of_date=_AS_OF,
        holdings=holdings,
        target_weights=tw,
        leaps_contracts=(),
        gtt_regime=None,
        defensive_sleeve=defensive_sleeve,
        leaps_pool=leaps_pool,
    )


def test_nav_breakdown_total_nav_identity_no_extras() -> None:
    """total_nav == base_nav when no LEAPS, defensive sleeve, or pool (I1)."""
    holdings = {"VTI": 60_000.0, "VXUS": 40_000.0}
    lp = _make_portfolio(holdings)
    nb = compute_nav_breakdown(lp)
    assert abs(nb.total_nav - (nb.base_nav + nb.leaps_nav + nb.defensive_sleeve + nb.leaps_pool)) < 1e-9
    assert abs(nb.total_nav - 100_000.0) < 1e-9


def test_nav_breakdown_leaps_mtm_included() -> None:
    """total_nav includes leaps_mtm (I1)."""
    holdings = {"VTI": 50_000.0}
    lp = _make_portfolio(holdings)
    nb = compute_nav_breakdown(lp, leaps_mtm=10_000.0)
    assert abs(nb.total_nav - 60_000.0) < 1e-9
    assert abs(nb.leaps_nav - 10_000.0) < 1e-9


def test_nav_breakdown_defensive_sleeve_included() -> None:
    """total_nav includes defensive_sleeve (I1)."""
    holdings = {"VTI": 40_000.0}
    lp = _make_portfolio(holdings, defensive_sleeve=20_000.0)
    nb = compute_nav_breakdown(lp)
    assert abs(nb.total_nav - 60_000.0) < 1e-9
    assert abs(nb.defensive_sleeve - 20_000.0) < 1e-9


def test_nav_breakdown_leaps_pool_included() -> None:
    """total_nav includes leaps_pool (I1)."""
    holdings = {"VTI": 40_000.0}
    lp = _make_portfolio(holdings, leaps_pool=5_000.0)
    nb = compute_nav_breakdown(lp)
    assert abs(nb.total_nav - 45_000.0) < 1e-9


def test_nav_breakdown_all_components() -> None:
    """total_nav is exact sum of all four components (I1) — boundary test."""
    holdings = {"VTI": 30_000.0, "VXUS": 20_000.0}
    lp = _make_portfolio(holdings, defensive_sleeve=10_000.0, leaps_pool=5_000.0)
    nb = compute_nav_breakdown(lp, leaps_mtm=8_000.0)
    expected_total = 30_000.0 + 20_000.0 + 8_000.0 + 10_000.0 + 5_000.0
    assert abs(nb.total_nav - expected_total) < 1e-9
    assert abs(nb.total_nav - (nb.base_nav + nb.leaps_nav + nb.defensive_sleeve + nb.leaps_pool)) < 1e-9


def test_nav_breakdown_is_frozen() -> None:
    """NavBreakdown is frozen — attribute assignment raises."""
    lp = _make_portfolio({"VTI": 10_000.0})
    nb = compute_nav_breakdown(lp)
    with pytest.raises((AttributeError, TypeError)):
        nb.total_nav = 0.0  # type: ignore[misc]


def test_nav_breakdown_zero_holdings() -> None:
    """Portfolio with zero holdings: all-zero NavBreakdown except leaps_mtm."""
    lp = LivePortfolio(
        as_of_date=_AS_OF,
        holdings={},
        target_weights={"VTI": 1.0},
        leaps_contracts=(),
        gtt_regime=None,
    )
    nb = compute_nav_breakdown(lp, leaps_mtm=5_000.0)
    assert nb.base_nav == 0.0
    assert abs(nb.total_nav - 5_000.0) < 1e-9


# ---------------------------------------------------------------------------
# F-010: compute_holdings_view
# ---------------------------------------------------------------------------


def test_holdings_view_actual_weight_sum_le_one() -> None:
    """sum(h.actual_weight) <= 1.0 + 1e-9 for base-only portfolio (I2)."""
    holdings = {"VTI": 50_000.0, "VXUS": 30_000.0, "GLD": 20_000.0}
    lp = _make_portfolio(holdings)
    nb = compute_nav_breakdown(lp)
    views = compute_holdings_view(lp, nb)
    total_w = sum(h.actual_weight for h in views)
    assert total_w <= 1.0 + 1e-9, f"actual_weight sum = {total_w}"


def test_holdings_view_actual_weight_lt_one_with_defensive_sleeve() -> None:
    """actual_weight sum < 1.0 when defensive_sleeve > 0 (I2, base assets only)."""
    holdings = {"VTI": 40_000.0, "VXUS": 20_000.0}
    lp = _make_portfolio(holdings, defensive_sleeve=10_000.0)
    nb = compute_nav_breakdown(lp)
    views = compute_holdings_view(lp, nb)
    total_w = sum(h.actual_weight for h in views)
    assert total_w < 1.0 - 1e-9, f"expected total_w < 1.0; got {total_w}"


def test_holdings_view_weight_drift_computation() -> None:
    """weight_drift = actual_weight - target_weight (signed)."""
    holdings = {"VTI": 80_000.0, "VXUS": 20_000.0}
    lp = LivePortfolio(
        as_of_date=_AS_OF,
        holdings=holdings,
        target_weights={"VTI": 0.6, "VXUS": 0.4},
        leaps_contracts=(),
        gtt_regime=None,
    )
    nb = compute_nav_breakdown(lp)
    views = {h.ticker: h for h in compute_holdings_view(lp, nb)}
    # VTI is overweight: actual=0.8, target=0.6 → drift=+0.2
    assert abs(views["VTI"].weight_drift - 0.2) < 1e-9
    # VXUS is underweight: actual=0.2, target=0.4 → drift=-0.2
    assert abs(views["VXUS"].weight_drift + 0.2) < 1e-9


def test_holdings_view_relative_drift_none_when_target_zero() -> None:
    """relative_drift is None when target_weight == 0."""
    holdings = {"VTI": 50_000.0, "VXUS": 50_000.0}
    lp = LivePortfolio(
        as_of_date=_AS_OF,
        holdings=holdings,
        target_weights={"VTI": 1.0, "VXUS": 0.0},
        leaps_contracts=(),
        gtt_regime=None,
    )
    nb = compute_nav_breakdown(lp)
    views = {h.ticker: h for h in compute_holdings_view(lp, nb)}
    assert views["VXUS"].relative_drift is None


def test_holdings_view_relative_drift_computed_correctly() -> None:
    """relative_drift = weight_drift / target_weight when target_weight != 0."""
    holdings = {"VTI": 80_000.0, "VXUS": 20_000.0}
    lp = LivePortfolio(
        as_of_date=_AS_OF,
        holdings=holdings,
        target_weights={"VTI": 0.6, "VXUS": 0.4},
        leaps_contracts=(),
        gtt_regime=None,
    )
    nb = compute_nav_breakdown(lp)
    views = {h.ticker: h for h in compute_holdings_view(lp, nb)}
    # VTI: drift=0.2, target=0.6 → relative = 0.2/0.6 ≈ 0.3333
    assert abs(views["VTI"].relative_drift - (0.2 / 0.6)) < 1e-9  # type: ignore[operator]
    # VXUS: drift=-0.2, target=0.4 → relative = -0.2/0.4 = -0.5
    assert abs(views["VXUS"].relative_drift + 0.5) < 1e-9  # type: ignore[operator]


def test_holdings_view_ticker_absent_from_target_weights() -> None:
    """Ticker in holdings but absent from target_weights gets target_weight=0.0."""
    holdings = {"VTI": 70_000.0, "EXTRA": 30_000.0}
    lp = LivePortfolio(
        as_of_date=_AS_OF,
        holdings=holdings,
        target_weights={"VTI": 1.0},
        leaps_contracts=(),
        gtt_regime=None,
    )
    nb = compute_nav_breakdown(lp)
    views = {h.ticker: h for h in compute_holdings_view(lp, nb)}
    assert views["EXTRA"].target_weight == 0.0
    assert views["EXTRA"].relative_drift is None


def test_holdings_view_zero_value_asset() -> None:
    """Zero-value asset produces actual_weight=0.0 without division error."""
    holdings = {"VTI": 100_000.0, "GLD": 0.0}
    lp = LivePortfolio(
        as_of_date=_AS_OF,
        holdings=holdings,
        target_weights={"VTI": 0.9, "GLD": 0.1},
        leaps_contracts=(),
        gtt_regime=None,
    )
    nb = compute_nav_breakdown(lp)
    views = {h.ticker: h for h in compute_holdings_view(lp, nb)}
    assert views["GLD"].actual_weight == 0.0
    assert views["GLD"].dollar_value == 0.0


def test_holdings_view_returns_tuple_of_holding_views() -> None:
    """Return type is tuple[HoldingView, ...]."""
    holdings = {"VTI": 50_000.0, "VXUS": 50_000.0}
    lp = _make_portfolio(holdings)
    nb = compute_nav_breakdown(lp)
    views = compute_holdings_view(lp, nb)
    assert isinstance(views, tuple)
    assert all(isinstance(h, HoldingView) for h in views)


def test_holdings_view_is_frozen() -> None:
    """HoldingView is frozen — attribute assignment raises."""
    holdings = {"VTI": 50_000.0}
    lp = _make_portfolio(holdings)
    nb = compute_nav_breakdown(lp)
    views = compute_holdings_view(lp, nb)
    with pytest.raises((AttributeError, TypeError)):
        views[0].actual_weight = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Property-based: I1 and I2 invariants
# ---------------------------------------------------------------------------


@given(
    base_vti=st.floats(min_value=1.0, max_value=1_000_000.0),
    base_vxus=st.floats(min_value=1.0, max_value=1_000_000.0),
    leaps_mtm=st.floats(min_value=0.0, max_value=500_000.0),
    defensive=st.floats(min_value=0.0, max_value=500_000.0),
    pool=st.floats(min_value=0.0, max_value=500_000.0),
)
@settings(max_examples=100)
def test_nav_breakdown_total_nav_identity_property(
    base_vti: float,
    base_vxus: float,
    leaps_mtm: float,
    defensive: float,
    pool: float,
) -> None:
    """Property I1: total_nav == base_nav + leaps_nav + defensive_sleeve + leaps_pool."""
    holdings = {"VTI": base_vti, "VXUS": base_vxus}
    lp = _make_portfolio(holdings, defensive_sleeve=defensive, leaps_pool=pool)
    nb = compute_nav_breakdown(lp, leaps_mtm=leaps_mtm)
    identity = nb.base_nav + nb.leaps_nav + nb.defensive_sleeve + nb.leaps_pool
    assert abs(nb.total_nav - identity) < 1e-9, f"I1 violated: {nb.total_nav} != {identity}"


@given(
    base_vti=st.floats(min_value=1.0, max_value=1_000_000.0),
    base_vxus=st.floats(min_value=1.0, max_value=1_000_000.0),
    defensive=st.floats(min_value=0.0, max_value=500_000.0),
    pool=st.floats(min_value=0.0, max_value=500_000.0),
)
@settings(max_examples=100)
def test_holdings_view_weight_sum_le_one_property(
    base_vti: float,
    base_vxus: float,
    defensive: float,
    pool: float,
) -> None:
    """Property I2: sum(actual_weight) <= 1.0 + 1e-9 for any base-only portfolio."""
    holdings = {"VTI": base_vti, "VXUS": base_vxus}
    lp = _make_portfolio(holdings, defensive_sleeve=defensive, leaps_pool=pool)
    nb = compute_nav_breakdown(lp)
    views = compute_holdings_view(lp, nb)
    total_w = sum(h.actual_weight for h in views)
    assert total_w <= 1.0 + 1e-9, f"I2 violated: sum(actual_weight) = {total_w}"
