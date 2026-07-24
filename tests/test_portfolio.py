"""Tests for portfolio.py — rebalance dates, contributions, and the backtest loop."""

import numpy as np
import pandas as pd
import pytest

from finance.data import PriceData
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance.portfolio import (
    BacktestResult,
    PortfolioConfig,
    apply_contribution,
    compute_target_weights,
    get_rebalance_dates,
    run_backtest,
)
from finance.returns import ReturnData, build_return_data

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_TICKERS = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
_EQUAL_WEIGHTS = {t: 1.0 / len(_TICKERS) for t in _TICKERS}

_QUARTER_END_MONTHS = {3, 6, 9, 12}


def _config(
    weights: dict[str, float] | None = None,
    initial_nav: float = 1_000_000.0,
    contribution: float = 0.0,
    leaps_config: LeapsConfig | None = None,
) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights=weights or dict(_EQUAL_WEIGHTS),
        initial_nav=initial_nav,
        monthly_contribution=contribution,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=leaps_config,
    )


def _make_return_data(
    n: int = 504,
    daily_ret: float = 0.0003,
    daily_vol: float = 0.01,
    seed: int = 42,
    start: str = "2015-01-02",
) -> ReturnData:
    """Synthetic ReturnData for 6 assets."""
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
        tickers=_TICKERS,
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
    )
    return build_return_data(pd_obj, apply_tey=False)


# ---------------------------------------------------------------------------
# get_rebalance_dates
# ---------------------------------------------------------------------------


def test_rebalance_dates_are_in_index() -> None:
    """Every rebalance date falls within the provided index."""
    idx = pd.bdate_range("2015-01-02", periods=504)
    dates = get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    for d in dates:
        assert d in idx


def test_rebalance_dates_in_quarter_end_months() -> None:
    """All rebalance dates land in Mar / Jun / Sep / Dec."""
    idx = pd.bdate_range("2015-01-02", periods=1008)
    dates = get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    for d in dates:
        assert d.month in _QUARTER_END_MONTHS


def test_rebalance_dates_are_last_day_of_month() -> None:
    """Each rebalance date is the last trading day of its month."""
    idx = pd.bdate_range("2015-01-02", periods=1008)
    dates = get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    for d in dates:
        later_in_month = idx[(idx.month == d.month) & (idx.year == d.year) & (idx > d)]
        assert len(later_in_month) == 0


def test_rebalance_dates_count_roughly_four_per_year() -> None:
    """For a 2-year window we get exactly 8 quarterly dates."""
    idx = pd.bdate_range("2015-01-02", "2016-12-31")
    dates = get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    assert len(dates) == 8


def test_rebalance_dates_sorted() -> None:
    """Returned list is chronologically sorted."""
    idx = pd.bdate_range("2015-01-02", periods=504)
    dates = get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# compute_target_weights
# ---------------------------------------------------------------------------


def test_compute_target_weights_user_specified_sums_to_one() -> None:
    """USER_SPECIFIED weights normalize to sum = 1.0."""
    cfg = _config(weights={"A": 2.0, "B": 2.0, "C": 1.0})
    current = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
    w = compute_target_weights(cfg, current, 1_000.0, pd.Timestamp("2020-01-02"))
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_compute_target_weights_user_specified_proportions() -> None:
    """USER_SPECIFIED weights preserve proportions of target_weights dict."""
    cfg = _config(weights={"A": 3.0, "B": 1.0})
    current = pd.Series({"A": 0.5, "B": 0.5})
    w = compute_target_weights(cfg, current, 1_000.0, pd.Timestamp("2020-01-02"))
    assert w["A"] == pytest.approx(0.75, abs=1e-9)
    assert w["B"] == pytest.approx(0.25, abs=1e-9)


# ---------------------------------------------------------------------------
# apply_contribution
# ---------------------------------------------------------------------------


def test_apply_contribution_total_equals_contribution() -> None:
    """Sum of allocated amounts equals the contribution."""
    weights = pd.Series({"A": 0.6, "B": 0.4})
    alloc = apply_contribution(nav=500_000.0, contribution=10_000.0, weights=weights)
    assert sum(alloc.values()) == pytest.approx(10_000.0, rel=1e-9)


def test_apply_contribution_proportional_to_weights() -> None:
    """Each asset receives weight[a] * contribution."""
    weights = pd.Series({"A": 0.7, "B": 0.3})
    alloc = apply_contribution(nav=1_000.0, contribution=5_000.0, weights=weights)
    assert alloc["A"] == pytest.approx(3_500.0, rel=1e-9)
    assert alloc["B"] == pytest.approx(1_500.0, rel=1e-9)


def test_apply_contribution_zero_contribution() -> None:
    """Zero contribution allocates zero to every asset."""
    weights = pd.Series({"A": 0.5, "B": 0.5})
    alloc = apply_contribution(nav=1_000.0, contribution=0.0, weights=weights)
    assert sum(alloc.values()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_backtest — basic structure
# ---------------------------------------------------------------------------


def test_run_backtest_returns_correct_type() -> None:
    """run_backtest returns a frozen BacktestResult."""
    rd = _make_return_data(252)
    result = run_backtest(rd, _config())
    assert isinstance(result, BacktestResult)
    with pytest.raises((AttributeError, TypeError)):
        result.config = _config()  # type: ignore[misc]


def test_run_backtest_nav_series_length() -> None:
    """NAV series has same length as return series."""
    rd = _make_return_data(252)
    result = run_backtest(rd, _config())
    assert len(result.nav_series) == len(rd.returns)


def test_run_backtest_weight_history_shape() -> None:
    """Weight history has shape (n_days, n_assets)."""
    rd = _make_return_data(252)
    result = run_backtest(rd, _config())
    assert result.weight_history.shape == (len(rd.returns), len(_TICKERS))


def test_run_backtest_return_series_length() -> None:
    """Return series has same length as return data."""
    rd = _make_return_data(252)
    result = run_backtest(rd, _config())
    assert len(result.return_series) == len(rd.returns)


def test_run_backtest_nav_positive() -> None:
    """NAV stays positive throughout the backtest."""
    rd = _make_return_data(504)
    result = run_backtest(rd, _config())
    assert (result.nav_series > 0).all()


# ---------------------------------------------------------------------------
# run_backtest — NAV math
# ---------------------------------------------------------------------------


def test_run_backtest_nav_starts_near_initial() -> None:
    """After day 1, NAV is initial_nav * (1 + first_day_return)."""
    rd = _make_return_data(252)
    cfg = _config(initial_nav=1_000_000.0, contribution=0.0)
    result = run_backtest(rd, cfg)
    first_ret = float(rd.returns.iloc[0].mean())  # equal weight
    expected = 1_000_000.0 * (1.0 + first_ret)
    assert result.nav_series.iloc[0] == pytest.approx(expected, rel=1e-6)


def test_run_backtest_flat_returns_nav_is_constant() -> None:
    """With zero returns, no contributions, and no rebalancing effect, NAV is constant."""
    n = 252
    idx = pd.bdate_range("2015-01-02", periods=n)
    returns = pd.DataFrame(0.0, index=idx, columns=list(_TICKERS))
    log_ret = pd.DataFrame(0.0, index=idx, columns=list(_TICKERS))
    rd = ReturnData(
        returns=returns,
        log_returns=log_ret,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=pd.Series(0.0, index=returns.index, name="risk_free_rate"),
    )
    cfg = _config(initial_nav=500_000.0, contribution=0.0)
    result = run_backtest(rd, cfg)
    assert result.nav_series.iloc[-1] == pytest.approx(500_000.0, rel=1e-9)


def test_run_backtest_contribution_grows_nav() -> None:
    """Monthly contributions increase NAV beyond what returns alone would produce."""
    rd = _make_return_data(504)
    cfg_no_contrib = _config(initial_nav=1_000_000.0, contribution=0.0)
    cfg_with_contrib = _config(initial_nav=1_000_000.0, contribution=10_000.0)
    result_no = run_backtest(rd, cfg_no_contrib)
    result_yes = run_backtest(rd, cfg_with_contrib)
    assert result_yes.nav_series.iloc[-1] > result_no.nav_series.iloc[-1]


def test_run_backtest_no_contribution_nav_from_returns() -> None:
    """Without contributions, final NAV equals initial_nav * cumulative growth."""
    n = 100
    idx = pd.bdate_range("2015-01-02", periods=n)
    # All assets have the same constant daily return so we can compute exactly
    r = 0.001
    returns = pd.DataFrame(r, index=idx, columns=list(_TICKERS))
    log_ret = pd.DataFrame(r, index=idx, columns=list(_TICKERS))
    rd = ReturnData(
        returns=returns,
        log_returns=log_ret,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=pd.Series(0.0, index=returns.index, name="risk_free_rate"),
    )
    cfg = _config(initial_nav=100_000.0, contribution=0.0)
    result = run_backtest(rd, cfg)
    expected = 100_000.0 * (1.0 + r) ** n
    assert result.nav_series.iloc[-1] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# run_backtest — weight drift and rebalancing
# ---------------------------------------------------------------------------


def test_run_backtest_weights_sum_to_one_each_day() -> None:
    """Realized weights sum to 1.0 on every trading day."""
    rd = _make_return_data(252)
    result = run_backtest(rd, _config())
    sums = result.weight_history.sum(axis=1)
    assert (sums - 1.0).abs().max() < 1e-9


def test_run_backtest_weights_drift_between_rebalances() -> None:
    """Weights are not perfectly equal every day (drift before rebalance)."""
    rd = _make_return_data(252)
    result = run_backtest(rd, _config())
    # Pick the middle of a quarter — weights should have drifted from 1/6
    mid = result.weight_history.iloc[30]
    max_dev = (mid - 1.0 / len(_TICKERS)).abs().max()
    assert max_dev > 1e-6  # some drift has occurred


def test_run_backtest_weights_snapped_on_rebalance_date() -> None:
    """On each quarterly rebalance date, weights are close to target."""
    rd = _make_return_data(504)
    cfg = _config()
    result = run_backtest(rd, cfg)
    idx = pd.DatetimeIndex(rd.returns.index)
    rdates = get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    tol = 1e-6
    for d in rdates:
        if d in result.weight_history.index:
            row = result.weight_history.loc[d]
            max_dev = (row - 1.0 / len(_TICKERS)).abs().max()
            assert max_dev < tol, f"Weights not snapped on {d}: max_dev={max_dev}"


# ---------------------------------------------------------------------------
# run_backtest — missing asset
# ---------------------------------------------------------------------------


def test_run_backtest_raises_on_missing_asset() -> None:
    """ValueError if a target_weights asset is absent from return_data."""
    rd = _make_return_data(100)
    cfg = _config(weights={"VTI": 0.5, "NONEXISTENT": 0.5})
    with pytest.raises(ValueError, match="missing from return_data"):
        run_backtest(rd, cfg)


# ---------------------------------------------------------------------------
# run_backtest — LEAPS overlay
# ---------------------------------------------------------------------------


def test_run_backtest_with_leaps_returns_ledger() -> None:
    """BacktestResult.leaps_ledger is not None when a ledger is passed."""
    from finance.leverage import run_leaps_simulation
    rd = _make_return_data(504)
    vti_prices = 200.0 * (1.0 + rd.returns["VTI"]).cumprod()
    leaps_cfg = LeapsConfig(account_type=AccountType.TAXABLE)
    ledger = run_leaps_simulation(vti_prices, 5_000.0, leaps_cfg)
    cfg = _config(leaps_config=leaps_cfg)
    result = run_backtest(rd, cfg, leaps_ledger=ledger)
    assert result.leaps_ledger is ledger


def test_run_backtest_no_leaps_ledger_is_none() -> None:
    """BacktestResult.leaps_ledger is None when no ledger is passed."""
    rd = _make_return_data(252)
    result = run_backtest(rd, _config())
    assert result.leaps_ledger is None


# ---------------------------------------------------------------------------
# run_backtest — config is stored
# ---------------------------------------------------------------------------


def test_run_backtest_config_stored() -> None:
    """BacktestResult.config is the exact PortfolioConfig that was passed."""
    rd = _make_return_data(252)
    cfg = _config()
    result = run_backtest(rd, cfg)
    assert result.config is cfg
