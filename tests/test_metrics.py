"""Tests for metrics.py — performance ratios, period slicing, and report assembly."""

import numpy as np
import pandas as pd
import pytest

from finance.data import PriceData
from finance.leverage import RebalanceRule, WeightStrategy
from finance.metrics import (
    TRADING_DAYS_PER_YEAR,
    PerformanceMetrics,
    PerformanceReport,
    annualized_return,
    annualized_std,
    build_performance_report,
    calmar_ratio,
    compute_metrics,
    max_drawdown,
    omega_ratio,
    return_excess_kurtosis,
    return_skewness,
    sharpe_ratio,
    slice_period,
    sortino_ratio,
)
from finance.portfolio import BacktestResult, PortfolioConfig
from finance.returns import ReturnData, build_return_data
from finance.volatility import build_volatility_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TICKERS = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
_WEIGHTS = {t: 1.0 / len(_TICKERS) for t in _TICKERS}


def _bdate_range(n: int, start: str = "2010-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _flat_returns(n: int = 252, daily_ret: float = 0.0004) -> pd.Series:
    """Constant-return series (easy to verify analytically)."""
    idx = _bdate_range(n)
    return pd.Series(daily_ret, index=idx)


def _nav_from_returns(returns: pd.Series, initial: float = 1000.0) -> pd.Series:
    return initial * (1.0 + returns).cumprod()


def _zero_rfr(returns: pd.Series) -> pd.Series:
    """Zero risk-free rate Series aligned to returns index."""
    return pd.Series(0.0, index=returns.index)


def _make_backtest_result(n: int = 504) -> BacktestResult:
    """Synthetic BacktestResult with 6 equal-weight assets over n trading days."""
    rng = np.random.default_rng(42)
    idx = _bdate_range(n)
    tickers = list(_TICKERS)
    returns_data = pd.DataFrame(
        {t: rng.normal(0.0004, 0.01, n) for t in tickers}, index=idx
    )
    port_returns = returns_data.mean(axis=1)
    nav = _nav_from_returns(port_returns)
    weights_df = pd.DataFrame(
        {t: 1.0 / len(tickers) for t in tickers}, index=idx
    )
    config = PortfolioConfig(
        target_weights=_WEIGHTS,
        initial_nav=1_000_000.0,
        monthly_contribution=10_000.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )
    return BacktestResult(
        nav_series=nav,
        weight_history=weights_df,
        return_series=port_returns,
        leaps_ledger=None,
        config=config,
    )


def _make_price_data(n: int = 505) -> PriceData:
    idx = pd.bdate_range("2010-01-04", periods=n)
    rng = np.random.default_rng(99)
    starts = {"VTI": 200.0, "VXUS": 60.0, "GLD": 170.0, "MUB": 55.0, "KMLM": 25.0, "VGIT": 65.0}
    prices = pd.DataFrame(
        {t: starts[t] * np.cumprod(1 + rng.normal(0.0003, 0.01, n)) for t in _TICKERS},
        index=idx,
    )
    dividends = pd.DataFrame(0.0, index=idx, columns=list(_TICKERS))
    return PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=pd.DataFrame(),
        tickers=_TICKERS,
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
    )


def _make_return_data(n: int = 504) -> ReturnData:
    idx = _bdate_range(n + 1)
    rng = np.random.default_rng(7)
    starts = {"VTI": 200.0, "VXUS": 60.0, "GLD": 170.0, "MUB": 55.0, "KMLM": 25.0, "VGIT": 65.0}
    prices_data = {
        t: starts[t] * np.cumprod(1 + rng.normal(0.0003, 0.01, n + 1))
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
    return build_return_data(pd_obj, apply_tey=False)


# ---------------------------------------------------------------------------
# annualized_return
# ---------------------------------------------------------------------------


def test_annualized_return_flat() -> None:
    """Flat daily return computes correct geometric annualization."""
    daily = 0.0004
    r = _flat_returns(TRADING_DAYS_PER_YEAR, daily)
    expected = (1.0 + daily) ** TRADING_DAYS_PER_YEAR - 1.0
    assert annualized_return(r) == pytest.approx(expected, rel=1e-9)


def test_annualized_return_empty() -> None:
    """Empty series returns 0.0."""
    assert annualized_return(pd.Series(dtype=float)) == 0.0


def test_annualized_return_half_year() -> None:
    """252/2 observations → result scales correctly by years = 0.5."""
    n = TRADING_DAYS_PER_YEAR // 2
    daily = 0.001
    r = _flat_returns(n, daily)
    assert annualized_return(r) > 0.0


def test_annualized_return_negative() -> None:
    """Negative daily returns give negative annualized return."""
    r = _flat_returns(TRADING_DAYS_PER_YEAR, -0.0004)
    assert annualized_return(r) < 0.0


# ---------------------------------------------------------------------------
# annualized_std
# ---------------------------------------------------------------------------


def test_annualized_std_known_value() -> None:
    """Std of constant series is 0."""
    r = _flat_returns(252, 0.001)
    assert annualized_std(r) == pytest.approx(0.0, abs=1e-12)


def test_annualized_std_iid() -> None:
    """i.i.d. series with daily vol v → annualized std ≈ v * sqrt(252)."""
    daily_vol = 0.01
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, daily_vol, 2000))
    result = annualized_std(r)
    expected = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert result == pytest.approx(expected, rel=0.05)


def test_annualized_std_single_obs() -> None:
    """Single observation returns 0.0."""
    assert annualized_std(pd.Series([0.01])) == 0.0


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_monotone_increasing() -> None:
    """Monotonically increasing NAV has 0 drawdown."""
    nav = pd.Series([100.0, 101.0, 102.0, 103.0])
    assert max_drawdown(nav) == pytest.approx(0.0)


def test_max_drawdown_known_value() -> None:
    """NAV that drops 50% then recovers has max drawdown = 0.5."""
    nav = pd.Series([100.0, 80.0, 50.0, 60.0, 90.0])
    assert max_drawdown(nav) == pytest.approx(0.5, rel=1e-9)


def test_max_drawdown_all_equal() -> None:
    """Flat NAV has 0 drawdown."""
    nav = pd.Series([100.0] * 20)
    assert max_drawdown(nav) == pytest.approx(0.0)


def test_max_drawdown_single_obs() -> None:
    """Single observation returns 0.0."""
    assert max_drawdown(pd.Series([100.0])) == 0.0


def test_max_drawdown_positive() -> None:
    """Result is always non-negative."""
    rng = np.random.default_rng(1)
    nav = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, 500)))
    assert max_drawdown(nav) >= 0.0


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


def test_sharpe_ratio_positive_mean() -> None:
    """Positive mean excess return with variance → positive Sharpe."""
    rng = np.random.default_rng(99)
    r = pd.Series(rng.normal(0.001, 0.01, 252))
    assert sharpe_ratio(r, _zero_rfr(r)) > 0.0


def test_sharpe_ratio_zero_std_returns_zero() -> None:
    """When excess return std is zero, returns 0.0."""
    r = pd.Series([0.001] * 252)
    rfr = pd.Series(0.001 * TRADING_DAYS_PER_YEAR, index=r.index)
    assert sharpe_ratio(r, rfr) == 0.0


def test_sharpe_ratio_iid_known() -> None:
    """i.i.d. returns with mean mu, std sigma → Sharpe ≈ mu/sigma * sqrt(252)."""
    mu = 0.001
    sigma = 0.01
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(mu, sigma, 5000))
    result = sharpe_ratio(r, _zero_rfr(r))
    expected = (mu / sigma) * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert result == pytest.approx(expected, rel=0.10)


def test_sharpe_ratio_negative_mean() -> None:
    """Negative mean returns produce negative Sharpe."""
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(-0.001, 0.01, 500))
    assert sharpe_ratio(r, _zero_rfr(r)) < 0.0


def test_sharpe_ratio_single_obs_returns_zero() -> None:
    r = pd.Series([0.01])
    assert sharpe_ratio(r, _zero_rfr(r)) == 0.0


def test_sharpe_ratio_nonzero_rfr_reduces_value() -> None:
    """A positive RFR lowers Sharpe vs. zero RFR for positive-return series."""
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    rfr_high = pd.Series(0.05, index=r.index)
    s_zero = sharpe_ratio(r, _zero_rfr(r))
    s_high = sharpe_ratio(r, rfr_high)
    assert s_zero > s_high


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------


def test_sortino_ratio_no_downside() -> None:
    """All-positive excess returns → infinite Sortino."""
    r = pd.Series([0.01, 0.02, 0.005, 0.03])
    assert sortino_ratio(r, _zero_rfr(r)) == float("inf")


def test_sortino_ratio_greater_than_sharpe_for_positive_skew() -> None:
    """With only upside volatility, Sortino > Sharpe for same mean."""
    rng = np.random.default_rng(5)
    r = pd.Series(np.clip(rng.normal(0.001, 0.01, 1000), -0.005, None))
    rfr = _zero_rfr(r)
    s = sharpe_ratio(r, rfr)
    so = sortino_ratio(r, rfr)
    assert so > s


def test_sortino_ratio_single_obs_returns_zero() -> None:
    r = pd.Series([0.01])
    assert sortino_ratio(r, _zero_rfr(r)) == 0.0


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------


def test_calmar_ratio_zero_drawdown() -> None:
    """Zero drawdown → calmar is inf."""
    r = _flat_returns(252, 0.001)
    nav = _nav_from_returns(r)
    assert calmar_ratio(r, nav) == float("inf")


def test_calmar_ratio_known_value() -> None:
    """Calmar = annualized_return / max_drawdown."""
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0.0005, 0.01, 504))
    nav = _nav_from_returns(r)
    expected = annualized_return(r) / max_drawdown(nav)
    assert calmar_ratio(r, nav) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# omega_ratio
# ---------------------------------------------------------------------------


def test_omega_ratio_all_positive() -> None:
    """All excess returns above zero → huge Omega."""
    r = pd.Series([0.01, 0.02, 0.03])
    result = omega_ratio(r, _zero_rfr(r))
    assert result > 1e10


def test_omega_ratio_all_negative() -> None:
    """All excess returns below zero → Omega < 1."""
    r = pd.Series([-0.01, -0.02, -0.03])
    result = omega_ratio(r, _zero_rfr(r))
    assert result < 1.0


def test_omega_ratio_equal_gains_losses() -> None:
    """Equal gains and losses → Omega ≈ 1.0."""
    r = pd.Series([0.01, -0.01, 0.01, -0.01])
    result = omega_ratio(r, _zero_rfr(r))
    assert result == pytest.approx(1.0, abs=1e-6)


def test_omega_ratio_empty() -> None:
    """Empty series returns 0.0."""
    empty: pd.Series = pd.Series(dtype=float)
    assert omega_ratio(empty, empty) == 0.0


# ---------------------------------------------------------------------------
# return_skewness
# ---------------------------------------------------------------------------


def test_return_skewness_few_obs_returns_zero() -> None:
    """Fewer than 4 observations → 0.0."""
    assert return_skewness(pd.Series([0.01, -0.01, 0.02])) == 0.0


def test_return_skewness_symmetric_near_zero() -> None:
    """Symmetric i.i.d. series has near-zero skewness."""
    rng = np.random.default_rng(20)
    r = pd.Series(rng.normal(0, 0.01, 2000))
    assert abs(return_skewness(r)) < 0.15


def test_return_skewness_right_skewed_positive() -> None:
    """Right-skewed series has positive skewness."""
    rng = np.random.default_rng(21)
    base = rng.normal(0.001, 0.01, 2000)
    spikes = rng.choice(len(base), size=30, replace=False)
    base[spikes] += 0.10
    r = pd.Series(base)
    assert return_skewness(r) > 0.0


# ---------------------------------------------------------------------------
# return_excess_kurtosis
# ---------------------------------------------------------------------------


def test_return_excess_kurtosis_few_obs_returns_zero() -> None:
    """Fewer than 4 observations → 0.0."""
    assert return_excess_kurtosis(pd.Series([0.01, -0.01, 0.02])) == 0.0


def test_return_excess_kurtosis_normal_near_zero() -> None:
    """Normal distribution has excess kurtosis ≈ 0."""
    rng = np.random.default_rng(22)
    r = pd.Series(rng.normal(0, 0.01, 5000))
    assert abs(return_excess_kurtosis(r)) < 0.3


def test_return_excess_kurtosis_fat_tails_positive() -> None:
    """Fat-tailed distribution has positive excess kurtosis."""
    rng = np.random.default_rng(23)
    r = pd.Series(rng.standard_t(df=4, size=5000) * 0.01)
    assert return_excess_kurtosis(r) > 0.0


# ---------------------------------------------------------------------------
# slice_period
# ---------------------------------------------------------------------------


def test_slice_period_within_range() -> None:
    """Slice returns only observations in [start, end]."""
    idx = pd.date_range("2020-01-01", periods=365, freq="D")
    r = pd.Series(0.001, index=idx)
    sliced = slice_period(r, "2020-03-01", "2020-03-31")
    assert sliced.index.min() >= pd.Timestamp("2020-03-01")
    assert sliced.index.max() <= pd.Timestamp("2020-03-31")


def test_slice_period_outside_range_is_empty() -> None:
    """Slice outside series range returns empty Series."""
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    r = pd.Series(0.001, index=idx)
    sliced = slice_period(r, "2025-01-01", "2025-06-01")
    assert len(sliced) == 0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_dataclass() -> None:
    """compute_metrics returns a frozen PerformanceMetrics."""
    r = _flat_returns(252, 0.0005)
    nav = _nav_from_returns(r)
    m = compute_metrics(r, nav, "Test", _zero_rfr(r))
    assert isinstance(m, PerformanceMetrics)
    with pytest.raises((AttributeError, TypeError)):
        m.sharpe = 0.0  # type: ignore[misc]


def test_compute_metrics_period_label() -> None:
    """period_label is stored correctly."""
    r = _flat_returns(252, 0.0005)
    nav = _nav_from_returns(r)
    m = compute_metrics(r, nav, "Full Period", _zero_rfr(r))
    assert m.period_label == "Full Period"


def test_compute_metrics_internal_consistency() -> None:
    """Each field matches calling the individual function directly."""
    rng = np.random.default_rng(10)
    r = pd.Series(rng.normal(0.0005, 0.01, 504))
    nav = _nav_from_returns(r)
    rfr = _zero_rfr(r)
    m = compute_metrics(r, nav, "Check", rfr)
    assert m.annualized_return == pytest.approx(annualized_return(r), rel=1e-9)
    assert m.annualized_std == pytest.approx(annualized_std(r), rel=1e-9)
    assert m.max_drawdown == pytest.approx(max_drawdown(nav), rel=1e-9)
    assert m.sharpe == pytest.approx(sharpe_ratio(r, rfr), rel=1e-9)
    assert m.sortino == pytest.approx(sortino_ratio(r, rfr), rel=1e-9)
    assert m.omega == pytest.approx(omega_ratio(r, rfr), rel=1e-9)


def test_compute_metrics_has_skewness_and_kurtosis() -> None:
    """skewness and excess_kurtosis fields are present and finite."""
    rng = np.random.default_rng(12)
    r = pd.Series(rng.normal(0.0005, 0.01, 504))
    nav = _nav_from_returns(r)
    m = compute_metrics(r, nav, "Test", _zero_rfr(r))
    assert isinstance(m.skewness, float)
    assert isinstance(m.excess_kurtosis, float)
    assert np.isfinite(m.skewness)
    assert np.isfinite(m.excess_kurtosis)


def test_compute_metrics_skewness_consistent_with_function() -> None:
    """skewness in metrics matches return_skewness(excess) directly."""
    rng = np.random.default_rng(13)
    r = pd.Series(rng.normal(0.0005, 0.01, 504))
    nav = _nav_from_returns(r)
    rfr = pd.Series(0.03, index=r.index)
    m = compute_metrics(r, nav, "Test", rfr)
    excess = r - rfr.reindex(r.index, method="ffill").fillna(0.0) / TRADING_DAYS_PER_YEAR
    assert m.skewness == pytest.approx(return_skewness(excess), rel=1e-9)
    assert m.excess_kurtosis == pytest.approx(return_excess_kurtosis(excess), rel=1e-9)


# ---------------------------------------------------------------------------
# build_performance_report
# ---------------------------------------------------------------------------


def test_build_performance_report_structure() -> None:
    """build_performance_report returns a frozen PerformanceReport."""
    br = _make_backtest_result(504)
    pd_obj = _make_price_data(505)
    rd = _make_return_data(504)
    vm = build_volatility_model(rd)
    report = build_performance_report(br, pd_obj, rd, vm)
    assert isinstance(report, PerformanceReport)
    assert isinstance(report.full_period, PerformanceMetrics)
    assert isinstance(report.crisis_periods, tuple)
    assert isinstance(report.vol_contribution_table, pd.DataFrame)
    assert isinstance(report.forward_vol_forecast, float)


def test_build_performance_report_full_period_label() -> None:
    """Full period metrics are labelled 'Full Period'."""
    br = _make_backtest_result(504)
    pd_obj = _make_price_data(505)
    rd = _make_return_data(504)
    vm = build_volatility_model(rd)
    report = build_performance_report(br, pd_obj, rd, vm)
    assert report.full_period.period_label == "Full Period"


def test_build_performance_report_vol_table_columns() -> None:
    """Vol contribution table has expected columns."""
    br = _make_backtest_result(504)
    pd_obj = _make_price_data(505)
    rd = _make_return_data(504)
    vm = build_volatility_model(rd)
    report = build_performance_report(br, pd_obj, rd, vm)
    assert set(report.vol_contribution_table.columns) == {
        "sigma_tilde", "sigma_hat", "rho_VTI", "contrib"
    }


def test_build_performance_report_forward_vol_positive() -> None:
    """Forward vol forecast is strictly positive."""
    br = _make_backtest_result(504)
    pd_obj = _make_price_data(505)
    rd = _make_return_data(504)
    vm = build_volatility_model(rd)
    report = build_performance_report(br, pd_obj, rd, vm)
    assert report.forward_vol_forecast > 0.0


def test_build_performance_report_crisis_excluded_when_no_overlap() -> None:
    """Crisis periods that don't overlap the backtest window are excluded."""
    br = _make_backtest_result(504)
    pd_obj = _make_price_data(505)
    rd = _make_return_data(504)
    vm = build_volatility_model(rd)
    report = build_performance_report(br, pd_obj, rd, vm)
    labels = {m.period_label for m in report.crisis_periods}
    assert "GFC" not in labels


def test_build_performance_report_immutable() -> None:
    """PerformanceReport is frozen."""
    br = _make_backtest_result(504)
    pd_obj = _make_price_data(505)
    rd = _make_return_data(504)
    vm = build_volatility_model(rd)
    report = build_performance_report(br, pd_obj, rd, vm)
    with pytest.raises((AttributeError, TypeError)):
        report.forward_vol_forecast = 0.0  # type: ignore[misc]


def test_build_performance_report_terminal_nav_none() -> None:
    """terminal_nav is None (Phase F placeholder)."""
    br = _make_backtest_result(504)
    pd_obj = _make_price_data(505)
    rd = _make_return_data(504)
    vm = build_volatility_model(rd)
    report = build_performance_report(br, pd_obj, rd, vm)
    assert report.terminal_nav is None


def test_build_performance_report_tax_summary_none() -> None:
    """tax_summary is None (Phase F placeholder)."""
    br = _make_backtest_result(504)
    pd_obj = _make_price_data(505)
    rd = _make_return_data(504)
    vm = build_volatility_model(rd)
    report = build_performance_report(br, pd_obj, rd, vm)
    assert report.tax_summary is None
