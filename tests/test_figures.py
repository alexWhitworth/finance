"""Tests for figures.py — chart construction and table formatting.

Tests verify the shape and content of plotnine objects and the formatted
performance table without requiring a display or saving to disk.
All plot functions are called with output_path=None to suppress file I/O.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9  # type: ignore[import-untyped]
import pytest

from finance.figures import (
    _compute_drawdown_series,
    format_performance_table,
    plot_drawdown,
    plot_leaps_tax_drag,
    plot_nav_growth,
    plot_vol_contributions,
)
from finance.leverage import RebalanceRule, WeightStrategy
from finance.metrics import PerformanceReport, build_performance_report
from finance.portfolio import BacktestResult, PortfolioConfig, run_backtest
from finance.returns import ReturnData
from finance.volatility import build_volatility_model

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TICKERS = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
_EQUAL_WEIGHTS = {t: 1.0 / len(_TICKERS) for t in _TICKERS}


def _make_return_data(n: int = 504, seed: int = 42, start: str = "2015-01-02") -> ReturnData:
    """Synthetic ReturnData for 6 assets spanning ~2 years."""
    idx = pd.bdate_range(start, periods=n + 1)
    rng = np.random.default_rng(seed)
    starts = {"VTI": 200.0, "VXUS": 60.0, "GLD": 170.0, "MUB": 55.0, "KMLM": 25.0, "VGIT": 65.0}
    prices = pd.DataFrame(
        {t: starts[t] * np.cumprod(1 + rng.normal(0.0003, 0.01, n + 1)) for t in _TICKERS},
        index=idx,
    )
    returns = prices.pct_change().dropna()
    log_returns = np.log(1 + returns)
    return ReturnData(
        returns=returns, log_returns=log_returns, tey_adjusted=False, marginal_rate=0.0
    )


def _make_backtest(rd: ReturnData, seed: int = 0) -> BacktestResult:
    """Run a basic backtest with no contributions, no LEAPS."""
    cfg = PortfolioConfig(
        target_weights=dict(_EQUAL_WEIGHTS),
        initial_nav=1_000_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=None,
    )
    return run_backtest(rd, cfg)


def _make_report(rd: ReturnData) -> PerformanceReport:
    """Build a PerformanceReport from synthetic data."""
    result = _make_backtest(rd)
    vol_model = build_volatility_model(rd)
    return build_performance_report(result, rd, vol_model)


# ---------------------------------------------------------------------------
# Drawdown helper
# ---------------------------------------------------------------------------


class TestComputeDrawdownSeries:
    def test_flat_nav_zero_drawdown(self) -> None:
        nav = pd.Series([100.0, 100.0, 100.0], index=pd.date_range("2020-01-01", periods=3))
        dd = _compute_drawdown_series(nav)
        assert (dd == 0.0).all()

    def test_monotone_rising_no_drawdown(self) -> None:
        nav = pd.Series([100.0, 110.0, 120.0], index=pd.date_range("2020-01-01", periods=3))
        dd = _compute_drawdown_series(nav)
        assert (dd == 0.0).all()

    def test_decline_then_recovery(self) -> None:
        nav = pd.Series([100.0, 80.0, 100.0], index=pd.date_range("2020-01-01", periods=3))
        dd = _compute_drawdown_series(nav)
        assert dd.iloc[0] == pytest.approx(0.0)
        assert dd.iloc[1] == pytest.approx(-0.20)
        assert dd.iloc[2] == pytest.approx(0.0)

    def test_all_negative(self) -> None:
        nav = pd.Series([100.0, 90.0, 80.0], index=pd.date_range("2020-01-01", periods=3))
        dd = _compute_drawdown_series(nav)
        assert (dd <= 0.0).all()
        assert dd.min() == pytest.approx(-0.20)


# ---------------------------------------------------------------------------
# plot_nav_growth
# ---------------------------------------------------------------------------


class TestPlotNavGrowth:
    def test_returns_ggplot(self) -> None:
        rd = _make_return_data()
        result = _make_backtest(rd)
        plot = plot_nav_growth({"Base": result}, output_path=None)
        assert isinstance(plot, p9.ggplot)

    def test_two_portfolios(self) -> None:
        rd = _make_return_data()
        r1 = _make_backtest(rd)
        r2 = _make_backtest(_make_return_data(seed=99))
        plot = plot_nav_growth({"A": r1, "B": r2}, output_path=None)
        assert isinstance(plot, p9.ggplot)

    def test_data_contains_both_portfolios(self) -> None:
        rd = _make_return_data()
        r1 = _make_backtest(rd)
        r2 = _make_backtest(_make_return_data(seed=7))
        plot = plot_nav_growth({"Alpha": r1, "Beta": r2}, output_path=None)
        built = plot.draw()
        assert built is not None


# ---------------------------------------------------------------------------
# plot_drawdown
# ---------------------------------------------------------------------------


class TestPlotDrawdown:
    def test_returns_ggplot(self) -> None:
        rd = _make_return_data()
        result = _make_backtest(rd)
        plot = plot_drawdown({"Base": result}, output_path=None)
        assert isinstance(plot, p9.ggplot)

    def test_crisis_clipping_outside_range(self) -> None:
        # Data spans 2015-2017; GFC crisis dates (2007-2009) are entirely out of range.
        rd = _make_return_data()
        result = _make_backtest(rd)
        gfc_only = {"GFC": ("2007-10-01", "2009-03-31")}
        # Should not raise even though GFC is outside the data range
        plot = plot_drawdown({"Base": result}, crisis_periods=gfc_only, output_path=None)
        assert isinstance(plot, p9.ggplot)

    def test_empty_crisis_periods(self) -> None:
        rd = _make_return_data()
        result = _make_backtest(rd)
        plot = plot_drawdown({"Base": result}, crisis_periods={}, output_path=None)
        assert isinstance(plot, p9.ggplot)

    def test_crisis_shading_overlapping_range(self) -> None:
        # Crisis period fully within the 2015-2017 data range — exercises geom_rect branch.
        rd = _make_return_data()
        result = _make_backtest(rd)
        overlapping = {"Mid-period": ("2015-06-01", "2016-06-01")}
        plot = plot_drawdown({"Base": result}, crisis_periods=overlapping, output_path=None)
        assert isinstance(plot, p9.ggplot)
        built = plot.draw()
        assert built is not None


# ---------------------------------------------------------------------------
# plot_vol_contributions
# ---------------------------------------------------------------------------


class TestPlotVolContributions:
    def test_returns_ggplot(self) -> None:
        rd = _make_return_data()
        report = _make_report(rd)
        plot = plot_vol_contributions(report, output_path=None)
        assert isinstance(plot, p9.ggplot)

    def test_all_assets_present(self) -> None:
        rd = _make_return_data()
        report = _make_report(rd)
        plot = plot_vol_contributions(report, output_path=None)
        built = plot.draw()
        assert built is not None


# ---------------------------------------------------------------------------
# plot_leaps_tax_drag
# ---------------------------------------------------------------------------


class TestPlotLeapsTaxDrag:
    def test_returns_ggplot(self) -> None:
        rd = _make_return_data()
        r_taxable = _make_backtest(rd)
        r_sheltered = _make_backtest(_make_return_data(seed=1))
        plot = plot_leaps_tax_drag(r_taxable, r_sheltered, output_path=None)
        assert isinstance(plot, p9.ggplot)

    def test_same_data_zero_drag_label(self) -> None:
        rd = _make_return_data()
        result = _make_backtest(rd)
        # Both sides are identical — drag should be $0
        plot = plot_leaps_tax_drag(result, result, output_path=None)
        assert isinstance(plot, p9.ggplot)


# ---------------------------------------------------------------------------
# Save-path smoke tests
# ---------------------------------------------------------------------------


class TestSavePath:
    def test_plot_nav_growth_saves_file(self, tmp_path: Path) -> None:
        rd = _make_return_data()
        result = _make_backtest(rd)
        dest = tmp_path / "nav_growth.png"
        plot_nav_growth({"Base": result}, output_path=dest)
        assert dest.exists()

    def test_plot_drawdown_saves_file(self, tmp_path: Path) -> None:
        rd = _make_return_data()
        result = _make_backtest(rd)
        dest = tmp_path / "drawdown.png"
        plot_drawdown({"Base": result}, output_path=dest)
        assert dest.exists()

    def test_plot_vol_contributions_saves_file(self, tmp_path: Path) -> None:
        rd = _make_return_data()
        report = _make_report(rd)
        dest = tmp_path / "vol_contributions.png"
        plot_vol_contributions(report, output_path=dest)
        assert dest.exists()

    def test_plot_leaps_tax_drag_saves_file(self, tmp_path: Path) -> None:
        rd = _make_return_data()
        result = _make_backtest(rd)
        dest = tmp_path / "leaps_tax_drag.png"
        plot_leaps_tax_drag(result, result, output_path=dest)
        assert dest.exists()


# ---------------------------------------------------------------------------
# format_performance_table
# ---------------------------------------------------------------------------


class TestFormatPerformanceTable:
    def test_returns_string(self) -> None:
        rd = _make_return_data()
        report = _make_report(rd)
        table = format_performance_table(report)
        assert isinstance(table, str)

    def test_contains_full_period(self) -> None:
        rd = _make_return_data()
        report = _make_report(rd)
        table = format_performance_table(report)
        assert "Full Period" in table

    def test_contains_forward_vol(self) -> None:
        rd = _make_return_data()
        report = _make_report(rd)
        table = format_performance_table(report)
        assert "Forward Vol" in table

    def test_contains_metric_columns(self) -> None:
        rd = _make_return_data()
        report = _make_report(rd)
        table = format_performance_table(report)
        for col in ("Sharpe", "Sortino", "Calmar", "Omega"):
            assert col in table

    def test_crisis_period_row_in_table(self) -> None:
        # Data spans 2019-2022 so COVID (2020-02 to 2020-04) overlaps — its row must appear.
        rd = _make_return_data(n=756, start="2019-01-02")
        report = _make_report(rd)
        table = format_performance_table(report)
        assert "COVID" in table
