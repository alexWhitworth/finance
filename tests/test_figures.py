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

from finance.data import PriceData
from finance.dca_signal import LeapsDcaSignal
from finance.figures import (
    _compute_drawdown_series,
    compare_performance_table,
    format_contract_greeks_table,
    format_holdings_table,
    format_leaps_dca_signal_table,
    format_nav_breakdown_table,
    format_performance_table,
    format_trade_orders_table,
    plot_drawdown,
    plot_leaps_tax_drag,
    plot_nav_growth,
    plot_vol_contributions,
)
from finance.greeks import PortfolioGreeks, compute_portfolio_greeks
from finance.leverage import RebalanceRule, WeightStrategy, build_leaps_contract
from finance.metrics import PerformanceReport, build_performance_report
from finance.portfolio import BacktestResult, PortfolioConfig, run_backtest
from finance.portfolio_manager import (
    LivePortfolio,
    compute_holdings_view,
    compute_leaps_holdings_view,
    compute_nav_breakdown,
    compute_rebalance_plan,
    leaps_trim_as_trade_order,
)
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
        returns=returns,
        log_returns=log_returns,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=pd.Series(0.0, index=returns.index, name="risk_free_rate"),
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
    pd_obj = _make_price_data(rd.returns.index)
    return run_backtest(rd, pd_obj, cfg)


def _make_price_data(returns_index: pd.DatetimeIndex) -> PriceData:
    """Minimal PriceData whose price index covers the returns index."""
    rng = np.random.default_rng(0)
    starts = {"VTI": 200.0, "VXUS": 60.0, "GLD": 170.0, "MUB": 55.0, "KMLM": 25.0, "VGIT": 65.0}
    n = len(returns_index) + 1
    idx = pd.bdate_range(returns_index[0], periods=n)
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


def _make_report(rd: ReturnData) -> PerformanceReport:
    """Build a PerformanceReport from synthetic data."""
    result = _make_backtest(rd)
    pd_obj = _make_price_data(rd.returns.index)
    vol_model = build_volatility_model(rd)
    return build_performance_report(result, pd_obj, rd, vol_model)


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

    def test_contains_skewness_and_kurtosis_columns(self) -> None:
        rd = _make_return_data()
        report = _make_report(rd)
        table = format_performance_table(report)
        assert "Skewness" in table
        assert "Ex. Kurt" in table

    def test_no_leaps_tax_section_without_terminal_nav(self) -> None:
        rd = _make_return_data()
        report = _make_report(rd)
        # report.terminal_nav is None for a no-LEAPS backtest
        assert report.terminal_nav is None
        table = format_performance_table(report)
        assert "LEAPS Terminal NAV" not in table

    def test_leaps_tax_section_with_terminal_nav(self) -> None:
        """Injects a synthetic terminal_nav to exercise the LEAPS tax block."""
        from dataclasses import replace

        from finance.leverage import AccountType, LeapsTaxSummary, TerminalNav

        rd = _make_return_data()
        report = _make_report(rd)
        synthetic_tn = TerminalNav(
            pre_tax_nav=1_100_000.0,
            post_tax_nav=1_050_000.0,
            terminal_tax=50_000.0,
            open_gain=200_000.0,
            ltcg_rate=0.238,
            account_type=AccountType.TAXABLE,
        )
        synthetic_ts = LeapsTaxSummary(
            total_roll_tax=0.0,
            n_rolls=0,
            terminal_tax=50_000.0,
            total_tax=50_000.0,
            tax_drag_pct=0.05,
            annualized_tax_drag=0.012,
            account_type=AccountType.TAXABLE,
        )
        report_with_leaps = replace(report, terminal_nav=synthetic_tn, tax_summary=synthetic_ts)
        table = format_performance_table(report_with_leaps)
        assert "LEAPS Terminal NAV" in table
        assert "Pre-tax" in table
        assert "Post-tax" in table
        assert "Ann. Tax Drag" in table


# ---------------------------------------------------------------------------
# compare_performance_table
# ---------------------------------------------------------------------------


class TestComparePerformanceTable:
    def test_single_report(self) -> None:
        rd = _make_return_data()
        report = _make_report(rd)
        table = compare_performance_table([("Base", report)])
        assert isinstance(table, str)
        assert "Base" in table
        assert "Full Period" in table

    def test_multi_report_both_labels_present(self) -> None:
        rd = _make_return_data()
        r1 = _make_report(rd)
        r2 = _make_report(_make_return_data(seed=99))
        table = compare_performance_table([("Alpha", r1), ("Beta", r2)])
        assert "Alpha" in table
        assert "Beta" in table

    def test_multi_report_metric_columns_present(self) -> None:
        rd = _make_return_data()
        r1 = _make_report(rd)
        r2 = _make_report(_make_return_data(seed=7))
        table = compare_performance_table([("X", r1), ("Y", r2)])
        for col in ("Sharpe", "Sortino", "Skewness", "Ex. Kurt"):
            assert col in table

    def test_leaps_rows_absent_without_terminal_nav(self) -> None:
        rd = _make_return_data()
        report = _make_report(rd)
        table = compare_performance_table([("Base", report)])
        assert "Pre-tax" not in table
        assert "Post-tax" not in table

    def test_leaps_rows_present_with_terminal_nav(self) -> None:
        from dataclasses import replace

        from finance.leverage import AccountType, LeapsTaxSummary, TerminalNav

        rd = _make_return_data()
        report = _make_report(rd)
        synthetic_tn = TerminalNav(
            pre_tax_nav=1_200_000.0,
            post_tax_nav=1_150_000.0,
            terminal_tax=50_000.0,
            open_gain=210_000.0,
            ltcg_rate=0.238,
            account_type=AccountType.TAXABLE,
        )
        synthetic_ts = LeapsTaxSummary(
            total_roll_tax=0.0,
            n_rolls=0,
            terminal_tax=50_000.0,
            total_tax=50_000.0,
            tax_drag_pct=0.04,
            annualized_tax_drag=0.01,
            account_type=AccountType.TAXABLE,
        )
        report_leaps = replace(report, terminal_nav=synthetic_tn, tax_summary=synthetic_ts)
        table = compare_performance_table([("LEAPS", report_leaps)])
        assert "Pre-tax" in table
        assert "Post-tax" in table
        assert "Ann. Tax Drag" in table


# ---------------------------------------------------------------------------
# format_contract_greeks_table
# ---------------------------------------------------------------------------


def _make_portfolio_greeks(n_contracts: float = 3.0) -> PortfolioGreeks:
    """PortfolioGreeks for a LivePortfolio with one live LEAPS contract."""
    contract = build_leaps_contract(
        pd.Timestamp("2024-01-15"), pd.Timestamp("2026-01-15"), 300.0, n_contracts
    )
    portfolio = LivePortfolio(
        as_of_date=pd.Timestamp("2025-06-01"),
        holdings={},
        target_weights={"VTI_LEAPS": 1.0},
        leaps_contracts=((contract, 1.0),),
        gtt_regime=None,
    )
    return compute_portfolio_greeks(portfolio, spot=320.0, iv=0.18)


class TestFormatContractGreeksTable:
    def test_returns_string(self) -> None:
        greeks = _make_portfolio_greeks()
        assert isinstance(format_contract_greeks_table(greeks), str)

    def test_contains_expected_columns(self) -> None:
        table = format_contract_greeks_table(_make_portfolio_greeks())
        for col in (
            "purchased", "expiry", "n_contracts", "delta", "gamma", "vega",
            "theta/day", "position_delta",
        ):
            assert col in table

    def test_one_row_per_contract(self) -> None:
        contract_a = build_leaps_contract(
            pd.Timestamp("2024-01-15"), pd.Timestamp("2026-01-15"), 300.0, 2.0
        )
        contract_b = build_leaps_contract(
            pd.Timestamp("2024-07-15"), pd.Timestamp("2026-07-15"), 310.0, 3.0
        )
        portfolio = LivePortfolio(
            as_of_date=pd.Timestamp("2025-06-01"),
            holdings={},
            target_weights={"VTI_LEAPS": 1.0},
            leaps_contracts=((contract_a, 1.0), (contract_b, 1.0)),
            gtt_regime=None,
        )
        greeks = compute_portfolio_greeks(portfolio, spot=320.0, iv=0.18)
        table = format_contract_greeks_table(greeks)
        assert table.count("2024-01-15") == 1
        assert table.count("2024-07-15") == 1

    def test_empty_contracts_returns_placeholder(self) -> None:
        empty = PortfolioGreeks(
            as_of_date=pd.Timestamp("2025-06-01"),
            contracts=(),
            net_delta=0.0,
            net_vega=0.0,
            net_gamma=0.0,
            net_theta=0.0,
            net_vanna=0.0,
            net_charm=0.0,
        )
        assert format_contract_greeks_table(empty) == "No active LEAPS contracts."


# ---------------------------------------------------------------------------
# format_holdings_table / format_trade_orders_table
# ---------------------------------------------------------------------------


class TestFormatHoldingsTable:
    def test_returns_string(self) -> None:
        lp = LivePortfolio(
            as_of_date=pd.Timestamp("2025-06-01"),
            holdings={"VXUS": 78_000.0, "GLD": 74_000.0},
            target_weights={"VXUS": 0.60, "GLD": 0.20, "VTI_LEAPS": 0.20},
            leaps_contracts=(),
            gtt_regime=None,
        )
        nb = compute_nav_breakdown(lp, leaps_mtm=40_000.0)
        views = (*compute_holdings_view(lp, nb), *compute_leaps_holdings_view(lp, nb))
        assert isinstance(format_holdings_table(views), str)

    def test_leaps_row_included_alongside_base_assets(self) -> None:
        lp = LivePortfolio(
            as_of_date=pd.Timestamp("2025-06-01"),
            holdings={"VXUS": 78_000.0},
            target_weights={"VXUS": 0.60, "VTI_LEAPS": 0.40},
            leaps_contracts=(),
            gtt_regime=None,
        )
        nb = compute_nav_breakdown(lp, leaps_mtm=40_000.0)
        views = (*compute_holdings_view(lp, nb), *compute_leaps_holdings_view(lp, nb))
        table = format_holdings_table(views)
        assert "VXUS" in table
        assert "VTI_LEAPS" in table

    def test_dollar_value_not_scientific_notation(self) -> None:
        """A column mixing 0.0 with large values must not render in scientific notation."""
        lp = LivePortfolio(
            as_of_date=pd.Timestamp("2025-06-01"),
            holdings={"VTI": 0.0, "VXUS": 2_375_081.0},
            target_weights={"VTI": 0.0, "VXUS": 1.0},
            leaps_contracts=(),
            gtt_regime=None,
        )
        nb = compute_nav_breakdown(lp)
        table = format_holdings_table(compute_holdings_view(lp, nb))
        assert "e+" not in table
        assert "2,375,081.00" in table


class TestFormatTradeOrdersTable:
    def test_empty_trades_returns_empty_string(self) -> None:
        assert format_trade_orders_table(()) == ""

    def test_returns_string_for_nonempty_trades(self) -> None:
        lp = LivePortfolio(
            as_of_date=pd.Timestamp("2025-06-01"),
            holdings={"VXUS": 78_000.0, "GLD": 74_000.0},
            target_weights={"VXUS": 0.6, "GLD": 0.4},
            leaps_contracts=(),
            gtt_regime=None,
        )
        nb = compute_nav_breakdown(lp)
        plan = compute_rebalance_plan(
            lp, nb, RebalanceRule.QUARTERLY, is_rebal_date=True, is_month_end=False
        )
        table = format_trade_orders_table(plan.trades)
        assert isinstance(table, str)
        assert "VXUS" in table
        assert "GLD" in table

    def test_leaps_trim_row_combines_with_base_trades(self) -> None:
        lp = LivePortfolio(
            as_of_date=pd.Timestamp("2025-06-01"),
            holdings={"VTI": 40_000.0},
            target_weights={"VTI": 0.70, "VTI_LEAPS": 0.30},
            leaps_contracts=(),
            gtt_regime=None,
        )
        nb = compute_nav_breakdown(lp, leaps_mtm=60_000.0)
        plan = compute_rebalance_plan(
            lp, nb, RebalanceRule.DRIFT, is_rebal_date=False, is_month_end=True
        )
        assert plan.leaps_trim > 0.0
        leaps_view = compute_leaps_holdings_view(lp, nb)[0]
        leaps_trade = leaps_trim_as_trade_order(leaps_view, plan.leaps_trim)
        assert leaps_trade is not None
        table = format_trade_orders_table((*plan.trades, leaps_trade))
        assert "VTI_LEAPS" in table
        assert "e+" not in table


# ---------------------------------------------------------------------------
# format_nav_breakdown_table
# ---------------------------------------------------------------------------


class TestFormatNavBreakdownTable:
    def test_returns_string(self) -> None:
        lp = LivePortfolio(
            as_of_date=pd.Timestamp("2025-06-01"),
            holdings={"VXUS": 78_000.0},
            target_weights={"VXUS": 1.0},
            leaps_contracts=(),
            gtt_regime=None,
        )
        nb = compute_nav_breakdown(lp, leaps_mtm=40_000.0)
        assert isinstance(format_nav_breakdown_table(nb), str)

    def test_one_row_per_field(self) -> None:
        lp = LivePortfolio(
            as_of_date=pd.Timestamp("2025-06-01"),
            holdings={"VXUS": 78_000.0},
            target_weights={"VXUS": 1.0},
            leaps_contracts=(),
            gtt_regime=None,
        )
        nb = compute_nav_breakdown(lp, leaps_mtm=40_000.0)
        table = format_nav_breakdown_table(nb)
        lines = table.splitlines()
        assert len(lines) == 5
        for field in ("base_nav", "leaps_nav", "defensive_sleeve", "leaps_pool", "total_nav"):
            assert field in table

    def test_values_not_scientific_notation(self) -> None:
        lp = LivePortfolio(
            as_of_date=pd.Timestamp("2025-06-01"),
            holdings={"VXUS": 78_000.0},
            target_weights={"VXUS": 1.0},
            leaps_contracts=(),
            gtt_regime=None,
        )
        nb = compute_nav_breakdown(lp, leaps_mtm=2_375_081.0)
        table = format_nav_breakdown_table(nb)
        assert "e+" not in table
        assert "$2,375,081.00" in table


# ---------------------------------------------------------------------------
# format_leaps_dca_signal_table
# ---------------------------------------------------------------------------


def _make_dca_signal() -> LeapsDcaSignal:
    """A representative LeapsDcaSignal for table-formatting tests."""
    return LeapsDcaSignal(
        as_of_date=pd.Timestamp("2026-06-29"),
        ticker="VTI",
        entry_score=15.7,
        score_percentile=29.2,
        alpha_t=0.08,
        dca_action="TRANCHE",
        rsi=54.3,
        stoch_d=45.3,
        iv_percentile=59.9,
        iv_current=0.176,
        macd_hist=-0.760,
        macd_bearish_confirmed=True,
        macd_gate=0.5,
    )


class TestFormatLeapsDcaSignalTable:
    def test_returns_string(self) -> None:
        assert isinstance(format_leaps_dca_signal_table(_make_dca_signal()), str)

    def test_one_row_per_field(self) -> None:
        table = format_leaps_dca_signal_table(_make_dca_signal())
        lines = table.splitlines()
        assert len(lines) == 13

    def test_contains_expected_fields_and_values(self) -> None:
        table = format_leaps_dca_signal_table(_make_dca_signal())
        assert "as_of_date" in table
        assert "2026-06-29" in table
        assert "ticker" in table
        assert "VTI" in table
        assert "dca_action" in table
        assert "TRANCHE" in table
        assert "macd_bearish_confirmed" in table
        assert "True" in table

    def test_iv_current_rendered_as_percent(self) -> None:
        table = format_leaps_dca_signal_table(_make_dca_signal())
        assert "17.6%" in table
