"""End-to-end integration test: full backtest pipeline → PerformanceReport.

Verifies the complete data → returns → volatility → backtest → metrics chain
using synthetic (offline) data.  No network I/O is performed.
The real-data smoke test (TestRealDataSmoke) loads data/price_data.parquet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9  # type: ignore[import-untyped]
import pytest

from finance.data import PriceData
from finance.figures import format_performance_table, plot_nav_growth
from finance.leverage import RebalanceRule, WeightStrategy
from finance.metrics import PerformanceReport, build_performance_report
from finance.portfolio import BacktestResult, PortfolioConfig, run_backtest
from finance.returns import ReturnData
from finance.volatility import build_volatility_model

# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

_TICKERS = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
_EQUAL_WEIGHTS = {t: 1.0 / len(_TICKERS) for t in _TICKERS}


def _synthetic_return_data(n_days: int = 756, seed: int = 0) -> ReturnData:
    """Generate ~3 years of synthetic daily returns for all 6 assets."""
    idx = pd.bdate_range("2020-01-02", periods=n_days + 1)
    rng = np.random.default_rng(seed)
    starts = {"VTI": 200.0, "VXUS": 60.0, "GLD": 170.0, "MUB": 55.0, "KMLM": 25.0, "VGIT": 65.0}
    prices = pd.DataFrame(
        {t: starts[t] * np.cumprod(1 + rng.normal(0.0003, 0.01, n_days + 1)) for t in _TICKERS},
        index=idx,
    )
    rets = prices.pct_change().dropna()
    log_rets = np.log(1 + rets)
    return ReturnData(
        returns=rets,
        log_returns=log_rets,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=pd.Series(0.0, index=rets.index, name="risk_free_rate"),
    )


def _base_config(contribution: float = 10_000.0) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights=dict(_EQUAL_WEIGHTS),
        initial_nav=1_000_000.0,
        monthly_contribution=contribution,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=None,
    )


# ---------------------------------------------------------------------------
# Integration test: core pipeline
# ---------------------------------------------------------------------------


def _synthetic_price_data(returns_index: pd.DatetimeIndex) -> PriceData:
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


@pytest.fixture(scope="module")
def pipeline() -> dict[str, object]:
    """Full pipeline result: return data → backtest → vol model → report."""
    rd = _synthetic_return_data()
    cfg = _base_config()
    pd_obj = _synthetic_price_data(rd.returns.index)
    result = run_backtest(rd, pd_obj, cfg)
    vol_model = build_volatility_model(rd)
    report = build_performance_report(result, pd_obj, rd, vol_model)
    return {"rd": rd, "cfg": cfg, "result": result, "vol_model": vol_model, "report": report}


class TestFullBacktestPipeline:
    """Full data → returns → volatility → backtest → report chain."""

    # --- BacktestResult structural checks ---

    def test_nav_series_length(self, pipeline: dict[str, object]) -> None:
        result: BacktestResult = pipeline["result"]  # type: ignore[assignment]
        rd: ReturnData = pipeline["rd"]  # type: ignore[assignment]
        assert len(result.nav_series) == len(rd.returns)

    def test_nav_positive(self, pipeline: dict[str, object]) -> None:
        result: BacktestResult = pipeline["result"]  # type: ignore[assignment]
        assert (result.nav_series > 0).all()

    def test_nav_starts_near_initial(self, pipeline: dict[str, object]) -> None:
        result: BacktestResult = pipeline["result"]  # type: ignore[assignment]
        cfg: PortfolioConfig = pipeline["cfg"]  # type: ignore[assignment]
        assert result.nav_series.iloc[0] == pytest.approx(cfg.initial_nav, rel=0.01)

    def test_weights_sum_to_one(self, pipeline: dict[str, object]) -> None:
        result: BacktestResult = pipeline["result"]  # type: ignore[assignment]
        row_sums = result.weight_history.sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-6

    def test_return_series_finite(self, pipeline: dict[str, object]) -> None:
        result: BacktestResult = pipeline["result"]  # type: ignore[assignment]
        assert result.return_series.isna().sum() == 0

    # --- PerformanceReport structural checks ---

    def test_report_full_period_present(self, pipeline: dict[str, object]) -> None:
        report: PerformanceReport = pipeline["report"]  # type: ignore[assignment]
        assert report.full_period.period_label == "Full Period"

    def test_report_metrics_finite(self, pipeline: dict[str, object]) -> None:
        report: PerformanceReport = pipeline["report"]  # type: ignore[assignment]
        m = report.full_period
        for field in (m.annualized_return, m.annualized_std, m.max_drawdown, m.sharpe):
            assert np.isfinite(field)

    def test_report_max_drawdown_non_negative(self, pipeline: dict[str, object]) -> None:
        report: PerformanceReport = pipeline["report"]  # type: ignore[assignment]
        assert report.full_period.max_drawdown >= 0.0

    def test_vol_contribution_table_shape(self, pipeline: dict[str, object]) -> None:
        report: PerformanceReport = pipeline["report"]  # type: ignore[assignment]
        tbl = report.vol_contribution_table
        assert set(tbl.columns) >= {"sigma_tilde", "sigma_hat", "rho_VTI", "contrib"}
        assert len(tbl) == len(_TICKERS)

    def test_vol_contributions_sum_to_one(self, pipeline: dict[str, object]) -> None:
        report: PerformanceReport = pipeline["report"]  # type: ignore[assignment]
        assert report.vol_contribution_table["contrib"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_forward_vol_positive(self, pipeline: dict[str, object]) -> None:
        report: PerformanceReport = pipeline["report"]  # type: ignore[assignment]
        assert report.forward_vol_forecast > 0.0

    # --- Figures layer ---

    def test_format_performance_table(self, pipeline: dict[str, object]) -> None:
        report: PerformanceReport = pipeline["report"]  # type: ignore[assignment]
        table = format_performance_table(report)
        assert "Full Period" in table
        assert "Forward Vol" in table

    def test_plot_nav_growth_no_file_io(self, pipeline: dict[str, object]) -> None:
        result: BacktestResult = pipeline["result"]  # type: ignore[assignment]
        plot = plot_nav_growth({"Base": result}, output_path=None)
        assert isinstance(plot, p9.ggplot)


# ---------------------------------------------------------------------------
# Integration test: with monthly contributions
# ---------------------------------------------------------------------------


class TestContributionCompounding:
    def test_nav_grows_faster_with_contributions(self) -> None:
        rd = _synthetic_return_data()
        pd_obj = _synthetic_price_data(rd.returns.index)
        result_no_contrib = run_backtest(rd, pd_obj, _base_config(contribution=0.0))
        result_contrib = run_backtest(rd, pd_obj, _base_config(contribution=10_000.0))
        assert result_contrib.nav_series.iloc[-1] > result_no_contrib.nav_series.iloc[-1]


# ---------------------------------------------------------------------------
# Integration test: multi-portfolio report
# ---------------------------------------------------------------------------


class TestMultiPortfolioReport:
    def test_two_portfolios_different_nav(self) -> None:
        rd = _synthetic_return_data()
        # Equity-heavy vs bond-heavy
        equity_weights = {
            "VTI": 0.7, "VXUS": 0.15, "GLD": 0.05, "MUB": 0.05, "KMLM": 0.025, "VGIT": 0.025,
        }
        bond_weights = {
            "VTI": 0.1, "VXUS": 0.05, "GLD": 0.05, "MUB": 0.4, "KMLM": 0.1, "VGIT": 0.3,
        }
        cfg_equity = PortfolioConfig(
            target_weights=equity_weights,
            initial_nav=1_000_000.0,
            monthly_contribution=0.0,
            rebalance_rule=RebalanceRule.QUARTERLY,
            weight_strategy=WeightStrategy.USER_SPECIFIED,
            leaps_config=None,
        )
        cfg_bond = PortfolioConfig(
            target_weights=bond_weights,
            initial_nav=1_000_000.0,
            monthly_contribution=0.0,
            rebalance_rule=RebalanceRule.QUARTERLY,
            weight_strategy=WeightStrategy.USER_SPECIFIED,
            leaps_config=None,
        )
        pd_obj = _synthetic_price_data(rd.returns.index)
        r_equity = run_backtest(rd, pd_obj, cfg_equity)
        r_bond = run_backtest(rd, pd_obj, cfg_bond)
        # Final NAVs should differ (synthetic data creates return dispersion)
        assert r_equity.nav_series.iloc[-1] != pytest.approx(r_bond.nav_series.iloc[-1], rel=1e-3)

    def test_report_crisis_period_none_when_no_overlap(self) -> None:
        # Data starts in 2020, GFC (2007-2009) has no overlap → no crisis metrics for GFC
        rd = _synthetic_return_data()
        cfg = _base_config()
        pd_obj = _synthetic_price_data(rd.returns.index)
        result = run_backtest(rd, pd_obj, cfg)
        vol_model = build_volatility_model(rd)
        crisis = {"GFC": ("2007-10-01", "2009-03-31")}
        report = build_performance_report(result, pd_obj, rd, vol_model, crisis_periods=crisis)
        # GFC has no data overlap — crisis_periods tuple should be empty
        assert len(report.crisis_periods) == 0


# ---------------------------------------------------------------------------
# Real-data smoke test — loads data/price_data.parquet (committed to repo)
# ---------------------------------------------------------------------------

_PARQUET_PATH = Path(__file__).parent.parent / "data" / "price_data.parquet"
_ASSET_TICKERS = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")


@pytest.fixture(scope="module")
def real_pipeline() -> dict[str, object]:
    """Full pipeline result built from the committed parquet fixture."""
    df = pd.read_parquet(_PARQUET_PATH)

    # IRX is stored as an annualized decimal rate (e.g. 0.052); VIX as index level (e.g. 18.5).
    prices = df[list(_ASSET_TICKERS)]
    irx = df["IRX"]
    vix = df["VIX"] / 100.0  # index level → decimal IV

    dividends = pd.DataFrame(0.0, index=prices.index, columns=list(_ASSET_TICKERS))
    vol_prices = pd.DataFrame({"^VIX": vix}, index=prices.index)

    pd_obj = PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=vol_prices,
        tickers=_ASSET_TICKERS,
        start_date=str(prices.index[0].date()),
        end_date=str(prices.index[-1].date()),
        spliced=False,
    )

    from finance.returns import build_return_data

    rd = build_return_data(pd_obj, apply_tey=False, risk_free_series=irx)

    cfg = PortfolioConfig(
        target_weights={t: 1.0 / len(_ASSET_TICKERS) for t in _ASSET_TICKERS},
        initial_nav=1_000_000.0,
        monthly_contribution=5_000.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=None,
    )

    result = run_backtest(rd, pd_obj, cfg)
    vol_model = build_volatility_model(rd)
    report = build_performance_report(result, pd_obj, rd, vol_model)
    return {"rd": rd, "pd_obj": pd_obj, "result": result, "report": report}


class TestRealDataSmoke:
    """Smoke tests using the committed data/price_data.parquet fixture.

    Verifies that the full pipeline runs without error on real market data and
    produces structurally valid outputs. Catches schema drift between the parquet
    file and the code that would be invisible to synthetic-data tests.
    """

    def test_parquet_loads_expected_columns(self) -> None:
        """Parquet file contains all required asset and auxiliary columns."""
        df = pd.read_parquet(_PARQUET_PATH)
        for col in (*_ASSET_TICKERS, "IRX", "VIX"):
            assert col in df.columns, f"Expected column '{col}' missing from parquet"

    def test_nav_positive_throughout(self, real_pipeline: dict[str, object]) -> None:
        result: BacktestResult = real_pipeline["result"]  # type: ignore[assignment]
        assert (result.nav_series > 0).all()

    def test_nav_length_matches_returns(self, real_pipeline: dict[str, object]) -> None:
        result: BacktestResult = real_pipeline["result"]  # type: ignore[assignment]
        rd: ReturnData = real_pipeline["rd"]  # type: ignore[assignment]
        assert len(result.nav_series) == len(rd.returns)

    def test_weights_sum_to_one(self, real_pipeline: dict[str, object]) -> None:
        result: BacktestResult = real_pipeline["result"]  # type: ignore[assignment]
        assert (result.weight_history.sum(axis=1) - 1.0).abs().max() < 1e-9

    def test_return_series_no_nans(self, real_pipeline: dict[str, object]) -> None:
        result: BacktestResult = real_pipeline["result"]  # type: ignore[assignment]
        assert result.return_series.isna().sum() == 0

    def test_report_metrics_finite(self, real_pipeline: dict[str, object]) -> None:
        report: PerformanceReport = real_pipeline["report"]  # type: ignore[assignment]
        m = report.full_period
        for field in (m.annualized_return, m.annualized_std, m.max_drawdown, m.sharpe):
            assert np.isfinite(field), f"Non-finite metric: {field}"

    def test_vol_contributions_sum_to_one(self, real_pipeline: dict[str, object]) -> None:
        report: PerformanceReport = real_pipeline["report"]  # type: ignore[assignment]
        assert report.vol_contribution_table["contrib"].sum() == pytest.approx(1.0, abs=1e-6)
