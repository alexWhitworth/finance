"""Crisis-period performance analysis for a 6-asset diversified portfolio.

Fetches prices from 2010-02-01 to 2026-06-30 (AQMIX splice used as KMLM proxy
before 2021), runs a quarterly-rebalanced backtest, then slices the return
series into three pre-defined crisis windows (GFC, COVID, 2022 Rate Hike) and
prints PerformanceMetrics for each.  Also saves a drawdown chart with shaded
crisis bands to figures/crisis_drawdown.png.
"""

from pathlib import Path

from finance.data import build_price_data, fetch_risk_free_rate
from finance.figures import format_performance_table, plot_drawdown
from finance.leverage import RebalanceRule, WeightStrategy
from finance.metrics import CRISIS_PERIODS, PerformanceReport, build_performance_report
from finance.portfolio import PortfolioConfig, run_backtest
from finance.returns import build_return_data
from finance.volatility import build_volatility_model

WEIGHTS = {
    "VTI": 0.40,
    "VXUS": 0.20,
    "GLD": 0.10,
    "MUB": 0.10,
    "KMLM": 0.10,
    "VGIT": 0.10,
}

if __name__ == "__main__":
    START, END = "2010-02-01", "2026-06-30"

    print("=== Fetching Price Data ===")
    price_data = build_price_data(START, END, use_aqmix_splice=True)

    print("=== Fetching Risk-Free Rate ===")
    rfr_series = fetch_risk_free_rate(START, END)

    print("=== Building Returns ===")
    return_data = build_return_data(price_data, risk_free_series=rfr_series)

    config = PortfolioConfig(
        target_weights=WEIGHTS,
        initial_nav=1_000_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=None,
    )

    print("=== Running Backtest ===")
    result = run_backtest(return_data, config)

    print("=== Building Volatility Model ===")
    vol_model = build_volatility_model(return_data)

    print("=== Building Performance Report ===")
    report = build_performance_report(result, return_data, vol_model)

    print("\n=== Full-Period Metrics ===")
    print(format_performance_table(report))

    print("\n=== Crisis-Period Metrics ===")
    for metrics in report.crisis_periods:
        start, end = CRISIS_PERIODS[metrics.period_label]
        print(f"\n--- {metrics.period_label} ({start} → {end}) ---")
        crisis_report = PerformanceReport(
            full_period=metrics,
            crisis_periods=(),
            vol_contribution_table=report.vol_contribution_table,
            forward_vol_forecast=report.forward_vol_forecast,
        )
        print(format_performance_table(crisis_report))

    print("\n=== Saving Drawdown Chart ===")
    Path("figures").mkdir(exist_ok=True)
    plot_drawdown(
        {"Portfolio": result},
        output_path=Path("figures/crisis_drawdown.png"),
    )
    print("Chart saved to figures/crisis_drawdown.png")
