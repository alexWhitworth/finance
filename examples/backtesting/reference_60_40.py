"""Reference 60/40 and 80/20 VTI/BND backtest, run as a comparison baseline.

Mirrors basic_backtest.py's pipeline (splice, returns, quarterly-rebalanced
backtest, performance report) but replaces the 6-asset diversified mix with
two simple VTI/BND allocations: 60/40 and 80/20. BND is spliced onto VBMFX
(see finance.consts.SPLICE_MAP) since BND's own yfinance history only starts
2007-04-10.
"""

from pathlib import Path

from finance.data import build_price_data, fetch_risk_free_rate
from finance.figures import compare_performance_table, format_performance_table, plot_nav_growth
from finance.leverage import RebalanceRule, WeightStrategy
from finance.metrics import build_performance_report
from finance.portfolio import PortfolioConfig, run_backtest
from finance.returns import build_return_data
from finance.volatility import build_volatility_model

WEIGHTS_60_40 = {"VTI": 0.60, "BND": 0.40}
WEIGHTS_80_20 = {"VTI": 0.80, "BND": 0.20}

if __name__ == "__main__":
    START, END = "2000-09-01", "2026-06-30"

    print("=== Fetching Price Data ===")
    price_data = build_price_data(START, END, tickers=["VTI", "BND"], use_splice=True)

    print("=== Fetching Risk-Free Rate ===")
    rfr_series = fetch_risk_free_rate(START, END)

    print("=== Building Returns ===")
    return_data = build_return_data(price_data, risk_free_series=rfr_series)

    config_60_40 = PortfolioConfig(
        target_weights=WEIGHTS_60_40,
        initial_nav=1_000_000.0,
        monthly_contribution=10_000.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=None,
    )

    config_80_20 = PortfolioConfig(
        target_weights=WEIGHTS_80_20,
        initial_nav=1_000_000.0,
        monthly_contribution=10_000.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=None,
    )

    print("=== Running Backtest ===")
    result_60_40 = run_backtest(return_data, price_data, config_60_40)
    result_80_20 = run_backtest(return_data, price_data, config_80_20)

    print("=== Building Volatility Model ===")
    vol_model = build_volatility_model(return_data)

    print("=== Building Performance Report ===")
    report_60_40 = build_performance_report(result_60_40, price_data, return_data, vol_model)
    report_80_20 = build_performance_report(result_80_20, price_data, return_data, vol_model)

    print("=== Performance Report: 60/40 VTI/BND ===")
    print(format_performance_table(report_60_40))

    print("\n=== Performance Report: 80/20 VTI/BND ===")
    print(format_performance_table(report_80_20))

    print("\n=== Comparison ===")
    print(compare_performance_table([
        ("60/40 VTI/BND", report_60_40),
        ("80/20 VTI/BND", report_80_20),
    ]))

    print("=== Saving NAV Growth Chart ===")
    output_path = "outputs/figures/reference_60_40_nav.png"
    plot_nav_growth(
        {"60/40 VTI/BND": result_60_40, "80/20 VTI/BND": result_80_20},
        output_path=Path(output_path),
    )
    print(f"Chart saved to {output_path}")
