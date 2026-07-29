"""End-to-end backtest example for a 6-asset diversified portfolio.

Fetches prices with data splicing for assets without long price history (eg VTI -> VTSMX),
runs a quarterly-rebalanced backtest, computes performance metrics and 
a volatility model, then prints a formatted performance table and saves a 
NAV growth chart to figures/basic_backtest_nav.png.
"""

from pathlib import Path

from finance.data import build_price_data, fetch_risk_free_rate
from finance.figures import format_performance_table, plot_nav_growth
from finance.leverage import RebalanceRule, WeightStrategy
from finance.metrics import build_performance_report
from finance.portfolio import PortfolioConfig, run_backtest
from finance.returns import build_return_data
from finance.volatility import build_volatility_model

WEIGHTS = {
    "VTI": 0.40,
    "VXUS": 0.15,
    "GLD": 0.10,
    "MUB": 0.15,
    "KMLM": 0.10,
    "VGIT": 0.10,
}

if __name__ == "__main__":
    START, END = "2000-09-01", "2026-06-30"

    print("=== Fetching Price Data ===")
    price_data = build_price_data(START, END, use_splice=True)

    print("=== Fetching Risk-Free Rate ===")
    rfr_series = fetch_risk_free_rate(START, END)

    print("=== Building Returns ===")
    return_data = build_return_data(price_data, risk_free_series=rfr_series)

    config = PortfolioConfig(
        target_weights=WEIGHTS,
        initial_nav=1_000_000.0,
        monthly_contribution=10_000.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=None,
    )

    print("=== Running Backtest ===")
    result = run_backtest(return_data, price_data, config)

    print("=== Building Volatility Model ===")
    vol_model = build_volatility_model(return_data)

    print("=== Building Performance Report ===")
    report = build_performance_report(result, price_data, return_data, vol_model)

    print("=== Performance Report ===")
    print(format_performance_table(report))

    print("=== Saving NAV Growth Chart ===")
    output_path = "outputs/figures/basic_backtest_nav.png"
    plot_nav_growth(
        {"Portfolio": result},
        output_path=Path(output_path),
    )
    print(f"Chart saved to {output_path}")
