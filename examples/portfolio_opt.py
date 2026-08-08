"""
This example demonstrates how to run a backtest for a range of different portfolio weight 
combinations, and then compare the results.
"""

from pathlib import Path

from finance._portfolio_types import PortfolioConfig
from finance.data import build_price_data, fetch_risk_free_rate
from finance.figures import compare_performance_table, plot_nav_growth, plot_pareto
from finance.leverage import RebalanceRule, WeightStrategy
from finance.metrics import build_performance_report
from finance.portfolio import run_backtest
from finance.returns import build_return_data
from finance.volatility import build_volatility_model


INITIAL_NAV = 1_000_000.0
MONTHLY_CONTRIBUTION = 10_000.0
LTCG_RATE = 0.238
START, END = "2000-09-01", "2026-06-30"

target_flexible = 50

# Bounds scaled to integer percentage points
vxus_range = range(10, 31, 10)  # 0.10 to 0.30
flexible_range = range(5, 16, 5)  # 0.05 to 0.15

combinations: dict[str, dict] = {}

for vxus in vxus_range:
    for kmlm in flexible_range:
        for vgit in flexible_range:
            gld = target_flexible - vxus - kmlm - vgit
            if gld in flexible_range:
                w_vxus = round(vxus / 100.0, 2)
                w_kmlm = round(kmlm / 100.0, 2)
                w_vgit = round(vgit / 100.0, 2)
                w_gld = round(gld / 100.0, 2)
                w = {
                    "VTI": 0.45,
                    "VXUS": w_vxus,
                    "GLD": w_gld,
                    "MUB": 0.05,
                    "KMLM": w_kmlm,
                    "VGIT": w_vgit,
                }

                key = f"VXUS: {w_vxus} / K: {w_kmlm} / VGIT: {w_vgit} / G: {w_gld}"
                combinations[key] = w

if __name__ == "__main__":
    print("=== Fetching Inputs ===")
    price_data = build_price_data(START, END, use_splice=True, fetch_vol_indices=True)
    rfr_series = fetch_risk_free_rate(START, END)
    return_data = build_return_data(price_data, risk_free_series=rfr_series)
    vol_model = build_volatility_model(return_data)

    print("=== Running backtest + Reporting Loop ===")
    bt_d = {}
    reports = []
    reports_d = {}
    for name, weights in combinations.items():
        config = PortfolioConfig(
            target_weights=weights,
            initial_nav=INITIAL_NAV,
            monthly_contribution=MONTHLY_CONTRIBUTION,
            rebalance_rule=RebalanceRule.QUARTERLY,
            weight_strategy=WeightStrategy.USER_SPECIFIED,
            leaps_config=None,
        )

        print(f"=== Running Backtest: {name} ===")
        result = run_backtest(return_data, price_data, config)
        report = build_performance_report(
            result, price_data, return_data, vol_model
        )
        bt_d[name] = result
        reports.append((name, report))
        reports_d[name] = report


    print("\n=== Portfolio Comparison ===")
    print(compare_performance_table(reports))

    print("\n=== Charting ===")
    output_path_nav = Path("outputs/figures/portfolio_opt_nav.png")
    output_path_pareto = Path("outputs/figures/portfolio_opt_pareto.png")
    
    plot_nav_growth(bt_d, output_path=output_path_nav)
    
    extracted = {k: pr.full_period for k, pr in reports_d.items()}
    plot_pareto(extracted, output_path=output_path_pareto)