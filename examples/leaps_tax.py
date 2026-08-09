"""LEAPS overlay backtest: taxable vs. tax-sheltered account comparison.

Runs two identical backtests — one with a LEAPS overlay in a TAXABLE account,
one in a TAX_SHELTERED account — then prints a side-by-side performance
comparison and saves a tax-drag comparison chart to figures/leaps_tax_drag.png.

The base portfolio is a 6-asset quarterly-rebalanced allocation. The LEAPS
overlay is specified via a "VTI_LEAPS" key in target_weights; run_backtest
handles simulation internally using VIX-based dynamic implied volatility.
"""

from pathlib import Path

from finance._portfolio_types import PortfolioConfig
from finance.data import build_price_data, fetch_risk_free_rate
from finance.figures import compare_performance_table, plot_leaps_tax_drag
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance.metrics import build_performance_report
from finance.portfolio import run_backtest
from finance.returns import build_return_data
from finance.volatility import build_volatility_model

# VTI and VTI_LEAPS keys are independent: VTI_LEAPS does not replace VTI.
# The portfolio can hold both the underlying and LEAPS
BASE_WEIGHTS = {
    "VTI": 0.0,
    "VXUS": 0.20,
    "GLD": 0.10,
    "MUB": 0.10,
    "KMLM": 0.10,
    "VGIT": 0.10,
    "VTI_LEAPS": 0.4,
}

INITIAL_NAV = 1_000_000.0
MONTHLY_CONTRIBUTION = 10_000.0
FLOOR_IV = 0.1
LTCG_RATE = 0.238


if __name__ == "__main__":
    START, END = "2000-09-01", "2026-06-30"

    print("=== Fetching Price Data ===")
    price_data = build_price_data(START, END, use_splice=True, fetch_vol_indices=True)

    print("=== Fetching Risk-Free Rate ===")
    rfr_series = fetch_risk_free_rate(START, END)

    print("=== Building Returns ===")
    return_data = build_return_data(price_data, risk_free_series=rfr_series)

    print("=== Building Volatility Model ===")
    vol_model = build_volatility_model(return_data)

    taxable_config = PortfolioConfig(
        target_weights=BASE_WEIGHTS,
        initial_nav=INITIAL_NAV,
        monthly_contribution=MONTHLY_CONTRIBUTION,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(
            iv=FLOOR_IV,
            ltcg_rate=LTCG_RATE,
            account_type=AccountType.TAXABLE,
        ),
    )

    sheltered_config = PortfolioConfig(
        target_weights=BASE_WEIGHTS,
        initial_nav=INITIAL_NAV,
        monthly_contribution=MONTHLY_CONTRIBUTION,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(
            iv=FLOOR_IV,
            ltcg_rate=LTCG_RATE,
            account_type=AccountType.TAX_SHELTERED,
        ),
    )

    print("=== Running Taxable LEAPS Backtest ===")
    taxable_result = run_backtest(return_data, price_data, taxable_config)

    print("=== Running Tax-Sheltered LEAPS Backtest ===")
    sheltered_result = run_backtest(return_data, price_data, sheltered_config)

    print("=== Building Performance Reports ===")
    taxable_report = build_performance_report(taxable_result, price_data, return_data, vol_model)
    sheltered_report = build_performance_report(
        sheltered_result, price_data, return_data, vol_model
    )

    print("\n=== Portfolio Comparison ===")
    print(compare_performance_table([("Taxable", taxable_report), ("Sheltered", sheltered_report)]))

    print("\n=== Saving Tax Drag Chart ===")
    output_path = Path("outputs/figures/leaps_tax_drag.png")
    plot_leaps_tax_drag(taxable_result, sheltered_result, output_path=output_path)
    print(f"Chart saved to {output_path}")
