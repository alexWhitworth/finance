"""LEAPS overlay backtest: taxable vs. tax-sheltered account comparison.

Runs two identical backtests from 2015-01-01 to 2024-12-31 — one with a LEAPS
overlay in a TAXABLE account, one in a TAX_SHELTERED account — then prints a
side-by-side NAV summary and saves a tax-drag comparison chart to
figures/leaps_tax_drag.png.

The base portfolio is a 6-asset quarterly-rebalanced allocation. LEAPS contracts
on VTI are accumulated month-by-month via run_leaps_simulation; the resulting
ledger is passed to run_backtest, which marks LEAPS positions to market as an
overlay on top of the base holdings.
"""

from pathlib import Path

from finance.data import PriceData, build_price_data
from finance.figures import plot_leaps_tax_drag
from finance.leverage import (
    AccountType,
    LeapsConfig,
    RebalanceRule,
    WeightStrategy,
    run_leaps_simulation,
)
from finance.portfolio import BacktestResult, PortfolioConfig, run_backtest
from finance.returns import ReturnData, build_return_data

WEIGHTS = {
    "VTI": 0.35,
    "VXUS": 0.20,
    "GLD": 0.10,
    "MUB": 0.10,
    "KMLM": 0.10,
    "VGIT": 0.15,
}

INITIAL_NAV = 1_000_000.0
MONTHLY_CONTRIBUTION = 10_000.0
MONTHLY_LEAPS_CONTRIBUTION = 5_000.0
IV = 0.18
LTCG_RATE = 0.238


def _run_scenario(
    price_data: PriceData,
    return_data: ReturnData,
    account_type: AccountType,
) -> BacktestResult:
    """Run a full LEAPS backtest for one account type.

    Arguments:
        price_data: PriceData from build_price_data().
        return_data: ReturnData from build_return_data().
        account_type: AccountType.TAXABLE or AccountType.TAX_SHELTERED.

    Returns:
        BacktestResult with LEAPS overlay applied.
    """
    leaps_config = LeapsConfig(iv=IV, ltcg_rate=LTCG_RATE, account_type=account_type)

    ledger = run_leaps_simulation(
        price_series=price_data.prices["VTI"],
        monthly_contribution_to_leaps=MONTHLY_LEAPS_CONTRIBUTION,
        config=leaps_config,
    )

    config = PortfolioConfig(
        target_weights=WEIGHTS,
        initial_nav=INITIAL_NAV,
        monthly_contribution=MONTHLY_CONTRIBUTION,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=leaps_config,
    )

    return run_backtest(return_data, config, leaps_ledger=ledger)


if __name__ == "__main__":
    print("=== Fetching Price Data ===")
    price_data = build_price_data("2007-09-10", "2026-06-30", use_aqmix_splice=True)

    print("=== Building Returns ===")
    return_data = build_return_data(price_data)

    print("=== Running Taxable LEAPS Backtest ===")
    taxable_result = _run_scenario(price_data, return_data, AccountType.TAXABLE)

    print("=== Running Tax-Sheltered LEAPS Backtest ===")
    sheltered_result = _run_scenario(price_data, return_data, AccountType.TAX_SHELTERED)

    # --- Summary ---
    taxable_final = float(taxable_result.nav_series.iloc[-1])
    sheltered_final = float(sheltered_result.nav_series.iloc[-1])
    tax_drag = sheltered_final - taxable_final

    print()
    print("=" * 52)
    print("  LEAPS Backtest Summary (2007-09-10 → 2026-06-30)")
    print("=" * 52)
    print(f"  Taxable final NAV      : ${taxable_final:>15,.0f}")
    print(f"  Tax-Sheltered final NAV: ${sheltered_final:>15,.0f}")
    print(f"  Tax drag (dollar)      : ${tax_drag:>15,.0f}")
    print("=" * 52)

    print()
    print("=== Saving Tax Drag Chart ===")
    output_path = Path("figures/leaps_tax_drag.png")
    plot_leaps_tax_drag(taxable_result, sheltered_result, output_path=output_path)
    print(f"Chart saved to {output_path}")
