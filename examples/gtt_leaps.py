"""End-to-end GTT vs. buy-and-hold comparison for the 6-asset diversified portfolio.

Fetches prices with data splicing for assets without long price history (eg VTI -> VTSMX),
runs quarterly-rebalanced backtests for both GTT-enabled and GTT-disabled configurations, 
then prints a side-by-side performance comparison and saves a NAV growth chart 
to outputs/figures/gtt_comparison_nav.png.

Usage:
    uv run examples/basic_gtt.py 2>&1 | tee outputs/gtt_example.log

Notes:
    - Requires FRED_API_KEY in the environment (or .env) for UNRATE data.
    - VIX P90 threshold (0.272) is derived from the 1993-2026 VIX history
      (mild look-ahead bias; documented per GTT assumption A1).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from finance.data import build_price_data, fetch_risk_free_rate
from finance.figures import compare_performance_table, plot_nav_growth
from finance.gtt import fetch_gtt_signal_data
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance.metrics import build_performance_report
from finance.portfolio import run_backtest
from finance._portfolio_types import GttConfig, PortfolioConfig
from finance.returns import build_return_data
from finance.volatility import build_volatility_model

load_dotenv()

WEIGHTS = {
    "VTI": 0.0,
    "VXUS": 0.15,
    "GLD": 0.10,
    "MUB": 0.10,
    "KMLM": 0.15,
    "VGIT": 0.10,
    "VTI_LEAPS": 0.4,
}

DEFENSIVE_WEIGHTS = {
    "R_f": 0.25,
    "KMLM": 0.50,
    "VGIT": 0.25,
    "GLD": 0.00,
}

INITIAL_NAV = 1_000_000.0
MONTHLY_CONTRIBUTION = 10_000.0
VIX_P90_THRESHOLD = 0.272
LTCG_RATE = 0.238
FLOOR_IV = 0.1

if __name__ == "__main__":
    START, END = "2000-09-01", "2026-06-30"

    print("=== Fetching Price Data ===")
    price_data = build_price_data(START, END, use_splice=True, fetch_vol_indices=True)

    print("=== Fetching Risk-Free Rate ===")
    rfr_series = fetch_risk_free_rate(START, END)

    print("=== Building Returns ===")
    return_data = build_return_data(price_data, risk_free_series=rfr_series)

    print("=== Fetching GTT Signal ===")
    vti_prices = price_data.prices["VTI"].rename("VTI")
    gtt_signal = fetch_gtt_signal_data(
        START,
        END,
        vix_p90_threshold=VIX_P90_THRESHOLD,
        equity_prices=vti_prices,
    )
    n_defensive = int((gtt_signal.position_mask == 0).sum())
    n_total = len(gtt_signal.position_mask)
    print(f"  Signal: {n_defensive}/{n_total} defensive days ({100.0 * n_defensive / n_total:.1f}%)")

    base_leaps_qtr = PortfolioConfig(
            target_weights=WEIGHTS,
            initial_nav=INITIAL_NAV,
            monthly_contribution=MONTHLY_CONTRIBUTION,
            rebalance_rule=RebalanceRule.QUARTERLY,
            weight_strategy=WeightStrategy.USER_SPECIFIED,
            leaps_config=LeapsConfig(
                iv=FLOOR_IV,
                ltcg_rate=LTCG_RATE,
                account_type=AccountType.TAX_SHELTERED,
            ),
            gtt_config=None,
        )

    base_leaps_drfit = PortfolioConfig(
        target_weights=WEIGHTS,
        initial_nav=INITIAL_NAV,
        monthly_contribution=MONTHLY_CONTRIBUTION,
        rebalance_rule=RebalanceRule.DRIFT,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(
            iv=FLOOR_IV,
            ltcg_rate=LTCG_RATE,
            account_type=AccountType.TAX_SHELTERED,
        ),
        gtt_config=None,
    )

    gtt_leaps_drift = PortfolioConfig(
        target_weights=WEIGHTS,
        initial_nav=1_000_000.0,
        monthly_contribution=10_000.0,
        rebalance_rule=RebalanceRule.DRIFT,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(
            iv=FLOOR_IV,
            ltcg_rate=LTCG_RATE,
            account_type=AccountType.TAX_SHELTERED,
        ),
        gtt_config=GttConfig(
            vix_p90_threshold=VIX_P90_THRESHOLD,
            defensive_weights=DEFENSIVE_WEIGHTS,
        ),
    )

    print("=== Running Backtests ===")
    base_qtr_result = run_backtest(return_data, price_data, base_leaps_qtr)
    base_drift_result = run_backtest(return_data, price_data, base_leaps_drfit)
    gtt_result = run_backtest(return_data, price_data, gtt_leaps_drift, gtt_signal=gtt_signal)

    print("=== Building Volatility Model ===")
    vol_model = build_volatility_model(return_data)

    print("=== Building Performance Reports ===")
    base_qtr_report = build_performance_report(base_qtr_result, price_data, return_data, vol_model)
    base_drift_report = build_performance_report(base_drift_result, price_data, return_data, vol_model)
    gtt_report = build_performance_report(gtt_result, price_data, return_data, vol_model)

    print("=== Performance Comparison ===")
    print(compare_performance_table(
        [
            ("LEAPS QTR", base_qtr_report),
            ("LEAPS DRIFT", base_drift_report),
            ("GTT LEAPS DRIFT", gtt_report)
        ]
    ))

    print("=== Saving NAV Growth Chart ===")
    output_path = Path("outputs/figures/gtt_leaps_comparison_nav.png")
    plot_nav_growth(
        {
            "LEAPS QTR": base_qtr_result,
            "LEAPS Drift": base_drift_result,
            "GTT LEAPS Drift": gtt_result
        },
        output_path=output_path,
        position_mask=gtt_signal.position_mask,
    )
    print(f"Chart saved to {output_path}")
