"""Volatility contribution report for a 6-asset diversified portfolio.

Fetches prices, builds a VolatilityModel, prints the full vol contribution
table (90-day realized vol, EWMA vol, correlation with VTI, and contribution
fraction), prints the forward portfolio vol forecast, and saves a vol
contributions bar chart to figures/vol_contributions.png.
"""

from pathlib import Path

import pandas as pd

from finance._portfolio_types import PortfolioConfig
from finance.data import build_price_data, fetch_risk_free_rate
from finance.figures import plot_vol_contributions
from finance.leverage import RebalanceRule, WeightStrategy, AccountType, LeapsConfig
from finance.metrics import build_performance_report
from finance.portfolio import run_backtest
from finance.returns import build_return_data
from finance.volatility import (
    build_vol_contribution_table,
    build_volatility_model,
    forecast_portfolio_vol,
)

WEIGHTS = pd.Series(
    {
        "VTI_LEAPS": 0.4,
        "VXUS": 0.25,
        "GLD": 0.10,
        "MUB": 0.10,
        "KMLM": 0.10,
        "VGIT": 0.05,
    }
)


def _format_vol_table(tbl: pd.DataFrame) -> pd.DataFrame:
    """Format the raw vol contribution table for human-readable display.

    Arguments:
        tbl: DataFrame with columns [sigma_tilde, sigma_hat, rho_VTI, contrib]
            indexed by asset name.

    Returns:
        Formatted DataFrame with renamed columns and percentage/decimal strings.
    """
    display = pd.DataFrame(index=tbl.index)
    display.index.name = "Asset"
    display["σ̃_k (90d realized)"] = tbl["sigma_tilde"].map(lambda x: f"{x:.1%}")  # noqa: RUF001
    display["σ̂_k (EWMA)"] = tbl["sigma_hat"].map(lambda x: f"{x:.1%}")  # noqa: RUF001
    display["ρ̂_VTI"] = tbl["rho_VTI"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")  # noqa: RUF001
    display["Contrib"] = tbl["contrib"].map(lambda x: f"{x:.1%}")
    return display


if __name__ == "__main__":
    START, END = "2000-09-01", "2026-06-30"

    print("=== Fetching Price Data ===")
    tickers = [t for t in list(WEIGHTS.keys()) if t != 'VTI_LEAPS'] + ["VTI"]
    price_data = build_price_data(START, END, tickers=tickers, use_splice=True)

    print("=== Fetching Risk-Free Rate ===")
    rfr_series = fetch_risk_free_rate(START, END)

    print("=== Building Returns ===")
    return_data = build_return_data(price_data, risk_free_series=rfr_series)

    print("=== Building Volatility Model ===")
    vol_model = build_volatility_model(return_data)

    print("=== Vol Contribution Table ===")
    vol_table = build_vol_contribution_table(WEIGHTS, return_data, vol_model)
    with pd.option_context("display.max_colwidth", 20):
        print(_format_vol_table(vol_table).to_string())

    fwd_vol = forecast_portfolio_vol(WEIGHTS, vol_model)
    print(f"\nForward portfolio vol forecast: {fwd_vol:.2%}")

    print("\n=== Running Backtest (for performance report) ===")
    config = PortfolioConfig(
        target_weights={str(k): v for k, v in WEIGHTS.items()},
        initial_nav=1_000_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(
            iv=0.1,
            ltcg_rate=0.238,
            account_type=AccountType.TAX_SHELTERED,
        ),
    )
    result = run_backtest(return_data, price_data, config)
    report = build_performance_report(result, price_data, return_data, vol_model)

    print("=== Saving Vol Contributions Chart ===")
    output_path = "outputs/figures/vol_contributions.png"
    plot_vol_contributions(report, output_path=Path(output_path))
    print(f"Chart saved to {output_path}")
