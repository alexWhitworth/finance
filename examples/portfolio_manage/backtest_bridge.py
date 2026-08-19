"""Live portfolio management walkthrough — the weekly LivePortfolio workflow.

Runs a short LEAPS-overlay backtest to produce a realistic BacktestResult,
bridges it to a LivePortfolio via as_live_portfolio(), then walks through
the live management pipeline from plans/portfolio_management.md: NAV
breakdown, holdings drift, a rebalance simulation, LEAPS greeks, and a
volatility contribution report.

Usage:
    uv run examples/portfolio_manage/backtest_bridge.py
"""

from finance import (
    LivePortfolio,
    as_live_portfolio,
    compute_holdings_view,
    compute_nav_breakdown,
    compute_portfolio_greeks,
    compute_rebalance_plan,
    compute_volatility_report,
)
from finance._portfolio_types import PortfolioConfig
from finance.data import build_price_data, fetch_risk_free_rate
from finance.figures import (
    format_holdings_table,
    format_nav_breakdown_table,
    format_trade_orders_table,
)
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance.portfolio import run_backtest
from finance.portfolio_manager import compute_leaps_holdings_view
from finance.returns import build_return_data

WEIGHTS = {
    "VTI": 0.0,
    "VTI_LEAPS": 0.40,
    "VXUS": 0.20,
    "GLD": 0.10,
    "MUB": 0.15,
    "VGIT": 0.15,
}
INITIAL_NAV = 1_000_000.0
MONTHLY_CONTRIBUTION = 10_000.0
FLOOR_IV = 0.1
LTCG_RATE = 0.238


if __name__ == "__main__":
    START, END = "2000-09-01", "2026-06-30"

    print("=== Fetching Price Data ===")
    tickers = [t for t in WEIGHTS if t != "VTI_LEAPS"]
    price_data = build_price_data(START, END, tickers=tickers, use_splice=True)

    print("=== Fetching Risk-Free Rate ===")
    rfr_series = fetch_risk_free_rate(START, END)

    print("=== Building Returns ===")
    return_data = build_return_data(price_data, risk_free_series=rfr_series)

    print("=== Running Backtest (seeds a realistic end-of-history portfolio state) ===")
    config = PortfolioConfig(
        target_weights=WEIGHTS,
        initial_nav=INITIAL_NAV,
        monthly_contribution=MONTHLY_CONTRIBUTION,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(
            iv=FLOOR_IV, ltcg_rate=LTCG_RATE, account_type=AccountType.TAX_SHELTERED
        ),
    )
    result = run_backtest(return_data, price_data, config)

    print("\n=== Bridging BacktestResult -> LivePortfolio ===")
    portfolio: LivePortfolio = as_live_portfolio(result)
    print(
        f"as_of_date={portfolio.as_of_date.date()}  "
        f"n_leaps_contracts={len(portfolio.leaps_contracts)}"
    )

    print("\n=== NAV Breakdown ===")
    # leaps_mtm is caller-supplied (e.g. from a brokerage statement); here we reuse
    # the backtest's own mark-to-market rather than recomputing it via greeks.
    nav = compute_nav_breakdown(portfolio, leaps_mtm=result.final_state.leaps_value)
    print(format_nav_breakdown_table(nav))

    print("\n=== Holdings Drift (base assets + LEAPS sleeve) ===")
    holdings_view = (
        *compute_holdings_view(portfolio, nav),
        *compute_leaps_holdings_view(portfolio, nav),
    )
    print(format_holdings_table(holdings_view))

    print("\n=== Rebalance Simulation (assume today is a scheduled quarterly date) ===")
    plan = compute_rebalance_plan(
        portfolio, nav, RebalanceRule.QUARTERLY, is_rebal_date=True, is_month_end=True
    )
    print(f"would_trigger={plan.would_trigger}  trigger_reason={plan.trigger_reason}")
    print(format_trade_orders_table(plan.trades))

    print("\n=== LEAPS Greeks ===")
    if portfolio.leaps_contracts:
        spot = float(price_data.prices["VTI"].iloc[-1])
        greeks = compute_portfolio_greeks(portfolio, spot=spot, iv=FLOOR_IV)
        print(
            f"net_delta={greeks.net_delta:,.1f} sh  net_gamma={greeks.net_gamma:.4f}  "
            f"net_vega=${greeks.net_vega:,.0f}  net_theta=${greeks.net_theta:,.2f}/day"
        )
    else:
        print("No live LEAPS contracts as of this date.")

    print("\n=== Volatility Report ===")
    vol_report = compute_volatility_report(portfolio, return_data)
    print(f"Forecasted portfolio vol: {vol_report.portfolio_vol:.2%}\n")
    print(vol_report.contribution_table.round(4).to_string())
