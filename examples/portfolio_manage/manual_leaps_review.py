"""Manual portfolio review — user-entered holdings and LEAPS contracts.

Models the "cold start" path: a user types in their actual brokerage
holdings (VXUS, GLD, MUB, VGIT) and their actual DITM LEAPS contracts
directly, with no backtest involved (contrast with backtest_bridge.py,
which bridges from a BacktestResult via as_live_portfolio()). Computes
portfolio greeks for the LEAPS sleeve, then compares realized weights
against target weights — base assets and the LEAPS sleeve together —
under the DRIFT rebalance rule.

Usage:
    uv run examples/portfolio_manage/manual_leaps_review.py
"""

import pandas as pd

from finance import (
    LivePortfolio,
    compute_holdings_view,
    compute_nav_breakdown,
    compute_portfolio_greeks,
    compute_rebalance_plan,
)
from finance.consts import CONTRACT_MULTIPLIER
from finance.data import build_price_data
from finance.figures import (
    format_contract_greeks_table,
    format_holdings_table,
    format_nav_breakdown_table,
    format_trade_orders_table,
)
from finance.leverage import AccountType, RebalanceRule, build_leaps_contract
from finance.portfolio_manager import compute_leaps_holdings_view, leaps_trim_as_trade_order

# Target allocation: 40% deep-ITM VTI LEAPS sleeve, 60% diversifiers.
TARGET_WEIGHTS = {
    "VTI": 0.0,
    "VTI_LEAPS": 0.40,
    "VXUS": 0.20,
    "GLD": 0.10,
    "MUB": 0.15,
    "VGIT": 0.15,
}

# The user's actual current holdings in the diversifier sleeve (dollars).
# Deliberately drifted from target: GLD has run up, VXUS and MUB have lagged.
HOLDINGS = {
    "VTI": 0.0,
    "VXUS": 78_000.0,
    "GLD": 74_000.0,
    "MUB": 66_000.0,
    "VGIT": 72_000.0,
}


def _spot_at(price_series: pd.Series, date: pd.Timestamp) -> float:
    """Look up the most recent close on or before date.

    Arguments:
        price_series: Price Series with a DatetimeIndex.
        date: Date to look up.

    Returns:
        Close price on or before date.
    """
    return float(price_series.loc[:date].iloc[-1])


if __name__ == "__main__":
    START, END = "2024-06-01", "2026-08-19"

    print("=== Fetching VTI Price + Vol Data (for greeks and MTM) ===")
    price_data = build_price_data(
        START, END, tickers=["VTI"], use_splice=False, fetch_vol_indices=True
    )
    as_of = price_data.prices.index[-1]
    spot = float(price_data.prices["VTI"].iloc[-1])
    iv = float(price_data.vol_prices["VTI"].iloc[-1])
    print(f"as_of={as_of.date()}  VTI spot=${spot:.2f}  IV={iv:.1%}")

    print("\n=== Entering User's Actual LEAPS Contracts ===")
    vti_prices = price_data.prices["VTI"]
    purchase_a, expiry_a = pd.Timestamp("2025-01-15"), pd.Timestamp("2027-01-15")
    purchase_b, expiry_b = pd.Timestamp("2025-07-15"), pd.Timestamp("2027-07-15")
    contract_a = build_leaps_contract(
        purchase_a, expiry_a, _spot_at(vti_prices, purchase_a), 5.0,
        account_type=AccountType.TAX_SHELTERED,
    )
    contract_b = build_leaps_contract(
        purchase_b, expiry_b, _spot_at(vti_prices, purchase_b), 4.0,
        account_type=AccountType.TAX_SHELTERED,
    )
    leaps_contracts = ((contract_a, 1.0), (contract_b, 1.0))
    for c, _ in leaps_contracts:
        print(
            f"  {c.purchase_date.date()} -> {c.expiry_date.date()}: "
            f"{c.n_contracts:.1f} contracts, strike=${c.strike:.2f}, "
            f"spot_at_purchase=${c.spot_at_purchase:.2f}"
        )

    print("\n=== Building LivePortfolio from User Input ===")
    portfolio = LivePortfolio(
        as_of_date=as_of,
        holdings=HOLDINGS,
        target_weights=TARGET_WEIGHTS,
        leaps_contracts=leaps_contracts,
        gtt_regime=None,
    )

    print("\n=== LEAPS Portfolio Greeks ===")
    greeks = compute_portfolio_greeks(portfolio, spot=spot, iv=iv)
    print(format_contract_greeks_table(greeks))
    print(
        f"\nnet_delta={greeks.net_delta:,.1f} sh  net_gamma={greeks.net_gamma:.4f}  "
        f"net_vega=${greeks.net_vega:,.0f}  net_theta=${greeks.net_theta:,.2f}/day"
    )

    print("\n=== NAV Breakdown ===")
    # leaps_mtm is caller-supplied; derive it from the same greeks just computed
    # (price * n_contracts * CONTRACT_MULTIPLIER * leaps_scale per contract).
    leaps_mtm = sum(
        cg.price * cg.contract.n_contracts * CONTRACT_MULTIPLIER * cg.leaps_scale
        for cg in greeks.contracts
    )
    nav = compute_nav_breakdown(portfolio, leaps_mtm=leaps_mtm)
    print(format_nav_breakdown_table(nav))

    print("\n=== Holdings Drift vs. Target (base assets + LEAPS sleeve) ===")
    leaps_views = compute_leaps_holdings_view(portfolio, nav)
    holdings_view = (*compute_holdings_view(portfolio, nav), *leaps_views)
    print(format_holdings_table(holdings_view))

    print("\n=== DRIFT Rebalance Check (base assets + LEAPS sleeve) ===")
    plan = compute_rebalance_plan(
        portfolio, nav, RebalanceRule.DRIFT, is_rebal_date=False, is_month_end=True
    )
    print(f"would_trigger={plan.would_trigger}  trigger_reason={plan.trigger_reason}")
    leaps_trade = (
        leaps_trim_as_trade_order(leaps_views[0], plan.leaps_trim) if leaps_views else None
    )
    trades = (*plan.trades, leaps_trade) if leaps_trade is not None else plan.trades
    print(format_trade_orders_table(trades))
