"""Manual portfolio review — user-entered holdings and multi-underlying LEAPS.

Models the "cold start" path: a user types in their actual brokerage
holdings (VXUS, MUB, VGIT) and their actual DITM LEAPS contracts on three
underlyings (VTI, IWM, GLD) directly, with no backtest involved (contrast
with backtest_bridge.py, which bridges from a BacktestResult via
as_live_portfolio()). Computes portfolio greeks for each LEAPS sleeve —
routing spot/IV per contract ticker — then compares realized weights against
target weights under the DRIFT rebalance rule.

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
from finance.data import build_price_data
from finance.figures import (
    format_contract_greeks_table,
    format_holdings_table,
    format_nav_breakdown_table,
    format_trade_orders_table,
)
from finance.greeks import leaps_mtm_by_ticker
from finance.leverage import AccountType, RebalanceRule, build_leaps_contract
from finance.portfolio_manager import compute_leaps_holdings_view, leaps_trims_as_trade_orders

# Target allocation: 40% total deep-ITM LEAPS sleeve across three underlyings
# (VTI 20%, IWM 10%, GLD 10%), 60% diversifiers.
TARGET_WEIGHTS = {
    "VTI": 0.0,
    "VTI_LEAPS": 0.20,
    "IWM_LEAPS": 0.10,
    "GLD_LEAPS": 0.10,
    "VXUS": 0.20,
    "MUB": 0.20,
    "VGIT": 0.20,
}

# The user's actual current holdings in the diversifier sleeve (dollars).
# Deliberately drifted from target: VXUS and MUB have lagged, VGIT has run up.
HOLDINGS = {
    "VTI": 0.0,
    "VXUS": 78_000.0,
    "MUB": 66_000.0,
    "VGIT": 88_000.0,
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
    UNDERLYINGS = ["VTI", "IWM", "GLD"]

    print(f"=== Fetching Price + Vol Data for {', '.join(UNDERLYINGS)} ===")
    price_data = build_price_data(
        START, END, tickers=UNDERLYINGS, use_splice=False, fetch_vol_indices=True
    )
    as_of = price_data.prices.index[-1]
    # IWM has no dedicated vol index; proxy it with VTI's (U.S. equity via ^VIX).
    spot_map = {t: float(price_data.prices[t].iloc[-1]) for t in UNDERLYINGS}
    iv_map = {
        "VTI": float(price_data.vol_prices["VTI"].iloc[-1]),
        "IWM": float(price_data.vol_prices["VTI"].iloc[-1]),
        "GLD": float(price_data.vol_prices["GLD"].iloc[-1]),
    }
    print(
        f"as_of={as_of.date()}  spot={ {k: round(v, 2) for k, v in spot_map.items()} }  "
        f"iv={ {k: f'{v:.1%}' for k, v in iv_map.items()} }"
    )

    print("\n=== Entering User's Actual LEAPS Contracts (one per underlying) ===")
    vti_prices = price_data.prices["VTI"]
    iwm_prices = price_data.prices["IWM"]
    gld_prices = price_data.prices["GLD"]
    purchase_a, expiry_a = pd.Timestamp("2025-01-15"), pd.Timestamp("2027-01-15")
    purchase_b, expiry_b = pd.Timestamp("2025-07-15"), pd.Timestamp("2027-07-15")
    contract_vti = build_leaps_contract(
        purchase_a, expiry_a, _spot_at(vti_prices, purchase_a), 5.0,
        account_type=AccountType.TAX_SHELTERED, ticker="VTI",
    )
    contract_iwm = build_leaps_contract(
        purchase_b, expiry_b, _spot_at(iwm_prices, purchase_b), 4.0,
        account_type=AccountType.TAX_SHELTERED, ticker="IWM",
    )
    contract_gld = build_leaps_contract(
        purchase_a, expiry_a, _spot_at(gld_prices, purchase_a), 6.0,
        account_type=AccountType.TAX_SHELTERED, ticker="GLD",
    )
    leaps_contracts = ((contract_vti, 1.0), (contract_iwm, 1.0), (contract_gld, 1.0))
    for c, _ in leaps_contracts:
        print(
            f"  {c.ticker}  {c.purchase_date.date()} -> {c.expiry_date.date()}: "
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

    print("\n=== LEAPS Portfolio Greeks (routed per-ticker) ===")
    greeks = compute_portfolio_greeks(portfolio, spot=spot_map, iv=iv_map)
    print(format_contract_greeks_table(greeks))
    print(
        f"\nnet_delta={greeks.net_delta:,.1f} sh  net_gamma={greeks.net_gamma:.4f}  "
        f"net_vega=${greeks.net_vega:,.0f}  net_theta=${greeks.net_theta:,.2f}/day"
    )

    print("\n=== NAV Breakdown (per-ticker LEAPS MTM) ===")
    mtm_by_ticker = leaps_mtm_by_ticker(greeks)
    nav = compute_nav_breakdown(portfolio, leaps_mtm_by_ticker=mtm_by_ticker)
    print(format_nav_breakdown_table(nav))
    print(
        "  leaps_nav_by_ticker: "
        + ", ".join(f"{k}=${v:,.0f}" for k, v in nav.leaps_nav_by_ticker.items())
    )

    print("\n=== Holdings Drift vs. Target (base assets + LEAPS sleeves) ===")
    leaps_views = compute_leaps_holdings_view(portfolio, nav)
    holdings_view = (*compute_holdings_view(portfolio, nav), *leaps_views)
    print(format_holdings_table(holdings_view))

    print("\n=== DRIFT Rebalance Check (base assets + LEAPS sleeves) ===")
    plan = compute_rebalance_plan(
        portfolio, nav, RebalanceRule.DRIFT, is_rebal_date=False, is_month_end=True
    )
    print(f"would_trigger={plan.would_trigger}  trigger_reason={plan.trigger_reason}")
    leaps_trades = leaps_trims_as_trade_orders(leaps_views, plan.leaps_trim)
    trades = (*plan.trades, *leaps_trades) if leaps_trades else plan.trades
    print(format_trade_orders_table(trades))