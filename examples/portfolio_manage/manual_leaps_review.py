"""Manual portfolio review — user-entered holdings and multi-underlying LEAPS.

Models the "cold start" path: a user types in their actual brokerage
holdings (VXUS, MUB, VGIT) and their actual DITM LEAPS contracts on three
underlyings (VTI, IWM, GLD) directly, with no backtest involved (contrast
with backtest_bridge.py, which bridges from a BacktestResult via
as_live_portfolio()). Computes portfolio greeks for each LEAPS sleeve —
routing spot/IV per contract ticker, and on the surface path pricing each
contract against the per-ticker SSVI params and vol-index level — then
compares realized weights against target weights under the DRIFT rebalance
rule.

The surface path (D3/D5/D6) is used by default: per-ticker SsviParams come
from ``finance.consts.SSVI_PARAMS`` and the current vol-index level from the
last value of ``price_data.vol_prices[ticker]``. Any ticker absent from the
surface maps falls back to its scalar implied vol (cold starts).

Usage:
    uv run examples/portfolio_manage/manual_leaps_review.py        # live fetch
    uv run examples/portfolio_manage/manual_leaps_review.py --fixtures  # offline

``--fixtures`` builds the same VTI/IWM/GLD price + vol series from
deterministic synthetic data so the whole walkthrough runs without network
access (smoke tests use this path).

IWM's implied volatility comes from the CBOE Russell 2000 Volatility Index
(^RVX) via FRED series RVXCLS when FRED_API_KEY is set; otherwise it falls
back to ^VIX.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from finance import (
    LivePortfolio,
    compute_holdings_view,
    compute_nav_breakdown,
    compute_portfolio_greeks,
    compute_rebalance_plan,
)
from finance.consts import CONTRACT_MULTIPLIER, SSVI_PARAMS
from finance.data import PriceData, build_price_data
from finance.figures import (
    format_contract_greeks_table,
    format_holdings_table,
    format_nav_breakdown_table,
    format_trade_orders_table,
)
from finance.greeks import leaps_mtm_by_ticker
from finance.leverage import AccountType, RebalanceRule, build_leaps_contract
from finance.portfolio_manager import compute_leaps_holdings_view, leaps_trims_as_trade_orders

load_dotenv()

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

# Deterministic fixture levels (decimal) for the --fixtures path. These are
# in the listed-option IV range and exercise the skew across the three
# underlyings: IWM vol > VTI vol > GLD vol, with GLD possessing call skew.
FIXTURE_VOL_LEVELS = {"VTI": 0.18, "IWM": 0.22, "GLD": 0.15}


def _spot_at(price_series: pd.Series, date: pd.Timestamp) -> float:
    """Look up the most recent close on or before date.

    Arguments:
        price_series: Price Series with a DatetimeIndex.
        date: Date to look up.

    Returns:
        Close price on or before date.
    """
    return float(price_series.loc[:date].iloc[-1])


def _build_fixture_price_data(
    start: str, end: str, tickers: tuple[str, ...]
) -> PriceData:
    """Construct deterministic price + vol data for the fixture (--fixtures) path.

    Each underlying follows a smooth geometric-bridge drift from a hard-coded
    start level to a hard-coded end level, with per-ticker vol levels carried
    as flat deterministic columns so the surface path has a well-defined
    ``vol_prices`` and the output is reproducible with no network access.

    Arguments:
        start: Inclusive start date (YYYY-MM-DD).
        end: Inclusive end date (YYYY-MM-DD).
        tickers: Underlying tickers to synthesise.

    Returns:
        A PriceData whose prices/vol_prices are fully determined by the inputs.
    """
    idx = pd.bdate_range(start, end)
    drift = {
        "VTI": (230.0, 320.0),
        "IWM": (190.0, 250.0),
        "GLD": (200.0, 260.0),
    }
    prices = pd.DataFrame(index=idx, dtype=float)
    for t in tickers:
        start_level, end_level = drift[t]
        prices[t] = np.linspace(start_level, end_level, len(idx))
    vol_prices = pd.DataFrame(
        {t: FIXTURE_VOL_LEVELS[t] for t in tickers}, index=idx, dtype=float
    )
    return PriceData(
        prices=prices,
        dividends=pd.DataFrame(0.0, index=idx, columns=list(tickers)),
        vol_prices=vol_prices,
        tickers=tickers,
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
        ohlcv=pd.DataFrame(),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Arguments:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with a ``fixtures`` bool flag.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="build deterministic synthetic VTI/IWM/GLD price+vol data (no network)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    START, END = "2024-06-01", "2026-08-19"
    UNDERLYINGS = ("VTI", "IWM", "GLD")

    if args.fixtures:
        print(f"=== Building Fixture Price + Vol Data for {', '.join(UNDERLYINGS)} ===")
        price_data = _build_fixture_price_data(START, END, UNDERLYINGS)
    else:
        print(f"=== Fetching Price + Vol Data for {', '.join(UNDERLYINGS)} ===")
        price_data = build_price_data(
            START, END, tickers=list(UNDERLYINGS), use_splice=False, fetch_vol_indices=True
        )
    as_of = price_data.prices.index[-1]
    # IWM routes to the CBOE Russell 2000 Volatility Index (^RVX) via FRED RVXCLS.
    spot_map = {t: float(price_data.prices[t].iloc[-1]) for t in UNDERLYINGS}
    vol_index_map = {t: float(price_data.vol_prices[t].iloc[-1]) for t in UNDERLYINGS}
    params_map = {t: SSVI_PARAMS[t] for t in UNDERLYINGS if t in SSVI_PARAMS}
    # Cold-start fallback: any ticker without surface params uses its scalar IV.
    iv_map = {t: v for t, v in vol_index_map.items() if t not in params_map}
    print(
        f"as_of={as_of.date()}  spot={ {k: round(v, 2) for k, v in spot_map.items()} }  "
        f"vol_index={ {k: f'{v:.1%}' for k, v in vol_index_map.items()} }"
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

    print("\n=== LEAPS Portfolio Greeks (surface-routed per-ticker) ===")
    greeks = compute_portfolio_greeks(
        portfolio,
        spot=spot_map,
        iv=iv_map or None,
        params=params_map,
        vol_index=vol_index_map,
    )
    print(format_contract_greeks_table(greeks))
    for cg in greeks.contracts:
        mark = cg.price * cg.contract.n_contracts * CONTRACT_MULTIPLIER * cg.leaps_scale
        print(
            f"  {cg.contract.ticker}: sigma(k,t)={cg.iv:.2%}  mark=${mark:,.2f}  "
            f"headline_delta={cg.delta:.4f}  level_vega="
            f"{'n/a' if cg.level_vega is None else f'{cg.level_vega:.4f}'}"
        )
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
