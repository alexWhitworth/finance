"""Price DITM LEAPS and arbitrary options on the SSVI volatility surface.

A minimal, self-contained walkthrough of the surface path (D3/D4/D5/D6):

  1. fetch or synthesise spot + vol-index series for VTI / IWM / GLD;
  2. build real DITM LEAPS contracts per ticker and run them through
     ``compute_contract_greeks`` on the surface (sigma(k,t), sticky-moneyness
     delta, level-vega); and
  3. price an *arbitrary* strike/expiry (not necessarily the stored contract)
     from ``surface_implied_vol`` + Black-Scholes, with a side-by-side
     flat-vs-SSVI comparison.

Surface parameters are looked up per ticker from ``finance.consts.SSVI_PARAMS``;
the vol-index level comes from the last value of ``price_data.vol_prices[ticker]``
(already decimal for VTI/IWM/GLD — VIX-like indices are ÷100 at fetch).

Usage:
    uv run examples/portfolio_manage/ssvi_leaps.py            # live fetch
    uv run examples/portfolio_manage/ssvi_leaps.py --fixtures # offline
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from finance.consts import CONTRACT_MULTIPLIER, SSVI_PARAMS
from finance.data import PriceData, build_price_data
from finance.greeks import compute_contract_greeks
from finance.leverage import AccountType, bs_call_price, build_leaps_contract
from finance.vol_surface import surface_implied_vol

load_dotenv()

TICKERS = ("VTI", "IWM", "GLD")

# Deterministic fixture levels used by --fixtures (no network). The surrogate
# IVs land in the listed-option range and exercise the skew: IWM > VTI > GLD.
FIXTURE_SPOTS = {"VTI": 320.0, "IWM": 250.0, "GLD": 260.0}
FIXTURE_VOL_LEVELS = {"VTI": 0.18, "IWM": 0.22, "GLD": 0.15}

DIVIDEND_YIELD = {"VTI": 0.0106, "IWM": 0.0097, "GLD": 0.0}
RISK_FREE_RATE = 0.0456


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
        help="use deterministic synthetic spot/vol data (no network)",
    )
    return parser.parse_args(argv)


def _build_fixture_price_data(start: str, end: str) -> PriceData:
    """Build deterministic spot + flat vol-index data for the --fixtures path.

    Arguments:
        start: Inclusive start date (YYYY-MM-DD).
        end: Inclusive end date (YYYY-MM-DD).

    Returns:
        A PriceData whose ``vol_prices`` are flat fixture levels so the
        surface path is fully reproducible offline.

    """
    idx = pd.bdate_range(start, end)
    prices = pd.DataFrame(
        {t: np.full(len(idx), FIXTURE_SPOTS[t]) for t in TICKERS}, index=idx, dtype=float
    )
    vol_prices = pd.DataFrame(
        {t: FIXTURE_VOL_LEVELS[t] for t in TICKERS}, index=idx, dtype=float
    )
    return PriceData(
        prices=prices,
        dividends=pd.DataFrame(0.0, index=idx, columns=list(TICKERS)),
        vol_prices=vol_prices,
        tickers=TICKERS,
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
        ohlcv=pd.DataFrame(),
    )


def _print_contract_greeks(
    contract,
    spot: float,
    vol_index_level: float,
    as_of: pd.Timestamp,
) -> None:
    """Compute and print surface greeks for one DITM LEAPS contract.

    Arguments:
        contract: The LeapsContract being evaluated.
        spot: Current spot price of the contract's underlying.
        vol_index_level: Decimal vol-index level for the contract's ticker.
        as_of: Valuation date.

    """
    cg = compute_contract_greeks(
        contract,
        spot=spot,
        as_of_date=as_of,
        risk_free_rate=RISK_FREE_RATE,
        params=SSVI_PARAMS[contract.ticker],
        vol_index_level=vol_index_level,
    )
    bs_delta = cg.delta_sticky_strike if cg.delta_sticky_strike is not None else math.nan
    print(
        f"{contract.ticker:<6} {cg.iv:>9.2%} {contract.strike:>7.2f} "
        f"{contract.expiry_date.date()!s:>10} {cg.delta:>14.4f} "
        f"{bs_delta:>19.4f} "
        f"{cg.level_vega if cg.level_vega is not None else float('nan'):>10.4f}"
    )



if __name__ == "__main__":
    """Execute the SSVI walkthrough."""
    start, end = "2024-06-01", "2026-08-19"
    as_of = pd.Timestamp("2026-08-19")

    args = _parse_args()
    if args.fixtures:
        print("=== Building deterministic fixture data (no network) ===")
        price_data = _build_fixture_price_data(start, end)
    else:
        print("=== Fetching live spot + vol-index data ===")
        price_data = build_price_data(
            start, end, tickers=list(TICKERS), use_splice=False, fetch_vol_indices=True
        )

    spot = {t: float(price_data.prices[t].iloc[-1]) for t in TICKERS}
    vol_index = {t: float(price_data.vol_prices[t].iloc[-1]) for t in TICKERS}
    print(
        f"as_of={as_of.date()}  spot={ {k: f'{v:.2f}' for k, v in spot.items()} }  "
        f"vol_index={ {k: f'{v:.1%}' for k, v in vol_index.items()} }\n"
    )

    print("=== 1. DITM LEAPS contracts on the surface ===")
    print(
        f"{'ticker':<6} {'sigma(k,t)':>9} {'strike':>7} {'expiry':>10} "
        f"{'headline_delta':>14} {'delta_sticky_strike':>19} {'level_vega':>10}"
    )
    print("-" * 76)

    for ticker in TICKERS:
        purchase = pd.Timestamp("2025-01-15")
        spot_at_purchase = float(
            price_data.prices[ticker].loc[:purchase].iloc[-1]
        )
        contract = build_leaps_contract(
            purchase,
            pd.Timestamp("2027-01-15"),
            spot_at_purchase,
            5.0,
            account_type=AccountType.TAX_SHELTERED,
            dividend_yield=DIVIDEND_YIELD[ticker],
            ticker=ticker,
        )
        _print_contract_greeks(contract, spot[ticker], vol_index[ticker], as_of)

    print("\n=== 2. Arbitrary option on the surface (VTI) ===")
    ticker = "VTI"
    params = SSVI_PARAMS[ticker]
    s = spot[ticker]
    # DITM 50% strike, 2y expiry — same convention as the stored contract, but
    # any positive strike/expiry works identically; the params are per-ticker.
    strike = 0.50 * s
    t_years = 2.0
    sigma_ssvi = surface_implied_vol(
        spot=s,
        strike=strike,
        time_to_expiry_years=t_years,
        vol_index_level=vol_index[ticker],
        params=params,
        dividend_yield=DIVIDEND_YIELD[ticker],
        risk_free_rate=RISK_FREE_RATE,
    )
    sigma_flat = vol_index[ticker]
    price_ssvi = bs_call_price(
        s, strike, t_years, sigma_ssvi, RISK_FREE_RATE, DIVIDEND_YIELD[ticker]
    )
    price_flat = bs_call_price(
        s, strike, t_years, sigma_flat, RISK_FREE_RATE, DIVIDEND_YIELD[ticker]
    )
    forward = s * math.exp((RISK_FREE_RATE - DIVIDEND_YIELD[ticker]) * t_years)
    k = math.log(strike / forward)
    print(f"strike={strike:.2f}  spot={s:.2f}  t={t_years:.1f}y  k={k:+.4f}")
    print(f"  SSVI sigma(k,t) = {sigma_ssvi:.2%}   BS price = ${price_ssvi:.2f}/sh")
    print(
        f"  flat sigma      = {sigma_flat:.2%}   BS price = ${price_flat:.2f}/sh  "
        "(vol-index scalar as IV)"
    )
    print(
        f"  premium delta   = ${price_ssvi - price_flat:.2f}/sh on a "
        f"{CONTRACT_MULTIPLIER}-share multiplier"
    )