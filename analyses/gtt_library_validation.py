"""F-11: Directional validation of the multi-asset GTT portfolio.

Runs GTT-on vs GTT-off backtests over 1993-2026 and asserts directional
risk-reduction behavior:

  1. GTT-on max drawdown < GTT-off max drawdown (closer to zero).
  2. GTT-on Sharpe >= GTT-off Sharpe.
  3. GTT-on Sortino >= GTT-off Sortino.
  4. During the 2001 and 2008 employment-driven recession windows, GTT-on
     cumulative return exceeds GTT-off (positive excess).

The EDA (outputs/gtt_findings.md, Table 5.1) used a 100% S&P 500 portfolio
and is a QUALITATIVE REFERENCE ONLY. This script validates the library's
multi-asset portfolio (VTI + VTI_LEAPS + KMLM/VGIT/GLD defensive sleeve)
against itself with GTT toggled on vs off.

Exits non-zero if any assertion fails.

Usage:
    uv run analyses/gtt_library_validation.py 2>&1 | tee outputs/gtt_validation.log

Notes:
    - Requires a FRED_API_KEY environment variable for the UNRATE fetch, or a
      keyless FRED connection (rate-limited but functional for one-off runs).
    - VIX P90 threshold (0.272) is computed from the full 1993-2026 VIX history,
      which introduces a mild look-ahead bias in the threshold itself. This is
      documented per assumption A1 in plans/implement_gtt.md; the library does
      not protect against it by design.
    - 2022 rate-driven bear is a known weak spot for the combined signal (UE blind,
      VIX late). The recession-window assertions are therefore scoped to the
      employment-driven recessions only (2001, 2008).
"""

import sys

import pandas as pd

from finance.data import build_price_data, fetch_risk_free_rate
from finance.gtt import fetch_gtt_signal_data
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance.metrics import max_drawdown, sharpe_ratio, sortino_ratio
from finance.portfolio import GttConfig, PortfolioConfig, run_backtest
from finance.returns import build_return_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

START_DATE = "1993-01-01"
END_DATE = "2026-06-30"

# VIX P90 threshold from the full 1993-2026 window (documented look-ahead
# per assumption A1; the library does not protect against this by design).
VIX_P90_THRESHOLD = 0.272

# Multi-asset target weights: VTI (base equity) + VTI_LEAPS carve-out +
# diversified bond/commodity sleeve + international equity.
TARGET_WEIGHTS: dict[str, float] = {
    "VTI": 0.10,
    "VTI_LEAPS": 0.30,
    "VXUS": 0.15,
    "GLD": 0.15,
    "MUB": 0.15,
    "KMLM": 0.075,
    "VGIT": 0.075,
}

DEFENSIVE_WEIGHTS: dict[str, float] = {
    "R_f": 0.25,
    "KMLM": 0.25,
    "VGIT": 0.25,
    "GLD": 0.25,
}

INITIAL_NAV = 1_000_000.0
MONTHLY_CONTRIBUTION = 2_000.0

# Employment-driven recession windows for recession-excess assertions.
# 2022 is deliberately excluded (rate-driven bear, UE blind).
RECESSION_WINDOWS: dict[str, tuple[str, str]] = {
    "2001": ("2001-03-01", "2001-11-30"),
    "2008": ("2007-12-01", "2009-06-30"),
}

# ---------------------------------------------------------------------------
# I/O — fetch all data once
# ---------------------------------------------------------------------------


def _fetch_data() -> tuple[object, object, object]:
    """Fetch prices, risk-free rate, and GTT signal. Returns (pd, rd, gtt_signal)."""
    print(f"Fetching prices {START_DATE} -> {END_DATE} …")
    tickers = [t for t in TARGET_WEIGHTS if not t.endswith("_LEAPS")]
    price_data = build_price_data(
        START_DATE,
        END_DATE,
        tickers=tickers,
        use_splice=True,
        fetch_vol_indices=True,
    )

    print("Fetching risk-free rate …")
    rfr = fetch_risk_free_rate(START_DATE, END_DATE)

    print("Computing return data …")
    return_data = build_return_data(price_data, apply_tey=True, risk_free_series=rfr)

    print("Fetching GTT signals (FRED + yfinance) …")
    vti_prices: pd.Series = price_data.prices["VTI"].rename("VTI")
    gtt_signal = fetch_gtt_signal_data(
        START_DATE,
        END_DATE,
        vix_p90_threshold=VIX_P90_THRESHOLD,
        equity_prices=vti_prices,
    )

    n_defensive = int((gtt_signal.position_mask == 0).sum())
    n_total = len(gtt_signal.position_mask)
    print(f"  Signal: {n_defensive}/{n_total} defensive days "
          f"({100.0 * n_defensive / n_total:.1f}%)")

    return price_data, return_data, gtt_signal


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _make_gtt_config() -> PortfolioConfig:
    """Full multi-asset config with GTT overlay."""
    return PortfolioConfig(
        target_weights=dict(TARGET_WEIGHTS),
        initial_nav=INITIAL_NAV,
        monthly_contribution=MONTHLY_CONTRIBUTION,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(account_type=AccountType.TAXABLE),
        gtt_config=GttConfig(
            vix_p90_threshold=VIX_P90_THRESHOLD,
            defensive_weights=dict(DEFENSIVE_WEIGHTS),
        ),
    )


def _make_baseline_config() -> PortfolioConfig:
    """Same multi-asset config, GTT disabled (buy-and-hold baseline)."""
    return PortfolioConfig(
        target_weights=dict(TARGET_WEIGHTS),
        initial_nav=INITIAL_NAV,
        monthly_contribution=MONTHLY_CONTRIBUTION,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(account_type=AccountType.TAXABLE),
        gtt_config=None,
    )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _recession_cumulative_return(nav: pd.Series, start: str, end: str) -> float:
    """Cumulative return of nav_series over the recession window."""
    window = nav.loc[start:end]
    if len(window) < 2:
        return 0.0
    return float(window.iloc[-1] / window.iloc[0] - 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run GTT-on vs GTT-off directional validation. Returns exit code (0=pass)."""
    price_data, return_data, gtt_signal = _fetch_data()

    # ---- GTT-on backtest ----
    print("\nRunning GTT-on backtest …")
    gtt_result = run_backtest(
        return_data, price_data, _make_gtt_config(), gtt_signal=gtt_signal
    )

    # ---- GTT-off (buy-and-hold) backtest ----
    print("Running GTT-off (buy-and-hold) backtest …")
    base_result = run_backtest(return_data, price_data, _make_baseline_config())

    rfr = return_data.risk_free_rate

    gtt_mdd = max_drawdown(gtt_result.nav_series)
    base_mdd = max_drawdown(base_result.nav_series)
    gtt_sharpe = sharpe_ratio(gtt_result.return_series, rfr)
    base_sharpe = sharpe_ratio(base_result.return_series, rfr)
    gtt_sortino = sortino_ratio(gtt_result.return_series, rfr)
    base_sortino = sortino_ratio(base_result.return_series, rfr)

    print("\n" + "=" * 62)
    print(f"{'Metric':<28} {'GTT-on':>10} {'GTT-off':>10} {'Pass?':>8}")
    print("-" * 62)

    failures: list[str] = []

    # AC-1: max drawdown
    mdd_pass = gtt_mdd < base_mdd
    print(f"{'Max Drawdown':<28} {gtt_mdd:>10.4f} {base_mdd:>10.4f} "
          f"{'PASS' if mdd_pass else 'FAIL':>8}")
    if not mdd_pass:
        failures.append(
            f"FAIL: GTT-on max drawdown {gtt_mdd:.4f} is not less than "
            f"GTT-off {base_mdd:.4f}"
        )

    # AC-2: Sharpe
    sharpe_pass = gtt_sharpe >= base_sharpe
    print(f"{'Sharpe Ratio':<28} {gtt_sharpe:>10.4f} {base_sharpe:>10.4f} "
          f"{'PASS' if sharpe_pass else 'FAIL':>8}")
    if not sharpe_pass:
        failures.append(
            f"FAIL: GTT-on Sharpe {gtt_sharpe:.4f} < GTT-off {base_sharpe:.4f}"
        )

    # AC-3: Sortino
    sortino_pass = gtt_sortino >= base_sortino
    print(f"{'Sortino Ratio':<28} {gtt_sortino:>10.4f} {base_sortino:>10.4f} "
          f"{'PASS' if sortino_pass else 'FAIL':>8}")
    if not sortino_pass:
        failures.append(
            f"FAIL: GTT-on Sortino {gtt_sortino:.4f} < GTT-off {base_sortino:.4f}"
        )

    # AC-4: recession-window excess
    print("-" * 62)
    for label, (start, end) in RECESSION_WINDOWS.items():
        gtt_ret = _recession_cumulative_return(gtt_result.nav_series, start, end)
        base_ret = _recession_cumulative_return(base_result.nav_series, start, end)
        excess = gtt_ret - base_ret
        recession_pass = excess > 0.0
        print(f"{'Recession excess ' + label:<28} {gtt_ret:>10.4f} {base_ret:>10.4f} "
              f"{'PASS' if recession_pass else 'FAIL':>8}  (excess={excess:+.4f})")
        if not recession_pass:
            failures.append(
                f"FAIL: {label} recession excess is not positive "
                f"(GTT-on {gtt_ret:.4f}, GTT-off {base_ret:.4f})"
            )

    # Informational: terminal NAV and GTT close event count
    gtt_nav = gtt_result.nav_series.iloc[-1]
    base_nav = base_result.nav_series.iloc[-1]
    print("-" * 62)
    print(f"{'Terminal NAV (GTT-on)':<28} {gtt_nav:>10,.0f}")
    print(f"{'Terminal NAV (GTT-off)':<28} {base_nav:>10,.0f}")
    if gtt_result.leaps_ledger is not None:
        n_closes = len(gtt_result.leaps_ledger.gtt_close_events)
        total_tax = sum(e.tax_paid for e in gtt_result.leaps_ledger.gtt_close_events)
        print(f"{'GTT force-close events':<28} {n_closes:>10}")
        print(f"{'GTT close tax drag ($)':<28} {total_tax:>10,.0f}")

    print("=" * 62)

    if failures:
        print("\nFAILURES:")
        for msg in failures:
            print(f"  {msg}")
        print(f"\nResult: {len(failures)} assertion(s) FAILED")
        return 1

    print("\nResult: All directional assertions PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
