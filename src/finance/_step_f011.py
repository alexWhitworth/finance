"""F-011: Pure NAV arithmetic helpers — _compute_nav_before_contrib and _compute_port_return.

Extracted from run_backtest lines 691 and 694. Both functions are pure arithmetic
with no side effects. PortfolioState is defined here pending integration of F-001
into finance.portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finance.leverage import LeapsContract, LeapsGttCloseEvent, LeapsLedger


@dataclass(frozen=True)
class PortfolioState:
    """Immutable snapshot of all mutable per-day loop state.

    Every mutation produces a new instance via dataclasses.replace().
    Accumulators (all_window_ledgers, all_gtt_closes) are tuples to ensure
    pure step functions.

    Attributes:
        holdings: Dollar value per base asset; governed assets zeroed during
            defensive windows.
        defensive_sleeve: Governed equity capital swept in during GTT defensive windows.
        leaps_pool: Force-closed LEAPS net proceeds parked during defensive windows.
        leaps_value: Current LEAPS mark-to-market value.
        prev_total_nav: End-of-day NAV from t-1; denominator of port_return calculation.
        prev_regime: GTT regime on t-1: 1=Long, 0=Defensive.
        prev_date_ts: Trading date of t-1; used for force-close spot/IV lookup.
        leaps_ledger: Active per-window LEAPS simulation ledger.
        leaps_scale: Surviving fraction per contract after drift partial-closes.
        all_window_ledgers: Immutable accumulator: one ledger per Long window.
        all_gtt_closes: Immutable accumulator: all GTT force-close events.
    """

    holdings: dict[str, float]
    defensive_sleeve: float
    leaps_pool: float
    leaps_value: float
    prev_total_nav: float
    prev_regime: int
    prev_date_ts: pd.Timestamp | None
    leaps_ledger: LeapsLedger | None
    leaps_scale: dict[LeapsContract, float]
    all_window_ledgers: tuple[LeapsLedger, ...]
    all_gtt_closes: tuple[LeapsGttCloseEvent, ...]


def _compute_nav_before_contrib(state: PortfolioState) -> float:
    """Return pre-contribution NAV: sum(holdings) + leaps_value + defensive_sleeve + leaps_pool.

    Arguments:
        state: Current PortfolioState (read-only).

    Returns:
        Pre-contribution NAV as a float.
    """
    return sum(state.holdings.values()) + state.leaps_value + state.defensive_sleeve + state.leaps_pool


def _compute_port_return(nav_before: float, prev_total_nav: float) -> float:
    """Return daily portfolio return excluding contributions.

    Arguments:
        nav_before: Pre-contribution NAV for today.
        prev_total_nav: End-of-day NAV from the prior trading day (state.prev_total_nav).

    Returns:
        Simple return: nav_before / prev_total_nav - 1.0.
    """
    return nav_before / prev_total_nav - 1.0
