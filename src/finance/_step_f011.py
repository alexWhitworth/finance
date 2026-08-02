"""F-011: Pure NAV arithmetic helpers — _compute_nav_before_contrib and _compute_port_return.

Extracted from run_backtest lines 691 and 694. Both functions are pure arithmetic
with no side effects.
"""

from __future__ import annotations

from finance.portfolio import PortfolioState


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
