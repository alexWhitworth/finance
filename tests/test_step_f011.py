"""Tests for F-011: _compute_nav_before_contrib and _compute_port_return.

Pure arithmetic functions — tested precisely with known values and property-based
strategies to verify the accounting oracle and plausibility invariants.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from finance._step_f011 import (
    PortfolioState,
    _compute_nav_before_contrib,
    _compute_port_return,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_state(
    holdings: dict[str, float],
    leaps_value: float = 0.0,
    sleeve: float = 0.0,
    pool: float = 0.0,
    prev_nav: float | None = None,
) -> PortfolioState:
    """Construct a minimal PortfolioState for unit testing.

    Arguments:
        holdings: Dollar values per asset ticker.
        leaps_value: Current LEAPS mark-to-market.
        sleeve: Defensive sleeve balance.
        pool: LEAPS pool balance.
        prev_nav: Prior-day NAV; defaults to sum of all components.

    Returns:
        PortfolioState with all required fields populated.
    """
    return PortfolioState(
        holdings=holdings,
        defensive_sleeve=sleeve,
        leaps_pool=pool,
        leaps_value=leaps_value,
        prev_total_nav=prev_nav or sum(holdings.values()) + leaps_value + sleeve + pool,
        prev_regime=1,
        prev_date_ts=None,
        leaps_ledger=None,
        leaps_scale={},
        all_window_ledgers=(),
        all_gtt_closes=(),
    )


# ---------------------------------------------------------------------------
# _compute_nav_before_contrib — deterministic tests
# ---------------------------------------------------------------------------


def test_nav_before_known_state() -> None:
    """Known state: sum of all components equals expected NAV."""
    state = make_state(
        holdings={"VTI": 50000.0, "VXUS": 20000.0},
        leaps_value=8000.0,
        sleeve=5000.0,
        pool=2000.0,
    )
    assert _compute_nav_before_contrib(state) == pytest.approx(85000.0)


def test_nav_before_all_zero() -> None:
    """All-zero state returns exactly 0.0."""
    state = make_state(
        holdings={"VTI": 0.0, "VXUS": 0.0},
        leaps_value=0.0,
        sleeve=0.0,
        pool=0.0,
        prev_nav=1.0,  # avoid division-by-zero in make_state default
    )
    assert _compute_nav_before_contrib(state) == 0.0


def test_nav_before_no_gtt_leaps() -> None:
    """Without GTT/LEAPS, result equals sum of holdings alone."""
    holdings = {"VTI": 40000.0, "VXUS": 25000.0, "BND": 10000.0}
    state = make_state(holdings=holdings, sleeve=0.0, pool=0.0, leaps_value=0.0)
    assert _compute_nav_before_contrib(state) == pytest.approx(sum(holdings.values()))


# ---------------------------------------------------------------------------
# _compute_port_return — deterministic tests
# ---------------------------------------------------------------------------


def test_port_return_positive() -> None:
    """5 % gain: nav_before=10500, prev=10000."""
    assert _compute_port_return(10500.0, 10000.0) == pytest.approx(0.05, rel=1e-14)


def test_port_return_negative() -> None:
    """5 % loss: nav_before=9500, prev=10000."""
    assert _compute_port_return(9500.0, 10000.0) == pytest.approx(-0.05, rel=1e-14)


# ---------------------------------------------------------------------------
# Hypothesis: _compute_nav_before_contrib
# ---------------------------------------------------------------------------


@given(
    holdings_vals=st.fixed_dictionaries(
        {"VTI": st.floats(0.0, 1e6), "VXUS": st.floats(0.0, 1e6)}
    ),
    leaps=st.floats(0.0, 1e5),
    sleeve=st.floats(0.0, 1e5),
    pool=st.floats(0.0, 1e5),
)
def test_nav_before_equals_sum_of_components(
    holdings_vals: dict[str, float],
    leaps: float,
    sleeve: float,
    pool: float,
) -> None:
    """nav_before equals the explicit sum of all four components."""
    state = make_state(
        holdings=holdings_vals,
        leaps_value=leaps,
        sleeve=sleeve,
        pool=pool,
        prev_nav=1.0,  # arbitrary; not used by this function
    )
    expected = holdings_vals["VTI"] + holdings_vals["VXUS"] + leaps + sleeve + pool
    assert _compute_nav_before_contrib(state) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Hypothesis: _compute_port_return
# ---------------------------------------------------------------------------


@given(
    nav_before=st.floats(min_value=1.0, max_value=1e7),
    prev=st.floats(min_value=1.0, max_value=1e7),
)
def test_port_return_above_minus_one(nav_before: float, prev: float) -> None:
    """For any positive NAV inputs, port_return > -1.0."""
    assert _compute_port_return(nav_before, prev) > -1.0
