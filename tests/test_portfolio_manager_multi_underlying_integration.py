"""Integration tests for F-003: multi-underlying NAV, drift, and rebalance trim.

Covers the M1, M4, and M5 invariants end-to-end across two LEAPS sleeves with
divergent realized MTM under equal target weights.
"""

from __future__ import annotations

import pandas as pd

from finance.consts import CONTRACT_MULTIPLIER
from finance.greeks import compute_portfolio_greeks, leaps_mtm_by_ticker
from finance.leverage import AccountType, LeapsContract, RebalanceRule, build_leaps_contract
from finance.portfolio_manager import (
    compute_holdings_view,
    compute_leaps_holdings_view,
    compute_nav_breakdown,
    compute_rebalance_plan,
    LivePortfolio,
)

_AS_OF = pd.Timestamp("2025-06-01")
_PURCHASE = pd.Timestamp("2024-01-15")
_EXPIRY = pd.Timestamp("2026-01-15")


def _tagged_contract(ticker: str, n_contracts: float) -> LeapsContract:
    """Build a LeapsContract tagged with *ticker* for the integration fixture."""
    return build_leaps_contract(
        _PURCHASE, _EXPIRY, 200.0, n_contracts, ticker=ticker, account_type=AccountType.TAXABLE
    )


def _divergent_portfolio() -> tuple[LivePortfolio, dict[str, float]]:
    """Portfolio with two equal-target LEAPS sleeves at divergent realized MTM.

    Base NAV = 60k. VTI_LEAPS ran up to 26k; IWM_LEAPS stayed flat at 20k.
    Equal 20% target weights would mask this divergence under the old
    target-proportional split (both would show 23k).
    """
    c_vti = _tagged_contract("VTI", 2.0)
    c_iwm = _tagged_contract("IWM", 2.0)
    portfolio = LivePortfolio(
        as_of_date=_AS_OF,
        holdings={"VXUS": 30_000.0, "GLD": 30_000.0},
        target_weights={
            "VXUS": 0.30,
            "GLD": 0.30,
            "VTI_LEAPS": 0.20,
            "IWM_LEAPS": 0.20,
        },
        leaps_contracts=((c_vti, 1.0), (c_iwm, 1.0)),
        gtt_regime=None,
    )
    return portfolio, {"VTI": 26_000.0, "IWM": 20_000.0}


def test_m1_leaps_nav_equals_sum_of_by_ticker() -> None:
    """M1: leaps_nav == sum(leaps_nav_by_ticker.values()) within 1e-9."""
    portfolio, mtm = _divergent_portfolio()
    nav = compute_nav_breakdown(portfolio, leaps_mtm_by_ticker=mtm)
    assert nav.leaps_nav == 46_000.0
    assert nav.leaps_nav_by_ticker == {"VTI": 26_000.0, "IWM": 20_000.0}
    assert abs(nav.leaps_nav - sum(nav.leaps_nav_by_ticker.values())) < 1e-9


def test_m4_holdings_view_reflects_actual_per_ticker_exposure() -> None:
    """M4: divergent MTM yields unequal actual_weight, not target-proportional split."""
    portfolio, mtm = _divergent_portfolio()
    nav = compute_nav_breakdown(portfolio, leaps_mtm_by_ticker=mtm)

    views = {v.ticker: v for v in compute_leaps_holdings_view(portfolio, nav)}

    total_nav = nav.total_nav  # 60k + 46k = 106k
    assert len(views) == 2
    # Old behavior would report equal dollar_value (23k each); new reports actual.
    assert views["VTI_LEAPS"].dollar_value == 26_000.0
    assert views["IWM_LEAPS"].dollar_value == 20_000.0
    assert views["VTI_LEAPS"].actual_weight == 26_000.0 / total_nav
    assert views["IWM_LEAPS"].actual_weight == 20_000.0 / total_nav
    assert views["VTI_LEAPS"].actual_weight != views["IWM_LEAPS"].actual_weight


def test_m5_leaps_trim_contains_only_overweight_tickers() -> None:
    """M5: leaps_trim keys only tickers whose sleeve is actually overweight."""
    c_vti = _tagged_contract("VTI", 2.0)
    c_iwm = _tagged_contract("IWM", 2.0)
    portfolio = LivePortfolio(
        as_of_date=_AS_OF,
        holdings={"VXUS": 30_000.0, "GLD": 30_000.0},
        target_weights={
            "VXUS": 0.20,
            "GLD": 0.20,
            "VTI_LEAPS": 0.30,
            "IWM_LEAPS": 0.30,
        },
        leaps_contracts=((c_vti, 1.0), (c_iwm, 1.0)),
        gtt_regime=None,
    )
    # total_nav = 60k + 40k + 20k = 120k.
    # VTI_LEAPS target 36k vs actual 40k → overweight by 4k.
    # IWM_LEAPS target 36k vs actual 20k → underweight, must be absent.
    nav = compute_nav_breakdown(
        portfolio, leaps_mtm_by_ticker={"VTI": 40_000.0, "IWM": 20_000.0}
    )
    plan = compute_rebalance_plan(
        portfolio, nav, RebalanceRule.QUARTERLY, is_rebal_date=True, is_month_end=False
    )
    assert plan.leaps_trim == {"VTI": 4_000.0}
    assert "IWM" not in plan.leaps_trim


def test_m5_leaps_trim_empty_when_none_overweight() -> None:
    """M5: empty leaps_trim when no sleeve is overweight."""
    c_vti = _tagged_contract("VTI", 2.0)
    portfolio = LivePortfolio(
        as_of_date=_AS_OF,
        holdings={"VXUS": 40_000.0},
        target_weights={"VXUS": 0.60, "VTI_LEAPS": 0.40},
        leaps_contracts=((c_vti, 1.0),),
        gtt_regime=None,
    )
    # total_nav = 40k + 40k = 80k; target VTI_LEAPS = 32k, actual 40k → overweight.
    # Use an underweight sleeve instead: actual 20k vs target 32k.
    nav = compute_nav_breakdown(portfolio, leaps_mtm_by_ticker={"VTI": 20_000.0})
    plan = compute_rebalance_plan(
        portfolio, nav, RebalanceRule.QUARTERLY, is_rebal_date=True, is_month_end=False
    )
    assert plan.leaps_trim == {}


def test_end_to_end_greeks_to_nav_to_rebalance() -> None:
    """Full chain: greeks → leaps_mtm_by_ticker → NAV → holdings → rebalance."""
    portfolio, _ = _divergent_portfolio()
    spot = {"VTI": 210.0, "IWM": 180.0}
    iv = {"VTI": 0.20, "IWM": 0.30}
    greeks = compute_portfolio_greeks(portfolio, spot=spot, iv=iv)
    mtm = leaps_mtm_by_ticker(greeks)

    assert set(mtm) == {"VTI", "IWM"}
    manual = {
        ticker: sum(
            c.price * c.contract.n_contracts * CONTRACT_MULTIPLIER * c.leaps_scale
            for c in greeks.contracts if c.contract.ticker == ticker
        )
        for ticker in {"VTI", "IWM"}
    }
    for ticker in ("VTI", "IWM"):
        assert abs(mtm[ticker] - manual[ticker]) < 1e-9

    nav = compute_nav_breakdown(portfolio, leaps_mtm_by_ticker=mtm)
    assert abs(nav.leaps_nav - sum(mtm.values())) < 1e-9

    views = (*compute_holdings_view(portfolio, nav), *compute_leaps_holdings_view(portfolio, nav))
    assert {v.ticker for v in views} == {"VXUS", "GLD", "VTI_LEAPS", "IWM_LEAPS"}

    plan = compute_rebalance_plan(
        portfolio, nav, RebalanceRule.DRIFT, is_rebal_date=False, is_month_end=True
    )
    assert isinstance(plan.leaps_trim, dict)