"""Integration tests for F-008: compute_portfolio_greeks with a LivePortfolio.

Verifies the I12 invariant: net_delta ∈ (0, n_contracts_total * CONTRACT_MULTIPLIER)
for any live LEAPS portfolio with valid inputs.
"""

import pandas as pd

from finance.consts import CONTRACT_MULTIPLIER
from finance.greeks import compute_portfolio_greeks
from finance.leverage import AccountType, LeapsContract, create_leaps_contract
from finance.portfolio_manager import LivePortfolio

_AS_OF = pd.Timestamp("2024-01-15")
_EXPIRY = pd.Timestamp("2026-01-21")


def _make_contract(
    *,
    strike: float = 100.0,
    spot: float = 200.0,
    n_contracts: float = 2.0,
    expiry: pd.Timestamp = _EXPIRY,
) -> LeapsContract:
    return LeapsContract(
        purchase_date=pd.Timestamp("2022-01-21"),
        expiry_date=expiry,
        strike=strike,
        spot_at_purchase=spot,
        premium_paid=45.0,
        notional=spot * CONTRACT_MULTIPLIER,
        n_contracts=n_contracts,
        account_type=AccountType.TAXABLE,
        dividend_yield=0.013,
    )


def _make_portfolio(
    leaps_pairs: tuple[tuple[LeapsContract, float], ...],
) -> LivePortfolio:
    return LivePortfolio(
        as_of_date=_AS_OF,
        holdings={"VTI": 85_000.0},
        target_weights={"VTI": 1.0},
        leaps_contracts=leaps_pairs,
        gtt_regime=None,
    )


class TestI12DomainBounds:
    """I12: net_delta ∈ (0, n_contracts_total * CONTRACT_MULTIPLIER) for live LEAPS."""

    def test_single_contract_delta_in_bounds(self) -> None:
        """Single contract DITM LEAPS: net_delta in valid domain."""
        contract = _make_contract(strike=100.0, n_contracts=2.0, spot=200.0)
        portfolio = _make_portfolio(((contract, 1.0),))
        pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.18, risk_free_rate=0.04)

        n_total = contract.n_contracts
        upper_bound = n_total * CONTRACT_MULTIPLIER
        assert pg.net_delta > 0.0
        assert pg.net_delta < upper_bound

    def test_multi_contract_delta_in_bounds(self) -> None:
        """Multiple LEAPS contracts: net_delta bounded by total contract count."""
        c1 = _make_contract(strike=100.0, n_contracts=3.0)
        c2 = _make_contract(strike=130.0, n_contracts=2.0)
        portfolio = _make_portfolio(((c1, 1.0), (c2, 0.8)))

        pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.18, risk_free_rate=0.04)

        # Upper bound: sum of (n_contracts * CONTRACT_MULTIPLIER) across all pairs
        n_total = c1.n_contracts + c2.n_contracts
        upper_bound = n_total * CONTRACT_MULTIPLIER
        assert pg.net_delta > 0.0
        assert pg.net_delta < upper_bound

    def test_partial_scale_reduces_net_delta(self) -> None:
        """leaps_scale < 1 reduces net_delta proportionally."""
        contract = _make_contract(strike=100.0, n_contracts=5.0)
        portfolio_full = _make_portfolio(((contract, 1.0),))
        portfolio_half = _make_portfolio(((contract, 0.5),))

        pg_full = compute_portfolio_greeks(portfolio_full, spot=200.0, iv=0.18)
        pg_half = compute_portfolio_greeks(portfolio_half, spot=200.0, iv=0.18)

        import numpy as np
        np.testing.assert_allclose(pg_half.net_delta, pg_full.net_delta * 0.5, rtol=1e-9)

    def test_net_delta_positive_for_ditm_leaps(self) -> None:
        """Deep ITM LEAPS (strike = 50% of spot): net_delta strongly positive."""
        # LEAPS_STRIKE_RATIO is 0.50 — this is the target product
        contract = _make_contract(strike=100.0, spot=200.0, n_contracts=1.0)
        portfolio = _make_portfolio(((contract, 1.0),))
        pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.18)

        # Delta of a 2-year DITM call at 50% moneyness should be > 0.9
        assert pg.net_delta > 0.0
        assert pg.contracts[0].delta > 0.9

    def test_net_theta_negative_for_live_leaps(self) -> None:
        """Long LEAPS: net_theta is negative (time decay).

        Uses dividend_yield=0.0 and risk_free_rate=0.04 to ensure the standard
        theta < 0 invariant holds. With q > 0 and r = 0, deep-ITM calls can have
        positive theta (dividend term dominates), which is mathematically correct
        but outside the scope of this invariant test.
        """
        # Create a contract without dividend yield to ensure theta < 0
        contract = LeapsContract(
            purchase_date=pd.Timestamp("2022-01-21"),
            expiry_date=_EXPIRY,
            strike=120.0,
            spot_at_purchase=200.0,
            premium_paid=45.0,
            notional=200.0 * CONTRACT_MULTIPLIER,
            n_contracts=2.0,
            account_type=AccountType.TAXABLE,
            dividend_yield=0.0,
        )
        portfolio = _make_portfolio(((contract, 1.0),))
        pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.18, risk_free_rate=0.04)
        assert pg.net_theta < 0.0

    def test_contracts_count_matches_input(self) -> None:
        """Number of ContractGreeks matches number of leaps_contracts."""
        c1 = _make_contract(strike=100.0)
        c2 = _make_contract(strike=120.0)
        c3 = _make_contract(strike=140.0)
        portfolio = _make_portfolio(((c1, 1.0), (c2, 0.9), (c3, 0.7)))
        pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.18)
        assert len(pg.contracts) == 3

    def test_consistency_via_create_leaps_contract(self) -> None:
        """Integration: LeapsContract created via create_leaps_contract → valid PortfolioGreeks."""
        purchase_date = pd.Timestamp("2022-01-21")
        contract = create_leaps_contract(
            purchase_date=purchase_date,
            spot=200.0,
            capital_to_deploy=15_000.0,
            iv=0.18,
            account_type=AccountType.TAXABLE,
        )
        # Forward 1 year; contract is still live
        as_of = pd.Timestamp("2023-01-20")
        portfolio = LivePortfolio(
            as_of_date=as_of,
            holdings={"VTI": 85_000.0},
            target_weights={"VTI": 1.0},
            leaps_contracts=((contract, 1.0),),
            gtt_regime=None,
        )
        pg = compute_portfolio_greeks(portfolio, spot=220.0, iv=0.20, risk_free_rate=0.04)

        assert pg.net_delta > 0.0
        assert pg.net_theta < 0.0
        n_total = contract.n_contracts
        assert pg.net_delta < n_total * CONTRACT_MULTIPLIER
