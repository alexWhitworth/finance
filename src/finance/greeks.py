"""LEAPS portfolio greeks — per-contract and aggregate Black-Scholes greeks.

All functions are pure (no I/O). Builds on the scalar BS functions in leverage.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finance.consts import CONTRACT_MULTIPLIER, TIME_FLOOR
from finance.leverage import (
    LeapsContract,
    bs_call_charm,
    bs_call_delta,
    bs_call_gamma,
    bs_call_price,
    bs_call_theta,
    bs_call_vanna,
    bs_call_vega,
)
from finance.portfolio_manager import LivePortfolio


@dataclass(frozen=True)
class ContractGreeks:
    """Black-Scholes greeks for a single LeapsContract.

    Position-level fields (position_delta, position_vega, position_theta) are
    scaled by (n_contracts * CONTRACT_MULTIPLIER * leaps_scale).

    Attributes:
        contract: Source LeapsContract.
        as_of_date: Evaluation date.
        spot: Spot price used.
        iv: Implied volatility used.
        risk_free_rate: Risk-free rate used.
        time_to_expiry: Years to expiry (floored at TIME_FLOOR).
        price: BS call price per share.
        delta: dV/dS. In (0, 1) for calls.
        gamma: d²V/dS². Always positive for long calls.
        vega: dV/d(sigma) per unit IV move.
        theta: dV/dt in dollars per calendar day (negative for long calls).
        vanna: dDelta/dVol.
        charm: dDelta/dt. Delta decay per calendar day.
        leaps_scale: Surviving fraction applied.
        position_delta: delta * n_contracts * CONTRACT_MULTIPLIER * leaps_scale.
        position_vega: vega * n_contracts * CONTRACT_MULTIPLIER * leaps_scale.
        position_theta: theta * n_contracts * CONTRACT_MULTIPLIER * leaps_scale.
    """

    contract: LeapsContract
    as_of_date: pd.Timestamp
    spot: float
    iv: float
    risk_free_rate: float
    time_to_expiry: float
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    vanna: float
    charm: float
    leaps_scale: float
    position_delta: float
    position_vega: float
    position_theta: float


@dataclass(frozen=True)
class PortfolioGreeks:
    """Aggregate greeks across all active LEAPS contracts in a LivePortfolio.

    All net_ fields are zero when leaps_contracts is empty.

    Attributes:
        as_of_date: Evaluation date.
        contracts: Per-contract greeks.
        net_delta: Sum of position_delta.
        net_vega: Sum of position_vega.
        net_gamma: Sum of per-position scaled gamma.
        net_theta: Total dollar theta per calendar day.
        net_vanna: Sum of per-position scaled vanna.
        net_charm: Sum of per-position scaled charm.
    """

    as_of_date: pd.Timestamp
    contracts: tuple[ContractGreeks, ...]
    net_delta: float
    net_vega: float
    net_gamma: float
    net_theta: float
    net_vanna: float
    net_charm: float


def compute_contract_greeks(
    contract: LeapsContract,
    spot: float,
    iv: float,
    as_of_date: pd.Timestamp,
    risk_free_rate: float = 0.0,
    leaps_scale: float = 1.0,
) -> ContractGreeks:
    """Compute all Black-Scholes greeks for one LeapsContract.

    Time to expiry is computed as (expiry_date - as_of_date).days / 365, then
    floored at TIME_FLOOR. Position-level fields are scaled by
    n_contracts * CONTRACT_MULTIPLIER * leaps_scale.

    Arguments:
        contract: The LeapsContract to evaluate.
        spot: Current spot price of the underlying.
        iv: Implied volatility (annualized decimal, e.g. 0.18).
        as_of_date: Evaluation date.
        risk_free_rate: Continuously compounded risk-free rate. Default 0.0.
        leaps_scale: Surviving fraction in (0, 1]. Default 1.0.

    Returns:
        ContractGreeks with all per-share and position-level greeks populated.
    """
    raw_t = (contract.expiry_date - as_of_date).days / 365.0
    t = max(raw_t, TIME_FLOOR)
    q = contract.dividend_yield
    k = contract.strike

    price = bs_call_price(spot, k, t, iv, risk_free_rate, q)
    delta = bs_call_delta(spot, k, t, iv, risk_free_rate, q)
    gamma = bs_call_gamma(spot, k, t, iv, risk_free_rate, q)
    vega = bs_call_vega(spot, k, t, iv, risk_free_rate, q)
    theta = bs_call_theta(spot, k, t, iv, risk_free_rate, q)
    vanna = bs_call_vanna(spot, k, t, iv, risk_free_rate, q)
    charm = bs_call_charm(spot, k, t, iv, risk_free_rate, q)

    position_scale = contract.n_contracts * CONTRACT_MULTIPLIER * leaps_scale
    return ContractGreeks(
        contract=contract,
        as_of_date=as_of_date,
        spot=spot,
        iv=iv,
        risk_free_rate=risk_free_rate,
        time_to_expiry=t,
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        vanna=vanna,
        charm=charm,
        leaps_scale=leaps_scale,
        position_delta=delta * position_scale,
        position_vega=vega * position_scale,
        position_theta=theta * position_scale,
    )


def compute_portfolio_greeks(
    portfolio: LivePortfolio,
    spot: float,
    iv: float,
    risk_free_rate: float = 0.0,
) -> PortfolioGreeks:
    """Compute aggregate greeks across all active LEAPS contracts in a LivePortfolio.

    Returns a zero-valued PortfolioGreeks with an empty contracts tuple when
    portfolio.leaps_contracts is empty.

    Arguments:
        portfolio: LivePortfolio with leaps_contracts to evaluate.
        spot: Current spot price of the underlying.
        iv: Implied volatility (annualized decimal).
        risk_free_rate: Continuously compounded risk-free rate. Default 0.0.

    Returns:
        PortfolioGreeks with per-contract details and net aggregates.
    """
    if not portfolio.leaps_contracts:
        return PortfolioGreeks(
            as_of_date=portfolio.as_of_date,
            contracts=(),
            net_delta=0.0,
            net_vega=0.0,
            net_gamma=0.0,
            net_theta=0.0,
            net_vanna=0.0,
            net_charm=0.0,
        )

    contract_greeks = tuple(
        compute_contract_greeks(contract, spot, iv, portfolio.as_of_date, risk_free_rate, scale)
        for contract, scale in portfolio.leaps_contracts
    )

    def _position_scale(cg: ContractGreeks) -> float:
        return cg.contract.n_contracts * CONTRACT_MULTIPLIER * cg.leaps_scale

    return PortfolioGreeks(
        as_of_date=portfolio.as_of_date,
        contracts=contract_greeks,
        net_delta=sum(cg.position_delta for cg in contract_greeks),
        net_vega=sum(cg.position_vega for cg in contract_greeks),
        net_gamma=sum(cg.gamma * _position_scale(cg) for cg in contract_greeks),
        net_theta=sum(cg.position_theta for cg in contract_greeks),
        net_vanna=sum(cg.vanna * _position_scale(cg) for cg in contract_greeks),
        net_charm=sum(cg.charm * _position_scale(cg) for cg in contract_greeks),
    )
