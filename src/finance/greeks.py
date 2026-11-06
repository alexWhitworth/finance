"""LEAPS portfolio greeks — per-contract and aggregate Black-Scholes greeks.

All functions are pure (no I/O). Builds on the scalar BS functions in leverage.py.

Two IV sourcing modes:

- Flat path (default): the caller supplies a scalar ``iv`` per contract; plain
  BSM greeks at that IV. Identical bit-for-bit to the pre-surface implementation.
- Surface path: the caller supplies per-underlying ``SsviParams`` and a decimal
  vol-index level; IV is sourced from ``vol_surface.surface_implied_vol`` and the
  greeks follow D5/D6: headline delta is sticky-moneyness (plain BSM delta
  exposed as ``delta_sticky_strike`` via ``vol_convention``), and ``level_vega``
  is dC/dv0 (sensitivity to vol-index variance), populated in addition to the
  local BSM vega.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import pandas as pd

from finance.consts import CONTRACT_MULTIPLIER, TIME_FLOOR, SsviParams
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
from finance.vol_surface import (
    atm_total_variance,
    ssvi_total_variance,
    surface_implied_vol,
)


class VolConvention(enum.Enum):
    """Delta convention on the surface path (D5).

    Attributes:
        STICKY_MONEYNESS: Headline delta accounts for the surface sliding with
            the underlying: delta_sm = delta_BS - (vega/S) * d(sigma)/dk.
            Internally consistent with the k-parameterized surface (Derman 1999).
        STICKY_STRIKE: Headline delta is plain BSM delta at the quoted strike.
    """

    STICKY_MONEYNESS = "sticky_moneyness"
    STICKY_STRIKE = "sticky_strike"


@dataclass(frozen=True)
class ContractGreeks:
    """Black-Scholes greeks for a single LeapsContract.

    Position-level fields (position_delta, position_vega, position_theta) are
    scaled by (n_contracts * CONTRACT_MULTIPLIER * leaps_scale).

    Attributes:
        contract: Source LeapsContract.
        as_of_date: Evaluation date.
        spot: Spot price used.
        iv: Implied volatility used (surface-sourced on the surface path).
        risk_free_rate: Risk-free rate used.
        time_to_expiry: Years to expiry (floored at TIME_FLOOR).
        price: BS call price per share.
        delta: Headline delta: sticky-moneyness on the surface path with the
            default convention (D5); plain BSM delta on the flat path and on
            the surface path with vol_convention=STICKY_STRIKE.
        gamma: d²V/dS². Always positive for long calls.
        vega: Local dV/d(sigma) per unit IV move.
        theta: dV/dt in dollars per calendar day (negative for long calls).
        vanna: dDelta/dVol.
        charm: dDelta/dt. Delta decay per calendar day.
        leaps_scale: Surviving fraction applied.
        position_delta: delta * n_contracts * CONTRACT_MULTIPLIER * leaps_scale.
        position_vega: vega * n_contracts * CONTRACT_MULTIPLIER * leaps_scale.
        position_theta: theta * n_contracts * CONTRACT_MULTIPLIER * leaps_scale.
        delta_sticky_strike: Plain BSM delta at sigma(k, t); None on the flat
            path (the delta field is the answer there).
        level_vega: dC/dv0 = vega * (1 - e^(-kappa*tau))/kappa, sensitivity to
            the vol-index instantaneous variance; None on the flat path.

    Notes:
        Surface-path vanna/charm are evaluated at the local surface IV
        (partial derivatives holding the smile point fixed); only delta
        (sticky-moneyness) and level_vega carry surface-dynamics corrections.
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
    delta_sticky_strike: float | None = None
    level_vega: float | None = None


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
    iv: float | None = None,
    as_of_date: pd.Timestamp | None = None,
    risk_free_rate: float = 0.0,
    leaps_scale: float = 1.0,
    params: SsviParams | None = None,
    vol_index_level: float | None = None,
    vol_convention: VolConvention = VolConvention.STICKY_MONEYNESS,
) -> ContractGreeks:
    """Compute all Black-Scholes greeks for one LeapsContract.

    Time to expiry is computed as (expiry_date - as_of_date).days / 365, then
    floored at TIME_FLOOR. Position-level fields are scaled by
    n_contracts * CONTRACT_MULTIPLIER * leaps_scale.

    Flat path (default): plain BSM greeks at the scalar ``iv``; ``delta``
    carries the plain BSM delta and ``delta_sticky_strike``/``level_vega``
    stay None. Identical bit-for-bit to the pre-surface implementation.

    Surface path: pass ``params`` and ``vol_index_level`` (and ``iv=None``);
    IV is sourced from ``surface_implied_vol``. ``delta`` then carries the
    headline convention selected by ``vol_convention`` (sticky-moneyness by
    default, D5), ``delta_sticky_strike`` records the plain BSM delta, and
    ``level_vega`` records dC/dv0 (D6).

    Arguments:
        contract: The LeapsContract to evaluate.
        spot: Current spot price of the underlying.
        iv: Implied volatility (annualized decimal, e.g. 0.18) for the flat
            path; None when surface inputs are given.
        as_of_date: Evaluation date.
        risk_free_rate: Continuously compounded risk-free rate. Default 0.0.
        leaps_scale: Surviving fraction in (0, 1]. Default 1.0.
        params: Per-underlying SsviParams for the surface path; None -> flat path.
        vol_index_level: Decimal vol-index level for the surface path.
        vol_convention: Headline-delta convention on the surface path.

    Returns:
        ContractGreeks with all per-share and position-level greeks populated.

    Raises:
        ValueError: If exactly one of params/vol_index_level is None, or if
            neither iv nor (params, vol_index_level) is provided, or if
            vol_convention is not a VolConvention.
    """
    surface_mode = params is not None
    if surface_mode != (vol_index_level is not None):
        raise ValueError(
            "surface inputs must be given together: "
            "params and vol_index_level are both required (or both None)"
        )
    if as_of_date is None:
        raise ValueError("as_of_date is required")
    if not surface_mode and iv is None:
        raise ValueError("either iv or (params, vol_index_level) must be provided")
    if not isinstance(vol_convention, VolConvention):
        raise ValueError("vol_convention must be a VolConvention member")

    raw_t = (contract.expiry_date - as_of_date).days / 365.0
    t = max(raw_t, TIME_FLOOR)
    q = contract.dividend_yield
    k = contract.strike

    if surface_mode:
        assert params is not None and vol_index_level is not None
        # Surface inputs take precedence when both are somehow supplied; iv is
        # the flat-path input by contract (API-compute_contract_greeks).
        iv_used = surface_implied_vol(
            spot=spot,
            strike=k,
            time_to_expiry_years=t,
            vol_index_level=vol_index_level,
            params=params,
            dividend_yield=q,
            risk_free_rate=risk_free_rate,
        )
    else:
        assert iv is not None
        iv_used = iv

    price = bs_call_price(spot, k, t, iv_used, risk_free_rate, q)
    delta_bs = bs_call_delta(spot, k, t, iv_used, risk_free_rate, q)
    gamma = bs_call_gamma(spot, k, t, iv_used, risk_free_rate, q)
    vega = bs_call_vega(spot, k, t, iv_used, risk_free_rate, q)
    theta = bs_call_theta(spot, k, t, iv_used, risk_free_rate, q)
    vanna = bs_call_vanna(spot, k, t, iv_used, risk_free_rate, q)
    charm = bs_call_charm(spot, k, t, iv_used, risk_free_rate, q)

    delta = delta_bs
    delta_sticky_strike: float | None = None
    level_vega: float | None = None
    if surface_mode:
        assert params is not None and vol_index_level is not None
        # Sticky-moneyness delta (D5): dV/dS at fixed k = dV/dS|_sigma - (vega/S) * dsigma/dk.
        forward = spot * math.exp((risk_free_rate - q) * t)
        log_moneyness = math.log(k / forward)
        theta_atm = atm_total_variance(vol_index_level, t, params)
        dsigma_dk = _dsigma_dk(log_moneyness, theta_atm, params, t)
        delta_sm = delta_bs - (vega / spot) * dsigma_dk

        # Level-vega (D6): dC/dv0 = vega * dtheta/dv0, dtheta/dv0 = (1-e^(-kappa*tau))/kappa.
        dtheta_dv0 = (1.0 - math.exp(-params.kappa * t)) / params.kappa
        level_vega = vega * dtheta_dv0

        if vol_convention is VolConvention.STICKY_MONEYNESS:
            delta = delta_sm
        delta_sticky_strike = delta_bs

    position_scale = contract.n_contracts * CONTRACT_MULTIPLIER * leaps_scale
    return ContractGreeks(
        contract=contract,
        as_of_date=as_of_date,
        spot=spot,
        iv=iv_used,
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
        delta_sticky_strike=delta_sticky_strike,
        level_vega=level_vega,
    )


def _dsigma_dk(
    log_moneyness: float,
    theta: float,
    params: SsviParams,
    t: float,
) -> float:
    """d(sigma)/dk along the SSVI smile at fixed (theta, t).

    sigma(k, t) = sqrt(w(k, theta) / t), so
    dsigma/dk = w'(k) / (2 * t * sigma) with w'(k) the SSVI total-variance
    slope: dw/dk = (theta/2) * (rho*phi + (phi*(phi*k + rho)) / sqrt((phi*k + rho)^2 + 1 - rho^2)).

    Arguments:
        log_moneyness: k = ln(K/F).
        theta: ATM total variance at this expiry.
        params: Per-underlying parameter set.
        t: Time to expiry in years (floored at TIME_FLOOR by the caller).

    Returns:
        d(sigma)/dk, sign matching rho (negative put skew for rho < 0).

    Raises:
        ValueError: If theta <= 0.
    """
    if theta <= 0.0:
        raise ValueError("theta must be positive")
    phi = params.eta * math.pow(theta, -params.gamma) * math.pow(1.0 + theta, params.gamma - 1.0)
    rho = params.rho
    radicand = (phi * log_moneyness + rho) ** 2 + 1.0 - rho**2
    dw_dk = (theta / 2.0) * (
        rho * phi + (phi * (phi * log_moneyness + rho)) / math.sqrt(max(radicand, 0.0))
    )
    w = ssvi_total_variance(log_moneyness, theta, params)
    sigma = math.sqrt(w / t)
    return dw_dk / (2.0 * t * sigma)


def compute_portfolio_greeks(
    portfolio: LivePortfolio,
    spot: dict[str, float],
    iv: dict[str, float] | None = None,
    risk_free_rate: float = 0.0,
    params: dict[str, SsviParams] | None = None,
    vol_index: dict[str, float] | None = None,
    vol_convention: VolConvention = VolConvention.STICKY_MONEYNESS,
) -> PortfolioGreeks:
    """Compute aggregate greeks across all active LEAPS contracts in a LivePortfolio.

    Flat path: each contract is priced against its own underlying by routing
    ``contract.ticker`` to ``spot[contract.ticker]`` and ``iv[contract.ticker]``.

    Surface path: pass per-ticker ``params`` and ``vol_index`` maps; each
    contract routes to ``params[contract.ticker]`` and
    ``vol_index[contract.ticker]``. A ticker missing from the surface maps
    falls back to the scalar ``iv[contract.ticker]`` if provided, else raises.

    Returns a zero-valued PortfolioGreeks with an empty contracts tuple when
    portfolio.leaps_contracts is empty.

    Arguments:
        portfolio: LivePortfolio with leaps_contracts to evaluate.
        spot: Current spot price per underlying, keyed by ticker.
        iv: Implied volatility (annualized decimal) per underlying (flat path),
            keyed by ticker; also used as per-ticker scalar fallback on the
            surface path.
        risk_free_rate: Continuously compounded risk-free rate. Default 0.0.
        params: Per-ticker SsviParams (surface path), keyed by ticker.
        vol_index: Per-ticker decimal vol-index levels (surface path).
        vol_convention: Headline-delta convention on the surface path.

    Returns:
        PortfolioGreeks with per-contract details and net aggregates.

    Raises:
        ValueError: If one or more live contracts' tickers are missing from
            spot and iv/params. The message lists every missing ticker, not
            just the first.
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

    params = params or {}
    vol_index = vol_index or {}
    iv = iv or {}

    def _surface_inputs_for(ticker: str) -> tuple[SsviParams | None, float | None]:
        if ticker in params and ticker in vol_index:
            return params[ticker], vol_index[ticker]
        if ticker in params or ticker in vol_index:
            raise ValueError(
                "surface maps must be given together: params and vol_index are "
                f"both required per ticker, got only one for {ticker}"
            )
        return None, None

    def _missing_reason(ticker: str) -> str | None:
        p, _ = _surface_inputs_for(ticker)
        if p is not None:
            return None
        if ticker not in iv:
            return "iv (or surface params/vol_index)"
        return None

    missing = sorted(
        {
            contract.ticker
            for contract, _ in portfolio.leaps_contracts
            if contract.ticker not in spot or _missing_reason(contract.ticker) is not None
        }
    )
    if missing:
        raise ValueError("spot/iv maps are missing the following tickers: " + ", ".join(missing))

    contract_greeks = tuple(
        compute_contract_greeks(
            contract,
            spot[contract.ticker],
            iv.get(contract.ticker),
            portfolio.as_of_date,
            risk_free_rate,
            scale,
            params=_surface_inputs_for(contract.ticker)[0],
            vol_index_level=_surface_inputs_for(contract.ticker)[1],
            vol_convention=vol_convention,
        )
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


def leaps_mtm_by_ticker(greeks: PortfolioGreeks) -> dict[str, float]:
    """Sum position mark-to-market grouped by underlying ticker.

    Position MTM is ``price * n_contracts * CONTRACT_MULTIPLIER * leaps_scale``
    per contract, summed within each ``contract.ticker``.

    Arguments:
        greeks: PortfolioGreeks with per-contract detail.

    Returns:
        Per-ticker MTM mapping. Empty dict when greeks.contracts is empty.
    """
    mtm: dict[str, float] = {}
    for cg in greeks.contracts:
        value = cg.price * cg.contract.n_contracts * CONTRACT_MULTIPLIER * cg.leaps_scale
        mtm[cg.contract.ticker] = mtm.get(cg.contract.ticker, 0.0) + value
    return mtm
