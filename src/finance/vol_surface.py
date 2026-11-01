"""SSVI implied-volatility surface in closed form.

Pure module: no I/O, no pandas, no solver, no hidden state. The single source
of implied volatility for LEAPS marks on the surface path.

The surface follows Gatheral & Jacquier (2014): an ATM total-variance curve
theta(tau) from an OU (mean-reverting) variance law anchored to the vol index,
wrapped by the SSVI smile w(k, theta). Implied vol is sigma = sqrt(w / t).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from finance.consts import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_RISK_FREE_RATE,
    TIME_FLOOR,
    SsviParams,
)


def atm_total_variance(
    vol_index_level: float,
    time_to_expiry_years: float,
    params: SsviParams,
) -> float:
    """ATM total variance theta(tau) under the OU variance law.

    theta = sigma_inf^2 * tau + (v0 - sigma_inf^2) * (1 - e^(-kappa*tau)) / kappa,
    with v0 = vol_index_level^2 (the 30-day variance strike approximation,
    Demeterfi et al. 1999). Calendar-arb-free by construction: theta is
    nondecreasing in tau whenever v0, sigma_inf^2 >= 0.

    Arguments:
        vol_index_level: Decimal annualized vol-index level (e.g. 0.18).
        time_to_expiry_years: Time to expiry in years; floored at TIME_FLOOR.
        params: Per-underlying parameter set.

    Returns:
        theta, the ATM total variance sigma_atm^2 * tau.

    Raises:
        ValueError: If params.kappa <= 0 or params.sigma_inf <= 0.
    """
    if params.kappa <= 0.0:
        raise ValueError("params.kappa must be positive")
    if params.sigma_inf <= 0.0:
        raise ValueError("params.sigma_inf must be positive")

    tau = max(time_to_expiry_years, TIME_FLOOR)
    v0 = vol_index_level**2
    sigma_inf2 = params.sigma_inf**2
    return sigma_inf2 * tau + (v0 - sigma_inf2) * (1.0 - math.exp(-params.kappa * tau)) / (
        params.kappa
    )


def ssvi_total_variance(
    log_moneyness: float,
    theta: float,
    params: SsviParams,
) -> float:
    """SSVI total variance w(k, theta) at log-forward-moneyness k.

    w = (theta/2) * {1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2)},
    phi(theta) = eta * theta^(-gamma) * (1 + theta)^(gamma - 1).

    Arguments:
        log_moneyness: k = ln(K/F) with forward F.
        theta: ATM total variance (from atm_total_variance); strictly > 0.
        params: Per-underlying parameter set.

    Returns:
        Strictly positive total variance w(k, theta).

    Raises:
        ValueError: If theta <= 0.
    """
    if theta <= 0.0:
        raise ValueError("theta must be positive")
    if abs(params.rho) >= 1.0:
        raise ValueError("params.rho magnitude must be < 1")
    if params.eta <= 0.0:
        raise ValueError("params.eta must be positive")
    if not 0.0 < params.gamma <= 1.0:
        raise ValueError("params.gamma must be in (0, 1]")

    phi = (
        params.eta
        * math.pow(theta, -params.gamma)
        * math.pow(1.0 + theta, params.gamma - 1.0)
    )
    rho = params.rho
    radicand = (phi * log_moneyness + rho) ** 2 + 1.0 - rho**2
    return (theta / 2.0) * (1.0 + rho * phi * log_moneyness + math.sqrt(max(radicand, 0.0)))


def surface_implied_vol(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    vol_index_level: float,
    params: SsviParams,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """Closed-form implied vol for a strike/expiry on the SSVI surface.

    Forward F = spot * exp((r - q) * t); k = ln(K/F); sigma = sqrt(w(k, theta) / t)
    with theta from atm_total_variance and w from ssvi_total_variance.

    Arguments:
        spot: Current spot; must be > 0.
        strike: Option strike; must be > 0.
        time_to_expiry_years: Time to expiry; floored at TIME_FLOOR.
        vol_index_level: Decimal annualized vol-index level.
        params: Per-underlying parameter set.
        dividend_yield: Forward dividend yield q; default DEFAULT_DIVIDEND_YIELD.
        risk_free_rate: Maturity-matched r(T); default DEFAULT_RISK_FREE_RATE.

    Returns:
        sigma(k, t), decimal annualized.

    Raises:
        ValueError: If spot or strike <= 0.
    """
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")

    tau = max(time_to_expiry_years, TIME_FLOOR)
    theta = atm_total_variance(vol_index_level, tau, params)
    forward = spot * math.exp((risk_free_rate - dividend_yield) * tau)
    k = math.log(strike / forward)
    w = ssvi_total_variance(k, theta, params)
    return math.sqrt(w / tau)


def check_surface_arbitrage(
    params: SsviParams,
    max_vol_index_level: float = 0.80,
    expiry_grid_years: Sequence[float] = (0.083, 0.25, 0.5, 1.0, 2.0),
) -> "SurfaceArbitrageReport":
    """Constructive no-arb checks for one SsviParams set.

    Butterfly (Gatheral & Jacquier 2014): theta*phi*(1+|rho|) < 4 and
    theta*phi^2*(1+|rho|) <= 4 over a vol-index-level x expiry grid up to
    max_vol_index_level. Lee (2004) wing slopes theta*phi*(1±rho)/2 reported
    against the total-variance slope cap of 2 (moment formula: beta < 2). Calendar
    monotonicity is asserted (holds by the OU form).

    Arguments:
        params: Per-underlying parameter set.
        max_vol_index_level: Worst-case decimal vol-index level (default 0.80).
        expiry_grid_years: Expiry grid in years (default (0.083, 0.25, 0.5, 1.0, 2.0)).

    Returns:
        SurfaceArbitrageReport diagnostic; the caller decides on failure (no
        silent clamping).

    Raises:
        ValueError: If max_vol_index_level <= 0 or any expiry <= 0.
    """
    if max_vol_index_level <= 0.0:
        raise ValueError("max_vol_index_level must be positive")
    if not expiry_grid_years or any(tau <= 0.0 for tau in expiry_grid_years):
        raise ValueError("expiry_grid_years entries must be positive")

    levels: Sequence[float] = (
        0.10,
        0.20,
        0.40,
        float(max_vol_index_level),
    )

    butterfly_first_ok = True
    butterfly_second_ok = True
    lee_left = 0.0
    lee_right = 0.0
    calendar_monotone = True

    for level in levels:
        prev_theta: float | None = None
        for tau in expiry_grid_years:
            theta = atm_total_variance(level, tau, params)
            if prev_theta is not None and theta < prev_theta - 1e-12:
                calendar_monotone = False
            prev_theta = theta

            phi = (
                params.eta
                * math.pow(theta, -params.gamma)
                * math.pow(1.0 + theta, params.gamma - 1.0)
            )
            factor = 1.0 + abs(params.rho)
            if theta * phi * factor >= 4.0:
                butterfly_first_ok = False
            if theta * phi**2 * factor > 4.0:
                butterfly_second_ok = False

            left = theta * phi * (1.0 - params.rho) / 2.0
            right = theta * phi * (1.0 + params.rho) / 2.0
            lee_left = max(lee_left, left)
            lee_right = max(lee_right, right)

    return SurfaceArbitrageReport(
        butterfly_first_ok=butterfly_first_ok,
        butterfly_second_ok=butterfly_second_ok,
        lee_wing_slope_left=lee_left,
        lee_wing_slope_right=lee_right,
        calendar_monotone=calendar_monotone,
    )


@dataclass(frozen=True)
class SurfaceArbitrageReport:
    """No-arbitrage diagnostic over a worst-case vol-index x expiry grid.

    Attributes:
        butterfly_first_ok: theta*phi*(1+|rho|) < 4 holds across the grid.
        butterfly_second_ok: theta*phi^2*(1+|rho|) <= 4 holds across the grid.
        lee_wing_slope_left: Max left-wing total-variance slope theta*phi*(1-rho)/2.
        lee_wing_slope_right: Max right-wing total-variance slope theta*phi*(1+rho)/2.
        calendar_monotone: theta nondecreasing in tau across the grid.
    """

    butterfly_first_ok: bool
    butterfly_second_ok: bool
    lee_wing_slope_left: float
    lee_wing_slope_right: float
    calendar_monotone: bool
