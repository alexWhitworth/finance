"""Live portfolio management — bridge from backtest result to live portfolio state.

Provides the LivePortfolio, NavBreakdown, and HoldingView dataclasses, and the
pure functions as_live_portfolio(), compute_nav_breakdown(), and
compute_holdings_view(). All functions are pure (no I/O).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finance._portfolio_types import BacktestResult
from finance.leverage import LeapsContract, get_live_contracts


@dataclass(frozen=True)
class LivePortfolio:
    """User's current brokerage portfolio state as of a specific date.

    Self-contained and user-constructible without understanding backtest internals.
    All fields are immutable. Validates target_weights sum, leaps_scale range, and
    contract expiry on construction.

    Attributes:
        as_of_date: Date these holdings reflect.
        holdings: Base asset ticker → current dollar value.
        target_weights: Target allocation. Must sum to 1.0 within 1e-6.
        leaps_contracts: Active LEAPS contracts with surviving fraction
            (scale ∈ (0, 1]). Empty tuple if no LEAPS.
        gtt_regime: Current GTT regime: 1=Long, 0=Defensive, None=inactive.
        defensive_sleeve: Dollar value in GTT defensive allocation. Default 0.0.
        leaps_pool: Force-closed LEAPS proceeds parked during defensive window.
            Default 0.0.

    Raises:
        ValueError: If target_weights does not sum to 1.0 within 1e-6.
        ValueError: If any leaps_scale is not in (0, 1].
        ValueError: If any contract.expiry_date <= as_of_date.
    """

    as_of_date: pd.Timestamp
    holdings: dict[str, float]
    target_weights: dict[str, float]
    leaps_contracts: tuple[tuple[LeapsContract, float], ...]
    gtt_regime: int | None
    defensive_sleeve: float = 0.0
    leaps_pool: float = 0.0

    def __post_init__(self) -> None:
        total = sum(self.target_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"target_weights must sum to 1.0; got {total:.8f}"
            )
        for contract, scale in self.leaps_contracts:
            if not (0.0 < scale <= 1.0):
                raise ValueError(
                    f"leaps_scale must be in (0, 1]; got {scale} for contract "
                    f"purchased {contract.purchase_date.date()}"
                )
            if contract.expiry_date <= self.as_of_date:
                raise ValueError(
                    f"contract.expiry_date ({contract.expiry_date.date()}) must be "
                    f"> as_of_date ({self.as_of_date.date()})"
                )


@dataclass(frozen=True)
class NavBreakdown:
    """NAV decomposition for a LivePortfolio.

    Attributes:
        base_nav: Sum of base asset holdings.
        leaps_nav: Mark-to-market value of all active LEAPS contracts (caller-supplied).
        defensive_sleeve: GTT-swept capital.
        leaps_pool: Parked force-closed LEAPS proceeds.
        total_nav: base_nav + leaps_nav + defensive_sleeve + leaps_pool.
    """

    base_nav: float
    leaps_nav: float
    defensive_sleeve: float
    leaps_pool: float
    total_nav: float


@dataclass(frozen=True)
class HoldingView:
    """Single asset row in a portfolio weight drift analysis.

    Attributes:
        ticker: Asset identifier.
        dollar_value: Current dollar value.
        actual_weight: dollar_value / total_nav.
        target_weight: From LivePortfolio.target_weights (0.0 if absent).
        weight_drift: actual_weight - target_weight (signed).
        relative_drift: weight_drift / target_weight; None when target_weight == 0.
    """

    ticker: str
    dollar_value: float
    actual_weight: float
    target_weight: float
    weight_drift: float
    relative_drift: float | None


def compute_nav_breakdown(
    portfolio: LivePortfolio,
    leaps_mtm: float = 0.0,
) -> NavBreakdown:
    """Decompose LivePortfolio NAV into base, LEAPS, defensive, and pool components.

    Arguments:
        portfolio: LivePortfolio whose holdings to decompose.
        leaps_mtm: Current mark-to-market value of all active LEAPS contracts
            (caller-supplied; not computed here). Defaults to 0.0.

    Returns:
        NavBreakdown with total_nav == base_nav + leaps_nav + defensive_sleeve
        + leaps_pool within 1e-9 (I1).
    """
    base_nav = sum(portfolio.holdings.values())
    total_nav = base_nav + leaps_mtm + portfolio.defensive_sleeve + portfolio.leaps_pool
    return NavBreakdown(
        base_nav=base_nav,
        leaps_nav=leaps_mtm,
        defensive_sleeve=portfolio.defensive_sleeve,
        leaps_pool=portfolio.leaps_pool,
        total_nav=total_nav,
    )


def compute_holdings_view(
    portfolio: LivePortfolio,
    nav: NavBreakdown,
) -> tuple[HoldingView, ...]:
    """Compute per-asset weight drift against target_weights.

    Arguments:
        portfolio: LivePortfolio with base asset holdings and target weights.
        nav: NavBreakdown providing total_nav for weight normalization.

    Returns:
        Tuple of HoldingView — one per ticker in portfolio.holdings. The sum of
        actual_weight values is <= 1.0 + 1e-9 (I2) since holdings cover only base
        assets; defensive_sleeve and leaps components are excluded.
    """
    total = nav.total_nav
    views = []
    for ticker, value in portfolio.holdings.items():
        actual_w = value / total if total > 0.0 else 0.0
        target_w = portfolio.target_weights.get(ticker, 0.0)
        drift = actual_w - target_w
        rel = drift / target_w if target_w != 0.0 else None
        views.append(
            HoldingView(
                ticker=ticker,
                dollar_value=value,
                actual_weight=actual_w,
                target_weight=target_w,
                weight_drift=drift,
                relative_drift=rel,
            )
        )
    return tuple(views)


def as_live_portfolio(
    result: BacktestResult,
    gtt_active: bool = False,
) -> LivePortfolio:
    """Convert a completed BacktestResult into a LivePortfolio.

    Uses get_live_contracts() to filter contracts from final_state.leaps_ledger
    to those still active as of the last backtest date. Pulls leaps_scale from
    final_state.leaps_scale. When the backtest had no LEAPS overlay or all
    contracts have expired, leaps_contracts is an empty tuple.

    Arguments:
        result: Completed BacktestResult from run_backtest. Must have
            final_state populated (requires F-003/F-004).
        gtt_active: Whether the GTT overlay was active. When True, gtt_regime
            is set from final_state.prev_regime. When False, gtt_regime is None.
            Pass True only when the backtest was configured with a gtt_signal;
            on a non-GTT backtest, prev_regime is always 1 (Long) and the
            returned gtt_regime is semantically meaningless.

    Returns:
        LivePortfolio reflecting the portfolio state at the last backtest date.
    """
    state = result.final_state
    if state.prev_date_ts is None:
        raise ValueError(
            "BacktestResult.final_state.prev_date_ts is None — "
            "run_backtest produced an empty date range."
        )
    as_of_date: pd.Timestamp = state.prev_date_ts

    # Build leaps_contracts: (contract, scale) pairs for live contracts only
    leaps_pairs: tuple[tuple[LeapsContract, float], ...]
    if state.leaps_ledger is None:
        leaps_pairs = ()
    else:
        live = get_live_contracts(state.leaps_ledger, as_of_date)
        leaps_pairs = tuple(
            (c, state.leaps_scale.get(c, 1.0)) for c in live
        )

    gtt_regime: int | None = state.prev_regime if gtt_active else None

    return LivePortfolio(
        as_of_date=as_of_date,
        holdings=dict(state.holdings),
        target_weights=dict(result.config.target_weights),
        leaps_contracts=leaps_pairs,
        gtt_regime=gtt_regime,
        defensive_sleeve=state.defensive_sleeve,
        leaps_pool=state.leaps_pool,
    )
