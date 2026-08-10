"""Live portfolio management — bridge from backtest result to live portfolio state.

Provides the LivePortfolio dataclass and the as_live_portfolio() bridge function
that converts a completed BacktestResult into a LivePortfolio for live management.
All functions are pure (no I/O).
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
