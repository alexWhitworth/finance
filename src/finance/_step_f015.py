"""Step functions for F-015: final day-loop helpers.

Extracted from run_backtest (portfolio.py lines 931-979).
Each function is pure — no side effects, no mutation of inputs.

Functions:
    _compute_total_nav: Sum all NAV components post-mutation.
    _advance_state: Advance prev_* fields for the next iteration.
    _build_weight_row: Compute end-of-day realized weights (sums to 1.0).
    _assemble_leaps_ledger: Post-loop ledger assembly from per-window ledgers.
"""

from dataclasses import replace

import pandas as pd

from finance.leverage import LeapsLedger, LeapsPartialCloseEvent
from finance.portfolio import BacktestContext, DayInputs, PortfolioState


def _compute_total_nav(state: PortfolioState) -> float:
    """Return total portfolio NAV after all day mutations.

    Identical formula to _compute_nav_before_contrib but called post-mutation.
    The explicit function keeps pre/post-contribution NAV semantics distinct.

    Arguments:
        state: Current PortfolioState after all step functions have run.

    Returns:
        Sum of holdings values, leaps_value, defensive_sleeve, and leaps_pool.
    """
    return (
        sum(state.holdings.values())
        + state.leaps_value
        + state.defensive_sleeve
        + state.leaps_pool
    )


def _advance_state(
    state: PortfolioState,
    total_nav: float,
    inputs: DayInputs,
) -> PortfolioState:
    """Advance the prev_* fields for the next loop iteration.

    Only prev_total_nav, prev_regime, and prev_date_ts change.
    All other fields carry forward unchanged.

    Arguments:
        state: Current PortfolioState.
        total_nav: End-of-day NAV computed by _compute_total_nav.
        inputs: Per-day inputs containing today's regime and date.

    Returns:
        New PortfolioState with updated prev_total_nav, prev_regime, prev_date_ts.
    """
    return replace(
        state,
        prev_total_nav=total_nav,
        prev_regime=inputs.regime_t,
        prev_date_ts=inputs.date_ts,
    )


def _build_weight_row(
    state: PortfolioState,
    total_nav: float,
    ctx: BacktestContext,
) -> dict[str, float]:
    """Compute end-of-day realized weights for weight_history.

    Decomposition:
    - Base assets: holdings[a] / total_nav.
    - LEAPS keys: leaps_value * (w[k] / leaps_fraction) / total_nav.
    - GTT defensive parked capital: (defensive_sleeve + leaps_pool) * dw / total_nav
      for each defensive_weight key, added onto any existing row entry.

    Invariant: sum(result.values()) == 1.0 within 1e-9 for any valid state with
    total_nav > 0.

    Arguments:
        state: Current PortfolioState after all mutations.
        total_nav: End-of-day NAV; must equal sum of all components.
        ctx: BacktestContext supplying base_assets, leaps_keys, leaps_fraction,
            w, gtt_active, and defensive_weights.

    Returns:
        Dict mapping each asset key to its realized weight in [0.0, 1.0].
        Returns a zero-weight dict (all values 0.0) when total_nav <= 0.
    """
    if total_nav <= 0.0:
        row: dict[str, float] = dict.fromkeys(ctx.base_assets, 0.0)
        for k in ctx.leaps_keys:
            row[k] = 0.0
        return row

    row = {a: state.holdings[a] / total_nav for a in ctx.base_assets}

    for k in ctx.leaps_keys:
        share = float(ctx.w[k]) / ctx.leaps_fraction if ctx.leaps_fraction > 0 else 0.0
        row[k] = state.leaps_value * share / total_nav

    parked = state.defensive_sleeve + state.leaps_pool
    if ctx.gtt_active and parked > 0.0:
        for dk, dw in ctx.defensive_weights.items():
            row[dk] = row.get(dk, 0.0) + dw * parked / total_nav

    return row


def _assemble_leaps_ledger(
    state: PortfolioState,
    ctx: BacktestContext,
    final_date: pd.Timestamp,
) -> LeapsLedger | None:
    """Post-loop ledger assembly: freeze partial closes and concatenate windows.

    Two independent assembly passes (both may apply):

    1. Partial-close freeze: if state.leaps_scale is non-empty, each surviving
       contract fraction is recorded as a LeapsPartialCloseEvent appended to
       state.leaps_ledger.

    2. GTT multi-window assembly: if GTT + LEAPS are both active, all
       per-window ledgers in state.all_window_ledgers are concatenated into a
       single LeapsLedger with gtt_close_events=state.all_gtt_closes.

    Arguments:
        state: Final PortfolioState after the backtest loop.
        ctx: BacktestContext supplying gtt_active, use_leaps, and config.
        final_date: Last trading date of the backtest; used as close_date for
            partial-close events.

    Returns:
        Assembled LeapsLedger, or None if no LEAPS overlay was active.
    """
    leaps_ledger = state.leaps_ledger
    partial_close_list = list(
        leaps_ledger.partial_close_events if leaps_ledger is not None else ()
    )

    # Freeze surviving partial closes onto the per-window ledger.
    if leaps_ledger is not None and state.leaps_scale:
        for c, surviving in state.leaps_scale.items():
            continuation = replace(c, n_contracts=c.n_contracts * surviving)
            partial_close_list.append(
                LeapsPartialCloseEvent(
                    close_date=final_date,
                    original_contract=c,
                    continuation_contract=continuation,
                    n_contracts_closed=c.n_contracts * (1.0 - surviving),
                    net_proceeds=0.0,
                )
            )
        leaps_ledger = replace(leaps_ledger, partial_close_events=tuple(partial_close_list))

    # GTT multi-window: concatenate all per-window ledgers.
    if ctx.gtt_active and ctx.use_leaps and ctx.config.leaps_config is not None:
        leaps_ledger = LeapsLedger(
            contracts=tuple(c for wl in state.all_window_ledgers for c in wl.contracts),
            roll_events=tuple(e for wl in state.all_window_ledgers for e in wl.roll_events),
            account_type=ctx.config.leaps_config.account_type,
            partial_close_events=tuple(
                e for wl in state.all_window_ledgers for e in wl.partial_close_events
            ),
            gtt_close_events=state.all_gtt_closes,
        )

    return leaps_ledger
