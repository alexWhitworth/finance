"""F-005: _build_initial_state — day-0 capital deployment.

Extracts the initial-state setup from run_backtest (lines 701–738 of portfolio.py)
into a single pure function returning an immutable PortfolioState.

Responsibilities:
- Run the first LEAPS simulation scoped to the first Long window (if use_leaps).
- Compute base holdings from initial_nav * (1 - leaps_fraction) * base_target_w.
- Set all PortfolioState accumulators to their zero/empty values.

No I/O or side effects beyond calling run_leaps_simulation (which is a pure
deterministic computation over price series).
"""

from __future__ import annotations

import pandas as pd

from finance.leverage import LeapsContract, LeapsLedger, run_leaps_simulation
from finance.portfolio import BacktestContext, PortfolioState
from finance.portfolio import _long_windows


def _build_initial_state(ctx: BacktestContext) -> PortfolioState:
    """Build the PortfolioState for the first loop iteration.

    Runs the first LEAPS simulation scoped to the first Long window when LEAPS
    are active, then allocates base holdings from the remaining NAV fraction.
    All loop-accumulator fields (all_window_ledgers, all_gtt_closes, leaps_scale)
    are initialised to their empty/zero values.

    Arguments:
        ctx: Immutable BacktestContext produced by _build_context (F-004).

    Returns:
        PortfolioState with:
        - holdings: dollar value per base asset.
        - leaps_ledger: populated if LEAPS are active, else None.
        - all_window_ledgers: single-entry tuple if LEAPS active, else ().
        - all other accumulator fields at zero / empty defaults.

    Notes:
        When GTT is active and the first window is entirely Defensive (no Long day
        exists), run_leaps_simulation is called with an empty price series and
        returns an empty ledger (no contracts).
    """
    leaps_ledger: LeapsLedger | None = None
    all_window_ledgers: tuple[LeapsLedger, ...] = ()

    if (
        ctx.use_leaps
        and ctx.underlying_prices is not None
        and ctx.config.leaps_config is not None
    ):
        # Scope initial simulation to the first Long window under GTT; use the
        # full price series when GTT is inactive.
        if ctx.gtt_active and ctx.mask_aligned is not None:
            long_wins = _long_windows(ctx.mask_aligned)
            if long_wins:
                first_start, first_end = long_wins[0]
                win_prices: pd.Series = ctx.underlying_prices.loc[first_start:first_end]
            else:
                # Portfolio is never Long; no contracts should be created.
                win_prices = ctx.underlying_prices.iloc[0:0]
        else:
            win_prices = ctx.underlying_prices

        initial_leaps_capital = ctx.config.initial_nav * ctx.leaps_fraction
        leaps_ledger = run_leaps_simulation(
            win_prices,
            ctx.leaps_monthly,
            ctx.config.leaps_config,
            risk_free_series=ctx.return_data.risk_free_rate,
            iv_series=ctx.raw_vix,
            initial_capital=initial_leaps_capital,
        )
        all_window_ledgers = (leaps_ledger,)

    base_nav_init = ctx.config.initial_nav * (1.0 - ctx.leaps_fraction)
    holdings: dict[str, float] = {
        a: base_nav_init * float(ctx.base_target_w[a]) for a in ctx.base_assets
    }

    leaps_scale: dict[LeapsContract, float] = {}

    return PortfolioState(
        holdings=holdings,
        defensive_sleeve=0.0,
        leaps_pool=0.0,
        leaps_value=0.0,
        prev_total_nav=ctx.config.initial_nav,
        prev_regime=1,
        prev_date_ts=None,
        leaps_ledger=leaps_ledger,
        leaps_scale=leaps_scale,
        all_window_ledgers=all_window_ledgers,
        all_gtt_closes=(),
    )
