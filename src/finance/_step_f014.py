"""F-014: _apply_gtt_reentry step function (Bug 2 fix).

Extracts the Defensive->Long re-entry rebalance from run_backtest (lines 891-929).
Bug 2 fix: LEAPS contracts created on re-entry are priced at creation IV
(raw VIX floored at ctx.iv), NOT the smoothed MTM IV.

Invariant A2: sum(new_holdings.values()) + new_leaps_value == total within 1e-9
Invariant A4: new_leaps_value == total * ctx.leaps_fraction within 1e-6 (creation IV)
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from finance.leverage import (
    LeapsContract,
    _live_contracts,
    price_leaps_contract,
    run_leaps_simulation,
)
from finance.portfolio import BacktestContext, DayInputs, PortfolioState


def _apply_gtt_reentry(
    state: PortfolioState,
    inputs: DayInputs,
    ctx: BacktestContext,
) -> PortfolioState:
    """Apply the Defensive->Long GTT re-entry forced rebalance.

    Redeploys full NAV to target_weights on the first Long day after a defensive
    window. Step order is load-bearing:

    1. Compute total NAV (holdings + sleeve + pool).
    2. Allocate base holdings at (1 - leaps_fraction) * total * base_target_w.
    3. Zero defensive_sleeve and leaps_pool.
    4. If LEAPS active: run fresh simulation for new Long window; price creation
       contracts at creation IV = max(raw_vix_value, ctx.iv) — NOT smoothed MTM IV
       (Bug 2 fix).
    5. Reset leaps_scale to {} (clears orphaned keys from prior window).

    No-op unless ctx.gtt_active AND state.prev_regime == 0 AND inputs.regime_t == 1.

    Invariant A2: sum(new_holdings.values()) + new_leaps_value == total within 1e-9.
    Invariant A4: new_leaps_value == total * ctx.leaps_fraction within 1e-6.

    Arguments:
        state: Current PortfolioState snapshot.
        inputs: Per-day read-only inputs for the current trading day.
        ctx: Immutable backtest configuration and precomputed series.

    Returns:
        Updated PortfolioState with re-entry rebalance applied, or state unchanged
        if fire condition is not met.
    """
    if not (ctx.gtt_active and state.prev_regime == 0 and inputs.regime_t == 1):
        return state

    # Step 1: total NAV
    total = sum(state.holdings.values()) + state.defensive_sleeve + state.leaps_pool

    # Step 2: redeploy base holdings
    base_total = total * (1.0 - ctx.leaps_fraction)
    new_holdings = {a: base_total * float(ctx.base_target_w[a]) for a in ctx.base_assets}

    # Step 3: zero sleeve and pool
    new_sleeve = 0.0
    new_pool = 0.0

    # Defaults if LEAPS not active
    new_leaps_value = 0.0
    new_ledger = state.leaps_ledger
    new_window_ledgers = state.all_window_ledgers

    # Step 4: fresh LEAPS simulation for new Long window
    if ctx.use_leaps and ctx.underlying_prices is not None and ctx.config.leaps_config is not None:
        win_end = ctx.long_window_end.get(
            inputs.date_ts,
            pd.Timestamp(ctx.return_data.returns.index[-1]),
        )
        win_prices = ctx.underlying_prices.loc[inputs.date_ts:win_end]
        new_ledger = run_leaps_simulation(
            win_prices,
            ctx.leaps_monthly,
            ctx.config.leaps_config,
            risk_free_series=ctx.return_data.risk_free_rate,
            iv_series=ctx.raw_vix,
            initial_capital=total * ctx.leaps_fraction,
        )
        new_window_ledgers = state.all_window_ledgers + (new_ledger,)

        # Bug 2 fix: price at CREATION IV (raw VIX), NOT smoothed MTM IV.
        creation_iv = (
            ctx.iv
            if inputs.raw_vix_value is None
            else max(inputs.raw_vix_value, ctx.iv)
        )
        spot = inputs.spot if inputs.spot is not None else 0.0
        new_leaps_value = sum(
            price_leaps_contract(c, spot, inputs.date_ts, creation_iv, inputs.rfr)
            for c in _live_contracts(new_ledger, inputs.date_ts)
        )

    # Step 5: reset leaps_scale (clears orphaned keys from old window — risk R7)
    new_leaps_scale: dict[LeapsContract, float] = {}

    return replace(
        state,
        holdings=new_holdings,
        defensive_sleeve=new_sleeve,
        leaps_pool=new_pool,
        leaps_value=new_leaps_value,
        leaps_ledger=new_ledger,
        leaps_scale=new_leaps_scale,
        all_window_ledgers=new_window_ledgers,
    )
