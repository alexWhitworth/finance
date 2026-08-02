"""Step function F-010: _compute_leaps_mtm.

Marks live LEAPS contracts to market and stores the result in state.leaps_value.
Bug 1 fix: suppress MTM on re-entry days to prevent double-counting from the
stale old-window ledger.
"""

from dataclasses import replace

import pandas as pd

from finance.leverage import _live_contracts, price_leaps_contract
from finance.portfolio import BacktestContext, DayInputs, PortfolioState


def _compute_leaps_mtm(
    state: PortfolioState,
    inputs: DayInputs,
    ctx: BacktestContext,
) -> PortfolioState:
    """Mark live LEAPS contracts to market using smoothed MTM IV.

    Suppressed (leaps_value=0.0) when:
    - ctx.use_leaps is False, OR
    - state.leaps_ledger is None, OR
    - ctx.underlying_prices is None, OR
    - ctx.gtt_active and inputs.regime_t == 0 (defensive window), OR
    - ctx.gtt_active and ctx.use_leaps and state.prev_regime == 0 and inputs.regime_t == 1
      (re-entry day — Bug 1 fix: old ledger has stale contracts; _apply_gtt_reentry will overwrite)

    MTM IV: max(inputs.mtm_iv_value, ctx.iv) when mtm_iv_value is not None/NaN; else ctx.iv.

    Arguments:
        state: Current PortfolioState snapshot.
        inputs: Per-day read-only inputs for the current trading day.
        ctx: Immutable per-backtest configuration and precomputed series.

    Returns:
        New PortfolioState with leaps_value updated (or suppressed to 0.0).
    """
    # Suppression conditions
    if not ctx.use_leaps or state.leaps_ledger is None or ctx.underlying_prices is None:
        return replace(state, leaps_value=0.0)
    if ctx.gtt_active and inputs.regime_t == 0:
        return replace(state, leaps_value=0.0)
    if ctx.gtt_active and ctx.use_leaps and state.prev_regime == 0 and inputs.regime_t == 1:
        return replace(state, leaps_value=0.0)  # Bug 1 fix

    spot = inputs.spot
    assert spot is not None  # guaranteed when use_leaps
    rfr = inputs.rfr
    day_iv = ctx.iv
    if inputs.mtm_iv_value is not None and pd.notna(inputs.mtm_iv_value):
        day_iv = max(float(inputs.mtm_iv_value), ctx.iv)
    live = _live_contracts(state.leaps_ledger, inputs.date_ts)
    leaps_value: float = sum(
        price_leaps_contract(c, spot, inputs.date_ts, day_iv, rfr)
        * state.leaps_scale.get(c, 1.0)
        for c in live
    )
    return replace(state, leaps_value=leaps_value)
