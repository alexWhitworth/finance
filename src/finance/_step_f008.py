"""F-008: _apply_gtt_force_close — force-close all live LEAPS on Long->Defensive transition.

On the day the GTT regime flips from Long (prev_regime=1) to Defensive (regime_t=0),
every live LEAPS contract is closed at the previous day's (prev_date_ts) spot price
and raw-VIX IV. Net proceeds after LTCG tax are added to leaps_pool to ride the
defensive return.

Invariant A3: new_leaps_pool - old_leaps_pool == sum(evt.net_proceeds for new events).
"""

from dataclasses import replace

import pandas as pd

from finance.leverage import LeapsGttCloseEvent, _live_contracts, close_leaps_contract
from finance.portfolio import BacktestContext, DayInputs, PortfolioState


def _apply_gtt_force_close(
    state: PortfolioState,
    inputs: DayInputs,
    ctx: BacktestContext,
) -> PortfolioState:
    """Force-close all live LEAPS contracts on a Long->Defensive regime transition.

    Fires only when all five conditions are met:
      - ctx.gtt_active is True
      - state.prev_regime == 1 (was Long yesterday)
      - inputs.regime_t == 0 (is Defensive today)
      - state.leaps_ledger is not None
      - state.prev_date_ts is not None
      - ctx.underlying_prices is not None
      - ctx.config.leaps_config is not None

    Each live contract is priced at prev_date_ts using the raw-VIX IV (floored at
    ctx.iv) and closed via close_leaps_contract. If a leaps_scale entry exists for
    a contract, n_contracts is scaled before closing (partial-close fraction).
    Net proceeds are added to leaps_pool; the LeapsGttCloseEvent is appended to
    all_gtt_closes.

    Arguments:
        state: Current PortfolioState; must carry prev_regime, prev_date_ts, and
            leaps_ledger for this function to fire.
        inputs: Per-day read-only inputs for the current trading day.
        ctx: Immutable BacktestContext holding config and precomputed series.

    Returns:
        New PortfolioState with leaps_pool and all_gtt_closes updated if the
        transition condition fires, otherwise the original state unchanged.

    Notes:
        Invariant A3: new_state.leaps_pool - state.leaps_pool ==
            sum(e.net_proceeds for e in new_state.all_gtt_closes[len(state.all_gtt_closes):])
    """
    if not (
        ctx.gtt_active
        and state.prev_regime == 1
        and inputs.regime_t == 0
        and state.leaps_ledger is not None
        and state.prev_date_ts is not None
        and ctx.underlying_prices is not None
        and ctx.config.leaps_config is not None
    ):
        return state

    prev_ts: pd.Timestamp = state.prev_date_ts  # narrowed: not None after guard above
    close_spot = float(ctx.underlying_prices.loc[prev_ts])
    close_rfr = float(ctx.rfr_series.loc[prev_ts]) if ctx.rfr_series is not None else 0.0
    close_iv = ctx.iv
    if ctx.raw_vix is not None:
        close_iv = max(float(ctx.raw_vix.loc[prev_ts]), ctx.iv)

    new_closes: list[LeapsGttCloseEvent] = []
    pool_delta = 0.0

    for c in _live_contracts(state.leaps_ledger, prev_ts):
        scale = state.leaps_scale.get(c, 1.0)
        c_eff = replace(c, n_contracts=c.n_contracts * scale) if scale != 1.0 else c
        evt = close_leaps_contract(
            c_eff,
            prev_ts,
            close_spot,
            close_iv,
            ctx.config.leaps_config.ltcg_rate,
            close_rfr,
        )
        new_closes.append(evt)
        pool_delta += evt.net_proceeds

    if not new_closes:
        return state

    return replace(
        state,
        leaps_pool=state.leaps_pool + pool_delta,
        all_gtt_closes=state.all_gtt_closes + tuple(new_closes),
    )
