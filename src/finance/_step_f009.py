"""F-009: _apply_returns and _apply_defensive_compounding step functions.

Pure step functions extracted from run_backtest. Each takes an immutable
PortfolioState plus read-only DayInputs and BacktestContext and returns a
new PortfolioState via dataclasses.replace(). No side effects.
"""

from __future__ import annotations

from dataclasses import replace

from finance.portfolio import BacktestContext, DayInputs, PortfolioState


def _apply_returns(
    state: PortfolioState,
    inputs: DayInputs,
    ctx: BacktestContext,
) -> PortfolioState:
    """Compound base holdings by today's asset returns.

    Invariant: holdings_out[a] == holdings_in[a] * (1 + day_ret[a]) for all
    a in ctx.base_assets.

    Arguments:
        state: Current portfolio state.
        inputs: Per-day read-only inputs for today.
        ctx: Immutable backtest configuration.

    Returns:
        New PortfolioState with updated holdings. All other fields unchanged.
    """
    new_holdings = {
        a: state.holdings[a] * (1.0 + float(inputs.day_ret[a]))
        for a in ctx.base_assets
    }
    return replace(state, holdings=new_holdings)


def _apply_defensive_compounding(
    state: PortfolioState,
    inputs: DayInputs,
    ctx: BacktestContext,
) -> PortfolioState:
    """Compound defensive_sleeve and leaps_pool by the blended defensive return.

    Fires on every defensive day (regime_t == 0) and on the re-entry day
    (prev_regime == 0, regime_t == 1), where the parked capital earns one final
    defensive day return before being redeployed to target at the close.

    No-op when GTT is inactive or ctx.def_gross is None, or on a pure Long day
    (prev_regime == 1 and regime_t == 1).

    Invariant (defensive/re-entry day):
        defensive_sleeve_out == defensive_sleeve_in * (1 + def_gross_return)
        leaps_pool_out       == leaps_pool_in       * (1 + def_gross_return)

    Arguments:
        state: Current portfolio state.
        inputs: Per-day read-only inputs for today.
        ctx: Immutable backtest configuration.

    Returns:
        New PortfolioState with updated defensive_sleeve and leaps_pool.
        All other fields unchanged.
    """
    if not ctx.gtt_active or ctx.def_gross is None:
        return state
    if not (inputs.regime_t == 0 or state.prev_regime == 0):
        return state
    factor = 1.0 + inputs.def_gross_return
    return replace(
        state,
        defensive_sleeve=state.defensive_sleeve * factor,
        leaps_pool=state.leaps_pool * factor,
    )
