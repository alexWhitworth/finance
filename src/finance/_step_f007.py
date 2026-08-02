"""F-007 — _apply_gtt_open: sweep governed equity into the defensive sleeve.

Extracted from run_backtest lines 615–620. Converts the in-place mutation
(holdings[k] = 0.0; defensive_sleeve += holdings[k]) into a pure step
function over frozen dataclasses.

Accounting invariant A-open:
    sum(new.holdings.values()) + new.defensive_sleeve
    == sum(old.holdings.values()) + old.defensive_sleeve
"""

from dataclasses import replace

from finance._portfolio_types import BacktestContext, DayInputs, PortfolioState


def _apply_gtt_open(
    state: PortfolioState,
    inputs: DayInputs,
    ctx: BacktestContext,
) -> PortfolioState:
    """Sweep governed equity holdings into defensive_sleeve at open of a defensive day.

    No-op on Long days (regime_t==1) or when GTT inactive.

    Arguments:
        state: Current PortfolioState.
        inputs: Per-day inputs for this trading day.
        ctx: Immutable backtest context.

    Returns:
        New PortfolioState with governed holdings zeroed and defensive_sleeve
        increased by the total swept value.  Returns the original state object
        unchanged on no-op paths.
    """
    if not ctx.gtt_active or inputs.regime_t != 0:
        return state
    new_holdings = dict(state.holdings)
    new_sleeve = state.defensive_sleeve
    for k in ctx.governed_base:
        new_sleeve += new_holdings.get(k, 0.0)
        new_holdings[k] = 0.0
    return replace(state, holdings=new_holdings, defensive_sleeve=new_sleeve)
