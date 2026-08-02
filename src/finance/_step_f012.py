"""F-012: _apply_contribution step function.

Handles month-end contribution allocation extracted from run_backtest lines 837-851.
"""

from __future__ import annotations

from dataclasses import replace

from finance.portfolio import BacktestContext, DayInputs, PortfolioState, apply_contribution


def _apply_contribution(
    state: PortfolioState,
    inputs: DayInputs,
    ctx: BacktestContext,
    nav_before: float,
) -> PortfolioState:
    """Apply monthly contribution to holdings, sleeve, and LEAPS pool.

    On month-end days, allocates ctx.base_contribution across base_assets
    according to ctx.base_target_w. When the GTT overlay is active and
    regime_t == 0 (Defensive), governed tickers' allocation is diverted into
    the defensive sleeve and ctx.leaps_monthly is added to the LEAPS pool.
    On Long days (regime_t == 1), all allocation flows into holdings normally.

    Arguments:
        state: Current PortfolioState before contribution.
        inputs: DayInputs for the current trading day.
        ctx: BacktestContext with contribution amounts, weights, and GTT flags.
        nav_before: NAV computed before this contribution step; passed to
            apply_contribution() for future weight strategies that need it.

    Returns:
        Updated PortfolioState. On non-month-end days or when ctx.base_assets
        is empty, the original state is returned unchanged.

    Notes:
        Accounting invariant (Long month-end):
        sum(new.holdings.values()) == sum(old.holdings.values()) + ctx.base_contribution
        within floating-point tolerance (~1e-9).
    """
    if not inputs.is_month_end or not ctx.base_assets:
        return state

    alloc = apply_contribution(nav_before, ctx.base_contribution, ctx.base_target_w)
    new_holdings = dict(state.holdings)
    new_sleeve = state.defensive_sleeve
    new_pool = state.leaps_pool

    for a in ctx.base_assets:
        if ctx.gtt_active and inputs.regime_t == 0 and a in ctx.governed_base:
            new_sleeve += alloc[a]
        else:
            new_holdings[a] = new_holdings.get(a, 0.0) + alloc[a]

    if ctx.gtt_active and inputs.regime_t == 0:
        new_pool += ctx.leaps_monthly

    return replace(state, holdings=new_holdings, defensive_sleeve=new_sleeve, leaps_pool=new_pool)
