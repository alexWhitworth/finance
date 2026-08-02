"""F-013: _apply_rebalance — pure step function for base-asset rebalancing.

Handles both QUARTERLY (scheduled) and DRIFT (band-based monthly check)
rebalance paths. Both paths are NAV-neutral for base holdings (invariant A5).
"""

from dataclasses import replace

import pandas as pd

from finance.leverage import RebalanceRule, _live_contracts
from finance.portfolio import BacktestContext, DayInputs, PortfolioState, should_rebalance


def _apply_rebalance(
    state: PortfolioState,
    inputs: DayInputs,
    ctx: BacktestContext,
) -> PortfolioState:
    """Realign base holdings to base_target_w on scheduled or drift-triggered dates.

    Two rebalance paths are implemented:

    QUARTERLY: fires when ``inputs.is_rebal_date`` is True and ``ctx.base_assets``
    is non-empty.  Distributes ``sum(holdings)`` across assets by
    ``ctx.base_target_w``.  When GTT is active and today is defensive
    (``inputs.regime_t == 0``), the repopulated governed-asset allocation is
    immediately re-swept into ``state.defensive_sleeve`` and zeroed in
    ``holdings``.

    DRIFT: fires when ``ctx.config.rebalance_rule == RebalanceRule.DRIFT`` and
    ``inputs.is_month_end``.  Computes current weights (base + LEAPS relative to
    total NAV) and calls ``should_rebalance``.  If triggered, base holdings are
    realigned and any LEAPS overshoot beyond ``ctx.leaps_fraction`` is scaled down
    pro-rata; proceeds are redistributed to base holdings.

    Arguments:
        state: Current immutable portfolio snapshot.
        inputs: Per-day read-only scalars for the current trading day.
        ctx: Immutable backtest configuration and precomputed series.

    Returns:
        New PortfolioState with updated holdings, defensive_sleeve, leaps_value,
        and leaps_scale.  Returns ``state`` unchanged when neither rebalance path
        fires.

    Notes:
        Invariant A5: ``sum(holdings_out) == sum(holdings_in)`` within 1e-9 for
        the QUARTERLY path (no capital enters or leaves the base sleeve).
        The DRIFT path is also NAV-neutral for base holdings; LEAPS overshoot
        proceeds are re-injected into base holdings rather than lost.
    """
    holdings = dict(state.holdings)
    defensive_sleeve = state.defensive_sleeve
    leaps_value = state.leaps_value
    leaps_scale = dict(state.leaps_scale)

    # ------------------------------------------------------------------
    # QUARTERLY rebalance (invariant A5: NAV-neutral)
    # ------------------------------------------------------------------
    if inputs.is_rebal_date and ctx.base_assets:
        base_nav = sum(holdings.values())
        holdings = {a: base_nav * float(ctx.base_target_w[a]) for a in ctx.base_assets}
        # GTT defensive override: governed assets that were just repopulated
        # must be swept back into the sleeve.
        if ctx.gtt_active and inputs.regime_t == 0:
            for k in ctx.governed_base:
                defensive_sleeve += holdings[k]
                holdings[k] = 0.0

    # ------------------------------------------------------------------
    # DRIFT rebalance: monthly band check
    # ------------------------------------------------------------------
    if ctx.config.rebalance_rule == RebalanceRule.DRIFT and inputs.is_month_end:
        base_val = sum(holdings.values())
        total_val = base_val + leaps_value
        if total_val > 0.0:
            weights_now: dict[str, float] = {
                a: holdings[a] / total_val for a in ctx.base_assets
            }
            for k in ctx.leaps_keys:
                share = float(ctx.w[k]) / ctx.leaps_fraction if ctx.leaps_fraction > 0 else 0.0
                weights_now[k] = leaps_value * share / total_val
            current_weights = pd.Series(weights_now)
            if should_rebalance(current_weights, ctx.w, RebalanceRule.DRIFT):
                # Realign base assets to their target weights within the base sleeve.
                holdings = {a: base_val * float(ctx.base_target_w[a]) for a in ctx.base_assets}
                # Trim LEAPS overshoot back to target fraction at current total NAV.
                target_leaps_now = total_val * ctx.leaps_fraction
                if leaps_value > target_leaps_now and leaps_value > 0:
                    close_scale = target_leaps_now / leaps_value
                    net_proceeds = leaps_value - target_leaps_now
                    for c in _live_contracts(state.leaps_ledger, inputs.date_ts):  # type: ignore[arg-type]
                        leaps_scale[c] = leaps_scale.get(c, 1.0) * close_scale
                    # Return LEAPS proceeds to base holdings by base target weights.
                    if ctx.base_assets:
                        for a in ctx.base_assets:
                            holdings[a] += net_proceeds * float(ctx.base_target_w[a])
                    leaps_value = target_leaps_now

    # Return the same object if nothing changed (identity check for callers).
    if (
        holdings == state.holdings
        and defensive_sleeve == state.defensive_sleeve
        and leaps_value == state.leaps_value
        and leaps_scale == state.leaps_scale
    ):
        return state

    return replace(
        state,
        holdings=holdings,
        defensive_sleeve=defensive_sleeve,
        leaps_value=leaps_value,
        leaps_scale=leaps_scale,
    )
