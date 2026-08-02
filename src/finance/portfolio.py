"""Portfolio backtest engine — rebalancing, contribution, and NAV loop.

All business logic is pure. Receives ReturnData + PriceData + PortfolioConfig
and produces BacktestResult consumed by metrics.py.
"""

from __future__ import annotations

import pandas as pd

from finance._backtest_steps import (
    _build_context,
    _build_initial_state,
    _extract_day_inputs,
    _apply_gtt_open,
    _apply_gtt_force_close,
    _apply_defensive_compounding,
    _apply_returns,
    _compute_leaps_mtm,
    _compute_nav_before_contrib,
    _compute_port_return,
    _apply_contribution,
    _apply_rebalance,
    _apply_gtt_reentry,
    _advance_state,
    _assemble_leaps_ledger,
    _build_weight_row,
    _compute_total_nav,
    _get_rebalance_dates,
    _should_rebalance,
    apply_contribution,
    _defensive_gross_return,
    _gtt_governed_keys,
    _long_windows,
)
from finance._portfolio_types import (
    BacktestContext,
    BacktestResult,
    DayInputs,
    GttConfig,
    PortfolioConfig,
    PortfolioState,
)
from finance.data import PriceData
from finance.gtt import GttSignalData
from finance.returns import ReturnData

def run_backtest(
    return_data: ReturnData,
    price_data: PriceData,
    config: PortfolioConfig,
    gtt_signal: GttSignalData | None = None,
) -> BacktestResult:
    """Run the core portfolio backtest loop.

    For each trading day:
      a. Apply asset returns to holdings.
      b. Compute market return (before contribution, to exclude cash-flow effects).
      c. On month-end: apply monthly_contribution proportional to target weights.
      d. On rebalance date: realign holdings to target weights.
      e. If LEAPS keys present: include carved-out LEAPS mark-to-market in total NAV.

    GTT overlay (opt-in via a matched gtt_signal + config.gtt_config pair):
      When gtt_signal is provided, the GTT_EQUITY_TICKERS leg (VTI and its _LEAPS
      variant) is governed by gtt_signal.position_mask (1=Long, 0=Defensive). On
      defensive days the governed capital is moved into a fixed-weight defensive
      sleeve and live LEAPS contracts are force-closed; on re-entry a forced
      rebalance re-anchors the portfolio to target_weights. gtt_signal=None
      preserves the pre-GTT behavior exactly.

    LEAPS (Model B carve-out, triggered by any "*_LEAPS" key in target_weights):
      - The underlying (key without the "_LEAPS" suffix) must exist in
        price_data.prices for absolute spot pricing.
      - LEAPS capital is carved out of NAV: initial_nav * leaps_fraction is
        deployed day-1 and the LEAPS share of each monthly contribution flows
        into run_leaps_simulation; base holdings hold the remainder.
      - Dynamic IV: when price_data.vol_prices has a column keyed by the LEAPS
        underlying ticker (e.g. 'VTI'), raw values drive contract creation and
        rolls, while a VIX_MTM_WINDOW-day rolling mean drives daily MTM.
        config.leaps_config.iv is the floor throughout. Absent that column (or
        with an empty vol_prices), config.leaps_config.iv is used everywhere.
      - run_leaps_simulation is called internally; no external ledger accepted.

    Arguments:
        return_data: ReturnData containing daily simple returns for all assets.
        price_data: PriceData providing absolute asset prices (used for LEAPS spot).
        config: PortfolioConfig specifying weights, contributions, and rebalancing.
        gtt_signal: Optional pre-computed GTT signal. None disables the overlay and
            preserves the pre-GTT behavior exactly. When provided, config.gtt_config
            must also be set (and vice versa).

    Returns:
        BacktestResult with NAV series, weight history, return series, and ledger.

    Raises:
        ValueError: If any base asset in config.target_weights is absent from return_data.
        ValueError: If a LEAPS underlying (key without "_LEAPS") is absent from price_data.prices.
        ValueError: If more than one distinct LEAPS underlying is requested.
        ValueError: If LEAPS keys are present but config.leaps_config is None.
        ValueError: If exactly one of gtt_signal / config.gtt_config is set (both or
            neither required).
        ValueError: If gtt_signal is set and a non-R_f defensive_weights ticker is
            absent from return_data.
    """
    ctx = _build_context(return_data, price_data, config, gtt_signal)
    state = _build_initial_state(ctx)

    nav_values: list[float] = []
    return_values: list[float] = []
    weight_rows: list[dict[str, float]] = []

    for date in ctx.return_data.returns.index:
        date_ts = pd.Timestamp(date)
        inputs = _extract_day_inputs(date_ts, ctx)
        state = _apply_gtt_open(state, inputs, ctx)
        state = _apply_gtt_force_close(state, inputs, ctx)
        state = _apply_returns(state, inputs, ctx)
        state = _apply_defensive_compounding(state, inputs, ctx)
        state = _compute_leaps_mtm(state, inputs, ctx)
        nav_before = _compute_nav_before_contrib(state)
        port_return = _compute_port_return(nav_before, state.prev_total_nav)
        state = _apply_contribution(state, inputs, ctx, nav_before)
        state = _apply_rebalance(state, inputs, ctx)
        state = _apply_gtt_reentry(state, inputs, ctx)
        total_nav = _compute_total_nav(state)
        weight_row = _build_weight_row(state, total_nav, ctx)
        nav_values.append(total_nav)
        return_values.append(port_return)
        weight_rows.append(weight_row)
        state = _advance_state(state, total_nav, inputs)

    final_date = pd.Timestamp(ctx.return_data.returns.index[-1])
    leaps_ledger = _assemble_leaps_ledger(state, ctx, final_date)

    returns = ctx.return_data.returns
    nav_series = pd.Series(nav_values, index=returns.index, name="NAV")
    return_series = pd.Series(return_values, index=returns.index, name="portfolio_return")
    weight_history = pd.DataFrame(weight_rows, index=returns.index)
    if ctx.gtt_active:
        weight_history = weight_history.fillna(0.0)

    return BacktestResult(
        nav_series=nav_series,
        weight_history=weight_history,
        return_series=return_series,
        leaps_ledger=leaps_ledger,
        config=config,
    )
