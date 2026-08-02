"""Backtest step functions — pure per-day pipeline over PortfolioState.

Each function is a pure step: same inputs → same output, no side effects.
All state mutations return new PortfolioState instances via dataclasses.replace().

Functions (in loop order):
    _build_context          — pre-loop: validate inputs and build BacktestContext
    _build_initial_state    — pre-loop: day-0 capital deployment
    _extract_day_inputs     — per-day index lookup from BacktestContext
    _apply_gtt_open         — sweep governed equity into defensive sleeve
    _apply_gtt_force_close  — force-close LEAPS on Long->Defensive transition
    _apply_returns          — compound base holdings by daily asset returns
    _apply_defensive_compounding — compound defensive sleeve and LEAPS pool
    _compute_leaps_mtm      — mark live LEAPS to market (Bug 1 fix)
    _compute_nav_before_contrib — pre-contribution NAV
    _compute_port_return    — daily portfolio return excluding contributions
    _apply_contribution     — month-end contribution allocation
    _apply_rebalance        — QUARTERLY or DRIFT rebalance
    _apply_gtt_reentry      — Defensive->Long forced rebalance (Bug 2 fix)
    _compute_total_nav      — post-mutation total NAV
    _build_weight_row       — end-of-day realized weights for weight_history
    _advance_state          — advance prev_* fields for next iteration
    _assemble_leaps_ledger  — post-loop: freeze partial closes and concat windows
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from finance.consts import DEFAULT_IV, LEAPS_KEY_SUFFIX, VIX_MTM_WINDOW
from finance.data import PriceData
from finance.gtt import GttSignalData
from finance.leverage import (
    LeapsContract,
    LeapsGttCloseEvent,
    LeapsLedger,
    LeapsPartialCloseEvent,
    RebalanceRule,
    _live_contracts,
    close_leaps_contract,
    price_leaps_contract,
    run_leaps_simulation,
)
from finance.portfolio import (
    BacktestContext,
    DayInputs,
    GTT_RISK_FREE_KEY,
    PortfolioConfig,
    PortfolioState,
    _defensive_gross_return,
    _gtt_governed_keys,
    _long_windows,
    _reindex_position_mask,
    apply_contribution,
    get_rebalance_dates,
    should_rebalance,
)
from finance.returns import ReturnData


# ---------------------------------------------------------------------------
# Pre-loop: context and initial state
# ---------------------------------------------------------------------------


def _build_context(
    return_data: ReturnData,
    price_data: PriceData,
    config: PortfolioConfig,
    gtt_signal: GttSignalData | None,
) -> BacktestContext:
    """Validate all run_backtest inputs and precompute all series and index sets.

    Extracts the pre-loop setup from run_backtest: GTT validation, asset split,
    index set construction, GTT series alignment, and LEAPS series precomputation
    (excluding the run_leaps_simulation call, which belongs in _build_initial_state).

    Arguments:
        return_data: Daily simple returns and risk-free rate series.
        price_data: Adjusted close prices and optional vol-index prices.
        config: Full portfolio configuration, including optional LEAPS and GTT config.
        gtt_signal: GTT position mask (1=Long / 0=Defensive), or None to disable.

    Returns:
        Immutable BacktestContext populated with all 22 fields.

    Raises:
        ValueError: If exactly one of gtt_signal / config.gtt_config is set.
        ValueError: If non-R_f defensive_weights tickers are absent from return_data.
        ValueError: If any base asset in config.target_weights is absent from return_data.
        ValueError: If LEAPS keys are present but config.leaps_config is None.
        ValueError: If more than one distinct LEAPS underlying is requested.
        ValueError: If LEAPS underlying is absent from price_data.prices.
    """
    if (gtt_signal is None) != (config.gtt_config is None):
        raise ValueError(
            "gtt_signal and config.gtt_config must both be set or both be None; got "
            f"gtt_signal={'set' if gtt_signal is not None else 'None'}, "
            f"config.gtt_config={'set' if config.gtt_config is not None else 'None'}"
        )

    returns = return_data.returns

    if gtt_signal is not None:
        assert config.gtt_config is not None  # guaranteed by paired check above
        missing_def = [
            k
            for k in config.gtt_config.defensive_weights
            if k != GTT_RISK_FREE_KEY and k not in returns.columns
        ]
        if missing_def:
            raise ValueError(
                f"defensive_weights tickers absent from return_data: {missing_def}"
            )

    leaps_keys = tuple(k for k in config.target_weights if k.endswith(LEAPS_KEY_SUFFIX))
    base_assets = tuple(k for k in config.target_weights if k not in set(leaps_keys))

    missing = [a for a in base_assets if a not in returns.columns]
    if missing:
        raise ValueError(f"Assets missing from return_data: {missing}")

    use_leaps = len(leaps_keys) > 0
    if use_leaps and config.leaps_config is None:
        raise ValueError("LEAPS keys present in target_weights but leaps_config is None")

    w = pd.Series(config.target_weights)
    leaps_fraction = float(w[list(leaps_keys)].sum()) if leaps_keys else 0.0

    base_target_w = w[list(base_assets)]
    if len(base_assets) > 0 and base_target_w.sum() > 0:
        base_target_w = base_target_w / base_target_w.sum()

    idx = pd.DatetimeIndex(returns.index)

    rebal_dates: frozenset[pd.Timestamp] = frozenset(
        get_rebalance_dates(idx, config.rebalance_rule)
    )
    month_end_dates: frozenset[pd.Timestamp] = frozenset(
        pd.Timestamp(grp.index[-1])
        for _, grp in returns.groupby(idx.to_period("M"))
    )

    governed_base: tuple[str, ...] = ()
    defensive_weights: dict[str, float] = {}
    mask_aligned: pd.Series | None = None
    def_gross: pd.Series | None = None
    gtt_active = False
    long_window_end: dict[pd.Timestamp, pd.Timestamp] = {}

    if gtt_signal is not None:
        assert config.gtt_config is not None
        governed = _gtt_governed_keys(config.target_weights)
        governed_base = tuple(k for k in governed if k in set(base_assets))
        gtt_active = len(governed) > 0

    if gtt_active:
        assert gtt_signal is not None and config.gtt_config is not None
        defensive_weights = config.gtt_config.defensive_weights
        mask_aligned = _reindex_position_mask(gtt_signal.position_mask, idx)
        def_gross = _defensive_gross_return(
            returns, return_data.risk_free_rate, defensive_weights
        )
        long_window_end = dict(_long_windows(mask_aligned))

    underlying_prices: pd.Series | None = None
    iv: float = DEFAULT_IV
    rfr_series: pd.Series | None = None
    raw_vix: pd.Series | None = None
    mtm_iv_series: pd.Series | None = None
    leaps_monthly = 0.0

    if use_leaps:
        underlyings = {k.removesuffix(LEAPS_KEY_SUFFIX) for k in leaps_keys}
        if len(underlyings) > 1:
            raise ValueError(
                f"Only one LEAPS underlying is supported; got {sorted(underlyings)}"
            )
        underlying = next(iter(underlyings))
        if underlying not in price_data.prices.columns:
            raise ValueError(
                f"LEAPS underlying '{underlying}' absent from price_data.prices"
            )

        assert config.leaps_config is not None
        iv = config.leaps_config.iv
        underlying_prices = price_data.prices[underlying].reindex(idx, method="ffill")
        rfr_series = return_data.risk_free_rate.reindex(idx, method="ffill").fillna(0.0)

        if (
            not price_data.vol_prices.empty
            and underlying in price_data.vol_prices.columns
        ):
            raw_vix = price_data.vol_prices[underlying].reindex(idx, method="ffill")
            mtm_iv_series = raw_vix.rolling(VIX_MTM_WINDOW).mean().ffill()

        leaps_monthly = config.monthly_contribution * leaps_fraction

    base_contribution = config.monthly_contribution * (1.0 - leaps_fraction)

    return BacktestContext(
        base_assets=base_assets,
        leaps_keys=leaps_keys,
        leaps_fraction=leaps_fraction,
        base_target_w=base_target_w,
        governed_base=governed_base,
        gtt_active=gtt_active,
        defensive_weights=defensive_weights,
        use_leaps=use_leaps,
        iv=iv,
        leaps_monthly=leaps_monthly,
        base_contribution=base_contribution,
        config=config,
        return_data=return_data,
        underlying_prices=underlying_prices,
        raw_vix=raw_vix,
        mtm_iv_series=mtm_iv_series,
        rfr_series=rfr_series,
        mask_aligned=mask_aligned,
        def_gross=def_gross,
        rebal_dates=rebal_dates,
        month_end_dates=month_end_dates,
        long_window_end=long_window_end,
        w=w,
    )


def _build_initial_state(ctx: BacktestContext) -> PortfolioState:
    """Build the PortfolioState for the first loop iteration.

    Runs the first LEAPS simulation scoped to the first Long window when LEAPS
    are active, then allocates base holdings from the remaining NAV fraction.
    All loop-accumulator fields are initialised to their empty/zero values.

    Arguments:
        ctx: Immutable BacktestContext produced by _build_context.

    Returns:
        PortfolioState with holdings, leaps_ledger, and all_window_ledgers populated.

    Notes:
        When GTT is active and the first window is entirely Defensive, an empty
        ledger (no contracts) is returned.
    """
    leaps_ledger: LeapsLedger | None = None
    all_window_ledgers: tuple[LeapsLedger, ...] = ()

    if (
        ctx.use_leaps
        and ctx.underlying_prices is not None
        and ctx.config.leaps_config is not None
    ):
        if ctx.gtt_active and ctx.mask_aligned is not None:
            long_wins = _long_windows(ctx.mask_aligned)
            if long_wins:
                first_start, first_end = long_wins[0]
                win_prices: pd.Series = ctx.underlying_prices.loc[first_start:first_end]
            else:
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


# ---------------------------------------------------------------------------
# Per-day input extraction
# ---------------------------------------------------------------------------


def _extract_day_inputs(
    date: pd.Timestamp,
    ctx: BacktestContext,
) -> DayInputs:
    """Extract per-day scalar values from precomputed context series.

    Pure index lookup: no computation beyond series access and None guards.

    Arguments:
        date: The trading day to extract inputs for.
        ctx: Immutable BacktestContext holding all precomputed series.

    Returns:
        DayInputs populated with values at ``date`` from ctx.
    """
    day_ret: pd.Series = ctx.return_data.returns.loc[date]  # type: ignore[assignment]
    regime_t = int(ctx.mask_aligned.loc[date]) if ctx.mask_aligned is not None else 1
    def_gross_return = float(ctx.def_gross.loc[date]) if ctx.def_gross is not None else 0.0
    spot = float(ctx.underlying_prices.loc[date]) if ctx.underlying_prices is not None else None
    raw_vix_value = float(ctx.raw_vix.loc[date]) if ctx.raw_vix is not None else None

    mtm_iv_value: float | None = None
    if ctx.mtm_iv_series is not None:
        raw = ctx.mtm_iv_series.loc[date]
        mtm_iv_value = float(raw)  # may be NaN; callers guard with pd.notna()

    rfr = float(ctx.rfr_series.loc[date]) if ctx.rfr_series is not None else 0.0
    is_month_end = date in ctx.month_end_dates
    is_rebal_date = date in ctx.rebal_dates

    return DayInputs(
        date_ts=date,
        day_ret=day_ret,
        regime_t=regime_t,
        def_gross_return=def_gross_return,
        spot=spot,
        raw_vix_value=raw_vix_value,
        mtm_iv_value=mtm_iv_value,
        rfr=rfr,
        is_month_end=is_month_end,
        is_rebal_date=is_rebal_date,
    )


# ---------------------------------------------------------------------------
# GTT open / force-close
# ---------------------------------------------------------------------------


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
        increased by the total swept value. Returns state unchanged on no-op paths.

    Notes:
        Accounting invariant A-open:
        sum(new.holdings.values()) + new.defensive_sleeve
        == sum(old.holdings.values()) + old.defensive_sleeve
    """
    if not ctx.gtt_active or inputs.regime_t != 0:
        return state
    new_holdings = dict(state.holdings)
    new_sleeve = state.defensive_sleeve
    for k in ctx.governed_base:
        new_sleeve += new_holdings.get(k, 0.0)
        new_holdings[k] = 0.0
    return replace(state, holdings=new_holdings, defensive_sleeve=new_sleeve)


def _apply_gtt_force_close(
    state: PortfolioState,
    inputs: DayInputs,
    ctx: BacktestContext,
) -> PortfolioState:
    """Force-close all live LEAPS contracts on a Long->Defensive regime transition.

    Fires only when ctx.gtt_active, state.prev_regime==1, inputs.regime_t==0,
    state.leaps_ledger is not None, state.prev_date_ts is not None,
    ctx.underlying_prices is not None, and ctx.config.leaps_config is not None.

    Net proceeds (after LTCG tax) are added to leaps_pool; each close event is
    appended to all_gtt_closes.

    Arguments:
        state: Current PortfolioState.
        inputs: Per-day read-only inputs for the current trading day.
        ctx: Immutable BacktestContext.

    Returns:
        New PortfolioState with leaps_pool and all_gtt_closes updated, or state
        unchanged when the transition condition does not fire.

    Notes:
        Invariant A3: new_state.leaps_pool - state.leaps_pool ==
            sum(e.net_proceeds for e in new events)
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

    prev_ts: pd.Timestamp = state.prev_date_ts
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


# ---------------------------------------------------------------------------
# Returns and defensive compounding
# ---------------------------------------------------------------------------


def _apply_returns(
    state: PortfolioState,
    inputs: DayInputs,
    ctx: BacktestContext,
) -> PortfolioState:
    """Compound base holdings by today's asset returns.

    Arguments:
        state: Current portfolio state.
        inputs: Per-day read-only inputs for today.
        ctx: Immutable backtest configuration.

    Returns:
        New PortfolioState with updated holdings. All other fields unchanged.

    Notes:
        Invariant: holdings_out[a] == holdings_in[a] * (1 + day_ret[a])
        for all a in ctx.base_assets.
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
    defensive return before redeployment to target at the close.

    No-op when GTT is inactive, ctx.def_gross is None, or on a pure Long day
    (prev_regime == 1 and regime_t == 1).

    Arguments:
        state: Current portfolio state.
        inputs: Per-day read-only inputs for today.
        ctx: Immutable backtest configuration.

    Returns:
        New PortfolioState with updated defensive_sleeve and leaps_pool.
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


# ---------------------------------------------------------------------------
# LEAPS mark-to-market (Bug 1 fix)
# ---------------------------------------------------------------------------


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
    - ctx.gtt_active and ctx.use_leaps and state.prev_regime == 0 and regime_t == 1
      (re-entry day — Bug 1 fix: old ledger has stale contracts)

    Arguments:
        state: Current PortfolioState snapshot.
        inputs: Per-day read-only inputs for the current trading day.
        ctx: Immutable per-backtest configuration and precomputed series.

    Returns:
        New PortfolioState with leaps_value updated (or suppressed to 0.0).
    """
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


# ---------------------------------------------------------------------------
# NAV arithmetic
# ---------------------------------------------------------------------------


def _compute_nav_before_contrib(state: PortfolioState) -> float:
    """Return pre-contribution NAV: sum(holdings) + leaps_value + sleeve + pool.

    Arguments:
        state: Current PortfolioState (read-only).

    Returns:
        Pre-contribution NAV as a float.
    """
    return (
        sum(state.holdings.values())
        + state.leaps_value
        + state.defensive_sleeve
        + state.leaps_pool
    )


def _compute_port_return(nav_before: float, prev_total_nav: float) -> float:
    """Return daily portfolio return excluding contributions.

    Arguments:
        nav_before: Pre-contribution NAV for today.
        prev_total_nav: End-of-day NAV from the prior trading day.

    Returns:
        Simple return: nav_before / prev_total_nav - 1.0.
    """
    return nav_before / prev_total_nav - 1.0


# ---------------------------------------------------------------------------
# Contribution and rebalance
# ---------------------------------------------------------------------------


def _apply_contribution(
    state: PortfolioState,
    inputs: DayInputs,
    ctx: BacktestContext,
    nav_before: float,
) -> PortfolioState:
    """Apply monthly contribution to holdings, sleeve, and LEAPS pool.

    On month-end days, allocates ctx.base_contribution across base_assets
    proportional to ctx.base_target_w. When GTT is active and regime_t==0,
    governed tickers' allocation is diverted into the defensive sleeve and
    ctx.leaps_monthly is added to the LEAPS pool.

    Arguments:
        state: Current PortfolioState before contribution.
        inputs: DayInputs for the current trading day.
        ctx: BacktestContext with contribution amounts, weights, and GTT flags.
        nav_before: NAV computed before this contribution step.

    Returns:
        Updated PortfolioState, or state unchanged on non-month-end days.

    Notes:
        Accounting invariant (Long month-end):
        sum(new.holdings.values()) == sum(old.holdings.values()) + ctx.base_contribution
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


def _apply_rebalance(
    state: PortfolioState,
    inputs: DayInputs,
    ctx: BacktestContext,
) -> PortfolioState:
    """Realign base holdings to base_target_w on scheduled or drift-triggered dates.

    QUARTERLY: fires when inputs.is_rebal_date and ctx.base_assets is non-empty.
    Distributes sum(holdings) by ctx.base_target_w. When GTT is active and
    defensive (regime_t==0), repopulated governed assets are re-swept to sleeve.

    DRIFT: fires when config.rebalance_rule==DRIFT and inputs.is_month_end.
    Checks current weights; if outside the band, realigns and trims any LEAPS
    overshoot pro-rata (proceeds returned to base holdings, tax-free).

    Arguments:
        state: Current immutable portfolio snapshot.
        inputs: Per-day read-only scalars.
        ctx: Immutable backtest configuration and precomputed series.

    Returns:
        New PortfolioState with updated holdings, defensive_sleeve, leaps_value,
        and leaps_scale. Returns state unchanged when neither path fires.

    Notes:
        Invariant A5: sum(holdings_out) == sum(holdings_in) within 1e-9
        for the QUARTERLY path.
    """
    holdings = dict(state.holdings)
    defensive_sleeve = state.defensive_sleeve
    leaps_value = state.leaps_value
    leaps_scale = dict(state.leaps_scale)

    if inputs.is_rebal_date and ctx.base_assets:
        base_nav = sum(holdings.values())
        holdings = {a: base_nav * float(ctx.base_target_w[a]) for a in ctx.base_assets}
        if ctx.gtt_active and inputs.regime_t == 0:
            for k in ctx.governed_base:
                defensive_sleeve += holdings[k]
                holdings[k] = 0.0

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
                holdings = {a: base_val * float(ctx.base_target_w[a]) for a in ctx.base_assets}
                target_leaps_now = total_val * ctx.leaps_fraction
                if leaps_value > target_leaps_now and leaps_value > 0:
                    close_scale = target_leaps_now / leaps_value
                    net_proceeds = leaps_value - target_leaps_now
                    for c in _live_contracts(state.leaps_ledger, inputs.date_ts):  # type: ignore[arg-type]
                        leaps_scale[c] = leaps_scale.get(c, 1.0) * close_scale
                    if ctx.base_assets:
                        for a in ctx.base_assets:
                            holdings[a] += net_proceeds * float(ctx.base_target_w[a])
                    leaps_value = target_leaps_now

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


# ---------------------------------------------------------------------------
# GTT re-entry (Bug 2 fix)
# ---------------------------------------------------------------------------


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

    Arguments:
        state: Current PortfolioState snapshot.
        inputs: Per-day read-only inputs for the current trading day.
        ctx: Immutable backtest configuration and precomputed series.

    Returns:
        Updated PortfolioState with re-entry rebalance applied, or state unchanged.

    Notes:
        Invariant A2: sum(new_holdings.values()) + new_leaps_value == total within 1e-9.
        Invariant A4: new_leaps_value == total * ctx.leaps_fraction within 1e-6.
    """
    if not (ctx.gtt_active and state.prev_regime == 0 and inputs.regime_t == 1):
        return state

    total = sum(state.holdings.values()) + state.defensive_sleeve + state.leaps_pool
    base_total = total * (1.0 - ctx.leaps_fraction)
    new_holdings = {a: base_total * float(ctx.base_target_w[a]) for a in ctx.base_assets}
    new_sleeve = 0.0
    new_pool = 0.0
    new_leaps_value = 0.0
    new_ledger = state.leaps_ledger
    new_window_ledgers = state.all_window_ledgers

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


# ---------------------------------------------------------------------------
# Final day-loop helpers
# ---------------------------------------------------------------------------


def _compute_total_nav(state: PortfolioState) -> float:
    """Return total portfolio NAV after all day mutations.

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
    - GTT defensive parked capital: (sleeve + pool) * dw / total_nav for each
      defensive_weight key.

    Arguments:
        state: Current PortfolioState after all mutations.
        total_nav: End-of-day NAV; must equal sum of all components.
        ctx: BacktestContext supplying base_assets, leaps_keys, leaps_fraction,
            w, gtt_active, and defensive_weights.

    Returns:
        Dict mapping each asset key to its realized weight. Returns all-zero
        weights when total_nav <= 0.

    Notes:
        Invariant: sum(result.values()) == 1.0 within 1e-9 for total_nav > 0.
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
       contract fraction is recorded as a LeapsPartialCloseEvent.

    2. GTT multi-window assembly: if GTT + LEAPS are both active, all
       per-window ledgers in state.all_window_ledgers are concatenated.

    Arguments:
        state: Final PortfolioState after the backtest loop.
        ctx: BacktestContext supplying gtt_active, use_leaps, and config.
        final_date: Last trading date; used as close_date for partial-close events.

    Returns:
        Assembled LeapsLedger, or None if no LEAPS overlay was active.
    """
    leaps_ledger = state.leaps_ledger
    partial_close_list = list(
        leaps_ledger.partial_close_events if leaps_ledger is not None else ()
    )

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
