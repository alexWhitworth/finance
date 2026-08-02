"""F-012: _apply_contribution step function and supporting dataclasses.

Defines the PortfolioState, DayInputs, and BacktestContext immutable dataclasses
used across the run_backtest decomposition, plus the pure step function
_apply_contribution that handles month-end contribution allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from finance.leverage import LeapsContract, LeapsGttCloseEvent, LeapsLedger
from finance.portfolio import PortfolioConfig, apply_contribution
from finance.returns import ReturnData

# ---------------------------------------------------------------------------
# Dataclasses (DS-001, DS-002, DS-003)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioState:
    """Immutable snapshot of all mutable per-day loop state.

    Every mutation produces a new instance via dataclasses.replace().
    Accumulators (all_window_ledgers, all_gtt_closes) are tuples to ensure
    pure step functions.

    Attributes:
        holdings: Dollar value per base asset; governed assets zeroed during
            defensive windows. Dict reference is frozen; step functions
            construct new dicts.
        defensive_sleeve: Governed equity capital swept in during GTT
            defensive windows.
        leaps_pool: Force-closed LEAPS net proceeds parked during defensive
            windows.
        leaps_value: Current LEAPS mark-to-market. Set by _compute_leaps_mtm
            (suppressed on re-entry days) or overwritten by _apply_gtt_reentry.
        prev_total_nav: End-of-day NAV from t-1; denominator of port_return
            calculation.
        prev_regime: GTT regime on t-1: 1=Long, 0=Defensive. Drives
            transition detection.
        prev_date_ts: Trading date of t-1; used for force-close spot/IV
            lookup on Long->Defensive day.
        leaps_ledger: Active per-window LEAPS simulation ledger; replaced on
            each GTT re-entry.
        leaps_scale: Surviving fraction per contract after drift
            partial-closes; keyed by original contract, value in (0, 1].
        all_window_ledgers: Immutable accumulator of one ledger per Long
            window for final assembly.
        all_gtt_closes: Immutable accumulator of all GTT force-close events
            across all defensive transitions.
    """

    holdings: dict[str, float]
    defensive_sleeve: float
    leaps_pool: float
    leaps_value: float
    prev_total_nav: float
    prev_regime: int
    prev_date_ts: pd.Timestamp | None
    leaps_ledger: LeapsLedger | None
    leaps_scale: dict[LeapsContract, float]
    all_window_ledgers: tuple[LeapsLedger, ...]
    all_gtt_closes: tuple[LeapsGttCloseEvent, ...]


@dataclass(frozen=True)
class DayInputs:
    """Per-day read-only inputs extracted from precomputed series.

    No field contains data with timestamp > date_ts (temporal invariant T1).

    Attributes:
        date_ts: Current trading day.
        day_ret: Asset returns for this day, indexed by ticker.
        regime_t: GTT regime for today: 0=Defensive, 1=Long (from
            mask_aligned).
        def_gross_return: Blended defensive sleeve return for today (0.0 if
            GTT inactive).
        spot: Underlying LEAPS spot price at date_ts (None if no LEAPS).
        raw_vix_value: Raw VIX at date_ts (None if no vol_prices). Used as
            creation IV on re-entry.
        mtm_iv_value: 30-day rolling mean VIX at date_ts (None or NaN during
            29-day warmup). Used for daily MTM.
        rfr: Risk-free rate at date_ts.
        is_month_end: True if date_ts is the last trading day of a calendar
            month.
        is_rebal_date: True if date_ts is a scheduled quarterly rebalance
            date.
    """

    date_ts: pd.Timestamp
    day_ret: pd.Series
    regime_t: int
    def_gross_return: float
    spot: float | None
    raw_vix_value: float | None
    mtm_iv_value: float | None
    rfr: float
    is_month_end: bool
    is_rebal_date: bool


@dataclass(frozen=True)
class BacktestContext:
    """Immutable per-backtest configuration and precomputed series.

    Constructed once before the loop. Contains no per-day state
    (PortfolioState) or per-day scalars (DayInputs).

    Attributes:
        base_assets: Asset tickers that are NOT LEAPS carve-outs.
        leaps_keys: Asset keys ending in _LEAPS suffix.
        leaps_fraction: Sum of target weights for leaps_keys; fraction of NAV
            allocated to LEAPS.
        base_target_w: Normalized weights over base_assets only (sums to 1.0).
        governed_base: Subset of base_assets governed by GTT signal (VTI and
            any GTT_EQUITY_TICKERS).
        gtt_active: True when gtt_signal is provided and portfolio holds a
            governed ticker.
        defensive_weights: Weights for the defensive sleeve; may include R_f
            sentinel key.
        use_leaps: True when any leaps_keys are present in target_weights.
        iv: IV floor: config.leaps_config.iv or DEFAULT_IV.
        leaps_monthly: Monthly contribution fraction allocated to LEAPS.
        base_contribution: Monthly contribution fraction allocated to base
            holdings.
        config: Full portfolio config; passed through for run_leaps_simulation
            calls.
        return_data: Return data; passed through for risk_free_rate in
            run_leaps_simulation.
        underlying_prices: LEAPS underlying spot prices aligned to backtest
            index.
        raw_vix: Raw VIX series aligned to backtest index (unsmoothed).
        mtm_iv_series: 30-day rolling mean of raw_vix; NaN for first 29 days.
        rfr_series: Risk-free rate series aligned to backtest index.
        mask_aligned: 0/1 GTT position mask aligned to backtest index.
        def_gross: Blended daily defensive sleeve return series.
        rebal_dates: O(1)-lookup set of scheduled quarterly rebalance dates.
        month_end_dates: O(1)-lookup set of last trading days of each calendar
            month.
        long_window_end: Maps each Long-window start date to its end date;
            used to slice prices for re-entry LEAPS simulations.
        w: Full target weight Series (including LEAPS keys); used for drift
            check and weight_row assembly.
    """

    base_assets: tuple[str, ...]
    leaps_keys: tuple[str, ...]
    leaps_fraction: float
    base_target_w: pd.Series
    governed_base: tuple[str, ...]
    gtt_active: bool
    defensive_weights: dict[str, float]
    use_leaps: bool
    iv: float
    leaps_monthly: float
    base_contribution: float
    config: PortfolioConfig
    return_data: ReturnData
    underlying_prices: pd.Series | None
    raw_vix: pd.Series | None
    mtm_iv_series: pd.Series | None
    rfr_series: pd.Series | None
    mask_aligned: pd.Series | None
    def_gross: pd.Series | None
    rebal_dates: frozenset[pd.Timestamp] = field(default_factory=frozenset)
    month_end_dates: frozenset[pd.Timestamp] = field(default_factory=frozenset)
    long_window_end: dict[pd.Timestamp, pd.Timestamp] = field(default_factory=dict)
    w: pd.Series = field(default_factory=pd.Series)


# ---------------------------------------------------------------------------
# Step function F-012
# ---------------------------------------------------------------------------


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

    This is a pure step function: it reads state, inputs, and ctx and returns
    a new PortfolioState without side effects.

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

        Accounting invariant (Defensive month-end, gtt_active):
        (new.defensive_sleeve - old.defensive_sleeve) + governed allocation to holdings
        + (new.leaps_pool - old.leaps_pool) == ctx.base_contribution + ctx.leaps_monthly
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
