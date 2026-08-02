"""F-009: _apply_returns and _apply_defensive_compounding step functions.

Pure step functions extracted from run_backtest. Each takes an immutable
PortfolioState plus read-only DayInputs and BacktestContext and returns a
new PortfolioState via dataclasses.replace(). No side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from finance.leverage import LeapsContract, LeapsGttCloseEvent, LeapsLedger
from finance.portfolio import PortfolioConfig
from finance.returns import ReturnData


# ---------------------------------------------------------------------------
# PortfolioState (DS-001)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioState:
    """Immutable snapshot of all mutable per-day loop state.

    Every mutation produces a new instance via dataclasses.replace(). Accumulators
    (all_window_ledgers, all_gtt_closes) are tuples to ensure pure step functions.

    Attributes:
        holdings: Dollar value per base asset; governed assets zeroed during
            defensive windows. Dict reference is frozen; step functions construct
            new dicts.
        defensive_sleeve: Governed equity capital swept in during GTT defensive windows.
        leaps_pool: Force-closed LEAPS net proceeds parked during defensive windows.
        leaps_value: Current LEAPS mark-to-market.
        prev_total_nav: End-of-day NAV from t-1; denominator of port_return calculation.
        prev_regime: GTT regime on t-1: 1=Long, 0=Defensive.
        prev_date_ts: Trading date of t-1; used for force-close spot/IV lookup.
        leaps_ledger: Active per-window LEAPS simulation ledger; replaced on GTT re-entry.
        leaps_scale: Surviving fraction per contract after drift partial-closes.
        all_window_ledgers: Immutable accumulator: one ledger per Long window.
        all_gtt_closes: Immutable accumulator: all GTT force-close events.
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


# ---------------------------------------------------------------------------
# DayInputs (DS-002)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DayInputs:
    """Per-day read-only inputs extracted from precomputed series.

    No field contains data with timestamp > date_ts (temporal invariant T1).

    Attributes:
        date_ts: Current trading day.
        day_ret: Asset returns for this day, indexed by ticker.
        regime_t: GTT regime for today: 0=Defensive, 1=Long.
        def_gross_return: Blended defensive sleeve return for today (0.0 if GTT inactive).
        spot: Underlying LEAPS spot price at date_ts (None if no LEAPS).
        raw_vix_value: Raw VIX at date_ts (None if no vol_prices).
        mtm_iv_value: 30-day rolling mean VIX at date_ts (None during 29-day warmup).
        rfr: Risk-free rate at date_ts.
        is_month_end: True if date_ts is the last trading day of a calendar month.
        is_rebal_date: True if date_ts is a scheduled quarterly rebalance date.
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


# ---------------------------------------------------------------------------
# BacktestContext (DS-003)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestContext:
    """Immutable per-backtest configuration and precomputed series.

    Constructed once before the loop. Contains no per-day state (PortfolioState)
    or per-day scalars (DayInputs).

    Attributes:
        base_assets: Asset tickers that are NOT LEAPS carve-outs.
        leaps_keys: Asset keys ending in _LEAPS suffix.
        leaps_fraction: Sum of target weights for leaps_keys.
        base_target_w: Normalized weights over base_assets only (sums to 1.0).
        governed_base: Subset of base_assets governed by GTT signal.
        gtt_active: True when gtt_signal is provided and portfolio holds a governed ticker.
        defensive_weights: Weights for the defensive sleeve; may include R_f sentinel key.
        use_leaps: True when any leaps_keys are present in target_weights.
        iv: IV floor: config.leaps_config.iv or DEFAULT_IV.
        leaps_monthly: Monthly contribution fraction allocated to LEAPS.
        base_contribution: Monthly contribution fraction allocated to base holdings.
        config: Full portfolio config; passed through for run_leaps_simulation calls.
        return_data: Return data; passed through for risk_free_rate in run_leaps_simulation.
        underlying_prices: LEAPS underlying spot prices aligned to backtest index.
        raw_vix: Raw VIX series aligned to backtest index (unsmoothed).
        mtm_iv_series: 30-day rolling mean of raw_vix; NaN for first 29 days.
        rfr_series: Risk-free rate series aligned to backtest index.
        mask_aligned: 0/1 GTT position mask aligned to backtest index.
        def_gross: Blended daily defensive sleeve return series.
        rebal_dates: O(1)-lookup set of scheduled quarterly rebalance dates.
        month_end_dates: O(1)-lookup set of last trading days of each calendar month.
        long_window_end: Maps each Long-window start date to its end date.
        w: Full target weight Series (including LEAPS keys).
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
    rebal_dates: frozenset[pd.Timestamp]
    month_end_dates: frozenset[pd.Timestamp]
    long_window_end: dict[pd.Timestamp, pd.Timestamp]
    w: pd.Series


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------


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
