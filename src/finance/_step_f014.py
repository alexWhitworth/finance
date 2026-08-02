"""F-014: _apply_gtt_reentry step function (Bug 2 fix).

Extracts the Defensive->Long re-entry rebalance from run_backtest (lines 891-929).
Bug 2 fix: LEAPS contracts created on re-entry are priced at creation IV
(raw VIX floored at ctx.iv), NOT the smoothed MTM IV.

Invariant A2: sum(new_holdings.values()) + new_leaps_value == total within 1e-9
Invariant A4: new_leaps_value == total * ctx.leaps_fraction within 1e-6 (creation IV)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pandas as pd

from finance.leverage import (
    LeapsContract,
    LeapsGttCloseEvent,
    LeapsLedger,
    _live_contracts,
    price_leaps_contract,
    run_leaps_simulation,
)
from finance.returns import ReturnData

if TYPE_CHECKING:
    from finance.portfolio import PortfolioConfig


# ---------------------------------------------------------------------------
# Data schemas (DS-001, DS-002, DS-003)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioState:
    """Immutable snapshot of all mutable per-day loop state.

    Every mutation produces a new instance via dataclasses.replace().
    Accumulators (all_window_ledgers, all_gtt_closes) are tuples to ensure
    pure step functions.

    Attributes:
        holdings: Dollar value per base asset.
        defensive_sleeve: Governed equity capital swept in during GTT defensive windows.
        leaps_pool: Force-closed LEAPS net proceeds parked during defensive windows.
        leaps_value: Current LEAPS mark-to-market.
        prev_total_nav: End-of-day NAV from t-1.
        prev_regime: GTT regime on t-1: 1=Long, 0=Defensive.
        prev_date_ts: Trading date of t-1; used for force-close spot/IV lookup.
        leaps_ledger: Active per-window LEAPS simulation ledger.
        leaps_scale: Surviving fraction per contract after drift partial-closes.
        all_window_ledgers: One ledger per Long window for final assembly.
        all_gtt_closes: All GTT force-close events across all defensive transitions.
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
        regime_t: GTT regime for today: 0=Defensive, 1=Long.
        def_gross_return: Blended defensive sleeve return for today.
        spot: Underlying LEAPS spot price at date_ts (None if no LEAPS).
        raw_vix_value: Raw VIX at date_ts (None if no vol_prices). Used as creation IV on re-entry.
        mtm_iv_value: 30-day rolling mean VIX at date_ts. Used for daily MTM.
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


@dataclass(frozen=True)
class BacktestContext:
    """Immutable per-backtest configuration and precomputed series.

    Constructed once before the loop. Contains no per-day state or scalars.

    Attributes:
        base_assets: Asset tickers that are NOT LEAPS carve-outs.
        leaps_keys: Asset keys ending in _LEAPS suffix.
        leaps_fraction: Fraction of NAV allocated to LEAPS.
        base_target_w: Normalized weights over base_assets only (sums to 1.0).
        governed_base: Subset of base_assets governed by GTT signal.
        gtt_active: True when gtt_signal is provided and portfolio holds a governed ticker.
        defensive_weights: Weights for the defensive sleeve.
        use_leaps: True when any leaps_keys are present in target_weights.
        iv: IV floor: config.leaps_config.iv or DEFAULT_IV.
        leaps_monthly: Monthly contribution fraction allocated to LEAPS.
        base_contribution: Monthly contribution fraction allocated to base holdings.
        config: Full portfolio config.
        return_data: Return data; passed through for risk_free_rate.
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
# F-014: _apply_gtt_reentry
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

    Invariant A2: sum(new_holdings.values()) + new_leaps_value == total within 1e-9.
    Invariant A4: new_leaps_value == total * ctx.leaps_fraction within 1e-6.

    Arguments:
        state: Current PortfolioState snapshot.
        inputs: Per-day read-only inputs for the current trading day.
        ctx: Immutable backtest configuration and precomputed series.

    Returns:
        Updated PortfolioState with re-entry rebalance applied, or state unchanged
        if fire condition is not met.
    """
    if not (ctx.gtt_active and state.prev_regime == 0 and inputs.regime_t == 1):
        return state

    # Step 1: total NAV
    total = sum(state.holdings.values()) + state.defensive_sleeve + state.leaps_pool

    # Step 2: redeploy base holdings
    base_total = total * (1.0 - ctx.leaps_fraction)
    new_holdings = {a: base_total * float(ctx.base_target_w[a]) for a in ctx.base_assets}

    # Step 3: zero sleeve and pool
    new_sleeve = 0.0
    new_pool = 0.0

    # Defaults if LEAPS not active
    new_leaps_value = 0.0
    new_ledger = state.leaps_ledger
    new_window_ledgers = state.all_window_ledgers

    # Step 4: fresh LEAPS simulation for new Long window
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
        # creation_iv = max(raw_vix_value, ctx.iv) when raw_vix available; else ctx.iv.
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

    # Step 5: reset leaps_scale (clears orphaned keys from old window — risk R7)
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
