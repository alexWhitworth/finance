"""Portfolio backtest engine — rebalancing, contribution, and NAV loop.

All business logic is pure. Receives ReturnData + PriceData + PortfolioConfig
and produces BacktestResult consumed by metrics.py.
"""

from dataclasses import dataclass, field

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
)

from finance.consts import (
    GTT_DEFENSIVE_WEIGHTS_DEFAULT,
    GTT_SMA_WINDOW,
    GTT_UNRATE_TRADE_LAG_DAYS,
    GTT_VIX_CONSECUTIVE_DAYS,
)
from finance.data import PriceData
from finance.gtt import GttSignalData
from finance.leverage import (
    LeapsConfig,
    LeapsContract,
    LeapsGttCloseEvent,
    LeapsLedger,
    RebalanceRule,
    WeightStrategy,
)
from finance.returns import ReturnData

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


# Sentinel key in defensive_weights meaning T-bill cash (earns risk_free_rate/252
# per day). Exempt from the target_weights membership check.
GTT_RISK_FREE_KEY: str = "R_f"


@dataclass(frozen=True)
class GttConfig:
    """Configuration for the GTT (Growth Trend Timing) market-timing overlay.

    Opt-in via PortfolioConfig.gtt_config. Governs the GTT_EQUITY_TICKERS leg
    (currently VTI and its _LEAPS variant), moving it into a fixed-weight
    defensive sleeve when recession risk is detected and the price trend confirms.
    Extensible for future per-ticker signals (e.g. VXUS).

    Attributes:
        vix_p90_threshold: Fixed VIX P90 threshold as a decimal (e.g. 0.272 == 27.2%).
            Caller computes from desired history to avoid look-ahead; the library
            applies no look-ahead protection.
        sma_window: Rolling window (trading days) for the equity price SMA trend
            filter. Default GTT_SMA_WINDOW (200).
        vix_consecutive_days: N consecutive days VIX >= threshold required to fire
            VIX_5D. Default GTT_VIX_CONSECUTIVE_DAYS (5).
        unrate_trade_lag_days: Trading-day execution lag from the UNRATE publication
            date to the trade. The ~1-month reference→publication lag is handled
            inside compute_ue_signal, NOT by this field. Default
            GTT_UNRATE_TRADE_LAG_DAYS (1).
        defensive_weights: Weights the defensive sleeve holds when defensive. Must
            sum to 1.0 (abs tol 1e-6). Sentinel key "R_f" means T-bill cash whose
            daily gross return is risk_free_rate[t]/252 (a date-varying Series, not
            a scalar). Non-R_f keys must exist in target_weights. Default
            GTT_DEFENSIVE_WEIGHTS_DEFAULT.

    Raises:
        ValueError: If defensive_weights does not sum to 1.0 within 1e-6.
    """

    vix_p90_threshold: float
    sma_window: int = GTT_SMA_WINDOW
    vix_consecutive_days: int = GTT_VIX_CONSECUTIVE_DAYS
    unrate_trade_lag_days: int = GTT_UNRATE_TRADE_LAG_DAYS
    defensive_weights: dict[str, float] = field(
        default_factory=lambda: dict(GTT_DEFENSIVE_WEIGHTS_DEFAULT)
    )

    def __post_init__(self) -> None:
        total = sum(self.defensive_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"defensive_weights must sum to 1.0; got {total:.6f}"
            )


@dataclass(frozen=True)
class PortfolioConfig:
    """Specification for a single backtest run.

    Attributes:
        target_weights: Mapping of asset → notional weight. Must sum to 1.0.
        initial_nav: Starting portfolio value in dollars.
        monthly_contribution: Dollar amount added at each month-end.
        rebalance_rule: When to rebalance (QUARTERLY, etc.).
        weight_strategy: How target weights are determined.
        leaps_config: Optional LEAPS overlay configuration.
        gtt_config: Optional GTT market-timing overlay. None = GTT disabled
            (existing behavior unchanged). When set, non-R_f keys in
            defensive_weights must exist in target_weights.

    Raises:
        ValueError: If target_weights does not sum to 1.0 within 1e-6.
        ValueError: If gtt_config is set and a non-R_f defensive_weights key is
            absent from target_weights.
    """

    target_weights: dict[str, float]
    initial_nav: float
    monthly_contribution: float
    rebalance_rule: RebalanceRule
    weight_strategy: WeightStrategy
    leaps_config: LeapsConfig | None = None
    gtt_config: GttConfig | None = None

    def __post_init__(self) -> None:
        total = sum(self.target_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"target_weights must sum to 1.0; got {total:.6f}"
            )
        if self.gtt_config is not None:
            missing = [
                k
                for k in self.gtt_config.defensive_weights
                if k != GTT_RISK_FREE_KEY and k not in self.target_weights
            ]
            if missing:
                raise ValueError(
                    f"defensive_weights keys absent from target_weights: {missing}"
                )


@dataclass(frozen=True)
class BacktestContext:
    """Immutable per-backtest configuration and precomputed series.

    Constructed once before the main loop. Contains no per-day state
    (PortfolioState) or per-day scalars (DayInputs).

    Attributes:
        base_assets: Asset tickers that are NOT LEAPS carve-outs.
        leaps_keys: Asset keys ending in _LEAPS suffix.
        leaps_fraction: Sum of target weights for leaps_keys; fraction of NAV
            allocated to LEAPS.
        base_target_w: Normalized weights over base_assets only (sums to 1.0).
        governed_base: Subset of base_assets governed by GTT signal.
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
        rebal_dates: O(1)-lookup frozenset of scheduled quarterly rebalance
            dates.
        month_end_dates: O(1)-lookup frozenset of last trading days of each
            calendar month.
        long_window_end: Maps each Long-window start date to its end date; used
            to slice prices for re-entry LEAPS simulations.
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
    rebal_dates: frozenset[pd.Timestamp]
    month_end_dates: frozenset[pd.Timestamp]
    long_window_end: dict[pd.Timestamp, pd.Timestamp]
    w: pd.Series


@dataclass(frozen=True)
class BacktestResult:
    """Output of a completed backtest simulation.

    Attributes:
        nav_series: DatetimeIndex → end-of-day portfolio NAV (includes contributions).
        weight_history: DatetimeIndex x asset, end-of-day realized weights.
        return_series: DatetimeIndex, daily market return (excludes contributions).
        leaps_ledger: Full LEAPS history, or None if no LEAPS overlay was used.
        config: PortfolioConfig that produced this result.
    """

    nav_series: pd.Series
    weight_history: pd.DataFrame
    return_series: pd.Series
    leaps_ledger: LeapsLedger | None
    config: PortfolioConfig


@dataclass(frozen=True)
class PortfolioState:
    """Immutable snapshot of the portfolio at a single point in time.

    Used by the GTT backtest loop to carry all mutable state forward without
    mutation. Each loop iteration produces a new PortfolioState via
    ``dataclasses.replace``.

    Attributes:
        holdings: Dollar value per base asset (ticker → value).
        defensive_sleeve: Governed equity capital swept in during GTT defensive windows.
        leaps_pool: Force-closed LEAPS net proceeds parked during defensive windows.
        leaps_value: Current LEAPS mark-to-market value.
        prev_total_nav: End-of-day NAV from t-1.
        prev_regime: GTT regime on t-1 (1 = Long, 0 = Defensive).
        prev_date_ts: Trading date of t-1, or None on the first iteration.
        leaps_ledger: Active per-window LEAPS simulation ledger, or None.
        leaps_scale: Surviving fraction per contract in (0, 1].
        all_window_ledgers: Immutable accumulator of per-Long-window ledgers.
        all_gtt_closes: Immutable accumulator of GTT force-close events.
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
    """Per-day read-only inputs extracted from precomputed series before the step pipeline runs.

    No field contains data with timestamp > date_ts (temporal invariant T1). Enforced by
    construction in _extract_day_inputs.

    Attributes:
        date_ts: Current trading day.
        day_ret: Asset returns for this day, indexed by ticker.
        regime_t: GTT regime for today: 0=Defensive, 1=Long (from mask_aligned).
        def_gross_return: Blended defensive sleeve return for today (0.0 if GTT inactive).
        spot: Underlying LEAPS spot price at date_ts (None if no LEAPS).
        raw_vix_value: Raw VIX at date_ts (None if no vol_prices). Used as creation IV on
            re-entry.
        mtm_iv_value: 30-day rolling mean VIX at date_ts (None or NaN during 29-day warmup).
            Used for daily MTM.
        rfr: Risk-free rate at date_ts.
        is_month_end: True if date_ts is the last trading day of a calendar month.
        is_rebal_date: True if date_ts is a scheduled quarterly rebalance date.

    Notes:
        frozen=True prevents field reassignment but does not prevent in-place mutation of the
        day_ret Series. By convention, day_ret must never be mutated after construction.
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
# Pure helper functions
# ---------------------------------------------------------------------------

def compute_target_weights(
    config: PortfolioConfig,
    current_weights: pd.Series,
    current_nav: float,
    current_date: pd.Timestamp,
) -> pd.Series:
    """Return target portfolio weights for the current rebalance.

    USER_SPECIFIED: returns config.target_weights normalized to sum = 1.

    Arguments:
        config: PortfolioConfig containing the weight strategy and targets.
        current_weights: Realized weights just before rebalancing.
        current_nav: Portfolio NAV just before rebalancing.
        current_date: Rebalance execution date.

    Returns:
        Normalized target weight Series (sums to 1.0).
    """
    if config.weight_strategy == WeightStrategy.USER_SPECIFIED:
        raw = pd.Series(config.target_weights)
        return raw / raw.sum()
    raw = pd.Series(config.target_weights)  # pragma: no cover
    return raw / raw.sum()  # pragma: no cover

# ---------------------------------------------------------------------------
# Backtest orchestrator
# ---------------------------------------------------------------------------


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
