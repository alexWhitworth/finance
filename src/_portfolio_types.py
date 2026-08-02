
from dataclasses import dataclass, field
import pandas as pd

from finance.consts import (
    GTT_DEFENSIVE_WEIGHTS_DEFAULT,
    GTT_RISK_FREE_KEY,
    GTT_SMA_WINDOW,
    GTT_UNRATE_TRADE_LAG_DAYS,
    GTT_VIX_CONSECUTIVE_DAYS,
)

from finance.leverage import (
    LeapsConfig,
    LeapsContract,
    LeapsGttCloseEvent,
    LeapsLedger,
    RebalanceRule,
    WeightStrategy,
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
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
