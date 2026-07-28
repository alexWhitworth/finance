"""Portfolio backtest engine — rebalancing, contribution, and NAV loop.

All business logic is pure. Receives ReturnData + PriceData + PortfolioConfig
and produces BacktestResult consumed by metrics.py.
"""

from dataclasses import dataclass, field, replace

import pandas as pd

from finance.consts import (
    DEFAULT_IV,
    DRIFT_BAND_RELATIVE,
    GTT_DEFENSIVE_WEIGHTS_DEFAULT,
    GTT_EQUITY_TICKERS,
    GTT_SMA_WINDOW,
    GTT_UNRATE_TRADE_LAG_DAYS,
    GTT_VIX_CONSECUTIVE_DAYS,
    LEAPS_KEY_SUFFIX,
    VIX_MTM_WINDOW,
)
from finance.data import PriceData
from finance.gtt import GttSignalData
from finance.leverage import (
    LeapsConfig,
    LeapsContract,
    LeapsLedger,
    LeapsPartialCloseEvent,
    RebalanceRule,
    WeightStrategy,
    _live_contracts,
    price_leaps_contract,
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


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def get_rebalance_dates(
    index: pd.DatetimeIndex,
    rule: RebalanceRule,
) -> list[pd.Timestamp]:
    """Return all rebalancing dates within index for the given rule.

    QUARTERLY: last trading day of each quarter-end month (Mar / Jun / Sep / Dec).

    Arguments:
        index: DatetimeIndex of trading days in the backtest window.
        rule: RebalanceRule controlling the rebalancing schedule.

    Returns:
        Sorted list of Timestamp rebalance dates, all within index.
    """
    if rule == RebalanceRule.QUARTERLY:
        quarter_end_months = {3, 6, 9, 12}
        dates: list[pd.Timestamp] = []
        for period, grp in pd.Series(index, index=index).groupby(index.to_period("M")):
            if period.month in quarter_end_months:
                dates.append(pd.Timestamp(grp.iloc[-1]))
        return sorted(dates)
    # Unreachable with current enum, but guards future extensions
    return []  # pragma: no cover


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


def should_rebalance(
    current_weights: pd.Series,
    target_weights: pd.Series,
    rule: RebalanceRule,
    band: float = DRIFT_BAND_RELATIVE,
) -> bool:
    """Return True if rebalancing should be triggered under the given rule.

    QUARTERLY: always returns False — schedule is handled by get_rebalance_dates().
    DRIFT: returns True if any asset's relative weight deviation exceeds band.

    Relative deviation for asset i: |w_i - t_i| / t_i > band.
    Only assets present in both current_weights and target_weights are checked.
    Assets with a target weight of zero are skipped (division by zero guard).

    Arguments:
        current_weights: Realized portfolio weights at the check date.
        target_weights: Target weights from PortfolioConfig (need not be normalized).
        rule: RebalanceRule controlling the check logic.
        band: Relative drift threshold. Default DRIFT_BAND_RELATIVE (0.10 = ±10%).

    Returns:
        True if rebalancing is triggered, False otherwise.
    """
    if rule == RebalanceRule.QUARTERLY:
        return False
    common = current_weights.index.intersection(target_weights.index)
    for a in common:
        t = float(target_weights[a])
        if t == 0.0:
            continue
        if abs(float(current_weights[a]) - t) / t > band:
            return True
    return False


def apply_contribution(
    nav: float,
    contribution: float,
    weights: pd.Series,
) -> dict[str, float]:
    """Allocate a dollar contribution across assets proportional to weights.

    Arguments:
        nav: Current portfolio NAV (unused in USER_SPECIFIED mode; present for
            future risk-parity strategies that need the NAV context).
        contribution: Dollar amount to allocate.
        weights: Unit-normed asset weights (must sum to 1.0).

    Returns:
        Mapping of asset → dollar amount allocated from the contribution.
    """
    _ = nav  # reserved for future weight strategies that use NAV context
    return {str(a): contribution * float(weights[a]) for a in weights.index}


# ---------------------------------------------------------------------------
# GTT pre-compute helpers (pure; used by the run_backtest GTT branch)
# ---------------------------------------------------------------------------


def _gtt_governed_keys(target_weights: dict[str, float]) -> set[str]:
    """Return the target_weights keys governed by the GTT signal.

    A key is governed when it is a GTT_EQUITY_TICKERS ticker present in
    target_weights, or a "<ticker>_LEAPS" carve-out of such a ticker. When
    target_weights holds no governed ticker the result is empty and the GTT
    overlay is a no-op.

    Arguments:
        target_weights: Portfolio target-weight mapping (asset -> weight).

    Returns:
        Set of governed keys (subset of target_weights). Empty if none.
    """
    governed: set[str] = set()
    for key in target_weights:
        base = key.removesuffix(LEAPS_KEY_SUFFIX) if key.endswith(LEAPS_KEY_SUFFIX) else key
        if base in GTT_EQUITY_TICKERS:
            governed.add(key)
    return governed


def _reindex_position_mask(mask: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Align a position mask to the backtest index, defaulting gaps to Long.

    Missing dates (holiday misalignment) are forward-filled from the last known
    signal; leading dates before any signal exists default to 1 (Long), so the
    overlay never forces a defensive posture on unknown data.

    Arguments:
        mask: 0/1 position mask (1=Long, 0=Defensive), DatetimeIndex.
        index: Target backtest trading-day index.

    Returns:
        Int Series aligned to index, values in {0, 1}, no NaN.
    """
    return mask.reindex(index, method="ffill").fillna(1).astype(int)


def _long_windows(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the (start, end) date bounds of each contiguous Long (mask==1) run.

    Scans the aligned position mask chronologically and groups maximal runs of
    Long days. Defensive (mask==0) runs are skipped. Each returned tuple gives the
    first and last date (both inclusive) of one Long window, so a caller can slice
    a price series to that window for a per-window LEAPS simulation.

    Arguments:
        mask: Int position mask (1=Long, 0=Defensive) with a DatetimeIndex.

    Returns:
        List of (start_date, end_date) inclusive Long-window bounds, in
        chronological order. Empty if the mask never equals 1.
    """
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    dates = mask.index
    regimes = mask.to_numpy()
    start_i: int | None = None
    for i in range(len(regimes)):
        if regimes[i] == 1 and start_i is None:
            start_i = i
        elif regimes[i] == 0 and start_i is not None:
            windows.append((pd.Timestamp(dates[start_i]), pd.Timestamp(dates[i - 1])))
            start_i = None
    if start_i is not None:
        windows.append((pd.Timestamp(dates[start_i]), pd.Timestamp(dates[-1])))
    return windows


def _defensive_gross_return(
    returns: pd.DataFrame,
    rfr_series: pd.Series,
    defensive_weights: dict[str, float],
) -> pd.Series:
    """Compute the daily blended gross return of the defensive sleeve.

    The sleeve return on day t is Sum_i w_i * r_i(t), where the sentinel key
    "R_f" contributes w_Rf * rfr(t)/252 (a date-varying T-bill day return) and
    every other key contributes its own asset return. This is the single factor
    by which the parked defensive capital (and the LEAPS pool) compounds.

    Arguments:
        returns: Daily simple returns (DatetimeIndex x asset columns).
        rfr_series: Daily annualized risk-free rate (decimal), aligned to returns.
        defensive_weights: Sleeve weights summing to 1.0; may contain "R_f".

    Returns:
        Daily float Series of blended sleeve returns, indexed like returns.
    """
    blended = pd.Series(0.0, index=returns.index, name="defensive_gross_return")
    rfr_aligned = rfr_series.reindex(returns.index, method="ffill").fillna(0.0)
    for key, weight in defensive_weights.items():
        if key == GTT_RISK_FREE_KEY:
            blended = blended + weight * (rfr_aligned / 252.0)
        else:
            blended = blended + weight * returns[key]
    return blended


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
      - Dynamic IV: when price_data.vol_prices has a "^VIX" column, raw VIX drives
        contract creation and rolls, while a VIX_MTM_WINDOW-day rolling mean drives
        daily mark-to-market. config.leaps_config.iv is the floor throughout.
        Absent "^VIX", config.leaps_config.iv is used everywhere.
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
    from finance.leverage import run_leaps_simulation

    # GTT opt-in requires a matched (gtt_signal, config.gtt_config) pair.
    if (gtt_signal is None) != (config.gtt_config is None):
        raise ValueError(
            "gtt_signal and config.gtt_config must both be set or both be None; got "
            f"gtt_signal={'set' if gtt_signal is not None else 'None'}, "
            f"config.gtt_config={'set' if config.gtt_config is not None else 'None'}"
        )

    returns = return_data.returns

    if gtt_signal is not None:
        assert config.gtt_config is not None  # guaranteed by the paired check above
        missing_def = [
            k
            for k in config.gtt_config.defensive_weights
            if k != GTT_RISK_FREE_KEY and k not in returns.columns
        ]
        if missing_def:
            raise ValueError(
                f"defensive_weights tickers absent from return_data: {missing_def}"
            )

    # Split target weights into base assets and carved-out LEAPS keys (Model B)
    leaps_keys = [k for k in config.target_weights if k.endswith(LEAPS_KEY_SUFFIX)]
    base_assets = [k for k in config.target_weights if k not in leaps_keys]

    missing = [a for a in base_assets if a not in returns.columns]
    if missing:
        raise ValueError(f"Assets missing from return_data: {missing}")

    use_leaps = len(leaps_keys) > 0
    if use_leaps and config.leaps_config is None:
        raise ValueError("LEAPS keys present in target_weights but leaps_config is None")

    w = pd.Series(config.target_weights)
    leaps_fraction = float(w[leaps_keys].sum()) if leaps_keys else 0.0

    # Base-only target weights, renormalized among base assets (sum to 1.0)
    base_target_w = w[base_assets]
    if len(base_assets) > 0 and base_target_w.sum() > 0:
        base_target_w = base_target_w / base_target_w.sum()

    idx = pd.DatetimeIndex(returns.index)

    # Rebalance date set (O(1) lookups)
    rebal_dates: set[pd.Timestamp] = set(get_rebalance_dates(idx, config.rebalance_rule))

    # Month-end date set: last trading day of each calendar month
    month_end_dates: set[pd.Timestamp] = {
        pd.Timestamp(grp.index[-1])
        for _, grp in returns.groupby(idx.to_period("M"))
    }

    # LEAPS setup — carve capital out of NAV and run a fresh simulation (Model B)
    leaps_ledger: LeapsLedger | None = None
    underlying_prices: pd.Series | None = None
    iv = DEFAULT_IV
    rfr_series: pd.Series | None = None
    mtm_iv_series: pd.Series | None = None  # 30-day smoothed VIX for daily MTM

    if use_leaps:
        underlyings = {k.removesuffix(LEAPS_KEY_SUFFIX) for k in leaps_keys}
        if len(underlyings) > 1:
            raise ValueError(f"Only one LEAPS underlying is supported; got {sorted(underlyings)}")
        underlying = next(iter(underlyings))
        if underlying not in price_data.prices.columns:
            raise ValueError(f"LEAPS underlying '{underlying}' absent from price_data.prices")

        assert config.leaps_config is not None  # guarded above
        iv = config.leaps_config.iv
        underlying_prices = price_data.prices[underlying].reindex(idx, method="ffill")
        rfr_series = return_data.risk_free_rate.reindex(idx, method="ffill").fillna(0.0)

        # VIX-based dynamic IV: raw VIX for contract creation/roll (via iv_series),
        # 30-day rolling mean for daily MTM. config.iv is the floor throughout.
        raw_vix: pd.Series | None = None
        if not price_data.vol_prices.empty and "^VIX" in price_data.vol_prices.columns:
            raw_vix = price_data.vol_prices["^VIX"].reindex(idx, method="ffill")
            mtm_iv_series = raw_vix.rolling(VIX_MTM_WINDOW).mean().ffill()

        initial_leaps_capital = config.initial_nav * leaps_fraction
        leaps_monthly = config.monthly_contribution * leaps_fraction
        leaps_ledger = run_leaps_simulation(
            underlying_prices,
            leaps_monthly,
            config.leaps_config,
            risk_free_series=return_data.risk_free_rate,
            iv_series=raw_vix,
            initial_capital=initial_leaps_capital,
        )

    # Initialize base holdings: dollar value per base asset (LEAPS capital carved out)
    base_nav_init = config.initial_nav * (1.0 - leaps_fraction)
    holdings: dict[str, float] = {
        a: base_nav_init * float(base_target_w[a]) for a in base_assets
    }
    prev_total_nav = config.initial_nav

    # Base share of the monthly contribution (LEAPS share is handled inside the ledger)
    base_contribution = config.monthly_contribution * (1.0 - leaps_fraction)

    # Drift-rebalance state: cumulative surviving fraction per base contract.
    # Daily LEAPS MTM is scaled by this map so partial closes take effect
    # without rebuilding the frozen ledger inside the loop (frozen once at return).
    leaps_scale: dict[LeapsContract, float] = {}
    partial_close_list: list[LeapsPartialCloseEvent] = []
    target_leaps_value = config.initial_nav * leaps_fraction  # dollar target for LEAPS sleeve

    # GTT overlay state. When active, the governed (VTI / VTI_LEAPS) leg is moved
    # into a single defensive-sleeve scalar on defensive days and re-anchored to
    # target on re-entry. The sleeve compounds by the blended defensive return and
    # is decomposed at defensive_weights proportions for weight_history.
    # The overlay only engages when the signal is present AND the portfolio holds a
    # governed ticker; otherwise GTT is a complete no-op (no forced rebalance fires).
    defensive_sleeve = 0.0
    prev_regime = 1
    governed_base: list[str] = []
    defensive_weights: dict[str, float] = {}
    mask_aligned: pd.Series | None = None
    def_gross: pd.Series | None = None
    gtt_active = False
    if gtt_signal is not None:
        assert config.gtt_config is not None  # paired above
        governed = _gtt_governed_keys(config.target_weights)
        governed_base = [k for k in governed if k in base_assets]
        gtt_active = len(governed) > 0
    if gtt_active:
        assert gtt_signal is not None and config.gtt_config is not None
        if use_leaps:
            # LEAPS-under-GTT segmentation lands in F-10d; guard until then.
            raise NotImplementedError(
                "GTT overlay with a LEAPS carve-out is implemented in F-10d"
            )
        defensive_weights = config.gtt_config.defensive_weights
        mask_aligned = _reindex_position_mask(gtt_signal.position_mask, idx)
        def_gross = _defensive_gross_return(
            returns, return_data.risk_free_rate, defensive_weights
        )

    nav_values: list[float] = []
    return_values: list[float] = []
    weight_rows: list[dict[str, float]] = []

    for date in returns.index:
        date_ts = pd.Timestamp(date)
        day_ret = returns.loc[date_ts]

        # (GTT open) The governed leg is swept into the sleeve at the open of a
        # defensive day (mask is lag-adjusted: mask[t]=0 means hold defensive
        # *during* day t), so the freed VTI capital rides the defensive return
        # rather than VTI's. Re-entry to target happens at the close (below), so
        # weight_history on the re-entry day lands exactly on target_weights.
        regime_t = 1
        if gtt_active and mask_aligned is not None:
            regime_t = int(mask_aligned.loc[date_ts])
            if regime_t == 0:
                for k in governed_base:
                    defensive_sleeve += holdings[k]
                    holdings[k] = 0.0

        # (a) Apply daily returns to base holdings
        for a in base_assets:
            holdings[a] *= 1.0 + float(day_ret[a])  # type: ignore[arg-type]

        # (a') The defensive sleeve rides the blended defensive return while it holds
        # defensively-allocated capital: every defensive day (regime_t == 0) and the
        # re-entry day (prev_regime == 0), where it earns one final defensive day
        # before being redeployed to target at the close.
        if gtt_active and def_gross is not None and (regime_t == 0 or prev_regime == 0):
            defensive_sleeve *= 1.0 + float(def_gross.loc[date_ts])

        # (e) LEAPS gross mark-to-market (carved-out capital value), scaled by prior closes
        leaps_value = 0.0
        if leaps_ledger is not None and underlying_prices is not None:
            spot = float(underlying_prices.loc[date_ts])
            rfr = float(rfr_series.loc[date_ts]) if rfr_series is not None else 0.0
            day_iv = iv
            if mtm_iv_series is not None:
                smoothed = mtm_iv_series.loc[date_ts]
                if pd.notna(smoothed):
                    day_iv = max(float(smoothed), iv)
            live = _live_contracts(leaps_ledger, date_ts)
            leaps_value = sum(
                price_leaps_contract(c, spot, date_ts, day_iv, rfr) * leaps_scale.get(c, 1.0)
                for c in live
            )

        nav_before_contrib = sum(holdings.values()) + leaps_value + defensive_sleeve

        # (b) Market return — excludes the upcoming contribution
        port_return = nav_before_contrib / prev_total_nav - 1.0

        # (c) Month-end: add the base share of the contribution across base assets.
        # On a defensive day the governed-ticker share of that contribution is
        # diverted into the sleeve (base_target_w still sums to 1 over base_assets,
        # so the governed weight is its base_target_w mass).
        if date_ts in month_end_dates and base_assets:
            alloc = apply_contribution(nav_before_contrib, base_contribution, base_target_w)
            for a in base_assets:
                if gtt_active and regime_t == 0 and a in governed_base:
                    defensive_sleeve += alloc[a]
                else:
                    holdings[a] += alloc[a]

        # (d) QUARTERLY rebalance: realign base holdings to base target weights.
        # Runs first (Option C); the GTT defensive override below re-zeros the
        # governed leg into the sleeve after the rebalance repopulates it.
        if date_ts in rebal_dates and base_assets:
            base_nav = sum(holdings.values())
            for a in base_assets:
                holdings[a] = base_nav * float(base_target_w[a])
            if gtt_active and regime_t == 0:
                for k in governed_base:
                    defensive_sleeve += holdings[k]
                    holdings[k] = 0.0

        # (f) DRIFT rebalance: check monthly; trim LEAPS overshoot pro-rata (tax-free)
        if config.rebalance_rule == RebalanceRule.DRIFT and date_ts in month_end_dates:
            base_val = sum(holdings.values())
            total_val = base_val + leaps_value
            weights_now = {a: holdings[a] / total_val for a in base_assets}
            for k in leaps_keys:
                share = float(w[k]) / leaps_fraction if leaps_fraction > 0 else 0.0
                weights_now[k] = leaps_value * share / total_val
            current_weights = pd.Series(weights_now)
            if should_rebalance(current_weights, w, RebalanceRule.DRIFT):
                # Realign base assets to their targets within the base sleeve.
                for a in base_assets:
                    holdings[a] = base_val * float(base_target_w[a])
                # Trim LEAPS overshoot back to the target dollar sleeve, pro-rata.
                if leaps_value > target_leaps_value and leaps_value > 0:
                    close_scale = target_leaps_value / leaps_value
                    net_proceeds = leaps_value - target_leaps_value
                    for c in _live_contracts(leaps_ledger, date_ts):  # type: ignore[arg-type]
                        leaps_scale[c] = leaps_scale.get(c, 1.0) * close_scale
                    # Return proceeds to base holdings by base target weights (tax-free).
                    if base_assets:
                        for a in base_assets:
                            holdings[a] += net_proceeds * float(base_target_w[a])
                    leaps_value = target_leaps_value

        # (GTT close) Defensive -> Long re-entry: re-anchor the whole portfolio to
        # target on the combined NAV (base holdings + sleeve). This is the last
        # holdings mutation of the day, so weight_history lands exactly on target.
        if gtt_active and prev_regime == 0 and regime_t == 1 and base_assets:
            total = sum(holdings.values()) + defensive_sleeve
            for a in base_assets:
                holdings[a] = total * float(base_target_w[a])
            defensive_sleeve = 0.0

        total_nav = sum(holdings.values()) + leaps_value + defensive_sleeve

        nav_values.append(total_nav)
        return_values.append(port_return)
        row = {a: holdings[a] / total_nav for a in base_assets}
        for k in leaps_keys:
            share = float(w[k]) / leaps_fraction if leaps_fraction > 0 else 0.0
            row[k] = leaps_value * share / total_nav
        # Decompose the defensive sleeve across defensive_weights for weight_history;
        # governed keys were zeroed at the open so their rows are already 0.
        if gtt_active and defensive_sleeve > 0.0:
            for dk, dw in defensive_weights.items():
                row[dk] = row.get(dk, 0.0) + dw * defensive_sleeve / total_nav
        weight_rows.append(row)

        prev_total_nav = total_nav
        prev_regime = regime_t

    # Freeze accumulated partial closes onto the ledger once, at the return boundary.
    # Every entry in leaps_scale is a surviving fraction < 1.0 (only closes write it).
    if leaps_ledger is not None and leaps_scale:
        final_date = pd.Timestamp(returns.index[-1])
        for c, surviving in leaps_scale.items():
            continuation = replace(c, n_contracts=c.n_contracts * surviving)
            partial_close_list.append(
                LeapsPartialCloseEvent(
                    close_date=final_date,
                    original_contract=c,
                    continuation_contract=continuation,
                    n_contracts_closed=c.n_contracts * (1.0 - surviving),
                    net_proceeds=0.0,  # per-contract proceeds already booked to base at close time
                )
            )
        leaps_ledger = replace(leaps_ledger, partial_close_events=tuple(partial_close_list))

    nav_series = pd.Series(nav_values, index=returns.index, name="NAV")
    return_series = pd.Series(return_values, index=returns.index, name="portfolio_return")
    weight_history = pd.DataFrame(weight_rows, index=returns.index)
    if gtt_active:
        # A synthetic R_f column (and any defensive-only ticker) appears only on
        # defensive days; other days are absent -> NaN. Zero-fill for a dense frame.
        weight_history = weight_history.fillna(0.0)

    return BacktestResult(
        nav_series=nav_series,
        weight_history=weight_history,
        return_series=return_series,
        leaps_ledger=leaps_ledger,
        config=config,
    )
