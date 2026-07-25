"""Portfolio backtest engine — rebalancing, contribution, and NAV loop.

All business logic is pure. Receives ReturnData + PriceData + PortfolioConfig
and produces BacktestResult consumed by metrics.py.
"""

from dataclasses import dataclass

import pandas as pd

from finance.consts import DEFAULT_IV, DRIFT_BAND_RELATIVE, LEAPS_KEY_SUFFIX
from finance.data import PriceData
from finance.leverage import (
    LeapsConfig,
    LeapsLedger,
    RebalanceRule,
    WeightStrategy,
    compute_leaps_mtm,
)
from finance.returns import ReturnData

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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
    """

    target_weights: dict[str, float]
    initial_nav: float
    monthly_contribution: float
    rebalance_rule: RebalanceRule
    weight_strategy: WeightStrategy
    leaps_config: LeapsConfig | None = None


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
# Backtest orchestrator
# ---------------------------------------------------------------------------


def run_backtest(
    return_data: ReturnData,
    price_data: PriceData,
    config: PortfolioConfig,
) -> BacktestResult:
    """Run the core portfolio backtest loop.

    For each trading day:
      a. Apply asset returns to holdings.
      b. Compute market return (before contribution, to exclude cash-flow effects).
      c. On month-end: apply monthly_contribution proportional to target weights.
      d. On rebalance date: realign holdings to target weights.
      e. If leaps_config present: include LEAPS mark-to-market in total NAV.

    LEAPS overlay (when config.leaps_config is set):
      - Requires "VTI" in price_data.prices for absolute spot pricing.
      - run_leaps_simulation is called internally; no external ledger accepted.

    Arguments:
        return_data: ReturnData containing daily simple returns for all assets.
        price_data: PriceData providing absolute asset prices (used for LEAPS spot).
        config: PortfolioConfig specifying weights, contributions, and rebalancing.

    Returns:
        BacktestResult with NAV series, weight history, return series, and ledger.

    Raises:
        ValueError: If any base asset in config.target_weights is absent from return_data.
        ValueError: If a LEAPS underlying (key without "_LEAPS") is absent from price_data.prices.
        ValueError: If more than one distinct LEAPS underlying is requested.
        ValueError: If LEAPS keys are present but config.leaps_config is None.
    """
    from finance.leverage import run_leaps_simulation

    returns = return_data.returns

    # Split target weights into base assets and carved-out LEAPS keys (Model B)
    leaps_keys = [k for k in config.target_weights if k.endswith(LEAPS_KEY_SUFFIX)]
    base_assets = [k for k in config.target_weights if k not in leaps_keys]

    missing = [a for a in base_assets if a not in returns.columns]
    if missing:
        raise ValueError(f"Assets missing from return_data: {missing}")

    use_leaps = len(leaps_keys) > 0
    if use_leaps and config.leaps_config is None:
        raise ValueError("LEAPS keys present in target_weights but leaps_config is None")

    # Normalize all weights (base + LEAPS) to sum to 1.0
    raw_w = pd.Series({k: config.target_weights[k] for k in config.target_weights})
    norm_w = raw_w / raw_w.sum()
    leaps_fraction = float(sum(norm_w[k] for k in leaps_keys))

    # Base-only target weights, renormalized among base assets (sum to 1.0)
    base_target_w = pd.Series({a: norm_w[a] for a in base_assets})
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

        initial_leaps_capital = config.initial_nav * leaps_fraction
        leaps_monthly = config.monthly_contribution * leaps_fraction
        leaps_ledger = run_leaps_simulation(
            underlying_prices,
            leaps_monthly,
            config.leaps_config,
            risk_free_series=return_data.risk_free_rate,
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

    nav_values: list[float] = []
    return_values: list[float] = []
    weight_rows: list[dict[str, float]] = []

    for date in returns.index:
        date_ts = pd.Timestamp(date)
        day_ret = returns.loc[date_ts]

        # (a) Apply daily returns to base holdings
        for a in base_assets:
            holdings[a] *= 1.0 + float(day_ret[a])  # type: ignore[arg-type]

        # (e) LEAPS gross mark-to-market (carved-out capital value)
        leaps_value = 0.0
        if leaps_ledger is not None and underlying_prices is not None:
            spot = float(underlying_prices.loc[date_ts])
            rfr = float(rfr_series.loc[date_ts]) if rfr_series is not None else 0.0
            leaps_value = compute_leaps_mtm(leaps_ledger, date_ts, spot, iv, rfr)

        nav_before_contrib = sum(holdings.values()) + leaps_value

        # (b) Market return — excludes the upcoming contribution
        port_return = nav_before_contrib / prev_total_nav - 1.0

        # (c) Month-end: add the base share of the contribution across base assets
        if date_ts in month_end_dates and base_assets:
            alloc = apply_contribution(nav_before_contrib, base_contribution, base_target_w)
            for a in base_assets:
                holdings[a] += alloc[a]

        # (d) Rebalance: realign base holdings to base target weights
        if date_ts in rebal_dates and base_assets:
            base_nav = sum(holdings.values())
            for a in base_assets:
                holdings[a] = base_nav * float(base_target_w[a])

        total_nav = sum(holdings.values()) + leaps_value

        nav_values.append(total_nav)
        return_values.append(port_return)
        row = {a: holdings[a] / total_nav for a in base_assets}
        for k in leaps_keys:
            share = float(norm_w[k]) / leaps_fraction if leaps_fraction > 0 else 0.0
            row[k] = leaps_value * share / total_nav
        weight_rows.append(row)

        prev_total_nav = total_nav

    nav_series = pd.Series(nav_values, index=returns.index, name="NAV")
    return_series = pd.Series(return_values, index=returns.index, name="portfolio_return")
    weight_history = pd.DataFrame(weight_rows, index=returns.index)

    return BacktestResult(
        nav_series=nav_series,
        weight_history=weight_history,
        return_series=return_series,
        leaps_ledger=leaps_ledger,
        config=config,
    )
