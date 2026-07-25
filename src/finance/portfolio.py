"""Portfolio backtest engine — rebalancing, contribution, and NAV loop.

All business logic is pure. Receives ReturnData + PriceData + PortfolioConfig
and produces BacktestResult consumed by metrics.py.
"""

from dataclasses import dataclass

import pandas as pd

from finance.consts import DEFAULT_IV, DRIFT_BAND_RELATIVE
from finance.data import PriceData
from finance.leverage import (
    LeapsConfig,
    LeapsLedger,
    RebalanceRule,
    WeightStrategy,
    compute_leaps_nav_contribution,
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
        ValueError: If any asset in config.target_weights is absent from return_data.
        ValueError: If leaps_config is set but "VTI" is absent from price_data.prices.
    """
    from finance.leverage import run_leaps_simulation

    returns = return_data.returns
    assets = list(config.target_weights.keys())

    missing = [a for a in assets if a not in returns.columns]
    if missing:
        raise ValueError(f"Assets missing from return_data: {missing}")

    # Normalized target weights
    raw_w = pd.Series({a: config.target_weights[a] for a in assets})
    target_w = raw_w / raw_w.sum()

    idx = pd.DatetimeIndex(returns.index)

    # Rebalance date set (O(1) lookups)
    rebal_dates: set[pd.Timestamp] = set(get_rebalance_dates(idx, config.rebalance_rule))

    # Month-end date set: last trading day of each calendar month
    month_end_dates: set[pd.Timestamp] = {
        pd.Timestamp(grp.index[-1])
        for _, grp in returns.groupby(idx.to_period("M"))
    }

    # LEAPS setup — run simulation internally if config requests it
    leaps_ledger: LeapsLedger | None = None
    vti_prices: pd.Series | None = None
    iv = DEFAULT_IV
    rfr_series: pd.Series | None = None

    if config.leaps_config is not None:
        if "VTI" not in price_data.prices.columns:
            raise ValueError("leaps_config requires 'VTI' in price_data.prices")
        vti_prices = price_data.prices["VTI"].reindex(idx, method="ffill")
        iv = config.leaps_config.iv
        rfr_series = return_data.risk_free_rate.reindex(idx, method="ffill").fillna(0.0)
        leaps_ledger = run_leaps_simulation(
            vti_prices,
            config.monthly_contribution,
            config.leaps_config,
            risk_free_series=return_data.risk_free_rate,
        )

    # Initialize holdings: dollar value per asset
    holdings: dict[str, float] = {
        a: config.initial_nav * float(target_w[a]) for a in assets
    }
    prev_total_nav = config.initial_nav

    nav_values: list[float] = []
    return_values: list[float] = []
    weight_rows: list[dict[str, float]] = []

    for date in returns.index:
        date_ts = pd.Timestamp(date)
        day_ret = returns.loc[date_ts]

        # (a) Apply daily returns to holdings
        for a in assets:
            holdings[a] *= 1.0 + float(day_ret[a])  # type: ignore[arg-type]

        # (e) LEAPS mark-to-market contribution (computed before contribution cash flow)
        leaps_contrib = 0.0
        if leaps_ledger is not None and vti_prices is not None:
            spot = float(vti_prices.loc[date_ts])
            rfr = float(rfr_series.loc[date_ts]) if rfr_series is not None else 0.0
            leaps_contrib = compute_leaps_nav_contribution(
                leaps_ledger, date_ts, spot, iv, rfr
            )

        nav_before_contrib = sum(holdings.values()) + leaps_contrib

        # (b) Market return — excludes the upcoming contribution
        port_return = nav_before_contrib / prev_total_nav - 1.0

        # (c) Month-end: add contribution proportional to target weights
        if date_ts in month_end_dates:
            alloc = apply_contribution(nav_before_contrib, config.monthly_contribution, target_w)
            for a in assets:
                holdings[a] += alloc[a]

        # (d) Rebalance: realign base holdings to target weights
        if date_ts in rebal_dates:
            base_nav = sum(holdings.values())
            for a in assets:
                holdings[a] = base_nav * float(target_w[a])

        total_nav = sum(holdings.values()) + leaps_contrib

        nav_values.append(total_nav)
        return_values.append(port_return)
        weight_rows.append({a: holdings[a] / total_nav for a in assets})

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
