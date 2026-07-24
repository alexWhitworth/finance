"""Performance metrics and report assembly.

All functions are pure. Receives BacktestResult + ReturnData + VolatilityModel
and produces PerformanceMetrics / PerformanceReport.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance.consts import (
    CRISIS_PERIODS,
    MIN_CRISIS_OBSERVATIONS,
    RISK_FREE_RATE_DEFAULT,
    TRADING_DAYS_PER_YEAR,
)
from finance.portfolio import BacktestResult
from finance.returns import ReturnData
from finance.volatility import VolatilityModel, build_vol_contribution_table, forecast_portfolio_vol

RISK_FREE_RATE = RISK_FREE_RATE_DEFAULT


@dataclass(frozen=True)
class PerformanceMetrics:
    """Scalar performance statistics for one return series slice.

    Attributes:
        annualized_return: Geometric annualized return.
        annualized_std: Annualized standard deviation of daily returns.
        max_drawdown: Maximum peak-to-trough drawdown as a positive fraction.
        sharpe: Annualized Sharpe ratio.
        sortino: Sortino ratio using downside deviation.
        calmar: Annualized return divided by max drawdown.
        omega: Omega ratio (probability-weighted gains over losses).
        period_label: Human-readable name for this slice, e.g. "Full Period".
    """

    annualized_return: float
    annualized_std: float
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    omega: float
    period_label: str


@dataclass(frozen=True)
class PerformanceReport:
    """Complete performance report for a backtest run.

    Attributes:
        full_period: Metrics computed over the entire return history.
        crisis_periods: Per-period metrics for defined stress intervals.
        vol_contribution_table: DataFrame with columns
            [sigma_tilde, sigma_hat, rho_VTI, contrib], indexed by asset.
        forward_vol_forecast: Annualized one-step-ahead portfolio vol sigma_hat_p.
    """

    full_period: PerformanceMetrics
    crisis_periods: tuple[PerformanceMetrics, ...]
    vol_contribution_table: pd.DataFrame
    forward_vol_forecast: float


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def annualized_return(returns: pd.Series) -> float:
    """Compute geometric annualized return from daily simple returns.

    Arguments:
        returns: Daily simple return Series.

    Returns:
        Annualized return as a decimal (e.g. 0.08 for 8%).
    """
    n = len(returns)
    if n == 0:
        return 0.0
    growth_f = float(np.prod(1.0 + returns.to_numpy(dtype=float)))
    years = n / TRADING_DAYS_PER_YEAR
    return float(growth_f ** (1.0 / years) - 1.0)


def annualized_std(returns: pd.Series) -> float:
    """Compute annualized standard deviation of daily returns.

    Arguments:
        returns: Daily simple return Series.

    Returns:
        Annualized standard deviation.
    """
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(nav_series: pd.Series) -> float:
    """Compute maximum peak-to-trough drawdown.

    Arguments:
        nav_series: Portfolio NAV Series (DatetimeIndex, strictly positive).

    Returns:
        Maximum drawdown as a positive fraction (e.g. 0.50 for a 50% drawdown).
        Returns 0.0 if nav_series has fewer than 2 observations.
    """
    if len(nav_series) < 2:
        return 0.0
    running_max = nav_series.cummax()
    drawdowns = (nav_series - running_max) / running_max
    return float(-drawdowns.min())


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Compute annualized Sharpe ratio.

    Arguments:
        returns: Daily simple return Series.
        risk_free_rate: Annual risk-free rate (default 0.0).

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if std is zero.
    """
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    std = float(returns.std(ddof=1))
    if std < 1e-14:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Compute Sortino ratio using downside deviation.

    Arguments:
        returns: Daily simple return Series.
        risk_free_rate: Annual risk-free rate (default 0.0).

    Returns:
        Annualized Sortino ratio. Returns 0.0 if downside deviation is zero.
    """
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    downside = excess[excess < 0.0]
    if len(downside) == 0:
        return float("inf")
    downside_std = float(np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR))
    if downside_std == 0.0:
        return 0.0
    ann_ret = annualized_return(returns)
    return float((ann_ret - risk_free_rate) / downside_std)


def calmar_ratio(returns: pd.Series, nav_series: pd.Series) -> float:
    """Compute Calmar ratio (annualized return / max drawdown).

    Arguments:
        returns: Daily simple return Series.
        nav_series: Corresponding NAV Series (same index).

    Returns:
        Calmar ratio. Returns inf if max_drawdown is 0.
    """
    ann = annualized_return(returns)
    mdd = max_drawdown(nav_series)
    if mdd == 0.0:
        return float("inf")
    return float(ann / mdd)


def omega_ratio(returns: pd.Series, threshold: float = RISK_FREE_RATE) -> float:
    """Compute Omega ratio.

    Omega = sum(max(r - threshold, 0)) / (sum(max(threshold - r, 0)) + epsilon).

    Arguments:
        returns: Daily simple return Series.
        threshold: Annual return threshold (used as daily_threshold = threshold / 252).
            Typically the risk-free rate. Default RISK_FREE_RATE.

    Returns:
        Omega ratio. Returns inf if there are no returns below threshold.
    """
    if len(returns) == 0:
        return 0.0
    daily_threshold = threshold / TRADING_DAYS_PER_YEAR
    gains = (returns - daily_threshold).clip(lower=0.0).sum()
    losses = (daily_threshold - returns).clip(lower=0.0).sum()
    eps = 1e-12
    return float(gains / (losses + eps))


def slice_period(returns: pd.Series, start: str, end: str) -> pd.Series:
    """Slice a return Series to a date range (inclusive on both ends).

    Arguments:
        returns: Daily simple return Series with DatetimeIndex.
        start: ISO date string for the slice start (inclusive).
        end: ISO date string for the slice end (inclusive).

    Returns:
        Sliced return Series. May be empty if the range is outside the index.
    """
    result: pd.Series = returns.loc[start:end]
    return result


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


def compute_metrics(
    returns: pd.Series,
    nav_series: pd.Series,
    period_label: str,
    risk_free_rate: float = RISK_FREE_RATE,
) -> PerformanceMetrics:
    """Compute all performance metrics for a given return/NAV slice.

    Arguments:
        returns: Daily simple return Series.
        nav_series: Corresponding NAV Series (same index).
        period_label: Human-readable name for this slice.
        risk_free_rate: Annual risk-free rate (default 0.0).

    Returns:
        PerformanceMetrics dataclass with all scalar metrics populated.
    """
    return PerformanceMetrics(
        annualized_return=annualized_return(returns),
        annualized_std=annualized_std(returns),
        max_drawdown=max_drawdown(nav_series),
        sharpe=sharpe_ratio(returns, risk_free_rate),
        sortino=sortino_ratio(returns, risk_free_rate),
        calmar=calmar_ratio(returns, nav_series),
        omega=omega_ratio(returns, risk_free_rate),
        period_label=period_label,
    )


def build_performance_report(
    backtest_result: BacktestResult,
    return_data: ReturnData,
    vol_model: VolatilityModel,
    crisis_periods: dict[str, tuple[str, str]] = CRISIS_PERIODS,
) -> PerformanceReport:
    """Build a complete PerformanceReport from a finished backtest.

    Crisis-period metrics are included only when the backtest overlaps the
    crisis window by at least MIN_CRISIS_OBSERVATIONS trading days.

    The risk-free rate for each period (full or crisis) is taken as the mean
    of return_data.risk_free_rate over that period's trading days, reflecting
    the actual prevailing T-bill rate rather than a single hardcoded scalar.

    Arguments:
        backtest_result: Output of portfolio.run_backtest().
        return_data: ReturnData used to build the backtest (for vol table and
            risk-free rate series).
        vol_model: VolatilityModel at the backtest end date (for vol table).
        crisis_periods: Mapping of label → (start, end) ISO strings.

    Returns:
        PerformanceReport with full_period, crisis_periods, vol table, and
        forward_vol_forecast.
    """
    port_returns = backtest_result.return_series
    nav = backtest_result.nav_series
    rfr = return_data.risk_free_rate

    full_rfr = float(rfr.reindex(port_returns.index, method="ffill").fillna(0.0).mean())
    full_period = compute_metrics(port_returns, nav, "Full Period", full_rfr)

    crisis_metrics: list[PerformanceMetrics] = []
    for label, (start, end) in crisis_periods.items():
        crisis_ret = slice_period(port_returns, start, end)
        if len(crisis_ret) < MIN_CRISIS_OBSERVATIONS:
            continue
        crisis_nav = nav.loc[crisis_ret.index]
        crisis_rfr = float(rfr.reindex(crisis_ret.index, method="ffill").fillna(0.0).mean())
        crisis_metrics.append(compute_metrics(crisis_ret, crisis_nav, label, crisis_rfr))

    weights = pd.Series(backtest_result.config.target_weights)
    weights = weights / weights.sum()
    vol_table = build_vol_contribution_table(weights, return_data, vol_model)
    fwd_vol = forecast_portfolio_vol(weights, vol_model)

    return PerformanceReport(
        full_period=full_period,
        crisis_periods=tuple(crisis_metrics),
        vol_contribution_table=vol_table,
        forward_vol_forecast=fwd_vol,
    )
