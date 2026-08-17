"""Performance metrics and report assembly.

All functions are pure. Receives BacktestResult + ReturnData + VolatilityModel
and produces PerformanceMetrics / PerformanceReport.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from finance._portfolio_types import BacktestResult
from finance.consts import (
    CRISIS_PERIODS,
    MIN_CRISIS_OBSERVATIONS,
    TRADING_DAYS_PER_YEAR,
)
from finance.data import PriceData
from finance.leverage import (
    LeapsTaxSummary,
    TerminalNav,
    compute_leaps_tax_summary,
    compute_terminal_nav,
)
from finance.returns import ReturnData
from finance.volatility import VolatilityModel, build_vol_contribution_table, forecast_portfolio_vol


@dataclass(frozen=True)
class PerformanceMetrics:
    """Scalar performance statistics for one return series slice.

    Attributes:
        annualized_return: Geometric annualized return.
        annualized_std: Annualized standard deviation of daily returns.
        max_drawdown: Maximum peak-to-trough drawdown as a positive fraction.
        sharpe: Annualized Sharpe ratio (on excess returns).
        sortino: Sortino ratio using downside deviation of excess returns.
        calmar: Annualized return divided by max drawdown.
        omega: Omega ratio evaluated on excess returns.
        skewness: Third standardized moment of daily excess returns.
        excess_kurtosis: Fourth standardized moment minus 3 (0 = normal tails).
        period_label: Human-readable name for this slice, e.g. "Full Period".
    """

    annualized_return: float
    annualized_std: float
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    omega: float
    skewness: float
    excess_kurtosis: float
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
        final_nav: Portfolio NAV at the last backtest date (None for
            sub-period slices, e.g. crisis windows, that have no terminal state).
        terminal_nav: Pre/post-tax terminal NAV (None when no LEAPS overlay).
        tax_summary: LEAPS tax drag summary (None when no LEAPS overlay).
    """

    full_period: PerformanceMetrics
    crisis_periods: tuple[PerformanceMetrics, ...]
    vol_contribution_table: pd.DataFrame
    forward_vol_forecast: float
    final_nav: float | None = None
    terminal_nav: TerminalNav | None = None
    tax_summary: LeapsTaxSummary | None = None


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


def sharpe_ratio(returns: pd.Series, risk_free_rate: pd.Series) -> float:
    """Compute annualized Sharpe ratio using a time-varying risk-free rate.

    Excess return: r_e(t) = r_p(t) - R_f(t) / TRADING_DAYS_PER_YEAR.
    Sharpe = mean(r_e) / std(r_e) * sqrt(TRADING_DAYS_PER_YEAR).

    Arguments:
        returns: Daily simple return Series.
        risk_free_rate: Daily annualized risk-free rate Series (decimal, e.g. 0.05
            for 5%). Forward-filled and aligned to returns.index before subtraction.

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if std is zero or series is too short.
    """
    if len(returns) < 2:
        return 0.0
    rfr = risk_free_rate.reindex(returns.index, method="ffill").fillna(0.0)
    excess = returns - rfr / TRADING_DAYS_PER_YEAR
    std = float(excess.std(ddof=1))
    if std < 1e-14:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_rate: pd.Series) -> float:
    """Compute Sortino ratio using downside deviation of excess returns.

    Excess return: r_e(t) = r_p(t) - R_f(t) / TRADING_DAYS_PER_YEAR.
    Downside deviation: sigma_d = sqrt(mean(min(r_e(t), 0)^2)) * sqrt(252).
    Sortino = annualized_return(r_e) / sigma_d.

    Arguments:
        returns: Daily simple return Series.
        risk_free_rate: Daily annualized risk-free rate Series (decimal).
            Forward-filled and aligned to returns.index.

    Returns:
        Annualized Sortino ratio. Returns inf if no downside, 0.0 if too short.
    """
    if len(returns) < 2:
        return 0.0
    rfr = risk_free_rate.reindex(returns.index, method="ffill").fillna(0.0)
    excess = returns - rfr / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0.0]
    if len(downside) == 0:
        return float("inf")
    downside_std = float(np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR))
    if downside_std == 0.0:
        return 0.0
    return float(annualized_return(excess) / downside_std)


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


def omega_ratio(returns: pd.Series, risk_free_rate: pd.Series) -> float:
    """Compute Omega ratio evaluated on excess returns.

    Omega = sum(max(r_e(t), 0)) / (sum(max(-r_e(t), 0)) + epsilon)
    where r_e(t) = r_p(t) - R_f(t) / TRADING_DAYS_PER_YEAR.
    The threshold is implicitly zero on the excess return series.

    Arguments:
        returns: Daily simple return Series.
        risk_free_rate: Daily annualized risk-free rate Series (decimal).
            Forward-filled and aligned to returns.index.

    Returns:
        Omega ratio. Returns 0.0 for an empty series.
    """
    if len(returns) == 0:
        return 0.0
    rfr = risk_free_rate.reindex(returns.index, method="ffill").fillna(0.0)
    excess = returns - rfr / TRADING_DAYS_PER_YEAR
    gains = excess.clip(lower=0.0).sum()
    losses = (-excess).clip(lower=0.0).sum()
    eps = 1e-12
    return float(gains / (losses + eps))


def return_skewness(excess_returns: pd.Series) -> float:
    """Compute third standardized moment (skewness) of excess return series.

    Arguments:
        excess_returns: Daily excess return Series.

    Returns:
        Skewness. Returns 0.0 for fewer than 4 observations.
    """
    if len(excess_returns) < 4:
        return 0.0
    vals = excess_returns.to_numpy(dtype=float)
    return float(scipy_stats.skew(vals, bias=False))


def return_excess_kurtosis(excess_returns: pd.Series) -> float:
    """Compute excess kurtosis (fourth standardized moment minus 3) of excess returns.

    Arguments:
        excess_returns: Daily excess return Series.

    Returns:
        Excess kurtosis (0.0 implies normal tails). Returns 0.0 for < 4 observations.
    """
    if len(excess_returns) < 4:
        return 0.0
    vals = excess_returns.to_numpy(dtype=float)
    return float(scipy_stats.kurtosis(vals, bias=False))


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
    risk_free_rate: pd.Series,
) -> PerformanceMetrics:
    """Compute all performance metrics for a given return/NAV slice.

    Excess returns are computed once and reused for all ratio functions and
    distribution shape metrics.

    Arguments:
        returns: Daily simple return Series.
        nav_series: Corresponding NAV Series (same index).
        period_label: Human-readable name for this slice.
        risk_free_rate: Daily annualized risk-free rate Series (decimal).
            Aligned to returns.index with forward-fill before use.

    Returns:
        PerformanceMetrics dataclass with all scalar metrics populated.
    """
    rfr_aligned = risk_free_rate.reindex(returns.index, method="ffill").fillna(0.0)
    excess = returns - rfr_aligned / TRADING_DAYS_PER_YEAR
    return PerformanceMetrics(
        annualized_return=annualized_return(returns),
        annualized_std=annualized_std(returns),
        max_drawdown=max_drawdown(nav_series),
        sharpe=sharpe_ratio(returns, risk_free_rate),
        sortino=sortino_ratio(returns, risk_free_rate),
        calmar=calmar_ratio(returns, nav_series),
        omega=omega_ratio(returns, risk_free_rate),
        skewness=return_skewness(excess),
        excess_kurtosis=return_excess_kurtosis(excess),
        period_label=period_label,
    )


def build_performance_report(
    backtest_result: BacktestResult,
    price_data: PriceData,
    return_data: ReturnData,
    vol_model: VolatilityModel,
    crisis_periods: dict[str, tuple[str, str]] = CRISIS_PERIODS,
) -> PerformanceReport:
    """Build a complete PerformanceReport from a finished backtest.

    Crisis-period metrics are included only when the backtest overlaps the
    crisis window by at least MIN_CRISIS_OBSERVATIONS trading days.

    The risk-free rate for each period (full or crisis) is the sliced
    return_data.risk_free_rate Series for that window — no scalar averaging.

    Arguments:
        backtest_result: Output of portfolio.run_backtest().
        price_data: PriceData used to build the backtest. The final VTI spot
            price (price_data.prices["VTI"].iloc[-1]) is used for terminal NAV
            pricing when a LEAPS ledger is present.
        return_data: ReturnData used to build the backtest (for vol table and
            risk-free rate series).
        vol_model: VolatilityModel at the backtest end date (for vol table).
        crisis_periods: Mapping of label → (start, end) ISO strings.

    Returns:
        PerformanceReport with full_period, crisis_periods, vol table,
        forward_vol_forecast, and final_nav always populated; terminal_nav /
        tax_summary populated only when a LEAPS ledger is present (None otherwise).
    """
    port_returns = backtest_result.return_series
    nav = backtest_result.nav_series
    rfr = return_data.risk_free_rate

    full_rfr = rfr.reindex(port_returns.index, method="ffill").fillna(0.0)
    full_start = port_returns.index[0].strftime("%Y-%m")
    full_end = port_returns.index[-1].strftime("%Y-%m")
    full_period = compute_metrics(
        port_returns, nav, f"Full Period ({full_start}:{full_end})", full_rfr
    )

    crisis_metrics: list[PerformanceMetrics] = []
    for label, (start, end) in crisis_periods.items():
        crisis_ret = slice_period(port_returns, start, end)
        if len(crisis_ret) < MIN_CRISIS_OBSERVATIONS:
            continue
        crisis_nav = nav.loc[crisis_ret.index]
        crisis_rfr = rfr.reindex(crisis_ret.index, method="ffill").fillna(0.0)
        start_fmt = pd.Timestamp(start).strftime("%Y-%m")
        end_fmt = pd.Timestamp(end).strftime("%Y-%m")
        crisis_metrics.append(
            compute_metrics(crisis_ret, crisis_nav, f"{label} ({start_fmt}:{end_fmt})", crisis_rfr)
        )

    weights = pd.Series(backtest_result.config.target_weights)
    weights = weights / weights.sum()
    vol_table = build_vol_contribution_table(weights, return_data, vol_model)
    fwd_vol = forecast_portfolio_vol(weights, vol_model)

    t_nav: TerminalNav | None = None
    t_summary: LeapsTaxSummary | None = None
    ledger = backtest_result.leaps_ledger
    if ledger is not None:
        final_nav_val = float(nav.iloc[-1])
        final_date = nav.index[-1]
        final_spot = float(price_data.prices["VTI"].iloc[-1])
        t_nav = compute_terminal_nav(ledger, final_nav_val, final_date, final_spot)
        years = len(port_returns) / TRADING_DAYS_PER_YEAR
        t_summary = compute_leaps_tax_summary(ledger, t_nav, final_nav_val, years)

    return PerformanceReport(
        full_period=full_period,
        crisis_periods=tuple(crisis_metrics),
        vol_contribution_table=vol_table,
        forward_vol_forecast=fwd_vol,
        final_nav=float(nav.iloc[-1]),
        terminal_nav=t_nav,
        tax_summary=t_summary,
    )
