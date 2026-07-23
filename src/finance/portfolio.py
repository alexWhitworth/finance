"""Portfolio dataclasses and backtest engine.

Dataclass definitions (PortfolioConfig, BacktestResult) are used by metrics.py
and leverage.py. The full backtest loop will be implemented in Phase 6.
"""

from dataclasses import dataclass

import pandas as pd

from finance.leverage import LeapsConfig, LeapsLedger, RebalanceRule, WeightStrategy


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
        nav_series: DatetimeIndex → portfolio NAV over time.
        weight_history: DatetimeIndex x asset, realized daily weights.
        return_series: DatetimeIndex, daily portfolio simple returns.
        leaps_ledger: Full LEAPS history, or None if no LEAPS overlay was used.
        config: PortfolioConfig that produced this result.
    """

    nav_series: pd.Series
    weight_history: pd.DataFrame
    return_series: pd.Series
    leaps_ledger: LeapsLedger | None
    config: PortfolioConfig
