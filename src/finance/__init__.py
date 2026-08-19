"""Finance backtesting and forward projection library.

Two flows are exposed at the top level: the existing backtest engine
(``run_backtest``) and the live portfolio management layer (``LivePortfolio``
and friends), plus LEAPS greeks and the LEAPS DCA entry signal.
"""

from finance._portfolio_types import (
    BacktestResult,
    GttConfig,
    PortfolioConfig,
    PortfolioState,
)
from finance.dca_signal import LeapsDcaSignal, compute_leaps_dca_signal
from finance.greeks import (
    ContractGreeks,
    PortfolioGreeks,
    compute_contract_greeks,
    compute_portfolio_greeks,
)
from finance.portfolio import run_backtest
from finance.portfolio_manager import (
    GttStatus,
    HoldingView,
    LivePortfolio,
    NavBreakdown,
    RebalancePlan,
    TradeOrder,
    VolatilityReport,
    as_live_portfolio,
    compute_gtt_status,
    compute_holdings_view,
    compute_nav_breakdown,
    compute_rebalance_plan,
    compute_volatility_report,
)

__all__ = [
    "BacktestResult",
    "ContractGreeks",
    "GttConfig",
    "GttStatus",
    "HoldingView",
    "LeapsDcaSignal",
    "LivePortfolio",
    "NavBreakdown",
    "PortfolioConfig",
    "PortfolioGreeks",
    "PortfolioState",
    "RebalancePlan",
    "TradeOrder",
    "VolatilityReport",
    "as_live_portfolio",
    "compute_contract_greeks",
    "compute_gtt_status",
    "compute_holdings_view",
    "compute_leaps_dca_signal",
    "compute_nav_breakdown",
    "compute_portfolio_greeks",
    "compute_rebalance_plan",
    "compute_volatility_report",
    "run_backtest",
]
