"""Smoke test for the finance package's public API surface (F-016)."""

import finance


def test_all_public_names_importable() -> None:
    """Every name in finance.__all__ must be importable with no ImportError."""
    for name in finance.__all__:
        assert hasattr(finance, name), f"finance.{name} missing despite being in __all__"


def test_expected_public_names_present() -> None:
    """The full public surface from spec §4h must be present on the finance package."""
    expected = {
        "BacktestResult",
        "PortfolioConfig",
        "PortfolioState",
        "GttConfig",
        "run_backtest",
        "LivePortfolio",
        "NavBreakdown",
        "HoldingView",
        "RebalancePlan",
        "TradeOrder",
        "VolatilityReport",
        "GttStatus",
        "as_live_portfolio",
        "compute_nav_breakdown",
        "compute_holdings_view",
        "compute_rebalance_plan",
        "compute_volatility_report",
        "compute_gtt_status",
        "ContractGreeks",
        "PortfolioGreeks",
        "compute_contract_greeks",
        "compute_portfolio_greeks",
        "LeapsDcaSignal",
        "compute_leaps_dca_signal",
    }
    assert expected <= set(finance.__all__)
    for name in expected:
        assert hasattr(finance, name)
