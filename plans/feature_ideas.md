# Features

1. (Small, P0): `figures.py` - Add pareto frontier comparisons for `List` or `Dict` of `PerformanceMetrics`.
    - eg. Pairwise metric comparison across set of portfolio results
2. (Medium, P0): `portfolio.py` - Add functions for managing the portfolio on weekly/monthly basis.
    - Portfolio Ledger (`PortfolioState`) tracking and metrics
        - MTM
        - Weights and rebalance rules
    - DCA signals for Leaps
    - GTT signal
3. (Small/Med, P1) `returns.py` - Add per-asset contribution calculations and reporting
4. (Large, P1/P2) New module - forward forecasting of returns: VAR-GARCH bootstrap
5. (P2) Speed improvements - Migrate core functions from `Python` to `Rust`

# Testing

1. Add `mutmut`, particularly for `metrics.py`
2. 