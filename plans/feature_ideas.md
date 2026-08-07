# Features

- (Medium, P0): `portfolio.py` - Add functions for managing the portfolio on weekly/monthly basis.
    - See [portfolio_management](./portfolio_management.md)
- (Small/Med, P1) `returns.py` - Add per-asset contribution calculations and reporting
- (Large, P1/P2) New module - forward forecasting of returns: VAR-GARCH bootstrap
- (P2) Speed improvements - Migrate core functions from `Python` to `Rust`

# Testing

1. (P0) Add `mutmut`, particularly for `metrics.py` and `leverage.py`
2. 