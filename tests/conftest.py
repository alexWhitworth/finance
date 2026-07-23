"""Shared pytest fixtures for the finance test suite.

Fixtures defined here are available to all test modules without explicit import.
daily_dates, sample_prices, and sample_returns are used by Phase 3+ tests
(volatility.py, metrics.py, portfolio.py).
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_dates() -> pd.DatetimeIndex:
    """252 trading days starting 2020-01-02. Used by Phase 3+ tests."""
    return pd.bdate_range("2020-01-02", periods=252)


@pytest.fixture
def sample_prices(daily_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Synthetic price DataFrame for 6 assets over 252 business days. Used by Phase 3+ tests."""
    rng = np.random.default_rng(42)
    tickers = ["VTI", "VXUS", "GLD", "VTEB", "KMLM", "VGIT"]
    starts = {"VTI": 200.0, "VXUS": 60.0, "GLD": 170.0, "VTEB": 55.0, "KMLM": 25.0, "VGIT": 65.0}
    data = {}
    for t in tickers:
        shocks = rng.normal(0.0003, 0.01, size=len(daily_dates))
        prices = starts[t] * np.cumprod(1 + shocks)
        data[t] = prices
    return pd.DataFrame(data, index=daily_dates)


@pytest.fixture
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns derived from sample_prices. Used by Phase 3+ tests."""
    return sample_prices.pct_change().dropna()
