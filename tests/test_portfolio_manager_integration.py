"""Integration tests for as_live_portfolio — full backtest → LivePortfolio (I11).

These tests run a complete backtest and verify that the bridge function
produces a LivePortfolio whose target_weights sum to 1.0 ± 1e-6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance._portfolio_types import PortfolioConfig
from finance.data import PriceData
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance.portfolio import run_backtest
from finance.portfolio_manager import as_live_portfolio, compute_nav_breakdown
from finance.returns import ReturnData, build_return_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TICKERS = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
_EQUAL_WEIGHTS = {t: 1.0 / len(_TICKERS) for t in _TICKERS}

_LEAPS_WEIGHTS = {
    "VTI_LEAPS": 0.30, "VTI": 0.10, "VXUS": 0.15, "GLD": 0.15,
    "MUB": 0.10, "KMLM": 0.10, "VGIT": 0.10,
}


def _make_rd_and_pd(
    n: int = 756,
    daily_ret: float = 0.0003,
    daily_vol: float = 0.01,
    seed: int = 42,
    start: str = "2015-01-02",
) -> tuple[ReturnData, PriceData]:
    """Return matching (ReturnData, PriceData) pair from synthetic series.

    Arguments:
        n: Number of trading days.
        daily_ret: Mean daily return.
        daily_vol: Daily return standard deviation.
        seed: Random seed.
        start: Start date string.

    Returns:
        Tuple of (ReturnData, PriceData).
    """
    idx = pd.bdate_range(start, periods=n + 1)
    rng = np.random.default_rng(seed)
    starts = {
        "VTI": 200.0, "VXUS": 60.0, "GLD": 170.0,
        "MUB": 55.0, "KMLM": 25.0, "VGIT": 65.0,
    }
    prices_data = {
        t: starts[t] * np.cumprod(1 + rng.normal(daily_ret, daily_vol, n + 1))
        for t in _TICKERS
    }
    prices = pd.DataFrame(prices_data, index=idx)
    dividends = pd.DataFrame(0.0, index=idx, columns=list(_TICKERS))
    pd_obj = PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=pd.DataFrame(),
        tickers=_TICKERS,
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
    )
    return build_return_data(pd_obj, apply_tey=False), pd_obj


def _config(
    weights: dict[str, float] | None = None,
    initial_nav: float = 1_000_000.0,
    contribution: float = 5_000.0,
    leaps_config: LeapsConfig | None = None,
) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights=weights or dict(_EQUAL_WEIGHTS),
        initial_nav=initial_nav,
        monthly_contribution=contribution,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=leaps_config,
    )


# ---------------------------------------------------------------------------
# Integration: full backtest → as_live_portfolio
# ---------------------------------------------------------------------------


def test_integration_no_leaps_target_weights_sum_one() -> None:
    """Full backtest without LEAPS: LivePortfolio.target_weights sum to 1.0 ± 1e-6."""
    rd, pd_obj = _make_rd_and_pd(756)
    result = run_backtest(rd, pd_obj, _config())
    lp = as_live_portfolio(result)
    total = sum(lp.target_weights.values())
    assert abs(total - 1.0) < 1e-6, f"target_weights sum = {total}"


def test_integration_leaps_target_weights_sum_one() -> None:
    """Full backtest with LEAPS: LivePortfolio.target_weights sum to 1.0 ± 1e-6."""
    rd, pd_obj = _make_rd_and_pd(756)
    leaps_cfg = LeapsConfig(account_type=AccountType.TAXABLE)
    cfg = _config(weights=dict(_LEAPS_WEIGHTS), leaps_config=leaps_cfg)
    result = run_backtest(rd, pd_obj, cfg)
    lp = as_live_portfolio(result)
    total = sum(lp.target_weights.values())
    assert abs(total - 1.0) < 1e-6, f"target_weights sum = {total}"


def test_integration_as_of_date_is_last_nav_date() -> None:
    """LivePortfolio.as_of_date equals result.nav_series.index[-1]."""
    rd, pd_obj = _make_rd_and_pd(756)
    result = run_backtest(rd, pd_obj, _config())
    lp = as_live_portfolio(result)
    assert lp.as_of_date == pd.Timestamp(result.nav_series.index[-1])


def test_integration_leaps_live_contracts_all_active() -> None:
    """All returned leaps_contracts have expiry_date > LivePortfolio.as_of_date."""
    rd, pd_obj = _make_rd_and_pd(756)
    leaps_cfg = LeapsConfig(account_type=AccountType.TAXABLE)
    cfg = _config(weights=dict(_LEAPS_WEIGHTS), leaps_config=leaps_cfg)
    result = run_backtest(rd, pd_obj, cfg)
    lp = as_live_portfolio(result)
    for contract, _scale in lp.leaps_contracts:
        assert contract.expiry_date > lp.as_of_date, (
            f"expired contract in LivePortfolio: {contract.expiry_date} <= {lp.as_of_date}"
        )


def test_integration_defensive_sleeve_nonnegative() -> None:
    """LivePortfolio.defensive_sleeve is non-negative after a full backtest."""
    rd, pd_obj = _make_rd_and_pd(756)
    result = run_backtest(rd, pd_obj, _config())
    lp = as_live_portfolio(result)
    assert lp.defensive_sleeve >= 0.0


def test_integration_holdings_covers_base_assets() -> None:
    """LivePortfolio.holdings contains at least the base asset tickers."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    lp = as_live_portfolio(result)
    # Every equal-weight ticker should appear in holdings.
    for ticker in _EQUAL_WEIGHTS:
        assert ticker in lp.holdings, f"{ticker} missing from holdings"


def test_integration_leaps_scale_within_bounds() -> None:
    """All leaps_scale values are in (0, 1] after a full LEAPS backtest."""
    rd, pd_obj = _make_rd_and_pd(756)
    leaps_cfg = LeapsConfig(account_type=AccountType.TAXABLE)
    cfg = _config(weights=dict(_LEAPS_WEIGHTS), leaps_config=leaps_cfg)
    result = run_backtest(rd, pd_obj, cfg)
    lp = as_live_portfolio(result)
    for _contract, scale in lp.leaps_contracts:
        assert 0.0 < scale <= 1.0, f"scale {scale} out of bounds"


def test_integration_nav_breakdown_total_nav_matches_nav_series_no_leaps() -> None:
    """I11: compute_nav_breakdown(lp).total_nav ≈ result.nav_series.iloc[-1] within 1e-6 (no LEAPS)."""
    rd, pd_obj = _make_rd_and_pd(756)
    result = run_backtest(rd, pd_obj, _config())
    lp = as_live_portfolio(result)
    nb = compute_nav_breakdown(lp, leaps_mtm=0.0)
    expected = float(result.nav_series.iloc[-1])
    assert abs(nb.total_nav - expected) < 1e-6, (
        f"I11 violation: nav_breakdown.total_nav={nb.total_nav} "
        f"vs nav_series[-1]={expected}"
    )


def test_integration_nav_breakdown_total_nav_matches_nav_series_with_leaps() -> None:
    """I11: compute_nav_breakdown(lp, leaps_mtm=leaps_value).total_nav ≈ nav_series[-1] with LEAPS."""
    rd, pd_obj = _make_rd_and_pd(756)
    leaps_cfg = LeapsConfig(account_type=AccountType.TAXABLE)
    cfg = _config(weights=dict(_LEAPS_WEIGHTS), leaps_config=leaps_cfg)
    result = run_backtest(rd, pd_obj, cfg)
    lp = as_live_portfolio(result)
    # leaps_value from final_state is the last MTM valuation used in the backtest
    leaps_mtm = result.final_state.leaps_value
    nb = compute_nav_breakdown(lp, leaps_mtm=leaps_mtm)
    expected = float(result.nav_series.iloc[-1])
    assert abs(nb.total_nav - expected) < 1e-6, (
        f"I11 violation: nav_breakdown.total_nav={nb.total_nav} "
        f"vs nav_series[-1]={expected}"
    )


@pytest.mark.slow
def test_integration_longer_backtest_still_valid() -> None:
    """3-year backtest with contributions: LivePortfolio remains valid."""
    rd, pd_obj = _make_rd_and_pd(n=756, seed=99)
    leaps_cfg = LeapsConfig(account_type=AccountType.TAX_SHELTERED)
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS),
        initial_nav=500_000.0,
        contribution=8_000.0,
        leaps_config=leaps_cfg,
    )
    result = run_backtest(rd, pd_obj, cfg)
    lp = as_live_portfolio(result)
    assert abs(sum(lp.target_weights.values()) - 1.0) < 1e-6
    for c, s in lp.leaps_contracts:
        assert c.expiry_date > lp.as_of_date
        assert 0.0 < s <= 1.0
