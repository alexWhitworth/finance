"""Tests for the F-10 GTT branch of run_backtest.

Organized by implementation slice:
  * F-10a — signature, validation, and the gtt_signal=None regression gate.
  * F-10b.. — added as later slices land.

The synthetic 6-asset corpus and PriceData/ReturnData builders mirror
tests/test_portfolio.py so GTT results can be compared against the no-GTT baseline.
"""

import numpy as np
import pandas as pd
import pytest

from finance.data import PriceData
from finance.gtt import GttSignalData
from finance.leverage import RebalanceRule, WeightStrategy
from finance.portfolio import GttConfig, PortfolioConfig, run_backtest
from finance.returns import ReturnData, build_return_data

# ---------------------------------------------------------------------------
# Corpus builders (aligned with tests/test_portfolio.py)
# ---------------------------------------------------------------------------

_TICKERS = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
_EQUAL_WEIGHTS = {t: 1.0 / len(_TICKERS) for t in _TICKERS}


def _make_price_data(
    n: int = 504,
    daily_ret: float = 0.0003,
    daily_vol: float = 0.01,
    seed: int = 42,
    start: str = "2015-01-02",
) -> PriceData:
    """Synthetic PriceData for the 6-asset corpus."""
    idx = pd.bdate_range(start, periods=n + 1)
    rng = np.random.default_rng(seed)
    starts = {
        "VTI": 200.0, "VXUS": 60.0, "GLD": 170.0,
        "MUB": 55.0, "KMLM": 25.0, "VGIT": 65.0,
    }
    prices = pd.DataFrame(
        {t: starts[t] * np.cumprod(1 + rng.normal(daily_ret, daily_vol, n + 1)) for t in _TICKERS},
        index=idx,
    )
    dividends = pd.DataFrame(0.0, index=idx, columns=list(_TICKERS))
    return PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=pd.DataFrame(),
        tickers=_TICKERS,
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
    )


def _make_rd_and_pd(
    n: int = 504,
    daily_ret: float = 0.0003,
    daily_vol: float = 0.01,
    seed: int = 42,
    start: str = "2015-01-02",
) -> tuple[ReturnData, PriceData]:
    """Matching (ReturnData, PriceData) from one synthetic series."""
    pd_obj = _make_price_data(n, daily_ret, daily_vol, seed, start)
    return build_return_data(pd_obj, apply_tey=False), pd_obj


def _gtt_config() -> GttConfig:
    """GttConfig whose defensive tickers all exist in the 6-asset corpus."""
    return GttConfig(
        vix_p90_threshold=0.272,
        defensive_weights={"R_f": 0.25, "KMLM": 0.25, "VGIT": 0.25, "GLD": 0.25},
    )


def _config(
    weights: dict[str, float] | None = None,
    gtt_config: GttConfig | None = None,
    contribution: float = 0.0,
    rebalance_rule: RebalanceRule = RebalanceRule.QUARTERLY,
) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights=weights or dict(_EQUAL_WEIGHTS),
        initial_nav=1_000_000.0,
        monthly_contribution=contribution,
        rebalance_rule=rebalance_rule,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        gtt_config=gtt_config,
    )


def _all_long_signal(index: pd.DatetimeIndex, threshold: float = 0.272) -> GttSignalData:
    """A GttSignalData whose mask is 1 (Long) on every date in index."""
    mask = pd.Series(1, index=index, name="position_mask")
    zeros = pd.Series(0, index=index)
    return GttSignalData(
        position_mask=mask,
        ue_signal=zeros,
        vix_signal=zeros,
        vix_p90_threshold=threshold,
        unrate_start=pd.Timestamp(index[0]),
        vix_start=pd.Timestamp(index[0]),
    )


# ---------------------------------------------------------------------------
# F-10a — gtt_signal=None regression gate (exact equality vs pre-GTT behavior)
# ---------------------------------------------------------------------------


def test_gtt_none_matches_baseline_no_leaps() -> None:
    """gtt_signal=None reproduces the no-GTT result exactly (no LEAPS overlay)."""
    rd, pd_obj = _make_rd_and_pd(504)
    cfg = _config(contribution=10_000.0)
    baseline = run_backtest(rd, pd_obj, cfg)
    with_none = run_backtest(rd, pd_obj, cfg, gtt_signal=None)

    pd.testing.assert_series_equal(with_none.nav_series, baseline.nav_series)
    pd.testing.assert_series_equal(with_none.return_series, baseline.return_series)
    pd.testing.assert_frame_equal(with_none.weight_history, baseline.weight_history)
    assert with_none.leaps_ledger is None
    assert baseline.leaps_ledger is None


def test_gtt_none_matches_baseline_drift_rule() -> None:
    """gtt_signal=None regression holds under the DRIFT rebalance rule too."""
    rd, pd_obj = _make_rd_and_pd(504)
    cfg = _config(contribution=5_000.0, rebalance_rule=RebalanceRule.DRIFT)
    baseline = run_backtest(rd, pd_obj, cfg)
    with_none = run_backtest(rd, pd_obj, cfg, gtt_signal=None)
    pd.testing.assert_series_equal(with_none.nav_series, baseline.nav_series)
    pd.testing.assert_frame_equal(with_none.weight_history, baseline.weight_history)


def test_gtt_none_default_param_equals_explicit_none() -> None:
    """The default (omitted) gtt_signal equals passing gtt_signal=None explicitly."""
    rd, pd_obj = _make_rd_and_pd(252)
    cfg = _config()
    default_call = run_backtest(rd, pd_obj, cfg)
    explicit = run_backtest(rd, pd_obj, cfg, gtt_signal=None)
    pd.testing.assert_series_equal(default_call.nav_series, explicit.nav_series)


# ---------------------------------------------------------------------------
# F-10a — validation
# ---------------------------------------------------------------------------


def test_signal_without_config_raises() -> None:
    """gtt_signal provided but config.gtt_config None -> ValueError."""
    rd, pd_obj = _make_rd_and_pd(252)
    cfg = _config(gtt_config=None)  # no gtt_config
    sig = _all_long_signal(pd.DatetimeIndex(rd.returns.index))
    with pytest.raises(ValueError, match="both be set or both be None"):
        run_backtest(rd, pd_obj, cfg, gtt_signal=sig)


def test_config_without_signal_raises() -> None:
    """config.gtt_config set but gtt_signal None -> ValueError."""
    rd, pd_obj = _make_rd_and_pd(252)
    cfg = _config(gtt_config=_gtt_config())
    with pytest.raises(ValueError, match="both be set or both be None"):
        run_backtest(rd, pd_obj, cfg, gtt_signal=None)


def test_both_none_is_valid() -> None:
    """Neither gtt_signal nor gtt_config set is the ordinary (legacy) path."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config(gtt_config=None), gtt_signal=None)
    assert len(result.nav_series) == len(rd.returns)


def test_missing_defensive_ticker_raises() -> None:
    """A non-R_f defensive_weights ticker absent from return_data -> ValueError."""
    rd, pd_obj = _make_rd_and_pd(252)
    bad_gtt = GttConfig(
        vix_p90_threshold=0.272,
        defensive_weights={"R_f": 0.5, "SPY": 0.5},  # SPY not in the 6-asset corpus
    )
    # target_weights must still contain SPY to pass PortfolioConfig's own check.
    weights = dict(_EQUAL_WEIGHTS)
    weights["SPY"] = 0.0
    cfg = _config(weights=weights, gtt_config=bad_gtt)
    sig = _all_long_signal(pd.DatetimeIndex(rd.returns.index))
    with pytest.raises(ValueError, match="absent from return_data"):
        run_backtest(rd, pd_obj, cfg, gtt_signal=sig)


def test_missing_defensive_ticker_reports_name() -> None:
    """The ValueError names the offending ticker(s)."""
    rd, pd_obj = _make_rd_and_pd(252)
    bad_gtt = GttConfig(
        vix_p90_threshold=0.272,
        defensive_weights={"R_f": 0.5, "SPY": 0.5},
    )
    weights = dict(_EQUAL_WEIGHTS)
    weights["SPY"] = 0.0
    cfg = _config(weights=weights, gtt_config=bad_gtt)
    sig = _all_long_signal(pd.DatetimeIndex(rd.returns.index))
    with pytest.raises(ValueError, match="SPY"):
        run_backtest(rd, pd_obj, cfg, gtt_signal=sig)
