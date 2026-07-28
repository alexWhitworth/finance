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
from finance.portfolio import (
    GttConfig,
    PortfolioConfig,
    _defensive_gross_return,
    _gtt_governed_keys,
    _reindex_position_mask,
    run_backtest,
)
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


# ---------------------------------------------------------------------------
# F-10b — _gtt_governed_keys
# ---------------------------------------------------------------------------


def test_governed_keys_vti_and_leaps() -> None:
    """Both VTI and VTI_LEAPS are governed; unrelated assets are not."""
    keys = _gtt_governed_keys({"VTI": 0.5, "VTI_LEAPS": 0.2, "VXUS": 0.2, "GLD": 0.1})
    assert keys == {"VTI", "VTI_LEAPS"}


def test_governed_keys_plain_vti_only() -> None:
    """A portfolio with VTI but no LEAPS carve-out governs just VTI."""
    assert _gtt_governed_keys({"VTI": 0.6, "VXUS": 0.4}) == {"VTI"}


def test_governed_keys_empty_when_no_vti() -> None:
    """No GTT_EQUITY_TICKERS present -> empty set -> GTT no-op."""
    assert _gtt_governed_keys({"VXUS": 0.5, "GLD": 0.5}) == set()


def test_governed_keys_leaps_without_base() -> None:
    """A VTI_LEAPS carve-out is governed even if plain VTI is absent."""
    assert _gtt_governed_keys({"VTI_LEAPS": 0.3, "VXUS": 0.7}) == {"VTI_LEAPS"}


# ---------------------------------------------------------------------------
# F-10b — _reindex_position_mask
# ---------------------------------------------------------------------------


def test_reindex_mask_ffill_across_holiday_gap() -> None:
    """A missing (holiday) date forward-fills from the prior signal value."""
    src_idx = pd.to_datetime(["2020-01-02", "2020-01-06"])
    mask = pd.Series([0, 1], index=src_idx)
    target = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    out = _reindex_position_mask(mask, pd.DatetimeIndex(target))
    assert out.tolist() == [0, 0, 1]  # 01-03 ffills the 01-02 value (0)
    assert out.dtype == int


def test_reindex_mask_leading_gap_defaults_long() -> None:
    """Dates before any signal default to 1 (Long)."""
    src_idx = pd.to_datetime(["2020-01-10"])
    mask = pd.Series([0], index=src_idx)
    target = pd.to_datetime(["2020-01-08", "2020-01-09", "2020-01-10"])
    out = _reindex_position_mask(mask, pd.DatetimeIndex(target))
    assert out.tolist() == [1, 1, 0]


def test_reindex_mask_domain_is_zero_one() -> None:
    """Output is always in {0, 1}."""
    idx = pd.bdate_range("2021-01-04", periods=10)
    mask = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], index=idx)
    out = _reindex_position_mask(mask, pd.DatetimeIndex(idx))
    assert set(out.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# F-10b — _defensive_gross_return
# ---------------------------------------------------------------------------


def test_defensive_gross_return_blend_matches_hand_computation() -> None:
    """Blended sleeve return equals Sum_i w_i r_i with R_f -> rfr/252."""
    idx = pd.bdate_range("2021-01-04", periods=3)
    returns = pd.DataFrame(
        {"KMLM": [0.01, -0.02, 0.03], "VGIT": [0.0, 0.01, -0.01], "GLD": [0.02, 0.02, 0.0]},
        index=idx,
    )
    rfr = pd.Series([0.0252, 0.0252, 0.0252], index=idx)  # annualized 2.52%
    weights = {"R_f": 0.25, "KMLM": 0.25, "VGIT": 0.25, "GLD": 0.25}
    out = _defensive_gross_return(returns, rfr, weights)

    rf_day = 0.0252 / 252.0
    expected = [
        0.25 * rf_day + 0.25 * 0.01 + 0.25 * 0.0 + 0.25 * 0.02,
        0.25 * rf_day + 0.25 * -0.02 + 0.25 * 0.01 + 0.25 * 0.02,
        0.25 * rf_day + 0.25 * 0.03 + 0.25 * -0.01 + 0.25 * 0.0,
    ]
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-12)


def test_defensive_gross_return_no_rf_key() -> None:
    """Without an R_f key the blend is a pure weighted asset return."""
    idx = pd.bdate_range("2021-01-04", periods=2)
    returns = pd.DataFrame({"KMLM": [0.10, -0.05], "GLD": [0.00, 0.20]}, index=idx)
    rfr = pd.Series([0.05, 0.05], index=idx)  # ignored (no R_f weight)
    out = _defensive_gross_return(returns, rfr, {"KMLM": 0.6, "GLD": 0.4})
    np.testing.assert_allclose(
        out.to_numpy(), [0.6 * 0.10 + 0.4 * 0.0, 0.6 * -0.05 + 0.4 * 0.20], atol=1e-12
    )


def test_defensive_gross_return_zero_weight_ticker() -> None:
    """A zero-weight defensive ticker contributes nothing (boundary)."""
    idx = pd.bdate_range("2021-01-04", periods=2)
    returns = pd.DataFrame({"KMLM": [0.10, 0.10], "GLD": [0.20, 0.20]}, index=idx)
    rfr = pd.Series([0.0, 0.0], index=idx)
    out = _defensive_gross_return(returns, rfr, {"KMLM": 1.0, "GLD": 0.0})
    np.testing.assert_allclose(out.to_numpy(), [0.10, 0.10], atol=1e-12)


def test_defensive_gross_return_rf_only_earns_daily_rfr() -> None:
    """An all-R_f sleeve earns exactly rfr/252 each day."""
    idx = pd.bdate_range("2021-01-04", periods=2)
    returns = pd.DataFrame({"KMLM": [0.5, 0.5]}, index=idx)  # present but unweighted
    rfr = pd.Series([0.0504, 0.0504], index=idx)
    out = _defensive_gross_return(returns, rfr, {"R_f": 1.0})
    np.testing.assert_allclose(out.to_numpy(), [0.0504 / 252.0] * 2, atol=1e-15)
