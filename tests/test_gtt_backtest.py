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
    _long_windows,
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


def _signal_from_mask(mask_values: np.ndarray, index: pd.DatetimeIndex) -> GttSignalData:
    """Build a GttSignalData directly from a 0/1 mask array aligned to index."""
    zeros = pd.Series(0, index=index)
    return GttSignalData(
        position_mask=pd.Series(mask_values, index=index, name="position_mask"),
        ue_signal=zeros,
        vix_signal=zeros,
        vix_p90_threshold=0.272,
        unrate_start=pd.Timestamp(index[0]),
        vix_start=pd.Timestamp(index[0]),
    )


def _all_long_signal(index: pd.DatetimeIndex, threshold: float = 0.272) -> GttSignalData:
    """A GttSignalData whose mask is 1 (Long) on every date in index."""
    return _signal_from_mask(np.ones(len(index), dtype=int), index)


def _window_signal(index: pd.DatetimeIndex, lo: int, hi: int) -> GttSignalData:
    """Mask that is Defensive (0) on [lo, hi) and Long (1) elsewhere."""
    m = np.ones(len(index), dtype=int)
    m[lo:hi] = 0
    return _signal_from_mask(m, index)


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


# ---------------------------------------------------------------------------
# F-10c — equity-only GTT branch (no LEAPS)
# ---------------------------------------------------------------------------

_TICKER_ORDER = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")


def test_all_long_mask_equals_no_gtt() -> None:
    """An all-Long mask reproduces the no-GTT run within 1e-9 terminal NAV."""
    rd, pd_obj = _make_rd_and_pd(252)
    idx = pd.DatetimeIndex(rd.returns.index)
    baseline = run_backtest(rd, pd_obj, _config(contribution=10_000.0))
    gtt = run_backtest(
        rd, pd_obj,
        _config(contribution=10_000.0, gtt_config=_gtt_config()),
        gtt_signal=_all_long_signal(idx),
    )
    assert gtt.nav_series.iloc[-1] == pytest.approx(baseline.nav_series.iloc[-1], abs=1e-9)
    # Governed leg is never zeroed under an all-Long mask.
    assert (gtt.weight_history["VTI"] > 0).all()


def test_defensive_window_zeros_vti_and_redistributes() -> None:
    """During a defensive window VTI weight is exactly 0 and defensive tickers carry it."""
    rd, pd_obj = _make_rd_and_pd(252)
    idx = pd.DatetimeIndex(rd.returns.index)
    gtt = run_backtest(
        rd, pd_obj,
        _config(gtt_config=_gtt_config()),
        gtt_signal=_window_signal(idx, 50, 100),
    )
    wh = gtt.weight_history
    assert wh["VTI"].iloc[50:100].abs().max() == 0.0
    # Redistributed capital shows up under the defensive sleeve (R_f synthetic column).
    assert "R_f" in wh.columns
    assert (wh["R_f"].iloc[50:100] > 0).all()
    # R_f is 0 on Long days (sleeve empty).
    assert wh["R_f"].iloc[10] == 0.0
    # Every row still sums to 1.0.
    np.testing.assert_allclose(wh.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_defensive_sleeve_rf_only_earns_daily_rfr() -> None:
    """An all-R_f defensive sleeve compounds NAV by exactly rfr/252 each defensive day."""
    rd, pd_obj = _make_rd_and_pd(252)
    idx = pd.DatetimeIndex(rd.returns.index)
    # VTI is the sole weighted base asset; sleeve is pure R_f.
    weights = {"VTI": 1.0, "KMLM": 0.0, "VGIT": 0.0, "GLD": 0.0}
    gc = GttConfig(vix_p90_threshold=0.272, defensive_weights={"R_f": 1.0})
    cfg = PortfolioConfig(
        target_weights=weights, initial_nav=1_000_000.0, monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY, weight_strategy=WeightStrategy.USER_SPECIFIED,
        gtt_config=gc,
    )
    res = run_backtest(rd, pd_obj, cfg, gtt_signal=_window_signal(idx, 50, 100))
    rfr = rd.risk_free_rate.reindex(idx, method="ffill").fillna(0.0)
    expected = res.nav_series.iloc[49]
    for i in range(50, 100):
        expected *= 1.0 + rfr.iloc[i] / 252.0
    assert res.nav_series.iloc[99] == pytest.approx(expected, rel=1e-12)
    assert res.weight_history["R_f"].iloc[60] == pytest.approx(1.0, abs=1e-12)


def test_reentry_restores_target_weights() -> None:
    """On the first Long day after a defensive window, weights snap to target (1e-9)."""
    rd, pd_obj = _make_rd_and_pd(252)
    idx = pd.DatetimeIndex(rd.returns.index)
    gtt = run_backtest(
        rd, pd_obj,
        _config(gtt_config=_gtt_config()),
        gtt_signal=_window_signal(idx, 50, 100),  # re-entry on day 100
    )
    wh = gtt.weight_history
    target = 1.0 / len(_TICKER_ORDER)
    for a in _TICKER_ORDER:
        assert wh[a].iloc[100] == pytest.approx(target, abs=1e-9)


def test_rebalance_on_defensive_day_keeps_vti_zero() -> None:
    """A quarterly rebalance landing on a defensive day still yields 0 VTI (Option C)."""
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    # Find a quarterly rebalance date and force a defensive window covering it.
    from finance.portfolio import get_rebalance_dates

    rebal = get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    assert rebal, "expected at least one rebalance date in a 2y window"
    pos = idx.get_loc(rebal[len(rebal) // 2])
    gtt = run_backtest(
        rd, pd_obj,
        _config(gtt_config=_gtt_config()),
        gtt_signal=_window_signal(idx, pos - 2, pos + 3),  # defensive across the rebalance
    )
    # VTI repopulated by the rebalance is re-zeroed by the GTT override on that day.
    assert gtt.weight_history["VTI"].iloc[pos] == 0.0


def test_whipsaw_multiple_windows_all_long_between() -> None:
    """Multiple defensive windows each zero VTI; Long gaps between restore it."""
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    m = np.ones(len(idx), dtype=int)
    m[40:60] = 0
    m[120:140] = 0
    gtt = run_backtest(
        rd, pd_obj, _config(gtt_config=_gtt_config()), gtt_signal=_signal_from_mask(m, idx)
    )
    wh = gtt.weight_history
    assert wh["VTI"].iloc[40:60].abs().max() == 0.0
    assert wh["VTI"].iloc[120:140].abs().max() == 0.0
    assert wh["VTI"].iloc[80] > 0.0  # Long gap between the two windows
    np.testing.assert_allclose(wh.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_terminal_defensive_window_reports_no_vti() -> None:
    """A timeline ending inside a defensive window keeps VTI at 0 through the end."""
    rd, pd_obj = _make_rd_and_pd(252)
    idx = pd.DatetimeIndex(rd.returns.index)
    n = len(idx)
    gtt = run_backtest(
        rd, pd_obj, _config(gtt_config=_gtt_config()), gtt_signal=_window_signal(idx, n - 30, n)
    )
    assert gtt.weight_history["VTI"].iloc[-1] == 0.0
    assert gtt.weight_history["R_f"].iloc[-1] > 0.0


def test_gtt_config_without_vti_is_noop() -> None:
    """gtt_config set but target_weights hold no governed ticker -> GTT is a no-op."""
    rd, pd_obj = _make_rd_and_pd(252)
    idx = pd.DatetimeIndex(rd.returns.index)
    # No VTI in the portfolio; defensive_weights must still be corpus tickers.
    weights = {"VXUS": 0.5, "GLD": 0.25, "MUB": 0.25}
    gc = GttConfig(vix_p90_threshold=0.272, defensive_weights={"GLD": 0.5, "MUB": 0.5})
    cfg_gtt = PortfolioConfig(
        target_weights=weights, initial_nav=1_000_000.0, monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY, weight_strategy=WeightStrategy.USER_SPECIFIED,
        gtt_config=gc,
    )
    cfg_plain = PortfolioConfig(
        target_weights=weights, initial_nav=1_000_000.0, monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY, weight_strategy=WeightStrategy.USER_SPECIFIED,
    )
    gtt = run_backtest(rd, pd_obj, cfg_gtt, gtt_signal=_window_signal(idx, 50, 100))
    plain = run_backtest(rd, pd_obj, cfg_plain)
    # No governed ticker -> defensive window changes nothing.
    assert gtt.nav_series.iloc[-1] == pytest.approx(plain.nav_series.iloc[-1], abs=1e-9)


def test_gtt_with_leaps_not_yet_supported() -> None:
    """A LEAPS carve-out under GTT raises until F-10d lands (documents the guard)."""
    from finance.leverage import AccountType, LeapsConfig

    rd, pd_obj = _make_rd_and_pd(252)
    idx = pd.DatetimeIndex(rd.returns.index)
    weights = {"VTI": 0.4, "VTI_LEAPS": 0.2, "VXUS": 0.2, "GLD": 0.1, "MUB": 0.1}
    gc = GttConfig(vix_p90_threshold=0.272, defensive_weights={"R_f": 0.5, "GLD": 0.5})
    cfg = PortfolioConfig(
        target_weights=weights, initial_nav=1_000_000.0, monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY, weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(account_type=AccountType.TAXABLE),
        gtt_config=gc,
    )
    with pytest.raises(NotImplementedError, match="F-10d"):
        run_backtest(rd, pd_obj, cfg, gtt_signal=_all_long_signal(idx))


# ---------------------------------------------------------------------------
# F-10d.1 — _long_windows
# ---------------------------------------------------------------------------


def test_long_windows_all_long_single_window() -> None:
    """An all-Long mask yields one window spanning the whole index."""
    idx = pd.bdate_range("2021-01-04", periods=10)
    mask = pd.Series(1, index=idx)
    wins = _long_windows(mask)
    assert wins == [(idx[0], idx[-1])]


def test_long_windows_all_defensive_empty() -> None:
    """An all-Defensive mask yields no windows."""
    idx = pd.bdate_range("2021-01-04", periods=10)
    mask = pd.Series(0, index=idx)
    assert _long_windows(mask) == []


def test_long_windows_single_interior_defensive_gap() -> None:
    """One defensive gap splits the timeline into two Long windows."""
    idx = pd.bdate_range("2021-01-04", periods=10)
    m = np.ones(10, dtype=int)
    m[3:6] = 0  # defensive days 3,4,5
    wins = _long_windows(pd.Series(m, index=idx))
    assert wins == [(idx[0], idx[2]), (idx[6], idx[9])]


def test_long_windows_leading_and_trailing_defensive() -> None:
    """Leading and trailing defensive runs are excluded from the windows."""
    idx = pd.bdate_range("2021-01-04", periods=10)
    m = np.zeros(10, dtype=int)
    m[2:5] = 1  # the only Long run is days 2,3,4
    wins = _long_windows(pd.Series(m, index=idx))
    assert wins == [(idx[2], idx[4])]


def test_long_windows_single_day_windows() -> None:
    """Alternating regimes produce single-day Long windows."""
    idx = pd.bdate_range("2021-01-04", periods=5)
    m = np.array([1, 0, 1, 0, 1])
    wins = _long_windows(pd.Series(m, index=idx))
    assert wins == [(idx[0], idx[0]), (idx[2], idx[2]), (idx[4], idx[4])]
