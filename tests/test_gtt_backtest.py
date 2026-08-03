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
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance._backtest_steps import _defensive_gross_return, _gtt_governed_keys, _long_windows
from finance._portfolio_types import GttConfig, PortfolioConfig
from finance.portfolio import run_backtest
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
    from finance._backtest_steps import _get_rebalance_dates

    rebal = _get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
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


# ---------------------------------------------------------------------------
# F-10d.2 — first-window ledger (all-Long GTT+LEAPS == no-GTT LEAPS baseline)
# ---------------------------------------------------------------------------

_LEAPS_WEIGHTS = {
    "VTI_LEAPS": 0.30, "VTI": 0.10, "VXUS": 0.15, "GLD": 0.15,
    "MUB": 0.15, "KMLM": 0.075, "VGIT": 0.075,
}


def _leaps_gtt_config(gtt: bool) -> PortfolioConfig:
    """LEAPS portfolio config, optionally with the GTT overlay attached."""
    return PortfolioConfig(
        target_weights=dict(_LEAPS_WEIGHTS),
        initial_nav=1_000_000.0,
        monthly_contribution=5_000.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(account_type=AccountType.TAXABLE),
        gtt_config=_gtt_config() if gtt else None,
    )


def test_all_long_gtt_leaps_equals_no_gtt_leaps() -> None:
    """All-Long GTT+LEAPS reproduces the no-GTT LEAPS baseline exactly."""
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    baseline = run_backtest(rd, pd_obj, _leaps_gtt_config(gtt=False))
    gtt = run_backtest(rd, pd_obj, _leaps_gtt_config(gtt=True), gtt_signal=_all_long_signal(idx))

    pd.testing.assert_series_equal(gtt.nav_series, baseline.nav_series)
    pd.testing.assert_frame_equal(gtt.weight_history, baseline.weight_history)
    assert baseline.leaps_ledger is not None and gtt.leaps_ledger is not None
    assert len(gtt.leaps_ledger.contracts) == len(baseline.leaps_ledger.contracts)
    # No defensive transitions -> no GTT close events.
    assert gtt.leaps_ledger.gtt_close_events == ()


# ---------------------------------------------------------------------------
# F-10d.3 — Long->Defensive force-close (observable behavior; ledger events in d.5)
# ---------------------------------------------------------------------------


def _leaps_sheltered_gtt_config() -> PortfolioConfig:
    """TAX_SHELTERED LEAPS+GTT config (force-close realizes no tax)."""
    return PortfolioConfig(
        target_weights=dict(_LEAPS_WEIGHTS),
        initial_nav=1_000_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(account_type=AccountType.TAX_SHELTERED),
        gtt_config=_gtt_config(),
    )


def test_defensive_window_zeros_leaps_weight() -> None:
    """VTI_LEAPS weight is exactly 0 through a defensive window (contracts closed)."""
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    r = run_backtest(
        rd, pd_obj, _leaps_gtt_config(gtt=True), gtt_signal=_window_signal(idx, 200, 260)
    )
    assert r.weight_history["VTI_LEAPS"].iloc[200:260].abs().max() == 0.0
    # VTI (base equity) is also zeroed by the equity overlay.
    assert r.weight_history["VTI"].iloc[200:260].abs().max() == 0.0
    np.testing.assert_allclose(r.weight_history.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def _leaps_only_config(account_type: AccountType) -> PortfolioConfig:
    """VTI + VTI_LEAPS only, with an R_f-only defensive sleeve.

    During a defensive window the entire portfolio (base VTI sleeve + parked LEAPS
    pool) rides rfr/252, isolating the force-close/pool mechanics from other assets.
    """
    gc = GttConfig(vix_p90_threshold=0.272, defensive_weights={"R_f": 1.0})
    return PortfolioConfig(
        target_weights={"VTI": 0.5, "VTI_LEAPS": 0.5},
        initial_nav=1_000_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(account_type=account_type),
        gtt_config=gc,
    )


def test_force_close_nav_rides_rfr_tax_sheltered() -> None:
    """TAX_SHELTERED: whole VTI/LEAPS portfolio parked in R_f compounds by rfr/252.

    With no tax on the close, force-closing the LEAPS conserves capital, so every
    defensive day (including the transition day) the NAV grows by exactly rfr/252.
    """
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    r = run_backtest(
        rd, pd_obj, _leaps_only_config(AccountType.TAX_SHELTERED),
        gtt_signal=_window_signal(idx, 200, 260),
    )
    rfr = rd.risk_free_rate.reindex(idx, method="ffill").fillna(0.0)
    expected = r.nav_series.iloc[199]
    for i in range(200, 260):
        expected *= 1.0 + rfr.iloc[i] / 252.0
    assert r.nav_series.iloc[259] == pytest.approx(expected, rel=1e-12)


def test_taxable_close_realizes_drag_vs_sheltered() -> None:
    """A taxable force-close on gains leaves less parked capital than TAX_SHELTERED."""
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    taxable = run_backtest(
        rd, pd_obj, _leaps_only_config(AccountType.TAXABLE),
        gtt_signal=_window_signal(idx, 200, 260),
    )
    sheltered = run_backtest(
        rd, pd_obj, _leaps_only_config(AccountType.TAX_SHELTERED),
        gtt_signal=_window_signal(idx, 200, 260),
    )
    # Identical up to the close; the taxable account pays LTCG on the gained LEAPS
    # at the transition, so its parked NAV is strictly lower thereafter.
    assert taxable.nav_series.iloc[205] < sheltered.nav_series.iloc[205]


# ---------------------------------------------------------------------------
# F-10d.4 — Defensive->Long re-entry (forced rebalance seeds fresh LEAPS)
# ---------------------------------------------------------------------------


def test_reentry_restores_full_target_including_leaps() -> None:
    """On re-entry, weight_history lands exactly on target including the LEAPS leg."""
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    r = run_backtest(
        rd, pd_obj, _leaps_gtt_config(gtt=True), gtt_signal=_window_signal(idx, 200, 250)
    )
    tw = _LEAPS_WEIGHTS
    for k, target in tw.items():
        assert r.weight_history[k].iloc[250] == pytest.approx(target, abs=1e-9)
    # LEAPS leg equals leaps_fraction (0.30) at re-entry.
    assert r.weight_history["VTI_LEAPS"].iloc[250] == pytest.approx(0.30, abs=1e-9)


def test_reentry_leaps_capital_equals_fraction_of_nav() -> None:
    """Re-entry LEAPS MTM == leaps_fraction * total_NAV within a create-flooring basis.

    With a single fresh DITM contract seeded at leaps_fraction * NAV and no VIX
    (so pricing IV == config.iv), the day-of-re-entry LEAPS value equals
    leaps_fraction * total_NAV up to the create_leaps_contract share rounding.
    """
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    r = run_backtest(
        rd, pd_obj, _leaps_only_config(AccountType.TAX_SHELTERED),
        gtt_signal=_window_signal(idx, 200, 250),
    )
    nav = r.nav_series.iloc[250]
    leaps_val = r.weight_history["VTI_LEAPS"].iloc[250] * nav
    assert leaps_val == pytest.approx(0.5 * nav, rel=1e-9)


def test_pool_and_sleeve_compound_with_monthly_diversion() -> None:
    """Parked pool + sleeve compound by rfr/252 and absorb month-end contributions.

    Isolated setup: TAX_SHELTERED (close conserves capital), all-R_f sleeve, with a
    monthly contribution. Through the defensive window the entire portfolio is
    parked, so its value follows a deterministic recurrence: each day it grows by
    rfr/252, and on each month-end the full monthly_contribution (base share +
    diverted leaps share) is added. Re-entry NAV must match that recurrence exactly.
    """
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    contribution = 12_000.0
    gc = GttConfig(vix_p90_threshold=0.272, defensive_weights={"R_f": 1.0})
    cfg = PortfolioConfig(
        target_weights={"VTI": 0.5, "VTI_LEAPS": 0.5},
        initial_nav=1_000_000.0,
        monthly_contribution=contribution,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(account_type=AccountType.TAX_SHELTERED),
        gtt_config=gc,
    )
    lo, hi = 200, 250  # defensive window spans at least one month-end
    r = run_backtest(rd, pd_obj, cfg, gtt_signal=_window_signal(idx, lo, hi))

    month_end_dates = {grp.index[-1] for _, grp in rd.returns.groupby(idx.to_period("M"))}
    rfr = rd.risk_free_rate.reindex(idx, method="ffill").fillna(0.0)
    nav = r.nav_series.iloc[lo - 1]  # last Long day value
    for i in range(lo, hi + 1):
        nav *= 1.0 + rfr.iloc[i] / 252.0
        if idx[i] in month_end_dates:
            nav += contribution
    assert r.nav_series.iloc[hi] == pytest.approx(nav, rel=1e-9)


# ---------------------------------------------------------------------------
# F-10d.5 — final ledger assembly + gtt_close_events
# ---------------------------------------------------------------------------


def test_tax_sheltered_zero_close_tax() -> None:
    """TAX_SHELTERED force-closes realize zero tax across all gtt_close_events."""
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    r = run_backtest(
        rd, pd_obj, _leaps_only_config(AccountType.TAX_SHELTERED),
        gtt_signal=_window_signal(idx, 200, 260),
    )
    assert r.leaps_ledger is not None
    assert len(r.leaps_ledger.gtt_close_events) >= 1
    assert sum(e.tax_paid for e in r.leaps_ledger.gtt_close_events) == 0.0


def test_taxable_close_taxes_positive_gains() -> None:
    """A taxable force-close on a gained contract records positive tax."""
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    r = run_backtest(
        rd, pd_obj, _leaps_only_config(AccountType.TAXABLE),
        gtt_signal=_window_signal(idx, 200, 260),
    )
    assert r.leaps_ledger is not None
    closes = r.leaps_ledger.gtt_close_events
    assert len(closes) >= 1
    # The initial DITM contract has appreciated by day 200 -> positive gain -> tax.
    assert any(e.tax_paid > 0.0 for e in closes)
    for e in closes:
        assert e.tax_paid == pytest.approx(max(0.0, e.gain_realized) * 0.238, rel=1e-9)
        assert e.net_proceeds == pytest.approx(e.mtm_value - e.tax_paid, rel=1e-12)


def test_whipsaw_one_close_set_per_boundary() -> None:
    """Each Long->Defensive boundary produces exactly one close-set (one per date)."""
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    m = np.ones(len(idx), dtype=int)
    m[100:130] = 0
    m[250:280] = 0
    r = run_backtest(rd, pd_obj, _leaps_gtt_config(gtt=True), gtt_signal=_signal_from_mask(m, idx))
    assert r.leaps_ledger is not None
    close_dates = {e.close_date for e in r.leaps_ledger.gtt_close_events}
    assert len(close_dates) == 2  # two Long->Defensive transitions


def test_terminal_defensive_window_no_dangling_contracts() -> None:
    """A timeline ending in a defensive window closes out; no live contracts remain."""
    from finance.leverage import _live_contracts

    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    n = len(idx)
    r = run_backtest(
        rd, pd_obj, _leaps_only_config(AccountType.TAX_SHELTERED),
        gtt_signal=_window_signal(idx, n - 30, n),
    )
    assert r.leaps_ledger is not None
    assert len(r.leaps_ledger.gtt_close_events) >= 1
    assert _live_contracts(r.leaps_ledger, idx[-1]) == []


def test_whipsaw_multiple_reentries_restore_target() -> None:
    """After each defensive window the LEAPS leg is restored to target on re-entry."""
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    m = np.ones(len(idx), dtype=int)
    m[100:130] = 0
    m[250:280] = 0
    r = run_backtest(rd, pd_obj, _leaps_gtt_config(gtt=True), gtt_signal=_signal_from_mask(m, idx))
    # Re-entry days are 130 and 280; VTI_LEAPS restored to its 0.30 target.
    assert r.weight_history["VTI_LEAPS"].iloc[130] == pytest.approx(0.30, abs=1e-9)
    assert r.weight_history["VTI_LEAPS"].iloc[280] == pytest.approx(0.30, abs=1e-9)


def _make_pd_with_vix(vix_level: float = 0.25, n: int = 504) -> tuple[ReturnData, PriceData]:
    """(ReturnData, PriceData) with a constant '^VIX' column for dynamic-IV paths."""
    base = _make_price_data(n)
    vix = pd.DataFrame({"^VIX": vix_level}, index=base.prices.index)
    pd_obj = PriceData(
        prices=base.prices, dividends=base.dividends, vol_prices=vix,
        tickers=base.tickers, start_date=base.start_date,
        end_date=base.end_date, spliced=base.spliced,
    )
    return build_return_data(pd_obj, apply_tey=False), pd_obj


def test_gtt_leaps_with_dynamic_vix_closes_and_reenters() -> None:
    """With a '^VIX' column, force-close and re-entry use VIX-driven IV (>= floor)."""
    rd, pd_obj = _make_pd_with_vix(vix_level=0.25)
    idx = pd.DatetimeIndex(rd.returns.index)
    r = run_backtest(
        rd, pd_obj, _leaps_gtt_config(gtt=True), gtt_signal=_window_signal(idx, 200, 250)
    )
    assert r.leaps_ledger is not None
    assert len(r.leaps_ledger.gtt_close_events) >= 1
    # Re-entry restores the LEAPS leg to target.
    assert r.weight_history["VTI_LEAPS"].iloc[250] == pytest.approx(0.30, abs=1e-9)
    # VTI_LEAPS is flat through the defensive window.
    assert r.weight_history["VTI_LEAPS"].iloc[200:250].abs().max() == 0.0


def test_reentry_return_not_double_counted() -> None:
    """On the re-entry day the daily return must not double-count LEAPS value.

    Before the fix, leaps_value was computed from the old per-window ledger whose
    gtt_close_events is () — so _live_contracts returned already-force-closed
    contracts as still live.  Their economic value was already in leaps_pool, so
    nav_before_contrib double-counted them and port_return spiked artificially.

    The NAV series is self-consistent (it is computed after the re-entry rebalance
    correctly zeros leaps_pool and re-prices from the fresh ledger), so the test
    verifies that the compounded return series and the NAV series agree: compounding
    initial_nav by every (1 + return[t]) must reproduce nav[-1].  Uses a
    zero-contribution config so contributions do not appear in return_series.

    Anchor: prev_total_nav is initialized to initial_nav, so
    (1+r[0]) = nav[0]/initial_nav, and initial_nav * prod(1+r[t]) = nav[-1].
    """
    rd, pd_obj = _make_rd_and_pd(504)
    idx = pd.DatetimeIndex(rd.returns.index)
    r = run_backtest(
        rd, pd_obj, _leaps_only_config(AccountType.TAX_SHELTERED),
        gtt_signal=_window_signal(idx, 200, 250),
    )
    initial_nav = 1_000_000.0  # matches _leaps_only_config
    compounded_nav = initial_nav * float((1.0 + r.return_series).prod())
    # Without the fix this differs by the double-counted LEAPS value on re-entry.
    assert r.nav_series.iloc[-1] == pytest.approx(compounded_nav, rel=1e-9)


def test_reentry_creation_iv_matches_mtm_iv_no_gap() -> None:
    """Re-entry day leaps_value must equal the capital deployed (no IV gap).

    Before the fix, re-entry MTM used the 30-day smoothed VIX while contracts
    were created with raw VIX.  After a defensive window with elevated VIX,
    smoothed IV >> raw IV on re-entry, so leaps_value > capital_deployed,
    inflating total_nav and creating a spurious negative return spike the
    following day as the smoothed IV decayed.

    Setup: VIX is elevated (0.50) through the defensive window and drops to
    normal (0.20) on re-entry.  The 30-day rolling mean on re-entry day still
    reflects the elevated window, so smoothed IV >> raw IV.  The fix prices
    freshly created contracts at creation IV, making leaps_value == capital.
    """
    base = _make_price_data(n=504)
    idx = pd.DatetimeIndex(base.prices.index)
    lo, hi = 200, 250  # defensive window

    # VIX elevated during defensive window, drops to normal at re-entry
    vix_vals = np.full(len(idx), 0.20)
    vix_vals[lo:hi] = 0.50
    vix = pd.DataFrame({"VTI": vix_vals}, index=idx)
    pd_obj = PriceData(
        prices=base.prices, dividends=base.dividends, vol_prices=vix,
        tickers=base.tickers, start_date=base.start_date,
        end_date=base.end_date, spliced=base.spliced,
    )
    rd = build_return_data(pd_obj, apply_tey=False)

    r = run_backtest(
        rd, pd_obj, _leaps_only_config(AccountType.TAX_SHELTERED),
        gtt_signal=_window_signal(idx, lo, hi),
    )
    # On re-entry day (hi) the LEAPS weight must equal leaps_fraction (0.5)
    # within floating-point tolerance.  A valuation gap would push it away
    # from target and cause a spike in return_series.iloc[hi + 1].
    assert r.weight_history["VTI_LEAPS"].iloc[hi] == pytest.approx(0.5, abs=1e-6)
    # The day after re-entry must not show an anomalously large return.
    assert abs(r.return_series.iloc[hi + 1]) < 0.10


def test_gtt_leaps_never_long_opens_no_contracts() -> None:
    """An all-Defensive mask with a LEAPS carve-out opens no contracts."""
    rd, pd_obj = _make_rd_and_pd(252)
    idx = pd.DatetimeIndex(rd.returns.index)
    all_defensive = _signal_from_mask(np.zeros(len(idx), dtype=int), idx)
    r = run_backtest(rd, pd_obj, _leaps_gtt_config(gtt=True), gtt_signal=all_defensive)
    assert r.leaps_ledger is not None
    assert r.leaps_ledger.contracts == ()
    assert r.leaps_ledger.gtt_close_events == ()
    # VTI and VTI_LEAPS are flat for the entire (all-defensive) timeline.
    assert r.weight_history["VTI"].abs().max() == 0.0
    assert r.weight_history["VTI_LEAPS"].abs().max() == 0.0
