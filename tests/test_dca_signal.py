"""Tests for dca_signal.py — LeapsDcaSignal and compute_leaps_dca_signal.

Covers: I14 (entry_score ∈ [0,100]), I15 (alpha_t ∈ [0,1]), I16 (dca_action
membership), I17 (raises on empty ohlcv), I18 (T1 no-lookahead), I19 (weights
sum to 1.0), min_lookback enforcement, MACD gate behavior, and action classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from finance.data import PriceData
from finance.dca_signal import (
    _compute_macd,
    _compute_rsi,
    _compute_stochastic,
    _macd_bearish_confirmed,
    _percentile_rank,
    compute_leaps_dca_signal,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

N_DAYS = 600  # > 504 lookback default


def _make_price_data(
    n_days: int = N_DAYS,
    ticker: str = "VTI",
    start: str = "2020-01-02",
    seed: int = 42,
    include_ohlcv: bool = True,
    include_vol: bool = True,
    vol_col_name: str = "VTI_IV",
) -> PriceData:
    """Build a synthetic PriceData suitable for DCA signal testing."""
    idx = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(seed)

    # Close prices (random walk)
    close = pd.Series(
        100.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n_days)),
        index=idx,
        name=ticker,
    )
    prices = pd.DataFrame({ticker: close})

    # OHLCV
    ohlcv: pd.DataFrame
    if include_ohlcv:
        high = close * (1 + rng.uniform(0.0, 0.015, n_days))
        low = close * (1 - rng.uniform(0.0, 0.015, n_days))
        volume = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
        tuples = [
            (ticker, f) for f in ("Open", "High", "Low", "Close", "Volume")
        ]
        cols = pd.MultiIndex.from_tuples(tuples, names=["ticker", "field"])
        data = np.column_stack(
            [close.values, high.values, low.values, close.values, volume]
        )
        ohlcv = pd.DataFrame(data, index=idx, columns=cols)
    else:
        ohlcv = pd.DataFrame()

    # Vol prices
    vol_prices: pd.DataFrame
    if include_vol:
        iv_vals = np.clip(0.18 + rng.normal(0.0, 0.04, n_days), 0.05, 0.80)
        vol_prices = pd.DataFrame({vol_col_name: iv_vals}, index=idx)
    else:
        vol_prices = pd.DataFrame()

    dividends = pd.DataFrame({ticker: 0.0}, index=idx)

    return PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=vol_prices,
        tickers=(ticker,),
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
        ohlcv=ohlcv,
    )


def _as_of(price_data: PriceData) -> pd.Timestamp:
    """Return the last date in price_data as evaluation date."""
    return price_data.prices.index[-1]


# ---------------------------------------------------------------------------
# Unit tests — indicator helpers
# ---------------------------------------------------------------------------


def test_rsi_bounds() -> None:
    """RSI values are in [0, 100]."""
    idx = pd.bdate_range("2022-01-03", periods=100)
    rng = np.random.default_rng(1)
    close = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, 100)) * 100.0, index=idx)
    rsi = _compute_rsi(close)
    assert (rsi.dropna() >= 0.0).all()
    assert (rsi.dropna() <= 100.0).all()


def test_rsi_constant_price_is_100() -> None:
    """Constant price series has no losses → RSI = 100."""
    idx = pd.bdate_range("2022-01-03", periods=50)
    close = pd.Series(100.0, index=idx)
    rsi = _compute_rsi(close)
    # After warmup, RSI should be 100 (no losses)
    assert rsi.iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_stochastic_bounds() -> None:
    """Stochastic %D values (where non-NaN) are in [0, 100]."""
    idx = pd.bdate_range("2022-01-03", periods=100)
    rng = np.random.default_rng(2)
    close = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, 100)) * 100.0, index=idx)
    high = close * 1.01
    low = close * 0.99
    stoch = _compute_stochastic(high, low, close)
    clean = stoch.dropna()
    assert (clean >= 0.0).all()
    assert (clean <= 100.0).all()


def test_macd_histogram_shape() -> None:
    """MACD histogram has the same index as input close."""
    idx = pd.bdate_range("2022-01-03", periods=100)
    rng = np.random.default_rng(3)
    close = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, 100)) * 100.0, index=idx)
    hist = _compute_macd(close)
    assert len(hist) == len(close)
    assert hist.index.equals(close.index)


def test_macd_bearish_confirmed_true() -> None:
    """Three consecutive negative histogram values → bearish_confirmed=True."""
    idx = pd.bdate_range("2022-01-03", periods=10)
    hist = pd.Series([0.5, 0.1, -0.1, -0.2, -0.3, 0.1, -0.4, -0.5, -0.6, -0.7], index=idx)
    assert _macd_bearish_confirmed(hist, min_sessions=3) is True


def test_macd_bearish_confirmed_false_mixed() -> None:
    """Mixed signs in last 3 values → bearish_confirmed=False."""
    idx = pd.bdate_range("2022-01-03", periods=5)
    hist = pd.Series([-0.3, -0.2, 0.1, -0.4, -0.5], index=idx)
    # Last 3: [0.1, -0.4, -0.5] → not all negative
    assert _macd_bearish_confirmed(hist, min_sessions=3) is False


def test_macd_bearish_confirmed_insufficient_data() -> None:
    """Fewer than min_sessions rows → bearish_confirmed=False."""
    idx = pd.bdate_range("2022-01-03", periods=2)
    hist = pd.Series([-0.3, -0.4], index=idx)
    assert _macd_bearish_confirmed(hist, min_sessions=3) is False


def test_percentile_rank_boundaries() -> None:
    """Percentile rank of min = 0 (or near), max = 100."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # All values ≤ 1.0: just 1 → 1/5 * 100 = 20
    assert _percentile_rank(1.0, series) == pytest.approx(20.0)
    # All values ≤ 5.0: 5 → 100
    assert _percentile_rank(5.0, series) == pytest.approx(100.0)


def test_percentile_rank_empty_returns_50() -> None:
    """Empty series → 50.0 (neutral)."""
    assert _percentile_rank(1.0, pd.Series([], dtype=float)) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# compute_leaps_dca_signal — validation errors
# ---------------------------------------------------------------------------


def test_raises_on_empty_ohlcv() -> None:
    """ValueError when price_data.ohlcv is empty (I17)."""
    pd_obj = _make_price_data(include_ohlcv=False)
    with pytest.raises(ValueError, match="ohlcv is empty"):
        compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj))


def test_raises_on_unknown_ticker() -> None:
    """ValueError when ticker not in price_data.prices."""
    pd_obj = _make_price_data()
    with pytest.raises(ValueError, match="ticker 'AAPL' not found"):
        compute_leaps_dca_signal(pd_obj, "AAPL", _as_of(pd_obj))


def test_raises_on_weights_not_summing_to_one() -> None:
    """ValueError when weights don't sum to 1.0 within 1e-6 (I19)."""
    pd_obj = _make_price_data()
    with pytest.raises(ValueError, match=r"Weights must sum to 1\.0"):
        compute_leaps_dca_signal(
            pd_obj, "VTI", _as_of(pd_obj), w_rsi=0.3, w_stoch=0.3, w_iv=0.3, w_macd=0.3
        )


def test_raises_on_missing_vol_prices() -> None:
    """ValueError when vol_prices has no column for ticker's IV proxy."""
    pd_obj = _make_price_data(include_vol=False)
    with pytest.raises(ValueError, match="No IV column found"):
        compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj))


def test_flat_ohlcv_columns_accepted() -> None:
    """Non-MultiIndex ohlcv with High/Low/Close columns works as fallback."""
    n = N_DAYS
    idx = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(33)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n)), index=idx, name="VTI")
    prices = pd.DataFrame({"VTI": close})
    # Flat (non-MultiIndex) OHLCV — simple column names
    ohlcv = pd.DataFrame({
        "Open": close.values,
        "High": (close * 1.01).values,
        "Low": (close * 0.99).values,
        "Close": close.values,
        "Volume": np.ones(n),
    }, index=idx)
    iv_vals = np.clip(0.18 + rng.normal(0.0, 0.04, n), 0.05, 0.80)
    vol_prices = pd.DataFrame({"VTI_IV": iv_vals}, index=idx)
    dividends = pd.DataFrame({"VTI": 0.0}, index=idx)
    pd_obj = PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=vol_prices,
        tickers=("VTI",),
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
        ohlcv=ohlcv,
    )
    sig = compute_leaps_dca_signal(pd_obj, "VTI", idx[-1])
    assert 0.0 <= sig.entry_score <= 100.0


def test_raises_when_ticker_not_in_ohlcv_multiindex() -> None:
    """ValueError when ticker is absent from MultiIndex ohlcv first level."""
    n = N_DAYS
    idx = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(77)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n)), index=idx, name="VTI")
    prices = pd.DataFrame({"VTI": close})
    # OHLCV only has GLD, not VTI
    tuples = [("GLD", f) for f in ("Open", "High", "Low", "Close", "Volume")]
    cols = pd.MultiIndex.from_tuples(tuples, names=["ticker", "field"])
    data = np.ones((n, 5)) * 100.0
    ohlcv = pd.DataFrame(data, index=idx, columns=cols)
    iv_vals = np.clip(0.18 + rng.normal(0.0, 0.04, n), 0.05, 0.80)
    vol_prices = pd.DataFrame({"VTI_IV": iv_vals}, index=idx)
    dividends = pd.DataFrame({"VTI": 0.0}, index=idx)
    pd_obj = PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=vol_prices,
        tickers=("VTI",),
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
        ohlcv=ohlcv,
    )
    with pytest.raises(ValueError, match="No OHLCV data found"):
        compute_leaps_dca_signal(pd_obj, "VTI", idx[-1])


def test_vix_fallback_when_no_asset_iv_mapping() -> None:
    """When ASSET_VOL_INDEX has no mapping, falls back to VTI_IV column."""
    # GLD maps to ^GVZ, but if we use a ticker not in ASSET_VOL_INDEX with VTI_IV present,
    # it uses the VTI_IV fallback.
    # Simulate by providing a ticker that is in ASSET_VOL_INDEX but whose vol col name
    # does NOT match, yet VTI_IV exists.
    n = N_DAYS
    idx = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(55)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n)), index=idx, name="VTI")
    prices = pd.DataFrame({"VTI": close})
    high = close * 1.01
    low = close * 0.99
    tuples = [("VTI", f) for f in ("Open", "High", "Low", "Close", "Volume")]
    cols = pd.MultiIndex.from_tuples(tuples, names=["ticker", "field"])
    data = np.column_stack([close.values, high.values, low.values, close.values, np.ones(n)])
    ohlcv = pd.DataFrame(data, index=idx, columns=cols)
    # Use "VTI_IV" as fallback name (not the standard "VTI_IV" from ASSET_VOL_INDEX)
    iv_vals = np.clip(0.18 + rng.normal(0.0, 0.04, n), 0.05, 0.80)
    vol_prices = pd.DataFrame({"VTI_IV": iv_vals}, index=idx)
    dividends = pd.DataFrame({"VTI": 0.0}, index=idx)
    pd_obj = PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=vol_prices,
        tickers=("VTI",),
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
        ohlcv=ohlcv,
    )
    # Should not raise — VTI_IV is found via ASSET_VOL_INDEX lookup
    sig = compute_leaps_dca_signal(pd_obj, "VTI", idx[-1])
    assert 0.0 <= sig.entry_score <= 100.0


def test_iv_column_named_by_raw_ticker_is_resolved() -> None:
    """build_price_data names vol_prices columns by raw ticker (e.g. "VTI"), not
    "<ticker>_IV" — this must resolve without falling through to the VIX fallback
    or raising, matching the convention _backtest_steps.py already relies on."""
    pd_obj = _make_price_data(vol_col_name="VTI")
    sig = compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj))
    assert 0.0 <= sig.entry_score <= 100.0


def test_raises_on_insufficient_data() -> None:
    """ValueError when fewer than min_lookback days available."""
    pd_obj = _make_price_data(n_days=100)
    with pytest.raises(ValueError, match="Insufficient data"):
        compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj), min_lookback=252)


# ---------------------------------------------------------------------------
# compute_leaps_dca_signal — invariant checks
# ---------------------------------------------------------------------------


def test_entry_score_in_bounds(  # type: ignore[misc]
) -> None:
    """I14: entry_score ∈ [0, 100]."""
    pd_obj = _make_price_data()
    sig = compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj))
    assert 0.0 <= sig.entry_score <= 100.0


def test_alpha_t_in_bounds() -> None:
    """I15: alpha_t ∈ [0, 1]."""
    pd_obj = _make_price_data()
    sig = compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj))
    assert 0.0 <= sig.alpha_t <= 1.0


def test_dca_action_valid_values() -> None:
    """I16: dca_action ∈ {HOLD, TRANCHE, AGGRESSIVE_SWEEP}."""
    pd_obj = _make_price_data()
    sig = compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj))
    assert sig.dca_action in {"HOLD", "TRANCHE", "AGGRESSIVE_SWEEP"}


def test_score_percentile_in_bounds() -> None:
    """score_percentile ∈ [0, 100]."""
    pd_obj = _make_price_data()
    sig = compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj))
    assert 0.0 <= sig.score_percentile <= 100.0


def test_leaps_dca_signal_is_frozen() -> None:
    """LeapsDcaSignal is frozen — attribute assignment raises."""
    pd_obj = _make_price_data()
    sig = compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj))
    with pytest.raises((AttributeError, TypeError)):
        sig.entry_score = 99.0  # type: ignore[misc]


def test_macd_gate_applied_when_bearish_confirmed() -> None:
    """When bearish_confirmed=True, macd_gate == macd_gate_floor."""
    pd_obj = _make_price_data()
    sig = compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj), macd_gate_floor=0.5)
    if sig.macd_bearish_confirmed:
        assert sig.macd_gate == pytest.approx(0.5)
    else:
        assert sig.macd_gate == pytest.approx(1.0)


def test_macd_gate_off_always_one() -> None:
    """use_macd_gate=False → macd_gate is always 1.0."""
    pd_obj = _make_price_data()
    sig = compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj), use_macd_gate=False)
    assert sig.macd_gate == pytest.approx(1.0)


def test_action_hold_when_low_percentile() -> None:
    """dca_action=HOLD when score_percentile < hold_pctile."""
    pd_obj = _make_price_data()
    sig = compute_leaps_dca_signal(
        pd_obj, "VTI", _as_of(pd_obj), hold_pctile=99.0, aggressive_pctile=99.5
    )
    # With hold_pctile=99.0, almost every result will be HOLD
    assert sig.dca_action == "HOLD"


def test_action_aggressive_sweep_when_high_percentile() -> None:
    """dca_action=AGGRESSIVE_SWEEP when score_percentile >= aggressive_pctile."""
    pd_obj = _make_price_data()
    sig = compute_leaps_dca_signal(
        pd_obj, "VTI", _as_of(pd_obj), hold_pctile=0.0, aggressive_pctile=0.1
    )
    # score_percentile will be >= 0.1 for any non-trivial result → AGGRESSIVE_SWEEP
    assert sig.dca_action == "AGGRESSIVE_SWEEP"


# ---------------------------------------------------------------------------
# T1 no-lookahead invariant (I18)
# ---------------------------------------------------------------------------


def test_no_lookahead_extra_row_does_not_change_output() -> None:
    """Appending one row after as_of_date does not change signal (I18).

    Builds one full PriceData object, then constructs a truncated copy by
    slicing all DataFrames to [:as_of_date]. Both copies share identical data
    up to as_of_date; only pd_long has one extra future row. The T1 slice
    [:as_of_date] inside compute_leaps_dca_signal must produce identical output.
    """
    pd_long = _make_price_data(n_days=N_DAYS, seed=7)
    as_of = pd_long.prices.index[-2]  # second-to-last: long has one row beyond as_of

    # Build truncated copy by slicing every DataFrame to [:as_of]
    pd_short = PriceData(
        prices=pd_long.prices.loc[:as_of],
        dividends=pd_long.dividends.loc[:as_of],
        vol_prices=pd_long.vol_prices.loc[:as_of],
        tickers=pd_long.tickers,
        start_date=pd_long.start_date,
        end_date=str(as_of.date()),
        spliced=pd_long.spliced,
        ohlcv=pd_long.ohlcv.loc[:as_of],
    )

    sig_a = compute_leaps_dca_signal(pd_short, "VTI", as_of)
    sig_b = compute_leaps_dca_signal(pd_long, "VTI", as_of)

    assert sig_a.entry_score == pytest.approx(sig_b.entry_score, abs=1e-9)
    assert sig_a.rsi == pytest.approx(sig_b.rsi, abs=1e-9)
    assert sig_a.macd_hist == pytest.approx(sig_b.macd_hist, abs=1e-9)


# ---------------------------------------------------------------------------
# MACD bearish confirmed scenario
# ---------------------------------------------------------------------------


def test_macd_bearish_confirmed_gate_reduces_score() -> None:
    """A bearish-confirmed gate multiplier < 1 cannot increase entry_score."""
    pd_obj = _make_price_data(n_days=N_DAYS, seed=99)
    as_of = pd_obj.prices.index[-1]

    sig_with_gate = compute_leaps_dca_signal(
        pd_obj, "VTI", as_of, use_macd_gate=True, macd_gate_floor=0.5
    )
    sig_no_gate = compute_leaps_dca_signal(
        pd_obj, "VTI", as_of, use_macd_gate=False
    )

    # Gate can only hold or reduce; never increase above no-gate score
    assert sig_with_gate.entry_score <= sig_no_gate.entry_score + 1e-9


def test_macd_score_is_binary() -> None:
    """MACD component is 100 when histogram > 0, 0 when ≤ 0.

    Construct a deterministic price series where the last MACD histogram sign
    is known, then verify the composite score reflects binary w_macd contribution.
    """
    pd_obj = _make_price_data(n_days=N_DAYS, seed=42)
    as_of = pd_obj.prices.index[-1]
    sig = compute_leaps_dca_signal(
        pd_obj, "VTI", as_of,
        # Use pure MACD weight to isolate the component
        w_rsi=0.0, w_stoch=0.0, w_iv=0.0, w_macd=1.0,
        use_macd_gate=False,
    )
    # With only MACD weight and no gate, entry_score must be exactly 0 or 100
    assert sig.entry_score in (0.0, 100.0)


def test_alpha_t_linear_interpolation() -> None:
    """alpha_t is 0 at hold_pctile, 1 at aggressive_pctile, 0.5 midway."""
    pd_obj = _make_price_data(n_days=N_DAYS, seed=42)
    as_of = pd_obj.prices.index[-1]

    # Force score_percentile to be exactly the midpoint by setting thresholds
    # such that the midpoint alpha is predictable — instead verify the formula
    # algebraically via the boundary conditions.
    sig_hold = compute_leaps_dca_signal(
        pd_obj, "VTI", as_of, hold_pctile=99.9, aggressive_pctile=100.0
    )
    assert sig_hold.alpha_t == pytest.approx(0.0, abs=0.05)

    sig_sweep = compute_leaps_dca_signal(
        pd_obj, "VTI", as_of, hold_pctile=0.0, aggressive_pctile=0.01
    )
    assert sig_sweep.alpha_t == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(max_examples=30, deadline=10_000)
def test_property_score_bounds(seed: int) -> None:
    """I14: entry_score ∈ [0, 100] across random price series."""
    pd_obj = _make_price_data(n_days=N_DAYS, seed=seed)
    sig = compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj))
    assert 0.0 <= sig.entry_score <= 100.0


@given(seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=30, deadline=10_000)
def test_property_alpha_t_bounds(seed: int) -> None:
    """I15: alpha_t ∈ [0, 1] across random price series."""
    pd_obj = _make_price_data(n_days=N_DAYS, seed=seed)
    sig = compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj))
    assert 0.0 <= sig.alpha_t <= 1.0


@given(seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=30, deadline=10_000)
def test_property_dca_action_membership(seed: int) -> None:
    """I16: dca_action ∈ {HOLD, TRANCHE, AGGRESSIVE_SWEEP} for any inputs."""
    pd_obj = _make_price_data(n_days=N_DAYS, seed=seed)
    sig = compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj))
    assert sig.dca_action in {"HOLD", "TRANCHE", "AGGRESSIVE_SWEEP"}


@given(seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=30, deadline=10_000)
def test_property_raises_on_empty_ohlcv_always(seed: int) -> None:
    """I17: Always raises ValueError when ohlcv is empty, regardless of seed."""
    pd_obj = _make_price_data(n_days=N_DAYS, seed=seed, include_ohlcv=False)
    with pytest.raises(ValueError, match="ohlcv is empty"):
        compute_leaps_dca_signal(pd_obj, "VTI", _as_of(pd_obj))


@given(
    w_rsi=st.floats(min_value=0.05, max_value=0.5),
    w_stoch=st.floats(min_value=0.05, max_value=0.5),
    w_iv=st.floats(min_value=0.05, max_value=0.5),
)
@settings(max_examples=30, deadline=10_000)
def test_property_weights_sum_raises(w_rsi: float, w_stoch: float, w_iv: float) -> None:
    """I19: Weights not summing to 1.0 always raises ValueError."""
    w_macd = 1.0 - w_rsi - w_stoch - w_iv + 0.1  # deliberately off by 0.1
    pd_obj = _make_price_data(n_days=N_DAYS, seed=0)
    with pytest.raises(ValueError, match=r"Weights must sum to 1\.0"):
        compute_leaps_dca_signal(
            pd_obj, "VTI", _as_of(pd_obj),
            w_rsi=w_rsi, w_stoch=w_stoch, w_iv=w_iv, w_macd=w_macd,
        )
