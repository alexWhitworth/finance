"""LEAPS DCA entry signal — multi-factor composite score for DITM LEAPS cost-averaging.

All logic is pure (no I/O). Inputs arrive as slices of PriceData; the T1
no-lookahead invariant is enforced by slicing every series to [:as_of_date]
at the top of compute_leaps_dca_signal before any rolling window runs.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance.consts import ASSET_VOL_INDEX


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeapsDcaSignal:
    """Multi-factor composite entry score for DITM LEAPS DCA timing.

    All fields are computed as of as_of_date using only data up to (and
    including) that date (T1 / I18 invariant).

    Attributes:
        as_of_date: Evaluation date.
        ticker: Underlying ticker evaluated (e.g. VTI).
        entry_score: Composite score in [0, 100]. Higher = more favorable entry.
        score_percentile: entry_score rank within the lookback window (0–100).
        alpha_t: Tranche allocation fraction in [0, 1].
        dca_action: HOLD | TRANCHE | AGGRESSIVE_SWEEP.
        rsi: 14-day RSI at as_of_date.
        stoch_d: 5/3 Stochastic %D at as_of_date.
        iv_percentile: 252-day IV percentile rank (0–100).
        iv_current: Raw IV (decimal) at as_of_date.
        macd_hist: MACD histogram value at as_of_date.
        macd_bearish_confirmed: True if MACD histogram negative for ≥ 3 consecutive sessions.
        macd_gate: Gate multiplier applied (1.0 or macd_gate_floor).
    """

    as_of_date: pd.Timestamp
    ticker: str
    entry_score: float
    score_percentile: float
    alpha_t: float
    dca_action: str
    rsi: float
    stoch_d: float
    iv_percentile: float
    iv_current: float
    macd_hist: float
    macd_bearish_confirmed: bool
    macd_gate: float


# ---------------------------------------------------------------------------
# Technical indicator helpers (pure functions)
# ---------------------------------------------------------------------------


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI over a close price series.

    Arguments:
        close: Close price Series (DatetimeIndex).
        window: Lookback window in trading days.

    Returns:
        RSI Series in [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder's smoothing: EWM with alpha = 1/window
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(100.0)  # avg_loss == 0 → no losses → RSI = 100


def _compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 5,
    d_window: int = 3,
) -> pd.Series:
    """Fast Stochastic %D (SMA of %K).

    Arguments:
        high: High price Series.
        low: Low price Series.
        close: Close price Series.
        k_window: Lookback for %K computation.
        d_window: SMA window for %D smoothing.

    Returns:
        %D Series in [0, 100]. NaN for the first k_window + d_window - 2 rows.
    """
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    denom = (highest_high - lowest_low).replace(0.0, np.nan)
    pct_k = (close - lowest_low) / denom * 100.0
    pct_d: pd.Series = pct_k.rolling(window=d_window).mean()
    return pct_d


def _compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_window: int = 9,
) -> pd.Series:
    """MACD histogram (MACD line − signal line).

    Arguments:
        close: Close price Series.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal_window: Signal line EMA period.

    Returns:
        MACD histogram Series. Positive = bullish, negative = bearish.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line - signal_line


def _macd_bearish_confirmed(macd_hist: pd.Series, min_sessions: int = 3) -> bool:
    """True if the last min_sessions MACD histogram values are all negative.

    Arguments:
        macd_hist: Full MACD histogram Series up to as_of_date.
        min_sessions: Required consecutive negative sessions.

    Returns:
        True when at least min_sessions consecutive negative values end the series.
    """
    recent = macd_hist.dropna().iloc[-min_sessions:]
    if len(recent) < min_sessions:
        return False
    return bool((recent < 0.0).all())


def _percentile_rank(value: float, series: pd.Series) -> float:
    """Percentile rank of value within series (0–100, inclusive).

    Arguments:
        value: The value to rank.
        series: Reference distribution (NaN values excluded).

    Returns:
        Percentile rank in [0, 100].
    """
    clean = series.dropna()
    if clean.empty:
        return 50.0
    return float((clean <= value).sum() / len(clean) * 100.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_leaps_dca_signal(
    price_data: object,
    ticker: str,
    as_of_date: pd.Timestamp,
    hold_pctile: float = 25.0,
    aggressive_pctile: float = 75.0,
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal_window: int = 9,
    stoch_k: int = 5,
    stoch_d: int = 3,
    iv_window: int = 252,
    lookback: int = 504,
    min_lookback: int = 252,
    w_rsi: float = 0.20,
    w_stoch: float = 0.15,
    w_iv: float = 0.35,
    w_macd: float = 0.30,
    use_macd_gate: bool = True,
    macd_gate_floor: float = 0.5,
) -> "LeapsDcaSignal":
    """Compute the multi-factor DITM LEAPS DCA entry signal for a single ticker.

    Slices all series to [:as_of_date] before any computation to enforce the
    T1 no-lookahead invariant (I18).

    Arguments:
        price_data: PriceData instance with prices, ohlcv, and vol_prices.
        ticker: Underlying ticker to evaluate (e.g. "VTI").
        as_of_date: Evaluation date.
        hold_pctile: Score percentile below which dca_action = HOLD.
        aggressive_pctile: Score percentile at or above which dca_action = AGGRESSIVE_SWEEP.
        rsi_window: RSI lookback window in trading days.
        macd_fast: MACD fast EMA period.
        macd_slow: MACD slow EMA period.
        macd_signal_window: MACD signal line EMA period.
        stoch_k: Stochastic %K lookback window.
        stoch_d: Stochastic %D smoothing window.
        iv_window: IV percentile lookback window in trading days.
        lookback: Score history window for percentile ranking (trading days).
        min_lookback: Minimum required trading days up to as_of_date.
        w_rsi: RSI component weight.
        w_stoch: Stochastic component weight.
        w_iv: IV component weight.
        w_macd: MACD component weight.
        use_macd_gate: Apply MACD gate multiplier when bearish confirmed.
        macd_gate_floor: Gate multiplier when bearish confirmed and use_macd_gate=True.

    Returns:
        LeapsDcaSignal with all factor values and composite score.

    Raises:
        ValueError: If ticker not in price_data.prices.
        ValueError: If price_data.ohlcv is empty (fetch_ohlcv not requested).
        ValueError: If price_data.vol_prices has no column for ticker's IV proxy.
        ValueError: If w_rsi + w_stoch + w_iv + w_macd does not sum to 1.0 within 1e-6.
        ValueError: If fewer than min_lookback trading days available up to as_of_date.
    """
    from finance.data import PriceData

    pd_obj: PriceData = price_data  # type: ignore[assignment]

    # Validate weight sum
    weight_sum = w_rsi + w_stoch + w_iv + w_macd
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0 within 1e-6; got {weight_sum:.8f}"
        )

    # Validate ticker present
    if ticker not in pd_obj.prices.columns:
        raise ValueError(f"ticker '{ticker}' not found in price_data.prices")

    # Validate ohlcv populated
    if pd_obj.ohlcv.empty:
        raise ValueError(
            "price_data.ohlcv is empty; build_price_data must be called with fetch_ohlcv=True"
        )

    # T1: slice everything to [:as_of_date]
    close_all: pd.Series = pd_obj.prices[ticker].loc[:as_of_date]

    if len(close_all) < min_lookback:
        raise ValueError(
            f"Insufficient data: {len(close_all)} trading days available up to "
            f"{as_of_date.date()}, need at least {min_lookback}"
        )

    # Slice OHLCV for stochastic — handle MultiIndex (ticker, field) structure
    if isinstance(pd_obj.ohlcv.columns, pd.MultiIndex):
        ohlcv_ticker = pd_obj.ohlcv[ticker].loc[:as_of_date] if ticker in pd_obj.ohlcv.columns.get_level_values(0) else pd.DataFrame()
    else:
        ohlcv_ticker = pd_obj.ohlcv.loc[:as_of_date]

    if ohlcv_ticker.empty:
        raise ValueError(
            f"No OHLCV data found for ticker '{ticker}' in price_data.ohlcv"
        )

    high = ohlcv_ticker["High"] if "High" in ohlcv_ticker.columns else close_all
    low = ohlcv_ticker["Low"] if "Low" in ohlcv_ticker.columns else close_all
    close = ohlcv_ticker["Close"] if "Close" in ohlcv_ticker.columns else close_all

    # Resolve IV series: prefer ASSET_VOL_INDEX mapping, fallback to VIX, then raise
    vol_col: str | None = None
    mapped_index = ASSET_VOL_INDEX.get(ticker)
    if mapped_index is not None:
        iv_col_name = f"{ticker}_IV"
        if iv_col_name in pd_obj.vol_prices.columns:
            vol_col = iv_col_name
    if vol_col is None:
        # Fallback: any VIX-related column
        vix_cols = [c for c in pd_obj.vol_prices.columns if "VIX" in c.upper() or "VTI_IV" in c]
        if vix_cols:
            vol_col = vix_cols[0]
    if vol_col is None:
        raise ValueError(
            f"No IV column found for ticker '{ticker}' in price_data.vol_prices. "
            f"Available columns: {list(pd_obj.vol_prices.columns)}"
        )

    iv_series_all: pd.Series = pd_obj.vol_prices[vol_col].loc[:as_of_date]

    # -----------------------------------------------------------------------
    # Compute technical indicators
    # -----------------------------------------------------------------------

    # RSI
    rsi_series = _compute_rsi(close, window=rsi_window)
    rsi_val = float(rsi_series.iloc[-1])

    # Stochastic %D
    stoch_d_series = _compute_stochastic(high, low, close, k_window=stoch_k, d_window=stoch_d)
    stoch_d_clean = stoch_d_series.dropna()
    stoch_d_val = float(stoch_d_clean.iloc[-1]) if not stoch_d_clean.empty else 50.0

    # IV current and IV percentile
    iv_series_clean = iv_series_all.dropna()
    iv_current = float(iv_series_clean.iloc[-1]) if not iv_series_clean.empty else 0.0
    iv_window_series = iv_series_clean.iloc[-iv_window:] if len(iv_series_clean) >= iv_window else iv_series_clean
    iv_pct = _percentile_rank(iv_current, iv_window_series)

    # MACD histogram
    macd_hist_series = _compute_macd(close, fast=macd_fast, slow=macd_slow, signal_window=macd_signal_window)
    macd_hist_val = float(macd_hist_series.iloc[-1])

    # MACD bearish confirmed
    bearish = _macd_bearish_confirmed(macd_hist_series, min_sessions=3)
    gate = macd_gate_floor if (use_macd_gate and bearish) else 1.0

    # -----------------------------------------------------------------------
    # Factor scores in [0, 100] (higher = better entry)
    # -----------------------------------------------------------------------

    # RSI: oversold (low RSI) = good → score = 100 - rsi
    score_rsi = float(np.clip(100.0 - rsi_val, 0.0, 100.0))

    # Stochastic: oversold (low %D) = good → score = 100 - stoch_d
    score_stoch = float(np.clip(100.0 - stoch_d_val, 0.0, 100.0))

    # IV: low IV (cheap options) = good → score = 100 - iv_pct
    score_iv = float(np.clip(100.0 - iv_pct, 0.0, 100.0))

    # MACD: positive histogram (bullish trend) = 100; negative = 0 (binary regime signal)
    score_macd = 100.0 if macd_hist_val > 0.0 else 0.0

    # Composite score (before gate)
    raw_score = w_rsi * score_rsi + w_stoch * score_stoch + w_iv * score_iv + w_macd * score_macd
    entry_score = float(np.clip(raw_score, 0.0, 100.0) * gate)

    # -----------------------------------------------------------------------
    # Score percentile over lookback window of entry scores
    # -----------------------------------------------------------------------

    # Build rolling entry score series for lookback window
    # We need enough history to compute the lookback distribution.
    close_lookback = close.iloc[-lookback:]
    high_lookback = high.iloc[-lookback:]
    low_lookback = low.iloc[-lookback:]
    iv_lookback_full = iv_series_all.reindex(close_lookback.index, method="ffill")
    macd_lookback_hist = _compute_macd(
        close_lookback, fast=macd_fast, slow=macd_slow, signal_window=macd_signal_window
    )

    rsi_lookback = _compute_rsi(close_lookback, window=rsi_window)
    stoch_lookback = _compute_stochastic(
        high_lookback, low_lookback, close_lookback, k_window=stoch_k, d_window=stoch_d
    )

    # Compute factor scores for each day in lookback
    iv_lookback_pcts = iv_lookback_full.rolling(window=iv_window, min_periods=1).apply(
        lambda s: (s <= s.iloc[-1]).sum() / len(s) * 100.0, raw=False
    )

    scores_rsi = (100.0 - rsi_lookback).clip(0.0, 100.0)
    scores_stoch = (100.0 - stoch_lookback).clip(0.0, 100.0).fillna(50.0)
    scores_iv = (100.0 - iv_lookback_pcts).clip(0.0, 100.0)
    # Binary: positive MACD histogram = bullish trend = 100; negative = 0
    scores_macd = (macd_lookback_hist > 0.0).astype(float) * 100.0

    # Gate for each day in lookback
    gates_series = pd.Series(1.0, index=close_lookback.index)
    if use_macd_gate:
        for i in range(3, len(macd_lookback_hist) + 1):
            recent_3 = macd_lookback_hist.iloc[i - 3 : i]
            if (recent_3 < 0.0).all():
                gates_series.iloc[i - 1] = macd_gate_floor

    raw_scores = (
        w_rsi * scores_rsi
        + w_stoch * scores_stoch
        + w_iv * scores_iv
        + w_macd * scores_macd
    )
    entry_scores_lookback = (raw_scores.clip(0.0, 100.0) * gates_series).dropna()

    score_pct = _percentile_rank(entry_score, entry_scores_lookback)

    # -----------------------------------------------------------------------
    # alpha_t and dca_action
    # -----------------------------------------------------------------------

    # Linear interpolation in [hold_pctile, aggressive_pctile] band; 0 below, 1 above
    band = aggressive_pctile - hold_pctile
    alpha_t = float(np.clip(
        (score_pct - hold_pctile) / band if band > 0.0 else 1.0,
        0.0,
        1.0,
    ))

    if score_pct < hold_pctile:
        dca_action = "HOLD"
    elif score_pct >= aggressive_pctile:
        dca_action = "AGGRESSIVE_SWEEP"
    else:
        dca_action = "TRANCHE"

    return LeapsDcaSignal(
        as_of_date=as_of_date,
        ticker=ticker,
        entry_score=entry_score,
        score_percentile=score_pct,
        alpha_t=alpha_t,
        dca_action=dca_action,
        rsi=rsi_val,
        stoch_d=stoch_d_val,
        iv_percentile=iv_pct,
        iv_current=iv_current,
        macd_hist=macd_hist_val,
        macd_bearish_confirmed=bearish,
        macd_gate=gate,
    )
