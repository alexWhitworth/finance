"""GTT (Growth Trend Timing) signal computation.

Pure signal functions plus the I/O boundary for FRED (UNRATE) fetching. The pure
functions here (compute_ue_signal, compute_vix_signal, compute_position_mask) take
plain pandas Series and return lag-adjusted 0/1 signals; all network I/O is isolated
in fetch_gtt_signal_data (Phase 3).

Signal timing model (see plans/implement_gtt.md §1):
  * UE_12M: FRED indexes UNRATE at the reference-month start, but the print is not
    public until the BLS Employment Situation release on the first Friday of the
    FOLLOWING month. compute_ue_signal re-stamps each observation to that first
    Friday before the daily forward-fill, eliminating a ~1-month look-ahead leak.
  * VIX_5D: known at close t (rolling window [t-4, t]).
  * A single close t -> open t+1 execution lag is applied once, in
    compute_position_mask, shared by both signals.
"""

from dataclasses import dataclass

import pandas as pd

from finance.consts import (
    GTT_SMA_WINDOW,
    GTT_UNRATE_TRADE_LAG_DAYS,
    GTT_VIX_CONSECUTIVE_DAYS,
)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GttSignalData:
    """Pre-computed, lag-adjusted GTT signals, directly consumable by run_backtest.

    Attributes:
        position_mask: DatetimeIndex -> int, 1=Long / 0=Defensive. Already 1-day
            lag-adjusted (signal at close t -> position at open t+1).
        ue_signal: DatetimeIndex -> int, 1=UE_12M active (UNRATE >= trailing 12M MA,
            publication-dated).
        vix_signal: DatetimeIndex -> int, 1=VIX_5D active (VIX >= threshold for N
            consecutive days).
        vix_p90_threshold: Threshold used, stored for reproducibility.
        unrate_start: Earliest date in the UNRATE series used.
        vix_start: Earliest date in the VIX series used.
    """

    position_mask: pd.Series
    ue_signal: pd.Series
    vix_signal: pd.Series
    vix_p90_threshold: float
    unrate_start: pd.Timestamp
    vix_start: pd.Timestamp


# ---------------------------------------------------------------------------
# Pure signal functions
# ---------------------------------------------------------------------------


def _first_friday_of_following_month(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the first Friday of the month following ts's month.

    This is the BLS Employment Situation release cadence: the rate for reference
    month M is published on the first Friday of month M+1.

    Arguments:
        ts: A timestamp within the reference month (FRED indexes at month start).

    Returns:
        Timestamp of the first Friday of the following month.
    """
    first_of_next = ts + pd.offsets.MonthBegin(1)
    # weekday(): Mon=0 .. Fri=4. Offset to the first Friday on/after the 1st.
    offset = (4 - first_of_next.weekday()) % 7
    return first_of_next + pd.Timedelta(days=offset)


def compute_ue_signal(
    unrate: pd.Series,
    rolling_window_months: int = 12,
) -> pd.Series:
    """Compute the daily UE_12M recession-risk signal from monthly UNRATE.

    Publication-date alignment (Option B — deterministic BLS cadence): each monthly
    UNRATE observation (FRED-indexed at the reference-month start) is re-stamped to
    its true publication date (the first Friday of the following month) before the
    daily forward-fill, so a month's rate is never visible before it is published.
    No execution shift is applied here; that single close->open lag lives in
    compute_position_mask (shared with VIX_5D).

    Arguments:
        unrate: Monthly UNRATE series (DatetimeIndex at reference-month start, float).
        rolling_window_months: Trailing moving-average window in months. Default 12.

    Raises:
        ValueError: If unrate is empty or has fewer than rolling_window_months
            observations.

    Returns:
        Daily int Series (0/1): 1 where the reference month's UNRATE was >= its
        trailing 12-month MA, publication-dated and forward-filled to calendar days.

    Notes:
        On the rare month BLS deviates from the first-Friday cadence (holiday weeks,
        shutdowns) this is an approximation of <= a few days; documented, not
        corrected. FRED month-start indexing is verified empirically in Phase 3.
    """
    if unrate.empty:
        raise ValueError("unrate is empty; cannot compute UE_12M signal")
    if len(unrate) < rolling_window_months:
        raise ValueError(
            f"unrate has {len(unrate)} observations; "
            f"need at least rolling_window_months ({rolling_window_months})"
        )

    # Trailing N-month MA on the monthly series; warm-up (< N obs) -> NaN -> flag 0.
    ma = unrate.rolling(window=rolling_window_months, min_periods=rolling_window_months).mean()
    monthly_flag = (unrate >= ma).astype(int)  # NaN comparison -> False -> 0

    # Re-stamp each monthly observation to its publication date, then daily ffill.
    pub_index = pd.DatetimeIndex(
        [_first_friday_of_following_month(ts) for ts in unrate.index]
    )
    publication_dated = pd.Series(monthly_flag.to_numpy(), index=pub_index).sort_index()
    daily = publication_dated.resample("D").ffill().astype(int)
    daily.name = "ue_signal"
    return daily


def compute_vix_signal(
    vix: pd.Series,
    threshold: float,
    consecutive_days: int = GTT_VIX_CONSECUTIVE_DAYS,
) -> pd.Series:
    """Compute the daily VIX_5D recession-risk signal.

    Fires 1 when VIX has been at or above threshold for N consecutive days. Since the
    per-day above-threshold flags are 0/1, a rolling window summing to exactly N means
    all N days in the window were above threshold (a genuine consecutive run).

    Arguments:
        vix: Daily VIX series as a decimal (e.g. 0.20), DatetimeIndex.
        threshold: P90 threshold decimal (e.g. 0.272).
        consecutive_days: Number of consecutive days at/above threshold to fire.
            Default GTT_VIX_CONSECUTIVE_DAYS (5).

    Raises:
        ValueError: If consecutive_days < 1 or vix is empty.

    Returns:
        Daily int Series (0/1), DatetimeIndex aligned to vix.
    """
    if consecutive_days < 1:
        raise ValueError(f"consecutive_days must be >= 1; got {consecutive_days}")
    if vix.empty:
        raise ValueError("vix is empty; cannot compute VIX_5D signal")

    above = (vix >= threshold).astype(int)
    run = above.rolling(window=consecutive_days, min_periods=consecutive_days).sum()
    signal = (run >= consecutive_days).astype(int)  # run maxes at N; >= N iff == N
    signal.name = "vix_signal"
    return signal


def compute_position_mask(
    ue_signal: pd.Series,
    vix_signal: pd.Series,
    equity_prices: pd.Series,
    sma_window: int = GTT_SMA_WINDOW,
) -> pd.Series:
    """Combine the recession-risk signals with the SMA trend filter and apply the lag.

    Decision rule (evaluated at close t, observed at open t+1):
        recession_risk = UE_12M OR VIX_5D
        Defensive (0) iff recession_risk AND price < SMA200; else Long (1).
    During SMA warm-up (NaN SMA) price is treated as above-SMA, so the mask stays Long.
    The single 1-trading-day shift here is the shared execution lag for both signals:
    ue_signal is already publication-dated (its Friday close -> next trading day) and
    vix_signal is known at close t (-> t+1).

    Arguments:
        ue_signal: Daily 0/1 UE_12M signal.
        vix_signal: Daily 0/1 VIX_5D signal.
        equity_prices: Daily equity (VTI) price series for the SMA filter.
        sma_window: SMA window in trading days. Default GTT_SMA_WINDOW (200).

    Raises:
        ValueError: If equity_prices is empty, or if neither signal's index overlaps
            equity_prices (the series cannot be aligned).

    Returns:
        Daily int Series (0/1), 1-day lag-adjusted, index-aligned to equity_prices.
    """
    if equity_prices.empty:
        raise ValueError("equity_prices is empty; cannot compute position mask")
    signal_index = ue_signal.index.union(vix_signal.index)
    if signal_index.intersection(equity_prices.index).empty:
        raise ValueError(
            "signal and price indexes do not overlap; cannot align position mask"
        )

    # Align both signals to the price index; ffill carries the last known signal,
    # leading dates before any signal exists default to 0 (no recession risk).
    ue_a = ue_signal.reindex(equity_prices.index, method="ffill").fillna(0).astype(int)
    vix_a = vix_signal.reindex(equity_prices.index, method="ffill").fillna(0).astype(int)
    recession_risk = (ue_a + vix_a) > 0

    sma = equity_prices.rolling(window=sma_window, min_periods=sma_window).mean()
    below_sma = equity_prices < sma  # NaN SMA (warm-up) -> False -> treated as above

    defensive_today = recession_risk & below_sma
    position_today = (~defensive_today).astype(int)  # 1=Long, 0=Defensive

    # Single close t -> open t+1 execution lag; first day defaults to Long.
    position_mask = position_today.shift(1).fillna(1).astype(int)
    position_mask.name = "position_mask"
    return position_mask


# ---------------------------------------------------------------------------
# I/O boundary — network calls isolated here
# ---------------------------------------------------------------------------


def fetch_gtt_signal_data(  # pragma: no cover
    start_date: str,
    end_date: str,
    vix_p90_threshold: float,
    vix_consecutive_days: int = GTT_VIX_CONSECUTIVE_DAYS,
    unrate_trade_lag_days: int = GTT_UNRATE_TRADE_LAG_DAYS,
    sma_window: int = GTT_SMA_WINDOW,
    equity_prices: pd.Series | None = None,
) -> GttSignalData:
    """Fetch UNRATE (FRED) and VIX (yfinance), compute signals, and return GttSignalData.

    All computation is delegated to the pure signal functions; this function only
    handles network I/O. The returned position_mask is already 1-day lag-adjusted
    and directly consumable by run_backtest.

    Arguments:
        start_date: ISO start date (YYYY-MM-DD). Must be >= 1993-01-01 (VIX start).
        end_date: ISO end date (YYYY-MM-DD).
        vix_p90_threshold: Fixed VIX P90 threshold as a decimal (e.g. 0.272).
            Look-ahead protection is the caller's responsibility.
        vix_consecutive_days: Number of consecutive days at/above threshold to fire
            the VIX_5D signal. Default GTT_VIX_CONSECUTIVE_DAYS (5).
        unrate_trade_lag_days: Trading-day execution lag from the UNRATE publication
            date to the trade. Default GTT_UNRATE_TRADE_LAG_DAYS (1). The
            reference->publication lag is handled inside compute_ue_signal.
        sma_window: Equity price SMA window in trading days. Default GTT_SMA_WINDOW (200).
        equity_prices: VTI price series for the SMA filter. If None, fetched internally
            via yfinance (^VTI). Caller can supply a pre-fetched series to avoid a
            redundant network call.

    Raises:
        ValueError: If start_date < 1993-01-01 (VIX data unavailable before 1993).
        ValueError: If UNRATE fetch returns an empty series (FRED unreachable or no data).
        ValueError: If VIX fetch returns an empty series (yfinance unavailable or no data).

    Returns:
        GttSignalData with fully computed, lag-adjusted signals.

    Notes:
        FRED indexes UNRATE at the reference-month start; compute_ue_signal re-stamps
        each observation to the first Friday of the following month (its true BLS
        publication date) before forward-filling. The final close->open execution lag
        is applied in compute_position_mask (shared with VIX_5D). The unrate_trade_lag_days
        parameter is retained for API symmetry with GttConfig but the reference->publication
        shift is deterministic inside compute_ue_signal and does not require this offset.
    """
    import os

    import yfinance as yf
    from fredapi import Fred

    if start_date < "1993-01-01":
        raise ValueError(
            f"start_date {start_date!r} is before 1993-01-01; VIX data is unavailable "
            "before January 1993. GTT requires VIX history."
        )

    fred_key = os.environ.get("FRED_API_KEY", "")
    fred = Fred(api_key=fred_key) if fred_key else Fred()
    unrate_raw: pd.Series = fred.get_series(
        "UNRATE", observation_start=start_date, observation_end=end_date
    )
    if unrate_raw.empty:
        raise ValueError(
            "FRED returned an empty UNRATE series for the requested date range. "
            "Check that the FRED API is reachable and FRED_API_KEY is set."
        )
    unrate_raw.name = "UNRATE"

    if equity_prices is None:
        vti_raw = yf.download(
            "VTI", start=start_date, end=end_date, auto_adjust=True, progress=False
        )
        if vti_raw.empty:
            raise ValueError(
                "yfinance returned an empty VTI price series. "
                "Check network connectivity and that start_date >= VTI inception (2001-05-24)."
            )
        equity_prices = vti_raw["Close"].squeeze().rename("VTI")

    vix_raw = yf.download("^VIX", start=start_date, end=end_date, auto_adjust=True, progress=False)
    if vix_raw.empty:
        raise ValueError(
            "yfinance returned an empty ^VIX series. "
            "Check network connectivity and that start_date >= 1993-01-01."
        )
    vix_series: pd.Series = (vix_raw["Close"].squeeze() / 100.0).rename("VIX")

    ue_sig = compute_ue_signal(unrate_raw)
    vix_sig = compute_vix_signal(
        vix_series, threshold=vix_p90_threshold, consecutive_days=vix_consecutive_days
    )
    position_mask = compute_position_mask(ue_sig, vix_sig, equity_prices, sma_window=sma_window)

    return GttSignalData(
        position_mask=position_mask,
        ue_signal=ue_sig,
        vix_signal=vix_sig,
        vix_p90_threshold=vix_p90_threshold,
        unrate_start=pd.Timestamp(unrate_raw.index[0]),
        vix_start=pd.Timestamp(vix_series.index[0]),
    )
