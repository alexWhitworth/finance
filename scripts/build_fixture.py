"""One-time script: build data/backtest_fixture.parquet and data/unrate_fixture.parquet.

Fetches the full 2000-09-01 to 2026-06-30 corpus via build_price_data and
fetch_gtt_signal_data, then persists two parquet files consumed by the
F-022 real-corpus integration test (TestRealCorpusI2).

Fixture layout
--------------
data/backtest_fixture.parquet
    DatetimeIndex (trading days), columns:
        VTI, VXUS, GLD, MUB, KMLM, VGIT   — spliced adjusted close prices
        ^VIX                                — raw VIX (decimal, ÷100)
        ^IRX                                — raw 3-month T-bill yield (decimal)
        position_mask                       — 0/1 GTT mask, 1-day lag-adjusted

data/unrate_fixture.parquet
    DatetimeIndex (monthly), columns:
        UNRATE  — FRED monthly unemployment rate

The two parquets together allow TestRealCorpusI2 to reconstruct PriceData,
ReturnData, and GttSignalData without any network calls.

Prerequisites
-------------
- FRED_API_KEY in .env (or environment)
- data/kmlm_mlmi_pre.parquet present (KMLM splice proxy)
- Network access to yfinance and FRED

Usage
-----
    uv run scripts/build_fixture.py 2>&1 | tee outputs/build_fixture.log
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

from finance.data import build_price_data
from finance.gtt import (
    compute_position_mask,
    compute_ue_signal,
    compute_vix_signal,
)

load_dotenv()

START = "2000-09-01"
END = "2026-06-30"
VIX_P90_THRESHOLD = 0.272
ASSET_TICKERS = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
FIXTURE_PATH = Path("data/backtest_fixture.parquet")
UNRATE_PATH = Path("data/unrate_fixture.parquet")


def _fetch_vix(start: str, end: str) -> pd.Series:
    """Download raw ^VIX close and convert to decimal."""
    raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("yfinance returned empty ^VIX series")
    return (raw["Close"].squeeze() / 100.0).rename("^VIX")


def _fetch_irx(start: str, end: str) -> pd.Series:
    """Download ^IRX (3-month T-bill yield) and convert to decimal."""
    raw = yf.download("^IRX", start=start, end=end, progress=False)
    if raw.empty:
        raise ValueError("yfinance returned empty ^IRX series")
    return (raw["Close"].squeeze() / 100.0).ffill().rename("^IRX")


def _fetch_unrate(start: str, end: str) -> pd.Series:
    """Fetch FRED UNRATE monthly series."""
    fred_key = os.environ.get("FRED_API_KEY", "")
    fred = Fred(api_key=fred_key) if fred_key else Fred()
    unrate: pd.Series = fred.get_series(
        "UNRATE", observation_start=start, observation_end=end
    )
    if unrate.empty:
        raise ValueError("FRED returned empty UNRATE series")
    unrate.name = "UNRATE"
    return unrate


def main() -> None:
    """Fetch full corpus and write fixture parquets."""
    print(f"=== build_fixture.py  {START} → {END} ===\n")

    print("Fetching price data (with splice)…")
    price_data = build_price_data(
        START, END, tickers=list(ASSET_TICKERS), use_splice=True, fetch_vol_indices=False
    )
    prices = price_data.prices
    print(f"  prices: {len(prices)} rows, {prices.index[0].date()} → {prices.index[-1].date()}")

    print("Fetching ^VIX…")
    vix = _fetch_vix(START, END).reindex(prices.index, method="ffill")

    print("Fetching ^IRX…")
    irx = _fetch_irx(START, END).reindex(prices.index, method="ffill")

    print("Fetching UNRATE (FRED)…")
    unrate = _fetch_unrate(START, END)
    print(f"  UNRATE: {len(unrate)} monthly observations, "
          f"{unrate.index[0].date()} → {unrate.index[-1].date()}")

    print("Computing GTT position_mask…")
    ue_sig = compute_ue_signal(unrate)
    vix_sig = compute_vix_signal(vix, threshold=VIX_P90_THRESHOLD)
    vti_prices = prices["VTI"].rename("VTI")
    position_mask = compute_position_mask(ue_sig, vix_sig, vti_prices)
    position_mask_aligned = position_mask.reindex(prices.index, method="ffill").fillna(1).astype(int)
    n_defensive = int((position_mask_aligned == 0).sum())
    print(f"  Defensive days: {n_defensive}/{len(position_mask_aligned)} "
          f"({100.0 * n_defensive / len(position_mask_aligned):.1f}%)")

    print("Assembling fixture DataFrame…")
    fixture = pd.concat(
        [prices, vix, irx, position_mask_aligned.rename("position_mask")],
        axis=1,
    )
    # Drop any rows with any NaN (very first rows where splice hasn't filled yet)
    fixture = fixture.dropna()
    fixture.index.name = "Date"

    print(f"  fixture: {len(fixture)} rows, columns: {list(fixture.columns)}")
    assert len(fixture) >= 6400, (
        f"Expected >= 6400 rows; got {len(fixture)}. Check date range and splice."
    )

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_parquet(FIXTURE_PATH)
    print(f"  Saved → {FIXTURE_PATH}")

    print("Saving UNRATE fixture…")
    unrate_df = unrate.to_frame()
    unrate_df.index.name = "Date"
    unrate_df.to_parquet(UNRATE_PATH)
    print(f"  Saved → {UNRATE_PATH}  ({len(unrate_df)} rows)")

    # Final validation
    _validate(fixture, unrate_df)
    print("\n=== build_fixture.py complete ===")


def _validate(fixture: pd.DataFrame, unrate_df: pd.DataFrame) -> None:
    """Assert acceptance criteria from spec F-021."""
    print("\n--- Validation ---")
    required_cols = {"VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT", "^VIX", "^IRX", "position_mask"}
    missing = required_cols - set(fixture.columns)
    assert not missing, f"Missing columns: {missing}"
    print(f"  [PASS] All required columns present: {sorted(required_cols)}")

    assert len(fixture) >= 6400, f"Row count {len(fixture)} < 6400"
    print(f"  [PASS] Row count: {len(fixture)} >= 6400")

    start_ts = pd.Timestamp("2000-09-01")
    end_ts = pd.Timestamp("2026-06-30")
    assert fixture.index[0] <= start_ts + pd.Timedelta(days=30), (
        f"fixture start {fixture.index[0].date()} too late (expected ~{start_ts.date()})"
    )
    assert fixture.index[-1] >= end_ts - pd.Timedelta(days=30), (
        f"fixture end {fixture.index[-1].date()} too early (expected ~{end_ts.date()})"
    )
    print(f"  [PASS] Date range: {fixture.index[0].date()} → {fixture.index[-1].date()}")

    assert not fixture.isna().any().any(), "NaN values in fixture"
    print("  [PASS] No NaN values")

    assert "UNRATE" in unrate_df.columns, "UNRATE column missing from unrate fixture"
    print(f"  [PASS] UNRATE fixture: {len(unrate_df)} monthly rows")


if __name__ == "__main__":
    main()
