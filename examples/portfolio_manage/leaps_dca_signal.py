"""LEAPS DCA entry signal — multi-factor score for timing DITM LEAPS deployment.

Fetches VTI price history with OHLCV and VIX (IV proxy) data, computes the
composite DCA entry signal as of the most recent trading day, then sweeps
the signal over the last 12 month-ends to show how the score, tranche
allocation (alpha_t), and DCA action move together over time.

Usage:
    uv run examples/portfolio_manage/leaps_dca_signal.py
"""

import pandas as pd

from finance import LeapsDcaSignal, compute_leaps_dca_signal
from finance.data import build_price_data
from finance.figures import format_leaps_dca_signal_table

TICKER = "VTI"

if __name__ == "__main__":
    START, END = "2015-01-01", "2026-06-30"

    print("=== Fetching Price Data (OHLCV + VIX) ===")
    price_data = build_price_data(
        START,
        END,
        tickers=[TICKER],
        use_splice=False,
        fetch_vol_indices=True,
        fetch_ohlcv=True,
    )

    print("\n=== Latest DCA Signal ===")
    as_of = price_data.prices.index[-1]
    signal: LeapsDcaSignal = compute_leaps_dca_signal(price_data, TICKER, as_of)
    print(format_leaps_dca_signal_table(signal))

    print("\n=== 12-Month Signal Sweep (month-end snapshots) ===")
    trading_days = price_data.prices.index.to_series()
    month_ends = trading_days.resample("ME").last().dropna().iloc[-12:]
    rows = []
    for date in month_ends:
        sig = compute_leaps_dca_signal(price_data, TICKER, date)
        rows.append(
            {
                "date": sig.as_of_date.date(),
                "entry_score": round(sig.entry_score, 1),
                "percentile": round(sig.score_percentile, 1),
                "alpha_t": round(sig.alpha_t, 2),
                "action": sig.dca_action,
            }
        )
    print(pd.DataFrame(rows).set_index("date").to_string())
