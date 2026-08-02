"""F-006: _extract_day_inputs — pure per-day index lookup from BacktestContext.

No computation beyond index access and None guards. All extracted values are
from ctx series at index date_ts only (temporal invariant T1).
"""

import pandas as pd

from finance.portfolio import BacktestContext, DayInputs


def _extract_day_inputs(
    date: pd.Timestamp,
    ctx: BacktestContext,
) -> DayInputs:
    """Extract per-day scalar values from precomputed context series.

    Pure index lookup: no computation beyond series access and None guards.
    All returned fields satisfy the temporal invariant — no value in the
    returned DayInputs has a timestamp later than ``date``.

    Arguments:
        date: The trading day to extract inputs for.
        ctx: Immutable BacktestContext holding all precomputed series.

    Returns:
        DayInputs populated with values at ``date`` from ctx.
    """
    day_ret: pd.Series = ctx.return_data.returns.loc[date]  # type: ignore[assignment]
    regime_t = int(ctx.mask_aligned.loc[date]) if ctx.mask_aligned is not None else 1
    def_gross_return = float(ctx.def_gross.loc[date]) if ctx.def_gross is not None else 0.0
    spot = float(ctx.underlying_prices.loc[date]) if ctx.underlying_prices is not None else None
    raw_vix_value = float(ctx.raw_vix.loc[date]) if ctx.raw_vix is not None else None

    # mtm_iv_value: keep as float|None — may be NaN during 29-day warmup.
    # Callers guard with pd.notna() before use.
    mtm_iv_value: float | None = None
    if ctx.mtm_iv_series is not None:
        raw = ctx.mtm_iv_series.loc[date]
        mtm_iv_value = float(raw)  # may be NaN; callers guard with pd.notna()

    rfr = float(ctx.rfr_series.loc[date]) if ctx.rfr_series is not None else 0.0
    is_month_end = date in ctx.month_end_dates
    is_rebal_date = date in ctx.rebal_dates

    return DayInputs(
        date_ts=date,
        day_ret=day_ret,
        regime_t=regime_t,
        def_gross_return=def_gross_return,
        spot=spot,
        raw_vix_value=raw_vix_value,
        mtm_iv_value=mtm_iv_value,
        rfr=rfr,
        is_month_end=is_month_end,
        is_rebal_date=is_rebal_date,
    )
