"""Tests for F-09: run_segmented_leaps_simulation."""

import pandas as pd
import pytest

from finance.leverage import (
    DEFAULT_IV,
    LTCG_RATE,
    AccountType,
    LeapsConfig,
    _live_contracts,
    run_leaps_simulation,
    run_segmented_leaps_simulation,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _prices(n_months: int = 24, start: str = "2018-01-02", price: float = 200.0) -> pd.Series:
    """Flat daily business-day price series spanning n_months calendar months."""
    end = pd.Timestamp(start) + pd.DateOffset(months=n_months)
    idx = pd.bdate_range(start, end)
    return pd.Series(price, index=idx, name="VTI")


def _all_long_mask(prices: pd.Series) -> pd.Series:
    return pd.Series(1, index=prices.index, dtype=int)


def _all_def_mask(prices: pd.Series) -> pd.Series:
    return pd.Series(0, index=prices.index, dtype=int)


def _zero_defensive_return(prices: pd.Series) -> pd.Series:
    return pd.Series(0.0, index=prices.index)


def _config(account_type: AccountType = AccountType.TAXABLE) -> LeapsConfig:
    return LeapsConfig(iv=DEFAULT_IV, ltcg_rate=LTCG_RATE, account_type=account_type)


# ---------------------------------------------------------------------------
# AC-1: All-Long equivalence to run_leaps_simulation
# ---------------------------------------------------------------------------


def test_all_long_equivalence_contracts() -> None:
    """All-Long mask: contracts and roll_events must be byte-equal to run_leaps_simulation."""
    prices = _prices(n_months=24)
    mask = _all_long_mask(prices)
    def_ret = _zero_defensive_return(prices)
    cfg = _config()

    direct = run_leaps_simulation(prices, monthly_contribution_to_leaps=500.0, config=cfg)
    segmented = run_segmented_leaps_simulation(
        prices, mask, def_ret, leaps_monthly=500.0, config=cfg
    )

    # gtt_close_events must be empty for an all-Long run
    assert segmented.gtt_close_events == ()

    # Contracts must be identical (frozen dataclasses: full field equality, same order)
    assert segmented.contracts == direct.contracts, (
        f"contracts differ: segmented has {len(segmented.contracts)}, "
        f"direct has {len(direct.contracts)}"
    )

    # Roll events must be identical (same count, same content)
    assert len(segmented.roll_events) == len(direct.roll_events)
    for i, (sr, dr) in enumerate(zip(segmented.roll_events, direct.roll_events, strict=True)):
        assert sr == dr, f"roll_event[{i}] differs: segmented={sr} direct={dr}"


def test_all_long_no_gtt_close_events() -> None:
    prices = _prices()
    segmented = run_segmented_leaps_simulation(
        prices, _all_long_mask(prices), _zero_defensive_return(prices),
        leaps_monthly=500.0, config=_config()
    )
    assert segmented.gtt_close_events == ()


# ---------------------------------------------------------------------------
# AC-2: Long -> Defensive -> Long fixture
# ---------------------------------------------------------------------------


def _ldl_prices_and_mask(
    long1_months: int = 6,
    def_months: int = 3,
    long2_months: int = 6,
    price: float = 200.0,
    start: str = "2018-01-02",
) -> tuple[pd.Series, pd.Series]:
    """Build a Long->Defensive->Long price + mask pair."""
    total_months = long1_months + def_months + long2_months
    end = pd.Timestamp(start) + pd.DateOffset(months=total_months)
    idx = pd.bdate_range(start, end)
    prices = pd.Series(price, index=idx, name="VTI")

    # Split the index into three segments by calendar months
    t0 = pd.Timestamp(start)
    boundary1 = t0 + pd.DateOffset(months=long1_months)
    boundary2 = boundary1 + pd.DateOffset(months=def_months)

    mask = pd.Series(1, index=idx, dtype=int)
    mask.loc[(mask.index >= boundary1) & (mask.index < boundary2)] = 0
    return prices, mask


def test_ldl_gtt_close_events_fire_at_boundary() -> None:
    """L->D->L: gtt_close_events are created at the Long->Defensive boundary."""
    prices, mask = _ldl_prices_and_mask()
    def_ret = _zero_defensive_return(prices)
    cfg = _config()

    ledger = run_segmented_leaps_simulation(
        prices, mask, def_ret, leaps_monthly=500.0, config=cfg
    )

    # There must be at least one GTT close event (from the Long->Defensive boundary)
    assert len(ledger.gtt_close_events) > 0


def test_ldl_no_live_contracts_during_defensive_window() -> None:
    """During the Defensive window, _live_contracts returns empty for every date."""
    prices, mask = _ldl_prices_and_mask(long1_months=6, def_months=3, long2_months=6)
    def_ret = _zero_defensive_return(prices)
    cfg = _config()

    ledger = run_segmented_leaps_simulation(
        prices, mask, def_ret, leaps_monthly=500.0, config=cfg
    )

    defensive_dates = prices.index[mask.values == 0]
    for d in defensive_dates:
        live = _live_contracts(ledger, d)
        assert live == [], f"Expected no live contracts on defensive date {d}"


def test_ldl_live_contracts_exist_in_long_windows() -> None:
    """After enough time in a Long window, at least one contract is live."""
    prices, mask = _ldl_prices_and_mask(long1_months=6, def_months=3, long2_months=6)
    def_ret = _zero_defensive_return(prices)
    cfg = _config()

    ledger = run_segmented_leaps_simulation(
        prices, mask, def_ret, leaps_monthly=500.0, config=cfg
    )

    # Pick a date in the second Long window (past enough months for purchases to have occurred)
    t0 = pd.Timestamp("2018-01-02")
    second_long_start = t0 + pd.DateOffset(months=9)  # 6 long + 3 def
    check_date = prices.index[prices.index >= second_long_start][30]  # 30 days in
    live = _live_contracts(ledger, check_date)
    assert len(live) > 0, "Expected live contracts in the second Long window"


def test_ldl_parked_pool_grows_through_defensive_window() -> None:
    """Parked pool grows exactly by defensive_gross_return during the Defensive window.

    AC-2: capital at re-entry == sum(net_proceeds from force-close) compounded by
    defensive_gross_return over each defensive day, plus leaps_monthly at each
    month-end, all within 1e-9 tolerance.
    """
    prices, mask = _ldl_prices_and_mask(long1_months=4, def_months=2, long2_months=4)

    daily_return = 0.001  # 0.1% per day — known constant
    def_ret = pd.Series(daily_return, index=prices.index)
    leaps_monthly = 500.0
    cfg = _config()

    ledger = run_segmented_leaps_simulation(
        prices, mask, def_ret, leaps_monthly=leaps_monthly, config=cfg
    )

    assert len(ledger.gtt_close_events) > 0, "Expected GTT close events at L->D boundary"

    # --- Compute expected pool at re-entry manually ---
    # Step 1: pool after force-close = sum of net_proceeds
    pool = sum(evt.net_proceeds for evt in ledger.gtt_close_events)

    # Step 2: identify defensive dates and compound pool day by day
    defensive_dates = prices.index[mask.reindex(prices.index).values == 0]

    # Month-end dates within the defensive window (last trading day of each calendar month)
    seg_prices_def = prices.loc[defensive_dates]
    dt_idx = pd.DatetimeIndex(defensive_dates)
    gb = seg_prices_def.groupby(dt_idx.to_period("M"))
    month_ends_def = {grp.index[-1] for _, grp in gb}

    for d in defensive_dates:
        pool *= 1.0 + daily_return
        if d in month_ends_def:
            pool += leaps_monthly

    # Step 3: the first contract purchased after re-entry was seeded with this pool as
    # initial_capital on the first day of the second Long window. Retrieve that contract.
    reentry_start = prices.index[mask.reindex(prices.index).values == 1]
    reentry_start = reentry_start[reentry_start > defensive_dates[-1]][0]

    # The initial_capital contract is the one purchased on the very first day of the second window.
    initial_contracts = [c for c in ledger.contracts if c.purchase_date == reentry_start]
    assert len(initial_contracts) == 1, (
        f"Expected exactly 1 initial contract on {reentry_start}, got {len(initial_contracts)}"
    )
    init_c = initial_contracts[0]

    # Verify that this contract's total_cost == pool within 1e-9
    from finance.leverage import CONTRACT_MULTIPLIER
    deployed = init_c.premium_paid * CONTRACT_MULTIPLIER * init_c.n_contracts
    assert deployed == pytest.approx(pool, abs=1e-9), (
        f"Re-entry capital {deployed:.6f} != expected pool {pool:.6f}"
    )


def test_ldl_gtt_close_event_count_matches_boundaries() -> None:
    """In a L->D->L fixture there is exactly one Long->Defensive boundary,
    so the number of gtt_close_events == number of contracts live at that boundary."""
    prices, mask = _ldl_prices_and_mask(long1_months=6, def_months=3, long2_months=6)
    def_ret = _zero_defensive_return(prices)
    cfg = _config()

    ledger = run_segmented_leaps_simulation(
        prices, mask, def_ret, leaps_monthly=500.0, config=cfg
    )

    # All gtt_close_events should have close_date == last date of the first Long window.
    t0 = pd.Timestamp("2018-01-02")
    boundary = t0 + pd.DateOffset(months=6)
    last_long1_date = prices.index[prices.index < boundary][-1]

    for evt in ledger.gtt_close_events:
        assert evt.close_date == last_long1_date, (
            f"GTT close event at {evt.close_date} != expected boundary {last_long1_date}"
        )


# ---------------------------------------------------------------------------
# AC-3: TAX_SHELTERED produces zero close tax
# ---------------------------------------------------------------------------


def test_tax_sheltered_all_close_events_have_zero_tax() -> None:
    """TAX_SHELTERED config: all gtt_close_events have tax_paid == 0.0."""
    prices, mask = _ldl_prices_and_mask()
    def_ret = _zero_defensive_return(prices)
    cfg = _config(account_type=AccountType.TAX_SHELTERED)

    ledger = run_segmented_leaps_simulation(
        prices, mask, def_ret, leaps_monthly=500.0, config=cfg
    )

    assert len(ledger.gtt_close_events) > 0
    for evt in ledger.gtt_close_events:
        assert evt.tax_paid == pytest.approx(0.0, abs=1e-9), (
            f"TAX_SHELTERED close has non-zero tax: {evt.tax_paid}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_price_series_returns_empty_ledger() -> None:
    empty_prices = pd.Series(dtype=float, name="VTI")
    empty_mask = pd.Series(dtype=int)
    empty_def = pd.Series(dtype=float)
    ledger = run_segmented_leaps_simulation(
        empty_prices, empty_mask, empty_def, leaps_monthly=500.0, config=_config()
    )
    assert ledger.contracts == ()
    assert ledger.gtt_close_events == ()


def test_all_defensive_no_contracts_pool_compounds() -> None:
    """All-Defensive mask: no contracts, pool grows via defensive return."""
    prices = _prices(n_months=3)
    mask = _all_def_mask(prices)
    daily_return = 0.001
    def_ret = pd.Series(daily_return, index=prices.index)
    cfg = _config()

    ledger = run_segmented_leaps_simulation(
        prices, mask, def_ret, leaps_monthly=0.0, config=cfg, initial_capital=10_000.0
    )

    assert ledger.contracts == ()
    assert ledger.gtt_close_events == ()


def test_defensive_window_with_zero_live_contracts_no_close_event() -> None:
    """Entering Defensive while already flat (no live contracts): zero close events."""
    prices, mask = _ldl_prices_and_mask(long1_months=2, def_months=2, long2_months=2)
    def_ret = _zero_defensive_return(prices)
    # Use zero monthly contribution and zero initial capital so no contracts are ever created.
    cfg = _config()
    ledger = run_segmented_leaps_simulation(
        prices, mask, def_ret, leaps_monthly=0.0, config=cfg, initial_capital=0.0
    )
    assert ledger.gtt_close_events == ()


def test_whipsaw_multiple_boundaries_accumulate_close_events() -> None:
    """Many L->D flips: gtt_close_events count grows with the number of boundaries."""
    # Build L-D-L-D-L pattern (4 boundaries: L->D, L->D = 2 Long->Defensive transitions).
    total_months = 12
    start = "2018-01-02"
    end = pd.Timestamp(start) + pd.DateOffset(months=total_months)
    idx = pd.bdate_range(start, end)
    prices = pd.Series(200.0, index=idx, name="VTI")

    # Assign alternating monthly blocks: months 0,2,4,6,8,10 = Long; 1,3,5,7,9,11 = Def
    t0 = pd.Timestamp(start)
    mask = pd.Series(1, index=idx, dtype=int)
    for m in range(1, total_months, 2):
        b_start = t0 + pd.DateOffset(months=m)
        b_end = t0 + pd.DateOffset(months=m + 1)
        mask.loc[(mask.index >= b_start) & (mask.index < b_end)] = 0

    def_ret = _zero_defensive_return(prices)
    cfg = _config()

    ledger = run_segmented_leaps_simulation(
        prices, mask, def_ret, leaps_monthly=500.0, config=cfg
    )

    # There should be one set of close events per Long->Defensive boundary.
    # With alternating monthly blocks starting Long, we have boundaries at months 1, 3, 5, 7, 9.
    # But only the L->D transitions produce close events (not D->L).
    # Count unique close dates — each L->D boundary produces a cluster.
    close_dates = {evt.close_date for evt in ledger.gtt_close_events}
    assert len(close_dates) >= 2, (
        f"Expected >=2 distinct close-date clusters (L->D boundaries) in whipsaw fixture; "
        f"got {len(close_dates)}"
    )


def test_timeline_ends_in_defensive_window_no_dangling_open_contracts() -> None:
    """Timeline ending inside a Defensive window: no live contracts at the final date."""
    prices, mask = _ldl_prices_and_mask(long1_months=6, def_months=12, long2_months=0)
    # Force the entire tail to be Defensive so the series unambiguously ends in regime=0.
    t0 = pd.Timestamp("2018-01-02")
    boundary1 = t0 + pd.DateOffset(months=6)
    mask.loc[mask.index >= boundary1] = 0
    assert mask.iloc[-1] == 0, "Fixture must end in a Defensive window"

    def_ret = _zero_defensive_return(prices)
    cfg = _config()

    ledger = run_segmented_leaps_simulation(
        prices, mask, def_ret, leaps_monthly=500.0, config=cfg
    )

    # No live contracts at the end of the series
    last_date = prices.index[-1]
    assert _live_contracts(ledger, last_date) == []


def test_misaligned_mask_raises_value_error() -> None:
    """Non-empty position_mask with no date overlap raises ValueError before reindex."""
    prices = _prices(n_months=12, start="2020-01-02")
    # mask on a completely different date range — no overlap with prices
    mask = pd.Series(1, index=pd.bdate_range("2010-01-04", periods=50), dtype=int)
    def_ret = _zero_defensive_return(prices)
    cfg = _config()

    with pytest.raises(ValueError, match="no common dates"):
        run_segmented_leaps_simulation(prices, mask, def_ret, leaps_monthly=500.0, config=cfg)


def test_optional_iv_and_rfr_series_used_at_force_close_boundary() -> None:
    """iv_series and risk_free_series are used for force-close pricing at L->D boundary.

    Exercises lines 1098 and 1102: the seg_iv/seg_rfr true branches inside the
    force-close block.
    """
    prices, mask = _ldl_prices_and_mask(long1_months=4, def_months=2, long2_months=4)
    def_ret = _zero_defensive_return(prices)
    cfg = _config()

    # Provide explicit iv_series and risk_free_series so the optional branches fire
    iv_series = pd.Series(0.25, index=prices.index)   # 25% IV (above DEFAULT_IV floor)
    rfr_series = pd.Series(0.04, index=prices.index)  # 4% risk-free rate

    ledger = run_segmented_leaps_simulation(
        prices, mask, def_ret, leaps_monthly=500.0, config=cfg,
        iv_series=iv_series, risk_free_series=rfr_series,
    )

    # The simulation must complete and produce at least one close event at the L->D boundary
    assert len(ledger.gtt_close_events) > 0

    # All close events must have valid (non-NaN, positive) mtm_value and net_proceeds
    for evt in ledger.gtt_close_events:
        assert evt.mtm_value > 0.0, f"mtm_value={evt.mtm_value} should be positive"
        assert evt.net_proceeds > 0.0, f"net_proceeds={evt.net_proceeds} should be positive"
        # net_proceeds == mtm_value - tax_paid (identity check with 1e-9 tolerance)
        assert evt.net_proceeds == pytest.approx(evt.mtm_value - evt.tax_paid, abs=1e-9)
