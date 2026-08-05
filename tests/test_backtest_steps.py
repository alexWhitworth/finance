from __future__ import annotations

import dataclasses
import math
import re

import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from finance._backtest_steps import (
    _advance_state,
    _apply_contribution,
    _apply_defensive_compounding,
    _apply_gtt_force_close,
    _apply_gtt_open,
    _apply_gtt_reentry,
    _apply_rebalance,
    _apply_returns,
    _assemble_leaps_ledger,
    _build_context,
    _build_initial_state,
    _build_weight_row,
    _compute_leaps_mtm,
    _compute_nav_before_contrib,
    _compute_port_return,
    _compute_total_nav,
    _extract_day_inputs,
)
from finance.consts import DEFAULT_IV, VIX_MTM_WINDOW
from finance.leverage import (
    AccountType,
    LeapsConfig,
    LeapsContract,
    LeapsGttCloseEvent,
    LeapsLedger,
    RebalanceRule,
    _live_contracts,
    price_leaps_contract,
    run_leaps_simulation,
)
from tests.conftest import (
    _F009_DATE,
    _F012_DATE,
    _F014_DATES,
    _F014_PRICES,
    _F014_RE_ENTRY_DATE,
    _F014_RETURN_DATA,
    _F014_RFR,
    _F014_SPOT,
    _F015_DEFAULT_DATE,
    _make_config,
    _make_contribution_ctx,
    _make_dates,
    _make_day_inputs,
    _make_extract_ctx,
    _make_gtt_force_close_ctx,
    _make_gtt_open_ctx,
    _make_gtt_signal,
    _make_leaps_ctx,
    _make_leaps_mtm_ctx,
    _make_minimal_backtest_ctx,
    _make_no_leaps_ctx,
    _make_portfolio_state,
    _make_price_data,
    _make_rebalance_ctx,
    _make_reentry_ctx,
    _make_return_data,
    _make_returns_ctx,
    make_contract,
    make_ledger,
)

# ---------------------------------------------------------------------------
# 1. ValueError: gtt_signal set but gtt_config is None
# ---------------------------------------------------------------------------


def test_gtt_signal_without_gtt_config_raises() -> None:
    """Providing gtt_signal without gtt_config raises ValueError."""
    dates = _make_dates()
    rd = _make_return_data(dates)
    pd_ = _make_price_data(dates)
    config = _make_config()  # gtt_config=None
    signal = _make_gtt_signal(dates)

    with pytest.raises(ValueError, match=re.escape(
        "gtt_signal and config.gtt_config must both be set"
    )):
        _build_context(rd, pd_, config, signal)


# ---------------------------------------------------------------------------
# 2. ValueError: assets missing from return_data
# ---------------------------------------------------------------------------


def test_missing_asset_in_return_data_raises() -> None:
    """A weight for a ticker absent from return_data raises ValueError."""
    dates = _make_dates()
    rd = _make_return_data(dates, tickers=("VTI",))
    pd_ = _make_price_data(dates, tickers=("VTI",))
    # VXUS is in weights but not in returns
    config = _make_config(weights={"VTI": 0.7, "VXUS": 0.3})

    with pytest.raises(ValueError, match="Assets missing from return_data"):
        _build_context(rd, pd_, config, None)


# ---------------------------------------------------------------------------
# 3. ValueError: LEAPS keys present but leaps_config is None
# ---------------------------------------------------------------------------


def test_leaps_keys_without_leaps_config_raises() -> None:
    """VTI_LEAPS in target_weights without leaps_config raises ValueError."""
    dates = _make_dates()
    rd = _make_return_data(dates, tickers=("VTI",))
    pd_ = _make_price_data(dates, tickers=("VTI",))
    config = _make_config(
        weights={"VTI": 0.85, "VTI_LEAPS": 0.15},
        leaps_config=None,
    )

    with pytest.raises(ValueError, match="leaps_config is None"):
        _build_context(rd, pd_, config, None)


# ---------------------------------------------------------------------------
# 4. ValueError: multiple LEAPS underlyings
# ---------------------------------------------------------------------------


def test_multiple_leaps_underlyings_raises() -> None:
    """VTI_LEAPS and VXUS_LEAPS together raise ValueError."""
    dates = _make_dates()
    rd = _make_return_data(dates, tickers=("VTI", "VXUS"))
    pd_ = _make_price_data(dates, tickers=("VTI", "VXUS"))
    leaps_cfg = LeapsConfig(iv=0.18)
    config = _make_config(
        weights={"VTI": 0.5, "VTI_LEAPS": 0.25, "VXUS_LEAPS": 0.25},
        leaps_config=leaps_cfg,
    )

    with pytest.raises(ValueError, match="Only one LEAPS underlying is supported"):
        _build_context(rd, pd_, config, None)


# ---------------------------------------------------------------------------
# 5. rebal_dates is frozenset
# ---------------------------------------------------------------------------


def test_rebal_dates_is_frozenset() -> None:
    """ctx.rebal_dates is a frozenset instance."""
    dates = _make_dates(120)
    rd = _make_return_data(dates)
    pd_ = _make_price_data(dates)
    config = _make_config()

    ctx = _build_context(rd, pd_, config, None)
    assert isinstance(ctx.rebal_dates, frozenset)


# ---------------------------------------------------------------------------
# 6. month_end_dates is frozenset
# ---------------------------------------------------------------------------


def test_month_end_dates_is_frozenset() -> None:
    """ctx.month_end_dates is a frozenset instance."""
    dates = _make_dates(120)
    rd = _make_return_data(dates)
    pd_ = _make_price_data(dates)
    config = _make_config()

    ctx = _build_context(rd, pd_, config, None)
    assert isinstance(ctx.month_end_dates, frozenset)


# ---------------------------------------------------------------------------
# 7. mtm_iv_series causality
# ---------------------------------------------------------------------------


def test_mtm_iv_series_causality() -> None:
    """mtm_iv_series[t] equals manual rolling mean of raw_vix[t-29:t] within 1e-10.

    Uses at least 60 days so the rolling window is fully populated for day index 30+.
    """
    n = 80
    dates = _make_dates(n)
    rd = _make_return_data(dates, tickers=("VTI",), seed=42)
    pd_ = _make_price_data(dates, tickers=("VTI",), vol_tickers=("VTI",), seed=99)
    leaps_cfg = LeapsConfig(iv=0.18)
    config = _make_config(
        weights={"VTI": 0.85, "VTI_LEAPS": 0.15},
        leaps_config=leaps_cfg,
    )

    ctx = _build_context(rd, pd_, config, None)
    assert ctx.mtm_iv_series is not None
    assert ctx.raw_vix is not None

    # Pick a date at index 35 (well past the 29-day warmup)
    t_idx = 35
    t = pd.Timestamp(dates[t_idx])

    expected = float(ctx.raw_vix.iloc[t_idx - VIX_MTM_WINDOW + 1 : t_idx + 1].mean())
    actual = float(ctx.mtm_iv_series.loc[t])
    assert abs(actual - expected) < 1e-10


# ---------------------------------------------------------------------------
# 8. No GTT, no LEAPS baseline
# ---------------------------------------------------------------------------


def test_no_gtt_no_leaps_baseline() -> None:
    """Basic config produces ctx with gtt_active=False and use_leaps=False."""
    dates = _make_dates(60)
    rd = _make_return_data(dates)
    pd_ = _make_price_data(dates)
    config = _make_config()

    ctx = _build_context(rd, pd_, config, None)
    assert ctx.gtt_active is False
    assert ctx.use_leaps is False
    assert ctx.mask_aligned is None
    assert ctx.def_gross is None
    assert ctx.underlying_prices is None
    assert ctx.raw_vix is None
    assert ctx.mtm_iv_series is None


# ---------------------------------------------------------------------------
# _build_initial_state
# ---------------------------------------------------------------------------


def test_no_leaps_ledger_is_none() -> None:
    """When use_leaps=False, leaps_ledger must be None."""
    ctx = _make_no_leaps_ctx()
    state = _build_initial_state(ctx)
    assert state.leaps_ledger is None


def test_no_leaps_leaps_value_zero() -> None:
    """When use_leaps=False, leaps_value must be 0.0."""
    ctx = _make_no_leaps_ctx()
    state = _build_initial_state(ctx)
    assert state.leaps_value == 0.0


def test_no_leaps_all_window_ledgers_empty() -> None:
    """When use_leaps=False, all_window_ledgers must be an empty tuple."""
    ctx = _make_no_leaps_ctx()
    state = _build_initial_state(ctx)
    assert state.all_window_ledgers == ()


def test_no_leaps_holdings_sum_equals_initial_nav() -> None:
    """Without LEAPS, all NAV goes into base holdings."""
    ctx = _make_no_leaps_ctx(initial_nav=100_000.0)
    state = _build_initial_state(ctx)
    assert sum(state.holdings.values()) == pytest.approx(ctx.config.initial_nav, rel=1e-9)


def test_holdings_sum_equals_base_nav_slice() -> None:
    """Holdings sum == initial_nav * (1 - leaps_fraction) for any leaps_fraction."""
    ctx = _make_leaps_ctx()
    state = _build_initial_state(ctx)
    expected_base = ctx.config.initial_nav * (1.0 - ctx.leaps_fraction)
    assert sum(state.holdings.values()) == pytest.approx(expected_base, rel=1e-9)


def test_holdings_weights_match_base_target_w() -> None:
    """Holdings ratios must equal base_target_w ratios for a 60/40 base portfolio."""
    ctx = _make_no_leaps_ctx(weights={"VTI": 0.6, "VXUS": 0.4})
    state = _build_initial_state(ctx)
    ratio = state.holdings["VTI"] / state.holdings["VXUS"]
    assert ratio == pytest.approx(0.6 / 0.4, rel=1e-9)


def test_prev_total_nav_equals_initial_nav() -> None:
    """prev_total_nav must equal initial_nav regardless of LEAPS fraction."""
    for ctx in (_make_no_leaps_ctx(initial_nav=50_000.0), _make_leaps_ctx(initial_nav=50_000.0)):
        state = _build_initial_state(ctx)
        assert state.prev_total_nav == ctx.config.initial_nav


def test_sentinel_fields_always_at_defaults() -> None:
    """prev_regime=1, prev_date_ts=None, all_gtt_closes=() on first iteration."""
    ctx = _make_no_leaps_ctx()
    state = _build_initial_state(ctx)
    assert state.prev_regime == 1
    assert state.prev_date_ts is None
    assert state.all_gtt_closes == ()


def test_leaps_active_ledger_not_none() -> None:
    """When use_leaps=True and underlying_prices is set, leaps_ledger is not None."""
    ctx = _make_leaps_ctx()
    state = _build_initial_state(ctx)
    assert state.leaps_ledger is not None


def test_leaps_active_all_window_ledgers_has_one_entry() -> None:
    """When LEAPS are active, all_window_ledgers must contain exactly one ledger."""
    ctx = _make_leaps_ctx()
    state = _build_initial_state(ctx)
    assert len(state.all_window_ledgers) == 1
    assert state.all_window_ledgers[0] is state.leaps_ledger


def test_gtt_leaps_first_long_window_used() -> None:
    """With GTT active, no contract is purchased before the first Long day (index 4)."""
    dates = pd.bdate_range("2020-01-02", periods=126)
    mask = pd.Series([0] * 4 + [1] * (len(dates) - 4), index=dates, dtype=int)
    first_long_date = dates[4]

    ctx = _make_leaps_ctx(gtt_active=True, mask_aligned=mask)
    state = _build_initial_state(ctx)

    assert state.leaps_ledger is not None
    for contract in state.leaps_ledger.contracts:
        assert contract.purchase_date >= pd.Timestamp(first_long_date)


def test_gtt_leaps_all_defensive_mask_gives_empty_ledger() -> None:
    """When the mask is all Defensive, run_leaps_simulation returns an empty ledger."""
    dates = pd.bdate_range("2020-01-02", periods=126)
    mask = pd.Series([0] * len(dates), index=dates, dtype=int)

    ctx = _make_leaps_ctx(gtt_active=True, mask_aligned=mask)
    state = _build_initial_state(ctx)

    assert state.leaps_ledger is not None
    assert len(state.leaps_ledger.contracts) == 0


# ---------------------------------------------------------------------------
# _extract_day_inputs
# ---------------------------------------------------------------------------


def test_extract_all_fields_match_series() -> None:
    """All DayInputs fields equal expected values from the underlying series."""
    dates = pd.bdate_range("2022-01-03", periods=100)
    ctx = _make_extract_ctx(dates)
    date = dates[50]  # past warmup so mtm_iv_value is finite

    inputs = _extract_day_inputs(date, ctx)

    assert inputs.date_ts == date
    assert float(inputs.day_ret["VTI"]) == pytest.approx(
        float(ctx.return_data.returns.loc[date, "VTI"]), abs=1e-12
    )
    assert inputs.regime_t == int(ctx.mask_aligned.loc[date])  # type: ignore[index]
    assert inputs.def_gross_return == pytest.approx(
        float(ctx.def_gross.loc[date]), abs=1e-12  # type: ignore[index]
    )
    assert inputs.spot is None
    assert inputs.raw_vix_value == pytest.approx(
        float(ctx.raw_vix.loc[date]), abs=1e-12  # type: ignore[index]
    )
    assert inputs.mtm_iv_value == pytest.approx(
        float(ctx.mtm_iv_series.loc[date]), abs=1e-12  # type: ignore[index]
    )
    assert inputs.rfr == pytest.approx(
        float(ctx.rfr_series.loc[date]), abs=1e-12  # type: ignore[index]
    )
    assert isinstance(inputs.is_month_end, bool)
    assert inputs.is_month_end == (date in ctx.month_end_dates)
    assert isinstance(inputs.is_rebal_date, bool)
    assert inputs.is_rebal_date == (date in ctx.rebal_dates)


def test_extract_long_day_regime_is_1() -> None:
    """A date in the first half of the mask returns regime_t == 1."""
    dates = pd.bdate_range("2022-01-03", periods=60)
    ctx = _make_extract_ctx(dates)
    inputs = _extract_day_inputs(dates[10], ctx)
    assert inputs.regime_t == 1


def test_extract_defensive_day_regime_is_0() -> None:
    """A date in the second half of the mask returns regime_t == 0."""
    dates = pd.bdate_range("2022-01-03", periods=60)
    ctx = _make_extract_ctx(dates)
    inputs = _extract_day_inputs(dates[50], ctx)
    assert inputs.regime_t == 0


def test_extract_mtm_iv_nan_during_warmup() -> None:
    """mtm_iv_value is NaN for dates in the rolling-mean warmup window."""
    dates = pd.bdate_range("2022-01-03", periods=100)
    ctx = _make_extract_ctx(dates)
    # Replace mtm_iv_series with one that has no ffill so first 29 rows are NaN
    vix = ctx.raw_vix
    assert vix is not None
    ctx_no_fill = dataclasses.replace(ctx, mtm_iv_series=vix.rolling(30).mean())

    inputs = _extract_day_inputs(dates[5], ctx_no_fill)

    assert inputs.mtm_iv_value is not None
    assert math.isnan(inputs.mtm_iv_value)


def test_extract_raw_vix_none_returns_none() -> None:
    """When ctx.raw_vix is None, raw_vix_value and mtm_iv_value are both None."""
    dates = pd.bdate_range("2022-01-03", periods=30)
    ctx = _make_extract_ctx(dates)
    ctx_no_vix = dataclasses.replace(ctx, raw_vix=None, mtm_iv_series=None)

    inputs = _extract_day_inputs(dates[10], ctx_no_vix)

    assert inputs.raw_vix_value is None
    assert inputs.mtm_iv_value is None


def test_extract_is_month_end_true_and_false() -> None:
    """is_month_end is True for a date in month_end_dates and False otherwise."""
    dates = pd.bdate_range("2022-01-03", periods=60)
    ctx = _make_extract_ctx(dates)
    month_end_date = next(iter(ctx.month_end_dates))
    non_month_end_date = next(d for d in dates if d not in ctx.month_end_dates)

    assert _extract_day_inputs(month_end_date, ctx).is_month_end is True
    assert _extract_day_inputs(non_month_end_date, ctx).is_month_end is False


def test_extract_is_rebal_date_true() -> None:
    """is_rebal_date is True when date is in ctx.rebal_dates."""
    dates = pd.bdate_range("2022-01-03", periods=100)
    ctx = _make_extract_ctx(dates)
    rebal_date = dates[40]
    ctx_with_rebal = dataclasses.replace(ctx, rebal_dates=frozenset({rebal_date}))

    assert _extract_day_inputs(rebal_date, ctx_with_rebal).is_rebal_date is True
    assert _extract_day_inputs(dates[41], ctx_with_rebal).is_rebal_date is False


# ---------------------------------------------------------------------------
# _apply_gtt_open
# ---------------------------------------------------------------------------


def test_gtt_open_noop_on_long_day() -> None:
    """State is returned unchanged (same object) when regime_t == 1."""
    state = _make_portfolio_state({"VTI": 50_000.0, "VXUS": 20_000.0})
    ctx = _make_gtt_open_ctx(gtt_active=True)
    inputs = _make_day_inputs(regime_t=1)

    assert _apply_gtt_open(state, inputs, ctx) is state


def test_gtt_open_sweep_on_defensive_day() -> None:
    """Governed VTI is zeroed and moved into defensive_sleeve; VXUS untouched."""
    state = _make_portfolio_state({"VTI": 50_000.0, "VXUS": 20_000.0}, sleeve=5_000.0)
    ctx = _make_gtt_open_ctx(gtt_active=True)
    inputs = _make_day_inputs(regime_t=0)

    new = _apply_gtt_open(state, inputs, ctx)

    assert new.holdings["VTI"] == 0.0
    assert new.holdings["VXUS"] == 20_000.0
    assert new.defensive_sleeve == pytest.approx(55_000.0)


def test_gtt_open_noop_when_inactive() -> None:
    """State returned unchanged when gtt_active=False, even on a defensive day."""
    state = _make_portfolio_state({"VTI": 50_000.0, "VXUS": 20_000.0})
    ctx = _make_gtt_open_ctx(gtt_active=False)
    inputs = _make_day_inputs(regime_t=0)

    assert _apply_gtt_open(state, inputs, ctx) is state


def test_gtt_open_double_sweep_safety() -> None:
    """Sleeve never goes negative when governed holding is already 0.0."""
    state = _make_portfolio_state({"VTI": 0.0, "VXUS": 20_000.0}, sleeve=50_000.0)
    ctx = _make_gtt_open_ctx(gtt_active=True)
    inputs = _make_day_inputs(regime_t=0)

    new = _apply_gtt_open(state, inputs, ctx)

    assert new.holdings["VTI"] == 0.0
    assert new.defensive_sleeve == pytest.approx(50_000.0)


def test_gtt_open_accounting_invariant() -> None:
    """Total (holdings_sum + sleeve) is conserved within 1e-12."""
    state = _make_portfolio_state({"VTI": 50_000.0, "VXUS": 20_000.0}, sleeve=3_000.0)
    ctx = _make_gtt_open_ctx(gtt_active=True)
    inputs = _make_day_inputs(regime_t=0)

    before = sum(state.holdings.values()) + state.defensive_sleeve
    new = _apply_gtt_open(state, inputs, ctx)
    after = sum(new.holdings.values()) + new.defensive_sleeve

    assert abs(after - before) < 1e-12


@given(
    vti=st.floats(0, 1e6),
    vxus=st.floats(0, 1e6),
    sleeve=st.floats(0, 1e4),
)
@settings(max_examples=500)
def test_gtt_open_accounting_invariant_hypothesis(
    vti: float, vxus: float, sleeve: float
) -> None:
    """Total capital is conserved for all non-negative holding/sleeve combinations."""
    state = _make_portfolio_state({"VTI": vti, "VXUS": vxus}, sleeve=sleeve)
    ctx = _make_gtt_open_ctx(gtt_active=True)
    inputs = _make_day_inputs(regime_t=0)

    before = sum(state.holdings.values()) + state.defensive_sleeve
    new = _apply_gtt_open(state, inputs, ctx)
    after = sum(new.holdings.values()) + new.defensive_sleeve

    assert abs(after - before) < 1e-9


# ---------------------------------------------------------------------------
# _apply_gtt_force_close
# ---------------------------------------------------------------------------

_F008_T0 = pd.Timestamp("2020-01-02")
_F008_T1 = pd.Timestamp("2020-01-03")


def test_force_close_noop_not_a_transition_day() -> None:
    """When prev_regime==1 and regime_t==1, state is returned unchanged."""
    contract = make_contract()
    ledger = make_ledger(contract)
    state = _make_portfolio_state({"VTI": 85_000.0}, leaps_ledger=ledger, prev_date_ts=_F008_T0)
    ctx = _make_gtt_force_close_ctx()
    inputs = _make_day_inputs(date_ts=_F008_T1, regime_t=1)

    assert _apply_gtt_force_close(state, inputs, ctx) is state


def test_force_close_noop_leaps_ledger_none() -> None:
    """When leaps_ledger is None on a transition day, state is returned unchanged."""
    state = _make_portfolio_state({"VTI": 85_000.0}, leaps_ledger=None, prev_date_ts=_F008_T0)
    ctx = _make_gtt_force_close_ctx()
    inputs = _make_day_inputs(date_ts=_F008_T1, regime_t=0)

    assert _apply_gtt_force_close(state, inputs, ctx) is state


def test_force_close_noop_gtt_inactive() -> None:
    """When gtt_active=False, state is returned unchanged regardless of regime."""
    contract = make_contract()
    ledger = make_ledger(contract)
    state = _make_portfolio_state({"VTI": 85_000.0}, leaps_ledger=ledger, prev_date_ts=_F008_T0)
    ctx = _make_gtt_force_close_ctx(gtt_active=False)
    inputs = _make_day_inputs(date_ts=_F008_T1, regime_t=0)

    assert _apply_gtt_force_close(state, inputs, ctx) is state


def test_force_close_fires_a3_invariant() -> None:
    """Invariant A3: new_leaps_pool - old_leaps_pool == new_event.net_proceeds."""
    contract = make_contract()
    ledger = make_ledger(contract)
    state = _make_portfolio_state(
        {"VTI": 85_000.0}, pool=1000.0, leaps_ledger=ledger, prev_date_ts=_F008_T0
    )
    ctx = _make_gtt_force_close_ctx(spot=210.0)
    inputs = _make_day_inputs(
        date_ts=_F008_T1, regime_t=0, spot=210.0, raw_vix_value=0.25, mtm_iv_value=0.22
    )

    new_state = _apply_gtt_force_close(state, inputs, ctx)

    assert len(new_state.all_gtt_closes) == len(state.all_gtt_closes) + 1
    evt = new_state.all_gtt_closes[-1]
    assert abs((new_state.leaps_pool - state.leaps_pool) - evt.net_proceeds) < 1e-9


def test_force_close_scale_applied_to_n_contracts() -> None:
    """leaps_scale={contract: 0.5} halves n_contracts in the close event."""
    contract = make_contract(n=2.0)
    ledger = make_ledger(contract)
    state = _make_portfolio_state(
        {"VTI": 85_000.0},
        leaps_ledger=ledger,
        leaps_scale={contract: 0.5},
        prev_date_ts=_F008_T0,
    )
    ctx = _make_gtt_force_close_ctx(spot=210.0)
    inputs = _make_day_inputs(
        date_ts=_F008_T1, regime_t=0, spot=210.0, raw_vix_value=0.25, mtm_iv_value=0.22
    )

    new_state = _apply_gtt_force_close(state, inputs, ctx)

    assert len(new_state.all_gtt_closes) == 1
    assert abs(new_state.all_gtt_closes[0].contract.n_contracts - 2.0 * 0.5) < 1e-12


def test_force_close_tax_sheltered_tax_paid_zero() -> None:
    """TAX_SHELTERED account: tax_paid == 0.0 for the GTT close event."""
    contract = make_contract(account_type=AccountType.TAX_SHELTERED)
    ledger = LeapsLedger(
        contracts=(contract,), roll_events=(), account_type=AccountType.TAX_SHELTERED
    )
    state = _make_portfolio_state(
        {"VTI": 85_000.0}, leaps_ledger=ledger, prev_date_ts=_F008_T0
    )
    ctx = _make_gtt_force_close_ctx(spot=210.0)
    inputs = _make_day_inputs(
        date_ts=_F008_T1, regime_t=0, spot=210.0, raw_vix_value=0.25, mtm_iv_value=0.22
    )

    new_state = _apply_gtt_force_close(state, inputs, ctx)

    assert len(new_state.all_gtt_closes) == 1
    assert new_state.all_gtt_closes[0].tax_paid == 0.0


@given(
    spot=st.floats(min_value=50.0, max_value=600.0, allow_nan=False, allow_infinity=False),
    n_contracts=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_force_close_hypothesis_a3_invariant(spot: float, n_contracts: float) -> None:
    """Invariant A3 holds over varied spot prices and contract counts."""
    contract = make_contract(spot=spot, n=n_contracts)
    ledger = make_ledger(contract)
    initial_pool = 500.0
    state = _make_portfolio_state(
        {"VTI": 85_000.0}, pool=initial_pool, leaps_ledger=ledger, prev_date_ts=_F008_T0
    )
    ctx = _make_gtt_force_close_ctx(spot=spot)
    inputs = _make_day_inputs(
        date_ts=_F008_T1, regime_t=0, spot=spot, raw_vix_value=0.25, mtm_iv_value=0.22
    )

    new_state = _apply_gtt_force_close(state, inputs, ctx)

    new_events = new_state.all_gtt_closes[len(state.all_gtt_closes):]
    expected_pool = initial_pool + sum(e.net_proceeds for e in new_events)
    assert abs(new_state.leaps_pool - expected_pool) < 1e-9


# ---------------------------------------------------------------------------
# _apply_returns
# ---------------------------------------------------------------------------


def test_returns_flat_holdings_unchanged() -> None:
    """Flat returns (day_ret=0.0) leave holdings numerically identical."""
    state = _make_portfolio_state({"VTI": 5000.0, "VXUS": 3000.0})
    inputs = _make_day_inputs(
        date_ts=_F009_DATE, day_ret=pd.Series({"VTI": 0.0, "VXUS": 0.0})
    )
    ctx = _make_returns_ctx()

    new = _apply_returns(state, inputs, ctx)

    assert abs(new.holdings["VTI"] - 5000.0) < 1e-14
    assert abs(new.holdings["VXUS"] - 3000.0) < 1e-14


def test_returns_known_scale_holdings() -> None:
    """Known returns scale holdings by (1 + r) for each asset."""
    state = _make_portfolio_state({"VTI": 10_000.0, "VXUS": 5_000.0})
    inputs = _make_day_inputs(
        date_ts=_F009_DATE, day_ret=pd.Series({"VTI": 0.05, "VXUS": -0.02})
    )
    ctx = _make_returns_ctx()

    new = _apply_returns(state, inputs, ctx)

    assert new.holdings["VTI"] == pytest.approx(10_500.0)
    assert new.holdings["VXUS"] == pytest.approx(4_900.0)


@given(
    ret_vti=st.floats(-0.5, 0.5),
    ret_vxus=st.floats(-0.5, 0.5),
    h_vti=st.floats(100.0, 10_000.0),
    h_vxus=st.floats(100.0, 10_000.0),
)
@settings(max_examples=500)
def test_returns_exact_formula_property(
    ret_vti: float, ret_vxus: float, h_vti: float, h_vxus: float
) -> None:
    """Property: holdings_out[a] == holdings_in[a] * (1 + day_ret[a])."""
    state = _make_portfolio_state({"VTI": h_vti, "VXUS": h_vxus})
    inputs = _make_day_inputs(
        date_ts=_F009_DATE, day_ret=pd.Series({"VTI": ret_vti, "VXUS": ret_vxus})
    )
    ctx = _make_returns_ctx()

    new = _apply_returns(state, inputs, ctx)

    assert abs(new.holdings["VTI"] - h_vti * (1.0 + ret_vti)) < 1e-10
    assert abs(new.holdings["VXUS"] - h_vxus * (1.0 + ret_vxus)) < 1e-10


def test_returns_non_holdings_fields_unchanged() -> None:
    """Fields other than holdings are not mutated by _apply_returns."""
    state = _make_portfolio_state({"VTI": 5000.0, "VXUS": 3000.0}, sleeve=200.0, pool=100.0)
    inputs = _make_day_inputs(
        date_ts=_F009_DATE, day_ret=pd.Series({"VTI": 0.01, "VXUS": 0.02})
    )
    ctx = _make_returns_ctx()

    new = _apply_returns(state, inputs, ctx)

    assert new.defensive_sleeve == state.defensive_sleeve
    assert new.leaps_pool == state.leaps_pool
    assert new.leaps_value == state.leaps_value
    assert new.prev_total_nav == state.prev_total_nav
    assert new.prev_regime == state.prev_regime


# ---------------------------------------------------------------------------
# _apply_defensive_compounding
# ---------------------------------------------------------------------------

_def_gross_f009 = pd.Series([0.001], index=pd.DatetimeIndex([_F009_DATE]))


def test_defensive_compounding_noop_gtt_inactive() -> None:
    """No compounding when gtt_active=False; state returned unchanged."""
    state = _make_portfolio_state({"VTI": 0.0, "VXUS": 0.0}, sleeve=10_000.0, pool=2_000.0)
    inputs = _make_day_inputs(date_ts=_F009_DATE, regime_t=0, def_gross_return=0.001)
    ctx = _make_returns_ctx(gtt_active=False, def_gross=_def_gross_f009)

    assert _apply_defensive_compounding(state, inputs, ctx) is state


def test_defensive_compounding_noop_on_long_day() -> None:
    """No compounding on a pure Long day (prev_regime=1, regime_t=1)."""
    state = _make_portfolio_state(
        {"VTI": 0.0, "VXUS": 0.0}, sleeve=10_000.0, pool=2_000.0, prev_regime=1
    )
    inputs = _make_day_inputs(date_ts=_F009_DATE, regime_t=1, def_gross_return=0.001)
    ctx = _make_returns_ctx(gtt_active=True, def_gross=_def_gross_f009)

    new = _apply_defensive_compounding(state, inputs, ctx)

    assert new.defensive_sleeve == state.defensive_sleeve
    assert new.leaps_pool == state.leaps_pool


def test_defensive_compounding_defensive_day_compounds() -> None:
    """Defensive day: sleeve and pool both compounded by (1 + def_gross_return)."""
    state = _make_portfolio_state({"VTI": 0.0, "VXUS": 0.0}, sleeve=10_000.0, pool=2_000.0)
    inputs = _make_day_inputs(date_ts=_F009_DATE, regime_t=0, def_gross_return=0.001)
    ctx = _make_returns_ctx(gtt_active=True, def_gross=_def_gross_f009)

    new = _apply_defensive_compounding(state, inputs, ctx)

    assert new.defensive_sleeve == pytest.approx(10_010.0)
    assert new.leaps_pool == pytest.approx(2_002.0)


def test_defensive_compounding_reentry_day_compounds_once() -> None:
    """Re-entry day (prev_regime=0, regime_t=1) compounds once before redeployment."""
    state = _make_portfolio_state(
        {"VTI": 0.0, "VXUS": 0.0}, sleeve=8_000.0, pool=1_500.0, prev_regime=0
    )
    inputs = _make_day_inputs(date_ts=_F009_DATE, regime_t=1, def_gross_return=0.002)
    ctx = _make_returns_ctx(
        gtt_active=True, def_gross=pd.Series([0.002], index=pd.DatetimeIndex([_F009_DATE]))
    )

    new = _apply_defensive_compounding(state, inputs, ctx)

    assert new.defensive_sleeve == pytest.approx(8_016.0)
    assert new.leaps_pool == pytest.approx(1_503.0)


def test_defensive_compounding_zero_rate_no_change() -> None:
    """Zero def_gross_return leaves sleeve and pool exactly unchanged."""
    state = _make_portfolio_state({"VTI": 0.0, "VXUS": 0.0}, sleeve=10_000.0, pool=2_000.0)
    inputs = _make_day_inputs(date_ts=_F009_DATE, regime_t=0, def_gross_return=0.0)
    ctx = _make_returns_ctx(
        gtt_active=True, def_gross=pd.Series([0.0], index=pd.DatetimeIndex([_F009_DATE]))
    )

    new = _apply_defensive_compounding(state, inputs, ctx)

    assert new.defensive_sleeve == state.defensive_sleeve
    assert new.leaps_pool == state.leaps_pool


def test_defensive_compounding_noop_when_def_gross_none() -> None:
    """No compounding when def_gross series is None (GTT active but no series)."""
    state = _make_portfolio_state({"VTI": 0.0, "VXUS": 0.0}, sleeve=10_000.0, pool=2_000.0)
    inputs = _make_day_inputs(date_ts=_F009_DATE, regime_t=0, def_gross_return=0.001)
    ctx = _make_returns_ctx(gtt_active=True, def_gross=None)

    assert _apply_defensive_compounding(state, inputs, ctx) is state


def test_defensive_compounding_non_sleeve_fields_unchanged() -> None:
    """Fields other than sleeve and pool are not mutated."""
    state = _make_portfolio_state(
        {"VTI": 500.0, "VXUS": 300.0}, sleeve=1_000.0, pool=200.0
    )
    inputs = _make_day_inputs(date_ts=_F009_DATE, regime_t=0, def_gross_return=0.005)
    ctx = _make_returns_ctx(
        gtt_active=True, def_gross=pd.Series([0.005], index=pd.DatetimeIndex([_F009_DATE]))
    )

    new = _apply_defensive_compounding(state, inputs, ctx)

    assert new.holdings == state.holdings
    assert new.leaps_value == state.leaps_value
    assert new.prev_total_nav == state.prev_total_nav
    assert new.prev_regime == state.prev_regime


# ---------------------------------------------------------------------------
# _compute_leaps_mtm
# ---------------------------------------------------------------------------

_F010_DATE_PURCHASE = pd.Timestamp("2023-01-03")
_F010_DATE_EXPIRY = pd.Timestamp("2025-01-17")
_F010_DATE_MTM = pd.Timestamp("2023-06-01")


def test_leaps_mtm_bug1_reentry_suppression() -> None:
    """Bug 1 regression: MTM is suppressed exactly on re-entry days."""
    contract = make_contract(purchase_date=_F010_DATE_PURCHASE, expiry=_F010_DATE_EXPIRY)
    ledger = make_ledger(contract)
    state = _make_portfolio_state(
        {"VTI": 50000.0}, leaps_ledger=ledger, leaps_value=0.0, prev_regime=0
    )
    inputs = _make_day_inputs(date_ts=_F010_DATE_MTM, regime_t=1, spot=200.0)
    ctx = _make_leaps_mtm_ctx(use_leaps=True, gtt_active=True)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == 0.0


def test_leaps_mtm_normal_long_day_not_suppressed() -> None:
    """On a normal Long day (prev_regime=1, regime_t=1), MTM fires normally."""
    contract = make_contract(purchase_date=_F010_DATE_PURCHASE, expiry=_F010_DATE_EXPIRY)
    ledger = make_ledger(contract)
    state = _make_portfolio_state(
        {"VTI": 50000.0}, leaps_ledger=ledger, leaps_value=0.0, prev_regime=1
    )
    inputs = _make_day_inputs(date_ts=_F010_DATE_MTM, regime_t=1, spot=200.0)
    ctx = _make_leaps_mtm_ctx(use_leaps=True, gtt_active=True, iv=0.20)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value > 0.0


def test_leaps_mtm_defensive_day_suppression() -> None:
    """MTM is suppressed on defensive days (regime_t=0) when gtt_active=True."""
    contract = make_contract(purchase_date=_F010_DATE_PURCHASE, expiry=_F010_DATE_EXPIRY)
    ledger = make_ledger(contract)
    state = _make_portfolio_state(
        {"VTI": 50000.0}, leaps_ledger=ledger, leaps_value=0.0, prev_regime=1
    )
    inputs = _make_day_inputs(date_ts=_F010_DATE_MTM, regime_t=0, spot=200.0)
    ctx = _make_leaps_mtm_ctx(use_leaps=True, gtt_active=True)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == 0.0


def test_leaps_mtm_noop_when_use_leaps_false() -> None:
    """MTM returns leaps_value=0.0 when use_leaps is False."""
    contract = make_contract(purchase_date=_F010_DATE_PURCHASE, expiry=_F010_DATE_EXPIRY)
    ledger = make_ledger(contract)
    state = _make_portfolio_state(
        {"VTI": 50000.0}, leaps_ledger=ledger, leaps_value=0.0, prev_regime=1
    )
    inputs = _make_day_inputs(date_ts=_F010_DATE_MTM, regime_t=1, spot=200.0)
    ctx = _make_leaps_mtm_ctx(use_leaps=False, gtt_active=False)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == 0.0


def test_leaps_mtm_noop_when_ledger_none() -> None:
    """MTM returns leaps_value=0.0 when leaps_ledger is None."""
    state = _make_portfolio_state(
        {"VTI": 50000.0}, leaps_ledger=None, leaps_value=0.0, prev_regime=1
    )
    inputs = _make_day_inputs(date_ts=_F010_DATE_MTM, regime_t=1, spot=200.0)
    ctx = _make_leaps_mtm_ctx(use_leaps=True, gtt_active=False)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == 0.0


def test_leaps_mtm_noop_when_underlying_prices_none() -> None:
    """MTM returns leaps_value=0.0 when ctx.underlying_prices is None."""
    contract = make_contract(purchase_date=_F010_DATE_PURCHASE, expiry=_F010_DATE_EXPIRY)
    ledger = make_ledger(contract)
    state = _make_portfolio_state(
        {"VTI": 50000.0}, leaps_ledger=ledger, leaps_value=0.0, prev_regime=1
    )
    inputs = _make_day_inputs(date_ts=_F010_DATE_MTM, regime_t=1, spot=200.0)
    ctx = _make_leaps_mtm_ctx(use_leaps=True, gtt_active=False)
    ctx_none = dataclasses.replace(ctx, underlying_prices=None)

    new_state = _compute_leaps_mtm(state, inputs, ctx_none)

    assert new_state.leaps_value == 0.0


def test_leaps_mtm_normal_long_day_value() -> None:
    """MTM is computed correctly on a normal Long day via price_leaps_contract."""
    contract = make_contract(purchase_date=_F010_DATE_PURCHASE, expiry=_F010_DATE_EXPIRY)
    ledger = make_ledger(contract)
    state = _make_portfolio_state(
        {"VTI": 50000.0}, leaps_ledger=ledger, leaps_value=0.0, prev_regime=1
    )
    spot, rfr, iv = 205.0, 0.04, 0.20
    inputs = _make_day_inputs(date_ts=_F010_DATE_MTM, regime_t=1, spot=spot, rfr=rfr)
    ctx = _make_leaps_mtm_ctx(use_leaps=True, gtt_active=False, iv=iv)

    expected = price_leaps_contract(contract, spot, _F010_DATE_MTM, iv, rfr)
    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == pytest.approx(expected, rel=1e-9)


def test_leaps_mtm_nan_iv_falls_back_to_ctx_iv() -> None:
    """When inputs.mtm_iv_value is NaN, day_iv equals ctx.iv."""
    contract = make_contract(purchase_date=_F010_DATE_PURCHASE, expiry=_F010_DATE_EXPIRY)
    ledger = make_ledger(contract)
    state = _make_portfolio_state(
        {"VTI": 50000.0}, leaps_ledger=ledger, leaps_value=0.0, prev_regime=1
    )
    spot, rfr, iv = 200.0, 0.04, 0.20
    inputs = _make_day_inputs(
        date_ts=_F010_DATE_MTM, regime_t=1, spot=spot, rfr=rfr, mtm_iv_value=float("nan")
    )
    ctx = _make_leaps_mtm_ctx(use_leaps=True, gtt_active=False, iv=iv)

    expected = price_leaps_contract(contract, spot, _F010_DATE_MTM, iv, rfr)
    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == pytest.approx(expected, rel=1e-9)


def test_leaps_mtm_none_iv_falls_back_to_ctx_iv() -> None:
    """When inputs.mtm_iv_value is None, day_iv equals ctx.iv."""
    contract = make_contract(purchase_date=_F010_DATE_PURCHASE, expiry=_F010_DATE_EXPIRY)
    ledger = make_ledger(contract)
    state = _make_portfolio_state(
        {"VTI": 50000.0}, leaps_ledger=ledger, leaps_value=0.0, prev_regime=1
    )
    spot, rfr, iv = 200.0, 0.04, 0.20
    inputs = _make_day_inputs(
        date_ts=_F010_DATE_MTM, regime_t=1, spot=spot, rfr=rfr, mtm_iv_value=None
    )
    ctx = _make_leaps_mtm_ctx(use_leaps=True, gtt_active=False, iv=iv)

    expected = price_leaps_contract(contract, spot, _F010_DATE_MTM, iv, rfr)
    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == pytest.approx(expected, rel=1e-9)


def test_leaps_mtm_smoothed_iv_used_when_larger() -> None:
    """When mtm_iv_value > ctx.iv, day_iv = mtm_iv_value (the larger value)."""
    contract = make_contract(purchase_date=_F010_DATE_PURCHASE, expiry=_F010_DATE_EXPIRY)
    ledger = make_ledger(contract)
    state = _make_portfolio_state(
        {"VTI": 50000.0}, leaps_ledger=ledger, leaps_value=0.0, prev_regime=1
    )
    spot, rfr, iv_floor, mtm_iv = 200.0, 0.04, 0.20, 0.35
    inputs = _make_day_inputs(
        date_ts=_F010_DATE_MTM, regime_t=1, spot=spot, rfr=rfr, mtm_iv_value=mtm_iv
    )
    ctx = _make_leaps_mtm_ctx(use_leaps=True, gtt_active=False, iv=iv_floor)

    expected = price_leaps_contract(contract, spot, _F010_DATE_MTM, mtm_iv, rfr)
    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == pytest.approx(expected, rel=1e-9)


def test_leaps_mtm_scale_applied() -> None:
    """leaps_scale={contract: 0.5} halves the contract's MTM contribution."""
    contract = make_contract(purchase_date=_F010_DATE_PURCHASE, expiry=_F010_DATE_EXPIRY)
    ledger = make_ledger(contract)
    state = _make_portfolio_state(
        {"VTI": 50000.0},
        leaps_ledger=ledger,
        leaps_scale={contract: 0.5},
        leaps_value=0.0,
        prev_regime=1,
    )
    spot, rfr, iv = 200.0, 0.04, 0.20
    inputs = _make_day_inputs(date_ts=_F010_DATE_MTM, regime_t=1, spot=spot, rfr=rfr)
    ctx = _make_leaps_mtm_ctx(use_leaps=True, gtt_active=False, iv=iv)

    full_price = price_leaps_contract(contract, spot, _F010_DATE_MTM, iv, rfr)
    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value == pytest.approx(full_price * 0.5, rel=1e-9)


@given(
    spot=st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    iv=st.floats(min_value=0.10, max_value=0.80, allow_nan=False, allow_infinity=False),
    strike_ratio=st.floats(min_value=0.50, max_value=0.90, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_leaps_mtm_hypothesis_value_nonneg(
    spot: float, iv: float, strike_ratio: float
) -> None:
    """For any valid spot/iv/strike combination, leaps_value is always >= 0.0."""
    strike = spot * strike_ratio
    purchase_date = pd.Timestamp("2022-01-03")
    expiry = pd.Timestamp("2024-01-17")
    mtm_date = pd.Timestamp("2023-06-01")
    contract = LeapsContract(
        purchase_date=purchase_date,
        expiry_date=expiry,
        strike=strike,
        spot_at_purchase=spot,
        premium_paid=max(spot * 0.20, 1.0),
        notional=spot * 100.0,
        n_contracts=1.0,
        account_type=AccountType.TAXABLE,
    )
    ledger = LeapsLedger(
        contracts=(contract,), roll_events=(), account_type=AccountType.TAXABLE
    )
    state = _make_portfolio_state(
        {"VTI": 50000.0}, leaps_ledger=ledger, leaps_value=0.0, prev_regime=1
    )
    inputs = _make_day_inputs(date_ts=mtm_date, regime_t=1, spot=spot, rfr=0.04)
    ctx = _make_leaps_mtm_ctx(use_leaps=True, gtt_active=False, iv=iv)

    new_state = _compute_leaps_mtm(state, inputs, ctx)

    assert new_state.leaps_value >= 0.0


# ---------------------------------------------------------------------------
# _compute_nav_before_contrib / _compute_port_return
# ---------------------------------------------------------------------------


def test_nav_before_known_state() -> None:
    """Known state: sum of all components equals expected NAV."""
    state = _make_portfolio_state(
        {"VTI": 50000.0, "VXUS": 20000.0}, leaps_value=8000.0, sleeve=5000.0, pool=2000.0
    )
    assert _compute_nav_before_contrib(state) == pytest.approx(85000.0)


def test_nav_before_all_zero() -> None:
    """All-zero state returns exactly 0.0."""
    state = _make_portfolio_state(
        {"VTI": 0.0, "VXUS": 0.0},
        leaps_value=0.0,
        sleeve=0.0,
        pool=0.0,
        prev_total_nav=1.0,
    )
    assert _compute_nav_before_contrib(state) == 0.0


def test_nav_before_no_gtt_leaps() -> None:
    """Without GTT/LEAPS, result equals sum of holdings alone."""
    holdings = {"VTI": 40000.0, "VXUS": 25000.0, "BND": 10000.0}
    state = _make_portfolio_state(holdings)
    assert _compute_nav_before_contrib(state) == pytest.approx(sum(holdings.values()))


def test_port_return_positive() -> None:
    """5% gain: nav_before=10500, prev=10000."""
    assert _compute_port_return(10500.0, 10000.0) == pytest.approx(0.05, rel=1e-14)


def test_port_return_negative() -> None:
    """5% loss: nav_before=9500, prev=10000."""
    assert _compute_port_return(9500.0, 10000.0) == pytest.approx(-0.05, rel=1e-14)


@given(
    holdings_vals=st.fixed_dictionaries(
        {"VTI": st.floats(0.0, 1e6), "VXUS": st.floats(0.0, 1e6)}
    ),
    leaps=st.floats(0.0, 1e5),
    sleeve=st.floats(0.0, 1e5),
    pool=st.floats(0.0, 1e5),
)
def test_nav_before_equals_sum_of_components(
    holdings_vals: dict[str, float], leaps: float, sleeve: float, pool: float
) -> None:
    """nav_before equals the explicit sum of all four components."""
    state = _make_portfolio_state(
        holdings_vals, leaps_value=leaps, sleeve=sleeve, pool=pool, prev_total_nav=1.0
    )
    expected = holdings_vals["VTI"] + holdings_vals["VXUS"] + leaps + sleeve + pool
    assert _compute_nav_before_contrib(state) == pytest.approx(expected, rel=1e-12)


@given(
    nav_before=st.floats(min_value=1.0, max_value=1e7),
    prev=st.floats(min_value=1.0, max_value=1e7),
)
def test_port_return_above_minus_one(nav_before: float, prev: float) -> None:
    """For any positive NAV inputs, port_return > -1.0."""
    assert _compute_port_return(nav_before, prev) > -1.0


# ---------------------------------------------------------------------------
# _apply_contribution
# ---------------------------------------------------------------------------


def test_contribution_noop_non_month_end() -> None:
    """Non-month-end day returns the original state unchanged."""
    state = _make_portfolio_state({"VTI": 1000.0, "VXUS": 500.0})
    inputs = _make_day_inputs(date_ts=_F012_DATE, is_month_end=False)
    ctx = _make_contribution_ctx()
    assert _apply_contribution(state, inputs, ctx, nav_before=1500.0) is state


def test_contribution_noop_empty_base_assets() -> None:
    """Empty base_assets tuple returns the original state unchanged."""
    state = _make_portfolio_state({"VTI": 1000.0, "VXUS": 500.0})
    inputs = _make_day_inputs(date_ts=_F012_DATE, is_month_end=True)
    ctx = _make_contribution_ctx(base_assets=())
    assert _apply_contribution(state, inputs, ctx, nav_before=1500.0) is state


def test_contribution_month_end_long_day_holdings_increase() -> None:
    """Long month-end: full contribution flows to holdings proportional to weights."""
    state = _make_portfolio_state({"VTI": 1000.0, "VXUS": 0.0})
    inputs = _make_day_inputs(date_ts=_F012_DATE, regime_t=1, is_month_end=True)
    ctx = _make_contribution_ctx(
        base_assets=("VTI", "VXUS"),
        base_target_w={"VTI": 1.0, "VXUS": 0.0},
        base_contribution=500.0,
    )
    result = _apply_contribution(state, inputs, ctx, nav_before=1000.0)

    assert abs(result.holdings["VTI"] - (1000.0 + 500.0)) < 1e-9
    assert result.defensive_sleeve == state.defensive_sleeve
    assert result.leaps_pool == state.leaps_pool


def test_contribution_month_end_defensive_governed_to_sleeve() -> None:
    """Defensive month-end: governed asset allocation goes to sleeve, not holdings."""
    state = _make_portfolio_state(
        {"VTI": 1000.0, "VXUS": 500.0}, sleeve=200.0
    )
    inputs = _make_day_inputs(date_ts=_F012_DATE, regime_t=0, is_month_end=True)
    ctx = _make_contribution_ctx(
        base_assets=("VTI", "VXUS"),
        base_target_w={"VTI": 0.7, "VXUS": 0.3},
        governed_base=("VTI",),
        gtt_active=True,
        base_contribution=500.0,
    )
    result = _apply_contribution(state, inputs, ctx, nav_before=1500.0)

    assert abs(result.holdings["VXUS"] - (500.0 + 150.0)) < 1e-9
    assert abs(result.defensive_sleeve - (200.0 + 350.0)) < 1e-9
    assert result.holdings.get("VTI", 0.0) == pytest.approx(1000.0)


def test_contribution_month_end_defensive_leaps_pool_receives_monthly() -> None:
    """Defensive month-end with use_leaps=True: leaps_pool receives leaps_monthly."""
    state = _make_portfolio_state({"VTI": 1000.0, "VXUS": 500.0}, pool=50.0)
    inputs = _make_day_inputs(date_ts=_F012_DATE, regime_t=0, is_month_end=True)
    ctx = _make_contribution_ctx(
        base_assets=("VTI", "VXUS"),
        base_target_w={"VTI": 1.0, "VXUS": 0.0},
        governed_base=("VTI",),
        gtt_active=True,
        use_leaps=True,
        base_contribution=500.0,
        leaps_monthly=100.0,
    )
    result = _apply_contribution(state, inputs, ctx, nav_before=1500.0)

    assert abs(result.leaps_pool - (50.0 + 100.0)) < 1e-9


def test_contribution_month_end_reentry_governed_to_holdings() -> None:
    """Re-entry day (regime_t=1): governed allocation goes to holdings, not sleeve."""
    state = _make_portfolio_state(
        {"VTI": 1000.0, "VXUS": 500.0}, sleeve=300.0
    )
    inputs = _make_day_inputs(date_ts=_F012_DATE, regime_t=1, is_month_end=True)
    ctx = _make_contribution_ctx(
        base_assets=("VTI", "VXUS"),
        base_target_w={"VTI": 0.7, "VXUS": 0.3},
        governed_base=("VTI",),
        gtt_active=True,
        base_contribution=500.0,
        leaps_monthly=100.0,
    )
    result = _apply_contribution(state, inputs, ctx, nav_before=1500.0)

    assert abs(result.holdings["VTI"] - (1000.0 + 0.7 * 500.0)) < 1e-9
    assert abs(result.holdings["VXUS"] - (500.0 + 0.3 * 500.0)) < 1e-9
    assert result.defensive_sleeve == state.defensive_sleeve
    assert result.leaps_pool == state.leaps_pool


def test_contribution_accounting_invariant_long_month_end() -> None:
    """Long month-end: sum(holdings) increases by exactly base_contribution."""
    holdings_start = {"VTI": 600.0, "VXUS": 400.0}
    state = _make_portfolio_state(dict(holdings_start))
    inputs = _make_day_inputs(date_ts=_F012_DATE, regime_t=1, is_month_end=True)
    contribution = 500.0
    ctx = _make_contribution_ctx(
        base_assets=("VTI", "VXUS"),
        base_target_w={"VTI": 0.6, "VXUS": 0.4},
        base_contribution=contribution,
    )
    result = _apply_contribution(state, inputs, ctx, nav_before=1000.0)

    assert abs(sum(result.holdings.values()) - (sum(holdings_start.values()) + contribution)) < 1e-9


# F-020: Bug 3 fix tests
def test_contribution_long_month_end_use_leaps_credits_leaps_value() -> None:
    """Bug 3 fix: Long month-end with use_leaps=True credits leaps_monthly to leaps_value."""
    state = _make_portfolio_state({"VTI": 1000.0, "VXUS": 500.0}, pool=0.0)
    inputs = _make_day_inputs(date_ts=_F012_DATE, regime_t=1, is_month_end=True)
    ctx = _make_contribution_ctx(
        base_assets=("VTI", "VXUS"),
        base_target_w={"VTI": 0.7, "VXUS": 0.3},
        use_leaps=True,
        base_contribution=500.0,
        leaps_monthly=100.0,
    )
    result = _apply_contribution(state, inputs, ctx, nav_before=1500.0)

    assert abs(result.leaps_value - (state.leaps_value + 100.0)) < 1e-9
    assert result.leaps_pool == state.leaps_pool


def test_contribution_use_leaps_false_pool_unchanged() -> None:
    """No regression: use_leaps=False leaves leaps_pool unchanged regardless of regime."""
    state = _make_portfolio_state({"VTI": 1000.0}, pool=50.0)
    inputs = _make_day_inputs(date_ts=_F012_DATE, regime_t=0, is_month_end=True)
    ctx = _make_contribution_ctx(
        base_assets=("VTI",),
        gtt_active=True,
        use_leaps=False,
        base_contribution=500.0,
        leaps_monthly=100.0,
    )
    result = _apply_contribution(state, inputs, ctx, nav_before=1500.0)

    assert result.leaps_pool == state.leaps_pool


def test_contribution_leaps_monthly_zero_pool_unchanged() -> None:
    """Edge case: leaps_monthly=0.0 leaves leaps_pool unchanged even when use_leaps=True."""
    state = _make_portfolio_state({"VTI": 1000.0}, pool=75.0)
    inputs = _make_day_inputs(date_ts=_F012_DATE, regime_t=1, is_month_end=True)
    ctx = _make_contribution_ctx(
        base_assets=("VTI",),
        use_leaps=True,
        base_contribution=500.0,
        leaps_monthly=0.0,
    )
    result = _apply_contribution(state, inputs, ctx, nav_before=1000.0)

    assert result.leaps_pool == state.leaps_pool


# ---------------------------------------------------------------------------
# _apply_rebalance
# ---------------------------------------------------------------------------


def test_rebalance_noop_neither_rebal_nor_month_end() -> None:
    """State is unchanged when is_rebal_date=False and is_month_end=False."""
    state = _make_portfolio_state({"VTI": 60_000.0, "VXUS": 25_000.0, "GLD": 15_000.0})
    ctx = _make_rebalance_ctx()
    inputs = _make_day_inputs(is_rebal_date=False, is_month_end=False)
    assert _apply_rebalance(state, inputs, ctx) is state


def test_rebalance_quarterly_nav_neutral() -> None:
    """QUARTERLY rebalance preserves sum(holdings) within 1e-9 (invariant A5)."""
    holdings_in = {"VTI": 60_000.0, "VXUS": 20_000.0, "GLD": 20_000.0}
    base_target_w = pd.Series({"VTI": 0.6, "VXUS": 0.2, "GLD": 0.2})
    state = _make_portfolio_state(holdings_in)
    ctx = _make_rebalance_ctx(
        base_assets=("VTI", "VXUS", "GLD"), base_target_w=base_target_w
    )
    inputs = _make_day_inputs(is_rebal_date=True)
    result = _apply_rebalance(state, inputs, ctx)
    assert sum(result.holdings.values()) == pytest.approx(
        sum(holdings_in.values()), rel=1e-9
    )


def test_rebalance_quarterly_weights_correct() -> None:
    """After QUARTERLY rebalance, each asset's weight matches base_target_w."""
    holdings_in = {"VTI": 50_000.0, "VXUS": 30_000.0, "GLD": 20_000.0}
    base_target_w = pd.Series({"VTI": 0.6, "VXUS": 0.2, "GLD": 0.2})
    state = _make_portfolio_state(holdings_in)
    ctx = _make_rebalance_ctx(
        base_assets=("VTI", "VXUS", "GLD"), base_target_w=base_target_w
    )
    inputs = _make_day_inputs(is_rebal_date=True)
    result = _apply_rebalance(state, inputs, ctx)
    total = sum(result.holdings.values())
    assert result.holdings["VTI"] / total == pytest.approx(0.6, rel=1e-9)
    assert result.holdings["VXUS"] / total == pytest.approx(0.2, rel=1e-9)
    assert result.holdings["GLD"] / total == pytest.approx(0.2, rel=1e-9)


def test_rebalance_quarterly_gtt_defensive_governed_swept() -> None:
    """On a defensive QUARTERLY rebalance day, governed holdings are zeroed into sleeve."""
    holdings_in = {"VTI": 60_000.0, "VXUS": 20_000.0, "GLD": 20_000.0}
    base_target_w = pd.Series({"VTI": 0.6, "VXUS": 0.2, "GLD": 0.2})
    sleeve_in = 5_000.0
    state = _make_portfolio_state(holdings_in, sleeve=sleeve_in)
    ctx = _make_rebalance_ctx(
        base_assets=("VTI", "VXUS", "GLD"),
        base_target_w=base_target_w,
        governed_base=("VTI",),
        gtt_active=True,
    )
    inputs = _make_day_inputs(is_rebal_date=True, regime_t=0)
    result = _apply_rebalance(state, inputs, ctx)

    assert result.holdings["VTI"] == 0.0
    base_nav = sum(holdings_in.values())
    expected_vti_rebalanced = base_nav * 0.6
    assert result.defensive_sleeve == pytest.approx(sleeve_in + expected_vti_rebalanced, rel=1e-9)


def test_rebalance_drift_not_triggered_noop() -> None:
    """DRIFT rebalance is a no-op when current weights are within the band."""
    holdings_in = {"VTI": 50_000.0, "VXUS": 50_000.0}
    base_target_w = pd.Series({"VTI": 0.5, "VXUS": 0.5})
    w = pd.Series({"VTI": 0.5, "VXUS": 0.5})
    state = _make_portfolio_state(holdings_in)
    ctx = _make_rebalance_ctx(
        base_assets=("VTI", "VXUS"),
        base_target_w=base_target_w,
        rebalance_rule=RebalanceRule.DRIFT,
        w=w,
    )
    inputs = _make_day_inputs(is_month_end=True)
    result = _apply_rebalance(state, inputs, ctx)
    assert result.holdings == holdings_in


def test_rebalance_drift_triggered_realigns_holdings() -> None:
    """DRIFT rebalance realigns holdings when an asset has drifted beyond the band."""
    holdings_in = {"VTI": 80_000.0, "VXUS": 10_000.0, "GLD": 10_000.0}
    base_target_w = pd.Series({"VTI": 0.6, "VXUS": 0.2, "GLD": 0.2})
    w = pd.Series({"VTI": 0.6, "VXUS": 0.2, "GLD": 0.2})
    state = _make_portfolio_state(holdings_in)
    ctx = _make_rebalance_ctx(
        base_assets=("VTI", "VXUS", "GLD"),
        base_target_w=base_target_w,
        rebalance_rule=RebalanceRule.DRIFT,
        w=w,
    )
    inputs = _make_day_inputs(is_month_end=True)
    result = _apply_rebalance(state, inputs, ctx)
    total = sum(result.holdings.values())
    assert result.holdings["VTI"] / total == pytest.approx(0.6, rel=1e-9)
    assert result.holdings["VXUS"] / total == pytest.approx(0.2, rel=1e-9)
    assert result.holdings["GLD"] / total == pytest.approx(0.2, rel=1e-9)


@given(
    vti=st.floats(min_value=1.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    vxus=st.floats(min_value=1.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    gld=st.floats(min_value=1.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    w_raw=st.lists(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=3, max_size=3,
    ),
)
@settings(max_examples=300)
def test_rebalance_quarterly_a5_sum_conservation(
    vti: float, vxus: float, gld: float, w_raw: list[float]
) -> None:
    """Hypothesis: QUARTERLY rebalance conserves sum(holdings) within 1e-9."""
    total_w = sum(w_raw)
    assume(total_w > 1e-6)
    assets = ("VTI", "VXUS", "GLD")
    norm_w = [x / total_w for x in w_raw]
    base_target_w = pd.Series(dict(zip(assets, norm_w, strict=False)))
    holdings_in = {"VTI": vti, "VXUS": vxus, "GLD": gld}
    state = _make_portfolio_state(holdings_in)
    ctx = _make_rebalance_ctx(base_assets=assets, base_target_w=base_target_w)
    inputs = _make_day_inputs(is_rebal_date=True)
    result = _apply_rebalance(state, inputs, ctx)
    assert sum(result.holdings.values()) == pytest.approx(
        sum(holdings_in.values()), rel=1e-9
    )


# ---------------------------------------------------------------------------
# _apply_gtt_reentry
# ---------------------------------------------------------------------------


def test_reentry_noop_long_day() -> None:
    """State is returned unchanged on a normal Long day (prev_regime=1, regime_t=1)."""
    ctx = _make_reentry_ctx()
    state = _make_portfolio_state({"VTI": 75_000.0}, prev_regime=1)
    inputs = _make_day_inputs(
        date_ts=_F014_RE_ENTRY_DATE, regime_t=1, spot=_F014_SPOT, rfr=_F014_RFR
    )
    assert _apply_gtt_reentry(state, inputs, ctx) is state


def test_reentry_noop_defensive_day() -> None:
    """State is returned unchanged on a defensive day (prev_regime=1, regime_t=0)."""
    ctx = _make_reentry_ctx()
    state = _make_portfolio_state({"VTI": 75_000.0}, prev_regime=1)
    inputs = _make_day_inputs(
        date_ts=_F014_RE_ENTRY_DATE, regime_t=0, spot=_F014_SPOT, rfr=_F014_RFR
    )
    assert _apply_gtt_reentry(state, inputs, ctx) is state


def test_reentry_noop_gtt_inactive() -> None:
    """State is returned unchanged when gtt_active=False."""
    ctx = _make_reentry_ctx(gtt_active=False)
    state = _make_portfolio_state({"VTI": 75_000.0}, prev_regime=0)
    inputs = _make_day_inputs(
        date_ts=_F014_RE_ENTRY_DATE, regime_t=1, spot=_F014_SPOT, rfr=_F014_RFR
    )
    assert _apply_gtt_reentry(state, inputs, ctx) is state


def test_reentry_a2_nav_neutral() -> None:
    """After re-entry, sum(holdings) + leaps_value == total within 1e-9 (A2)."""
    ctx = _make_reentry_ctx(leaps_fraction=0.15)
    state = _make_portfolio_state(
        {"VTI": 75_000.0}, sleeve=20_000.0, pool=5_000.0, prev_regime=0
    )
    inputs = _make_day_inputs(
        date_ts=_F014_RE_ENTRY_DATE, regime_t=1, spot=_F014_SPOT, rfr=_F014_RFR
    )
    total_before = sum(state.holdings.values()) + state.defensive_sleeve + state.leaps_pool

    result = _apply_gtt_reentry(state, inputs, ctx)

    reconstructed_nav = sum(result.holdings.values()) + result.leaps_value
    assert reconstructed_nav == pytest.approx(total_before, abs=1e-9)
    assert result.defensive_sleeve == pytest.approx(0.0)
    assert result.leaps_pool == pytest.approx(0.0)


def test_reentry_a4_leaps_value_equals_capital_deployed() -> None:
    """leaps_value == total * leaps_fraction within 1e-6 (A4, Bug 2 regression test)."""
    leaps_fraction = 0.15
    ctx = _make_reentry_ctx(leaps_fraction=leaps_fraction)
    state = _make_portfolio_state(
        {"VTI": 75_000.0}, sleeve=20_000.0, pool=5_000.0, prev_regime=0
    )
    inputs = _make_day_inputs(
        date_ts=_F014_RE_ENTRY_DATE, regime_t=1, spot=_F014_SPOT, rfr=_F014_RFR, raw_vix_value=None
    )
    total_before = sum(state.holdings.values()) + state.defensive_sleeve + state.leaps_pool

    result = _apply_gtt_reentry(state, inputs, ctx)

    expected_leaps_capital = total_before * leaps_fraction
    assert result.leaps_value == pytest.approx(expected_leaps_capital, rel=1e-6)


def test_reentry_bug2_elevated_vix_priced_at_creation_iv() -> None:
    """Elevated raw_vix_value at re-entry: creation_iv=max(raw_vix_value, ctx.iv)."""
    raw_vix_value = 0.50
    ctx = _make_reentry_ctx(leaps_fraction=0.15, iv=DEFAULT_IV)
    state = _make_portfolio_state(
        {"VTI": 75_000.0}, sleeve=20_000.0, pool=5_000.0, prev_regime=0
    )
    inputs = _make_day_inputs(
        date_ts=_F014_RE_ENTRY_DATE, regime_t=1, spot=_F014_SPOT,
        rfr=_F014_RFR, raw_vix_value=raw_vix_value
    )
    total_before = sum(state.holdings.values()) + state.defensive_sleeve + state.leaps_pool

    result = _apply_gtt_reentry(state, inputs, ctx)

    creation_iv = max(raw_vix_value, DEFAULT_IV)
    assert result.leaps_ledger is not None
    win_prices = _F014_PRICES.loc[_F014_RE_ENTRY_DATE:_F014_DATES[-1]]
    manual_ledger = run_leaps_simulation(
        win_prices,
        ctx.leaps_monthly,
        ctx.config.leaps_config,
        risk_free_series=_F014_RETURN_DATA.risk_free_rate,
        iv_series=None,
        initial_capital=total_before * 0.15,
    )
    expected_leaps_value = sum(
        price_leaps_contract(c, _F014_SPOT, _F014_RE_ENTRY_DATE, creation_iv, _F014_RFR)
        for c in _live_contracts(manual_ledger, _F014_RE_ENTRY_DATE)
    )
    assert result.leaps_value == pytest.approx(expected_leaps_value, rel=1e-6)


def test_reentry_leaps_scale_reset() -> None:
    """leaps_scale is reset to {} on re-entry (clears orphaned keys from old window)."""
    old_contract = LeapsContract(
        purchase_date=pd.Timestamp("2019-01-02"),
        expiry_date=pd.Timestamp("2021-01-15"),
        strike=180.0,
        spot_at_purchase=200.0,
        premium_paid=25.0,
        notional=20_000.0,
        n_contracts=1.0,
        account_type=AccountType.TAXABLE,
    )
    ctx = _make_reentry_ctx(leaps_fraction=0.15)
    state = _make_portfolio_state(
        {"VTI": 75_000.0},
        sleeve=20_000.0,
        pool=5_000.0,
        prev_regime=0,
        leaps_scale={old_contract: 0.5},
    )
    inputs = _make_day_inputs(
        date_ts=_F014_RE_ENTRY_DATE, regime_t=1, spot=_F014_SPOT, rfr=_F014_RFR
    )

    result = _apply_gtt_reentry(state, inputs, ctx)

    assert result.leaps_scale == {}


@settings(max_examples=30, deadline=10_000)
@given(
    leaps_fraction=st.floats(min_value=0.0, max_value=0.30, allow_nan=False),
    total_nav=st.floats(min_value=10_000.0, max_value=100_000.0, allow_nan=False),
)
def test_reentry_hypothesis_a2_nav_conservation(
    leaps_fraction: float, total_nav: float
) -> None:
    """A2: sum(holdings) + leaps_value == total within 1e-9 for any valid inputs."""
    holdings_val = total_nav * 0.75
    sleeve_val = total_nav * 0.15
    pool_val = total_nav * 0.10

    ctx = _make_reentry_ctx(leaps_fraction=leaps_fraction)
    state = _make_portfolio_state(
        {"VTI": holdings_val}, sleeve=sleeve_val, pool=pool_val, prev_regime=0
    )
    inputs = _make_day_inputs(
        date_ts=_F014_RE_ENTRY_DATE, regime_t=1, spot=_F014_SPOT,
        rfr=_F014_RFR, raw_vix_value=None
    )

    result = _apply_gtt_reentry(state, inputs, ctx)

    reconstructed_nav = sum(result.holdings.values()) + result.leaps_value
    expected_total = holdings_val + sleeve_val + pool_val
    assert abs(reconstructed_nav - expected_total) < 1e-9


# ---------------------------------------------------------------------------
# _compute_total_nav
# ---------------------------------------------------------------------------


def test_total_nav_known_state_exact_sum() -> None:
    """Sum of all four components equals expected value."""
    state = _make_portfolio_state(
        {"VTI": 50_000.0, "VXUS": 20_000.0},
        leaps_value=7_500.0,
        sleeve=10_000.0,
        pool=5_000.0,
    )
    assert _compute_total_nav(state) == pytest.approx(92_500.0)


def test_total_nav_all_zero_state() -> None:
    """All-zero state returns 0.0."""
    state = _make_portfolio_state({"VTI": 0.0}, leaps_value=0.0, sleeve=0.0, pool=0.0)
    assert _compute_total_nav(state) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _advance_state
# ---------------------------------------------------------------------------


def test_advance_state_prev_fields_updated() -> None:
    """prev_total_nav, prev_regime, prev_date_ts are updated from arguments."""
    state = _make_portfolio_state(
        {"VTI": 80_000.0},
        leaps_value=12_000.0,
        sleeve=5_000.0,
        pool=3_000.0,
        prev_total_nav=90_000.0,
        prev_regime=0,
        prev_date_ts=pd.Timestamp("2023-01-02"),
    )
    inputs = _make_day_inputs(date_ts=pd.Timestamp("2023-03-31"), regime_t=1)

    result = _advance_state(state, 100_000.0, inputs)

    assert result.prev_total_nav == pytest.approx(100_000.0)
    assert result.prev_regime == 1
    assert result.prev_date_ts == pd.Timestamp("2023-03-31")


def test_advance_state_other_fields_unchanged() -> None:
    """All fields other than prev_* carry forward unchanged."""
    state = _make_portfolio_state(
        {"VTI": 80_000.0},
        leaps_value=12_000.0,
        sleeve=5_000.0,
        pool=3_000.0,
    )
    inputs = _make_day_inputs(date_ts=_F015_DEFAULT_DATE)
    result = _advance_state(state, 100_000.0, inputs)

    assert result.holdings == {"VTI": 80_000.0}
    assert result.leaps_value == pytest.approx(12_000.0)
    assert result.defensive_sleeve == pytest.approx(5_000.0)
    assert result.leaps_pool == pytest.approx(3_000.0)
    assert result.leaps_ledger is state.leaps_ledger
    assert result.leaps_scale is state.leaps_scale
    assert result.all_window_ledgers is state.all_window_ledgers
    assert result.all_gtt_closes is state.all_gtt_closes


# ---------------------------------------------------------------------------
# _build_weight_row
# ---------------------------------------------------------------------------


def test_build_weight_row_base_only_sums_to_one() -> None:
    """Base-only portfolio: weight row sums to 1.0."""
    state = _make_portfolio_state({"VTI": 60_000.0, "VXUS": 40_000.0})
    ctx = _make_minimal_backtest_ctx(
        base_assets=("VTI", "VXUS"),
        w=pd.Series({"VTI": 0.6, "VXUS": 0.4}),
    )
    row = _build_weight_row(state, 100_000.0, ctx)
    assert sum(row.values()) == pytest.approx(1.0, rel=1e-9)


def test_build_weight_row_leaps_key_weight() -> None:
    """LEAPS key weight computed as leaps_value * (w[k]/leaps_fraction) / total_nav."""
    state = _make_portfolio_state({"VTI": 85_000.0}, leaps_value=15_000.0)
    w = pd.Series({"VTI": 0.85, "VTI_LEAPS": 0.15})
    ctx = _make_minimal_backtest_ctx(
        base_assets=("VTI",),
        leaps_keys=("VTI_LEAPS",),
        leaps_fraction=0.15,
        use_leaps=True,
        w=w,
    )
    row = _build_weight_row(state, 100_000.0, ctx)
    assert row["VTI_LEAPS"] == pytest.approx(0.15, rel=1e-9)


def test_build_weight_row_gtt_defensive_parked_decomposed() -> None:
    """GTT active: parked capital split across defensive_weights."""
    state = _make_portfolio_state({"VTI": 70_000.0}, sleeve=30_000.0)
    ctx = _make_minimal_backtest_ctx(
        base_assets=("VTI",),
        gtt_active=True,
        governed_base=("VTI",),
        defensive_weights={"BIL": 0.5, "R_f": 0.5},
        w=pd.Series({"VTI": 1.0}),
    )
    row = _build_weight_row(state, 100_000.0, ctx)
    assert row["BIL"] == pytest.approx(0.15, rel=1e-9)
    assert row["R_f"] == pytest.approx(0.15, rel=1e-9)


def test_build_weight_row_combined_sums_to_one() -> None:
    """Base + LEAPS + GTT defensive: weight row sums to 1.0."""
    state = _make_portfolio_state(
        {"VTI": 55_000.0}, leaps_value=15_000.0, sleeve=20_000.0, pool=10_000.0
    )
    w = pd.Series({"VTI": 0.55, "VTI_LEAPS": 0.15, "BIL": 0.30})
    ctx = _make_minimal_backtest_ctx(
        base_assets=("VTI",),
        leaps_keys=("VTI_LEAPS",),
        leaps_fraction=0.15,
        use_leaps=True,
        gtt_active=True,
        governed_base=("VTI",),
        defensive_weights={"BIL": 0.5, "R_f": 0.5},
        w=w,
    )
    row = _build_weight_row(state, 100_000.0, ctx)
    assert sum(row.values()) == pytest.approx(1.0, rel=1e-9)


def test_build_weight_row_zero_total_nav_returns_zero_weights() -> None:
    """total_nav <= 0 returns all-zero weights without division error."""
    state = _make_portfolio_state({"VTI": 0.0})
    ctx = _make_minimal_backtest_ctx()
    row = _build_weight_row(state, 0.0, ctx)
    assert all(v == 0.0 for v in row.values())


# ---------------------------------------------------------------------------
# _assemble_leaps_ledger
# ---------------------------------------------------------------------------


def test_assemble_single_window_no_gtt(
    sample_ledger: LeapsLedger, sample_contract: LeapsContract
) -> None:
    """Non-GTT run: returns per-window ledger unchanged."""
    state = _make_portfolio_state({"VTI": 100_000.0}, leaps_ledger=sample_ledger)
    ctx = _make_minimal_backtest_ctx(gtt_active=False, use_leaps=True)
    result = _assemble_leaps_ledger(state, ctx, pd.Timestamp("2025-01-17"))
    assert result is not None
    assert result.contracts == sample_ledger.contracts


def test_assemble_two_window_gtt(sample_contract: LeapsContract) -> None:
    """GTT active: contracts from both windows are concatenated."""
    contract_b = LeapsContract(
        purchase_date=pd.Timestamp("2024-01-02"),
        expiry_date=pd.Timestamp("2026-01-16"),
        strike=170.0,
        spot_at_purchase=210.0,
        premium_paid=50.0,
        notional=21000.0,
        n_contracts=1.5,
        account_type=AccountType.TAXABLE,
    )
    ledger_w1 = LeapsLedger(
        contracts=(sample_contract,), roll_events=(), account_type=AccountType.TAXABLE
    )
    ledger_w2 = LeapsLedger(
        contracts=(contract_b,), roll_events=(), account_type=AccountType.TAXABLE
    )
    leaps_config = LeapsConfig(iv=0.20, ltcg_rate=0.20)
    state = _make_portfolio_state(
        {"VTI": 100_000.0},
        leaps_ledger=ledger_w2,
        all_window_ledgers=(ledger_w1, ledger_w2),
    )
    ctx = _make_minimal_backtest_ctx(
        gtt_active=True, use_leaps=True, leaps_config=leaps_config
    )
    result = _assemble_leaps_ledger(state, ctx, pd.Timestamp("2025-12-31"))
    assert result is not None
    assert len(result.contracts) == len(ledger_w1.contracts) + len(ledger_w2.contracts)


def test_assemble_gtt_close_events_attached(
    sample_contract: LeapsContract, sample_gtt_close: LeapsGttCloseEvent
) -> None:
    """Assembled ledger has gtt_close_events == state.all_gtt_closes."""
    ledger = LeapsLedger(
        contracts=(sample_contract,), roll_events=(), account_type=AccountType.TAXABLE
    )
    leaps_config = LeapsConfig(iv=0.20, ltcg_rate=0.20)
    state = _make_portfolio_state(
        {"VTI": 100_000.0},
        leaps_ledger=ledger,
        all_window_ledgers=(ledger,),
        all_gtt_closes=(sample_gtt_close,),
    )
    ctx = _make_minimal_backtest_ctx(gtt_active=True, use_leaps=True, leaps_config=leaps_config)
    result = _assemble_leaps_ledger(state, ctx, pd.Timestamp("2025-12-31"))
    assert result is not None
    assert result.gtt_close_events == state.all_gtt_closes


def test_assemble_leaps_scale_frozen_into_partial_close_events(
    sample_contract: LeapsContract, sample_ledger: LeapsLedger
) -> None:
    """leaps_scale entry creates a LeapsPartialCloseEvent on the ledger."""
    surviving_fraction = 0.8
    original_n = sample_contract.n_contracts
    state = _make_portfolio_state(
        {"VTI": 100_000.0},
        leaps_ledger=sample_ledger,
        leaps_scale={sample_contract: surviving_fraction},
    )
    ctx = _make_minimal_backtest_ctx(gtt_active=False, use_leaps=True)
    result = _assemble_leaps_ledger(state, ctx, pd.Timestamp("2025-01-17"))
    assert result is not None
    assert len(result.partial_close_events) == 1
    event = result.partial_close_events[0]
    assert event.continuation_contract.n_contracts == pytest.approx(
        original_n * surviving_fraction, rel=1e-9
    )
