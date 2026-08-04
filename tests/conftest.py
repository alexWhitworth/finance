"""Shared pytest fixtures for the finance test suite.

Fixtures defined here are available to all test modules without explicit import.
"""

import numpy as np
import pandas as pd
import pytest

from finance._portfolio_types import (
    BacktestContext,
    DayInputs,
    GttConfig,
    PortfolioConfig,
    PortfolioState,
)
from finance.consts import DEFAULT_IV
from finance.data import PriceData
from finance.gtt import GttSignalData
from finance.leverage import (
    AccountType,
    LeapsConfig,
    LeapsContract,
    LeapsGttCloseEvent,
    LeapsLedger,
    RebalanceRule,
    WeightStrategy,
)
from finance.returns import ReturnData

# ---------------------------------------------------------------------------
# Generic builder helpers (used across multiple test modules)
# ---------------------------------------------------------------------------


def _make_dates(n: int = 60) -> pd.DatetimeIndex:
    """Return n business days starting 2020-01-02."""
    return pd.bdate_range("2020-01-02", periods=n)


def _make_return_data(
    dates: pd.DatetimeIndex,
    tickers: tuple[str, ...] = ("VTI",),
    seed: int = 0,
    rfr_value: float = 0.04,
) -> ReturnData:
    """Build a minimal ReturnData for the given tickers and date index.

    Arguments:
        dates: DatetimeIndex of trading days.
        tickers: Asset tickers to include.
        seed: RNG seed for reproducibility.
        rfr_value: Constant risk-free rate value (default 0.04 annualised).
            Pass ``0.04 / 252`` for daily-scaled tests (e.g. F-005).

    Returns:
        ReturnData with synthetic returns and a flat risk-free-rate series.
    """
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame(
        {t: rng.normal(0.0003, 0.01, len(dates)) for t in tickers},
        index=dates,
    )
    log_rets = pd.DataFrame(np.log1p(rets.values), index=dates, columns=rets.columns)
    rfr = pd.Series(rfr_value, index=dates, name="risk_free_rate")
    return ReturnData(
        returns=rets,
        log_returns=log_rets,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr,
    )


def _make_price_data(
    dates: pd.DatetimeIndex,
    tickers: tuple[str, ...] = ("VTI",),
    vol_tickers: tuple[str, ...] = (),
    seed: int = 1,
) -> PriceData:
    """Build a minimal PriceData with optional vol_prices."""
    rng = np.random.default_rng(seed)
    prices = pd.DataFrame(
        {t: 200.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates))) for t in tickers},
        index=dates,
    )
    if vol_tickers:
        vol_prices = pd.DataFrame(
            {t: 0.20 + rng.normal(0, 0.02, len(dates)) for t in vol_tickers},
            index=dates,
        )
    else:
        vol_prices = pd.DataFrame(index=dates)
    return PriceData(
        prices=prices,
        dividends=pd.DataFrame(index=dates),
        vol_prices=vol_prices,
        tickers=tickers,
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        spliced=False,
    )


def _make_config(
    weights: dict[str, float] | None = None,
    leaps_config: LeapsConfig | None = None,
    gtt_config: GttConfig | None = None,
) -> PortfolioConfig:
    """Build a PortfolioConfig; defaults to {VTI: 1.0}."""
    if weights is None:
        weights = {"VTI": 1.0}
    return PortfolioConfig(
        target_weights=weights,
        initial_nav=10_000.0,
        monthly_contribution=500.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=leaps_config,
        gtt_config=gtt_config,
    )


def _make_gtt_signal(dates: pd.DatetimeIndex, regime: int = 1) -> GttSignalData:
    """Build a minimal GttSignalData with a constant position mask."""
    mask = pd.Series(regime, index=dates, name="position_mask", dtype=int)
    return GttSignalData(
        position_mask=mask,
        ue_signal=pd.Series(0, index=dates),
        vix_signal=pd.Series(0, index=dates),
        vix_p90_threshold=0.272,
        unrate_start=pd.Timestamp(dates[0]),
        vix_start=pd.Timestamp(dates[0]),
    )


def _make_gtt_config(dates: pd.DatetimeIndex) -> GttConfig:
    """Build a minimal GttConfig whose defensive_weights only use R_f."""
    return GttConfig(
        vix_p90_threshold=0.272,
        defensive_weights={"R_f": 1.0},
    )


# ---------------------------------------------------------------------------
# F-005: BacktestContext builders
# ---------------------------------------------------------------------------


def _make_no_leaps_ctx(
    initial_nav: float = 100_000.0,
    weights: dict[str, float] | None = None,
) -> BacktestContext:
    """Build a BacktestContext with LEAPS disabled.

    Arguments:
        initial_nav: Starting portfolio value.
        weights: Target weights dict; defaults to {"VTI": 1.0}.

    Returns:
        BacktestContext with use_leaps=False and all optional series set to None.
    """
    if weights is None:
        weights = {"VTI": 1.0}
    dates = pd.bdate_range("2020-01-02", periods=60)
    config = PortfolioConfig(
        target_weights=weights,
        initial_nav=initial_nav,
        monthly_contribution=1000.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )
    return_data = _make_return_data(dates, tickers=tuple(weights.keys()), rfr_value=0.04 / 252)
    w = pd.Series(weights)
    base_target_w = w / w.sum()
    rebal_dates: frozenset[pd.Timestamp] = frozenset(
        {pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30")}
    )
    month_end_dates: frozenset[pd.Timestamp] = frozenset(
        {pd.Timestamp(d) for d in dates[::20]}
    )
    return BacktestContext(
        base_assets=tuple(weights.keys()),
        leaps_keys=(),
        leaps_fraction=0.0,
        base_target_w=base_target_w,
        governed_base=(),
        gtt_active=False,
        defensive_weights={},
        use_leaps=False,
        iv=0.18,
        leaps_monthly=0.0,
        base_contribution=1000.0,
        config=config,
        return_data=return_data,
        underlying_prices=None,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=None,
        mask_aligned=None,
        def_gross=None,
        rebal_dates=rebal_dates,
        month_end_dates=month_end_dates,
        long_window_end={pd.Timestamp(dates[0]): pd.Timestamp(dates[-1])},
        w=w,
    )


def _make_leaps_ctx(
    *,
    gtt_active: bool = False,
    mask_aligned: pd.Series | None = None,
    initial_nav: float = 100_000.0,
    n_periods: int = 126,
) -> BacktestContext:
    """Build a BacktestContext with LEAPS enabled.

    Arguments:
        gtt_active: Whether the GTT overlay is active.
        mask_aligned: GTT position mask (1=Long, 0=Defensive); required when
            gtt_active=True.
        initial_nav: Starting portfolio value.
        n_periods: Number of business days in the simulated period.

    Returns:
        BacktestContext with use_leaps=True, underlying_prices set, and leaps_config set.
    """
    dates = pd.bdate_range("2020-01-02", periods=n_periods)
    leaps_config = LeapsConfig(iv=0.22, ltcg_rate=0.20)
    config = PortfolioConfig(
        target_weights={"VTI": 0.85, "VTI_LEAPS": 0.15},
        initial_nav=initial_nav,
        monthly_contribution=1000.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=leaps_config,
    )
    return_data = _make_return_data(dates, rfr_value=0.04 / 252)
    rng = np.random.default_rng(42)
    prices = pd.Series(
        200.0 * np.cumprod(1 + rng.normal(0, 0.01, n_periods)),
        index=dates,
    )
    rfr = pd.Series(0.04 / 252, index=dates, name="risk_free_rate")
    w = pd.Series({"VTI": 0.85, "VTI_LEAPS": 0.15})
    base_target_w = pd.Series({"VTI": 1.0})
    long_window_end: dict[pd.Timestamp, pd.Timestamp] = {}
    if mask_aligned is not None:
        from finance._backtest_steps import _long_windows
        for start, end in _long_windows(mask_aligned):
            long_window_end[start] = end
    else:
        long_window_end = {pd.Timestamp(dates[0]): pd.Timestamp(dates[-1])}

    rebal_dates: frozenset[pd.Timestamp] = frozenset(
        {pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30")}
    )
    month_end_dates: frozenset[pd.Timestamp] = frozenset(
        {pd.Timestamp(d) for d in dates[::20]}
    )
    return BacktestContext(
        base_assets=("VTI",),
        leaps_keys=("VTI_LEAPS",),
        leaps_fraction=0.15,
        base_target_w=base_target_w,
        governed_base=("VTI",) if gtt_active else (),
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0} if gtt_active else {},
        use_leaps=True,
        iv=0.22,
        leaps_monthly=1000.0 * 0.15,
        base_contribution=1000.0 * 0.85,
        config=config,
        return_data=return_data,
        underlying_prices=prices,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=rfr,
        mask_aligned=mask_aligned,
        def_gross=None,
        rebal_dates=rebal_dates,
        month_end_dates=month_end_dates,
        long_window_end=long_window_end,
        w=w,
    )


# ---------------------------------------------------------------------------
# F-006: BacktestContext builder for _extract_day_inputs tests
# ---------------------------------------------------------------------------


def _make_extract_ctx(dates: pd.DatetimeIndex) -> BacktestContext:
    """Build a minimal BacktestContext with deterministic series for _extract_day_inputs.

    Arguments:
        dates: DatetimeIndex of trading days to populate.

    Returns:
        BacktestContext with GTT active, raw_vix, mtm_iv_series, mask (first half
        Long, second half Defensive), and all optional series populated.
    """
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({"VTI": rng.normal(0.001, 0.01, len(dates))}, index=dates)
    rfr = pd.Series(0.04, index=dates)
    vix = pd.Series(0.20 + rng.normal(0, 0.02, len(dates)), index=dates)
    mtm_iv = vix.rolling(30).mean().ffill()

    mask = pd.Series([1] * len(dates), index=dates, dtype=int)
    mask.iloc[len(dates) // 2 :] = 0

    config = PortfolioConfig(
        target_weights={"VTI": 1.0},
        initial_nav=10000.0,
        monthly_contribution=500.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )
    return_data = ReturnData(
        returns=returns,
        log_returns=pd.DataFrame({"VTI": np.log1p(returns["VTI"].values)}, index=dates),
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr,
    )
    month_ends = frozenset(
        pd.Timestamp(g.index[-1])
        for _, g in returns.groupby(pd.DatetimeIndex(dates).to_period("M"))
    )
    return BacktestContext(
        base_assets=("VTI",),
        leaps_keys=(),
        leaps_fraction=0.0,
        base_target_w=pd.Series({"VTI": 1.0}),
        governed_base=("VTI",),
        gtt_active=True,
        defensive_weights={"R_f": 1.0},
        use_leaps=False,
        iv=0.20,
        leaps_monthly=0.0,
        base_contribution=500.0,
        config=config,
        return_data=return_data,
        underlying_prices=None,
        raw_vix=vix,
        mtm_iv_series=mtm_iv,
        rfr_series=rfr,
        mask_aligned=mask,
        def_gross=rfr * 0,
        rebal_dates=frozenset(),
        month_end_dates=month_ends,
        long_window_end={},
        w=pd.Series({"VTI": 1.0}),
    )


# ---------------------------------------------------------------------------
# Generic PortfolioState / DayInputs builders
# ---------------------------------------------------------------------------


def _make_portfolio_state(
    holdings: dict[str, float],
    *,
    sleeve: float = 0.0,
    pool: float = 0.0,
    leaps_value: float = 0.0,
    prev_regime: int = 1,
    prev_date_ts: pd.Timestamp | None = None,
    leaps_ledger: LeapsLedger | None = None,
    leaps_scale: dict[LeapsContract, float] | None = None,
    all_window_ledgers: tuple[LeapsLedger, ...] = (),
    all_gtt_closes: tuple = (),
    prev_total_nav: float | None = None,
) -> PortfolioState:
    """Build a minimal PortfolioState with sensible defaults.

    Arguments:
        holdings: Dollar value per base asset.
        sleeve: defensive_sleeve value.
        pool: leaps_pool value.
        leaps_value: LEAPS mark-to-market value.
        prev_regime: GTT regime from the previous day (1=Long, 0=Defensive).
        prev_date_ts: Timestamp of the previous trading day.
        leaps_ledger: Active LEAPS ledger, or None.
        leaps_scale: Per-contract scale factors; defaults to empty dict.
        all_window_ledgers: Tuple of all per-window ledgers.
        all_gtt_closes: Tuple of all GTT force-close events.
        prev_total_nav: Prior NAV; defaults to sum of all components.

    Returns:
        PortfolioState with all fields populated.
    """
    total = sum(holdings.values()) + sleeve + pool + leaps_value
    return PortfolioState(
        holdings=holdings,
        defensive_sleeve=sleeve,
        leaps_pool=pool,
        leaps_value=leaps_value,
        prev_total_nav=prev_total_nav if prev_total_nav is not None else total,
        prev_regime=prev_regime,
        prev_date_ts=prev_date_ts,
        leaps_ledger=leaps_ledger,
        leaps_scale=leaps_scale if leaps_scale is not None else {},
        all_window_ledgers=all_window_ledgers,
        all_gtt_closes=all_gtt_closes,
    )


def _make_day_inputs(
    *,
    date_ts: pd.Timestamp = pd.Timestamp("2020-01-02"),
    day_ret: pd.Series | None = None,
    regime_t: int = 1,
    def_gross_return: float = 0.0,
    spot: float | None = None,
    raw_vix_value: float | None = None,
    mtm_iv_value: float | None = None,
    rfr: float = 0.04,
    is_month_end: bool = False,
    is_rebal_date: bool = False,
) -> DayInputs:
    """Build a minimal DayInputs with sensible defaults.

    Arguments:
        date_ts: Trading date.
        day_ret: Asset return Series; defaults to {"VTI": 0.0}.
        regime_t: GTT regime (1=Long, 0=Defensive).
        def_gross_return: Blended defensive return for today.
        spot: Underlying spot price (None when no LEAPS).
        raw_vix_value: Raw VIX value (None when no vol series).
        mtm_iv_value: Smoothed MTM IV (None or NaN when in warmup).
        rfr: Risk-free rate.
        is_month_end: Whether this is a month-end day.
        is_rebal_date: Whether this is a rebalance date.

    Returns:
        DayInputs with all fields populated.
    """
    if day_ret is None:
        day_ret = pd.Series({"VTI": 0.0})
    return DayInputs(
        date_ts=date_ts,
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


# ---------------------------------------------------------------------------
# F-007: BacktestContext builder for _apply_gtt_open tests
# ---------------------------------------------------------------------------

_GTT_OPEN_DATES = pd.bdate_range("2020-01-02", periods=10)
_GTT_OPEN_RFR = pd.Series(0.04, index=_GTT_OPEN_DATES)


def _make_gtt_open_ctx(
    gtt_active: bool = True,
    governed_base: tuple[str, ...] = ("VTI",),
) -> BacktestContext:
    """Build a minimal BacktestContext for _apply_gtt_open tests.

    Arguments:
        gtt_active: Whether the GTT overlay is active.
        governed_base: GTT-governed subset of base assets.

    Returns:
        BacktestContext with VTI/VXUS base assets and deterministic series.
    """
    returns = pd.DataFrame(
        {"VTI": [0.001] * 10, "VXUS": [0.001] * 10}, index=_GTT_OPEN_DATES
    )
    config = PortfolioConfig(
        target_weights={"VTI": 0.7, "VXUS": 0.3},
        initial_nav=10_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )
    return_data = ReturnData(
        returns=returns,
        log_returns=pd.DataFrame(
            {"VTI": np.log1p(0.001), "VXUS": np.log1p(0.001)},
            index=_GTT_OPEN_DATES,
        ),
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=_GTT_OPEN_RFR,
    )
    return BacktestContext(
        base_assets=("VTI", "VXUS"),
        leaps_keys=(),
        leaps_fraction=0.0,
        base_target_w=pd.Series({"VTI": 0.7, "VXUS": 0.3}),
        governed_base=governed_base,
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0},
        use_leaps=False,
        iv=0.20,
        leaps_monthly=0.0,
        base_contribution=0.0,
        config=config,
        return_data=return_data,
        underlying_prices=None,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=_GTT_OPEN_RFR,
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end={},
        w=pd.Series({"VTI": 0.7, "VXUS": 0.3}),
    )


# ---------------------------------------------------------------------------
# F-008 / F-010: LeapsContract / LeapsLedger builders (shared)
# ---------------------------------------------------------------------------


def make_contract(
    *,
    purchase_date: pd.Timestamp = pd.Timestamp("2020-01-02"),
    expiry: pd.Timestamp = pd.Timestamp("2022-01-21"),
    strike: float = 160.0,
    spot: float = 200.0,
    premium: float = 45.0,
    notional: float = 20000.0,
    n: float = 1.0,
    account_type: AccountType = AccountType.TAXABLE,
) -> LeapsContract:
    """Construct a minimal LeapsContract."""
    return LeapsContract(
        purchase_date=purchase_date,
        expiry_date=expiry,
        strike=strike,
        spot_at_purchase=spot,
        premium_paid=premium,
        notional=notional,
        n_contracts=n,
        account_type=account_type,
    )


def make_ledger(contract: LeapsContract) -> LeapsLedger:
    """Construct a minimal LeapsLedger with one contract."""
    return LeapsLedger(
        contracts=(contract,),
        roll_events=(),
        account_type=contract.account_type,
    )


# ---------------------------------------------------------------------------
# F-015: pytest fixtures for LeapsContract / LeapsLedger / LeapsGttCloseEvent
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_contract() -> LeapsContract:
    """Minimal LeapsContract for use in fixtures."""
    return LeapsContract(
        purchase_date=pd.Timestamp("2023-01-03"),
        expiry_date=pd.Timestamp("2025-01-17"),
        strike=160.0,
        spot_at_purchase=200.0,
        premium_paid=45.0,
        notional=20000.0,
        n_contracts=2.0,
        account_type=AccountType.TAXABLE,
    )


@pytest.fixture
def sample_ledger(sample_contract: LeapsContract) -> LeapsLedger:
    """Minimal LeapsLedger with one contract."""
    return LeapsLedger(
        contracts=(sample_contract,),
        roll_events=(),
        account_type=AccountType.TAXABLE,
    )


@pytest.fixture
def sample_gtt_close(sample_contract: LeapsContract) -> LeapsGttCloseEvent:
    """Minimal LeapsGttCloseEvent."""
    return LeapsGttCloseEvent(
        close_date=pd.Timestamp("2023-06-01"),
        contract=sample_contract,
        mtm_value=5000.0,
        gain_realized=500.0,
        tax_paid=100.0,
        net_proceeds=4900.0,
    )


# ---------------------------------------------------------------------------
# Generic pytest fixtures
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# F-008: BacktestContext builder for _apply_gtt_force_close tests
# ---------------------------------------------------------------------------

_F008_DATES = pd.bdate_range("2019-06-03", periods=400)


def _make_gtt_force_close_ctx(
    *,
    gtt_active: bool = True,
    with_leaps: bool = True,
    spot: float = 210.0,
    raw_vix_value: float = 0.25,
) -> BacktestContext:
    """Build a minimal BacktestContext for _apply_gtt_force_close tests.

    Arguments:
        gtt_active: Whether GTT overlay is active.
        with_leaps: Whether LEAPS overlay is included.
        spot: Constant spot price for underlying_prices series.
        raw_vix_value: Constant raw VIX value.

    Returns:
        BacktestContext with underlying_prices and raw_vix populated.
    """
    rng = np.random.default_rng(42)
    simple = rng.normal(0.0003, 0.01, len(_F008_DATES))
    returns = pd.DataFrame({"VTI": simple}, index=_F008_DATES)
    log_rets = pd.DataFrame({"VTI": np.log1p(simple)}, index=_F008_DATES)
    rfr_series = pd.Series(0.04, index=_F008_DATES, name="risk_free_rate")
    return_data = ReturnData(
        returns=returns,
        log_returns=log_rets,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr_series,
    )
    weights = {"VTI": 0.85, "VTI_LEAPS": 0.15} if with_leaps else {"VTI": 1.0}
    config = PortfolioConfig(
        target_weights=weights,
        initial_nav=100_000.0,
        monthly_contribution=500.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(iv=0.20, ltcg_rate=0.238) if with_leaps else None,
    )
    underlying_prices = pd.Series(spot, index=_F008_DATES, name="VTI") if with_leaps else None
    raw_vix = pd.Series(raw_vix_value, index=_F008_DATES, name="VIX")
    return BacktestContext(
        base_assets=("VTI",),
        leaps_keys=("VTI_LEAPS",) if with_leaps else (),
        leaps_fraction=0.15 if with_leaps else 0.0,
        base_target_w=pd.Series({"VTI": 1.0}),
        governed_base=("VTI",),
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0},
        use_leaps=with_leaps,
        iv=0.20,
        leaps_monthly=500.0 * 0.15 if with_leaps else 0.0,
        base_contribution=500.0 * (0.85 if with_leaps else 1.0),
        config=config,
        return_data=return_data,
        underlying_prices=underlying_prices,
        raw_vix=raw_vix,
        mtm_iv_series=None,
        rfr_series=pd.Series(0.04, index=_F008_DATES),
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end={},
        w=pd.Series({"VTI": 0.85, "VTI_LEAPS": 0.15} if with_leaps else {"VTI": 1.0}),
    )


# ---------------------------------------------------------------------------
# F-009: BacktestContext builder for _apply_returns / _apply_defensive_compounding
# ---------------------------------------------------------------------------

_F009_DATE = pd.Timestamp("2020-01-02")


def _make_returns_ctx(
    gtt_active: bool = False,
    def_gross: pd.Series | None = None,
) -> BacktestContext:
    """Build a minimal BacktestContext for _apply_returns / _apply_defensive_compounding tests.

    Arguments:
        gtt_active: Whether GTT overlay is active.
        def_gross: Precomputed defensive gross return series (default None).

    Returns:
        BacktestContext with VTI/VXUS base assets and a single-row date index.
    """
    idx = pd.DatetimeIndex([_F009_DATE])
    returns = pd.DataFrame({"VTI": [0.0], "VXUS": [0.0]}, index=idx)
    rfr = pd.Series([0.04], index=idx)
    return_data = ReturnData(
        returns=returns,
        log_returns=returns,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr,
    )
    config = PortfolioConfig(
        target_weights={"VTI": 0.5, "VXUS": 0.5},
        initial_nav=10_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )
    return BacktestContext(
        base_assets=("VTI", "VXUS"),
        leaps_keys=(),
        leaps_fraction=0.0,
        base_target_w=pd.Series({"VTI": 0.5, "VXUS": 0.5}),
        governed_base=("VTI",) if gtt_active else (),
        gtt_active=gtt_active,
        defensive_weights={"VTI": 1.0} if gtt_active else {},
        use_leaps=False,
        iv=0.18,
        leaps_monthly=0.0,
        base_contribution=0.0,
        config=config,
        return_data=return_data,
        underlying_prices=None,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=pd.Series([0.04], index=idx),
        mask_aligned=None,
        def_gross=def_gross,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end={},
        w=pd.Series({"VTI": 0.5, "VXUS": 0.5}),
    )


# ---------------------------------------------------------------------------
# F-010: BacktestContext builder for _compute_leaps_mtm tests
# ---------------------------------------------------------------------------

_F010_DATES = pd.bdate_range("2023-01-03", periods=130)


def _make_leaps_mtm_ctx(
    *,
    use_leaps: bool = True,
    gtt_active: bool = True,
    underlying_prices: pd.Series | None = None,
    iv: float = 0.20,
) -> BacktestContext:
    """Build a BacktestContext with controllable fields for _compute_leaps_mtm.

    Arguments:
        use_leaps: Whether LEAPS overlay is active.
        gtt_active: Whether GTT signal is active.
        underlying_prices: Spot price series; defaults to a constant 200.0 series.
        iv: IV floor.

    Returns:
        BacktestContext with sensible defaults for F-010 tests.
    """
    rng = np.random.default_rng(42)
    simple = rng.normal(0.0003, 0.01, len(_F010_DATES))
    returns = pd.DataFrame({"VTI": simple}, index=_F010_DATES)
    return_data = ReturnData(
        returns=returns,
        log_returns=pd.DataFrame({"VTI": np.log1p(simple)}, index=_F010_DATES),
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=pd.Series(0.04, index=_F010_DATES, name="risk_free_rate"),
    )
    if underlying_prices is None and use_leaps:
        underlying_prices = pd.Series(200.0, index=_F010_DATES, name="VTI")
    config = PortfolioConfig(
        target_weights={"VTI": 0.85, "VTI_LEAPS": 0.15} if use_leaps else {"VTI": 1.0},
        initial_nav=10_000.0,
        monthly_contribution=500.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(iv=0.20) if use_leaps else None,
    )
    return BacktestContext(
        base_assets=("VTI",),
        leaps_keys=("VTI_LEAPS",) if use_leaps else (),
        leaps_fraction=0.15 if use_leaps else 0.0,
        base_target_w=pd.Series({"VTI": 1.0}),
        governed_base=("VTI",) if gtt_active else (),
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0},
        use_leaps=use_leaps,
        iv=iv,
        leaps_monthly=500.0 * 0.15 if use_leaps else 0.0,
        base_contribution=500.0 * (0.85 if use_leaps else 1.0),
        config=config,
        return_data=return_data,
        underlying_prices=underlying_prices,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=None,
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end={},
        w=pd.Series({"VTI": 0.85, "VTI_LEAPS": 0.15}) if use_leaps else pd.Series({"VTI": 1.0}),
    )


# ---------------------------------------------------------------------------
# F-012: BacktestContext builder for _apply_contribution tests
# ---------------------------------------------------------------------------

_F012_DATE = pd.Timestamp("2024-01-31")


def _make_contribution_ctx(
    base_assets: tuple[str, ...] = ("VTI", "VXUS"),
    base_target_w: dict[str, float] | None = None,
    governed_base: tuple[str, ...] = (),
    gtt_active: bool = False,
    use_leaps: bool = False,
    base_contribution: float = 500.0,
    leaps_monthly: float = 0.0,
) -> BacktestContext:
    """Build a minimal BacktestContext for _apply_contribution tests.

    Arguments:
        base_assets: Tuple of base asset tickers.
        base_target_w: Weight dict over base_assets (sums to 1.0). Defaults to equal-weight.
        governed_base: GTT-governed subset of base_assets.
        gtt_active: Whether the GTT overlay is active.
        use_leaps: Whether LEAPS are configured (gates leaps_monthly credit).
        base_contribution: Monthly dollar contribution for base holdings.
        leaps_monthly: Monthly dollar contribution for LEAPS pool.

    Returns:
        BacktestContext with minimal fields needed for _apply_contribution.
    """
    if base_target_w is None:
        n = len(base_assets)
        base_target_w = dict.fromkeys(base_assets, 1.0 / n) if n > 0 else {}
    w_series = pd.Series(base_target_w)
    config_weights = (
        {a: 1.0 / len(base_assets) for a in base_assets} if base_assets else {"VTI": 1.0}
    )
    config = PortfolioConfig(
        target_weights=config_weights,
        initial_nav=10_000.0,
        monthly_contribution=base_contribution,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )
    idx = pd.DatetimeIndex([_F012_DATE])
    returns = pd.DataFrame({a: [0.0] for a in (base_assets or ("VTI",))}, index=idx)
    return_data = ReturnData(
        returns=returns,
        log_returns=returns.copy(),
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=pd.Series([0.0], index=idx),
    )
    return BacktestContext(
        base_assets=base_assets,
        leaps_keys=(),
        leaps_fraction=0.0,
        base_target_w=w_series,
        governed_base=governed_base,
        gtt_active=gtt_active,
        defensive_weights={},
        use_leaps=use_leaps,
        iv=0.25,
        leaps_monthly=leaps_monthly,
        base_contribution=base_contribution,
        config=config,
        return_data=return_data,
        underlying_prices=None,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=None,
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end={},
        w=w_series,
    )


# ---------------------------------------------------------------------------
# F-013: BacktestContext builder for _apply_rebalance tests
# ---------------------------------------------------------------------------

_F013_DATES = pd.bdate_range("2023-01-02", periods=30)


def _make_rebalance_ctx(
    *,
    base_assets: tuple[str, ...] = ("VTI", "VXUS", "GLD"),
    base_target_w: pd.Series | None = None,
    governed_base: tuple[str, ...] = (),
    gtt_active: bool = False,
    rebalance_rule: RebalanceRule = RebalanceRule.QUARTERLY,
    leaps_keys: tuple[str, ...] = (),
    leaps_fraction: float = 0.0,
    w: pd.Series | None = None,
) -> BacktestContext:
    """Build a minimal BacktestContext for _apply_rebalance tests.

    Arguments:
        base_assets: Tuple of base asset tickers.
        base_target_w: Weight Series over base_assets; defaults to equal-weight.
        governed_base: GTT-governed subset of base_assets.
        gtt_active: Whether GTT overlay is active.
        rebalance_rule: QUARTERLY or DRIFT.
        leaps_keys: LEAPS ticker keys.
        leaps_fraction: LEAPS fraction of total NAV.
        w: Full portfolio weight Series (base + LEAPS); defaults derived from base_target_w.

    Returns:
        BacktestContext with sensible defaults for F-013 tests.
    """
    if base_target_w is None:
        n = len(base_assets)
        base_target_w = pd.Series(dict.fromkeys(base_assets, 1.0 / n))
    if w is None:
        w_dict = {a: float(base_target_w[a]) * (1.0 - leaps_fraction) for a in base_assets}
        for k in leaps_keys:
            w_dict[k] = leaps_fraction / max(len(leaps_keys), 1)
        w = pd.Series(w_dict)
    rng = np.random.default_rng(0)
    simple = rng.normal(0.0003, 0.01, len(_F013_DATES))
    returns = pd.DataFrame({"VTI": simple}, index=_F013_DATES)
    return_data = ReturnData(
        returns=returns,
        log_returns=pd.DataFrame({"VTI": np.log1p(simple)}, index=_F013_DATES),
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=pd.Series(0.04, index=_F013_DATES),
    )
    config = PortfolioConfig(
        target_weights={str(k): float(v) for k, v in w.items()},
        initial_nav=100_000.0,
        monthly_contribution=1_000.0,
        rebalance_rule=rebalance_rule,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(iv=0.20, ltcg_rate=0.20) if leaps_keys else None,
    )
    return BacktestContext(
        base_assets=base_assets,
        leaps_keys=leaps_keys,
        leaps_fraction=leaps_fraction,
        base_target_w=base_target_w,
        governed_base=governed_base,
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0},
        use_leaps=len(leaps_keys) > 0,
        iv=0.20,
        leaps_monthly=0.0,
        base_contribution=1_000.0,
        config=config,
        return_data=return_data,
        underlying_prices=None,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=None,
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset({pd.Timestamp("2023-03-31")}),
        month_end_dates=frozenset({pd.Timestamp("2023-01-31")}),
        long_window_end={},
        w=w,
    )


# ---------------------------------------------------------------------------
# F-014: shared corpus + BacktestContext builder for _apply_gtt_reentry tests
# ---------------------------------------------------------------------------

_F014_RNG = np.random.default_rng(42)
_F014_DATES = pd.bdate_range("2020-01-02", periods=252)
_F014_PRICES = pd.Series(
    200.0 * np.cumprod(1 + _F014_RNG.normal(0, 0.01, len(_F014_DATES))),
    index=_F014_DATES,
)
_F014_RETURNS_DF = pd.DataFrame({"VTI": _F014_PRICES.pct_change().dropna()})
_F014_RETURN_DATA = ReturnData(
    returns=_F014_RETURNS_DF,
    log_returns=np.log(1 + _F014_RETURNS_DF),
    tey_adjusted=False,
    marginal_rate=0.0,
    risk_free_rate=pd.Series(0.04, index=_F014_DATES),
)
_F014_LEAPS_CONFIG = LeapsConfig(iv=DEFAULT_IV, ltcg_rate=0.238, account_type=AccountType.TAXABLE)
_F014_PORTFOLIO_CONFIG = PortfolioConfig(
    target_weights={"VTI": 0.85, "VTI_LEAPS": 0.15},
    initial_nav=100_000.0,
    monthly_contribution=500.0,
    rebalance_rule=RebalanceRule.QUARTERLY,
    weight_strategy=WeightStrategy.USER_SPECIFIED,
    leaps_config=_F014_LEAPS_CONFIG,
)
_F014_RE_ENTRY_DATE = _F014_DATES[10]
_F014_SPOT = float(_F014_PRICES.loc[_F014_RE_ENTRY_DATE])
_F014_RFR = 0.04
_F014_LONG_WINDOW_END: dict[pd.Timestamp, pd.Timestamp] = {
    _F014_RE_ENTRY_DATE: _F014_DATES[-1],
}


def _make_reentry_ctx(
    leaps_fraction: float = 0.15,
    use_leaps: bool = True,
    gtt_active: bool = True,
    iv: float = DEFAULT_IV,
    raw_vix: pd.Series | None = None,
) -> BacktestContext:
    """Build a minimal BacktestContext for _apply_gtt_reentry tests.

    Arguments:
        leaps_fraction: Fraction of NAV allocated to LEAPS.
        use_leaps: Whether LEAPS overlay is active.
        gtt_active: Whether GTT overlay is active.
        iv: IV floor for LEAPS pricing.
        raw_vix: Optional raw VIX series.

    Returns:
        BacktestContext using the shared F-014 price corpus.
    """
    return BacktestContext(
        base_assets=("VTI",),
        leaps_keys=("VTI_LEAPS",),
        leaps_fraction=leaps_fraction,
        base_target_w=pd.Series({"VTI": 1.0}),
        governed_base=("VTI",),
        gtt_active=gtt_active,
        defensive_weights={"R_f": 1.0},
        use_leaps=use_leaps,
        iv=iv,
        leaps_monthly=500.0 * leaps_fraction,
        base_contribution=500.0 * (1.0 - leaps_fraction),
        config=_F014_PORTFOLIO_CONFIG,
        return_data=_F014_RETURN_DATA,
        underlying_prices=_F014_PRICES,
        raw_vix=raw_vix,
        mtm_iv_series=None,
        rfr_series=pd.Series(0.04, index=_F014_DATES),
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end=_F014_LONG_WINDOW_END,
        w=pd.Series({"VTI": 0.85, "VTI_LEAPS": 0.15}),
    )


# ---------------------------------------------------------------------------
# F-015: BacktestContext builder for _compute_total_nav / _advance_state /
#        _build_weight_row / _assemble_leaps_ledger tests
# ---------------------------------------------------------------------------

_F015_DATES = pd.bdate_range("2023-01-03", periods=30)
_F015_DEFAULT_DATE = pd.Timestamp("2023-03-31")


def _make_minimal_backtest_ctx(
    *,
    base_assets: tuple[str, ...] = ("VTI",),
    leaps_keys: tuple[str, ...] = (),
    leaps_fraction: float = 0.0,
    gtt_active: bool = False,
    governed_base: tuple[str, ...] = (),
    defensive_weights: dict[str, float] | None = None,
    use_leaps: bool = False,
    w: pd.Series | None = None,
    leaps_config: LeapsConfig | None = None,
) -> BacktestContext:
    """Build a minimal BacktestContext for F-015 step-function unit tests.

    Arguments:
        base_assets: Tuple of base asset tickers.
        leaps_keys: LEAPS ticker keys.
        leaps_fraction: Fraction of NAV allocated to LEAPS.
        gtt_active: Whether GTT overlay is active.
        governed_base: GTT-governed subset of base_assets.
        defensive_weights: Weights for defensive allocation; defaults to {"R_f": 1.0}.
        use_leaps: Whether LEAPS overlay is active.
        w: Full portfolio weight Series; defaults to equal-weight over base_assets.
        leaps_config: LEAPS configuration, required when use_leaps=True.

    Returns:
        BacktestContext with minimal fields for F-015 tests.
    """
    if defensive_weights is None:
        defensive_weights = {"R_f": 1.0}
    if w is None:
        w = pd.Series({a: 1.0 / len(base_assets) for a in base_assets})
    target_weights: dict[str, float] = {a: float(w[a]) for a in w.index}
    config = PortfolioConfig(
        target_weights=target_weights,
        initial_nav=100_000.0,
        monthly_contribution=500.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=leaps_config,
    )
    rng = np.random.default_rng(0)
    simple = rng.normal(0.0003, 0.01, len(_F015_DATES))
    returns = pd.DataFrame({"VTI": simple}, index=_F015_DATES)
    return_data = ReturnData(
        returns=returns,
        log_returns=pd.DataFrame({"VTI": np.log1p(simple)}, index=_F015_DATES),
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=pd.Series(0.04, index=_F015_DATES),
    )
    base_target_w = pd.Series({a: 1.0 / len(base_assets) for a in base_assets})
    return BacktestContext(
        base_assets=base_assets,
        leaps_keys=leaps_keys,
        leaps_fraction=leaps_fraction,
        base_target_w=base_target_w,
        governed_base=governed_base,
        gtt_active=gtt_active,
        defensive_weights=defensive_weights,
        use_leaps=use_leaps,
        iv=0.20,
        leaps_monthly=0.0,
        base_contribution=500.0,
        config=config,
        return_data=return_data,
        underlying_prices=None,
        raw_vix=None,
        mtm_iv_series=None,
        rfr_series=None,
        mask_aligned=None,
        def_gross=None,
        rebal_dates=frozenset(),
        month_end_dates=frozenset(),
        long_window_end={},
        w=w,
    )


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Synthetic price DataFrame for 6 assets over 252 business days."""
    dates = _make_dates(252)
    rng = np.random.default_rng(42)
    tickers = ["VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT"]
    starts = {"VTI": 200.0, "VXUS": 60.0, "GLD": 170.0, "MUB": 55.0, "KMLM": 25.0, "VGIT": 65.0}
    data = {}
    for t in tickers:
        shocks = rng.normal(0.0003, 0.01, size=len(dates))
        prices = starts[t] * np.cumprod(1 + shocks)
        data[t] = prices
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Percent-change returns from sample_prices."""
    return sample_prices.pct_change().dropna()
