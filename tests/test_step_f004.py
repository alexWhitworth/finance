"""Tests for F-004: _build_context extraction.

Covers all ValueError guards, frozenset types for rebal_dates and
month_end_dates, mtm_iv_series causality, and the no-GTT/no-LEAPS baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance._step_f004 import _build_context
from finance.consts import VIX_MTM_WINDOW
from finance.data import PriceData
from finance.gtt import GttSignalData
from finance.leverage import LeapsConfig, RebalanceRule, WeightStrategy
from finance.portfolio import GttConfig, PortfolioConfig
from finance.returns import ReturnData


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_dates(n: int = 60) -> pd.DatetimeIndex:
    """Return n business days starting 2020-01-02."""
    return pd.bdate_range("2020-01-02", periods=n)


def _make_return_data(
    dates: pd.DatetimeIndex,
    tickers: tuple[str, ...] = ("VTI",),
    seed: int = 0,
) -> ReturnData:
    """Build a minimal ReturnData for the given tickers and date index."""
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame(
        {t: rng.normal(0.0003, 0.01, len(dates)) for t in tickers},
        index=dates,
    )
    log_rets = pd.DataFrame(np.log1p(rets.values), index=dates, columns=rets.columns)
    rfr = pd.Series(0.04, index=dates, name="risk_free_rate")
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
# 1. ValueError: gtt_signal set but gtt_config is None
# ---------------------------------------------------------------------------


def test_gtt_signal_without_gtt_config_raises() -> None:
    """Providing gtt_signal without gtt_config raises ValueError."""
    dates = _make_dates()
    rd = _make_return_data(dates)
    pd_ = _make_price_data(dates)
    config = _make_config()  # gtt_config=None
    signal = _make_gtt_signal(dates)

    with pytest.raises(ValueError, match="gtt_signal and config.gtt_config must both be set"):
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
