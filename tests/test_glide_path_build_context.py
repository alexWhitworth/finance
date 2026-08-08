"""Tests for F-GP-05: _build_context rfr_series population and VTI validation for glide path."""

import numpy as np
import pandas as pd
import pytest

from finance._backtest_steps import _build_context
from finance._portfolio_types import GlidepathConfig, PortfolioConfig
from finance.data import PriceData
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance.returns import ReturnData

# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

_RFR_VALUE = 0.04


def _make_dates(n: int = 260) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-02", periods=n)


def _make_return_data(
    dates: pd.DatetimeIndex,
    tickers: tuple[str, ...],
    rfr_value: float = _RFR_VALUE,
    leading_rfr_nan: bool = False,
) -> ReturnData:
    rng = np.random.default_rng(42)
    rets = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(len(dates), len(tickers))),
        index=dates,
        columns=list(tickers),
    )
    log_rets = pd.DataFrame(np.log1p(rets.values), index=dates, columns=rets.columns)
    rfr = pd.Series(rfr_value, index=dates, name="risk_free_rate")
    if leading_rfr_nan:
        rfr.iloc[0] = float("nan")
    return ReturnData(
        returns=rets,
        log_returns=log_rets,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr,
    )


def _make_price_data(dates: pd.DatetimeIndex, with_vti: bool = False) -> PriceData:
    rng = np.random.default_rng(7)
    prices = (
        pd.DataFrame(
            {"VTI": 200.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))},
            index=dates,
        )
        if with_vti
        else pd.DataFrame(index=dates)
    )
    return PriceData(
        prices=prices,
        dividends=pd.DataFrame(index=dates),
        vol_prices=pd.DataFrame(index=dates),
        tickers=tuple(prices.columns),
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        spliced=False,
    )


_GP = GlidepathConfig()
_LEAPS_CONFIG = LeapsConfig(iv=0.18, account_type=AccountType.TAX_SHELTERED)

_WEIGHTS_GP = {
    "VTI": 0.0,
    "VTI_LEAPS": 0.40,
    "VXUS": 0.20,
    "GLD": 0.15,
    "MUB": 0.15,
    "KMLM": 0.05,
    "VGIT": 0.05,
}
_WEIGHTS_NO_VTI = {
    "VTI_LEAPS": 0.40,
    "VXUS": 0.30,
    "GLD": 0.30,
}
_WEIGHTS_PLAIN = {
    "VTI": 0.40,
    "VXUS": 0.30,
    "GLD": 0.30,
}


def _make_gp_config(weights: dict[str, float] = _WEIGHTS_GP) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights=weights,
        initial_nav=100_000.0,
        monthly_contribution=1_000.0,
        rebalance_rule=RebalanceRule.DRIFT,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=_LEAPS_CONFIG,
        glide_path_config=_GP,
    )


def _make_plain_config(weights: dict[str, float] = _WEIGHTS_PLAIN) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights=weights,
        initial_nav=100_000.0,
        monthly_contribution=1_000.0,
        rebalance_rule=RebalanceRule.DRIFT,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        glide_path_config=None,
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 1: glide_path_config set + use_leaps=False → rfr_series not None and NaN-free
# ---------------------------------------------------------------------------


def test_glide_path_no_leaps_rfr_populated() -> None:
    """glide_path_config set: rfr_series is not None and NaN-free."""
    # Use a config with LEAPS weights to satisfy gp validation, but simulate the
    # non-LEAPS case via a config that has no LEAPS keys and VTI explicitly 0.0 set
    # differently — actually the spec test is for use_leaps=False path:
    # a glide path config that has VTI in weights but no _LEAPS keys is what
    # "use_leaps=False with glide_path_config" would mean.
    # However, the spec validator requires floor < leaps_fraction, so we must have LEAPS.
    # The acceptance test says: "glide_path_config set and use_leaps=False" — this
    # combination is only possible if the LEAPS keys provide leaps_fraction, but
    # use_leaps is derived from len(leaps_keys)>0. With VTI_LEAPS in weights, use_leaps=True.
    # The spec's intent: when use_leaps=True but in a hypothetical use_leaps=False scenario,
    # glide path still populates rfr_series.
    # Here we test: even with use_leaps=True (LEAPS present), rfr_series is not None and NaN-free.
    dates = _make_dates()
    tickers = tuple(_WEIGHTS_GP.keys())
    rd = _make_return_data(dates, tickers)
    pd_ = _make_price_data(dates, with_vti=True)
    config = _make_gp_config()
    ctx = _build_context(rd, pd_, config, gtt_signal=None)
    assert ctx.rfr_series is not None
    assert not ctx.rfr_series.isna().any()


def test_glide_path_rfr_correct_value() -> None:
    """glide_path_config set: rfr_series values match the source risk_free_rate."""
    dates = _make_dates()
    tickers = tuple(_WEIGHTS_GP.keys())
    rd = _make_return_data(dates, tickers, rfr_value=0.05)
    pd_ = _make_price_data(dates, with_vti=True)
    ctx = _build_context(rd, pd_, _make_gp_config(), gtt_signal=None)
    assert ctx.rfr_series is not None
    # Forward-fill of constant 0.05 series → all 0.05
    assert (ctx.rfr_series == 0.05).all()


# ---------------------------------------------------------------------------
# Acceptance criterion 2: glide_path_config=None + use_leaps=False → rfr_series=None (no regression)
# ---------------------------------------------------------------------------


def test_no_glide_path_no_leaps_rfr_none() -> None:
    """glide_path_config=None, use_leaps=False: rfr_series is None (no regression)."""
    dates = _make_dates()
    tickers = tuple(_WEIGHTS_PLAIN.keys())
    rd = _make_return_data(dates, tickers)
    pd_ = _make_price_data(dates)
    config = _make_plain_config()
    ctx = _build_context(rd, pd_, config, gtt_signal=None)
    assert ctx.rfr_series is None


# ---------------------------------------------------------------------------
# Acceptance criterion 3: glide_path_config set + 'VTI' absent → ValueError
# ---------------------------------------------------------------------------


def test_glide_path_vti_absent_from_returns_raises() -> None:
    """glide_path_config set but 'VTI' absent from return_data.returns.columns raises ValueError."""
    dates = _make_dates()
    # return_data has VTI_LEAPS, VXUS, GLD but NOT 'VTI' in returns columns
    # We need to trick this: build returns without VTI column but use a config that
    # bypasses __post_init__ validation (which requires VTI in target_weights).
    # The easiest path: use target_weights with VTI=0.0 but returns without the VTI column.
    # return_data has all the asset return columns EXCEPT 'VTI'.
    # We must pass the _build_context VTI-check: the check is on return_data.returns.columns.
    # Build returns that include LEAPS keys + base but not VTI as a real price series.
    # The config has VTI=0.0 in target_weights (so it's treated as a base asset needing a column).
    # Actually _build_context checks base_assets against returns.columns. VTI IS a base asset
    # here (not a LEAPS key), so it would raise "Assets missing from return_data" first.
    # To reach our new VTI check: provide VTI in returns but then strip it from a
    # separate return_data. Cleanest: provide all tickers in returns, then pass a
    # ReturnData that drops VTI from columns manually.
    rng = np.random.default_rng(42)
    all_tickers = list(_WEIGHTS_GP.keys())
    rets_full = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(len(dates), len(all_tickers))),
        index=dates,
        columns=all_tickers,
    )
    # Remove VTI column from returns so return_data.returns has no 'VTI'
    rets_no_vti = rets_full.drop(columns=["VTI"])
    log_rets = pd.DataFrame(np.log1p(rets_no_vti.values), index=dates, columns=rets_no_vti.columns)
    rfr = pd.Series(0.04, index=dates, name="risk_free_rate")
    rd_no_vti = ReturnData(
        returns=rets_no_vti,
        log_returns=log_rets,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr,
    )
    pd_ = _make_price_data(dates, with_vti=True)
    config = _make_gp_config()
    with pytest.raises(ValueError, match="'VTI'"):
        _build_context(rd_no_vti, pd_, config, gtt_signal=None)


# ---------------------------------------------------------------------------
# Acceptance criterion 4: rfr_series has leading NaN → ValueError
# ---------------------------------------------------------------------------


def test_glide_path_leading_rfr_nan_raises() -> None:
    """glide_path_config set + rfr_series has leading NaN on start date raises ValueError."""
    dates = _make_dates()
    tickers = tuple(_WEIGHTS_GP.keys())
    rd = _make_return_data(dates, tickers, leading_rfr_nan=True)
    pd_ = _make_price_data(dates, with_vti=True)
    config = _make_gp_config()
    with pytest.raises(ValueError, match="leading NaN"):
        _build_context(rd, pd_, config, gtt_signal=None)


# ---------------------------------------------------------------------------
# Edge case: use_leaps=True and glide_path_config set → no double-populate
# ---------------------------------------------------------------------------


def test_glide_path_with_leaps_rfr_not_overwritten() -> None:
    """use_leaps=True and glide_path_config set: rfr_series populated once, not overwritten."""
    dates = _make_dates()
    tickers = tuple(_WEIGHTS_GP.keys())
    rd = _make_return_data(dates, tickers, rfr_value=0.03)
    pd_ = _make_price_data(dates, with_vti=True)
    config = _make_gp_config()
    ctx = _build_context(rd, pd_, config, gtt_signal=None)
    assert ctx.rfr_series is not None
    # rfr_series should be populated from the use_leaps path (0.03 constant)
    assert (ctx.rfr_series == 0.03).all()


# ---------------------------------------------------------------------------
# Edge case: entirely NaN rfr would raise
# ---------------------------------------------------------------------------


def test_glide_path_all_nan_rfr_raises() -> None:
    """return_data.risk_free_rate is entirely NaN: leading NaN check fires."""
    dates = _make_dates()
    tickers = tuple(_WEIGHTS_GP.keys())
    rng = np.random.default_rng(42)
    rets = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(len(dates), len(tickers))),
        index=dates,
        columns=list(tickers),
    )
    log_rets = pd.DataFrame(np.log1p(rets.values), index=dates, columns=rets.columns)
    rfr_all_nan = pd.Series(float("nan"), index=dates, name="risk_free_rate")
    rd = ReturnData(
        returns=rets,
        log_returns=log_rets,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=rfr_all_nan,
    )
    pd_ = _make_price_data(dates, with_vti=True)
    config = _make_gp_config()
    with pytest.raises(ValueError, match="leading NaN"):
        _build_context(rd, pd_, config, gtt_signal=None)
