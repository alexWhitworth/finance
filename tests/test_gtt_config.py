"""Tests for GTT constants and GttConfig / PortfolioConfig validation (F-01, F-02)."""

from dataclasses import FrozenInstanceError

import pytest

from finance.consts import (
    GTT_DEFENSIVE_WEIGHTS_DEFAULT,
    GTT_EQUITY_TICKERS,
    GTT_SMA_WINDOW,
    GTT_UNRATE_TRADE_LAG_DAYS,
    GTT_VIX_CONSECUTIVE_DAYS,
)
from finance.leverage import RebalanceRule, WeightStrategy
from finance.portfolio import GttConfig, PortfolioConfig

# ---------------------------------------------------------------------------
# F-01: constants
# ---------------------------------------------------------------------------


def test_gtt_constants_exact_values_and_types() -> None:
    assert GTT_SMA_WINDOW == 200
    assert GTT_VIX_CONSECUTIVE_DAYS == 5
    assert GTT_UNRATE_TRADE_LAG_DAYS == 1
    assert isinstance(GTT_EQUITY_TICKERS, frozenset)
    assert GTT_EQUITY_TICKERS == frozenset({"VTI"})
    assert GTT_DEFENSIVE_WEIGHTS_DEFAULT == {
        "R_f": 0.25,
        "KMLM": 0.25,
        "VGIT": 0.25,
        "GLD": 0.25,
    }


# ---------------------------------------------------------------------------
# F-01: GttConfig construction & validation
# ---------------------------------------------------------------------------


def test_gttconfig_defaults() -> None:
    cfg = GttConfig(vix_p90_threshold=0.272)
    assert cfg.vix_p90_threshold == 0.272
    assert cfg.sma_window == 200
    assert cfg.vix_consecutive_days == 5
    assert cfg.unrate_trade_lag_days == 1
    assert cfg.defensive_weights == {
        "R_f": 0.25,
        "KMLM": 0.25,
        "VGIT": 0.25,
        "GLD": 0.25,
    }


def test_gttconfig_default_weights_are_independent_instances() -> None:
    # default_factory must not share a mutable dict across instances.
    a = GttConfig(vix_p90_threshold=0.272)
    b = GttConfig(vix_p90_threshold=0.272)
    assert a.defensive_weights is not b.defensive_weights


def test_gttconfig_is_frozen() -> None:
    cfg = GttConfig(vix_p90_threshold=0.272)
    with pytest.raises(FrozenInstanceError):
        cfg.vix_p90_threshold = 0.30  # type: ignore[misc]


def test_gttconfig_weights_sum_below_one_raises() -> None:
    with pytest.raises(ValueError, match=r"defensive_weights must sum to 1\.0"):
        GttConfig(vix_p90_threshold=0.272, defensive_weights={"R_f": 0.5, "GLD": 0.4})


def test_gttconfig_weights_sum_above_one_raises() -> None:
    with pytest.raises(ValueError, match=r"defensive_weights must sum to 1\.0"):
        GttConfig(vix_p90_threshold=0.272, defensive_weights={"R_f": 0.6, "GLD": 0.5})


def test_gttconfig_weights_within_tolerance_succeeds() -> None:
    # Sum drifts by < 1e-6 → accepted.
    cfg = GttConfig(
        vix_p90_threshold=0.272,
        defensive_weights={"R_f": 0.3333333, "KMLM": 0.3333333, "GLD": 0.3333334},
    )
    assert abs(sum(cfg.defensive_weights.values()) - 1.0) <= 1e-6


def test_gttconfig_all_cash_defensive_is_valid() -> None:
    cfg = GttConfig(vix_p90_threshold=0.272, defensive_weights={"R_f": 1.0})
    assert cfg.defensive_weights == {"R_f": 1.0}


# ---------------------------------------------------------------------------
# F-02: PortfolioConfig gtt_config field & validation
# ---------------------------------------------------------------------------

_TICKERS = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
_EQUAL_WEIGHTS = {t: 1.0 / len(_TICKERS) for t in _TICKERS}


def _portfolio_config(gtt_config: GttConfig | None) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights=dict(_EQUAL_WEIGHTS),
        initial_nav=1_000_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        gtt_config=gtt_config,
    )


def test_portfolioconfig_gtt_none_default() -> None:
    cfg = _portfolio_config(gtt_config=None)
    assert cfg.gtt_config is None


def test_portfolioconfig_valid_defensive_keys_present_in_target() -> None:
    # Default defensive sleeve (KMLM/VGIT/GLD) all exist in _EQUAL_WEIGHTS.
    gtt = GttConfig(vix_p90_threshold=0.272)
    cfg = _portfolio_config(gtt_config=gtt)
    assert cfg.gtt_config is gtt


def test_portfolioconfig_rf_only_sleeve_validates() -> None:
    # 'R_f' sentinel is exempt from the target_weights membership check.
    gtt = GttConfig(vix_p90_threshold=0.272, defensive_weights={"R_f": 1.0})
    cfg = _portfolio_config(gtt_config=gtt)
    assert cfg.gtt_config is gtt


def test_portfolioconfig_unknown_defensive_ticker_raises() -> None:
    gtt = GttConfig(
        vix_p90_threshold=0.272,
        defensive_weights={"R_f": 0.5, "TLT": 0.5},  # TLT absent from target_weights
    )
    with pytest.raises(ValueError, match=r"absent from target_weights.*TLT"):
        _portfolio_config(gtt_config=gtt)


def test_portfolioconfig_defensive_key_with_zero_target_weight_is_valid() -> None:
    # Key present in target_weights (even at 0 weight) passes membership check.
    weights = dict(_EQUAL_WEIGHTS)
    weights["GLD"] = 0.0
    # renormalize so target sums to 1.0
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    gtt = GttConfig(vix_p90_threshold=0.272, defensive_weights={"R_f": 0.5, "GLD": 0.5})
    cfg = PortfolioConfig(
        target_weights=weights,
        initial_nav=1_000_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        gtt_config=gtt,
    )
    assert cfg.gtt_config is gtt
