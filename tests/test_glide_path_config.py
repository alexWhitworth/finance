"""Tests for F-GP-01: GlidepathConfig dataclass and PortfolioConfig extension."""

import dataclasses

import pytest

from finance._portfolio_types import GlidepathConfig, PortfolioConfig
from finance.leverage import RebalanceRule, WeightStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_WEIGHTS_VTI = {
    "VTI": 0.0,
    "VTI_LEAPS": 0.40,
    "VXUS": 0.20,
    "GLD": 0.15,
    "MUB": 0.15,
    "KMLM": 0.05,
    "VGIT": 0.05,
}

_BASE_WEIGHTS_NO_VTI = {
    "VTI_LEAPS": 0.40,
    "VXUS": 0.20,
    "GLD": 0.15,
    "MUB": 0.15,
    "KMLM": 0.05,
    "VGIT": 0.05,
}

_GP_DEFAULT = GlidepathConfig()


def _make_config(
    weights: dict[str, float] = _BASE_WEIGHTS_VTI,
    rule: RebalanceRule = RebalanceRule.DRIFT,
    glide_path_config: GlidepathConfig | None = _GP_DEFAULT,
) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights=weights,
        initial_nav=100_000.0,
        monthly_contribution=1_000.0,
        rebalance_rule=rule,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        glide_path_config=glide_path_config,
    )


# ---------------------------------------------------------------------------
# GlidepathConfig: construction and defaults
# ---------------------------------------------------------------------------


def test_glidepath_config_defaults() -> None:
    """GlidepathConfig() constructs with defaults half_life_multiple=2.0, floor=0.05, vti_alpha=0.65."""
    gp = GlidepathConfig()
    assert gp.half_life_multiple == 2.0
    assert gp.floor == 0.05
    assert gp.vti_alpha == 0.65


def test_glidepath_config_custom_values() -> None:
    """GlidepathConfig accepts custom values for all three fields."""
    gp = GlidepathConfig(half_life_multiple=3.5, floor=0.0, vti_alpha=1.0)
    assert gp.half_life_multiple == 3.5
    assert gp.floor == 0.0
    assert gp.vti_alpha == 1.0


def test_glidepath_config_frozen() -> None:
    """GlidepathConfig is frozen: attribute assignment raises FrozenInstanceError."""
    gp = GlidepathConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        gp.floor = 0.10  # type: ignore[misc]


def test_glidepath_config_vti_alpha_zero_valid() -> None:
    """vti_alpha=0.0 is a valid degenerate: all freed weight to base assets."""
    gp = GlidepathConfig(vti_alpha=0.0)
    assert gp.vti_alpha == 0.0


def test_glidepath_config_vti_alpha_one_valid() -> None:
    """vti_alpha=1.0 is a valid degenerate: all freed weight to VTI."""
    gp = GlidepathConfig(vti_alpha=1.0)
    assert gp.vti_alpha == 1.0


def test_glidepath_config_floor_zero_valid() -> None:
    """floor=0.0 allows full de-lever to zero LEAPS."""
    gp = GlidepathConfig(floor=0.0)
    assert gp.floor == 0.0


# ---------------------------------------------------------------------------
# PortfolioConfig: valid glide-path construction
# ---------------------------------------------------------------------------


def test_portfolio_config_glide_path_valid() -> None:
    """PortfolioConfig with valid glide_path_config and DRIFT constructs without error."""
    cfg = _make_config()
    assert cfg.glide_path_config is not None
    assert cfg.rebalance_rule == RebalanceRule.DRIFT


def test_portfolio_config_glide_path_none_drift_unchanged() -> None:
    """PortfolioConfig with glide_path_config=None and DRIFT behaves identically to pre-change."""
    cfg = _make_config(glide_path_config=None)
    assert cfg.glide_path_config is None
    assert cfg.rebalance_rule == RebalanceRule.DRIFT


def test_portfolio_config_glide_path_none_quarterly_unchanged() -> None:
    """PortfolioConfig with glide_path_config=None and QUARTERLY constructs correctly."""
    weights = {
        "VTI": 0.60,
        "VXUS": 0.20,
        "GLD": 0.10,
        "MUB": 0.10,
    }
    cfg = PortfolioConfig(
        target_weights=weights,
        initial_nav=100_000.0,
        monthly_contribution=500.0,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        glide_path_config=None,
    )
    assert cfg.glide_path_config is None
    assert cfg.rebalance_rule == RebalanceRule.QUARTERLY


# ---------------------------------------------------------------------------
# PortfolioConfig: __post_init__ validation failures
# ---------------------------------------------------------------------------


def test_portfolio_config_glide_path_requires_drift() -> None:
    """PortfolioConfig with glide_path_config set and rebalance_rule != DRIFT raises ValueError."""
    with pytest.raises(ValueError, match="rebalance_rule=DRIFT"):
        _make_config(rule=RebalanceRule.QUARTERLY)


def test_portfolio_config_glide_path_requires_vti_in_weights() -> None:
    """PortfolioConfig with glide_path_config set and 'VTI' absent raises ValueError."""
    with pytest.raises(ValueError, match="'VTI' in target_weights"):
        _make_config(weights=_BASE_WEIGHTS_NO_VTI)


def test_portfolio_config_glide_path_floor_gte_leaps_fraction_raises() -> None:
    """glide_path_config.floor >= leaps_fraction (0.40) raises ValueError."""
    with pytest.raises(ValueError, match="floor"):
        _make_config(glide_path_config=GlidepathConfig(floor=0.40))


def test_portfolio_config_glide_path_floor_equals_leaps_fraction_raises() -> None:
    """floor exactly equal to leaps_fraction (0.40) raises ValueError (strict <)."""
    with pytest.raises(ValueError, match="floor"):
        _make_config(glide_path_config=GlidepathConfig(floor=0.40))


def test_portfolio_config_glide_path_floor_just_below_leaps_fraction_valid() -> None:
    """floor just below leaps_fraction (0.40 - epsilon) is valid."""
    gp = GlidepathConfig(floor=0.40 - 1e-9)
    cfg = _make_config(glide_path_config=gp)
    assert cfg.glide_path_config is not None


def test_portfolio_config_glide_path_half_life_zero_raises() -> None:
    """half_life_multiple == 0 raises ValueError."""
    with pytest.raises(ValueError, match="half_life_multiple"):
        _make_config(glide_path_config=GlidepathConfig(half_life_multiple=0.0))


def test_portfolio_config_glide_path_half_life_negative_raises() -> None:
    """half_life_multiple < 0 raises ValueError."""
    with pytest.raises(ValueError, match="half_life_multiple"):
        _make_config(glide_path_config=GlidepathConfig(half_life_multiple=-1.0))


def test_portfolio_config_glide_path_vti_alpha_above_one_raises() -> None:
    """vti_alpha > 1.0 raises ValueError."""
    with pytest.raises(ValueError, match="vti_alpha"):
        _make_config(glide_path_config=GlidepathConfig(vti_alpha=1.001))


def test_portfolio_config_glide_path_vti_alpha_below_zero_raises() -> None:
    """vti_alpha < 0.0 raises ValueError."""
    with pytest.raises(ValueError, match="vti_alpha"):
        _make_config(glide_path_config=GlidepathConfig(vti_alpha=-0.001))


def test_portfolio_config_glide_path_leaps_fraction_zero_raises() -> None:
    """leaps_fraction==0 with glide_path_config set violates floor < w0, raises ValueError."""
    weights_no_leaps = {
        "VTI": 0.0,
        "VXUS": 0.40,
        "GLD": 0.25,
        "MUB": 0.20,
        "KMLM": 0.10,
        "VGIT": 0.05,
    }
    with pytest.raises(ValueError, match="floor"):
        _make_config(weights=weights_no_leaps)
