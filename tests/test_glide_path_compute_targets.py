"""Tests for F-GP-03: compute_glide_target_weights() pure function."""

import math

import numpy as np
import pandas as pd
import pytest

from finance._backtest_steps import compute_glide_target_weights, glide_path_leaps_weight
from finance._portfolio_types import GlidepathConfig, PortfolioConfig
from finance.leverage import RebalanceRule, WeightStrategy

# ---------------------------------------------------------------------------
# Standard fixtures
# ---------------------------------------------------------------------------

_TARGET_WEIGHTS = {
    "VTI": 0.0,
    "VTI_LEAPS": 0.40,
    "VXUS": 0.20,
    "GLD": 0.15,
    "MUB": 0.15,
    "KMLM": 0.05,
    "VGIT": 0.05,
}
_W0 = 0.40
_BASE_SUM = 0.60  # VXUS+GLD+MUB+KMLM+VGIT
_GP = GlidepathConfig(half_life_multiple=2.0, floor=0.05, vti_alpha=0.65)


def _make_config(
    weights: dict[str, float] | None = None,
    gp: GlidepathConfig = _GP,
) -> PortfolioConfig:
    w = weights if weights is not None else _TARGET_WEIGHTS
    return PortfolioConfig(
        target_weights=w,
        initial_nav=100_000.0,
        monthly_contribution=1_000.0,
        rebalance_rule=RebalanceRule.DRIFT,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        glide_path_config=gp,
    )


def _expected_at(m: float) -> dict[str, float]:
    """Hand-compute expected weights at given m for _TARGET_WEIGHTS + _GP."""
    w_leaps = glide_path_leaps_weight(m, _W0, _GP.floor, _GP.half_life_multiple)
    w_freed = _W0 - w_leaps
    vti = _GP.vti_alpha * w_freed
    base_scale = (1.0 - _GP.vti_alpha) * w_freed / _BASE_SUM
    return {
        "VTI": vti,
        "VTI_LEAPS": w_leaps,
        "VXUS": 0.20 + 0.20 * base_scale,
        "GLD": 0.15 + 0.15 * base_scale,
        "MUB": 0.15 + 0.15 * base_scale,
        "KMLM": 0.05 + 0.05 * base_scale,
        "VGIT": 0.05 + 0.05 * base_scale,
    }


# ---------------------------------------------------------------------------
# Sum invariant: result.sum() == 1.0 within 1e-12
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [0.5, 1.0, 1.5, 2.0, 5.0, 10.0])
def test_sum_equals_one(m: float) -> None:
    """result.sum() == 1.0 within 1e-12 at tabulated m values."""
    result = compute_glide_target_weights(m, _make_config(), _GP)
    np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12, err_msg=f"m={m}")


# ---------------------------------------------------------------------------
# Identity check: at m==1.0, result matches config.target_weights
# ---------------------------------------------------------------------------


def test_identity_at_m_one() -> None:
    """At m==1.0, result equals config.target_weights (with VTI==0.0) for all keys."""
    cfg = _make_config()
    result = compute_glide_target_weights(1.0, cfg, _GP)
    for k, v in _TARGET_WEIGHTS.items():
        np.testing.assert_allclose(
            float(result[k]), v, atol=1e-12, err_msg=f"key={k}"
        )


def test_identity_at_m_below_one() -> None:
    """At m < 1.0, w_freed==0 so result also equals config.target_weights."""
    cfg = _make_config()
    result = compute_glide_target_weights(0.5, cfg, _GP)
    for k, v in _TARGET_WEIGHTS.items():
        np.testing.assert_allclose(float(result[k]), v, atol=1e-12, err_msg=f"key={k}")


# ---------------------------------------------------------------------------
# Tabulated values at m=1.0, 2.0, 5.0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1.0, 2.0, 5.0])
def test_tabulated_weights(m: float) -> None:
    """Tabulated weights at m=1.0, 2.0, 5.0 match hand-computed references within 1e-12."""
    cfg = _make_config()
    result = compute_glide_target_weights(m, cfg, _GP)
    expected = _expected_at(m)
    for k, v in expected.items():
        np.testing.assert_allclose(float(result[k]), v, atol=1e-12, err_msg=f"m={m}, key={k}")


def test_vti_weight_at_m2_explicit() -> None:
    """At m==2.0, result['VTI'] == (w0 - glide_path_leaps_weight(m)) * vti_alpha within 1e-12."""
    cfg = _make_config()
    result = compute_glide_target_weights(2.0, cfg, _GP)
    w_leaps = glide_path_leaps_weight(2.0, _W0, _GP.floor, _GP.half_life_multiple)
    expected_vti = (_W0 - w_leaps) * _GP.vti_alpha
    np.testing.assert_allclose(float(result["VTI"]), expected_vti, atol=1e-12)


# ---------------------------------------------------------------------------
# Monotonicity: VTI and each base weight non-decreasing across m in [1.0, 10.0]
# ---------------------------------------------------------------------------


def test_monotonicity_vti_and_base_weights() -> None:
    """VTI weight and each base weight are non-decreasing across m in [1.0, 10.0] (100 points)."""
    cfg = _make_config()
    ms = np.linspace(1.0, 10.0, 100)
    results = [compute_glide_target_weights(float(m), cfg, _GP) for m in ms]
    monitored = ["VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT"]
    for k in monitored:
        vals = [float(r[k]) for r in results]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1] - 1e-14, (
                f"{k} not non-decreasing at i={i}: {vals[i-1]} -> {vals[i]}"
            )


def test_leaps_weight_non_increasing_in_m() -> None:
    """LEAPS weight is non-increasing across m in [1.0, 10.0]."""
    cfg = _make_config()
    ms = np.linspace(1.0, 10.0, 100)
    leaps_vals = [float(compute_glide_target_weights(float(m), cfg, _GP)["VTI_LEAPS"]) for m in ms]
    for i in range(1, len(leaps_vals)):
        assert leaps_vals[i] <= leaps_vals[i - 1] + 1e-14, (
            f"LEAPS not non-increasing at i={i}: {leaps_vals[i-1]} -> {leaps_vals[i]}"
        )


# ---------------------------------------------------------------------------
# All base weights >= original target_weights for m > 1.0
# ---------------------------------------------------------------------------


def test_base_weights_ge_original_for_m_gt_one() -> None:
    """All base asset weights >= their original target_weights for m > 1.0."""
    cfg = _make_config()
    base_keys = ["VXUS", "GLD", "MUB", "KMLM", "VGIT"]
    for m in [1.5, 2.0, 5.0, 10.0]:
        result = compute_glide_target_weights(m, cfg, _GP)
        for k in base_keys:
            assert float(result[k]) >= _TARGET_WEIGHTS[k] - 1e-14, (
                f"{k} at m={m}: {float(result[k])} < {_TARGET_WEIGHTS[k]}"
            )


# ---------------------------------------------------------------------------
# Edge cases: vti_alpha degenerate values
# ---------------------------------------------------------------------------


def test_vti_alpha_zero_all_freed_to_base() -> None:
    """vti_alpha==0.0: VTI weight is always 0.0 for all m."""
    gp = GlidepathConfig(half_life_multiple=2.0, floor=0.05, vti_alpha=0.0)
    cfg = _make_config(gp=gp)
    for m in [1.0, 2.0, 5.0]:
        result = compute_glide_target_weights(m, cfg, gp)
        np.testing.assert_allclose(float(result["VTI"]), 0.0, atol=1e-15, err_msg=f"m={m}")
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12, err_msg=f"m={m}")


def test_vti_alpha_one_all_freed_to_vti() -> None:
    """vti_alpha==1.0: all freed weight goes to VTI; base weights equal original at all m."""
    gp = GlidepathConfig(half_life_multiple=2.0, floor=0.05, vti_alpha=1.0)
    cfg = _make_config(gp=gp)
    base_keys = ["VXUS", "GLD", "MUB", "KMLM", "VGIT"]
    for m in [1.0, 2.0, 5.0]:
        result = compute_glide_target_weights(m, cfg, gp)
        for k in base_keys:
            np.testing.assert_allclose(
                float(result[k]), _TARGET_WEIGHTS[k], atol=1e-12,
                err_msg=f"{k} at m={m}"
            )
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12, err_msg=f"m={m}")


def test_m_below_one_vti_remains_zero() -> None:
    """m < 1.0: w_freed==0, result['VTI'] == 0.0."""
    cfg = _make_config()
    result = compute_glide_target_weights(0.5, cfg, _GP)
    np.testing.assert_allclose(float(result["VTI"]), 0.0, atol=1e-15)


def test_single_base_asset_proportional_trivially_correct() -> None:
    """Single base asset receives all freed non-VTI weight correctly."""
    weights = {"VTI": 0.0, "VTI_LEAPS": 0.40, "VXUS": 0.60}
    gp = GlidepathConfig(half_life_multiple=2.0, floor=0.05, vti_alpha=0.65)
    cfg = PortfolioConfig(
        target_weights=weights,
        initial_nav=100_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.DRIFT,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        glide_path_config=gp,
    )
    result = compute_glide_target_weights(2.0, cfg, gp)
    np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12)
    w_leaps = glide_path_leaps_weight(2.0, 0.40, 0.05, 2.0)
    w_freed = 0.40 - w_leaps
    expected_vti = 0.65 * w_freed
    expected_vxus = 0.60 + 0.35 * w_freed  # all non-VTI freed weight goes to VXUS
    np.testing.assert_allclose(float(result["VTI"]), expected_vti, atol=1e-12)
    np.testing.assert_allclose(float(result["VXUS"]), expected_vxus, atol=1e-12)


# ---------------------------------------------------------------------------
# ValueError paths
# ---------------------------------------------------------------------------


def test_vti_absent_from_target_weights_raises() -> None:
    """ValueError if 'VTI' not in config.target_weights."""
    weights_no_vti = {
        "VTI_LEAPS": 0.40,
        "VXUS": 0.30,
        "GLD": 0.30,
    }
    # Build a config without glide_path_config to bypass __post_init__ validation,
    # then call the function directly to test its own guard.
    cfg_raw = PortfolioConfig(
        target_weights=weights_no_vti,
        initial_nav=100_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.DRIFT,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        glide_path_config=None,
    )
    with pytest.raises(ValueError, match="'VTI'"):
        compute_glide_target_weights(1.0, cfg_raw, _GP)


def test_base_sum_zero_raises() -> None:
    """ValueError if base_sum == 0 (no non-LEAPS, non-VTI assets)."""
    weights_leaps_vti_only = {"VTI": 0.0, "VTI_LEAPS": 1.0}
    # bypass PortfolioConfig.__post_init__ glide_path validation (no gp field)
    cfg_raw = PortfolioConfig(
        target_weights=weights_leaps_vti_only,
        initial_nav=100_000.0,
        monthly_contribution=0.0,
        rebalance_rule=RebalanceRule.DRIFT,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        glide_path_config=None,
    )
    with pytest.raises(ValueError, match="base_sum"):
        compute_glide_target_weights(1.0, cfg_raw, _GP)
