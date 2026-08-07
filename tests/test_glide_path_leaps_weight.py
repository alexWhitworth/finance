"""Tests for F-GP-02: glide_path_leaps_weight() pure function."""

import math

import numpy as np
import pytest

from finance._backtest_steps import glide_path_leaps_weight

# Standard parameters used throughout
W0 = 0.40
FLOOR = 0.05
HLM = 2.0  # half_life_multiple

# ---------------------------------------------------------------------------
# Tabulated value assertions (hand-computed references)
# ---------------------------------------------------------------------------
# Formula: lam = ln(2)/2.0; w(m) = 0.05 + 0.35 * exp(-lam * max(m-1,0))


def _expected(m: float) -> float:
    lam = math.log(2.0) / HLM
    return FLOOR + (W0 - FLOOR) * math.exp(-lam * max(m - 1.0, 0.0))


@pytest.mark.parametrize(
    "m,expected",
    [
        (0.5,  W0),       # m < 1: full weight
        (1.0,  W0),       # m == 1: exp(0)==1 → w0 exactly
        (1.5,  _expected(1.5)),
        (2.0,  _expected(2.0)),   # == 0.05 + 0.35*exp(-ln2/2) = 0.05 + 0.35*0.7071 ≈ 0.2975
        (3.0,  FLOOR + (W0 - FLOOR) * math.exp(-math.log(2.0))),  # spec: == 0.225
        (10.0, _expected(10.0)),
    ],
)
def test_tabulated_values(m: float, expected: float) -> None:
    """Tabulated outputs at six m values match hand-computed references within 1e-12."""
    result = glide_path_leaps_weight(m, W0, FLOOR, HLM)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_m_at_break_even_equals_w0() -> None:
    """m == 1.0: result equals w0 exactly (exp(0) == 1)."""
    result = glide_path_leaps_weight(1.0, W0, FLOOR, HLM)
    np.testing.assert_allclose(result, W0, atol=1e-12)


def test_m_below_one_equals_w0() -> None:
    """m < 1.0: max(m-1,0)==0 so result == w0 exactly."""
    for m in (0.0, 0.01, 0.5, 0.9999):
        result = glide_path_leaps_weight(m, W0, FLOOR, HLM)
        np.testing.assert_allclose(result, W0, atol=1e-12, err_msg=f"m={m}")


def test_m_at_half_life_multiple_halves_active_weight() -> None:
    """At m == 1 + half_life_multiple, active weight (w0-floor) has halved."""
    m = 1.0 + HLM
    result = glide_path_leaps_weight(m, W0, FLOOR, HLM)
    expected = FLOOR + (W0 - FLOOR) / 2.0
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_spec_m3_value() -> None:
    """Spec acceptance criterion: m=3.0 == 0.05 + 0.35*exp(-ln2) == 0.225 within 1e-12."""
    result = glide_path_leaps_weight(3.0, W0, FLOOR, HLM)
    expected = FLOOR + (W0 - FLOOR) * math.exp(-math.log(2.0))
    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_allclose(result, 0.225, atol=1e-12)


def test_large_m_approaches_floor() -> None:
    """m == 1000: exp term rounds to 0.0; result == floor within 1e-15."""
    result = glide_path_leaps_weight(1000.0, W0, FLOOR, HLM)
    np.testing.assert_allclose(result, FLOOR, atol=1e-15)


# ---------------------------------------------------------------------------
# Bounds check: result always in [floor, w0]
# ---------------------------------------------------------------------------


def test_result_in_bounds_sweep() -> None:
    """Result is in [floor, w0] for all m in [0.0, 100.0] (1000 points)."""
    ms = np.linspace(0.0, 100.0, 1000)
    for m in ms:
        r = glide_path_leaps_weight(float(m), W0, FLOOR, HLM)
        assert FLOOR <= r <= W0, f"Out of bounds at m={m}: {r}"


# ---------------------------------------------------------------------------
# Monotonicity: non-increasing for m >= 1.0
# ---------------------------------------------------------------------------


def test_monotone_non_increasing_for_m_gte_1() -> None:
    """Result is monotone non-increasing for m in [1.0, 100.0] (1000 points)."""
    ms = np.linspace(1.0, 100.0, 1000)
    values = [glide_path_leaps_weight(float(m), W0, FLOOR, HLM) for m in ms]
    for i in range(1, len(values)):
        assert values[i] <= values[i - 1] + 1e-15, (
            f"Not non-increasing at i={i}: values[{i-1}]={values[i-1]}, "
            f"values[{i}]={values[i]}"
        )


# ---------------------------------------------------------------------------
# Boundary: just either side of m == 1.0
# ---------------------------------------------------------------------------


def test_boundary_just_below_one_equals_w0() -> None:
    """m = 1.0 - 1e-9: result == w0 (max clamps to 0)."""
    result = glide_path_leaps_weight(1.0 - 1e-9, W0, FLOOR, HLM)
    np.testing.assert_allclose(result, W0, atol=1e-12)


def test_boundary_just_above_one_less_than_w0() -> None:
    """m = 1.0 + 1e-9: result < w0 (decay has started)."""
    result = glide_path_leaps_weight(1.0 + 1e-9, W0, FLOOR, HLM)
    assert result < W0


# ---------------------------------------------------------------------------
# ValueError paths
# ---------------------------------------------------------------------------


def test_half_life_multiple_zero_raises() -> None:
    """half_life_multiple == 0 raises ValueError."""
    with pytest.raises(ValueError, match="half_life_multiple"):
        glide_path_leaps_weight(1.0, W0, FLOOR, 0.0)


def test_half_life_multiple_negative_raises() -> None:
    """half_life_multiple < 0 raises ValueError."""
    with pytest.raises(ValueError, match="half_life_multiple"):
        glide_path_leaps_weight(1.0, W0, FLOOR, -1.0)


def test_floor_equals_w0_raises() -> None:
    """floor == w0 raises ValueError (strict <)."""
    with pytest.raises(ValueError, match="floor"):
        glide_path_leaps_weight(1.0, W0, W0, HLM)


def test_floor_greater_than_w0_raises() -> None:
    """floor > w0 raises ValueError."""
    with pytest.raises(ValueError, match="floor"):
        glide_path_leaps_weight(1.0, W0, W0 + 0.01, HLM)


# ---------------------------------------------------------------------------
# Degenerate-but-valid cases
# ---------------------------------------------------------------------------


def test_floor_zero_valid() -> None:
    """floor=0.0 is valid; result approaches 0.0 for large m."""
    result = glide_path_leaps_weight(1.0, W0, 0.0, HLM)
    np.testing.assert_allclose(result, W0, atol=1e-12)
    result_large = glide_path_leaps_weight(1000.0, W0, 0.0, HLM)
    np.testing.assert_allclose(result_large, 0.0, atol=1e-15)


def test_small_half_life_multiple_fast_decay() -> None:
    """half_life_multiple=0.5: active weight halves at m=1.5 (one half-life from m=1)."""
    result = glide_path_leaps_weight(1.5, W0, FLOOR, 0.5)
    expected = FLOOR + (W0 - FLOOR) / 2.0
    np.testing.assert_allclose(result, expected, atol=1e-12)
