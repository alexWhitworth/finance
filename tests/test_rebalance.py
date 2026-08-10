"""Tests for finance.rebalance — should_rebalance public API (F-002 / AC-002)."""

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from finance.leverage import RebalanceRule
from finance.rebalance import should_rebalance


# ---------------------------------------------------------------------------
# QUARTERLY rule — always False (I8)
# ---------------------------------------------------------------------------


def test_quarterly_rule_returns_false_when_not_rebal_date() -> None:
    """QUARTERLY rule must never trigger from weight drift alone (I8).

    The caller is responsible for consulting rebalance-date sets; should_rebalance
    itself always returns False for QUARTERLY so the drift path cannot fire.
    """
    current = pd.Series({"A": 0.70, "B": 0.30})
    target = pd.Series({"A": 0.50, "B": 0.50})
    assert should_rebalance(current, target, RebalanceRule.QUARTERLY) is False


def test_quarterly_rule_returns_false_even_at_extreme_drift() -> None:
    """QUARTERLY never triggers regardless of how far weights have drifted."""
    current = pd.Series({"A": 0.99, "B": 0.01})
    target = pd.Series({"A": 0.50, "B": 0.50})
    assert should_rebalance(current, target, RebalanceRule.QUARTERLY) is False


@given(
    a=st.floats(min_value=0.01, max_value=0.99),
)
@settings(max_examples=50)
def test_quarterly_rule_never_triggers_property(a: float) -> None:
    """Property: QUARTERLY rule always returns False for any weight pair."""
    current = pd.Series({"A": a, "B": 1.0 - a})
    target = pd.Series({"A": 0.50, "B": 0.50})
    assert should_rebalance(current, target, RebalanceRule.QUARTERLY) is False


# ---------------------------------------------------------------------------
# DRIFT rule — band boundary edge cases
# ---------------------------------------------------------------------------


def test_drift_triggers_when_exactly_above_band() -> None:
    """DRIFT fires when |w_i - t_i| / t_i strictly exceeds the band."""
    # target A = 0.50, band = 0.10; breach threshold at w_A > 0.55
    current = pd.Series({"A": 0.56, "B": 0.44})
    target = pd.Series({"A": 0.50, "B": 0.50})
    assert should_rebalance(current, target, RebalanceRule.DRIFT, band=0.10) is True


def test_drift_does_not_trigger_below_exact_band() -> None:
    """DRIFT does NOT fire when relative deviation is strictly below band.

    0.549 vs 0.50: deviation = 0.049/0.50 = 0.098 < 0.10.
    (Avoids the 0.55 floating-point edge where IEEE-754 yields > 0.10.)
    """
    current = pd.Series({"A": 0.549, "B": 0.451})
    target = pd.Series({"A": 0.50, "B": 0.50})
    assert should_rebalance(current, target, RebalanceRule.DRIFT, band=0.10) is False


def test_drift_does_not_trigger_below_band() -> None:
    """DRIFT does NOT fire when relative deviation is below the band."""
    current = pd.Series({"A": 0.54, "B": 0.46})
    target = pd.Series({"A": 0.50, "B": 0.50})
    assert should_rebalance(current, target, RebalanceRule.DRIFT, band=0.10) is False


def test_drift_zero_target_weight_skipped() -> None:
    """Assets with target_weight == 0 are skipped (division by zero guard)."""
    current = pd.Series({"A": 0.50, "B": 0.50})
    target = pd.Series({"A": 0.50, "B": 0.0})
    # B has target 0 so it must be skipped; A is on target → no trigger
    assert should_rebalance(current, target, RebalanceRule.DRIFT) is False


def test_drift_only_common_assets_checked() -> None:
    """Assets absent from either series are not checked."""
    current = pd.Series({"A": 0.99, "C": 0.01})  # C not in target
    target = pd.Series({"A": 0.99, "B": 0.01})   # B not in current
    # A is on target; neither B nor C appears in both → no trigger
    assert should_rebalance(current, target, RebalanceRule.DRIFT) is False


# ---------------------------------------------------------------------------
# Empty weights edge case
# ---------------------------------------------------------------------------


def test_drift_empty_weights_returns_false() -> None:
    """Empty weights Series returns False for DRIFT (no assets to check)."""
    current: pd.Series = pd.Series(dtype=float)
    target: pd.Series = pd.Series(dtype=float)
    assert should_rebalance(current, target, RebalanceRule.DRIFT) is False


# ---------------------------------------------------------------------------
# Public API importability (F-002 / AC-002)
# ---------------------------------------------------------------------------


def test_should_rebalance_importable_from_finance_rebalance() -> None:
    """should_rebalance is importable from finance.rebalance."""
    from finance.rebalance import should_rebalance as sr
    assert callable(sr)


def test_backtest_steps_no_longer_defines_should_rebalance() -> None:
    """_backtest_steps no longer owns the original _should_rebalance definition.

    The only reference remaining in _backtest_steps is the backward-compat alias
    that points to finance.rebalance.should_rebalance.
    """
    import finance._backtest_steps as bs
    import finance.rebalance as rm
    # The alias in _backtest_steps must resolve to the same object as rebalance.should_rebalance
    assert bs._should_rebalance is rm.should_rebalance
