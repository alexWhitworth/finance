"""Rebalance trigger logic — pure function determining when rebalancing should fire.

All business logic is pure: same inputs → same output, no side effects.
"""

import pandas as pd

from finance.consts import DRIFT_BAND_RELATIVE
from finance.leverage import RebalanceRule


def should_rebalance(
    current_weights: pd.Series,
    target_weights: pd.Series,
    rule: RebalanceRule,
    band: float = DRIFT_BAND_RELATIVE,
) -> bool:
    """Return True if rebalancing should be triggered under the given rule.

    QUARTERLY: always returns False — schedule is handled by the caller via
    rebalance date sets.
    DRIFT: returns True if any asset's relative weight deviation exceeds band.

    Relative deviation for asset i: |w_i - t_i| / t_i > band.
    Only assets present in both current_weights and target_weights are checked.
    Assets with a target weight of zero are skipped (division by zero guard).

    Arguments:
        current_weights: Realized portfolio weights at the check date.
        target_weights: Target weights from PortfolioConfig (need not be normalized).
        rule: RebalanceRule controlling the check logic.
        band: Relative drift threshold. Default DRIFT_BAND_RELATIVE (0.10 = ±10%).

    Returns:
        True if rebalancing is triggered, False otherwise.
    """
    if rule == RebalanceRule.QUARTERLY:
        return False
    common = current_weights.index.intersection(target_weights.index)
    for a in common:
        t = float(target_weights[a])
        if t == 0.0:
            continue
        if abs(float(current_weights[a]) - t) / t > band:
            return True
    return False
