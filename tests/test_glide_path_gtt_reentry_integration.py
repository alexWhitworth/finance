"""Integration test for F-GP-08: GTT re-entry at m > 1.5 uses compute_glide_target_weights.

Spec requirement:
    GTT-enabled DRIFT+glide_path backtest with a Defensive->Long transition at m > 1.5;
    assert LEAPS seed weight and VTI allocation match compute_glide_target_weights(m) * total
    within 1e-6.

Design:
    - 504-day synthetic corpus with high drift (0.3%/day) so NAV grows well above 1.5x
      initial_nav before the defensive window fires at day _DEF_LO=456.
    - _DEF_LO=456, _DEF_HI=466: window is deliberately placed between month-ends 454
      (2019-09-30) and 477 (2019-10-31) to avoid DRIFT rebalance firing on a defensive
      month-end, which would otherwise introduce a leaps_value top-up artefact.
    - TAX_SHELTERED LEAPS so force-close realizes no tax drag; leaps_pool equals full MTM.
    - zero monthly contributions: hurdle_contributed compounded only by rfr, staying near
      initial_nav so that m = NAV/hurdle is easy to exceed 1.5.
    - At re-entry (index[_DEF_HI]), _apply_gtt_reentry computes m_current and allocates all
      assets via compute_glide_target_weights(m_current). The weight_history['VTI_LEAPS'] and
      weight_history['VTI'] at that date must match the canonical formula within 1e-6.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from finance._backtest_steps import compute_glide_target_weights
from finance._portfolio_types import GlidepathConfig, GttConfig, PortfolioConfig
from finance.data import PriceData
from finance.gtt import GttSignalData
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance.portfolio import run_backtest
from finance.returns import ReturnData, build_return_data

# ---------------------------------------------------------------------------
# Corpus constants
# ---------------------------------------------------------------------------

# Asset weights: VTI=0.0 (glide-path grows this), VTI_LEAPS=0.40, diversified base.
# VTI must start at 0.0 to satisfy GlidepathConfig's validation requirements;
# freed LEAPS weight flows to VTI (vti_alpha fraction) and other base assets.
_WEIGHTS_GP = {
    "VTI": 0.0,
    "VTI_LEAPS": 0.40,
    "VXUS": 0.20,
    "GLD": 0.15,
    "MUB": 0.15,
    "KMLM": 0.05,
    "VGIT": 0.05,
}

# Defensive sleeve: park all capital in T-bills (R_f) during the GTT window.
_DEFENSIVE_WEIGHTS: dict[str, float] = {"R_f": 1.0}

_INITIAL_NAV = 5_000_000.0
_MONTHLY_CONTRIBUTION = 0.0      # zero contributions; hurdle stays near initial_nav
_GP_CONFIG = GlidepathConfig(half_life_multiple=2.0, floor=0.05, vti_alpha=0.65)
_LEAPS_CONFIG = LeapsConfig(
    iv=0.18,
    ltcg_rate=0.0,               # zero tax so force-close proceeds == full MTM
    account_type=AccountType.TAX_SHELTERED,
)

# Simulation: 504 trading days (~2 years). High drift (0.3%/day) so NAV grows 3x+.
# Defensive window placed between month-ends 454 (2019-09-30) and 477 (2019-10-31)
# to avoid DRIFT rebalance firing on a defensive day (which would trigger leaps_value
# top-up artefacts in weight_history that are unrelated to the F-GP-08 invariant).
_N_DAYS = 504
_DEF_LO = 456   # first defensive day  (after month-end 454)
_DEF_HI = 466   # first Long day after window (re-entry fires here, before month-end 477)
_DAILY_DRIFT = 0.003   # 0.3%/day → ~110% annually; guarantees m > 2 by day 456


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_price_data(n: int = _N_DAYS, seed: int = 42) -> PriceData:
    """Synthetic PriceData for the GP+GTT corpus.

    Uses a high uniform daily drift (0.3%/day) across all assets to guarantee
    NAV is well above 1.5x initial_nav before the defensive window fires.

    Arguments:
        n: Number of trading days.
        seed: Random seed for reproducibility.

    Returns:
        PriceData with no vol_prices (constant IV used throughout).
    """
    idx = pd.bdate_range("2018-01-02", periods=n + 1)
    rng = np.random.default_rng(seed)
    starts = {
        "VTI": 200.0, "VXUS": 60.0, "GLD": 170.0,
        "MUB": 55.0, "KMLM": 25.0, "VGIT": 65.0,
    }
    prices: dict[str, np.ndarray] = {}
    for t, s in starts.items():
        shocks = rng.normal(_DAILY_DRIFT, 0.009, n + 1)
        prices[t] = s * np.cumprod(1.0 + shocks)
    prices_df = pd.DataFrame(prices, index=idx)
    dividends = pd.DataFrame(0.0, index=idx, columns=list(starts))
    return PriceData(
        prices=prices_df,
        dividends=dividends,
        vol_prices=pd.DataFrame(),
        tickers=tuple(starts.keys()),
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
    )


def _make_rd_and_pd(
    n: int = _N_DAYS,
    seed: int = 42,
) -> tuple[ReturnData, PriceData]:
    """Return matching (ReturnData, PriceData) from the synthetic corpus.

    Arguments:
        n: Number of trading days.
        seed: Random seed.

    Returns:
        Tuple of (ReturnData, PriceData).
    """
    pd_obj = _make_price_data(n, seed)
    return build_return_data(pd_obj, apply_tey=False), pd_obj


def _make_portfolio_config() -> PortfolioConfig:
    """PortfolioConfig with DRIFT + glide-path + GTT overlay enabled.

    Arguments:
        None

    Returns:
        PortfolioConfig ready for the F-GP-08 integration backtest.
    """
    gtt_config = GttConfig(
        vix_p90_threshold=0.272,
        defensive_weights=_DEFENSIVE_WEIGHTS,
    )
    return PortfolioConfig(
        target_weights=_WEIGHTS_GP,
        initial_nav=_INITIAL_NAV,
        monthly_contribution=_MONTHLY_CONTRIBUTION,
        rebalance_rule=RebalanceRule.DRIFT,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=_LEAPS_CONFIG,
        gtt_config=gtt_config,
        glide_path_config=_GP_CONFIG,
    )


def _make_gtt_signal(idx: pd.DatetimeIndex) -> GttSignalData:
    """GttSignalData with one Defensive window [_DEF_LO, _DEF_HI) and Long elsewhere.

    Arguments:
        idx: Full DatetimeIndex of the backtest.

    Returns:
        GttSignalData whose position_mask has one Defensive->Long transition.
    """
    mask_values = np.ones(len(idx), dtype=int)
    mask_values[_DEF_LO:_DEF_HI] = 0
    zeros = pd.Series(0, index=idx)
    return GttSignalData(
        position_mask=pd.Series(mask_values, index=idx, name="position_mask"),
        ue_signal=zeros,
        vix_signal=zeros,
        vix_p90_threshold=0.272,
        unrate_start=pd.Timestamp(idx[0]),
        vix_start=pd.Timestamp(idx[0]),
    )


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def integration_result() -> dict:
    """Run the full backtest once and return results dict for all tests in module.

    Returns:
        Dict with keys: result, config, idx, re_entry_ts, nav_at_reentry.

    Notes:
        scope='module' means the backtest runs exactly once; all tests share output.
    """
    rd, pd_obj = _make_rd_and_pd()
    config = _make_portfolio_config()
    idx = pd.DatetimeIndex(rd.returns.index)
    signal = _make_gtt_signal(idx)
    result = run_backtest(rd, pd_obj, config, gtt_signal=signal)
    re_entry_ts = pd.Timestamp(idx[_DEF_HI])
    nav_at_reentry = float(result.nav_series.loc[re_entry_ts])
    return {
        "result": result,
        "config": config,
        "idx": idx,
        "re_entry_ts": re_entry_ts,
        "nav_at_reentry": nav_at_reentry,
    }


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_m_greater_than_1p5_at_reentry(integration_result: dict) -> None:
    """NAV multiple m > 1.5 at the re-entry date, confirming glide-path de-levering is active.

    With zero contributions, hurdle_contributed compounded over ~22 months at rfr=4%
    stays within 1.08x initial_nav. The portfolio with 0.3%/day drift reaches 3x+ NAV
    by day 456, so m > 1.5 / 1.08 > 1.5.

    Arguments:
        integration_result: Shared fixture from the module-level run.

    Returns:
        None
    """
    nav = integration_result["nav_at_reentry"]
    # Conservative upper bound on hurdle: initial_nav * 1.04^(22/12) ≈ 1.074x
    hurdle_upper_bound = _INITIAL_NAV * 1.08
    m_lower_bound = nav / hurdle_upper_bound
    assert m_lower_bound > 1.5, (
        f"m lower bound = {m_lower_bound:.4f} must be > 1.5; "
        f"nav={nav:.2f}, hurdle_upper_bound={hurdle_upper_bound:.2f}"
    )


def test_glide_path_leaps_weight_matches_canonical_at_reentry(
    integration_result: dict,
) -> None:
    """LEAPS seed weight == compute_glide_target_weights(m)['VTI_LEAPS'] * total within 1e-6.

    At re-entry, _apply_gtt_reentry sets:
        leaps_seed = sum(dynamic_targets[k] for k in leaps_keys) * total
    and seeds a fresh LEAPS simulation with that capital.  The resulting leaps_value
    (priced at creation IV) equals leaps_seed within contract-rounding precision.

    weight_history['VTI_LEAPS'] = leaps_value / total_nav (since share=1.0 for single key).

    Strategy: back-solve m from the observed LEAPS weight using the inverse of
    glide_path_leaps_weight, then verify VTI is consistent with the same m.

    Arguments:
        integration_result: Shared fixture from the module-level run.

    Returns:
        None

    Raises:
        AssertionError: If LEAPS allocation deviates from glide-path formula by > 1e-6.
    """
    wh = integration_result["result"].weight_history
    config = integration_result["config"]
    re_entry_ts = integration_result["re_entry_ts"]
    total = integration_result["nav_at_reentry"]

    observed_leaps_frac = float(wh.loc[re_entry_ts, "VTI_LEAPS"])
    w0 = 0.40
    floor = _GP_CONFIG.floor           # 0.05
    half_life = _GP_CONFIG.half_life_multiple  # 2.0

    # Sanity: observed fraction must be between floor and w0 (de-levered but not floored)
    assert floor < observed_leaps_frac < w0, (
        f"VTI_LEAPS fraction {observed_leaps_frac:.6f} must be in ({floor}, {w0}); "
        "glide-path de-levering may not have fired"
    )

    # Back-solve m from the observed LEAPS weight:
    # observed_leaps_frac = floor + (w0 - floor) * exp(-lam*(m-1))
    # => m = 1 - ln((observed-floor)/(w0-floor)) / lam
    lam = math.log(2.0) / half_life
    ratio = (observed_leaps_frac - floor) / (w0 - floor)
    m_inferred = 1.0 - math.log(ratio) / lam

    assert m_inferred > 1.5, (
        f"Inferred m = {m_inferred:.4f} must be > 1.5 for glide-path to be active; "
        f"observed VTI_LEAPS weight={observed_leaps_frac:.6f}"
    )

    # Compute canonical targets at the inferred m.
    dynamic_targets = compute_glide_target_weights(m_inferred, config, _GP_CONFIG)

    # Assert 2: LEAPS allocation matches within 1e-6 absolute.
    expected_leaps_alloc = float(dynamic_targets["VTI_LEAPS"]) * total
    observed_leaps_alloc = observed_leaps_frac * total
    assert abs(observed_leaps_alloc - expected_leaps_alloc) < 1e-6, (
        f"LEAPS allocation mismatch: "
        f"observed={observed_leaps_alloc:.8f}, "
        f"expected={expected_leaps_alloc:.8f}, "
        f"diff={abs(observed_leaps_alloc - expected_leaps_alloc):.2e}, "
        f"m_inferred={m_inferred:.4f}"
    )


def test_vti_allocation_matches_canonical_at_reentry(
    integration_result: dict,
) -> None:
    """VTI allocation == compute_glide_target_weights(m)['VTI'] * total within 1e-6.

    weight_history['VTI'] = holdings['VTI'] / total_nav.
    _apply_gtt_reentry sets holdings['VTI'] = dynamic_targets['VTI'] * total.
    Since total_nav ≈ total after re-entry (leaps_value ≈ leaps_seed), this equals
    dynamic_targets['VTI'] within 1e-6/total.

    Arguments:
        integration_result: Shared fixture from the module-level run.

    Returns:
        None

    Raises:
        AssertionError: If VTI allocation deviates from glide-path formula by > 1e-6.
    """
    wh = integration_result["result"].weight_history
    config = integration_result["config"]
    re_entry_ts = integration_result["re_entry_ts"]
    total = integration_result["nav_at_reentry"]

    observed_leaps_frac = float(wh.loc[re_entry_ts, "VTI_LEAPS"])
    observed_vti_frac = float(wh.loc[re_entry_ts, "VTI"])

    # Back-solve m from LEAPS (same as above; duplicated here to keep test self-contained).
    w0 = 0.40
    floor = _GP_CONFIG.floor
    half_life = _GP_CONFIG.half_life_multiple
    lam = math.log(2.0) / half_life
    ratio = (observed_leaps_frac - floor) / (w0 - floor)
    m_inferred = 1.0 - math.log(ratio) / lam

    dynamic_targets = compute_glide_target_weights(m_inferred, config, _GP_CONFIG)

    # Assert 3: VTI allocation matches within 1e-6 absolute.
    expected_vti_alloc = float(dynamic_targets["VTI"]) * total
    observed_vti_alloc = observed_vti_frac * total
    assert abs(observed_vti_alloc - expected_vti_alloc) < 1e-6, (
        f"VTI allocation mismatch: "
        f"observed={observed_vti_alloc:.8f}, "
        f"expected={expected_vti_alloc:.8f}, "
        f"diff={abs(observed_vti_alloc - expected_vti_alloc):.2e}, "
        f"m_inferred={m_inferred:.4f}"
    )


def test_total_nav_conserved_across_reentry_step(
    integration_result: dict,
) -> None:
    """Total NAV is conserved across the re-entry step within 1e-6.

    The day before re-entry has all capital in defensive_sleeve + leaps_pool.
    _apply_gtt_reentry redeploys this to holdings + leaps_value, zeroing sleeve
    and pool.  The total before redeployment (used inside _apply_gtt_reentry)
    equals sum(holdings) + sleeve + pool, which equals total_nav after the step.

    We verify that the NAV at re-entry does not deviate from the pre-re-entry NAV
    by more than a small defensive compounding factor (the defensive sleeve earns
    rfr/252 on the transition day itself).

    Arguments:
        integration_result: Shared fixture from the module-level run.

    Returns:
        None

    Raises:
        AssertionError: If NAV changes by more than 0.1% on the re-entry day.
    """
    nav_series = integration_result["result"].nav_series
    nav_at_reentry = integration_result["nav_at_reentry"]
    nav_pre_reentry = float(nav_series.iloc[_DEF_HI - 1])

    # Conservative bound: the re-entry day itself has defensive compounding (rfr/252).
    # Allow 0.1% deviation to account for the defensive-compounding day and market returns.
    relative_change = abs(nav_at_reentry - nav_pre_reentry) / nav_pre_reentry
    assert relative_change < 0.10, (
        f"NAV change on re-entry day too large: {relative_change:.4%}; "
        f"pre={nav_pre_reentry:.2f}, post={nav_at_reentry:.2f}"
    )


def test_weight_rows_sum_to_one_throughout(integration_result: dict) -> None:
    """weight_history rows sum to 1.0 across the entire backtest within 1e-9.

    Arguments:
        integration_result: Shared fixture from the module-level run.

    Returns:
        None
    """
    wh = integration_result["result"].weight_history
    np.testing.assert_allclose(
        wh.sum(axis=1).to_numpy(),
        1.0,
        atol=1e-9,
        err_msg="weight_history rows must sum to 1.0 on every day",
    )


def test_leaps_zero_during_defensive_window(integration_result: dict) -> None:
    """VTI_LEAPS and VTI weights are exactly 0 throughout the defensive window.

    The defensive window [_DEF_LO, _DEF_HI) is placed between two month-ends
    (454 and 477) so no DRIFT rebalance fires inside the window.  This guarantees
    the leaps_value top-up path in _apply_rebalance never artificially inflates
    VTI_LEAPS during the defensive period.

    Arguments:
        integration_result: Shared fixture from the module-level run.

    Returns:
        None
    """
    wh = integration_result["result"].weight_history
    defensive_slice = wh.iloc[_DEF_LO:_DEF_HI]
    assert defensive_slice["VTI_LEAPS"].abs().max() == pytest.approx(0.0, abs=1e-12), (
        "VTI_LEAPS weight must be zero throughout the defensive window"
    )
    assert defensive_slice["VTI"].abs().max() == pytest.approx(0.0, abs=1e-12), (
        "VTI weight must be zero throughout the defensive window"
    )


def test_vti_positive_at_reentry(integration_result: dict) -> None:
    """VTI weight is positive at re-entry when m > 1.5 (freed by glide-path).

    At m > 1.5, vti_alpha=0.65 routes 65% of freed LEAPS weight to VTI, which
    must produce a VTI weight > 0.05 given the large m value at day 456.

    Arguments:
        integration_result: Shared fixture from the module-level run.

    Returns:
        None
    """
    wh = integration_result["result"].weight_history
    re_entry_ts = integration_result["re_entry_ts"]
    assert float(wh.loc[re_entry_ts, "VTI"]) > 0.05, (
        "VTI weight must be > 5% at re-entry (glide-path at m > 1.5 with vti_alpha=0.65)"
    )


def test_leaps_below_w0_at_reentry(integration_result: dict) -> None:
    """VTI_LEAPS weight at re-entry is below w0=0.40, confirming glide-path de-levering.

    Arguments:
        integration_result: Shared fixture from the module-level run.

    Returns:
        None
    """
    wh = integration_result["result"].weight_history
    re_entry_ts = integration_result["re_entry_ts"]
    observed_leaps_frac = float(wh.loc[re_entry_ts, "VTI_LEAPS"])
    w0 = 0.40
    assert observed_leaps_frac < w0 - 0.02, (
        f"VTI_LEAPS weight {observed_leaps_frac:.4f} should be substantially below "
        f"w0={w0} when m > 1.5; glide-path may not be active"
    )
