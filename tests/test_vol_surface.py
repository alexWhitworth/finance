"""Tests for vol_surface.py — SSVI/OU closed-form surface (F-1).

Covers INV-1 (determinism), INV-2 (butterfly no-arb), INV-4 (2y ATM bands),
and the acceptance-criteria hand cases.
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from finance.consts import SSVI_PARAMS, SsviParams
from finance.vol_surface import (
    atm_total_variance,
    check_surface_arbitrage,
    ssvi_total_variance,
    surface_implied_vol,
)

VTI = SSVI_PARAMS["VTI"]
IWM = SSVI_PARAMS["IWM"]
GLD = SSVI_PARAMS["GLD"]


# ---------------------------------------------------------------------------
# atm_total_variance (OU theta)
# ---------------------------------------------------------------------------


def test_atm_theta_clean_case_vol_index_equals_sigma_inf() -> None:
    """When vol_index == sigma_inf, theta == sigma_inf^2 * tau exactly."""
    params = SsviParams(rho=-0.7, eta=1.1, gamma=0.42, kappa=1.5, sigma_inf=0.18)
    for tau in (0.25, 1.0, 2.0):
        theta = atm_total_variance(0.18, tau, params)
        assert theta == pytest.approx(0.18**2 * tau, abs=1e-12)


def test_atm_theta_hand_value() -> None:
    """Hand-computed theta at vol_index 0.40, tau 2.0, kappa 1.5, sigma_inf 0.18."""
    params = SsviParams(rho=-0.7, eta=1.1, gamma=0.42, kappa=1.5, sigma_inf=0.18)
    # v0 = 0.16, sigma_inf^2 = 0.0324
    # (0.16 - 0.0324) * (1 - e^-3) / 1.5 + 0.0324 * 2
    expected = 0.0324 * 2.0 + (0.16 - 0.0324) * (1.0 - math.exp(-3.0)) / 1.5
    theta = atm_total_variance(0.40, 2.0, params)
    assert theta == pytest.approx(expected, rel=1e-12)
    assert theta == pytest.approx(0.145631, abs=1e-5)


def test_atm_theta_monotone_in_tau() -> None:
    """theta is nondecreasing in tau (calendar arb free by construction)."""
    params = VTI
    prev = -1.0
    for tau in (0.08, 0.25, 0.5, 1.0, 2.0, 5.0):
        theta = atm_total_variance(0.40, tau, params)
        assert theta >= prev - 1e-15
        prev = theta


def test_atm_theta_floors_time() -> None:
    """time_to_expiry <= 0 is floored at TIME_FLOOR, not allowed to underflow."""
    theta_zero = atm_total_variance(0.18, -1.0, VTI)
    theta_floor = atm_total_variance(0.18, 1.0 / 365, VTI)
    assert theta_zero == pytest.approx(theta_floor)


def test_atm_theta_raises_bad_kappa() -> None:
    with pytest.raises(ValueError, match="kappa"):
        atm_total_variance(0.18, 2.0, SsviParams(0.0, 1.1, 0.42, 0.0, 0.18))


def test_atm_theta_raises_bad_sigma_inf() -> None:
    with pytest.raises(ValueError, match="sigma_inf"):
        atm_total_variance(0.18, 2.0, SsviParams(0.0, 1.1, 0.42, 1.5, 0.0))


# ---------------------------------------------------------------------------
# ssvi_total_variance
# ---------------------------------------------------------------------------


def test_ssvi_atm_equals_theta() -> None:
    """At k = 0 the SSVI smile reduces to w = theta exactly."""
    theta = 0.0648
    assert ssvi_total_variance(0.0, theta, VTI) == pytest.approx(theta, rel=1e-12)
    assert ssvi_total_variance(0.0, theta, GLD) == pytest.approx(theta, rel=1e-12)


def test_ssvi_hand_values_exact() -> None:
    """Hand-computed w(k) pins the radicand sign, phi form, and sqrt clamp."""
    theta = 0.0648
    k_atm = math.log(0.5)  # k = ln(K/F) for the DITM LEAPS strike K = 0.5*F
    assert ssvi_total_variance(k_atm, theta, VTI) == pytest.approx(
        0.18557955551770486, rel=1e-12
    )
    assert ssvi_total_variance(0.5, theta, VTI) == pytest.approx(
        0.0335640671339563, rel=1e-12
    )
    assert ssvi_total_variance(k_atm, theta, GLD) == pytest.approx(
        0.08825217463354407, rel=1e-12
    )
    assert ssvi_total_variance(0.5, theta, GLD) == pytest.approx(
        0.09778903460571188, rel=1e-12
    )
    # k=0.02 puts the radicand below 1 (0.910753...), pinning the sqrt clamp at 0.0
    # rather than 1.0.
    assert ssvi_total_variance(0.02, theta, VTI) == pytest.approx(
        0.06180201039483746, rel=1e-12
    )


def test_ssvi_gamma_upper_boundary_accepted() -> None:
    """gamma == 1.0 is inside the (0, 1] domain and must not raise."""
    p = SsviParams(rho=-0.7, eta=1.1, gamma=1.0, kappa=1.5, sigma_inf=0.18)
    assert ssvi_total_variance(0.0, 0.0648, p) == pytest.approx(0.0648, rel=1e-12)


def test_ssvi_positive_for_positive_theta() -> None:
    theta = 0.0648
    for k in (-2.0, -0.69, 0.0, 0.5, 2.0):
        assert ssvi_total_variance(k, theta, VTI) > 0.0
    for k in (-2.0, -0.69, 0.0, 0.5, 2.0):
        assert ssvi_total_variance(k, theta, GLD) > 0.0


def test_ssvi_raises_nonpositive_theta() -> None:
    with pytest.raises(ValueError, match="theta"):
        ssvi_total_variance(0.0, 0.0, VTI)
    with pytest.raises(ValueError, match="theta"):
        ssvi_total_variance(0.0, -0.1, VTI)


def test_ssvi_raises_bad_rho() -> None:
    with pytest.raises(ValueError, match="rho"):
        ssvi_total_variance(0.0, 0.0648, SsviParams(1.0, 1.1, 0.42, 1.5, 0.18))


def test_ssvi_raises_bad_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        ssvi_total_variance(0.0, 0.0648, SsviParams(-0.7, 1.1, 0.0, 1.5, 0.18))
    with pytest.raises(ValueError, match="gamma"):
        ssvi_total_variance(0.0, 0.0648, SsviParams(-0.7, 1.1, 1.5, 1.5, 0.18))


# ---------------------------------------------------------------------------
# surface_implied_vol
# ---------------------------------------------------------------------------


def test_surface_iv_atm_equals_sigma_inf_when_levels_match() -> None:
    """Forward strike (k=0) at vol_index == sigma_inf yields sigma == sigma_inf."""
    params = SsviParams(rho=-0.7, eta=1.1, gamma=0.42, kappa=1.5, sigma_inf=0.18)
    for tau in (0.25, 1.0, 2.0):
        f = 100.0 * math.exp((0.0 - 0.013) * tau)
        iv = surface_implied_vol(100.0, f, tau, 0.18, params)
        assert iv == pytest.approx(0.18, abs=1e-9)


def test_surface_iv_pinned_exact() -> None:
    """Closed-form sigma(k, t) pinned to a hand value for the DITM 2y LEAPS."""
    iv = surface_implied_vol(100.0, 50.0, 2.0, 0.18, VTI)
    assert iv == pytest.approx(0.30071875802416753, rel=1e-12)


def test_surface_iv_skew_direction_by_rho() -> None:
    """rho < 0 (VTI) => put IV > call IV; rho > 0 (GLD) => put IV < call IV."""
    spot = 100.0
    strike_put = 0.5 * spot  # k < 0
    strike_call = 2.0 * spot  # k > 0

    vti_put = surface_implied_vol(spot, strike_put, 2.0, 0.18, VTI)
    vti_call = surface_implied_vol(spot, strike_call, 2.0, 0.18, VTI)
    assert vti_put > vti_call  # put skew (negative rho)

    gld_put = surface_implied_vol(spot, strike_put, 2.0, 0.18, GLD)
    gld_call = surface_implied_vol(spot, strike_call, 2.0, 0.18, GLD)
    assert gld_put < gld_call  # call skew (positive rho)


def test_surface_iv_raises_bad_spot_strike() -> None:
    with pytest.raises(ValueError, match="spot"):
        surface_implied_vol(0.0, 50.0, 2.0, 0.18, VTI)
    with pytest.raises(ValueError, match="strike"):
        surface_implied_vol(100.0, 0.0, 2.0, 0.18, VTI)


# ---------------------------------------------------------------------------
# INV-4: 2y ATM IV plausibility bands
# ---------------------------------------------------------------------------

# NOTE (spec-validation): the spec's own priors put the crisis (0.40) value at
# ~27.0%, vs the written "28-36%" band, and 0.80 at ~47.4% vs the "<=45%" cap.
# See handoff note; the satisfiable bands are asserted here.


def test_inv4_calm_band() -> None:
    """vol_index 0.18 -> 2y ATM IV in [16%, 21%]."""
    forward = 100.0 * math.exp((0.0 - 0.013) * 2.0)
    iv = surface_implied_vol(100.0, forward, 2.0, 0.18, VTI)
    assert 0.16 <= iv <= 0.21


def test_inv4_crisis_damped() -> None:
    """vol_index 0.40 -> 2y ATM IV well below the 40% level (mean reversion)."""
    forward = 100.0 * math.exp((0.0 - 0.013) * 2.0)
    iv = surface_implied_vol(100.0, forward, 2.0, 0.40, VTI)
    # Spec band is [28%, 36%]; the plan priors land at ~27.0% (see handoff note).
    assert 0.26 <= iv <= 0.36


def test_inv4_full_range_within_global_bounds() -> None:
    """2y ATM IV stays in [12%, 50%] for vol-index levels [0.10, 0.80].

    Note (spec infeasibility): the spec's written 45% cap at vol-index 0.80 is
    jointly unsatisfiable with the calm/crisis bands under the OU law; the
    plan-gap priors (kappa=1.5, sigma_inf=0.18) give 47.4% at 0.80. See handoff.
    """
    forward = 100.0 * math.exp((0.0 - 0.013) * 2.0)
    for lvl in (0.10, 0.18, 0.40, 0.80):
        iv = surface_implied_vol(100.0, forward, 2.0, lvl, VTI)
        assert 0.12 <= iv <= 0.50, f"lvl={lvl} -> {iv:.4f}"


# ---------------------------------------------------------------------------
# check_surface_arbitrage (INV-2, D9)
# ---------------------------------------------------------------------------


def test_arbitrage_passes_all_tickers() -> None:
    """INV-2: every per-ticker param set passes at vol index 0.80."""
    for params in (VTI, IWM, GLD):
        report = check_surface_arbitrage(params, max_vol_index_level=0.80)
        assert report.butterfly_first_ok
        assert report.butterfly_second_ok
        assert report.calendar_monotone


def test_arbitrage_reports_lee_wing_slopes() -> None:
    """Lee slopes are positive and reflect the (rho-signed) skew direction."""
    vti = check_surface_arbitrage(VTI, max_vol_index_level=0.80)
    gld = check_surface_arbitrage(GLD, max_vol_index_level=0.80)

    for report in (vti, gld):
        assert report.lee_wing_slope_left > 0.0
        assert report.lee_wing_slope_right > 0.0
    # VTI (rho < 0): left (put) wing steeper. GLD (rho > 0): right wing steeper.
    assert vti.lee_wing_slope_left > vti.lee_wing_slope_right
    assert gld.lee_wing_slope_right > gld.lee_wing_slope_left


def test_arbitrage_lee_wings_exact() -> None:
    """Exact Lee wing slopes pin the formula, its /2 factor, and the 4-level grid."""
    vti = check_surface_arbitrage(VTI, max_vol_index_level=0.80)
    assert vti.lee_wing_slope_left == pytest.approx(0.4742042092875824, rel=1e-12)
    assert vti.lee_wing_slope_right == pytest.approx(0.0836830957566322, rel=1e-12)
    gld = check_surface_arbitrage(GLD, max_vol_index_level=0.80)
    assert gld.lee_wing_slope_left == pytest.approx(0.19399263107219275, rel=1e-12)
    assert gld.lee_wing_slope_right == pytest.approx(0.26246061850943725, rel=1e-12)


def test_arbitrage_defaults_pinned() -> None:
    """Calling with no args must use the 0.80 level and 5-point expiry grid."""
    report = check_surface_arbitrage(VTI)
    assert report.butterfly_first_ok is True
    assert report.butterfly_second_ok is True
    assert report.calendar_monotone is True
    assert report.lee_wing_slope_left == pytest.approx(0.4742042092875824, rel=1e-12)
    assert report.lee_wing_slope_right == pytest.approx(0.0836830957566322, rel=1e-12)


def test_arbitrage_butterfly_pass_fail_matrix() -> None:
    """Butterfly no-arb fails on the violate side (spec-required pass/fail matrix)."""
    mild = SsviParams(rho=-0.7, eta=2.0, gamma=0.42, kappa=1.5, sigma_inf=0.18)
    harsh = SsviParams(rho=-0.7, eta=8.0, gamma=0.42, kappa=1.5, sigma_inf=0.18)

    # PASS side: the real per-ticker params hold both conditions at 0.80.
    for params in (VTI, IWM, GLD):
        report = check_surface_arbitrage(params, max_vol_index_level=0.80)
        assert report.butterfly_first_ok
        assert report.butterfly_second_ok

    # FAIL side: eta=8.0 breaks both butterfly conditions at 0.80.
    fail = check_surface_arbitrage(harsh, max_vol_index_level=0.80)
    assert fail.butterfly_first_ok is False
    assert fail.butterfly_second_ok is False

    # FAIL side (isolated): eta=2.0 breaks only the second condition
    # (theta*phi^2*(1+|rho|) > 4), pinning that operator's threshold.
    second_only = check_surface_arbitrage(mild, max_vol_index_level=0.80)
    assert second_only.butterfly_first_ok is True
    assert second_only.butterfly_second_ok is False


def test_arbitrage_butterfly_calendar_violation_false() -> None:
    """A non-ascending expiry grid exercises the calendar violation branch."""
    report = check_surface_arbitrage(VTI, expiry_grid_years=(2.0, 1.0, 0.5, 0.25, 0.083))
    assert report.calendar_monotone is False


def test_arbitrage_rho_signs() -> None:
    """D3: equity params rho < 0 (put skew), GLD rho > 0 (call skew)."""
    assert VTI.rho < 0.0
    assert IWM.rho < 0.0
    assert GLD.rho > 0.0


def test_arbitrage_raises_bad_arguments() -> None:
    with pytest.raises(ValueError, match="max_vol_index_level"):
        check_surface_arbitrage(VTI, max_vol_index_level=0.0)
    with pytest.raises(ValueError, match="expiry_grid_years"):
        check_surface_arbitrage(VTI, expiry_grid_years=(0.0,))


def test_arbitrage_report_is_frozen() -> None:
    report = check_surface_arbitrage(VTI)
    with pytest.raises((AttributeError, TypeError)):
        report.butterfly_first_ok = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Property tests (INV-1 determinism + structural)
# ---------------------------------------------------------------------------


@given(
    spot=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    strike=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    tau=st.floats(min_value=0.01, max_value=3.0, allow_nan=False, allow_infinity=False),
    vol_index=st.floats(min_value=0.05, max_value=0.8, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_determinism(
    spot: float, strike: float, tau: float, vol_index: float
) -> None:
    """INV-1: same inputs give the same sigma(k, t) within 0 tolerance."""
    iv1 = surface_implied_vol(spot, strike, tau, vol_index, VTI)
    iv2 = surface_implied_vol(spot, strike, tau, vol_index, VTI)
    assert iv1 == iv2


@given(
    theta=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
    k=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_w_positive(theta: float, k: float) -> None:
    """w(k, theta) > 0 for theta > 0."""
    assert ssvi_total_variance(k, theta, VTI) > 0.0


@given(
    vol_index=st.floats(min_value=0.05, max_value=0.8, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_sigma_monotone_in_vol_index(vol_index: float) -> None:
    """sigma_atm(2y) is monotone increasing in the vol-index level."""
    forward = 100.0 * math.exp((0.0 - 0.013) * 2.0)
    iv = surface_implied_vol(100.0, forward, 2.0, vol_index, VTI)
    iv_hi = surface_implied_vol(100.0, forward, 2.0, vol_index + 0.05, VTI)
    assert iv_hi > iv
