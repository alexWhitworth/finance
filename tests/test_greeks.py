"""Tests for finance/greeks.py — F-007/F-008: ContractGreeks, PortfolioGreeks, compute functions."""

import math

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from finance.consts import CONTRACT_MULTIPLIER, TIME_FLOOR
from finance.greeks import compute_contract_greeks, compute_portfolio_greeks
from finance.leverage import AccountType, LeapsContract, bs_call_price
from finance.portfolio_manager import LivePortfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AS_OF = pd.Timestamp("2024-01-15")
_EXPIRY = pd.Timestamp("2026-01-16")  # ~2 years out


def _make_contract(
    *,
    expiry: pd.Timestamp = _EXPIRY,
    strike: float = 100.0,
    spot_at_purchase: float = 200.0,
    premium: float = 45.0,
    n_contracts: float = 2.0,
    dividend_yield: float = 0.0,
) -> LeapsContract:
    """Build a minimal LeapsContract."""
    return LeapsContract(
        purchase_date=pd.Timestamp("2024-01-15"),
        expiry_date=expiry,
        strike=strike,
        spot_at_purchase=spot_at_purchase,
        premium_paid=premium,
        notional=spot_at_purchase * CONTRACT_MULTIPLIER,
        n_contracts=n_contracts,
        account_type=AccountType.TAXABLE,
        dividend_yield=dividend_yield,
    )


# ---------------------------------------------------------------------------
# I7: ContractGreeks.price == bs_call_price with identical inputs
# ---------------------------------------------------------------------------


def test_price_matches_bs_call_price() -> None:
    """I7: price field == bs_call_price with identical inputs within 1e-9."""
    contract = _make_contract(strike=100.0)
    spot = 200.0
    iv = 0.20
    cg = compute_contract_greeks(contract, spot, iv, _AS_OF, risk_free_rate=0.05)

    expected = bs_call_price(spot, 100.0, cg.time_to_expiry, iv, 0.05, contract.dividend_yield)
    np.testing.assert_allclose(cg.price, expected, atol=1e-9)


def test_price_atm_reference() -> None:
    """ATM call: price should be positive for any T > 0 and IV > 0."""
    contract = _make_contract(strike=200.0, spot_at_purchase=200.0)
    cg = compute_contract_greeks(contract, 200.0, 0.20, _AS_OF)
    assert cg.price > 0.0


# ---------------------------------------------------------------------------
# I4: delta ∈ (0, 1) for calls
# ---------------------------------------------------------------------------


def test_delta_in_open_interval() -> None:
    """I4: delta must be in (0, 1) for any valid LEAPS call."""
    contract = _make_contract(strike=100.0)
    cg = compute_contract_greeks(contract, 200.0, 0.20, _AS_OF)
    assert 0.0 < cg.delta < 1.0


def test_delta_deep_itm_near_one() -> None:
    """Moderately deep ITM call: delta should be high (> 0.9) but within (0, 1).

    Note: extreme moneyness (e.g. strike=10, spot=200) saturates float64 CDF to 1.0.
    Use strike=80 (40% below spot=200) which is deep ITM but not numerically degenerate.
    """
    contract = _make_contract(strike=80.0)
    cg = compute_contract_greeks(contract, 200.0, 0.20, _AS_OF)
    assert 0.9 < cg.delta < 1.0


def test_delta_deep_otm_near_zero() -> None:
    """Deep OTM call: delta should be close to 0 but strictly > 0."""
    # strike = 500, spot = 50  => very deep OTM
    contract = _make_contract(strike=500.0, spot_at_purchase=50.0)
    cg = compute_contract_greeks(contract, 50.0, 0.20, _AS_OF)
    assert 0.0 < cg.delta < 0.1


# ---------------------------------------------------------------------------
# I5: gamma > 0 for long calls
# ---------------------------------------------------------------------------


def test_gamma_positive() -> None:
    """I5: gamma must be positive for any long call."""
    contract = _make_contract(strike=100.0)
    cg = compute_contract_greeks(contract, 200.0, 0.20, _AS_OF)
    assert cg.gamma > 0.0


# ---------------------------------------------------------------------------
# I6: theta < 0 for long call with T > TIME_FLOOR
# ---------------------------------------------------------------------------


def test_theta_negative() -> None:
    """I6: theta must be negative for long call with T > TIME_FLOOR."""
    contract = _make_contract()
    cg = compute_contract_greeks(contract, 200.0, 0.20, _AS_OF, risk_free_rate=0.0)
    assert cg.theta < 0.0


def test_theta_negative_with_rate() -> None:
    """Theta remains negative with a positive risk-free rate."""
    contract = _make_contract()
    cg = compute_contract_greeks(contract, 200.0, 0.20, _AS_OF, risk_free_rate=0.05)
    assert cg.theta < 0.0


# ---------------------------------------------------------------------------
# TIME_FLOOR edge case — near-expiry contract must not crash
# ---------------------------------------------------------------------------


def test_time_floor_applied_no_crash() -> None:
    """time_to_expiry just above TIME_FLOOR: no crash, gamma/charm approach extremes."""
    # Expiry is 1 day from as_of_date; raw_t ≈ 1/365 == TIME_FLOOR, floored correctly.
    expiry_near = _AS_OF + pd.Timedelta(days=1)
    contract = _make_contract(expiry=expiry_near, strike=100.0)
    cg = compute_contract_greeks(contract, 200.0, 0.20, _AS_OF)
    assert cg.time_to_expiry >= TIME_FLOOR
    assert math.isfinite(cg.gamma)
    assert math.isfinite(cg.charm)


def test_time_to_expiry_floored_at_time_floor() -> None:
    """time_to_expiry is always >= TIME_FLOOR even if contract expired."""
    expiry_past = _AS_OF - pd.Timedelta(days=5)
    contract = LeapsContract(
        purchase_date=pd.Timestamp("2022-01-15"),
        expiry_date=expiry_past,
        strike=100.0,
        spot_at_purchase=200.0,
        premium_paid=45.0,
        notional=20000.0,
        n_contracts=1.0,
        account_type=AccountType.TAXABLE,
        dividend_yield=0.0,
    )
    cg = compute_contract_greeks(contract, 200.0, 0.20, _AS_OF)
    assert cg.time_to_expiry == TIME_FLOOR


# ---------------------------------------------------------------------------
# leaps_scale: position fields scaled proportionally
# ---------------------------------------------------------------------------


def test_position_scale_proportional() -> None:
    """Position-level fields scale proportionally with leaps_scale."""
    contract = _make_contract(n_contracts=1.0)
    cg_full = compute_contract_greeks(contract, 200.0, 0.20, _AS_OF, leaps_scale=1.0)
    cg_half = compute_contract_greeks(contract, 200.0, 0.20, _AS_OF, leaps_scale=0.5)

    np.testing.assert_allclose(cg_half.position_delta, cg_full.position_delta * 0.5, rtol=1e-9)
    np.testing.assert_allclose(cg_half.position_vega, cg_full.position_vega * 0.5, rtol=1e-9)
    np.testing.assert_allclose(cg_half.position_theta, cg_full.position_theta * 0.5, rtol=1e-9)


def test_position_delta_formula() -> None:
    """position_delta == delta * n_contracts * CONTRACT_MULTIPLIER * leaps_scale."""
    contract = _make_contract(n_contracts=3.0)
    cg = compute_contract_greeks(contract, 200.0, 0.20, _AS_OF, leaps_scale=0.8)
    expected = cg.delta * 3.0 * CONTRACT_MULTIPLIER * 0.8
    np.testing.assert_allclose(cg.position_delta, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# All greeks fields are finite
# ---------------------------------------------------------------------------


def test_all_greeks_finite() -> None:
    """All ContractGreeks fields must be finite for a standard LEAPS contract."""
    contract = _make_contract()
    cg = compute_contract_greeks(contract, 200.0, 0.20, _AS_OF, risk_free_rate=0.04)
    for field_name in (
        "price", "delta", "gamma", "vega", "theta", "vanna", "charm",
        "position_delta", "position_vega", "position_theta",
    ):
        val = getattr(cg, field_name)
        assert math.isfinite(val), f"{field_name} is not finite: {val}"


# ---------------------------------------------------------------------------
# Dividend yield is forwarded from contract
# ---------------------------------------------------------------------------


def test_dividend_yield_forwarded() -> None:
    """contract.dividend_yield is used in BS computation (non-zero yield lowers price)."""
    contract_no_q = _make_contract(dividend_yield=0.0)
    contract_with_q = _make_contract(dividend_yield=0.02)
    spot, iv, rfr = 200.0, 0.20, 0.04
    cg_no_q = compute_contract_greeks(contract_no_q, spot, iv, _AS_OF, risk_free_rate=rfr)
    cg_with_q = compute_contract_greeks(contract_with_q, spot, iv, _AS_OF, risk_free_rate=rfr)
    # Dividend yield reduces call price
    assert cg_with_q.price < cg_no_q.price


# ---------------------------------------------------------------------------
# Property-based tests
#
# Strategy notes:
#   - iv_min=0.10 avoids float64 underflow at extreme moneyness / low-vol combos
#   - moneyness bounded at 0.4-2.5x spot to stay in the "valid LEAPS" regime
#     (strike < 0.4x spot or > 2.5x spot saturates CDF to 0 or 1 in float64)
# ---------------------------------------------------------------------------

_valid_spot = st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False)
_valid_iv = st.floats(min_value=0.10, max_value=1.0, allow_nan=False, allow_infinity=False)
_valid_rfr = st.floats(min_value=0.0, max_value=0.15, allow_nan=False, allow_infinity=False)
_valid_scale = st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
_valid_n = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
# Strike as a moneyness fraction of spot (0.4-2.5x), then multiplied in the test
_valid_moneyness = st.floats(min_value=0.4, max_value=2.5, allow_nan=False, allow_infinity=False)


def _d1(spot: float, strike: float, t: float, iv: float, r: float) -> float:
    return (math.log(spot / strike) + (r + 0.5 * iv**2) * t) / (iv * math.sqrt(t))


@given(spot=_valid_spot, iv=_valid_iv, rfr=_valid_rfr, mono=_valid_moneyness, scale=_valid_scale)
@settings(max_examples=300)
def test_property_delta_in_unit_interval(
    spot: float, iv: float, rfr: float, mono: float, scale: float
) -> None:
    """I4: delta ∈ (0, 1) for valid-regime inputs (strike within 40%-250% of spot).

    Note: Filters inputs where float64 N(d1) saturates to exactly 0 or 1.
    This is a float64 limitation, not a formula error.
    """
    strike = round(spot * mono, 2)
    assume(strike > 1.0)
    # Reject inputs that would saturate CDF in float64
    t = ((_EXPIRY - _AS_OF).days) / 365.0
    d1 = _d1(spot, strike, max(t, TIME_FLOOR), iv, rfr)
    assume(abs(d1) < 8.0)  # N(±8) is indistinguishable from 0/1 in float64
    contract = _make_contract(strike=strike)
    cg = compute_contract_greeks(contract, spot, iv, _AS_OF, risk_free_rate=rfr, leaps_scale=scale)
    assert 0.0 < cg.delta < 1.0


@given(spot=_valid_spot, iv=_valid_iv, rfr=_valid_rfr, mono=_valid_moneyness)
@settings(max_examples=300)
def test_property_gamma_positive(
    spot: float, iv: float, rfr: float, mono: float
) -> None:
    """I5: gamma > 0 for valid-regime inputs."""
    strike = round(spot * mono, 2)
    assume(strike > 1.0)
    contract = _make_contract(strike=strike)
    cg = compute_contract_greeks(contract, spot, iv, _AS_OF, risk_free_rate=rfr)
    assert cg.gamma > 0.0


@given(spot=_valid_spot, iv=_valid_iv, rfr=_valid_rfr, mono=_valid_moneyness)
@settings(max_examples=300)
def test_property_theta_negative(
    spot: float, iv: float, rfr: float, mono: float
) -> None:
    """I6: theta < 0 for valid-regime inputs with T > TIME_FLOOR."""
    strike = round(spot * mono, 2)
    assume(strike > 1.0)
    contract = _make_contract(strike=strike, expiry=pd.Timestamp("2026-01-16"))
    cg = compute_contract_greeks(contract, spot, iv, _AS_OF, risk_free_rate=rfr)
    assert cg.time_to_expiry > TIME_FLOOR
    assert cg.theta < 0.0


@given(spot=_valid_spot, iv=_valid_iv, rfr=_valid_rfr, mono=_valid_moneyness)
@settings(max_examples=300)
def test_property_price_consistency_i7(
    spot: float, iv: float, rfr: float, mono: float
) -> None:
    """I7: price == bs_call_price with identical inputs within 1e-9."""
    strike = round(spot * mono, 2)
    assume(strike > 1.0)
    contract = _make_contract(strike=strike, dividend_yield=0.0)
    cg = compute_contract_greeks(contract, spot, iv, _AS_OF, risk_free_rate=rfr)
    expected = bs_call_price(spot, strike, cg.time_to_expiry, iv, rfr, 0.0)
    np.testing.assert_allclose(cg.price, expected, atol=1e-9)


@given(spot=_valid_spot, iv=_valid_iv, rfr=_valid_rfr, mono=_valid_moneyness, n=_valid_n)
@settings(max_examples=200)
def test_property_position_delta_formula(
    spot: float, iv: float, rfr: float, mono: float, n: float
) -> None:
    """position_delta == delta * n_contracts * CONTRACT_MULTIPLIER * leaps_scale."""
    strike = round(spot * mono, 2)
    assume(strike > 1.0)
    contract = _make_contract(strike=strike, n_contracts=n)
    scale = 0.75
    cg = compute_contract_greeks(contract, spot, iv, _AS_OF, risk_free_rate=rfr, leaps_scale=scale)
    expected = cg.delta * n * CONTRACT_MULTIPLIER * scale
    np.testing.assert_allclose(cg.position_delta, expected, rtol=1e-12)


# ===========================================================================
# F-008: compute_portfolio_greeks unit tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers for LivePortfolio construction
# ---------------------------------------------------------------------------


def _make_live_portfolio(
    leaps_contracts: tuple[tuple[LeapsContract, float], ...] = (),
) -> LivePortfolio:
    """Build a minimal LivePortfolio with controllable leaps_contracts."""
    return LivePortfolio(
        as_of_date=_AS_OF,
        holdings={"VTI": 10_000.0},
        target_weights={"VTI": 1.0},
        leaps_contracts=leaps_contracts,
        gtt_regime=None,
    )


# ---------------------------------------------------------------------------
# Empty portfolio → all-zero PortfolioGreeks
# ---------------------------------------------------------------------------


def test_empty_portfolio_all_zero_net_fields() -> None:
    """Empty leaps_contracts → PortfolioGreeks with all net_ fields = 0.0."""
    portfolio = _make_live_portfolio(leaps_contracts=())
    pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.20)

    assert pg.net_delta == 0.0
    assert pg.net_vega == 0.0
    assert pg.net_gamma == 0.0
    assert pg.net_theta == 0.0
    assert pg.net_vanna == 0.0
    assert pg.net_charm == 0.0
    assert pg.contracts == ()
    assert pg.as_of_date == _AS_OF


def test_empty_portfolio_contracts_tuple_empty() -> None:
    """Empty leaps_contracts → contracts tuple is empty."""
    portfolio = _make_live_portfolio()
    pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.20)
    assert len(pg.contracts) == 0


# ---------------------------------------------------------------------------
# Single contract portfolio
# ---------------------------------------------------------------------------


def test_single_contract_net_delta_equals_position_delta() -> None:
    """Single contract: net_delta == contracts[0].position_delta."""
    contract = _make_contract(strike=100.0, n_contracts=1.0)
    portfolio = _make_live_portfolio(leaps_contracts=((contract, 1.0),))
    pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.20, risk_free_rate=0.04)

    assert len(pg.contracts) == 1
    np.testing.assert_allclose(pg.net_delta, pg.contracts[0].position_delta, rtol=1e-12)
    np.testing.assert_allclose(pg.net_vega, pg.contracts[0].position_vega, rtol=1e-12)
    np.testing.assert_allclose(pg.net_theta, pg.contracts[0].position_theta, rtol=1e-12)


def test_single_contract_as_of_date() -> None:
    """as_of_date on PortfolioGreeks matches portfolio.as_of_date."""
    contract = _make_contract(strike=100.0)
    portfolio = _make_live_portfolio(leaps_contracts=((contract, 1.0),))
    pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.20)
    assert pg.as_of_date == _AS_OF


# ---------------------------------------------------------------------------
# Multiple contracts: net_ == sum of per-contract position values (I12 oracle)
# ---------------------------------------------------------------------------


def test_multi_contract_net_delta_is_sum() -> None:
    """net_delta == sum(cg.position_delta) for any number of contracts."""
    c1 = _make_contract(strike=100.0, n_contracts=2.0)
    c2 = _make_contract(strike=150.0, n_contracts=3.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c1, 1.0), (c2, 0.75)))
    pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.20)

    expected_delta = sum(cg.position_delta for cg in pg.contracts)
    np.testing.assert_allclose(pg.net_delta, expected_delta, rtol=1e-12)


def test_multi_contract_net_vega_is_sum() -> None:
    """net_vega == sum(cg.position_vega) for multiple contracts."""
    c1 = _make_contract(strike=100.0, n_contracts=1.0)
    c2 = _make_contract(strike=120.0, n_contracts=2.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c1, 1.0), (c2, 1.0)))
    pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.20)

    expected_vega = sum(cg.position_vega for cg in pg.contracts)
    np.testing.assert_allclose(pg.net_vega, expected_vega, rtol=1e-12)


def test_multi_contract_net_gamma_scaled() -> None:
    """net_gamma == sum(gamma * n_contracts * CONTRACT_MULTIPLIER * leaps_scale)."""
    c1 = _make_contract(strike=100.0, n_contracts=2.0)
    c2 = _make_contract(strike=130.0, n_contracts=1.5)
    portfolio = _make_live_portfolio(leaps_contracts=((c1, 0.8), (c2, 0.5)))
    pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.20)

    expected_gamma = sum(
        cg.gamma * cg.contract.n_contracts * CONTRACT_MULTIPLIER * cg.leaps_scale
        for cg in pg.contracts
    )
    np.testing.assert_allclose(pg.net_gamma, expected_gamma, rtol=1e-12)


def test_multi_contract_net_theta_is_sum() -> None:
    """net_theta == sum(cg.position_theta)."""
    c1 = _make_contract(strike=100.0, n_contracts=1.0)
    c2 = _make_contract(strike=150.0, n_contracts=1.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c1, 1.0), (c2, 1.0)))
    pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.20)

    expected_theta = sum(cg.position_theta for cg in pg.contracts)
    np.testing.assert_allclose(pg.net_theta, expected_theta, rtol=1e-12)


def test_different_leaps_scales() -> None:
    """Multiple contracts with different leaps_scale values: net fields are correct."""
    c1 = _make_contract(strike=100.0, n_contracts=5.0)
    c2 = _make_contract(strike=100.0, n_contracts=5.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c1, 1.0), (c2, 0.3)))
    pg = compute_portfolio_greeks(portfolio, spot=200.0, iv=0.20)

    assert len(pg.contracts) == 2
    assert pg.contracts[0].leaps_scale == 1.0
    assert pg.contracts[1].leaps_scale == 0.3
    # Position delta for half-scale should be ~0.3x of full-scale (same contract params)
    np.testing.assert_allclose(
        pg.contracts[1].position_delta,
        pg.contracts[0].position_delta * 0.3,
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# Property-based: net_delta == sum(position_deltas) (I12 partial check)
# ---------------------------------------------------------------------------


@given(
    spot=_valid_spot,
    iv=_valid_iv,
    mono1=_valid_moneyness,
    mono2=_valid_moneyness,
    n1=st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False),
    n2=st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False),
    scale1=_valid_scale,
    scale2=_valid_scale,
)
@settings(max_examples=150)
def test_property_net_delta_equals_sum(
    spot: float,
    iv: float,
    mono1: float,
    mono2: float,
    n1: float,
    n2: float,
    scale1: float,
    scale2: float,
) -> None:
    """Property: net_delta == sum(cg.position_delta for cg in contracts)."""
    s1 = round(spot * mono1, 2)
    s2 = round(spot * mono2, 2)
    assume(s1 > 1.0 and s2 > 1.0)
    c1 = _make_contract(strike=s1, n_contracts=n1)
    c2 = _make_contract(strike=s2, n_contracts=n2)
    portfolio = _make_live_portfolio(leaps_contracts=((c1, scale1), (c2, scale2)))
    pg = compute_portfolio_greeks(portfolio, spot=spot, iv=iv)
    expected = sum(cg.position_delta for cg in pg.contracts)
    np.testing.assert_allclose(pg.net_delta, expected, rtol=1e-12)
