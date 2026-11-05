"""Tests for finance/greeks.py — F-007/F-008: ContractGreeks, PortfolioGreeks, compute functions."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from finance.consts import CONTRACT_MULTIPLIER, TIME_FLOOR
from finance.greeks import (
    compute_contract_greeks,
    compute_portfolio_greeks,
    leaps_mtm_by_ticker,
)
from finance.leverage import (
    AccountType,
    LeapsContract,
    bs_call_charm,
    bs_call_delta,
    bs_call_gamma,
    bs_call_price,
    bs_call_theta,
    bs_call_vanna,
    bs_call_vega,
)
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
        "price",
        "delta",
        "gamma",
        "vega",
        "theta",
        "vanna",
        "charm",
        "position_delta",
        "position_vega",
        "position_theta",
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
def test_property_gamma_positive(spot: float, iv: float, rfr: float, mono: float) -> None:
    """I5: gamma > 0 for valid-regime inputs."""
    strike = round(spot * mono, 2)
    assume(strike > 1.0)
    contract = _make_contract(strike=strike)
    cg = compute_contract_greeks(contract, spot, iv, _AS_OF, risk_free_rate=rfr)
    assert cg.gamma > 0.0


@given(spot=_valid_spot, iv=_valid_iv, rfr=_valid_rfr, mono=_valid_moneyness)
@settings(max_examples=300)
def test_property_theta_negative(spot: float, iv: float, rfr: float, mono: float) -> None:
    """I6: theta < 0 for valid-regime inputs with T > TIME_FLOOR."""
    strike = round(spot * mono, 2)
    assume(strike > 1.0)
    contract = _make_contract(strike=strike, expiry=pd.Timestamp("2026-01-16"))
    cg = compute_contract_greeks(contract, spot, iv, _AS_OF, risk_free_rate=rfr)
    assert cg.time_to_expiry > TIME_FLOOR
    assert cg.theta < 0.0


@given(spot=_valid_spot, iv=_valid_iv, rfr=_valid_rfr, mono=_valid_moneyness)
@settings(max_examples=300)
def test_property_price_consistency_i7(spot: float, iv: float, rfr: float, mono: float) -> None:
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
    pg = compute_portfolio_greeks(portfolio, spot={"VTI": 200.0}, iv={"VTI": 0.20})

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
    pg = compute_portfolio_greeks(portfolio, spot={"VTI": 200.0}, iv={"VTI": 0.20})
    assert len(pg.contracts) == 0


# ---------------------------------------------------------------------------
# Single contract portfolio
# ---------------------------------------------------------------------------


def test_single_contract_net_delta_equals_position_delta() -> None:
    """Single contract: net_delta == contracts[0].position_delta."""
    contract = _make_contract(strike=100.0, n_contracts=1.0)
    portfolio = _make_live_portfolio(leaps_contracts=((contract, 1.0),))
    pg = compute_portfolio_greeks(
        portfolio, spot={"VTI": 200.0}, iv={"VTI": 0.20}, risk_free_rate=0.04
    )

    assert len(pg.contracts) == 1
    np.testing.assert_allclose(pg.net_delta, pg.contracts[0].position_delta, rtol=1e-12)
    np.testing.assert_allclose(pg.net_vega, pg.contracts[0].position_vega, rtol=1e-12)
    np.testing.assert_allclose(pg.net_theta, pg.contracts[0].position_theta, rtol=1e-12)


def test_single_contract_as_of_date() -> None:
    """as_of_date on PortfolioGreeks matches portfolio.as_of_date."""
    contract = _make_contract(strike=100.0)
    portfolio = _make_live_portfolio(leaps_contracts=((contract, 1.0),))
    pg = compute_portfolio_greeks(portfolio, spot={"VTI": 200.0}, iv={"VTI": 0.20})
    assert pg.as_of_date == _AS_OF


# ---------------------------------------------------------------------------
# Multiple contracts: net_ == sum of per-contract position values (I12 oracle)
# ---------------------------------------------------------------------------


def test_multi_contract_net_delta_is_sum() -> None:
    """net_delta == sum(cg.position_delta) for any number of contracts."""
    c1 = _make_contract(strike=100.0, n_contracts=2.0)
    c2 = _make_contract(strike=150.0, n_contracts=3.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c1, 1.0), (c2, 0.75)))
    pg = compute_portfolio_greeks(portfolio, spot={"VTI": 200.0}, iv={"VTI": 0.20})

    expected_delta = sum(cg.position_delta for cg in pg.contracts)
    np.testing.assert_allclose(pg.net_delta, expected_delta, rtol=1e-12)


def test_multi_contract_net_vega_is_sum() -> None:
    """net_vega == sum(cg.position_vega) for multiple contracts."""
    c1 = _make_contract(strike=100.0, n_contracts=1.0)
    c2 = _make_contract(strike=120.0, n_contracts=2.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c1, 1.0), (c2, 1.0)))
    pg = compute_portfolio_greeks(portfolio, spot={"VTI": 200.0}, iv={"VTI": 0.20})

    expected_vega = sum(cg.position_vega for cg in pg.contracts)
    np.testing.assert_allclose(pg.net_vega, expected_vega, rtol=1e-12)


def test_multi_contract_net_gamma_scaled() -> None:
    """net_gamma == sum(gamma * n_contracts * CONTRACT_MULTIPLIER * leaps_scale)."""
    c1 = _make_contract(strike=100.0, n_contracts=2.0)
    c2 = _make_contract(strike=130.0, n_contracts=1.5)
    portfolio = _make_live_portfolio(leaps_contracts=((c1, 0.8), (c2, 0.5)))
    pg = compute_portfolio_greeks(portfolio, spot={"VTI": 200.0}, iv={"VTI": 0.20})

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
    pg = compute_portfolio_greeks(portfolio, spot={"VTI": 200.0}, iv={"VTI": 0.20})

    expected_theta = sum(cg.position_theta for cg in pg.contracts)
    np.testing.assert_allclose(pg.net_theta, expected_theta, rtol=1e-12)


def test_different_leaps_scales() -> None:
    """Multiple contracts with different leaps_scale values: net fields are correct."""
    c1 = _make_contract(strike=100.0, n_contracts=5.0)
    c2 = _make_contract(strike=100.0, n_contracts=5.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c1, 1.0), (c2, 0.3)))
    pg = compute_portfolio_greeks(portfolio, spot={"VTI": 200.0}, iv={"VTI": 0.20})

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
    pg = compute_portfolio_greeks(portfolio, spot={"VTI": spot}, iv={"VTI": iv})
    expected = sum(cg.position_delta for cg in pg.contracts)
    np.testing.assert_allclose(pg.net_delta, expected, rtol=1e-12)


# ===========================================================================
# F-002: multi-ticker routing and leaps_mtm_by_ticker
# ===========================================================================


def _make_ticker_contract(ticker: str, strike: float, n_contracts: float) -> LeapsContract:
    """Build a minimal LeapsContract tagged with the given ticker."""
    return LeapsContract(
        purchase_date=pd.Timestamp("2024-01-15"),
        expiry_date=_EXPIRY,
        strike=strike,
        spot_at_purchase=200.0,
        premium_paid=45.0,
        notional=200.0 * CONTRACT_MULTIPLIER,
        n_contracts=n_contracts,
        account_type=AccountType.TAXABLE,
        dividend_yield=0.0,
        ticker=ticker,
    )


def test_multi_ticker_routing_uses_own_spot_and_iv() -> None:
    """M3: each contract is priced with spot/iv for its own contract.ticker."""
    c_vti = _make_ticker_contract("VTI", strike=100.0, n_contracts=2.0)
    c_iwm = _make_ticker_contract("IWM", strike=100.0, n_contracts=2.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c_vti, 1.0), (c_iwm, 1.0)))

    spot = {"VTI": 200.0, "IWM": 180.0}
    iv = {"VTI": 0.20, "IWM": 0.35}
    pg = compute_portfolio_greeks(portfolio, spot=spot, iv=iv)

    by_ticker = {cg.contract.ticker: cg for cg in pg.contracts}
    assert by_ticker["VTI"].spot == 200.0
    assert by_ticker["VTI"].iv == 0.20
    assert by_ticker["IWM"].spot == 180.0
    assert by_ticker["IWM"].iv == 0.35


def test_multi_ticker_missing_tickers_listed_all() -> None:
    """M2: ValueError lists every missing ticker, not just the first."""
    c_vti = _make_ticker_contract("VTI", strike=100.0, n_contracts=1.0)
    c_iwm = _make_ticker_contract("IWM", strike=100.0, n_contracts=1.0)
    c_gld = _make_ticker_contract("GLD", strike=100.0, n_contracts=1.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c_vti, 1.0), (c_iwm, 1.0), (c_gld, 1.0)))

    with pytest.raises(ValueError, match=r"IWM.*GLD|GLD.*IWM"):
        compute_portfolio_greeks(portfolio, spot={"VTI": 200.0}, iv={"VTI": 0.20})


def test_multi_ticker_missing_both_spot_and_iv() -> None:
    """A ticker missing only from iv is still reported."""
    c_vti = _make_ticker_contract("VTI", strike=100.0, n_contracts=1.0)
    c_iwm = _make_ticker_contract("IWM", strike=100.0, n_contracts=1.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c_vti, 1.0), (c_iwm, 1.0)))

    with pytest.raises(ValueError, match="IWM"):
        compute_portfolio_greeks(portfolio, spot={"VTI": 200.0, "IWM": 180.0}, iv={"VTI": 0.20})


def test_leaps_mtm_by_ticker_sums_per_ticker() -> None:
    """leaps_mtm_by_ticker's per-ticker sum matches a manual per-ticker sum."""
    c_vti = _make_ticker_contract("VTI", strike=100.0, n_contracts=2.0)
    c_iwm = _make_ticker_contract("IWM", strike=100.0, n_contracts=3.0)
    c_vti2 = _make_ticker_contract("VTI", strike=120.0, n_contracts=1.5)
    portfolio = _make_live_portfolio(leaps_contracts=((c_vti, 1.0), (c_iwm, 0.8), (c_vti2, 1.0)))
    spot = {"VTI": 210.0, "IWM": 180.0}
    iv = {"VTI": 0.20, "IWM": 0.30}
    pg = compute_portfolio_greeks(portfolio, spot=spot, iv=iv)

    mtm = leaps_mtm_by_ticker(pg)

    expected: dict[str, float] = {}
    for cg in pg.contracts:
        value = cg.price * cg.contract.n_contracts * CONTRACT_MULTIPLIER * cg.leaps_scale
        expected[cg.contract.ticker] = expected.get(cg.contract.ticker, 0.0) + value
    assert set(mtm) == {"VTI", "IWM"}
    np.testing.assert_allclose(mtm["VTI"], expected["VTI"], atol=1e-9)
    np.testing.assert_allclose(mtm["IWM"], expected["IWM"], atol=1e-9)


def test_leaps_mtm_by_ticker_empty_contracts() -> None:
    """leaps_mtm_by_ticker returns an empty dict when contracts is empty."""
    portfolio = _make_live_portfolio(leaps_contracts=())
    pg = compute_portfolio_greeks(portfolio, spot={"VTI": 200.0}, iv={"VTI": 0.20})
    assert leaps_mtm_by_ticker(pg) == {}


def test_extra_unused_tickers_harmless() -> None:
    """spot/iv dicts with extra unused tickers are harmless."""
    c_vti = _make_ticker_contract("VTI", strike=100.0, n_contracts=1.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c_vti, 1.0),))
    spot = {"VTI": 200.0, "IWM": 180.0, "GLD": 170.0}
    iv = {"VTI": 0.20, "IWM": 0.30, "GLD": 0.25}
    pg = compute_portfolio_greeks(portfolio, spot=spot, iv=iv)
    assert len(pg.contracts) == 1


# ---------------------------------------------------------------------------
# Property-based: per-contract spot/iv routing always uses the contract's own ticker
# ---------------------------------------------------------------------------

_UNDERLYING_TICKERS = ("VTI", "IWM", "GLD", "SPY", "QQQ")


@st.composite
def _multi_ticker_portfolio(draw):
    """Draw a multi-ticker portfolio and its matching spot/iv maps.

    Returns:
        (portfolio, spot_map, iv_map) with 2-4 distinct tickers, each ticker
        carrying 1-3 tagged contracts and its own random spot/iv values.
    """
    tickers = draw(
        st.lists(st.sampled_from(_UNDERLYING_TICKERS), min_size=2, max_size=4, unique=True)
    )
    spot_map = {t: draw(_valid_spot) for t in tickers}
    iv_map = {t: draw(_valid_iv) for t in tickers}

    pairs = []
    for ticker in tickers:
        for _ in range(draw(st.integers(min_value=1, max_value=3))):
            contract = _make_ticker_contract(
                ticker, strike=draw(_valid_spot), n_contracts=draw(_valid_n)
            )
            pairs.append((contract, draw(_valid_scale)))
    portfolio = _make_live_portfolio(leaps_contracts=tuple(pairs))
    return portfolio, spot_map, iv_map


@given(fixture=_multi_ticker_portfolio())
@settings(max_examples=150)
def test_property_multi_ticker_routing_uses_own_spot_and_iv(fixture) -> None:
    """Property: each ContractGreeks.spot/.iv equals the map value for its own ticker."""
    portfolio, spot_map, iv_map = fixture
    pg = compute_portfolio_greeks(portfolio, spot=spot_map, iv=iv_map)

    for cg in pg.contracts:
        assert cg.spot == spot_map[cg.contract.ticker]
        assert cg.iv == iv_map[cg.contract.ticker]


# ===========================================================================
# F-2: Surface-aware greeks (D5 sticky-moneyness delta, D6 level-vega)
# ===========================================================================

from finance.consts import SsviParams  # noqa: E402
from finance.greeks import VolConvention  # noqa: E402
from finance.vol_surface import atm_total_variance, surface_implied_vol  # noqa: E402

_SSVI_EQUITY = SsviParams(rho=-0.7, eta=1.1, gamma=0.42, kappa=1.5, sigma_inf=0.18)
_SSVI_GLD = SsviParams(rho=0.15, eta=0.9, gamma=0.42, kappa=1.5, sigma_inf=0.18)


def _dsigma_dk_reference(log_moneyness: float, theta: float, params: SsviParams, t: float) -> float:
    """Independent finite-difference d(sigma)/dk along the SSVI smile."""
    from finance.greeks import _dsigma_dk

    h = 1e-6
    w_plus = surface_implied_vol.__globals__["ssvi_total_variance"](
        log_moneyness + h, theta, params
    )
    w_minus = surface_implied_vol.__globals__["ssvi_total_variance"](
        log_moneyness - h, theta, params
    )
    analytic = _dsigma_dk(log_moneyness, theta, params, t)
    # cross-check the finite difference against the analytic value
    fd = (math.sqrt(w_plus / t) - math.sqrt(w_minus / t)) / (2 * h)
    np.testing.assert_allclose(analytic, fd, atol=1e-5)
    return analytic


def test_flat_path_bit_for_bit_regression() -> None:
    """params=None reproduces the pre-change flat implementation exactly."""
    contract = _make_contract(strike=100.0, n_contracts=2.0)
    cg = compute_contract_greeks(
        contract, 200.0, 0.20, _AS_OF, risk_free_rate=0.04, leaps_scale=0.9
    )

    t = max((contract.expiry_date - _AS_OF).days / 365.0, TIME_FLOOR)
    scale = 2.0 * CONTRACT_MULTIPLIER * 0.9
    np.testing.assert_array_equal(
        [
            cg.time_to_expiry,
            cg.iv,
            cg.price,
            cg.delta,
            cg.gamma,
            cg.vega,
            cg.theta,
            cg.vanna,
            cg.charm,
            cg.leaps_scale,
            cg.position_delta,
            cg.position_vega,
            cg.position_theta,
        ],
        [
            t,
            0.20,
            bs_call_price(200.0, 100.0, t, 0.20, 0.04, contract.dividend_yield),
            bs_call_delta(200.0, 100.0, t, 0.20, 0.04, contract.dividend_yield),
            bs_call_gamma(200.0, 100.0, t, 0.20, 0.04, contract.dividend_yield),
            bs_call_vega(200.0, 100.0, t, 0.20, 0.04, contract.dividend_yield),
            bs_call_theta(200.0, 100.0, t, 0.20, 0.04, contract.dividend_yield),
            bs_call_vanna(200.0, 100.0, t, 0.20, 0.04, contract.dividend_yield),
            bs_call_charm(200.0, 100.0, t, 0.20, 0.04, contract.dividend_yield),
            0.9,
            bs_call_delta(200.0, 100.0, t, 0.20, 0.04, contract.dividend_yield) * scale,
            bs_call_vega(200.0, 100.0, t, 0.20, 0.04, contract.dividend_yield) * scale,
            bs_call_theta(200.0, 100.0, t, 0.20, 0.04, contract.dividend_yield) * scale,
        ],
    )
    assert cg.delta_sticky_strike is None
    assert cg.level_vega is None


def test_surface_iv_sourced_from_vol_surface() -> None:
    """Surface path: iv field equals surface_implied_vol composed independently."""
    contract = _make_contract(strike=100.0)
    spot, vol_index, rfr = 200.0, 0.18, 0.03
    cg = compute_contract_greeks(
        contract,
        spot,
        None,
        _AS_OF,
        risk_free_rate=rfr,
        params=_SSVI_EQUITY,
        vol_index_level=vol_index,
    )
    expected_iv = surface_implied_vol(
        spot=spot,
        strike=100.0,
        time_to_expiry_years=cg.time_to_expiry,
        vol_index_level=vol_index,
        params=_SSVI_EQUITY,
        dividend_yield=contract.dividend_yield,
        risk_free_rate=rfr,
    )
    assert cg.iv == expected_iv


def test_surface_delta_sm_formula_k_minus_0_69() -> None:
    """Hand-computed delta_sm at k = -0.69 (D5): delta_sm = delta_BS - (vega/S)*dsigma/dk."""
    contract = _make_contract(strike=100.0)
    spot, vol_index, rfr = 200.0, 0.18, 0.03
    cg = compute_contract_greeks(
        contract,
        spot,
        None,
        _AS_OF,
        risk_free_rate=rfr,
        params=_SSVI_EQUITY,
        vol_index_level=vol_index,
    )
    t = cg.time_to_expiry
    theta = atm_total_variance(vol_index, t, _SSVI_EQUITY)
    forward = spot * math.exp((rfr - contract.dividend_yield) * t)
    k = math.log(100.0 / forward)  # deep ITM, ~ -0.69-0.75
    assert -0.80 < k < -0.60
    dsigma_dk = _dsigma_dk_reference(k, theta, _SSVI_EQUITY, t)
    expected = cg.delta_sticky_strike - (cg.vega / spot) * dsigma_dk
    np.testing.assert_allclose(cg.delta, expected, atol=1e-12)
    assert 0.0 < cg.delta < 1.0


def test_surface_delta_sign_flips_with_rho_gld() -> None:
    """Near ATM the dsigma/dk sign follows rho (D3/D5), flipping the delta_sm correction.

    At k ~ 0 the SSVI smile slope is theta*phi*rho: negative for the equity put
    skew (rho < 0 -> delta_sm > delta_BS) and positive for the GLD call skew
    (rho > 0 -> delta_sm < delta_BS).
    """
    contract = _make_contract(strike=200.0, spot_at_purchase=200.0)
    cg_vti = compute_contract_greeks(
        contract,
        200.0,
        None,
        _AS_OF,
        risk_free_rate=0.0,
        params=_SSVI_EQUITY,
        vol_index_level=0.18,
    )
    cg_gld = compute_contract_greeks(
        contract,
        200.0,
        None,
        _AS_OF,
        risk_free_rate=0.0,
        params=_SSVI_GLD,
        vol_index_level=0.18,
    )
    assert cg_vti.delta > cg_vti.delta_sticky_strike  # put skew: dsigma/dk < 0
    assert cg_gld.delta < cg_gld.delta_sticky_strike  # call skew: dsigma/dk > 0


def test_surface_delta_atm_rho_term_vanishes() -> None:
    """k = 0 (ATM): the rho*phi*k term vanishes; w = theta and dsigma/dk stays finite."""
    contract = _make_contract(strike=200.0, spot_at_purchase=200.0)
    cg = compute_contract_greeks(
        contract,
        200.0,
        None,
        _AS_OF,
        risk_free_rate=0.0,
        params=_SSVI_EQUITY,
        vol_index_level=0.18,
    )
    t = cg.time_to_expiry
    theta = atm_total_variance(0.18, t, _SSVI_EQUITY)
    assert cg.iv == math.sqrt(theta / t)  # w(k=0) = theta
    # At k=0 the smile slope is positive for either skew sign (curvature term)
    _dsigma_dk_reference(0.0, theta, _SSVI_EQUITY, t)


def test_level_vega_damping_kappa_1_5_tau_2y() -> None:
    """D6: level_vega == vega * (1 - e^(-kappa*tau))/kappa within 1e-9 at kappa=1.5, tau=2y."""
    cg = compute_contract_greeks(
        _make_contract(strike=100.0),
        200.0,
        None,
        _AS_OF,
        risk_free_rate=0.03,
        params=_SSVI_EQUITY,
        vol_index_level=0.18,
    )
    expected = cg.vega * (1.0 - math.exp(-1.5 * cg.time_to_expiry)) / 1.5
    np.testing.assert_allclose(cg.level_vega, expected, atol=1e-9)
    # ~0.63 damping vs local vega at tau ~ 2y, kappa = 1.5
    assert 0.5 < cg.level_vega / cg.vega < 0.8
    assert cg.level_vega < cg.vega


def test_surface_path_marks_within_parity_bounds() -> None:
    """INV-3: parity bounds on the surface path, calm and crisis v0, k = -0.69, t = 2y."""
    contract = _make_contract(strike=100.0)
    for vol_index in (0.18, 0.80):
        cg = compute_contract_greeks(
            contract,
            200.0,
            None,
            _AS_OF,
            risk_free_rate=0.03,
            params=_SSVI_EQUITY,
            vol_index_level=vol_index,
        )
        q = contract.dividend_yield
        lower = max(
            0.0,
            200.0 * math.exp(-q * cg.time_to_expiry) - 100.0 * math.exp(-0.03 * cg.time_to_expiry),
        )
        upper = 200.0 * math.exp(-q * cg.time_to_expiry)
        assert lower <= cg.price <= upper


def test_surface_greeks_finite() -> None:
    """All surface-path fields finite, including the new nullable ones."""
    cg = compute_contract_greeks(
        _make_contract(strike=100.0),
        200.0,
        None,
        _AS_OF,
        risk_free_rate=0.03,
        params=_SSVI_EQUITY,
        vol_index_level=0.18,
    )
    for field_name in (
        "price",
        "delta",
        "gamma",
        "vega",
        "theta",
        "vanna",
        "charm",
        "position_delta",
        "position_vega",
        "position_theta",
        "delta_sticky_strike",
        "level_vega",
    ):
        val = getattr(cg, field_name)
        assert val is not None and math.isfinite(val), f"{field_name} not finite: {val}"


def test_sticky_strike_convention_headline() -> None:
    """vol_convention=STICKY_STRIKE: headline delta is the plain BSM delta."""
    contract = _make_contract(strike=100.0)
    cg_sm = compute_contract_greeks(
        contract,
        200.0,
        None,
        _AS_OF,
        risk_free_rate=0.03,
        params=_SSVI_EQUITY,
        vol_index_level=0.18,
    )
    cg_ss = compute_contract_greeks(
        contract,
        200.0,
        None,
        _AS_OF,
        risk_free_rate=0.03,
        params=_SSVI_EQUITY,
        vol_index_level=0.18,
        vol_convention=VolConvention.STICKY_STRIKE,
    )
    assert cg_ss.delta == cg_ss.delta_sticky_strike
    assert cg_ss.delta_sticky_strike == cg_sm.delta_sticky_strike
    assert cg_ss.delta != cg_sm.delta
    # everything except the headline delta is unchanged
    np.testing.assert_allclose(
        cg_ss.position_delta,
        cg_ss.delta * cg_ss.contract.n_contracts * CONTRACT_MULTIPLIER * cg_ss.leaps_scale,
        rtol=1e-12,
    )


def test_partial_surface_inputs_raise() -> None:
    """Edge case: params without vol_index_level (and vice versa) -> ValueError."""
    contract = _make_contract()
    with pytest.raises(ValueError, match="must be given together"):
        compute_contract_greeks(
            contract, 200.0, None, _AS_OF, params=_SSVI_EQUITY, vol_index_level=None
        )
    with pytest.raises(ValueError, match="must be given together"):
        compute_contract_greeks(contract, 200.0, 0.20, _AS_OF, params=None, vol_index_level=0.18)


def test_neither_iv_nor_surface_inputs_raise() -> None:
    """Neither iv nor (params, vol_index_level) -> ValueError."""
    with pytest.raises(ValueError, match="either iv or"):
        compute_contract_greeks(_make_contract(), 200.0, None, _AS_OF)


def test_portfolio_surface_routing_per_ticker() -> None:
    """Per-ticker routing on the surface path: VTI gets equity params, GLD gets gold."""
    c_vti = _make_ticker_contract("VTI", strike=100.0, n_contracts=1.0)
    c_gld = _make_ticker_contract("GLD", strike=100.0, n_contracts=1.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c_vti, 1.0), (c_gld, 1.0)))
    pg = compute_portfolio_greeks(
        portfolio,
        spot={"VTI": 200.0, "GLD": 180.0},
        iv=None,
        risk_free_rate=0.03,
        params={"VTI": _SSVI_EQUITY, "GLD": _SSVI_GLD},
        vol_index={"VTI": 0.18, "GLD": 0.15},
    )
    by_ticker = {cg.contract.ticker: cg for cg in pg.contracts}
    expected_vti_iv = surface_implied_vol(
        spot=200.0,
        strike=100.0,
        time_to_expiry_years=by_ticker["VTI"].time_to_expiry,
        vol_index_level=0.18,
        params=_SSVI_EQUITY,
        dividend_yield=c_vti.dividend_yield,
        risk_free_rate=0.03,
    )
    assert by_ticker["VTI"].iv == expected_vti_iv
    assert by_ticker["VTI"].level_vega is not None
    assert by_ticker["GLD"].delta > by_ticker["GLD"].delta_sticky_strike  # call skew
    expected_delta = sum(cg.position_delta for cg in pg.contracts)
    np.testing.assert_allclose(pg.net_delta, expected_delta, rtol=1e-12)


def test_portfolio_surface_missing_ticker_scalar_fallback() -> None:
    """Ticker absent from surface maps falls back to scalar iv when provided."""
    c_vti = _make_ticker_contract("VTI", strike=100.0, n_contracts=1.0)
    c_iwm = _make_ticker_contract("IWM", strike=100.0, n_contracts=1.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c_vti, 1.0), (c_iwm, 1.0)))
    pg = compute_portfolio_greeks(
        portfolio,
        spot={"VTI": 200.0, "IWM": 180.0},
        iv={"IWM": 0.25},
        risk_free_rate=0.03,
        params={"VTI": _SSVI_EQUITY},
        vol_index={"VTI": 0.18},
    )
    by_ticker = {cg.contract.ticker: cg for cg in pg.contracts}
    assert by_ticker["VTI"].level_vega is not None
    assert by_ticker["IWM"].level_vega is None
    assert by_ticker["IWM"].iv == 0.25


def test_portfolio_surface_missing_ticker_fails_loud() -> None:
    """Ticker absent from both surface maps and iv -> ValueError listing it."""
    c_vti = _make_ticker_contract("VTI", strike=100.0, n_contracts=1.0)
    c_iwm = _make_ticker_contract("IWM", strike=100.0, n_contracts=1.0)
    c_gld = _make_ticker_contract("GLD", strike=100.0, n_contracts=1.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c_vti, 1.0), (c_iwm, 1.0), (c_gld, 1.0)))
    with pytest.raises(ValueError, match=r"IWM.*GLD|GLD.*IWM"):
        compute_portfolio_greeks(
            portfolio,
            spot={"VTI": 200.0, "IWM": 180.0, "GLD": 170.0},
            iv=None,
            params={"VTI": _SSVI_EQUITY},
            vol_index={"VTI": 0.18},
        )


def test_portfolio_partial_surface_map_raises() -> None:
    """params map has a ticker vol_index lacks (or vice versa) -> ValueError."""
    c_vti = _make_ticker_contract("VTI", strike=100.0, n_contracts=1.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c_vti, 1.0),))
    with pytest.raises(ValueError, match="must be given together"):
        compute_portfolio_greeks(
            portfolio,
            spot={"VTI": 200.0},
            iv=None,
            params={"VTI": _SSVI_EQUITY},
            vol_index={"IWM": 0.18},
        )


def test_portfolio_flat_path_bit_for_bit() -> None:
    """Flat portfolio path unchanged: scalar iv dict yields identical results."""
    c_vti = _make_ticker_contract("VTI", strike=100.0, n_contracts=2.0)
    c_iwm = _make_ticker_contract("IWM", strike=150.0, n_contracts=1.0)
    portfolio = _make_live_portfolio(leaps_contracts=((c_vti, 0.8), (c_iwm, 1.0)))
    pg = compute_portfolio_greeks(
        portfolio,
        spot={"VTI": 200.0, "IWM": 180.0},
        iv={"VTI": 0.20, "IWM": 0.30},
        risk_free_rate=0.04,
    )
    for cg in pg.contracts:
        assert cg.delta_sticky_strike is None
        assert cg.level_vega is None
        assert cg.iv in (0.20, 0.30)


# ---------------------------------------------------------------------------
# Property: surface-path marks within parity bounds; delta in (0, 1) both ways
# ---------------------------------------------------------------------------

_valid_vol_index = st.floats(min_value=0.08, max_value=0.80, allow_nan=False, allow_infinity=False)


@given(
    spot=_valid_spot,
    vol_index=_valid_vol_index,
    rfr=_valid_rfr,
    mono=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    params_choice=st.sampled_from([("equity", _SSVI_EQUITY), ("gld", _SSVI_GLD)]),
)
@settings(max_examples=200)
def test_property_surface_parity_and_delta_bounds(
    spot: float, vol_index: float, rfr: float, mono: float, params_choice: tuple
) -> None:
    """INV-3: mark within parity bounds for random valid surface inputs; delta in (0, 1)."""
    strike = round(spot * mono, 2)
    assume(strike > 1.0)
    _, params = params_choice
    contract = _make_contract(strike=strike)
    for convention in (VolConvention.STICKY_MONEYNESS, VolConvention.STICKY_STRIKE):
        cg = compute_contract_greeks(
            contract,
            spot,
            None,
            _AS_OF,
            risk_free_rate=rfr,
            params=params,
            vol_index_level=vol_index,
            vol_convention=convention,
        )
        q = contract.dividend_yield
        lower = max(
            0.0,
            spot * math.exp(-q * cg.time_to_expiry) - strike * math.exp(-rfr * cg.time_to_expiry),
        )
        upper = spot * math.exp(-q * cg.time_to_expiry)
        assert lower <= cg.price <= upper
        assert 0.0 < cg.delta < 1.0
        assert 0.0 < cg.delta_sticky_strike < 1.0
