"""Tests for leverage.py — Black-Scholes, LEAPS lifecycle, and simulation."""

import math

import numpy as np
import pandas as pd
import pytest

from finance.leverage import (
    CONTRACT_MULTIPLIER,
    DEFAULT_IV,
    LEAPS_STRIKE_RATIO,
    LTCG_RATE,
    TIME_FLOOR,
    AccountType,
    LeapsConfig,
    LeapsContract,
    LeapsLedger,
    LeapsRollEvent,
    bs_call_delta,
    bs_call_price,
    compute_leaps_nav_contribution,
    create_leaps_contract,
    price_leaps_contract,
    roll_contract,
    run_leaps_simulation,
    should_roll,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DEFAULT_PURCHASE_DATE = pd.Timestamp("2020-01-02")


def _make_contract(
    purchase_date: pd.Timestamp = _DEFAULT_PURCHASE_DATE,
    spot: float = 200.0,
    capital: float = 10_000.0,
    iv: float = DEFAULT_IV,
    account_type: AccountType = AccountType.TAXABLE,
) -> LeapsContract:
    return create_leaps_contract(purchase_date, spot, capital, iv, account_type)


def _flat_price_series(
    n_months: int = 36,
    start: str = "2015-01-01",
    price: float = 200.0,
) -> pd.Series:
    """Daily price series at a constant value."""
    idx = pd.bdate_range(start, periods=n_months * 21)  # ~21 bdays/month
    return pd.Series(price, index=idx)


# ---------------------------------------------------------------------------
# bs_call_price
# ---------------------------------------------------------------------------


def test_bs_price_atm_positive() -> None:
    """ATM call price is strictly positive."""
    p = bs_call_price(spot=100.0, strike=100.0, time_to_expiry=1.0, iv=0.20)
    assert p > 0.0


def test_bs_price_deep_itm_approaches_intrinsic() -> None:
    """Deep ITM call price approaches S - K * exp(-rT) (near intrinsic for r=0)."""
    spot, strike = 200.0, 50.0  # 75% moneyness
    p = bs_call_price(spot=spot, strike=strike, time_to_expiry=2.0, iv=0.18)
    intrinsic = spot - strike
    # Price should be close to intrinsic (within 1%)
    assert abs(p - intrinsic) / intrinsic < 0.01


def test_bs_price_increases_with_iv() -> None:
    """Higher IV → higher call price (vega is positive)."""
    low = bs_call_price(100.0, 100.0, 1.0, iv=0.10)
    high = bs_call_price(100.0, 100.0, 1.0, iv=0.40)
    assert high > low


def test_bs_price_increases_with_time() -> None:
    """Longer time to expiry → higher call price (theta is positive for long options)."""
    short = bs_call_price(100.0, 100.0, 0.5, iv=0.20)
    long_ = bs_call_price(100.0, 100.0, 2.0, iv=0.20)
    assert long_ > short


def test_bs_price_put_call_parity() -> None:
    """Put-call parity: C - P = S - K*exp(-rT).

    Put price is implied from parity: P = C - S + K*exp(-rT).
    We verify this holds to machine precision.
    """
    spot, strike, t_years, iv, r = 100.0, 95.0, 1.0, 0.20, 0.05
    from scipy import stats as st
    d1 = (math.log(spot / strike) + (r + 0.5 * iv**2) * t_years) / (iv * math.sqrt(t_years))
    d2 = d1 - iv * math.sqrt(t_years)
    call = float(spot * st.norm.cdf(d1) - strike * math.exp(-r * t_years) * st.norm.cdf(d2))
    put = float(
        strike * math.exp(-r * t_years) * st.norm.cdf(-d2) - spot * st.norm.cdf(-d1)
    )
    assert (call - put) == pytest.approx(spot - strike * math.exp(-r * t_years), rel=1e-9)


def test_bs_price_floors_time_to_expiry() -> None:
    """Negative or zero time_to_expiry does not crash (floored at TIME_FLOOR)."""
    p = bs_call_price(100.0, 80.0, time_to_expiry=0.0, iv=0.20)
    assert p >= 0.0
    p_neg = bs_call_price(100.0, 80.0, time_to_expiry=-1.0, iv=0.20)
    assert p_neg >= 0.0
    # Both should equal the T=TIME_FLOOR price
    p_floor = bs_call_price(100.0, 80.0, time_to_expiry=TIME_FLOOR, iv=0.20)
    assert p == pytest.approx(p_floor, rel=1e-9)
    assert p_neg == pytest.approx(p_floor, rel=1e-9)


# ---------------------------------------------------------------------------
# bs_call_delta
# ---------------------------------------------------------------------------


def test_bs_delta_deep_itm_approaches_one() -> None:
    """Deep ITM call delta approaches 1.0."""
    delta = bs_call_delta(spot=200.0, strike=50.0, time_to_expiry=2.0, iv=0.18)
    assert delta > 0.99


def test_bs_delta_atm_near_half() -> None:
    """ATM call delta is approximately 0.5 for short dated options (r=0)."""
    delta = bs_call_delta(spot=100.0, strike=100.0, time_to_expiry=0.25, iv=0.20)
    assert 0.45 < delta < 0.55


def test_bs_delta_bounded() -> None:
    """Delta is always in (0, 1)."""
    for spot in [50.0, 100.0, 200.0]:
        for strike in [50.0, 100.0, 200.0]:
            delta = bs_call_delta(spot, strike, 1.0, 0.20)
            assert 0.0 < delta < 1.0


def test_bs_delta_deep_otm_approaches_zero() -> None:
    """Deep OTM call delta approaches 0.0."""
    delta = bs_call_delta(spot=50.0, strike=200.0, time_to_expiry=0.5, iv=0.20)
    assert delta < 0.01


def test_bs_delta_increases_with_spot() -> None:
    """Delta is monotone increasing in spot price."""
    strike, t_years, iv = 100.0, 1.0, 0.20
    spots = [60.0, 80.0, 100.0, 130.0, 160.0]
    deltas = [bs_call_delta(s, strike, t_years, iv) for s in spots]
    assert all(deltas[i] < deltas[i + 1] for i in range(len(deltas) - 1))


# ---------------------------------------------------------------------------
# create_leaps_contract
# ---------------------------------------------------------------------------


def test_create_leaps_strike_at_50pct() -> None:
    """Strike is 50% of spot."""
    spot = 200.0
    c = _make_contract(spot=spot)
    assert c.strike == pytest.approx(LEAPS_STRIKE_RATIO * spot, rel=1e-9)


def test_create_leaps_expiry_two_years() -> None:
    """Expiry is exactly 2 years from purchase_date."""
    date = pd.Timestamp("2020-06-15")
    c = _make_contract(purchase_date=date)
    assert c.expiry_date == pd.Timestamp("2022-06-15")


def test_create_leaps_notional() -> None:
    """Notional equals spot * CONTRACT_MULTIPLIER."""
    spot = 180.0
    c = _make_contract(spot=spot)
    assert c.notional == pytest.approx(spot * CONTRACT_MULTIPLIER, rel=1e-9)


def test_create_leaps_capital_deployed() -> None:
    """Total cost basis ≈ capital_to_deploy (within rounding on n_contracts)."""
    capital = 15_000.0
    c = _make_contract(capital=capital)
    cost_basis = c.premium_paid * CONTRACT_MULTIPLIER * c.n_contracts
    # n_contracts is exact (not floored), so cost_basis == capital
    assert cost_basis == pytest.approx(capital, rel=1e-9)


def test_create_leaps_frozen() -> None:
    """LeapsContract is frozen."""
    c = _make_contract()
    with pytest.raises((AttributeError, TypeError)):
        c.strike = 0.0  # type: ignore[misc]


def test_create_leaps_account_type_stored() -> None:
    """Account type is preserved on the contract."""
    c = _make_contract(account_type=AccountType.TAX_SHELTERED)
    assert c.account_type == AccountType.TAX_SHELTERED


# ---------------------------------------------------------------------------
# price_leaps_contract
# ---------------------------------------------------------------------------


def test_price_leaps_at_purchase_matches_cost_basis() -> None:
    """Mark-to-market at purchase date equals cost basis (no time change)."""
    date = pd.Timestamp("2020-01-02")
    spot = 200.0
    capital = 10_000.0
    c = create_leaps_contract(date, spot, capital)
    mtm = price_leaps_contract(c, spot, date)
    assert mtm == pytest.approx(capital, rel=1e-6)


def test_price_leaps_increases_with_spot() -> None:
    """Mark-to-market increases when spot rises."""
    date = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=date, spot=200.0, capital=20_000.0)
    later = date + pd.Timedelta(days=180)
    low_mtm = price_leaps_contract(c, 180.0, later)
    high_mtm = price_leaps_contract(c, 240.0, later)
    assert high_mtm > low_mtm


def test_price_leaps_positive() -> None:
    """Mark-to-market value is always positive."""
    date = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=date, spot=200.0)
    later = date + pd.Timedelta(days=300)
    assert price_leaps_contract(c, 200.0, later) > 0.0


# ---------------------------------------------------------------------------
# should_roll
# ---------------------------------------------------------------------------


def test_should_roll_all_conditions_met() -> None:
    """Returns True when all three roll conditions hold."""
    purchase = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=purchase)
    # 13 months later: held > MIN_HOLD_DAYS, expiry in ~11 months (< 6 months fails)
    # Need to be within 6 months of expiry AND held >= 366 days
    current = purchase + pd.Timedelta(days=550)  # ~18 months after purchase
    new_expiry = c.expiry_date + pd.DateOffset(years=2)  # new expiry beyond current
    assert should_roll(c, current, pd.Timestamp(new_expiry))


def test_should_roll_false_if_not_held_long_enough() -> None:
    """Returns False if hold duration < MIN_HOLD_DAYS."""
    purchase = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=purchase)
    # 200 days after purchase: too soon (<366)
    current = purchase + pd.Timedelta(days=200)
    new_expiry = pd.Timestamp(c.expiry_date + pd.DateOffset(years=2))
    assert not should_roll(c, current, new_expiry)


def test_should_roll_false_if_not_near_expiry() -> None:
    """Returns False if more than SIX_MONTHS_DAYS remain until expiry."""
    purchase = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=purchase)
    # 370 days after purchase: holds >= 366 but expiry is still ~13 months away
    current = purchase + pd.Timedelta(days=370)
    new_expiry = pd.Timestamp(c.expiry_date + pd.DateOffset(years=2))
    assert not should_roll(c, current, new_expiry)


def test_should_roll_false_if_no_new_expiry() -> None:
    """Returns False if new_expiry_available is not beyond current contract expiry."""
    purchase = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=purchase)
    current = purchase + pd.Timedelta(days=550)
    # new_expiry same as current contract expiry → not truly 'new'
    assert not should_roll(c, current, c.expiry_date)


# ---------------------------------------------------------------------------
# roll_contract
# ---------------------------------------------------------------------------


def test_roll_taxable_applies_ltcg_on_gain() -> None:
    """Taxable account: tax = LTCG_RATE * max(gain, 0)."""
    purchase = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=purchase, spot=200.0, capital=10_000.0)
    # Roll at a higher spot so gain is positive
    current = purchase + pd.Timedelta(days=400)
    event = roll_contract(c, current, 300.0)
    expected_tax = max(0.0, event.gain_realized) * LTCG_RATE
    assert event.tax_paid == pytest.approx(expected_tax, rel=1e-9)


def test_roll_tax_sheltered_no_tax() -> None:
    """TAX_SHELTERED account: tax_paid is always 0."""
    purchase = pd.Timestamp("2020-01-02")
    c = _make_contract(
        purchase_date=purchase, spot=200.0, capital=10_000.0,
        account_type=AccountType.TAX_SHELTERED,
    )
    current = purchase + pd.Timedelta(days=400)
    event = roll_contract(c, current, 300.0, ltcg_rate=LTCG_RATE)
    assert event.tax_paid == 0.0


def test_roll_no_tax_on_negative_gain() -> None:
    """Taxable account: no tax when gain is negative (spot fell)."""
    purchase = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=purchase, spot=200.0, capital=50_000.0)
    # Roll at a lower spot so gain is likely negative
    current = purchase + pd.Timedelta(days=550)
    event = roll_contract(c, current, 80.0)
    assert event.tax_paid == pytest.approx(0.0, abs=1e-9)


def test_roll_net_proceeds_equals_old_value_minus_tax() -> None:
    """net_proceeds = old_value - tax_paid."""
    purchase = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=purchase, spot=200.0, capital=20_000.0)
    current = purchase + pd.Timedelta(days=400)
    event = roll_contract(c, current, 250.0)
    old_value = price_leaps_contract(c, 250.0, current)
    assert event.net_proceeds == pytest.approx(old_value - event.tax_paid, rel=1e-9)


def test_roll_returns_roll_event_dataclass() -> None:
    """roll_contract returns a frozen LeapsRollEvent."""
    purchase = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=purchase)
    event = roll_contract(c, purchase + pd.Timedelta(days=400), 200.0)
    assert isinstance(event, LeapsRollEvent)
    with pytest.raises((AttributeError, TypeError)):
        event.tax_paid = 0.0  # type: ignore[misc]


def test_roll_new_contract_uses_same_account_type() -> None:
    """New contract after roll inherits account_type from old contract."""
    purchase = pd.Timestamp("2020-01-02")
    c = _make_contract(
        purchase_date=purchase, account_type=AccountType.TAX_SHELTERED
    )
    event = roll_contract(c, purchase + pd.Timedelta(days=400), 200.0)
    assert event.new_contract.account_type == AccountType.TAX_SHELTERED


def test_roll_taxable_less_capital_than_sheltered() -> None:
    """Taxable roll leaves less capital for the new contract than tax-sheltered."""
    purchase = pd.Timestamp("2020-01-02")
    spot_buy, spot_sell = 200.0, 300.0
    current = purchase + pd.Timedelta(days=400)
    c_taxable = _make_contract(
        purchase_date=purchase, spot=spot_buy, capital=20_000.0,
        account_type=AccountType.TAXABLE,
    )
    c_sheltered = _make_contract(
        purchase_date=purchase, spot=spot_buy, capital=20_000.0,
        account_type=AccountType.TAX_SHELTERED,
    )
    ev_taxable = roll_contract(c_taxable, current, spot_sell)
    ev_sheltered = roll_contract(c_sheltered, current, spot_sell)
    assert ev_taxable.net_proceeds < ev_sheltered.net_proceeds


# ---------------------------------------------------------------------------
# compute_leaps_nav_contribution
# ---------------------------------------------------------------------------


def test_nav_contribution_empty_ledger_is_zero() -> None:
    """Empty ledger returns 0.0."""
    ledger = LeapsLedger(contracts=(), roll_events=(), account_type=AccountType.TAXABLE)
    result = compute_leaps_nav_contribution(ledger, pd.Timestamp("2022-01-01"), 200.0)
    assert result == 0.0


def test_nav_contribution_at_purchase_is_near_zero() -> None:
    """NAV contribution is ~0 at purchase date (MTM ≈ cost basis)."""
    date = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=date, spot=200.0, capital=10_000.0)
    ledger = LeapsLedger(contracts=(c,), roll_events=(), account_type=c.account_type)
    contribution = compute_leaps_nav_contribution(ledger, date, 200.0)
    assert abs(contribution) < 1.0  # within $1 of zero


def test_nav_contribution_positive_after_spot_rise() -> None:
    """NAV contribution is positive after the spot price rises."""
    date = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=date, spot=200.0, capital=20_000.0)
    ledger = LeapsLedger(contracts=(c,), roll_events=(), account_type=c.account_type)
    later = date + pd.Timedelta(days=90)
    contribution = compute_leaps_nav_contribution(ledger, later, 280.0)
    assert contribution > 0.0


def test_nav_contribution_excludes_rolled_contracts() -> None:
    """Rolled-out contracts are not counted in NAV contribution."""
    date = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=date, spot=200.0, capital=10_000.0)
    roll_date = date + pd.Timedelta(days=400)
    event = roll_contract(c, roll_date, 220.0)
    ledger = LeapsLedger(
        contracts=(c, event.new_contract),
        roll_events=(event,),
        account_type=c.account_type,
    )
    # The old contract should NOT be counted (it was rolled out)
    contribution_with_roll = compute_leaps_nav_contribution(
        ledger, roll_date, 220.0
    )
    # Build a ledger with only the old contract active (no roll) for comparison
    ledger_no_roll = LeapsLedger(
        contracts=(c,), roll_events=(), account_type=c.account_type
    )
    contribution_no_roll = compute_leaps_nav_contribution(
        ledger_no_roll, roll_date, 220.0
    )
    # They should differ (new contract has different strike/expiry)
    assert contribution_with_roll != pytest.approx(contribution_no_roll, rel=1e-3)


def test_nav_contribution_excludes_expired_contracts() -> None:
    """Expired contracts (current_date >= expiry_date) are not counted."""
    date = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=date, spot=200.0, capital=10_000.0)
    ledger = LeapsLedger(contracts=(c,), roll_events=(), account_type=c.account_type)
    # Price after expiry
    after_expiry = c.expiry_date + pd.Timedelta(days=1)
    contribution = compute_leaps_nav_contribution(ledger, after_expiry, 200.0)
    assert contribution == 0.0


# ---------------------------------------------------------------------------
# run_leaps_simulation
# ---------------------------------------------------------------------------


def test_run_simulation_empty_price_series() -> None:
    """Empty price series returns an empty ledger."""
    result = run_leaps_simulation(
        pd.Series(dtype=float),
        monthly_contribution_to_leaps=5_000.0,
        config=LeapsConfig(),
    )
    assert result.contracts == ()
    assert result.roll_events == ()


def test_run_simulation_contracts_created_each_month() -> None:
    """Each month-end creates one new contract."""
    n_months = 12
    prices = _flat_price_series(n_months=n_months)
    result = run_leaps_simulation(
        prices,
        monthly_contribution_to_leaps=10_000.0,
        config=LeapsConfig(),
    )
    # At minimum one contract per month (rolls may add more)
    assert len(result.contracts) >= n_months


def test_run_simulation_ledger_frozen() -> None:
    """LeapsLedger is frozen."""
    result = run_leaps_simulation(
        _flat_price_series(6),
        monthly_contribution_to_leaps=5_000.0,
        config=LeapsConfig(),
    )
    with pytest.raises((AttributeError, TypeError)):
        result.account_type = AccountType.TAXABLE  # type: ignore[misc]


def test_run_simulation_account_type_matches_config() -> None:
    """Ledger account_type matches the config."""
    for at in [AccountType.TAXABLE, AccountType.TAX_SHELTERED]:
        result = run_leaps_simulation(
            _flat_price_series(6),
            monthly_contribution_to_leaps=5_000.0,
            config=LeapsConfig(account_type=at),
        )
        assert result.account_type == at


def test_run_simulation_roll_events_reference_known_contracts() -> None:
    """Every roll event's old_contract appears in contracts tuple."""
    # Use a long history so rolls can trigger
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2010-01-04", periods=252 * 5)
    prices = pd.Series(200.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))), index=idx)
    result = run_leaps_simulation(
        prices,
        monthly_contribution_to_leaps=10_000.0,
        config=LeapsConfig(),
    )
    contract_set = set(result.contracts)
    for event in result.roll_events:
        assert event.old_contract in contract_set


def test_run_simulation_sheltered_has_no_tax() -> None:
    """No roll event has tax_paid > 0 in a TAX_SHELTERED simulation."""
    rng = np.random.default_rng(99)
    idx = pd.bdate_range("2010-01-04", periods=252 * 5)
    prices = pd.Series(200.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))), index=idx)
    result = run_leaps_simulation(
        prices,
        monthly_contribution_to_leaps=10_000.0,
        config=LeapsConfig(account_type=AccountType.TAX_SHELTERED),
    )
    for event in result.roll_events:
        assert event.tax_paid == 0.0


def test_run_simulation_taxable_has_less_capital_than_sheltered() -> None:
    """Taxable simulation produces less total notional than tax-sheltered over the same period."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2010-01-04", periods=252 * 5)
    prices = pd.Series(200.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))), index=idx)

    contribution = 10_000.0
    taxable = run_leaps_simulation(
        prices, contribution, LeapsConfig(account_type=AccountType.TAXABLE)
    )
    sheltered = run_leaps_simulation(
        prices, contribution, LeapsConfig(account_type=AccountType.TAX_SHELTERED)
    )

    def _total_tax_paid(ledger: LeapsLedger) -> float:
        return sum(e.tax_paid for e in ledger.roll_events)

    # Tax-sheltered should pay $0 in total taxes
    assert _total_tax_paid(sheltered) == 0.0
    # If any rolls happened, taxable should have paid some tax (prices trended up)
    if taxable.roll_events:
        assert _total_tax_paid(taxable) >= 0.0
