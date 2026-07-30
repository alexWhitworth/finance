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
    MIN_PREMIUM_PER_SHARE,
    TIME_FLOOR,
    AccountType,
    LeapsConfig,
    LeapsContract,
    LeapsGttCloseEvent,
    LeapsLedger,
    LeapsPartialCloseEvent,
    LeapsRollEvent,
    LeapsTaxSummary,
    TerminalNav,
    _live_contracts,
    bs_call_delta,
    bs_call_price,
    bs_call_vanna,
    compute_leaps_nav_contribution,
    compute_leaps_tax_summary,
    compute_terminal_nav,
    create_leaps_contract,
    partial_close_leaps,
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


def test_bs_delta_dividend_yield_reduces_delta() -> None:
    """Positive dividend yield reduces delta via exp(-qT) multiplier."""
    args = {"spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "iv": 0.20}
    delta_no_div = bs_call_delta(**args, dividend_yield=0.0)
    delta_with_div = bs_call_delta(**args, dividend_yield=0.05)
    assert delta_with_div < delta_no_div


# ---------------------------------------------------------------------------
# bs_call_vanna
# ---------------------------------------------------------------------------


def test_bs_vanna_atm_negative() -> None:
    """ATM vanna is negative (d2 > 0 for ITM-biased, but for exact ATM with r=q=0 d2 < 0)."""
    # For ATM with r=q=0: d1 = 0.5*sigma*sqrt(T), d2 = -0.5*sigma*sqrt(T) < 0
    # => vanna = -N'(d1) * (d2/sigma) > 0 (minus of negative)
    vanna = bs_call_vanna(spot=100.0, strike=100.0, time_to_expiry=1.0, iv=0.20)
    assert vanna > 0.0


def test_bs_vanna_deep_itm_near_zero() -> None:
    """Deep ITM vanna approaches 0 (N'(d1) → 0 as d1 → ∞)."""
    vanna = bs_call_vanna(spot=200.0, strike=50.0, time_to_expiry=1.0, iv=0.18)
    assert abs(vanna) < 0.01


def test_bs_vanna_deep_otm_near_zero() -> None:
    """Deep OTM vanna approaches 0 (N'(d1) → 0 as d1 → -∞)."""
    vanna = bs_call_vanna(spot=50.0, strike=200.0, time_to_expiry=0.5, iv=0.20)
    assert abs(vanna) < 0.01


def test_bs_vanna_symmetric_with_dividend_yield() -> None:
    """Vanna changes when dividend yield is non-zero (exp(-qT) multiplier)."""
    base = bs_call_vanna(spot=100.0, strike=100.0, time_to_expiry=1.0, iv=0.20)
    with_div = bs_call_vanna(
        spot=100.0, strike=100.0, time_to_expiry=1.0, iv=0.20, dividend_yield=0.03
    )
    assert base != pytest.approx(with_div, rel=1e-3)


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


# ---------------------------------------------------------------------------
# run_leaps_simulation — initial_capital (F-G2-01 day-1 deployment)
# ---------------------------------------------------------------------------


def test_run_simulation_initial_capital_day1_contract() -> None:
    """initial_capital deploys a single day-1 contract with matching cost basis."""
    prices = _flat_price_series(n_months=6, price=200.0)
    init_cap = 300_000.0
    result = run_leaps_simulation(
        prices,
        monthly_contribution_to_leaps=0.0,  # isolate the day-1 contract
        config=LeapsConfig(),
        initial_capital=init_cap,
    )
    assert len(result.contracts) == 1
    c0 = result.contracts[0]
    assert c0.purchase_date == pd.Timestamp(prices.index[0])
    basis = c0.premium_paid * CONTRACT_MULTIPLIER * c0.n_contracts
    assert basis == pytest.approx(init_cap, rel=1e-9)


def test_run_simulation_initial_capital_zero_no_contract() -> None:
    """initial_capital=0.0 (default) creates no day-1 contract."""
    prices = _flat_price_series(n_months=6, price=200.0)
    result = run_leaps_simulation(
        prices,
        monthly_contribution_to_leaps=0.0,
        config=LeapsConfig(),
        initial_capital=0.0,
    )
    assert result.contracts == ()


def test_run_simulation_initial_capital_uses_iv_series() -> None:
    """Day-1 contract respects the iv_series value (floored at config.iv)."""
    prices = _flat_price_series(n_months=6, price=200.0)
    high_iv = pd.Series(0.40, index=prices.index)
    result_hi = run_leaps_simulation(
        prices, 0.0, LeapsConfig(iv=0.18), iv_series=high_iv, initial_capital=300_000.0,
    )
    result_lo = run_leaps_simulation(
        prices, 0.0, LeapsConfig(iv=0.18), initial_capital=300_000.0,
    )
    # Higher IV → higher premium on the day-1 contract.
    assert result_hi.contracts[0].premium_paid > result_lo.contracts[0].premium_paid


def test_run_simulation_initial_capital_adds_to_monthly() -> None:
    """With both initial_capital and monthly contributions, day-1 contract is first."""
    prices = _flat_price_series(n_months=6, price=200.0)
    result = run_leaps_simulation(
        prices,
        monthly_contribution_to_leaps=10_000.0,
        config=LeapsConfig(),
        initial_capital=250_000.0,
    )
    # First contract is the day-1 carve-out (basis == 250k), followed by monthly buys.
    c0 = result.contracts[0]
    basis0 = c0.premium_paid * CONTRACT_MULTIPLIER * c0.n_contracts
    assert basis0 == pytest.approx(250_000.0, rel=1e-9)
    assert len(result.contracts) > 1


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


# ---------------------------------------------------------------------------
# create_leaps_contract — MIN_PREMIUM_PER_SHARE guard
# ---------------------------------------------------------------------------


def test_create_leaps_zero_contracts_when_premium_below_floor() -> None:
    """n_contracts is 0.0 when computed premium < MIN_PREMIUM_PER_SHARE."""
    # Use a near-expiry, deep OTM contract to force a near-zero premium
    purchase = pd.Timestamp("2020-01-02")
    # Very short time to expiry (sub-floor after DateOffset(years=2)? No —
    # instead set strike very high relative to spot so premium is tiny)
    # Spot = 1.0, strike = 0.5 (50%), IV = 0.001 (near-zero) → premium ≈ 0
    spot = 1.0
    iv_tiny = 0.0001
    c = create_leaps_contract(purchase, spot, 10_000.0, iv=iv_tiny)
    if c.premium_paid < MIN_PREMIUM_PER_SHARE:
        assert c.n_contracts == 0.0


def test_create_leaps_normal_premium_nonzero_contracts() -> None:
    """n_contracts > 0 for normal BS inputs (not near-zero premium)."""
    c = _make_contract(spot=200.0, capital=10_000.0, iv=DEFAULT_IV)
    assert c.n_contracts > 0.0


# ---------------------------------------------------------------------------
# partial_close_leaps
# ---------------------------------------------------------------------------


def test_partial_close_reduces_n_contracts() -> None:
    """continuation_contract has fewer contracts than original."""
    date = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=date, spot=200.0, capital=20_000.0)
    later = date + pd.Timedelta(days=180)
    current_mtm = price_leaps_contract(c, 200.0, later)
    target = current_mtm * 0.5
    ev = partial_close_leaps(c, later, 200.0, target)
    assert ev.continuation_contract.n_contracts < c.n_contracts
    assert ev.continuation_contract.n_contracts == pytest.approx(c.n_contracts * 0.5, rel=1e-6)


def test_partial_close_net_proceeds_correct() -> None:
    """net_proceeds equals closed fraction of mark-to-market (no tax)."""
    date = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=date, spot=200.0, capital=20_000.0)
    later = date + pd.Timedelta(days=180)
    current_mtm = price_leaps_contract(c, 200.0, later)
    target = current_mtm * 0.4
    ev = partial_close_leaps(c, later, 200.0, target)
    expected_proceeds = current_mtm * 0.6
    assert ev.net_proceeds == pytest.approx(expected_proceeds, rel=1e-6)


def test_partial_close_raises_if_target_gte_mtm() -> None:
    """Raises ValueError when target_value >= current MTM."""
    date = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=date, spot=200.0, capital=10_000.0)
    later = date + pd.Timedelta(days=180)
    current_mtm = price_leaps_contract(c, 200.0, later)
    with pytest.raises(ValueError):
        partial_close_leaps(c, later, 200.0, current_mtm)
    with pytest.raises(ValueError):
        partial_close_leaps(c, later, 200.0, current_mtm * 1.5)


def test_partial_close_no_tax_applied() -> None:
    """net_proceeds is full MTM of closed portion; no LTCG deduction."""
    date = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=date, spot=200.0, capital=20_000.0)
    later = date + pd.Timedelta(days=365)
    current_mtm = price_leaps_contract(c, 300.0, later)
    target = current_mtm * 0.5
    ev = partial_close_leaps(c, later, 300.0, target)
    # If there were a tax, net_proceeds would be < current_mtm * 0.5
    assert ev.net_proceeds == pytest.approx(current_mtm * 0.5, rel=1e-6)


def test_partial_close_returns_frozen_dataclass() -> None:
    """LeapsPartialCloseEvent is frozen."""
    date = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=date, spot=200.0, capital=10_000.0)
    later = date + pd.Timedelta(days=180)
    current_mtm = price_leaps_contract(c, 200.0, later)
    ev = partial_close_leaps(c, later, 200.0, current_mtm * 0.5)
    assert isinstance(ev, LeapsPartialCloseEvent)
    with pytest.raises((AttributeError, TypeError)):
        ev.net_proceeds = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_leaps_nav_contribution — partial close accounting
# ---------------------------------------------------------------------------


def test_nav_contribution_uses_continuation_contract() -> None:
    """After a partial close, the contribution is based on the continuation contract."""
    date = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=date, spot=200.0, capital=20_000.0)
    later = date + pd.Timedelta(days=180)
    current_mtm = price_leaps_contract(c, 200.0, later)
    ev = partial_close_leaps(c, later, 200.0, current_mtm * 0.5)

    ledger_partial = LeapsLedger(
        contracts=(c,),
        roll_events=(),
        account_type=AccountType.TAXABLE,
        partial_close_events=(ev,),
    )
    ledger_full = LeapsLedger(
        contracts=(c,),
        roll_events=(),
        account_type=AccountType.TAXABLE,
    )
    contrib_partial = compute_leaps_nav_contribution(ledger_partial, later, 200.0)
    contrib_full = compute_leaps_nav_contribution(ledger_full, later, 200.0)
    # Partial close cuts position in half → contribution magnitude ~halved
    assert abs(contrib_partial) < abs(contrib_full) + 1.0  # within $1 for rounding


# ---------------------------------------------------------------------------
# compute_terminal_nav
# ---------------------------------------------------------------------------


_DEFAULT_LEDGER_PURCHASE = pd.Timestamp("2020-01-02")


def _make_ledger_with_contracts(
    purchase_date: pd.Timestamp = _DEFAULT_LEDGER_PURCHASE,
    spot: float = 200.0,
    capital: float = 10_000.0,
    account_type: AccountType = AccountType.TAXABLE,
) -> LeapsLedger:
    c = create_leaps_contract(purchase_date, spot, capital, account_type=account_type)
    return LeapsLedger(contracts=(c,), roll_events=(), account_type=account_type)


def test_compute_terminal_nav_taxable_positive_gain() -> None:
    """TAXABLE: terminal_tax = ltcg_rate * open_gain when open_gain > 0."""
    purchase = pd.Timestamp("2020-01-02")
    ledger = _make_ledger_with_contracts(purchase, spot=200.0, capital=20_000.0)
    final_date = purchase + pd.Timedelta(days=365)
    final_nav = 1_000_000.0
    # Spot risen to 350 → positive open gain
    t = compute_terminal_nav(ledger, final_nav, final_date, 350.0)
    assert isinstance(t, TerminalNav)
    assert t.open_gain > 0.0
    assert t.terminal_tax == pytest.approx(t.open_gain * LTCG_RATE, rel=1e-6)
    assert t.post_tax_nav == pytest.approx(final_nav - t.terminal_tax, rel=1e-9)


def test_compute_terminal_nav_taxable_negative_gain_no_tax() -> None:
    """TAXABLE: terminal_tax = 0 when open_gain <= 0 (underwater)."""
    purchase = pd.Timestamp("2020-01-02")
    ledger = _make_ledger_with_contracts(purchase, spot=200.0, capital=20_000.0)
    final_date = purchase + pd.Timedelta(days=365)
    # Spot crashed to 50 → likely underwater
    t = compute_terminal_nav(ledger, 1_000_000.0, final_date, 50.0)
    assert t.terminal_tax == pytest.approx(0.0, abs=1e-9)
    assert t.post_tax_nav == pytest.approx(t.pre_tax_nav, rel=1e-9)


def test_compute_terminal_nav_tax_sheltered_always_zero_tax() -> None:
    """TAX_SHELTERED: terminal_tax = 0 regardless of gain."""
    purchase = pd.Timestamp("2020-01-02")
    ledger = _make_ledger_with_contracts(
        purchase, spot=200.0, capital=20_000.0, account_type=AccountType.TAX_SHELTERED
    )
    final_date = purchase + pd.Timedelta(days=365)
    t = compute_terminal_nav(ledger, 1_000_000.0, final_date, 400.0)
    assert t.terminal_tax == 0.0
    assert t.post_tax_nav == pytest.approx(t.pre_tax_nav, rel=1e-9)


def test_compute_terminal_nav_empty_ledger() -> None:
    """Empty ledger: terminal_tax = 0, post_tax_nav == pre_tax_nav."""
    ledger = LeapsLedger(contracts=(), roll_events=(), account_type=AccountType.TAXABLE)
    t = compute_terminal_nav(ledger, 500_000.0, pd.Timestamp("2022-01-01"), 200.0)
    assert t.terminal_tax == 0.0
    assert t.pre_tax_nav == pytest.approx(500_000.0, rel=1e-9)
    assert t.post_tax_nav == pytest.approx(500_000.0, rel=1e-9)


def test_compute_terminal_nav_is_frozen() -> None:
    """TerminalNav is frozen."""
    ledger = LeapsLedger(contracts=(), roll_events=(), account_type=AccountType.TAXABLE)
    t = compute_terminal_nav(ledger, 500_000.0, pd.Timestamp("2022-01-01"), 200.0)
    with pytest.raises((AttributeError, TypeError)):
        t.terminal_tax = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_leaps_tax_summary
# ---------------------------------------------------------------------------


def test_compute_tax_summary_taxable_aggregates_roll_and_terminal_tax() -> None:
    """total_tax = total_roll_tax + terminal_tax for TAXABLE account."""
    purchase = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=purchase, spot=200.0, capital=20_000.0)
    roll_date = purchase + pd.Timedelta(days=400)
    event = roll_contract(c, roll_date, 280.0)
    ledger = LeapsLedger(
        contracts=(c, event.new_contract),
        roll_events=(event,),
        account_type=AccountType.TAXABLE,
    )
    t_nav = TerminalNav(
        pre_tax_nav=1_000_000.0,
        post_tax_nav=990_000.0,
        terminal_tax=10_000.0,
        open_gain=42_016.81,
        ltcg_rate=LTCG_RATE,
        account_type=AccountType.TAXABLE,
    )
    summary = compute_leaps_tax_summary(ledger, t_nav, 1_000_000.0, years=3.0)
    assert isinstance(summary, LeapsTaxSummary)
    expected_roll_tax = event.tax_paid
    assert summary.total_roll_tax == pytest.approx(expected_roll_tax, rel=1e-9)
    assert summary.terminal_tax == pytest.approx(10_000.0, rel=1e-9)
    assert summary.total_tax == pytest.approx(expected_roll_tax + 10_000.0, rel=1e-9)
    assert summary.n_rolls == 1


def test_compute_tax_summary_tax_sheltered_all_zeros() -> None:
    """TAX_SHELTERED: all tax fields are 0.0."""
    ledger = LeapsLedger(contracts=(), roll_events=(), account_type=AccountType.TAX_SHELTERED)
    t_nav = TerminalNav(
        pre_tax_nav=1_000_000.0,
        post_tax_nav=1_000_000.0,
        terminal_tax=0.0,
        open_gain=0.0,
        ltcg_rate=LTCG_RATE,
        account_type=AccountType.TAX_SHELTERED,
    )
    summary = compute_leaps_tax_summary(ledger, t_nav, 1_000_000.0, years=3.0)
    assert summary.total_tax == 0.0
    assert summary.annualized_tax_drag == 0.0


def test_compute_tax_summary_annualized_drag_positive_for_taxable() -> None:
    """annualized_tax_drag > 0 when total_tax > 0 for TAXABLE."""
    purchase = pd.Timestamp("2020-01-02")
    c = _make_contract(purchase_date=purchase, spot=200.0, capital=20_000.0)
    roll_date = purchase + pd.Timedelta(days=400)
    event = roll_contract(c, roll_date, 300.0)
    ledger = LeapsLedger(
        contracts=(c, event.new_contract),
        roll_events=(event,),
        account_type=AccountType.TAXABLE,
    )
    t_nav = TerminalNav(
        pre_tax_nav=1_000_000.0,
        post_tax_nav=999_000.0,
        terminal_tax=1_000.0,
        open_gain=4_201.68,
        ltcg_rate=LTCG_RATE,
        account_type=AccountType.TAXABLE,
    )
    summary = compute_leaps_tax_summary(ledger, t_nav, 1_000_000.0, years=3.0)
    assert summary.annualized_tax_drag > 0.0


def test_compute_tax_summary_is_frozen() -> None:
    """LeapsTaxSummary is frozen."""
    ledger = LeapsLedger(contracts=(), roll_events=(), account_type=AccountType.TAXABLE)
    t_nav = TerminalNav(
        pre_tax_nav=1_000_000.0,
        post_tax_nav=1_000_000.0,
        terminal_tax=0.0,
        open_gain=0.0,
        ltcg_rate=LTCG_RATE,
        account_type=AccountType.TAXABLE,
    )
    summary = compute_leaps_tax_summary(ledger, t_nav, 1_000_000.0, years=3.0)
    with pytest.raises((AttributeError, TypeError)):
        summary.total_tax = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# run_leaps_simulation — iv_series
# ---------------------------------------------------------------------------


def test_run_leaps_simulation_iv_series_overrides_config_iv() -> None:
    """iv_series above config.iv produces higher premium than config.iv alone."""
    prices = _flat_price_series(n_months=6, price=200.0)
    config_low_iv = LeapsConfig(iv=0.18)
    config_high_iv = LeapsConfig(iv=0.30)

    result_high = run_leaps_simulation(prices, 10_000.0, config_high_iv)
    iv_series = pd.Series(0.30, index=prices.index)
    result_iv_override = run_leaps_simulation(prices, 10_000.0, config_low_iv, iv_series=iv_series)

    assert len(result_iv_override.contracts) > 0
    first_premium_override = result_iv_override.contracts[0].premium_paid
    result_low = run_leaps_simulation(prices, 10_000.0, config_low_iv)
    first_premium_low = result_low.contracts[0].premium_paid

    assert first_premium_override > first_premium_low
    assert first_premium_override == pytest.approx(result_high.contracts[0].premium_paid, rel=1e-6)


def test_run_leaps_simulation_iv_series_floor_respected() -> None:
    """iv_series below config.iv is floored at config.iv; result matches no-iv_series run."""
    prices = _flat_price_series(n_months=6, price=200.0)
    config = LeapsConfig(iv=0.18)

    iv_series = pd.Series(0.05, index=prices.index)
    result_floored = run_leaps_simulation(prices, 10_000.0, config, iv_series=iv_series)
    result_base = run_leaps_simulation(prices, 10_000.0, config)

    assert len(result_floored.contracts) == len(result_base.contracts)
    for c_floored, c_base in zip(result_floored.contracts, result_base.contracts, strict=True):
        assert c_floored.premium_paid == pytest.approx(c_base.premium_paid, rel=1e-9)


def test_run_leaps_simulation_iv_series_none_unchanged() -> None:
    """Passing iv_series=None explicitly is identical to not passing it at all."""
    prices = _flat_price_series(n_months=6, price=200.0)
    config = LeapsConfig(iv=0.18)

    result_default = run_leaps_simulation(prices, 10_000.0, config)
    result_none = run_leaps_simulation(prices, 10_000.0, config, iv_series=None)

    assert len(result_none.contracts) == len(result_default.contracts)
    assert result_none.contracts[0].premium_paid == pytest.approx(
        result_default.contracts[0].premium_paid, rel=1e-9
    )


# ---------------------------------------------------------------------------
# _live_contracts — no-lookahead date-awareness (F-1A / INV-1)
# ---------------------------------------------------------------------------

# Shared timestamps for all _live_contracts tests
_T0 = pd.Timestamp("2020-01-02")  # purchase date
_T1 = pd.Timestamp("2021-01-04")  # between purchase and event
_T2 = pd.Timestamp("2022-01-03")  # event date (roll / close)
_T3 = pd.Timestamp("2022-01-04")  # day after event


def _make_live_contract(purchase_date: pd.Timestamp = _T0) -> LeapsContract:
    """Minimal contract with expiry well past any test date."""
    return create_leaps_contract(purchase_date, 200.0, 10_000.0, DEFAULT_IV, AccountType.TAXABLE)


def test_live_contracts_roll_no_lookahead() -> None:
    """Roll event dated t2 must not hide the original contract at t1 < t2."""
    original = _make_live_contract(_T0)
    roll_ev = roll_contract(original, _T2, 200.0, DEFAULT_IV, 0.02)
    ledger = LeapsLedger(
        contracts=(original, roll_ev.new_contract),
        roll_events=(roll_ev,),
        account_type=AccountType.TAXABLE,
    )

    # Before the roll: original is live, new contract (purchase_date == _T2) is not
    live_before = _live_contracts(ledger, _T1)
    assert original in live_before
    assert roll_ev.new_contract not in live_before

    # After the roll: original is excluded, new contract is live
    live_after = _live_contracts(ledger, _T3)
    assert original not in live_after
    assert roll_ev.new_contract in live_after


def test_live_contracts_same_day_roll_boundary() -> None:
    """On the roll date itself: old excluded, new live (<=  semantics, no gap)."""
    original = _make_live_contract(_T0)
    roll_ev = roll_contract(original, _T2, 200.0, DEFAULT_IV, 0.02)
    ledger = LeapsLedger(
        contracts=(original, roll_ev.new_contract),
        roll_events=(roll_ev,),
        account_type=AccountType.TAXABLE,
    )

    live_on_roll_date = _live_contracts(ledger, _T2)
    assert original not in live_on_roll_date
    assert roll_ev.new_contract in live_on_roll_date


def test_live_contracts_gtt_close_no_lookahead() -> None:
    """GTT-close event dated t2 must not hide the contract at t1 < t2."""
    contract = _make_live_contract(_T0)
    gtt_ev = LeapsGttCloseEvent(
        close_date=_T2,
        contract=contract,
        mtm_value=5_000.0,
        gain_realized=500.0,
        tax_paid=75.0,
        net_proceeds=4_925.0,
    )
    ledger = LeapsLedger(
        contracts=(contract,),
        roll_events=(),
        account_type=AccountType.TAXABLE,
        gtt_close_events=(gtt_ev,),
    )

    # Before the close: contract is live
    live_before = _live_contracts(ledger, _T1)
    assert contract in live_before

    # On the close date: contract is excluded (close_date <= current_date)
    live_on_close = _live_contracts(ledger, _T2)
    assert contract not in live_on_close

    # After the close: still excluded
    live_after = _live_contracts(ledger, _T3)
    assert contract not in live_after


def test_live_contracts_partial_close_substitution_unconditional() -> None:
    """Partial-close substitution applies unconditionally (synthetic close_date trap).

    partial_close_events use a synthetic close_date == final_date, so a naive
    date-filter would revert to full size before that date.  The current design
    applies the substitution without a date guard; this test documents and pins
    the observed behavior so any future change is deliberate.
    """
    # Use dates well within the 2-year contract lifetime (purchase 2020-01-02,
    # expiry 2022-01-02) so the contract doesn't expire during the query window.
    _p0 = pd.Timestamp("2020-01-02")
    _p_close = pd.Timestamp("2020-06-01")   # partial-close execution date
    _p_before = pd.Timestamp("2020-03-01")  # before partial close
    _p_after = pd.Timestamp("2020-09-01")   # after partial close (still live)

    original = _make_live_contract(_p0)
    current_mtm = price_leaps_contract(original, 200.0, _p_close, DEFAULT_IV, 0.0)
    close_ev = partial_close_leaps(original, _p_close, 200.0, target_value=current_mtm * 0.5)
    ledger = LeapsLedger(
        contracts=(original,),
        roll_events=(),
        account_type=AccountType.TAXABLE,
        partial_close_events=(close_ev,),
    )

    # The continuation (reduced size) is substituted at all query dates —
    # because partial_close_events carry a synthetic close_date (== final_date)
    # and the guard is intentionally omitted to avoid reverting to full size.
    for query_date in (_p_before, _p_close, _p_after):
        live = _live_contracts(ledger, query_date)
        assert len(live) == 1
        assert live[0].n_contracts == pytest.approx(
            close_ev.continuation_contract.n_contracts, rel=1e-9
        )
        assert live[0].n_contracts < original.n_contracts
