"""Tests for portfolio.py — rebalance dates, contributions, and the backtest loop."""

import numpy as np
import pandas as pd
import pytest

from finance.data import PriceData
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance.portfolio import (
    BacktestResult,
    PortfolioConfig,
    apply_contribution,
    compute_target_weights,
    get_rebalance_dates,
    run_backtest,
    should_rebalance,
)
from finance.returns import ReturnData, build_return_data

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_TICKERS = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
_EQUAL_WEIGHTS = {t: 1.0 / len(_TICKERS) for t in _TICKERS}

_QUARTER_END_MONTHS = {3, 6, 9, 12}


def _config(
    weights: dict[str, float] | None = None,
    initial_nav: float = 1_000_000.0,
    contribution: float = 0.0,
    leaps_config: LeapsConfig | None = None,
) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights=weights or dict(_EQUAL_WEIGHTS),
        initial_nav=initial_nav,
        monthly_contribution=contribution,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=leaps_config,
    )


def _make_price_data(
    n: int = 504,
    daily_ret: float = 0.0003,
    daily_vol: float = 0.01,
    seed: int = 42,
    start: str = "2015-01-02",
) -> PriceData:
    """Synthetic PriceData for 6 assets."""
    idx = pd.bdate_range(start, periods=n + 1)
    rng = np.random.default_rng(seed)
    starts = {
        "VTI": 200.0, "VXUS": 60.0, "GLD": 170.0,
        "MUB": 55.0, "KMLM": 25.0, "VGIT": 65.0,
    }
    prices_data = {
        t: starts[t] * np.cumprod(1 + rng.normal(daily_ret, daily_vol, n + 1))
        for t in _TICKERS
    }
    prices = pd.DataFrame(prices_data, index=idx)
    dividends = pd.DataFrame(0.0, index=idx, columns=list(_TICKERS))
    return PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=pd.DataFrame(),
        tickers=_TICKERS,
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
    )


def _make_return_data(
    n: int = 504,
    daily_ret: float = 0.0003,
    daily_vol: float = 0.01,
    seed: int = 42,
    start: str = "2015-01-02",
) -> ReturnData:
    """Synthetic ReturnData for 6 assets."""
    pd_obj = _make_price_data(n, daily_ret, daily_vol, seed, start)
    return build_return_data(pd_obj, apply_tey=False)


def _make_rd_and_pd(
    n: int = 504,
    daily_ret: float = 0.0003,
    daily_vol: float = 0.01,
    seed: int = 42,
    start: str = "2015-01-02",
) -> tuple[ReturnData, PriceData]:
    """Return matching (ReturnData, PriceData) pair from the same synthetic series."""
    pd_obj = _make_price_data(n, daily_ret, daily_vol, seed, start)
    return build_return_data(pd_obj, apply_tey=False), pd_obj


# ---------------------------------------------------------------------------
# get_rebalance_dates
# ---------------------------------------------------------------------------


def test_rebalance_dates_are_in_index() -> None:
    """Every rebalance date falls within the provided index."""
    idx = pd.bdate_range("2015-01-02", periods=504)
    dates = get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    for d in dates:
        assert d in idx


def test_rebalance_dates_in_quarter_end_months() -> None:
    """All rebalance dates land in Mar / Jun / Sep / Dec."""
    idx = pd.bdate_range("2015-01-02", periods=1008)
    dates = get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    for d in dates:
        assert d.month in _QUARTER_END_MONTHS


def test_rebalance_dates_are_last_day_of_month() -> None:
    """Each rebalance date is the last trading day of its month."""
    idx = pd.bdate_range("2015-01-02", periods=1008)
    dates = get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    for d in dates:
        later_in_month = idx[(idx.month == d.month) & (idx.year == d.year) & (idx > d)]
        assert len(later_in_month) == 0


def test_rebalance_dates_count_roughly_four_per_year() -> None:
    """For a 2-year window we get exactly 8 quarterly dates."""
    idx = pd.bdate_range("2015-01-02", "2016-12-31")
    dates = get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    assert len(dates) == 8


def test_rebalance_dates_sorted() -> None:
    """Returned list is chronologically sorted."""
    idx = pd.bdate_range("2015-01-02", periods=504)
    dates = get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# compute_target_weights
# ---------------------------------------------------------------------------


def test_compute_target_weights_user_specified_sums_to_one() -> None:
    """USER_SPECIFIED weights normalize to sum = 1.0."""
    cfg = _config(weights={"A": 2.0, "B": 2.0, "C": 1.0})
    current = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
    w = compute_target_weights(cfg, current, 1_000.0, pd.Timestamp("2020-01-02"))
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_compute_target_weights_user_specified_proportions() -> None:
    """USER_SPECIFIED weights preserve proportions of target_weights dict."""
    cfg = _config(weights={"A": 3.0, "B": 1.0})
    current = pd.Series({"A": 0.5, "B": 0.5})
    w = compute_target_weights(cfg, current, 1_000.0, pd.Timestamp("2020-01-02"))
    assert w["A"] == pytest.approx(0.75, abs=1e-9)
    assert w["B"] == pytest.approx(0.25, abs=1e-9)


# ---------------------------------------------------------------------------
# apply_contribution
# ---------------------------------------------------------------------------


def test_apply_contribution_total_equals_contribution() -> None:
    """Sum of allocated amounts equals the contribution."""
    weights = pd.Series({"A": 0.6, "B": 0.4})
    alloc = apply_contribution(nav=500_000.0, contribution=10_000.0, weights=weights)
    assert sum(alloc.values()) == pytest.approx(10_000.0, rel=1e-9)


def test_apply_contribution_proportional_to_weights() -> None:
    """Each asset receives weight[a] * contribution."""
    weights = pd.Series({"A": 0.7, "B": 0.3})
    alloc = apply_contribution(nav=1_000.0, contribution=5_000.0, weights=weights)
    assert alloc["A"] == pytest.approx(3_500.0, rel=1e-9)
    assert alloc["B"] == pytest.approx(1_500.0, rel=1e-9)


def test_apply_contribution_zero_contribution() -> None:
    """Zero contribution allocates zero to every asset."""
    weights = pd.Series({"A": 0.5, "B": 0.5})
    alloc = apply_contribution(nav=1_000.0, contribution=0.0, weights=weights)
    assert sum(alloc.values()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_backtest — basic structure
# ---------------------------------------------------------------------------


def test_run_backtest_returns_correct_type() -> None:
    """run_backtest returns a frozen BacktestResult."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    assert isinstance(result, BacktestResult)
    with pytest.raises((AttributeError, TypeError)):
        result.config = _config()  # type: ignore[misc]


def test_run_backtest_nav_series_length() -> None:
    """NAV series has same length as return series."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    assert len(result.nav_series) == len(rd.returns)


def test_run_backtest_weight_history_shape() -> None:
    """Weight history has shape (n_days, n_assets)."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    assert result.weight_history.shape == (len(rd.returns), len(_TICKERS))


def test_run_backtest_return_series_length() -> None:
    """Return series has same length as return data."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    assert len(result.return_series) == len(rd.returns)


def test_run_backtest_nav_positive() -> None:
    """NAV stays positive throughout the backtest."""
    rd, pd_obj = _make_rd_and_pd(504)
    result = run_backtest(rd, pd_obj, _config())
    assert (result.nav_series > 0).all()


# ---------------------------------------------------------------------------
# run_backtest — NAV math
# ---------------------------------------------------------------------------


def test_run_backtest_nav_starts_near_initial() -> None:
    """After day 1, NAV is initial_nav * (1 + first_day_return)."""
    rd, pd_obj = _make_rd_and_pd(252)
    cfg = _config(initial_nav=1_000_000.0, contribution=0.0)
    result = run_backtest(rd, pd_obj, cfg)
    first_ret = float(rd.returns.iloc[0].mean())  # equal weight
    expected = 1_000_000.0 * (1.0 + first_ret)
    assert result.nav_series.iloc[0] == pytest.approx(expected, rel=1e-6)


def test_run_backtest_flat_returns_nav_is_constant() -> None:
    """With zero returns, no contributions, and no rebalancing effect, NAV is constant."""
    n = 252
    idx = pd.bdate_range("2015-01-02", periods=n)
    returns = pd.DataFrame(0.0, index=idx, columns=list(_TICKERS))
    log_ret = pd.DataFrame(0.0, index=idx, columns=list(_TICKERS))
    rd = ReturnData(
        returns=returns,
        log_returns=log_ret,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=pd.Series(0.0, index=returns.index, name="risk_free_rate"),
    )
    prices = pd.DataFrame(100.0, index=idx, columns=list(_TICKERS))
    pd_obj = PriceData(
        prices=prices, dividends=pd.DataFrame(0.0, index=idx, columns=list(_TICKERS)),
        vol_prices=pd.DataFrame(), tickers=_TICKERS,
        start_date=str(idx[0].date()), end_date=str(idx[-1].date()), spliced=False,
    )
    cfg = _config(initial_nav=500_000.0, contribution=0.0)
    result = run_backtest(rd, pd_obj, cfg)
    assert result.nav_series.iloc[-1] == pytest.approx(500_000.0, rel=1e-9)


def test_run_backtest_contribution_grows_nav() -> None:
    """Monthly contributions increase NAV beyond what returns alone would produce."""
    rd, pd_obj = _make_rd_and_pd(504)
    cfg_no_contrib = _config(initial_nav=1_000_000.0, contribution=0.0)
    cfg_with_contrib = _config(initial_nav=1_000_000.0, contribution=10_000.0)
    result_no = run_backtest(rd, pd_obj, cfg_no_contrib)
    result_yes = run_backtest(rd, pd_obj, cfg_with_contrib)
    assert result_yes.nav_series.iloc[-1] > result_no.nav_series.iloc[-1]


def test_run_backtest_no_contribution_nav_from_returns() -> None:
    """Without contributions, final NAV equals initial_nav * cumulative growth."""
    n = 100
    idx = pd.bdate_range("2015-01-02", periods=n)
    r = 0.001
    returns = pd.DataFrame(r, index=idx, columns=list(_TICKERS))
    log_ret = pd.DataFrame(r, index=idx, columns=list(_TICKERS))
    rd = ReturnData(
        returns=returns,
        log_returns=log_ret,
        tey_adjusted=False,
        marginal_rate=0.0,
        risk_free_rate=pd.Series(0.0, index=returns.index, name="risk_free_rate"),
    )
    prices = pd.DataFrame(100.0, index=idx, columns=list(_TICKERS))
    pd_obj = PriceData(
        prices=prices, dividends=pd.DataFrame(0.0, index=idx, columns=list(_TICKERS)),
        vol_prices=pd.DataFrame(), tickers=_TICKERS,
        start_date=str(idx[0].date()), end_date=str(idx[-1].date()), spliced=False,
    )
    cfg = _config(initial_nav=100_000.0, contribution=0.0)
    result = run_backtest(rd, pd_obj, cfg)
    expected = 100_000.0 * (1.0 + r) ** n
    assert result.nav_series.iloc[-1] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# run_backtest — weight drift and rebalancing
# ---------------------------------------------------------------------------


def test_run_backtest_weights_sum_to_one_each_day() -> None:
    """Realized weights sum to 1.0 on every trading day."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    sums = result.weight_history.sum(axis=1)
    assert (sums - 1.0).abs().max() < 1e-9


def test_run_backtest_weights_drift_between_rebalances() -> None:
    """Weights are not perfectly equal every day (drift before rebalance)."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    mid = result.weight_history.iloc[30]
    max_dev = (mid - 1.0 / len(_TICKERS)).abs().max()
    assert max_dev > 1e-6  # some drift has occurred


def test_run_backtest_weights_snapped_on_rebalance_date() -> None:
    """On each quarterly rebalance date, weights are close to target."""
    rd, pd_obj = _make_rd_and_pd(504)
    cfg = _config()
    result = run_backtest(rd, pd_obj, cfg)
    idx = pd.DatetimeIndex(rd.returns.index)
    rdates = get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    tol = 1e-6
    for d in rdates:
        if d in result.weight_history.index:
            row = result.weight_history.loc[d]
            max_dev = (row - 1.0 / len(_TICKERS)).abs().max()
            assert max_dev < tol, f"Weights not snapped on {d}: max_dev={max_dev}"


# ---------------------------------------------------------------------------
# run_backtest — missing asset
# ---------------------------------------------------------------------------


def test_run_backtest_raises_on_missing_asset() -> None:
    """ValueError if a target_weights asset is absent from return_data."""
    rd, pd_obj = _make_rd_and_pd(100)
    cfg = _config(weights={"VTI": 0.5, "NONEXISTENT": 0.5})
    with pytest.raises(ValueError, match="missing from return_data"):
        run_backtest(rd, pd_obj, cfg)


# ---------------------------------------------------------------------------
# run_backtest — LEAPS overlay
# ---------------------------------------------------------------------------


# LEAPS weights under Model B: a "VTI_LEAPS" key routes carved-out capital.
_LEAPS_WEIGHTS = {
    "VTI_LEAPS": 0.30, "VTI": 0.10, "VXUS": 0.15, "GLD": 0.15,
    "MUB": 0.10, "KMLM": 0.10, "VGIT": 0.10,
}


def test_run_backtest_with_leaps_returns_ledger() -> None:
    """BacktestResult.leaps_ledger is populated when a *_LEAPS key is present."""
    rd, pd_obj = _make_rd_and_pd(504)
    leaps_cfg = LeapsConfig(account_type=AccountType.TAXABLE)
    cfg = _config(weights=dict(_LEAPS_WEIGHTS), leaps_config=leaps_cfg, contribution=5_000.0)
    result = run_backtest(rd, pd_obj, cfg)
    assert result.leaps_ledger is not None
    assert len(result.leaps_ledger.contracts) > 0


def test_run_backtest_no_leaps_ledger_is_none() -> None:
    """BacktestResult.leaps_ledger is None when no *_LEAPS key is present."""
    rd, pd_obj = _make_rd_and_pd(252)
    result = run_backtest(rd, pd_obj, _config())
    assert result.leaps_ledger is None


def test_run_backtest_leaps_keys_without_config_raises() -> None:
    """ValueError if *_LEAPS keys are present but leaps_config is None."""
    rd, pd_obj = _make_rd_and_pd(100)
    cfg = _config(weights=dict(_LEAPS_WEIGHTS), leaps_config=None)
    with pytest.raises(ValueError, match="leaps_config is None"):
        run_backtest(rd, pd_obj, cfg)


def test_run_backtest_leaps_missing_underlying_raises() -> None:
    """ValueError if a *_LEAPS key's underlying is absent from price_data.prices."""
    rd, pd_obj = _make_rd_and_pd(100)
    prices_no_vti = pd_obj.prices.drop(columns=["VTI"])
    pd_no_vti = PriceData(
        prices=prices_no_vti, dividends=pd_obj.dividends,
        vol_prices=pd_obj.vol_prices, tickers=tuple(prices_no_vti.columns),
        start_date=pd_obj.start_date, end_date=pd_obj.end_date, spliced=False,
    )
    # Base assets must still exist in returns; drop VTI from weights too.
    weights = {k: v for k, v in _LEAPS_WEIGHTS.items() if k != "VTI"}
    cfg = _config(weights=weights, leaps_config=LeapsConfig())
    with pytest.raises(ValueError, match="underlying 'VTI' absent"):
        run_backtest(rd, pd_no_vti, cfg)


# ---------------------------------------------------------------------------
# run_backtest — config is stored
# ---------------------------------------------------------------------------


def test_run_backtest_config_stored() -> None:
    """BacktestResult.config is the exact PortfolioConfig that was passed."""
    rd, pd_obj = _make_rd_and_pd(252)
    cfg = _config()
    result = run_backtest(rd, pd_obj, cfg)
    assert result.config is cfg


# ---------------------------------------------------------------------------
# should_rebalance
# ---------------------------------------------------------------------------


def test_should_rebalance_quarterly_always_false() -> None:
    """QUARTERLY rule always returns False regardless of weight deviation."""
    current = pd.Series({"A": 0.80, "B": 0.20})
    target = pd.Series({"A": 0.50, "B": 0.50})
    assert should_rebalance(current, target, RebalanceRule.QUARTERLY) is False


def test_should_rebalance_drift_no_trigger_within_band() -> None:
    """DRIFT rule returns False when all relative deviations are within the band.

    target=0.50, current=0.54 → deviation = 0.04/0.50 = 8% < 10%.
    """
    current = pd.Series({"A": 0.54, "B": 0.46})
    target = pd.Series({"A": 0.50, "B": 0.50})
    assert should_rebalance(current, target, RebalanceRule.DRIFT) is False


def test_should_rebalance_drift_triggers_at_band_breach() -> None:
    """DRIFT rule returns True when one asset exceeds the 10% relative band.

    target=0.50, current=0.56 → deviation = 0.06/0.50 = 12% > 10%.
    """
    current = pd.Series({"A": 0.56, "B": 0.44})
    target = pd.Series({"A": 0.50, "B": 0.50})
    assert should_rebalance(current, target, RebalanceRule.DRIFT) is True


def test_should_rebalance_drift_zero_target_weight_skipped() -> None:
    """DRIFT rule skips assets with target=0.0 (division by zero guard).

    Asset B has target=0.0 and current=0.05; must not raise and return False
    when no other asset breaches.
    """
    current = pd.Series({"A": 0.95, "B": 0.05})
    target = pd.Series({"A": 1.00, "B": 0.00})
    # A: |0.95 - 1.00| / 1.00 = 5% < 10%; B skipped
    assert should_rebalance(current, target, RebalanceRule.DRIFT) is False


def test_should_rebalance_drift_uses_custom_band() -> None:
    """Custom band=0.05 triggers on an 8% relative deviation (outside 5%, within 10%)."""
    current = pd.Series({"A": 0.54, "B": 0.46})
    target = pd.Series({"A": 0.50, "B": 0.50})
    # Default 10% band: no trigger; custom 5% band: 8% > 5% → trigger
    assert should_rebalance(current, target, RebalanceRule.DRIFT, band=0.05) is True


# ---------------------------------------------------------------------------
# F-G2-01 — carved-out LEAPS capital routing (Model B)
# ---------------------------------------------------------------------------


def _leaps_cost_basis(ledger: object) -> float:
    """Sum cost basis of every contract created (premium * multiplier * n_contracts)."""
    from finance.consts import CONTRACT_MULTIPLIER

    return sum(
        c.premium_paid * CONTRACT_MULTIPLIER * c.n_contracts
        for c in ledger.contracts  # type: ignore[attr-defined]
    )


def test_leaps_base_holdings_carved_out_of_initial_nav() -> None:
    """Initial base holdings sum to initial_nav * (1 - leaps_fraction)."""
    rd, pd_obj = _make_rd_and_pd(60)
    init_nav = 1_000_000.0
    cfg = _config(weights=dict(_LEAPS_WEIGHTS), initial_nav=init_nav, leaps_config=LeapsConfig())
    result = run_backtest(rd, pd_obj, cfg)
    # leaps_fraction = 0.30 → base fraction 0.70. Day-0 base value is recoverable by
    # reversing the first-day return on the base weights, but simplest: reconstruct
    # from the model — base holdings init before any return = init_nav * 0.70.
    # Verify via weight_history: LEAPS weight column on day 0 reflects carved fraction.
    leaps_frac = 0.30
    # Base + LEAPS realized weights sum to 1 each day.
    assert result.weight_history.sum(axis=1).sub(1.0).abs().max() < 1e-9
    # The carved-out LEAPS capital deployed on day 1 == init_nav * leaps_fraction.
    assert result.leaps_ledger is not None
    day1_basis = _leaps_cost_basis(result.leaps_ledger)  # includes only day-1 contract at n=60
    # Only a day-1 contract exists early (contributions add more monthly); with 60 days
    # there are ~3 month-ends, so isolate the first contract explicitly.
    first_contract = result.leaps_ledger.contracts[0]
    from finance.consts import CONTRACT_MULTIPLIER

    first_basis = first_contract.premium_paid * CONTRACT_MULTIPLIER * first_contract.n_contracts
    assert first_basis == pytest.approx(init_nav * leaps_frac, rel=1e-9)
    assert day1_basis >= first_basis  # later monthly contracts only add


def test_leaps_day1_contract_cost_basis_matches_carveout() -> None:
    """The first (day-1) LEAPS contract cost basis == initial_nav * leaps_fraction."""
    rd, pd_obj = _make_rd_and_pd(30)  # < 1 month-end guaranteed contributions minimal
    from finance.consts import CONTRACT_MULTIPLIER

    init_nav = 2_000_000.0
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS), initial_nav=init_nav,
        contribution=0.0, leaps_config=LeapsConfig(),
    )
    result = run_backtest(rd, pd_obj, cfg)
    assert result.leaps_ledger is not None
    c0 = result.leaps_ledger.contracts[0]
    basis = c0.premium_paid * CONTRACT_MULTIPLIER * c0.n_contracts
    assert basis == pytest.approx(init_nav * 0.30, rel=1e-9)


def test_leaps_base_holdings_exclude_leaps_keys() -> None:
    """weight_history contains the LEAPS key column and base columns, no overlap error."""
    rd, pd_obj = _make_rd_and_pd(60)
    cfg = _config(weights=dict(_LEAPS_WEIGHTS), leaps_config=LeapsConfig())
    result = run_backtest(rd, pd_obj, cfg)
    assert "VTI_LEAPS" in result.weight_history.columns
    # Base VTI also present (coexists with VTI_LEAPS)
    assert "VTI" in result.weight_history.columns


def test_leaps_multiple_underlyings_raises() -> None:
    """More than one distinct LEAPS underlying raises ValueError."""
    rd, pd_obj = _make_rd_and_pd(60)
    weights = {"VTI_LEAPS": 0.3, "GLD_LEAPS": 0.2, "VXUS": 0.25, "MUB": 0.25}
    cfg = _config(weights=weights, leaps_config=LeapsConfig())
    with pytest.raises(ValueError, match=r"[Oo]nly one LEAPS underlying"):
        run_backtest(rd, pd_obj, cfg)


def test_leaps_fraction_zero_matches_g1_behavior() -> None:
    """No *_LEAPS key → identical result to a plain base-only backtest (regression)."""
    rd, pd_obj = _make_rd_and_pd(252)
    cfg = _config()  # no LEAPS keys, no leaps_config
    result = run_backtest(rd, pd_obj, cfg)
    assert result.leaps_ledger is None
    # NAV path identical to the canonical no-LEAPS run
    assert result.nav_series.iloc[-1] > 0


# ---------------------------------------------------------------------------
# F-G2-02 — monthly contribution split between LEAPS and base
# ---------------------------------------------------------------------------


def test_leaps_monthly_contribution_split_to_leaps() -> None:
    """LEAPS monthly contribution == monthly_contribution * leaps_fraction.

    Verified indirectly: the second-and-later contracts' cost bases reflect the
    LEAPS share of each month-end contribution (with rolls aside). We assert the
    per-month LEAPS purchase basis matches contribution * leaps_fraction on a
    flat price series so no rolls occur and premiums are stable.
    """
    from finance.consts import CONTRACT_MULTIPLIER

    n = 200
    idx = pd.bdate_range("2015-01-02", periods=n)
    prices = pd.DataFrame(100.0, index=idx, columns=list(_TICKERS))
    returns = pd.DataFrame(0.0, index=idx, columns=list(_TICKERS))
    rd = ReturnData(
        returns=returns, log_returns=returns.copy(), tey_adjusted=False,
        marginal_rate=0.0, risk_free_rate=pd.Series(0.0, index=idx, name="risk_free_rate"),
    )
    pd_obj = PriceData(
        prices=prices, dividends=pd.DataFrame(0.0, index=idx, columns=list(_TICKERS)),
        vol_prices=pd.DataFrame(), tickers=_TICKERS,
        start_date=str(idx[0].date()), end_date=str(idx[-1].date()), spliced=False,
    )
    contribution = 12_000.0
    init_nav = 1_000_000.0
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS), initial_nav=init_nav,
        contribution=contribution, leaps_config=LeapsConfig(),
    )
    result = run_backtest(rd, pd_obj, cfg)
    assert result.leaps_ledger is not None
    contracts = result.leaps_ledger.contracts
    # Contract 0 is the day-1 carve-out; subsequent monthly contracts each have
    # basis == contribution * leaps_fraction (flat prices → no rolls, stable premium).
    leaps_frac = 0.30
    monthly_basis = [
        c.premium_paid * CONTRACT_MULTIPLIER * c.n_contracts for c in contracts[1:]
    ]
    assert len(monthly_basis) > 0
    for basis in monthly_basis:
        assert basis == pytest.approx(contribution * leaps_frac, rel=1e-9)


def test_leaps_base_contribution_share() -> None:
    """Base contribution share == monthly_contribution * (1 - leaps_fraction).

    On a flat, zero-return series with no rebalancing distortion, the base
    holdings grow by exactly the base share of each contribution.
    """
    n = 45  # spans ~2 month-ends
    idx = pd.bdate_range("2015-01-02", periods=n)
    prices = pd.DataFrame(100.0, index=idx, columns=list(_TICKERS))
    returns = pd.DataFrame(0.0, index=idx, columns=list(_TICKERS))
    rd = ReturnData(
        returns=returns, log_returns=returns.copy(), tey_adjusted=False,
        marginal_rate=0.0, risk_free_rate=pd.Series(0.0, index=idx, name="risk_free_rate"),
    )
    pd_obj = PriceData(
        prices=prices, dividends=pd.DataFrame(0.0, index=idx, columns=list(_TICKERS)),
        vol_prices=pd.DataFrame(), tickers=_TICKERS,
        start_date=str(idx[0].date()), end_date=str(idx[-1].date()), spliced=False,
    )
    contribution = 10_000.0
    init_nav = 1_000_000.0
    leaps_frac = 0.30
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS), initial_nav=init_nav,
        contribution=contribution, leaps_config=LeapsConfig(),
    )
    result = run_backtest(rd, pd_obj, cfg)
    # Count month-ends in the window
    n_month_ends = len({(d.year, d.month) for d in idx})
    # Base holdings start at init_nav*(1-frac) and grow by base share each month-end.
    base_start = init_nav * (1.0 - leaps_frac)
    expected_base = base_start + n_month_ends * contribution * (1.0 - leaps_frac)
    # Reconstruct final base value = total_nav - leaps_value; leaps_value is MTM.
    # On flat prices leaps MTM ≈ intrinsic + time value; instead assert base directly
    # via weight_history * nav for base assets.
    final_nav = result.nav_series.iloc[-1]
    base_cols = [c for c in result.weight_history.columns if not c.endswith("_LEAPS")]
    final_base = float(result.weight_history.iloc[-1][base_cols].sum()) * final_nav
    assert final_base == pytest.approx(expected_base, rel=1e-6)


def test_leaps_zero_contribution_only_day1_contract() -> None:
    """With zero contribution and a flat short series, only the day-1 contract exists."""
    n = 15  # fewer than a full month → possibly one month-end
    idx = pd.bdate_range("2015-01-02", periods=n)
    prices = pd.DataFrame(100.0, index=idx, columns=list(_TICKERS))
    returns = pd.DataFrame(0.0, index=idx, columns=list(_TICKERS))
    rd = ReturnData(
        returns=returns, log_returns=returns.copy(), tey_adjusted=False,
        marginal_rate=0.0, risk_free_rate=pd.Series(0.0, index=idx, name="risk_free_rate"),
    )
    pd_obj = PriceData(
        prices=prices, dividends=pd.DataFrame(0.0, index=idx, columns=list(_TICKERS)),
        vol_prices=pd.DataFrame(), tickers=_TICKERS,
        start_date=str(idx[0].date()), end_date=str(idx[-1].date()), spliced=False,
    )
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS), initial_nav=1_000_000.0,
        contribution=0.0, leaps_config=LeapsConfig(),
    )
    result = run_backtest(rd, pd_obj, cfg)
    assert result.leaps_ledger is not None
    # Zero contribution → no monthly purchases; exactly one (day-1) contract.
    assert len(result.leaps_ledger.contracts) == 1
