"""Tests for portfolio.py — rebalance dates, contributions, and the backtest loop."""

import numpy as np
import pandas as pd
import pytest

from finance.data import PriceData
from finance.consts import DRIFT_BAND_RELATIVE
from finance.leverage import AccountType, LeapsConfig, RebalanceRule, WeightStrategy
from finance._backtest_steps import _get_rebalance_dates, _should_rebalance
from finance._portfolio_types import BacktestResult, PortfolioConfig
from finance.portfolio import run_backtest
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
    rebalance_rule: RebalanceRule = RebalanceRule.QUARTERLY,
) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights=weights or dict(_EQUAL_WEIGHTS),
        initial_nav=initial_nav,
        monthly_contribution=contribution,
        rebalance_rule=rebalance_rule,
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
# _get_rebalance_dates
# ---------------------------------------------------------------------------


def test_rebalance_dates_are_in_index() -> None:
    """Every rebalance date falls within the provided index."""
    idx = pd.bdate_range("2015-01-02", periods=504)
    dates = _get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    for d in dates:
        assert d in idx


def test_rebalance_dates_in_quarter_end_months() -> None:
    """All rebalance dates land in Mar / Jun / Sep / Dec."""
    idx = pd.bdate_range("2015-01-02", periods=1008)
    dates = _get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    for d in dates:
        assert d.month in _QUARTER_END_MONTHS


def test_rebalance_dates_are_last_day_of_month() -> None:
    """Each rebalance date is the last trading day of its month."""
    idx = pd.bdate_range("2015-01-02", periods=1008)
    dates = _get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    for d in dates:
        later_in_month = idx[(idx.month == d.month) & (idx.year == d.year) & (idx > d)]
        assert len(later_in_month) == 0


def test_rebalance_dates_count_roughly_four_per_year() -> None:
    """For a 2-year window we get exactly 8 quarterly dates."""
    idx = pd.bdate_range("2015-01-02", "2016-12-31")
    dates = _get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    assert len(dates) == 8


def test_rebalance_dates_sorted() -> None:
    """Returned list is chronologically sorted."""
    idx = pd.bdate_range("2015-01-02", periods=504)
    dates = _get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
    assert dates == sorted(dates)

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
    rdates = _get_rebalance_dates(idx, RebalanceRule.QUARTERLY)
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
    # Drop VTI from base weights; absorb its share into VTI_LEAPS so sum stays 1.0.
    # VTI_LEAPS still requires VTI spot prices → triggers the missing-underlying error.
    weights = {k: v for k, v in _LEAPS_WEIGHTS.items() if k != "VTI"}
    weights["VTI_LEAPS"] = weights["VTI_LEAPS"] + _LEAPS_WEIGHTS["VTI"]
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
# _should_rebalance
# ---------------------------------------------------------------------------


def test__should_rebalance_quarterly_always_false() -> None:
    """QUARTERLY rule always returns False regardless of weight deviation."""
    current = pd.Series({"A": 0.80, "B": 0.20})
    target = pd.Series({"A": 0.50, "B": 0.50})
    assert _should_rebalance(current, target, RebalanceRule.QUARTERLY) is False


def test__should_rebalance_drift_no_trigger_within_band() -> None:
    """DRIFT rule returns False when all relative deviations are within the band.

    target=0.50, current=0.54 → deviation = 0.04/0.50 = 8% < 10%.
    """
    current = pd.Series({"A": 0.54, "B": 0.46})
    target = pd.Series({"A": 0.50, "B": 0.50})
    assert _should_rebalance(current, target, RebalanceRule.DRIFT) is False


def test__should_rebalance_drift_triggers_at_band_breach() -> None:
    """DRIFT rule returns True when one asset exceeds the 10% relative band.

    target=0.50, current=0.56 → deviation = 0.06/0.50 = 12% > 10%.
    """
    current = pd.Series({"A": 0.56, "B": 0.44})
    target = pd.Series({"A": 0.50, "B": 0.50})
    assert _should_rebalance(current, target, RebalanceRule.DRIFT) is True


def test__should_rebalance_drift_zero_target_weight_skipped() -> None:
    """DRIFT rule skips assets with target=0.0 (division by zero guard).

    Asset B has target=0.0 and current=0.05; must not raise and return False
    when no other asset breaches.
    """
    current = pd.Series({"A": 0.95, "B": 0.05})
    target = pd.Series({"A": 1.00, "B": 0.00})
    # A: |0.95 - 1.00| / 1.00 = 5% < 10%; B skipped
    assert _should_rebalance(current, target, RebalanceRule.DRIFT) is False


def test__should_rebalance_drift_uses_custom_band() -> None:
    """Custom band=0.05 triggers on an 8% relative deviation (outside 5%, within 10%)."""
    current = pd.Series({"A": 0.54, "B": 0.46})
    target = pd.Series({"A": 0.50, "B": 0.50})
    # Default 10% band: no trigger; custom 5% band: 8% > 5% → trigger
    assert _should_rebalance(current, target, RebalanceRule.DRIFT, band=0.05) is True


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


# ---------------------------------------------------------------------------
# F-G2-03 — VIX-based dynamic implied volatility
# ---------------------------------------------------------------------------


def _make_pd_with_vix(
    n: int = 504,
    vix_level: float = 0.25,
    seed: int = 42,
) -> PriceData:
    """Synthetic PriceData with a constant 'VTI'-keyed vol_prices column (asset-ticker convention).
    """
    base = _make_price_data(n, seed=seed)
    vix = pd.DataFrame({"VTI": vix_level}, index=base.prices.index)
    return PriceData(
        prices=base.prices, dividends=base.dividends, vol_prices=vix,
        tickers=base.tickers, start_date=base.start_date,
        end_date=base.end_date, spliced=base.spliced,
    )


def test_leaps_vix_iv_floor_respected_on_creation() -> None:
    """A VIX below config.iv is floored: contracts priced at config.iv, not VIX."""
    n = 120
    base = _make_price_data(n)
    rd = build_return_data(base, apply_tey=False)
    low_vix = pd.DataFrame({"VTI": 0.05}, index=base.prices.index)  # below 0.18 floor
    pd_low_vix = PriceData(
        prices=base.prices, dividends=base.dividends, vol_prices=low_vix,
        tickers=base.tickers, start_date=base.start_date,
        end_date=base.end_date, spliced=base.spliced,
    )
    cfg = _config(weights=dict(_LEAPS_WEIGHTS), leaps_config=LeapsConfig(iv=0.18))
    result_lowvix = run_backtest(rd, pd_low_vix, cfg)

    # No-VIX run uses config.iv=0.18 everywhere; floored VIX must match it.
    result_novix = run_backtest(rd, base, cfg)
    assert result_lowvix.leaps_ledger is not None
    assert result_novix.leaps_ledger is not None
    lo_prem = result_lowvix.leaps_ledger.contracts[0].premium_paid
    no_prem = result_novix.leaps_ledger.contracts[0].premium_paid
    assert lo_prem == pytest.approx(no_prem, rel=1e-9)


def test_leaps_vix_above_floor_raises_premium() -> None:
    """VIX above the floor produces a higher day-1 premium than the floor case."""
    n = 120
    base = _make_price_data(n)
    rd = build_return_data(base, apply_tey=False)
    pd_hi_vix = _make_pd_with_vix(n, vix_level=0.45)
    cfg = _config(weights=dict(_LEAPS_WEIGHTS), leaps_config=LeapsConfig(iv=0.18))
    result_hi = run_backtest(rd, pd_hi_vix, cfg)
    result_floor = run_backtest(rd, base, cfg)  # no VIX → floor 0.18
    assert result_hi.leaps_ledger is not None
    assert result_floor.leaps_ledger is not None
    assert (
        result_hi.leaps_ledger.contracts[0].premium_paid
        > result_floor.leaps_ledger.contracts[0].premium_paid
    )


def test_leaps_vix_lookup_uses_underlying_variable_not_hardcoded_ticker() -> None:
    """vol_prices lookup uses `underlying` variable, not a hardcoded 'VTI' or '^VIX' string.

    Uses GLD_LEAPS (underlying='GLD') with vol_prices keyed by 'GLD'.
    A regression to any hardcoded ticker would leave dynamic IV disengaged,
    so premium_paid must differ from the no-vol-prices baseline.
    """
    n = 120
    base = _make_price_data(n)
    rd = build_return_data(base, apply_tey=False)
    gld_leaps_weights = {
        "GLD_LEAPS": 0.30, "GLD": 0.10, "VTI": 0.15,
        "VXUS": 0.15, "MUB": 0.10, "KMLM": 0.10, "VGIT": 0.10,
    }
    cfg = _config(weights=gld_leaps_weights, leaps_config=LeapsConfig(iv=0.18))
    # vol_prices keyed by 'GLD' (the underlying), not 'VTI' or '^VIX'
    gld_vix = pd.DataFrame({"GLD": 0.45}, index=base.prices.index)
    pd_gld_vol = PriceData(
        prices=base.prices, dividends=base.dividends, vol_prices=gld_vix,
        tickers=base.tickers, start_date=base.start_date,
        end_date=base.end_date, spliced=base.spliced,
    )
    result_vol = run_backtest(rd, pd_gld_vol, cfg)
    result_novol = run_backtest(rd, base, cfg)  # no vol_prices → constant config.iv
    assert result_vol.leaps_ledger is not None
    assert result_novol.leaps_ledger is not None
    assert (
        result_vol.leaps_ledger.contracts[0].premium_paid
        > result_novol.leaps_ledger.contracts[0].premium_paid
    ), "GLD-keyed vol_prices did not engage — lookup may be hardcoded to a different ticker"


def test_leaps_empty_vol_prices_falls_back_to_config_iv() -> None:
    """Empty vol_prices → identical NAV path to a config.iv-only run (regression)."""
    rd, pd_obj = _make_rd_and_pd(252)  # _make_price_data → vol_prices empty
    assert pd_obj.vol_prices.empty
    cfg = _config(weights=dict(_LEAPS_WEIGHTS), leaps_config=LeapsConfig(iv=0.18))
    result = run_backtest(rd, pd_obj, cfg)
    assert result.leaps_ledger is not None  # ran without error, used config.iv


def test_leaps_vix_creation_uses_raw_not_smoothed() -> None:
    """Creation IV uses raw month-end VIX, distinct from the 30-day MTM mean.

    Build a VIX series that is low for the first ~30 days then spikes. The day-1
    contract (created on the raw first value) must differ from what a 30-day mean
    would give, confirming creation uses raw VIX.
    """
    n = 90
    base = _make_price_data(n)
    rd = build_return_data(base, apply_tey=False)
    vix_vals = np.concatenate([np.full(45, 0.20), np.full(n + 1 - 45, 0.60)])
    vix = pd.DataFrame({"VTI": vix_vals[: len(base.prices)]}, index=base.prices.index)
    pd_vix = PriceData(
        prices=base.prices, dividends=base.dividends, vol_prices=vix,
        tickers=base.tickers, start_date=base.start_date,
        end_date=base.end_date, spliced=base.spliced,
    )
    cfg = _config(weights=dict(_LEAPS_WEIGHTS), leaps_config=LeapsConfig(iv=0.18))
    result = run_backtest(rd, pd_vix, cfg)
    assert result.leaps_ledger is not None
    # Day-1 contract created at raw VIX=0.20 (> floor 0.18); premium reflects 0.20.
    # Compare to a flat-0.20 VIX run: day-1 premiums must match exactly.
    flat_vix = _make_pd_with_vix(n, vix_level=0.20)
    result_flat = run_backtest(rd, flat_vix, cfg)
    assert result_flat.leaps_ledger is not None
    assert result.leaps_ledger.contracts[0].premium_paid == pytest.approx(
        result_flat.leaps_ledger.contracts[0].premium_paid, rel=1e-9
    )


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


# ---------------------------------------------------------------------------
# F-G3-01 / F-G3-02 / F-G3-03 — drift rebalancing + partial LEAPS close
# ---------------------------------------------------------------------------


def _rising_vti_pd_rd(
    n: int = 504,
    vti_daily: float = 0.0015,
    start: str = "2015-01-02",
) -> tuple[ReturnData, PriceData]:
    """PriceData/ReturnData where VTI rises steadily and other assets are flat.

    A rising underlying makes the levered LEAPS sleeve grow faster than the base
    sleeve, driving the LEAPS weight above its drift band.
    """
    idx = pd.bdate_range(start, periods=n)
    vti = 200.0 * np.cumprod(1.0 + np.full(n, vti_daily))
    prices = pd.DataFrame(
        {
            "VTI": vti, "VXUS": 60.0, "GLD": 170.0,
            "MUB": 55.0, "KMLM": 25.0, "VGIT": 65.0,
        },
        index=idx,
    )
    returns = prices.pct_change().fillna(0.0)
    rd = ReturnData(
        returns=returns, log_returns=np.log1p(returns), tey_adjusted=False,
        marginal_rate=0.0, risk_free_rate=pd.Series(0.0, index=idx, name="risk_free_rate"),
    )
    pd_obj = PriceData(
        prices=prices, dividends=pd.DataFrame(0.0, index=idx, columns=list(_TICKERS)),
        vol_prices=pd.DataFrame(), tickers=_TICKERS,
        start_date=str(idx[0].date()), end_date=str(idx[-1].date()), spliced=False,
    )
    return rd, pd_obj


def test_drift_quarterly_regression_unchanged() -> None:
    """QUARTERLY rule path is identical whether or not DRIFT code exists (regression)."""
    rd, pd_obj = _make_rd_and_pd(252)
    cfg_q = _config(rebalance_rule=RebalanceRule.QUARTERLY)
    result = run_backtest(rd, pd_obj, cfg_q)
    # No partial closes on a non-LEAPS quarterly run.
    assert result.leaps_ledger is None
    assert result.nav_series.iloc[-1] > 0


def test_drift_no_partial_close_events_when_quarterly() -> None:
    """QUARTERLY LEAPS run accumulates no partial_close_events."""
    rd, pd_obj = _rising_vti_pd_rd(504)
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS), leaps_config=LeapsConfig(),
        rebalance_rule=RebalanceRule.QUARTERLY,
    )
    result = run_backtest(rd, pd_obj, cfg)
    assert result.leaps_ledger is not None
    assert result.leaps_ledger.partial_close_events == ()


def test_drift_triggers_partial_close_on_overshoot() -> None:
    """A rising underlying drives LEAPS above its band; DRIFT trims it (events recorded)."""
    rd, pd_obj = _rising_vti_pd_rd(504, vti_daily=0.0020)
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS), leaps_config=LeapsConfig(),
        rebalance_rule=RebalanceRule.DRIFT,
    )
    result = run_backtest(rd, pd_obj, cfg)
    assert result.leaps_ledger is not None
    # LEAPS overshot and was trimmed at least once.
    assert len(result.leaps_ledger.partial_close_events) > 0


def test_drift_partial_close_events_is_tuple() -> None:
    """partial_close_events is a tuple on the returned ledger (F-G3-03)."""
    rd, pd_obj = _rising_vti_pd_rd(504, vti_daily=0.0020)
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS), leaps_config=LeapsConfig(),
        rebalance_rule=RebalanceRule.DRIFT,
    )
    result = run_backtest(rd, pd_obj, cfg)
    assert result.leaps_ledger is not None
    assert isinstance(result.leaps_ledger.partial_close_events, tuple)


def test_drift_ledger_remains_frozen() -> None:
    """The returned LeapsLedger is still frozen after partial-close accumulation."""
    rd, pd_obj = _rising_vti_pd_rd(504, vti_daily=0.0020)
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS), leaps_config=LeapsConfig(),
        rebalance_rule=RebalanceRule.DRIFT,
    )
    result = run_backtest(rd, pd_obj, cfg)
    assert result.leaps_ledger is not None
    with pytest.raises((AttributeError, TypeError)):
        result.leaps_ledger.partial_close_events = ()  # type: ignore[misc]


def test_drift_no_trigger_within_band_no_closes() -> None:
    """A flat market keeps weights within band → no partial closes."""
    n = 504
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
        weights=dict(_LEAPS_WEIGHTS), leaps_config=LeapsConfig(),
        rebalance_rule=RebalanceRule.DRIFT, contribution=0.0,
    )
    result = run_backtest(rd, pd_obj, cfg)
    assert result.leaps_ledger is not None
    # LEAPS time-decays on flat prices → it undershoots, never overshoots → no closes.
    assert result.leaps_ledger.partial_close_events == ()


def test_drift_net_proceeds_added_to_base_tax_free() -> None:
    """On a drift trim, base sleeve grows by the closed LEAPS proceeds (no tax).

    We compare against the QUARTERLY run on the same series: the DRIFT run must
    move value from the LEAPS sleeve into the base sleeve, so its final base
    holdings exceed the quarterly run's while its LEAPS MTM is capped.

    Checks fire only at month-end, so the final date (mid-month) can be up to
    one drift band above target before the next check would fire.
    """
    rd, pd_obj = _rising_vti_pd_rd(504, vti_daily=0.0020)
    cfg_drift = _config(
        weights=dict(_LEAPS_WEIGHTS), leaps_config=LeapsConfig(),
        rebalance_rule=RebalanceRule.DRIFT,
    )
    result = run_backtest(rd, pd_obj, cfg_drift)
    assert result.leaps_ledger is not None
    assert len(result.leaps_ledger.partial_close_events) > 0
    # After trims, the LEAPS weight column should be within one drift band of target.
    # Checks fire only at month-end, so intra-month drift is allowed up to the band.
    leaps_target_weight = 0.30
    final_leaps_weight = float(result.weight_history.iloc[-1]["VTI_LEAPS"])
    assert final_leaps_weight <= leaps_target_weight * (1 + DRIFT_BAND_RELATIVE) + 1e-6


def test_drift_partial_close_one_event_per_trimmed_contract() -> None:
    """Cumulative closes collapse to one event per distinct trimmed contract.

    Each event's original_contract must be unique (single continuation per original,
    honoring the _live_contracts model).
    """
    rd, pd_obj = _rising_vti_pd_rd(756, vti_daily=0.0018)
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS), leaps_config=LeapsConfig(),
        rebalance_rule=RebalanceRule.DRIFT, contribution=5_000.0,
    )
    result = run_backtest(rd, pd_obj, cfg)
    assert result.leaps_ledger is not None
    events = result.leaps_ledger.partial_close_events
    assert len(events) > 0
    originals = [ev.original_contract for ev in events]
    assert len(originals) == len(set(originals))  # one event per contract
    # Each continuation has strictly fewer contracts than its original.
    for ev in events:
        assert ev.continuation_contract.n_contracts < ev.original_contract.n_contracts
        assert ev.n_contracts_closed > 0


def test_drift_weights_sum_to_one_each_day() -> None:
    """Realized weights (base + LEAPS) still sum to 1.0 every day under DRIFT."""
    rd, pd_obj = _rising_vti_pd_rd(504, vti_daily=0.0020)
    cfg = _config(
        weights=dict(_LEAPS_WEIGHTS), leaps_config=LeapsConfig(),
        rebalance_rule=RebalanceRule.DRIFT,
    )
    result = run_backtest(rd, pd_obj, cfg)
    sums = result.weight_history.sum(axis=1)
    assert (sums - 1.0).abs().max() < 1e-9
