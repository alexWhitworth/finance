"""Tests for volatility.py — EWMA vol, rolling covariance, and contribution table."""

import numpy as np
import pandas as pd
import pytest

from finance.data import PriceData
from finance.returns import ReturnData, build_return_data
from finance.volatility import (
    EWMA_LAMBDA,
    TRADING_DAYS_PER_YEAR,
    VolatilityModel,
    build_covariance_matrix,
    build_vol_contribution_table,
    build_volatility_model,
    compute_ewma_vol,
    compute_realized_vol,
    compute_rolling_weekly_corr,
    compute_vol_contributions,
    forecast_portfolio_vol,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _return_series(n: int = 300, vol: float = 0.01, seed: int = 0) -> pd.Series:
    """Synthetic daily return series with known daily vol."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.Series(rng.normal(0.0, vol, n), index=idx, name="VTI")


def _return_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic 6-asset daily return DataFrame."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    tickers = ["VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT"]
    vols = [0.012, 0.013, 0.009, 0.003, 0.008, 0.004]
    data = {t: rng.normal(0.0, v, n) for t, v in zip(tickers, vols, strict=True)}
    return pd.DataFrame(data, index=idx)


def _make_return_data(n: int = 400) -> ReturnData:
    """Build a ReturnData from synthetic prices for 6 assets."""
    idx = pd.bdate_range("2019-01-02", periods=n + 1)
    rng = np.random.default_rng(7)
    tickers = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")
    starts = {"VTI": 200.0, "VXUS": 60.0, "GLD": 170.0, "MUB": 55.0, "KMLM": 25.0, "VGIT": 65.0}
    prices_data = {
        t: starts[t] * np.cumprod(1 + rng.normal(0.0003, 0.01, n + 1)) for t in tickers
    }
    prices = pd.DataFrame(prices_data, index=idx)
    dividends = pd.DataFrame(0.0, index=idx, columns=list(tickers))
    pd_obj = PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=pd.DataFrame(),
        tickers=tickers,
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
    )
    return build_return_data(pd_obj, apply_tey=False)


# ---------------------------------------------------------------------------
# compute_ewma_vol
# ---------------------------------------------------------------------------


def test_ewma_vol_length_matches_input() -> None:
    """Output series has same length as input."""
    r = _return_series(200)
    vol = compute_ewma_vol(r)
    assert len(vol) == len(r)


def test_ewma_vol_all_positive() -> None:
    """EWMA vol is strictly positive for all observations."""
    r = _return_series(200)
    vol = compute_ewma_vol(r)
    assert (vol > 0).all()


def test_ewma_vol_convergence_known_vol() -> None:
    """For a long i.i.d. series with daily vol v, EWMA converges near v*sqrt(252)."""
    daily_vol = 0.01
    # Use a long series (1000 obs) so the EWMA has time to converge
    r = _return_series(n=1000, vol=daily_vol, seed=42)
    vol = compute_ewma_vol(r, lambda_=EWMA_LAMBDA)
    # Tail average should be within 10% of true annualized vol
    tail_avg = float(vol.iloc[-100:].mean())
    expected = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert tail_avg == pytest.approx(expected, rel=0.10)


def test_ewma_vol_recursive_formula() -> None:
    """Verify the EWMA recursive formula at a single step.

    With a 3-observation series the warmup window is all 3 obs (warmup=min(30,3)=3),
    so var[0] = var(r[:3], ddof=1). Then var[1] = lambda*var[0] + (1-lambda)*r[0]^2.
    """
    vals = [0.02, -0.01, 0.015]
    r = pd.Series(vals, index=pd.bdate_range("2022-01-03", periods=3))
    lambda_ = 0.94
    vol = compute_ewma_vol(r, lambda_=lambda_)
    # var[0] = sample variance of the full 3-obs warmup window
    var0 = float(np.var(r.values, ddof=1))
    # var[1] = lambda * var[0] + (1-lambda) * r[0]^2
    var1 = lambda_ * var0 + (1.0 - lambda_) * vals[0] ** 2
    expected_vol1 = float(np.sqrt(var1 * TRADING_DAYS_PER_YEAR))
    assert vol.iloc[1] == pytest.approx(expected_vol1, rel=1e-9)


def test_ewma_vol_raises_invalid_lambda() -> None:
    """lambda_ outside (0, 1) raises ValueError."""
    r = _return_series()
    with pytest.raises(ValueError, match="lambda_"):
        compute_ewma_vol(r, lambda_=0.0)
    with pytest.raises(ValueError, match="lambda_"):
        compute_ewma_vol(r, lambda_=1.0)
    with pytest.raises(ValueError, match="lambda_"):
        compute_ewma_vol(r, lambda_=1.5)


def test_ewma_vol_raises_empty() -> None:
    """Empty series raises ValueError."""
    r = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="empty"):
        compute_ewma_vol(r)


# ---------------------------------------------------------------------------
# compute_rolling_weekly_corr
# ---------------------------------------------------------------------------


def test_rolling_corr_shape() -> None:
    """Output is N x N for N assets."""
    returns = _return_df(400)
    corr = compute_rolling_weekly_corr(returns)
    n = returns.shape[1]
    assert corr.shape == (n, n)


def test_rolling_corr_diagonal_is_one() -> None:
    """Diagonal of correlation matrix is 1.0."""
    returns = _return_df(400)
    corr = compute_rolling_weekly_corr(returns)
    np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)


def test_rolling_corr_symmetric() -> None:
    """Correlation matrix is symmetric."""
    returns = _return_df(400)
    corr = compute_rolling_weekly_corr(returns)
    np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)


def test_rolling_corr_bounded() -> None:
    """All correlations are in [-1, 1]."""
    returns = _return_df(400)
    corr = compute_rolling_weekly_corr(returns)
    assert (corr.values >= -1.0 - 1e-10).all()
    assert (corr.values <= 1.0 + 1e-10).all()


def test_rolling_corr_uses_window() -> None:
    """With window < total weeks, only last window_weeks rows are used."""
    returns = _return_df(600)
    # Full-history correlation vs short-window correlation should differ
    corr_full = compute_rolling_weekly_corr(returns, window_weeks=500)
    corr_short = compute_rolling_weekly_corr(returns, window_weeks=10)
    # They should not be identical
    assert not np.allclose(corr_full.values, corr_short.values)


def test_rolling_corr_raises_insufficient_data() -> None:
    """Fewer than 2 weekly observations raises ValueError."""
    # 4 business days resamples to exactly 1 Friday → only 1 weekly obs
    idx = pd.bdate_range("2022-01-03", periods=4)  # Mon-Thu, no Friday
    rng = np.random.default_rng(5)
    returns = pd.DataFrame({"A": rng.normal(0, 0.01, 4), "B": rng.normal(0, 0.01, 4)}, index=idx)
    with pytest.raises(ValueError, match="2 weeks"):
        compute_rolling_weekly_corr(returns)


def test_rolling_corr_known_perfect_correlation() -> None:
    """Two identical return series have correlation 1.0."""
    idx = pd.bdate_range("2020-01-02", periods=300)
    r = np.random.default_rng(0).normal(0, 0.01, 300)
    returns = pd.DataFrame({"A": r, "B": r}, index=idx)
    corr = compute_rolling_weekly_corr(returns)
    assert corr.loc["A", "B"] == pytest.approx(1.0, abs=1e-10)


def test_rolling_corr_known_perfect_neg_correlation() -> None:
    """Two perfectly negatively correlated daily series have weekly correlation ~ -1.0.

    Weekly compounding of -r vs r is not exactly -1, so we allow 1% tolerance.
    """
    idx = pd.bdate_range("2020-01-02", periods=300)
    r = np.random.default_rng(1).normal(0, 0.01, 300)
    returns = pd.DataFrame({"A": r, "B": -r}, index=idx)
    corr = compute_rolling_weekly_corr(returns)
    assert corr.loc["A", "B"] == pytest.approx(-1.0, abs=0.01)


# ---------------------------------------------------------------------------
# build_covariance_matrix
# ---------------------------------------------------------------------------


def test_cov_matrix_shape() -> None:
    """Output is N x N."""
    returns = _return_df(400)
    vols = pd.Series(dict.fromkeys(returns.columns, 0.15))
    corr = compute_rolling_weekly_corr(returns)
    cov = build_covariance_matrix(vols, corr)
    n = len(returns.columns)
    assert cov.shape == (n, n)


def test_cov_matrix_diagonal_known_value() -> None:
    """Diagonal entry = sigma_i^2 + ridge for uncorrelated assets."""
    assets = ["A", "B"]
    vols = pd.Series({"A": 0.10, "B": 0.20})
    corr = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=assets, columns=assets)
    from finance.volatility import COV_RIDGE
    cov = build_covariance_matrix(vols, corr)
    assert cov.loc["A", "A"] == pytest.approx(0.10**2 + COV_RIDGE, rel=1e-9)
    assert cov.loc["B", "B"] == pytest.approx(0.20**2 + COV_RIDGE, rel=1e-9)


def test_cov_matrix_off_diagonal_known_value() -> None:
    """Off-diagonal entry = sigma_i * rho_{ij} * sigma_j."""
    assets = ["A", "B"]
    vols = pd.Series({"A": 0.10, "B": 0.20})
    rho = 0.5
    corr = pd.DataFrame([[1.0, rho], [rho, 1.0]], index=assets, columns=assets)
    cov = build_covariance_matrix(vols, corr)
    expected_off = 0.10 * rho * 0.20
    assert cov.loc["A", "B"] == pytest.approx(expected_off, rel=1e-9)
    assert cov.loc["B", "A"] == pytest.approx(expected_off, rel=1e-9)


def test_cov_matrix_positive_definite() -> None:
    """Covariance matrix is positive definite (all eigenvalues > 0)."""
    returns = _return_df(400)
    vols = pd.Series(dict.fromkeys(returns.columns, 0.15))
    corr = compute_rolling_weekly_corr(returns)
    cov = build_covariance_matrix(vols, corr)
    eigenvalues = np.linalg.eigvalsh(cov.values)
    assert (eigenvalues > 0).all()


def test_cov_matrix_raises_missing_assets() -> None:
    """Raises if ewma_vols is missing an asset from corr_matrix."""
    assets = ["A", "B", "C"]
    vols = pd.Series({"A": 0.10, "B": 0.20})  # missing C
    corr = pd.DataFrame(np.eye(3), index=assets, columns=assets)
    with pytest.raises(ValueError, match="missing assets"):
        build_covariance_matrix(vols, corr)


# ---------------------------------------------------------------------------
# compute_vol_contributions
# ---------------------------------------------------------------------------


def test_vol_contributions_sum_to_one() -> None:
    """Contributions sum to exactly 1.0 for any valid weights and cov matrix."""
    returns = _return_df(400)
    vols = pd.Series({c: float(compute_ewma_vol(_return_series()).iloc[-1])
                      for c in returns.columns})
    corr = compute_rolling_weekly_corr(returns)
    cov = build_covariance_matrix(vols, corr)
    weights = pd.Series({c: 1.0 / len(returns.columns) for c in returns.columns})
    contribs = compute_vol_contributions(weights, cov)
    assert contribs.sum() == pytest.approx(1.0, abs=1e-10)


def test_vol_contributions_single_asset_is_one() -> None:
    """Single asset with weight 1.0 has contribution 1.0."""
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    weights = pd.Series({"A": 1.0})
    contribs = compute_vol_contributions(weights, cov)
    assert contribs["A"] == pytest.approx(1.0)


def test_vol_contributions_equal_weights_uncorrelated() -> None:
    """Equal weights + zero correlation → equal contributions."""
    assets = ["A", "B", "C", "D"]
    sigma = 0.15
    n = len(assets)
    cov = pd.DataFrame(sigma**2 * np.eye(n), index=assets, columns=assets)
    weights = pd.Series(dict.fromkeys(assets, 1.0 / n))
    contribs = compute_vol_contributions(weights, cov)
    np.testing.assert_allclose(contribs.values, 1.0 / n, atol=1e-10)


def test_vol_contributions_raises_unnormalized_weights() -> None:
    """Weights not summing to 1.0 raise ValueError."""
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=["A", "B"], columns=["A", "B"])
    weights = pd.Series({"A": 0.6, "B": 0.6})  # sums to 1.2
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        compute_vol_contributions(weights, cov)


def test_vol_contributions_all_nonnegative_positive_cov() -> None:
    """With a positive-definite covariance, all contributions are non-negative."""
    returns = _return_df(400)
    vols = pd.Series(dict.fromkeys(returns.columns, 0.15))
    corr = compute_rolling_weekly_corr(returns)
    cov = build_covariance_matrix(vols, corr)
    weights = pd.Series({c: 1.0 / len(returns.columns) for c in returns.columns})
    contribs = compute_vol_contributions(weights, cov)
    assert (contribs >= -1e-10).all()


# ---------------------------------------------------------------------------
# compute_realized_vol
# ---------------------------------------------------------------------------


def test_realized_vol_nan_in_warmup() -> None:
    """First window_days - 1 values are NaN."""
    r = _return_series(200)
    rvol = compute_realized_vol(r, window_days=90)
    assert rvol.iloc[:89].isna().all()
    assert not rvol.iloc[89:].isna().any()


def test_realized_vol_known_constant_series() -> None:
    """Constant return series has zero realized vol."""
    idx = pd.bdate_range("2022-01-03", periods=100)
    r = pd.Series(0.001, index=idx, name="X")
    rvol = compute_realized_vol(r, window_days=10)
    # std of a constant = 0; annualized = 0
    assert rvol.dropna().abs().max() == pytest.approx(0.0, abs=1e-12)


def test_realized_vol_known_value() -> None:
    """Realized vol of an i.i.d. series approximates daily_vol * sqrt(252)."""
    daily_vol = 0.02
    rng = np.random.default_rng(99)
    idx = pd.bdate_range("2020-01-02", periods=500)
    r = pd.Series(rng.normal(0, daily_vol, 500), index=idx)
    rvol = compute_realized_vol(r, window_days=252)
    tail_mean = float(rvol.dropna().mean())
    expected = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert tail_mean == pytest.approx(expected, rel=0.10)


# ---------------------------------------------------------------------------
# build_volatility_model
# ---------------------------------------------------------------------------


def test_build_volatility_model_returns_correct_types() -> None:
    """build_volatility_model returns a VolatilityModel with expected fields."""
    rd = _make_return_data(400)
    vm = build_volatility_model(rd)
    assert isinstance(vm, VolatilityModel)
    assert isinstance(vm.ewma_vols, pd.Series)
    assert isinstance(vm.rolling_corr, pd.DataFrame)
    assert isinstance(vm.cov_matrix, pd.DataFrame)
    assert vm.lambda_ == EWMA_LAMBDA


def test_build_volatility_model_vols_positive() -> None:
    """All per-asset EWMA vols are positive."""
    rd = _make_return_data(400)
    vm = build_volatility_model(rd)
    assert (vm.ewma_vols > 0).all()


def test_build_volatility_model_immutable() -> None:
    """VolatilityModel is frozen."""
    rd = _make_return_data(400)
    vm = build_volatility_model(rd)
    with pytest.raises((AttributeError, TypeError)):
        vm.lambda_ = 0.9  # type: ignore[misc]


def test_build_volatility_model_as_of_date() -> None:
    """Slicing to an earlier as_of_date produces different vols than the full history."""
    rd = _make_return_data(600)
    vm_full = build_volatility_model(rd)
    mid_date = rd.returns.index[200]
    vm_mid = build_volatility_model(rd, as_of_date=mid_date)
    # EWMA vols at mid-date should differ from full-history vols
    assert not np.allclose(vm_full.ewma_vols.values, vm_mid.ewma_vols.values)


def test_build_volatility_model_raises_bad_date() -> None:
    """as_of_date before first return raises ValueError."""
    rd = _make_return_data(200)
    bad_date = rd.returns.index[0] - pd.Timedelta(days=5)
    with pytest.raises(ValueError, match="as_of_date"):
        build_volatility_model(rd, as_of_date=bad_date)


def test_build_volatility_model_excludes_vol_index_tickers() -> None:
    """Vol index tickers (e.g. ^VIX) present in return_data are silently excluded."""
    rd = _make_return_data(400)
    # Inject a fake ^VIX column into the returns DataFrame
    returns_with_vix = rd.returns.copy()
    returns_with_vix["^VIX"] = 0.20
    log_with_vix = rd.log_returns.copy()
    log_with_vix["^VIX"] = 0.20
    from finance.returns import ReturnData
    rd_with_vix = ReturnData(
        returns=returns_with_vix,
        log_returns=log_with_vix,
        tey_adjusted=rd.tey_adjusted,
        marginal_rate=rd.marginal_rate,
        risk_free_rate=rd.risk_free_rate,
    )
    vm = build_volatility_model(rd_with_vix)
    assert "^VIX" not in vm.ewma_vols.index
    assert "^VIX" not in vm.cov_matrix.columns


# ---------------------------------------------------------------------------
# build_vol_contribution_table
# ---------------------------------------------------------------------------


def test_vol_contribution_table_columns() -> None:
    """Table has expected columns."""
    rd = _make_return_data(400)
    vm = build_volatility_model(rd)
    tickers = list(rd.returns.columns)
    weights = pd.Series({t: 1.0 / len(tickers) for t in tickers})
    table = build_vol_contribution_table(weights, rd, vm)
    assert set(table.columns) == {"sigma_tilde", "sigma_hat", "rho_VTI", "contrib"}


def test_vol_contribution_table_contrib_sums_to_one() -> None:
    """contrib column sums to 1.0."""
    rd = _make_return_data(400)
    vm = build_volatility_model(rd)
    tickers = list(rd.returns.columns)
    weights = pd.Series({t: 1.0 / len(tickers) for t in tickers})
    table = build_vol_contribution_table(weights, rd, vm)
    assert table["contrib"].sum() == pytest.approx(1.0, abs=1e-10)


def test_vol_contribution_table_vti_rho_is_one() -> None:
    """VTI's correlation with itself is 1.0."""
    rd = _make_return_data(400)
    vm = build_volatility_model(rd)
    tickers = list(rd.returns.columns)
    weights = pd.Series({t: 1.0 / len(tickers) for t in tickers})
    table = build_vol_contribution_table(weights, rd, vm)
    assert table.loc["VTI", "rho_VTI"] == pytest.approx(1.0, abs=1e-10)


def test_vol_contribution_table_sigma_hat_matches_model() -> None:
    """sigma_hat column matches vol_model.ewma_vols."""
    rd = _make_return_data(400)
    vm = build_volatility_model(rd)
    tickers = list(rd.returns.columns)
    weights = pd.Series({t: 1.0 / len(tickers) for t in tickers})
    table = build_vol_contribution_table(weights, rd, vm)
    np.testing.assert_allclose(
        table["sigma_hat"].values,
        vm.ewma_vols[tickers].values,
        rtol=1e-9,
    )


def test_vol_contribution_table_no_vti_rho_is_nan() -> None:
    """When VTI is absent from the universe, rho_VTI is NaN for all assets."""
    # Build a 2-asset universe without VTI
    idx = pd.bdate_range("2019-01-02", periods=401)
    rng = np.random.default_rng(11)
    tickers_no_vti = ("VXUS", "GLD")
    prices_data = {t: 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, 401)) for t in tickers_no_vti}
    prices = pd.DataFrame(prices_data, index=idx)
    dividends = pd.DataFrame(0.0, index=idx, columns=list(tickers_no_vti))
    from finance.data import PriceData
    pd_obj = PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=pd.DataFrame(),
        tickers=tickers_no_vti,
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
    )
    from finance.returns import build_return_data
    rd_no_vti = build_return_data(pd_obj, apply_tey=False)
    vm_no_vti = build_volatility_model(rd_no_vti)
    weights = pd.Series(dict.fromkeys(tickers_no_vti, 0.5))
    table = build_vol_contribution_table(weights, rd_no_vti, vm_no_vti)
    assert table["rho_VTI"].isna().all()


def test_vol_contribution_table_sigma_tilde_nan_for_short_series() -> None:
    """sigma_tilde is NaN when the return series is shorter than the 90-day window."""
    # Use only 50 obs — shorter than the 90-day realized vol window
    rd_short = _make_return_data(n=50)
    vm_short = build_volatility_model(rd_short)
    tickers = list(rd_short.returns.columns)
    weights = pd.Series({t: 1.0 / len(tickers) for t in tickers})
    table = build_vol_contribution_table(weights, rd_short, vm_short)
    # All sigma_tilde values should be NaN (rolling window not yet full)
    assert table["sigma_tilde"].isna().all()


# ---------------------------------------------------------------------------
# forecast_portfolio_vol
# ---------------------------------------------------------------------------


def test_forecast_portfolio_vol_positive() -> None:
    """Forecasted portfolio vol is positive."""
    rd = _make_return_data(400)
    vm = build_volatility_model(rd)
    tickers = list(rd.returns.columns)
    weights = pd.Series({t: 1.0 / len(tickers) for t in tickers})
    pv = forecast_portfolio_vol(weights, vm)
    assert pv > 0.0


def test_forecast_portfolio_vol_single_asset() -> None:
    """Single-asset portfolio vol equals that asset's EWMA vol (+ ridge effect)."""
    assets = ["A"]
    sigma = 0.15
    from finance.volatility import COV_RIDGE
    cov = pd.DataFrame([[sigma**2 + COV_RIDGE]], index=assets, columns=assets)
    vm = VolatilityModel(
        ewma_vols=pd.Series({"A": sigma}),
        rolling_corr=pd.DataFrame([[1.0]], index=assets, columns=assets),
        cov_matrix=cov,
        lambda_=EWMA_LAMBDA,
    )
    weights = pd.Series({"A": 1.0})
    pv = forecast_portfolio_vol(weights, vm)
    expected = np.sqrt(sigma**2 + COV_RIDGE)
    assert pv == pytest.approx(expected, rel=1e-9)


def test_forecast_portfolio_vol_diversification() -> None:
    """Portfolio of uncorrelated equal-weight assets has lower vol than any single asset."""
    n = 4
    assets = [f"A{i}" for i in range(n)]
    sigma = 0.15
    cov_vals = sigma**2 * np.eye(n)
    cov = pd.DataFrame(cov_vals, index=assets, columns=assets)
    vm = VolatilityModel(
        ewma_vols=pd.Series(dict.fromkeys(assets, sigma)),
        rolling_corr=pd.DataFrame(np.eye(n), index=assets, columns=assets),
        cov_matrix=cov,
        lambda_=EWMA_LAMBDA,
    )
    weights = pd.Series(dict.fromkeys(assets, 1.0 / n))
    pv = forecast_portfolio_vol(weights, vm)
    # Equal-weight uncorrelated: sigma_p = sigma / sqrt(n)
    assert pv == pytest.approx(sigma / np.sqrt(n), rel=1e-6)
