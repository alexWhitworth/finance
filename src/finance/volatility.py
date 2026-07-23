"""EWMA volatility forecasting, rolling covariance, and volatility contribution.

All functions are pure. No I/O. Receives ReturnData from returns.py and produces
VolatilityModel consumed by metrics.py and portfolio.py.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance.returns import ReturnData

EWMA_LAMBDA: float = 0.95
ROLLING_CORR_WINDOW_WEEKS: int = 156  # 36 months ≈ 156 weeks
TRADING_DAYS_PER_YEAR: int = 252
COV_RIDGE: float = 1e-8  # added to diagonal to guarantee positive definiteness


@dataclass(frozen=True)
class VolatilityModel:
    """EWMA-based volatility and covariance model for a set of assets.

    Attributes:
        ewma_vols: Per-asset annualized EWMA volatility at as_of_date.
        rolling_corr: N x N correlation matrix from 36-month rolling weekly returns.
        cov_matrix: Sigma_hat_{t+1}, N x N forward covariance estimate.
        lambda_: EWMA decay parameter used to build ewma_vols.
    """

    ewma_vols: pd.Series
    rolling_corr: pd.DataFrame
    cov_matrix: pd.DataFrame
    lambda_: float


def compute_ewma_vol(
    returns: pd.Series,
    lambda_: float = EWMA_LAMBDA,
) -> pd.Series:
    """Compute time-series of EWMA annualized volatility from daily returns.

    Uses the recursive formula: sigma^2_{t+1} = lambda * sigma^2_t + (1-lambda) * r^2_t.
    Initialized with the variance of the first 30 observations (or all if fewer).

    Convention: output index i represents the forecast *entering* day i, incorporating
    returns through r[i-2]. The final observation r[-1] is not folded in, so iloc[-1]
    is a one-step-ahead forecast as of the second-to-last return. This is the standard
    RiskMetrics convention for a realized-vol series.

    Arguments:
        returns: Daily simple return Series (DatetimeIndex).
        lambda_: EWMA decay parameter. Must be in (0, 1). Defaults to 0.95.

    Returns:
        Series of annualized EWMA volatility (same index as returns).

    Raises:
        ValueError: If lambda_ is not in (0, 1).
        ValueError: If returns is empty.
    """
    if not (0.0 < lambda_ < 1.0):
        raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}")
    if returns.empty:
        raise ValueError("returns series is empty")

    r = returns.values.astype(float)
    n = len(r)
    var = np.empty(n)

    # Initialise with variance of warm-up window
    warmup = min(30, n)
    warmup_vals = np.asarray(r[:warmup], dtype=float)
    var[0] = float(np.var(warmup_vals, ddof=1)) if warmup > 1 else float(r[0] ** 2)

    for i in range(1, n):
        var[i] = lambda_ * var[i - 1] + (1.0 - lambda_) * r[i - 1] ** 2

    annualized_vol = np.sqrt(var * TRADING_DAYS_PER_YEAR)
    return pd.Series(annualized_vol, index=returns.index, name=returns.name)


def compute_rolling_weekly_corr(
    returns: pd.DataFrame,
    window_weeks: int = ROLLING_CORR_WINDOW_WEEKS,
) -> pd.DataFrame:
    """Compute the most recent rolling N x N correlation matrix from weekly returns.

    Resamples daily returns to weekly (Friday close), then applies a rolling
    window of window_weeks. Returns the correlation matrix at the last available
    observation.

    Arguments:
        returns: Daily simple return DataFrame (DatetimeIndex x assets).
        window_weeks: Rolling window length in weeks. Defaults to 156 (36 months).

    Returns:
        N x N correlation DataFrame (assets x assets). Diagonal is 1.0.

    Raises:
        ValueError: If fewer than 2 weeks of data are available.
    """
    weekly = returns.resample("W-FRI").apply(lambda x: (1 + x).prod() - 1)
    if len(weekly) < 2:
        raise ValueError(
            f"Need at least 2 weeks of data; got {len(weekly)}. "
            "Increase the date range."
        )
    effective_window = min(window_weeks, len(weekly))
    tail = weekly.iloc[-effective_window:]
    corr: pd.DataFrame = tail.corr()
    return corr


def build_covariance_matrix(
    ewma_vols: pd.Series,
    corr_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Build the forward covariance matrix from EWMA vols and rolling correlations.

    Sigma_hat_{ij} = sigma_hat_i * rho_hat_{ij} * sigma_hat_j.
    A small ridge (1e-8 * I) is added to guarantee positive definiteness.

    Arguments:
        ewma_vols: Per-asset annualized EWMA volatility (index = asset names).
        corr_matrix: N x N correlation matrix (index and columns = asset names).

    Returns:
        N x N covariance DataFrame aligned to corr_matrix index/columns.

    Raises:
        ValueError: If ewma_vols assets do not match corr_matrix index.
    """
    assets = corr_matrix.index.tolist()
    missing = [a for a in assets if a not in ewma_vols.index]
    if missing:
        raise ValueError(f"ewma_vols missing assets: {missing}")

    vols: np.ndarray[tuple[int], np.dtype[np.float64]] = ewma_vols[assets].to_numpy(dtype=float)
    corr: np.ndarray[tuple[int, int], np.dtype[np.float64]] = (
        corr_matrix.loc[assets, assets].to_numpy(dtype=float)
    )
    cov = np.outer(vols, vols) * corr
    # Ridge for numerical stability
    cov += COV_RIDGE * np.eye(len(assets))
    return pd.DataFrame(cov, index=assets, columns=assets)


def compute_vol_contributions(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """Compute per-asset marginal volatility contributions (unit-normed).

    Contrib_k = w_k * (Sigma_hat w)_k / (w^T Sigma_hat w).
    By construction, contributions sum to exactly 1.0.

    Arguments:
        weights: Unit-normed portfolio weights (sum must equal 1.0, within 1e-6).
        cov_matrix: N x N covariance matrix aligned to weights index.

    Returns:
        Series of volatility contributions, same index as weights.

    Raises:
        ValueError: If weights do not sum to approximately 1.0.
        ValueError: If portfolio variance is non-positive.
    """
    if abs(weights.sum() - 1.0) > 1e-6:
        raise ValueError(
            f"weights must sum to 1.0; got {weights.sum():.8f}. "
            "Normalize before calling."
        )

    assets = weights.index.tolist()
    w = weights[assets].values.astype(float)
    sigma = cov_matrix.loc[assets, assets].values.astype(float)

    sigma_w = sigma @ w
    port_var = float(w @ sigma_w)

    if port_var <= 0.0:
        raise ValueError(f"Portfolio variance is non-positive: {port_var}")

    contrib = w * sigma_w / port_var
    return pd.Series(contrib, index=assets)


def compute_realized_vol(
    returns: pd.Series,
    window_days: int = 90,
) -> pd.Series:
    """Compute rolling realized (historical) annualized volatility.

    Arguments:
        returns: Daily simple return Series (DatetimeIndex).
        window_days: Rolling window in trading days. Defaults to 90.

    Returns:
        Series of annualized realized vol (NaN for first window_days - 1 observations).
    """
    result: pd.Series = returns.rolling(window=window_days).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return result


def build_volatility_model(
    return_data: ReturnData,
    as_of_date: pd.Timestamp | None = None,
) -> VolatilityModel:
    """Build a VolatilityModel snapshot at a given date.

    Slices return history up to as_of_date, computes EWMA vols per asset,
    rolling weekly correlations, and the forward covariance matrix.

    Arguments:
        return_data: ReturnData from returns.build_return_data().
        as_of_date: Snapshot date (inclusive). Defaults to the last available date.

    Returns:
        VolatilityModel with ewma_vols, rolling_corr, cov_matrix, and lambda_.

    Raises:
        ValueError: If as_of_date is before the first available return date.
    """
    returns = return_data.returns
    if as_of_date is not None:
        if as_of_date < returns.index[0]:
            raise ValueError(
                f"as_of_date {as_of_date} is before the first return date "
                f"{returns.index[0]}"
            )
        returns = returns.loc[:as_of_date]

    ewma_vols = pd.Series(
        {col: float(compute_ewma_vol(returns[col]).iloc[-1]) for col in returns.columns},
        name="ewma_vol",
    )
    rolling_corr = compute_rolling_weekly_corr(returns)
    cov_matrix = build_covariance_matrix(ewma_vols, rolling_corr)

    return VolatilityModel(
        ewma_vols=ewma_vols,
        rolling_corr=rolling_corr,
        cov_matrix=cov_matrix,
        lambda_=EWMA_LAMBDA,
    )


def build_vol_contribution_table(
    weights: pd.Series,
    return_data: ReturnData,
    vol_model: VolatilityModel,
) -> pd.DataFrame:
    """Assemble the full volatility contribution table.

    Columns: Asset, sigma_tilde_k (90-day realized), sigma_hat_k (EWMA),
    rho_VTI_k (rolling correlation with VTI), Contrib_k (sums to 1).

    Arguments:
        weights: Unit-normed portfolio weights (sum = 1.0).
        return_data: ReturnData providing daily returns for realized vol.
        vol_model: VolatilityModel providing EWMA vols, correlations, covariance.

    Returns:
        DataFrame with columns [sigma_tilde, sigma_hat, rho_VTI, contrib],
        indexed by asset name.

    Notes:
        rho_VTI is set to NaN for assets not present in rolling_corr.
        If VTI is not in the universe, the rho_VTI column is all NaN.
    """
    assets = weights.index.tolist()
    returns = return_data.returns

    realized_vol_series = {
        a: compute_realized_vol(returns[a]).iloc[-1] for a in assets
    }
    contributions = compute_vol_contributions(weights, vol_model.cov_matrix)

    rho_vti: dict[str, float] = {}
    if "VTI" in vol_model.rolling_corr.columns:
        for a in assets:
            if a in vol_model.rolling_corr.index:
                rho_vti[a] = float(vol_model.rolling_corr.loc[a, "VTI"])
            else:
                rho_vti[a] = float("nan")
    else:
        rho_vti = {a: float("nan") for a in assets}

    rows = {
        "sigma_tilde": pd.Series(realized_vol_series),
        "sigma_hat": vol_model.ewma_vols[assets],
        "rho_VTI": pd.Series(rho_vti),
        "contrib": contributions,
    }
    return pd.DataFrame(rows, index=assets)


def forecast_portfolio_vol(
    weights: pd.Series,
    vol_model: VolatilityModel,
) -> float:
    """Forecast one-step-ahead annualized portfolio volatility using Sigma_hat.

    Arguments:
        weights: Unit-normed portfolio weights (sum = 1.0).
        vol_model: VolatilityModel containing cov_matrix.

    Returns:
        Annualized portfolio volatility sigma_hat_p = sqrt(w^T Sigma_hat w).
    """
    assets = weights.index.tolist()
    w = weights[assets].values.astype(float)
    sigma = vol_model.cov_matrix.loc[assets, assets].values.astype(float)
    port_var = float(w @ sigma @ w)
    return float(np.sqrt(max(port_var, 0.0)))
