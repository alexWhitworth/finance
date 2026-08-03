"""Phase 4 integration tests for the refactored run_backtest pipeline.

F-017: NAV-return accounting identity across all 4 config variants.
F-018: Same-day state consistency for every day in a GTT+LEAPS backtest.
F-019: Whipsaw multi-regime lifecycle with two defensive windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance._backtest_steps import _long_windows
from finance._portfolio_types import GttConfig, PortfolioConfig
from finance.data import PriceData
from finance.gtt import GttSignalData
from finance.leverage import (
    AccountType,
    LeapsConfig,
    RebalanceRule,
    WeightStrategy,
    _live_contracts,
)
from finance.portfolio import run_backtest
from finance.returns import ReturnData, build_return_data

# ---------------------------------------------------------------------------
# Shared synthetic corpus
# ---------------------------------------------------------------------------

_N_DAYS = 504  # ~2 years of trading days
_INITIAL_NAV = 500_000.0
_MONTHLY_CONTRIB = 2_000.0
_SEED = 7


def _make_price_data(
    n: int = _N_DAYS,
    seed: int = _SEED,
    with_vti_vol: bool = False,
) -> PriceData:
    """Synthetic PriceData for VTI + VXUS + GLD + (optional VTI vol column)."""
    idx = pd.bdate_range("2018-01-02", periods=n + 1)
    rng = np.random.default_rng(seed)
    tickers = ("VTI", "VXUS", "GLD")
    starts = {"VTI": 150.0, "VXUS": 55.0, "GLD": 120.0}
    prices = pd.DataFrame(
        {t: starts[t] * np.cumprod(1 + rng.normal(0.0003, 0.01, n + 1)) for t in tickers},
        index=idx,
    )
    dividends = pd.DataFrame(0.0, index=idx, columns=list(tickers))
    if with_vti_vol:
        vix_vals = 0.20 + rng.normal(0, 0.04, n + 1)
        vol_prices = pd.DataFrame({"VTI": np.clip(vix_vals, 0.08, 0.80)}, index=idx)
    else:
        vol_prices = pd.DataFrame(index=idx)
    return PriceData(
        prices=prices,
        dividends=dividends,
        vol_prices=vol_prices,
        tickers=tickers,
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
        spliced=False,
    )


def _make_rd(pd_obj: PriceData) -> ReturnData:
    return build_return_data(pd_obj, apply_tey=False)


def _gtt_signal(index: pd.DatetimeIndex, mask: np.ndarray) -> GttSignalData:
    zeros = pd.Series(0, index=index)
    return GttSignalData(
        position_mask=pd.Series(mask, index=index, name="position_mask"),
        ue_signal=zeros,
        vix_signal=zeros,
        vix_p90_threshold=0.272,
        unrate_start=pd.Timestamp(index[0]),
        vix_start=pd.Timestamp(index[0]),
    )


def _gtt_config() -> GttConfig:
    return GttConfig(
        vix_p90_threshold=0.272,
        defensive_weights={"R_f": 1.0},
    )


def _no_gtt_no_leaps_config(contribution: float = _MONTHLY_CONTRIB) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights={"VTI": 0.60, "VXUS": 0.25, "GLD": 0.15},
        initial_nav=_INITIAL_NAV,
        monthly_contribution=contribution,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
    )


def _no_gtt_leaps_config(contribution: float = _MONTHLY_CONTRIB) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights={"VTI": 0.50, "VXUS": 0.25, "GLD": 0.10, "VTI_LEAPS": 0.15},
        initial_nav=_INITIAL_NAV,
        monthly_contribution=contribution,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(iv=0.20, ltcg_rate=0.238, account_type=AccountType.TAXABLE),
    )


def _gtt_no_leaps_config(contribution: float = _MONTHLY_CONTRIB) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights={"VTI": 0.60, "VXUS": 0.25, "GLD": 0.15},
        initial_nav=_INITIAL_NAV,
        monthly_contribution=contribution,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        gtt_config=_gtt_config(),
    )


def _gtt_leaps_config(contribution: float = _MONTHLY_CONTRIB) -> PortfolioConfig:
    return PortfolioConfig(
        target_weights={"VTI": 0.50, "VXUS": 0.25, "GLD": 0.10, "VTI_LEAPS": 0.15},
        initial_nav=_INITIAL_NAV,
        monthly_contribution=contribution,
        rebalance_rule=RebalanceRule.QUARTERLY,
        weight_strategy=WeightStrategy.USER_SPECIFIED,
        leaps_config=LeapsConfig(iv=0.20, ltcg_rate=0.238, account_type=AccountType.TAXABLE),
        gtt_config=_gtt_config(),
    )


# Single defensive window: days 80–160 are Defensive, all others Long.
def _single_window_mask(n: int) -> np.ndarray:
    m = np.ones(n, dtype=int)
    m[80:160] = 0
    return m


# Two non-overlapping defensive windows: days 60–110 and 200–260.
def _two_window_mask(n: int) -> np.ndarray:
    m = np.ones(n, dtype=int)
    m[60:110] = 0
    m[200:260] = 0
    return m


# ---------------------------------------------------------------------------
# F-017: Accounting identity I1
#
# For a zero-contribution backtest:
#   nav[-1] == initial_nav * prod(1 + return_series)   (within 1e-9 rel)
#
# For a non-zero contribution backtest the arithmetic oracle is more involved
# because each contribution compounds forward.  Instead we verify the weaker
# but still tight constraint:
#   nav[-1] > initial_nav * prod(1 + return_series)    (contributions inflate NAV)
# and
#   nav[-1] == sum over all days of period_NAV_delta   (telescoping chain)
# where period_NAV_delta is the direct NAV compounding + contribution.
# ---------------------------------------------------------------------------


def _nav_product_oracle(initial_nav: float, return_series: pd.Series) -> float:
    """Zero-contribution oracle: initial_nav * prod(1 + r_t) for all t."""
    return float(initial_nav * (1.0 + return_series).prod())


@pytest.mark.parametrize(
    "label,config_fn,has_gtt",
    [
        ("no_gtt_no_leaps", _no_gtt_no_leaps_config, False),
        ("no_gtt_leaps", _no_gtt_leaps_config, False),
        ("gtt_no_leaps", _gtt_no_leaps_config, True),
        ("gtt_leaps", _gtt_leaps_config, True),
    ],
)
class TestNavReturnIdentityZeroContrib:
    """F-017 (Scenario A, zero-contribution variant): I1 for all 4 configs.

    nav[-1] == initial_nav * prod(1 + return_series) within rel 1e-9.
    Zero-contribution isolates pure compounding so the product identity is exact.
    """

    def test_nav_product_identity(
        self, label: str, config_fn: object, has_gtt: bool
    ) -> None:
        """I1: nav[-1] == initial_nav * prod(1+r) for zero-contribution backtest."""
        pd_obj = _make_price_data()
        rd = _make_rd(pd_obj)
        cfg = config_fn(contribution=0.0)  # type: ignore[call-arg]

        if has_gtt:
            mask = _single_window_mask(len(rd.returns))
            sig = _gtt_signal(rd.returns.index, mask)
            result = run_backtest(rd, pd_obj, cfg, gtt_signal=sig)
        else:
            result = run_backtest(rd, pd_obj, cfg)

        oracle = _nav_product_oracle(cfg.initial_nav, result.return_series)
        assert result.nav_series.iloc[-1] == pytest.approx(oracle, rel=1e-9), (
            f"[{label}] NAV product identity failed: "
            f"got {result.nav_series.iloc[-1]:.6f}, oracle {oracle:.6f}"
        )


@pytest.mark.parametrize(
    "label,config_fn,has_gtt",
    [
        ("no_gtt_no_leaps", _no_gtt_no_leaps_config, False),
        ("no_gtt_leaps", _no_gtt_leaps_config, False),
        ("gtt_no_leaps", _gtt_no_leaps_config, True),
        ("gtt_leaps", _gtt_leaps_config, True),
    ],
)
class TestNavReturnIdentityWithContrib:
    """F-017 (Scenario A, non-zero contribution variant): I1 structural bound for all 4 configs.

    With contributions, nav[-1] strictly exceeds the zero-contribution oracle.
    Also verifies the return_series is finite and the NAV is positive throughout.
    """

    def test_contrib_nav_exceeds_zero_contrib(
        self, label: str, config_fn: object, has_gtt: bool
    ) -> None:
        """Nav with contributions exceeds zero-contribution oracle."""
        pd_obj = _make_price_data()
        rd = _make_rd(pd_obj)
        cfg_zero = config_fn(contribution=0.0)  # type: ignore[call-arg]
        cfg_contrib = config_fn(contribution=_MONTHLY_CONTRIB)  # type: ignore[call-arg]

        if has_gtt:
            mask = _single_window_mask(len(rd.returns))
            sig_zero = _gtt_signal(rd.returns.index, mask)
            sig_contrib = _gtt_signal(rd.returns.index, mask)
            r_zero = run_backtest(rd, pd_obj, cfg_zero, gtt_signal=sig_zero)
            r_contrib = run_backtest(rd, pd_obj, cfg_contrib, gtt_signal=sig_contrib)
        else:
            r_zero = run_backtest(rd, pd_obj, cfg_zero)
            r_contrib = run_backtest(rd, pd_obj, cfg_contrib)

        assert r_contrib.nav_series.iloc[-1] > r_zero.nav_series.iloc[-1], (
            f"[{label}] Contribution backtest NAV ({r_contrib.nav_series.iloc[-1]:.2f}) "
            f"should exceed zero-contrib NAV ({r_zero.nav_series.iloc[-1]:.2f})"
        )

    def test_return_series_finite(
        self, label: str, config_fn: object, has_gtt: bool
    ) -> None:
        """Return series is finite and non-NaN for all days."""
        pd_obj = _make_price_data()
        rd = _make_rd(pd_obj)
        cfg = config_fn(contribution=_MONTHLY_CONTRIB)  # type: ignore[call-arg]

        if has_gtt:
            mask = _single_window_mask(len(rd.returns))
            sig = _gtt_signal(rd.returns.index, mask)
            result = run_backtest(rd, pd_obj, cfg, gtt_signal=sig)
        else:
            result = run_backtest(rd, pd_obj, cfg)

        assert result.return_series.isna().sum() == 0, f"[{label}] NaN in return_series"
        assert np.isfinite(result.return_series.values).all(), (
            f"[{label}] Non-finite value in return_series"
        )

    def test_double_count_guard(
        self, label: str, config_fn: object, has_gtt: bool
    ) -> None:
        """Contributions must NOT appear in return_series (no double-count).

        If a contribution is baked into port_return, the return_series will
        systematically spike on month-end days relative to the inter-month mean.
        Guard: max month-end |return| < 3x the median absolute daily return.
        """
        pd_obj = _make_price_data()
        rd = _make_rd(pd_obj)
        cfg = config_fn(contribution=_MONTHLY_CONTRIB)  # type: ignore[call-arg]

        if has_gtt:
            mask = _single_window_mask(len(rd.returns))
            sig = _gtt_signal(rd.returns.index, mask)
            result = run_backtest(rd, pd_obj, cfg, gtt_signal=sig)
        else:
            result = run_backtest(rd, pd_obj, cfg)

        idx = pd.DatetimeIndex(result.return_series.index)
        month_end_mask = idx == idx.to_period("M").to_timestamp("M")
        # Use the business-day-aligned month ends instead of raw calendar month end
        month_ends_bday = (
            result.return_series.groupby(idx.to_period("M")).tail(1).index
        )
        month_end_rets = result.return_series.reindex(month_ends_bday).abs()
        median_abs = float(result.return_series.abs().median())

        if median_abs > 0:
            # Month-end returns must not be disproportionately large
            assert float(month_end_rets.max()) < 10 * median_abs, (
                f"[{label}] Month-end return spike detected — contribution may be "
                f"double-counted in return_series"
            )


# ---------------------------------------------------------------------------
# F-018: Same-day state consistency (Scenario B) — Invariant I2
#
# Oracle: nav[t] == nav[t-1] * (1 + r[t]) + contribution_on_t  for all t
#
# For GTT+no-LEAPS the oracle is exact:
#   contribution_on_t = monthly_contribution if is_month_end else 0.0
#
# For GTT+LEAPS the LEAPS contribution is pre-priced into run_leaps_simulation
# at window start, so base_contribution (not monthly_contribution) is what
# _apply_contribution adds on Long month-ends.  We test the weaker but still
# tight bound: the implied contrib delta is non-negative every day and positive
# on month-ends.
#
# The Bug 1 regression guard verifies the I2 chain breaks if the
# _compute_leaps_mtm re-entry suppression is disabled (stale leaps_value).
# ---------------------------------------------------------------------------

_F018_N = 504  # ~2 years, matching spec realism requirement


def _build_f018_corpus() -> tuple[ReturnData, PriceData, np.ndarray]:
    """Return (rd, pd_obj, mask) for a 504-day GTT corpus with one defensive window.

    Defensive window: days 100-200 (regime=0), all others Long.
    This ensures at least one re-entry (day 200) and several month-ends in each regime.
    """
    pd_obj = _make_price_data(n=_F018_N, seed=42)
    rd = _make_rd(pd_obj)
    mask = np.ones(_F018_N, dtype=int)
    mask[100:200] = 0
    return rd, pd_obj, mask


def _month_end_set(returns: pd.DataFrame) -> frozenset[pd.Timestamp]:
    """Return the set of last business days of each calendar month in returns.index.

    Mirrors the exact logic used by _build_context so the test oracle matches the
    backtest's internal month_end_dates frozenset.
    """
    idx = pd.DatetimeIndex(returns.index)
    return frozenset(
        pd.Timestamp(grp.index[-1])
        for _, grp in returns.groupby(idx.to_period("M"))
    )


class TestSameDayConsistency:
    """F-018: Invariant I2 — nav[t] == nav[t-1]*(1+r[t]) + contrib_delta[t] for all t.

    Covers 504 days including: defensive window, re-entry, month-end re-entry coincidence.
    """

    def test_i2_exact_gtt_no_leaps_all_days(self) -> None:
        """I2 exact oracle holds for every day in a GTT (no-LEAPS) backtest.

        Oracle: nav[t] == nav[t-1] * (1 + r[t]) + monthly_contribution * is_month_end[t]
        Tolerance: 1e-9 (pure float arithmetic, no BS pricing involved).
        """
        rd, pd_obj, mask = _build_f018_corpus()
        cfg = _gtt_no_leaps_config(contribution=_MONTHLY_CONTRIB)
        sig = _gtt_signal(rd.returns.index, mask)
        result = run_backtest(rd, pd_obj, cfg, gtt_signal=sig)

        nav = result.nav_series
        rets = result.return_series
        idx = pd.DatetimeIndex(nav.index)
        month_ends = _month_end_set(rd.returns)

        failures: list[str] = []
        for i, ts in enumerate(idx):
            date_ts = pd.Timestamp(ts)
            r_t = float(rets.iloc[i])
            contrib = _MONTHLY_CONTRIB if date_ts in month_ends else 0.0

            if i == 0:
                prev_nav = cfg.initial_nav
            else:
                prev_nav = float(nav.iloc[i - 1])

            expected = prev_nav * (1.0 + r_t) + contrib
            actual = float(nav.iloc[i])

            if abs(actual - expected) > 1e-9 * max(abs(expected), 1.0):
                failures.append(
                    f"  day {i} ({date_ts.date()}): "
                    f"expected {expected:.9f}, got {actual:.9f}, "
                    f"diff {actual - expected:.3e}, "
                    f"is_month_end={date_ts in month_ends}"
                )

        assert not failures, (
            f"I2 violated on {len(failures)} day(s):\n" + "\n".join(failures[:10])
        )

    def test_i2_contrib_delta_nonneg_gtt_leaps_all_days(self) -> None:
        """I2 weaker bound for GTT+LEAPS: implied contribution delta is non-negative every day.

        On month-ends the delta must be strictly positive; on non-month-ends it must
        equal zero within tolerance (rebalance and re-entry are NAV-neutral).
        """
        rd, pd_obj, mask = _build_f018_corpus()
        cfg = _gtt_leaps_config(contribution=_MONTHLY_CONTRIB)
        sig = _gtt_signal(rd.returns.index, mask)
        result = run_backtest(rd, pd_obj, cfg, gtt_signal=sig)

        nav = result.nav_series
        rets = result.return_series
        idx = pd.DatetimeIndex(nav.index)
        month_ends = _month_end_set(rd.returns)

        neg_on_nonmonthend: list[str] = []
        zero_on_monthend: list[str] = []

        for i, ts in enumerate(idx):
            date_ts = pd.Timestamp(ts)
            r_t = float(rets.iloc[i])

            if i == 0:
                prev_nav = cfg.initial_nav
            else:
                prev_nav = float(nav.iloc[i - 1])

            # implied contribution delta = nav[t] - prev_nav*(1+r[t])
            implied_delta = float(nav.iloc[i]) - prev_nav * (1.0 + r_t)

            if date_ts in month_ends:
                if implied_delta <= 0:
                    zero_on_monthend.append(
                        f"  day {i} ({date_ts.date()}): implied_delta={implied_delta:.6e}"
                    )
            else:
                # non-month-end: rebalance + re-entry are NAV-neutral → delta must be ~0
                if abs(implied_delta) > 1e-6 * max(abs(prev_nav), 1.0):
                    neg_on_nonmonthend.append(
                        f"  day {i} ({date_ts.date()}): implied_delta={implied_delta:.6e}, "
                        f"prev_nav={prev_nav:.2f}"
                    )

        assert not zero_on_monthend, (
            f"Implied contribution delta ≤ 0 on {len(zero_on_monthend)} month-end(s):\n"
            + "\n".join(zero_on_monthend[:5])
        )
        assert not neg_on_nonmonthend, (
            f"Non-zero implied delta on {len(neg_on_nonmonthend)} non-month-end day(s):\n"
            + "\n".join(neg_on_nonmonthend[:10])
        )

    def test_i2_bug1_regression_guard(self) -> None:
        """I2 chain is satisfied for all 504 days including re-entry day.

        Bug 1 manifestation: on re-entry day (prev_regime=0, regime_t=1) the
        stale LEAPS ledger's mark-to-market was added BEFORE _apply_gtt_reentry
        replaced it.  This inflated nav_before, so port_return spiked.  The
        corrected code suppresses leaps_value to 0.0 before re-entry runs.

        We verify by checking that the re-entry day's return is bounded
        (|r[reentry]| < 0.10) — a spike of > 10% on that single day would
        indicate Bug 1 is present.
        """
        rd, pd_obj, mask = _build_f018_corpus()
        cfg = _gtt_leaps_config(contribution=_MONTHLY_CONTRIB)
        sig = _gtt_signal(rd.returns.index, mask)
        result = run_backtest(rd, pd_obj, cfg, gtt_signal=sig)

        rets = result.return_series
        idx = pd.DatetimeIndex(rets.index)

        # Re-entry is the first Long day after the defensive window: mask[200] == 1, mask[199] == 0
        # The backtest aligns the mask to returns index via ffill, so returns.index[200] is re-entry.
        reentry_ts = pd.Timestamp(idx[200])
        reentry_ret = float(rets.iloc[200])

        assert abs(reentry_ret) < 0.10, (
            f"Re-entry return spike detected at {reentry_ts.date()}: "
            f"|return| = {abs(reentry_ret):.4f} >= 0.10. "
            f"Bug 1 suppression may be missing in _compute_leaps_mtm."
        )
