"""Visualization layer for portfolio backtest results.

All functions return a plotnine ``ggplot`` object and optionally save to
``figures/<filename>.png``.  Each function is pure modulo filesystem I/O —
pass ``output_path=None`` to suppress saving.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import asdict
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
from matplotlib.figure import Figure

from finance.consts import CRISIS_PERIODS, NBER_RECESSION_PERIODS
from finance.dca_signal import LeapsDcaSignal
from finance.greeks import PortfolioGreeks
from finance.metrics import PerformanceMetrics, PerformanceReport
from finance.portfolio import BacktestResult
from finance.portfolio_manager import HoldingView, NavBreakdown, TradeOrder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIGURES_DIR = Path("figures")


def _save(plot: p9.ggplot, path: Path) -> None:
    """Save *plot* to *path*, creating parent directories as needed.

    Arguments:
        plot: The plotnine ggplot object to save.
        path: Destination file path (PNG).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plot.save(str(path), verbose=False)


def _nber_rects(
    date_min: pd.Timestamp, date_max: pd.Timestamp, y_lo: float, y_hi: float
) -> pd.DataFrame | None:
    """Build a DataFrame of NBER recession rectangles clipped to [date_min, date_max].

    Arguments:
        date_min: Left clip boundary (inclusive).
        date_max: Right clip boundary (inclusive).
        y_lo: Bottom of each rectangle.
        y_hi: Top of each rectangle.

    Returns:
        DataFrame with columns [xmin, xmax, ymin, ymax], or None if no overlap.
    """
    rows = []
    for s, e in NBER_RECESSION_PERIODS:
        xmin = max(pd.Timestamp(s), date_min)
        xmax = min(pd.Timestamp(e), date_max)
        if xmin < xmax:
            rows.append({"xmin": xmin, "xmax": xmax, "ymin": y_lo, "ymax": y_hi})
    return pd.DataFrame(rows) if rows else None


def _gtt_defensive_rects(position_mask: pd.Series, y_lo: float, y_hi: float) -> pd.DataFrame | None:
    """Build a DataFrame of GTT-defensive (position_mask == 0) rectangles.

    Arguments:
        position_mask: Daily 0/1 Series; 0 = defensive.
        y_lo: Bottom of each rectangle.
        y_hi: Top of each rectangle.

    Returns:
        DataFrame with columns [xmin, xmax, ymin, ymax], or None if mask never fires.
    """
    is_def = position_mask == 0
    starts = position_mask.index[is_def & ~is_def.shift(1, fill_value=False)]
    ends = position_mask.index[is_def & ~is_def.shift(-1, fill_value=False)]
    n = min(len(starts), len(ends))
    if n == 0:
        return None
    return pd.DataFrame({
        "xmin": starts[:n],
        "xmax": ends[:n],
        "ymin": y_lo,
        "ymax": y_hi,
    })


def _nav_log_breaks(y_lo: float, y_hi: float) -> list[float]:
    """Generate round-number ($M) tick positions spanning [y_lo, y_hi] on a log scale.

    Uses a 1-2-5-per-decade sequence (e.g. 1, 2, 5, 10, 20, 50) so ticks land on
    clean dollar amounts anchored to the actual NAV range, instead of plotnine's
    default log-scale ticks which rarely align with the data.

    Arguments:
        y_lo: Lower bound of the NAV range, in millions.
        y_hi: Upper bound of the NAV range, in millions.

    Returns:
        Sorted list of tick positions (in millions) covering the range.
    """
    if y_lo <= 0.0 or y_hi <= y_lo:
        return [y_hi]
    lo_exp = math.floor(math.log10(y_lo))
    hi_exp = math.ceil(math.log10(y_hi))
    candidates = sorted(m * 10.0**e for e in range(lo_exp, hi_exp + 1) for m in (1, 2, 5))
    breaks = [b for b in candidates if y_lo * 0.9 <= b <= y_hi * 1.1]
    return breaks if breaks else [y_lo, y_hi]


def _format_nav_millions(breaks: list[float]) -> list[str]:
    """Format $-millions tick positions as human-readable dollar labels.

    Arguments:
        breaks: Tick positions in millions of dollars.

    Returns:
        Labels like "$500K", "$1M", "$2.5M" matching each break.
    """
    labels = []
    for b in breaks:
        if b < 1.0:
            labels.append(f"${b * 1000:,.0f}K")
        elif b >= 1000.0:
            billions = b / 1000.0
            fmt = f"${int(billions):,}B" if billions == int(billions) else f"${billions:,.1f}B"
            labels.append(fmt)
        elif b == int(b):
            labels.append(f"${int(b):,}M")
        else:
            labels.append(f"${b:,.1f}M")
    return labels


def _compute_drawdown_series(nav: pd.Series) -> pd.Series:
    """Return the drawdown series for *nav* (negative fractions).

    Arguments:
        nav: NAV time series.

    Returns:
        Series of drawdown fractions ≤ 0.
    """
    peak = nav.cummax()
    return (nav - peak) / peak


# ---------------------------------------------------------------------------
# 1. NAV growth comparison
# ---------------------------------------------------------------------------


def plot_nav_growth(
    results: dict[str, BacktestResult],
    output_path: Path | None = _FIGURES_DIR / "nav_growth.png",
    position_mask: pd.Series | None = None,
) -> p9.ggplot:
    """Line chart comparing NAV growth across multiple portfolio configurations.

    Grey bands mark NBER recession periods. When *position_mask* is supplied,
    yellow bands mark GTT-defensive periods (position_mask == 0).

    Arguments:
        results: Mapping of label → BacktestResult.
        output_path: Destination PNG path, or None to skip saving.
        position_mask: Optional daily 0/1 Series from GttSignalData; 0 = defensive.
            When provided, defensive periods are shaded yellow.

    Returns:
        A plotnine ggplot object.
    """
    frames: list[pd.DataFrame] = []
    for label, result in results.items():
        df = result.nav_series.rename("nav").to_frame()
        df["portfolio"] = label
        df.index.name = "date"
        frames.append(df.reset_index())

    data = pd.concat(frames, ignore_index=True)
    data["nav_millions"] = data["nav"] / 1_000_000

    date_min = data["date"].min()
    date_max = data["date"].max()
    y_lo = float(data["nav_millions"].min() * 0.95)
    y_hi = float(data["nav_millions"].max() * 1.05)

    base = p9.ggplot()

    nber = _nber_rects(date_min, date_max, y_lo, y_hi)
    if nber is not None:
        base = base + p9.geom_rect(
            data=nber,
            mapping=p9.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
            fill="#d0d0d0", alpha=0.6, inherit_aes=False,
        )

    if position_mask is not None:
        gtt = _gtt_defensive_rects(position_mask, y_lo, y_hi)
        if gtt is not None:
            base = base + p9.geom_rect(
                data=gtt,
                mapping=p9.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                fill="#f5c242", alpha=0.45, inherit_aes=False,
            )

    subtitle = "Gray = NBER recession"
    if position_mask is not None:
        subtitle += "  |  Yellow = GTT defensive"

    y_breaks = _nav_log_breaks(y_lo, y_hi)
    y_labels = _format_nav_millions(y_breaks)

    plot = (
        base
        + p9.geom_line(
            data=data, mapping=p9.aes(x="date", y="nav_millions", color="portfolio"), size=0.8
        )
        + p9.scale_x_datetime(date_labels="%Y", date_minor_breaks="1 year")
        + p9.scale_y_log10(breaks=y_breaks, labels=y_labels, minor_breaks=[])
        + p9.labs(
            title=f"Portfolio NAV Growth\n{subtitle}",
            x="Date",
            y="NAV",
            color="Portfolio",
        )
        + p9.theme_grey()
        + p9.theme(figure_size=(10, 5), legend_position="bottom")
    )

    if output_path is not None:
        _save(plot, output_path)
    return plot


# ---------------------------------------------------------------------------
# 2. Drawdown chart with crisis period shading
# ---------------------------------------------------------------------------


def plot_drawdown(
    results: dict[str, BacktestResult],
    crisis_periods: dict[str, tuple[str, str]] = CRISIS_PERIODS,
    output_path: Path | None = _FIGURES_DIR / "drawdown.png",
) -> p9.ggplot:
    """Drawdown chart with shaded crisis period bands.

    Arguments:
        results: Mapping of label → BacktestResult.
        crisis_periods: Dict of crisis label → (start_date, end_date) strings.
        output_path: Destination PNG path, or None to skip saving.

    Returns:
        A plotnine ggplot object.
    """
    frames: list[pd.DataFrame] = []
    for label, result in results.items():
        dd = _compute_drawdown_series(result.nav_series)
        df = dd.rename("drawdown").to_frame()
        df["portfolio"] = label
        df.index.name = "date"
        frames.append(df.reset_index())

    data = pd.concat(frames, ignore_index=True)
    data["drawdown_pct"] = data["drawdown"] * 100

    # Build crisis period rectangles clipped to the data range
    date_min = data["date"].min()
    date_max = data["date"].max()

    crisis_frames: list[pd.DataFrame] = []
    for crisis_label, (start, end) in crisis_periods.items():
        xmin = max(pd.Timestamp(start), date_min)
        xmax = min(pd.Timestamp(end), date_max)
        if xmin < xmax:
            crisis_frames.append(
                pd.DataFrame({"xmin": [xmin], "xmax": [xmax], "crisis": [crisis_label]})
            )

    base = p9.ggplot(data, p9.aes(x="date", y="drawdown_pct"))

    if crisis_frames:
        crisis_data = pd.concat(crisis_frames, ignore_index=True)
        base = base + p9.geom_rect(
            data=crisis_data,
            mapping=p9.aes(xmin="xmin", xmax="xmax", fill="crisis"),
            ymin=-float("inf"),
            ymax=0,
            alpha=0.15,
            inherit_aes=False,
        )

    plot = (
        base
        + p9.geom_line(p9.aes(color="portfolio"), size=0.8)
        + p9.geom_hline(yintercept=0, linetype="dashed", color="grey", size=0.4)
        + p9.scale_x_datetime(date_labels="%Y", date_minor_breaks="1 year")
        + p9.labs(
            title="Portfolio Drawdown",
            x="Date",
            y="Drawdown (%)",
            color="Portfolio",
            fill="Crisis Period",
        )
        + p9.theme_grey()
        + p9.theme(figure_size=(10, 5), legend_position="bottom")
    )

    if output_path is not None:
        _save(plot, output_path)
    return plot


# ---------------------------------------------------------------------------
# 3. Volatility contribution bar chart
# ---------------------------------------------------------------------------


def plot_vol_contributions(
    report: PerformanceReport,
    output_path: Path | None = _FIGURES_DIR / "vol_contributions.png",
) -> p9.ggplot:
    """Horizontal stacked bar chart of per-asset volatility contributions.

    Arguments:
        report: PerformanceReport containing vol_contribution_table.
        output_path: Destination PNG path, or None to skip saving.

    Returns:
        A plotnine ggplot object.
    """
    tbl = report.vol_contribution_table.copy()
    tbl.index.name = "asset"
    tbl = tbl.reset_index()
    tbl = tbl.sort_values("contrib", ascending=False)
    tbl["contrib_pct"] = tbl["contrib"] * 100
    tbl["asset"] = pd.Categorical(tbl["asset"], categories=tbl["asset"].tolist())

    plot = (
        p9.ggplot(tbl, p9.aes(x="asset", y="contrib_pct", fill="asset"))
        + p9.geom_col(show_legend=False)
        + p9.geom_text(
            p9.aes(label="contrib_pct"),
            format_string="{:.1f}%",
            ha="left",
            size=9,
            nudge_y=0.5,
        )
        + p9.coord_flip()
        + p9.labs(
            title="Volatility Contribution by Asset",
            x="Asset",
            y="Contribution (%)",
        )
        + p9.theme_grey()
        + p9.theme(figure_size=(8, 5))
    )

    if output_path is not None:
        _save(plot, output_path)
    return plot


# ---------------------------------------------------------------------------
# 4. LEAPS tax drag comparison
# ---------------------------------------------------------------------------


def plot_leaps_tax_drag(
    taxable_result: BacktestResult,
    sheltered_result: BacktestResult,
    output_path: Path | None = _FIGURES_DIR / "leaps_tax_drag.png",
    position_mask: pd.Series | None = None,
) -> p9.ggplot:
    """Compare taxable vs. tax-sheltered LEAPS NAV trajectories.

    Grey bands mark NBER recession periods. When *position_mask* is supplied,
    yellow bands mark GTT-defensive periods (position_mask == 0).

    Arguments:
        taxable_result: BacktestResult from a TAXABLE account simulation.
        sheltered_result: BacktestResult from a TAX_SHELTERED account simulation.
        output_path: Destination PNG path, or None to skip saving.
        position_mask: Optional daily 0/1 Series from GttSignalData; 0 = defensive.
            When provided, defensive periods are shaded yellow.

    Returns:
        A plotnine ggplot object.
    """
    frames = [
        _result_to_df(taxable_result, "Taxable"),
        _result_to_df(sheltered_result, "Tax-Sheltered"),
    ]
    data = pd.concat(frames, ignore_index=True)
    data["nav_millions"] = data["nav"] / 1_000_000

    date_min = data["date"].min()
    date_max = data["date"].max()
    y_lo = float(data["nav_millions"].min() * 0.95)
    y_hi = float(data["nav_millions"].max() * 1.05)

    t_nav = taxable_result.nav_series
    s_nav = sheltered_result.nav_series
    common_idx = t_nav.index.intersection(s_nav.index)
    drag_label = (
        f"Tax drag: ${float(s_nav.loc[common_idx[-1]] - t_nav.loc[common_idx[-1]]):,.0f}"
        if len(common_idx) else ""
    )

    base = p9.ggplot()

    nber = _nber_rects(date_min, date_max, y_lo, y_hi)
    if nber is not None:
        base = base + p9.geom_rect(
            data=nber,
            mapping=p9.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
            fill="#d0d0d0", alpha=0.6, inherit_aes=False,
        )

    if position_mask is not None:
        gtt = _gtt_defensive_rects(position_mask, y_lo, y_hi)
        if gtt is not None:
            base = base + p9.geom_rect(
                data=gtt,
                mapping=p9.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                fill="#f5c242", alpha=0.45, inherit_aes=False,
            )

    subtitle_parts = ["Gray = NBER recession"]
    if position_mask is not None:
        subtitle_parts.append("Yellow = GTT defensive")
    subtitle = "  |  ".join(subtitle_parts)

    y_breaks = _nav_log_breaks(y_lo, y_hi)
    y_labels = _format_nav_millions(y_breaks)

    plot = (
        base
        + p9.geom_line(
            data=data, mapping=p9.aes(x="date", y="nav_millions", color="account"), size=0.8
        )
        + p9.scale_x_datetime(date_labels="%Y", date_minor_breaks="1 year")
        + p9.scale_y_log10(breaks=y_breaks, labels=y_labels, minor_breaks=[])
        + p9.labs(
            title=f"LEAPS Tax Drag: Taxable vs. Tax-Sheltered\n{drag_label}  {subtitle}",
            x="Date",
            y="NAV",
            color="Account Type",
        )
        + p9.theme_grey()
        + p9.theme(figure_size=(10, 5), legend_position="bottom")
    )

    if output_path is not None:
        _save(plot, output_path)
    return plot


def _result_to_df(result: BacktestResult, label: str) -> pd.DataFrame:
    """Convert a BacktestResult NAV series to a tidy DataFrame with an account label.

    Arguments:
        result: BacktestResult to convert.
        label: Account label string.

    Returns:
        DataFrame with columns [date, nav, account].
    """
    df = result.nav_series.rename("nav").to_frame()
    df["account"] = label
    df.index.name = "date"
    return df.reset_index()


# ---------------------------------------------------------------------------
# 5. Performance report table (console)
# ---------------------------------------------------------------------------


def format_performance_table(report: PerformanceReport) -> str:
    """Format a PerformanceReport as a human-readable table string.

    Arguments:
        report: PerformanceReport to display.

    Returns:
        A formatted string suitable for printing to stdout.
    """
    rows: list[dict[str, object]] = []
    for m in (report.full_period, *report.crisis_periods):
        rows.append(_metrics_to_row(m))

    df = pd.DataFrame(rows).set_index("Period")
    col_labels = [
        "Ann. Return", "Ann. Std", "Max DD",
        "Sharpe", "Sortino", "Calmar", "Omega",
        "Skewness", "Ex. Kurt",
    ]
    df.columns = pd.Index(col_labels)

    lines = [
        "=" * 100,
        "  Performance Report",
        "=" * 100,
        df.to_string(float_format=lambda x: f"{x:.4f}"),
        "-" * 100,
        f"  Forward Vol Forecast : {report.forward_vol_forecast:.4f}",
    ]

    if report.final_nav is not None:
        lines.append(f"  Terminal NAV         : ${report.final_nav:>15,.2f}")

    if report.terminal_nav is not None:
        tn = report.terminal_nav
        ts = report.tax_summary
        lines += [
            "-" * 100,
            "  LEAPS Terminal NAV",
            f"    Pre-tax  NAV : ${tn.pre_tax_nav:>15,.2f}",
            f"    Post-tax NAV : ${tn.post_tax_nav:>15,.2f}",
        ]
        if ts is not None:
            lines += [
                f"    Total Tax Drag  : {ts.tax_drag_pct:.4f}",
                f"    Ann. Tax Drag   : {ts.annualized_tax_drag:.4f}",
            ]

    lines.append("=" * 100)
    return os.linesep.join(lines)


def format_contract_greeks_table(greeks: PortfolioGreeks) -> str:
    """Format per-contract LEAPS greeks as a human-readable table string.

    Arguments:
        greeks: PortfolioGreeks with per-contract detail.

    Returns:
        A formatted string suitable for printing to stdout. A single line
        reading "No active LEAPS contracts." when greeks.contracts is empty.
    """
    if not greeks.contracts:
        return "No active LEAPS contracts."

    rows = [
        {
            "purchased": cg.contract.purchase_date.date(),
            "expiry": cg.contract.expiry_date.date(),
            "n_contracts": cg.contract.n_contracts,
            "delta": round(cg.delta, 3),
            "gamma": f"{cg.gamma:.4f}",
            "vega": f"{cg.vega:.4f}",
            "theta/day": round(cg.theta, 2),
            "position_delta": round(cg.position_delta, 1),
        }
        for cg in greeks.contracts
    ]
    return pd.DataFrame(rows).to_string(index=False)


def format_nav_breakdown_table(nav: NavBreakdown) -> str:
    """Format a NavBreakdown as a one-value-per-row table string.

    Arguments:
        nav: NavBreakdown to display.

    Returns:
        A formatted string suitable for printing to stdout.
    """
    rows = [
        ("base_nav", f"${nav.base_nav:,.2f}"),
        ("leaps_nav", f"${nav.leaps_nav:,.2f}"),
        ("defensive_sleeve", f"${nav.defensive_sleeve:,.2f}"),
        ("leaps_pool", f"${nav.leaps_pool:,.2f}"),
        ("total_nav", f"${nav.total_nav:,.2f}"),
    ]
    df = pd.DataFrame(rows, columns=["field", "value"]).set_index("field")
    return df.to_string(header=False, index_names=False)


def format_leaps_dca_signal_table(signal: LeapsDcaSignal) -> str:
    """Format a LeapsDcaSignal as a one-value-per-row table string.

    Arguments:
        signal: LeapsDcaSignal to display.

    Returns:
        A formatted string suitable for printing to stdout.
    """
    rows = [
        ("as_of_date", str(signal.as_of_date.date())),
        ("ticker", signal.ticker),
        ("entry_score", f"{signal.entry_score:.1f}"),
        ("score_percentile", f"{signal.score_percentile:.1f}"),
        ("alpha_t", f"{signal.alpha_t:.2f}"),
        ("dca_action", signal.dca_action),
        ("rsi", f"{signal.rsi:.1f}"),
        ("stoch_d", f"{signal.stoch_d:.1f}"),
        ("iv_current", f"{signal.iv_current:.1%}"),
        ("iv_percentile", f"{signal.iv_percentile:.1f}"),
        ("macd_hist", f"{signal.macd_hist:.3f}"),
        ("macd_bearish_confirmed", str(signal.macd_bearish_confirmed)),
        ("macd_gate", f"{signal.macd_gate:.2f}"),
    ]
    df = pd.DataFrame(rows, columns=["field", "value"]).set_index("field")
    return df.to_string(header=False, index_names=False)


def format_holdings_table(views: tuple[HoldingView, ...]) -> str:
    """Format a HoldingView tuple as a human-readable weight-drift table string.

    Arguments:
        views: Per-asset holding views, e.g. from compute_holdings_view() and
            compute_leaps_holdings_view() concatenated together.

    Returns:
        A formatted string suitable for printing to stdout.
    """
    df = pd.DataFrame([asdict(v) for v in views]).set_index("ticker")
    df["dollar_value"] = df["dollar_value"].map(lambda x: f"{x:,.2f}")
    df["actual_weight"] = df["actual_weight"].map(lambda x: f"{x:.1%}")
    df["target_weight"] = df["target_weight"].map(lambda x: f"{x:.1%}")
    df["weight_drift"] = df["weight_drift"].map(lambda x: f"{x:+.1%}")
    return df[["dollar_value", "actual_weight", "target_weight", "weight_drift"]].to_string()


def format_trade_orders_table(trades: tuple[TradeOrder, ...]) -> str:
    """Format a TradeOrder tuple as a human-readable trade table string.

    Arguments:
        trades: Trade instructions, e.g. from RebalancePlan.trades with
            leaps_trim_as_trade_order() appended.

    Returns:
        A formatted string suitable for printing to stdout, or an empty
        string when trades is empty.
    """
    if not trades:
        return ""
    df = pd.DataFrame([asdict(t) for t in trades]).set_index("ticker")
    df = df[["current_value", "target_value", "trade_amount"]]
    return df.map(lambda x: f"{x:,.2f}").to_string()


def compare_performance_table(reports: list[tuple[str, PerformanceReport]]) -> str:
    """Format multiple PerformanceReports side-by-side as a human-readable table string.

    Each report contributes one column per period (full period + crisis periods).
    Rows are metrics; columns are labelled "<name> | <period>".

    Arguments:
        reports: List of (label, PerformanceReport) pairs in display order.

    Returns:
        A formatted string suitable for printing to stdout.
    """
    col_order = [
        "Ann. Return", "Ann. Std", "Max DD",
        "Sharpe", "Sortino", "Calmar", "Omega",
        "Skewness", "Ex. Kurt",
    ]

    columns: dict[str, pd.Series] = {}
    for label, report in reports:
        for m in (report.full_period, *report.crisis_periods):
            col_label = f"{label} | {m.period_label}"
            row = _metrics_to_row(m)
            columns[col_label] = pd.Series(
                {k: row[k] for k in ("ann_return", "ann_std", "max_drawdown",
                                     "sharpe", "sortino", "calmar", "omega",
                                     "skewness", "excess_kurtosis")},
                index=["ann_return", "ann_std", "max_drawdown",
                       "sharpe", "sortino", "calmar", "omega",
                       "skewness", "excess_kurtosis"],
            )

    df = pd.DataFrame(columns).T
    df.columns = pd.Index(col_order)

    lines = [
        "=" * 100,
        "  Portfolio Comparison",
        "=" * 100,
        df.to_string(float_format=lambda x: f"{x:.4f}"),
        "-" * 100,
    ]
    for label, report in reports:
        lines.append(
            f"  [{label}] Forward Vol Forecast : {report.forward_vol_forecast:.4f}"
        )
        if report.final_nav is not None:
            lines.append(f"  [{label}] Terminal NAV : ${report.final_nav:>15,.2f}")
        if report.terminal_nav is not None:
            tn = report.terminal_nav
            ts = report.tax_summary
            lines.append(
                f"  [{label}] Pre-tax NAV: ${tn.pre_tax_nav:>15,.2f}  "
                f"Post-tax NAV: ${tn.post_tax_nav:>15,.2f}"
            )
            if ts is not None:
                lines.append(
                    f"  [{label}] Tax Drag: {ts.tax_drag_pct:.4f}  "
                    f"Ann. Tax Drag: {ts.annualized_tax_drag:.4f}"
                )

    lines.append("=" * 100)
    return os.linesep.join(lines)


def _metrics_to_row(m: PerformanceMetrics) -> dict[str, object]:
    """Convert a PerformanceMetrics dataclass to a dict row for a DataFrame.

    Arguments:
        m: PerformanceMetrics instance.

    Returns:
        Dict suitable for use as a DataFrame row.
    """
    return {
        "Period": m.period_label,
        "ann_return": m.annualized_return,
        "ann_std": m.annualized_std,
        "max_drawdown": m.max_drawdown,
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "calmar": m.calmar,
        "omega": m.omega,
        "skewness": m.skewness,
        "excess_kurtosis": m.excess_kurtosis,
    }


# ---------------------------------------------------------------------------
# 6. Pareto frontier dot plot
# ---------------------------------------------------------------------------

_METRIC_LABELS: dict[str, str] = {
    "annualized_return": "Ann. Return",
    "annualized_std": "Ann. Std",
    "max_drawdown": "Max Drawdown",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "calmar": "Calmar",
    "omega": "Omega",
    "skewness": "Skewness",
    "excess_kurtosis": "Ex. Kurtosis",
}

# Metrics where higher is better (used to orient the Pareto frontier).
_HIGHER_IS_BETTER: frozenset[str] = frozenset({
    "annualized_return", "sharpe", "sortino", "calmar", "omega", "skewness",
})


def _pareto_frontier(df: pd.DataFrame, x_metric: str, y_metric: str) -> pd.DataFrame:
    """Return the Pareto-dominant rows from *df* for the given metric pair.

    A point is Pareto-dominant when no other point is at least as good on both
    axes and strictly better on one.  "Better" is defined per
    ``_HIGHER_IS_BETTER``: higher is better for return/ratio metrics, lower is
    better for risk metrics (std, max_drawdown).

    Arguments:
        df: DataFrame with columns matching *x_metric* and *y_metric*.
        x_metric: PerformanceMetrics field name for the x-axis.
        y_metric: PerformanceMetrics field name for the y-axis.

    Returns:
        Subset of *df* containing only Pareto-dominant rows, sorted by
        x-axis value so the dashed frontier line connects correctly.
    """
    x = df[x_metric].to_numpy()
    y = df[y_metric].to_numpy()

    x_sign = 1.0 if x_metric in _HIGHER_IS_BETTER else -1.0
    y_sign = 1.0 if y_metric in _HIGHER_IS_BETTER else -1.0

    xs = x * x_sign
    ys = y * y_sign

    dominated = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            if xs[j] >= xs[i] and ys[j] >= ys[i] and (xs[j] > xs[i] or ys[j] > ys[i]):
                dominated[i] = True
                break

    frontier = df[~dominated].copy()
    return frontier.sort_values(x_metric)


def _plot_pareto(
    portfolios: dict[str, PerformanceMetrics],
    metrics: tuple[str, str],
) -> p9.ggplot:
    """Dot plot of portfolios on two metrics with a Pareto frontier overlay.

    Arguments:
        portfolios: Mapping of portfolio label → PerformanceMetrics.
        metrics: Two-tuple of PerformanceMetrics field names (x_metric, y_metric).

    Returns:
        A plotnine ggplot object.

    Raises:
        ValueError: If either metric name is not a field of PerformanceMetrics.
    """
    x_metric, y_metric = metrics
    valid = set(_METRIC_LABELS)
    for m in (x_metric, y_metric):
        if m not in valid:
            raise ValueError(f"Unknown metric {m!r}. Valid options: {sorted(valid)}")

    rows = [
        {"label": label, x_metric: getattr(pm, x_metric), y_metric: getattr(pm, y_metric)}
        for label, pm in portfolios.items()
    ]
    data = pd.DataFrame(rows)

    frontier = _pareto_frontier(data, x_metric, y_metric)

    x_lab = _METRIC_LABELS[x_metric]
    y_lab = _METRIC_LABELS[y_metric]

    plot = (
        p9.ggplot(data, p9.aes(x=x_metric, y=y_metric, color="label"))
        + p9.geom_point(size=3)
        + p9.geom_line(
            data=frontier,
            mapping=p9.aes(x=x_metric, y=y_metric),
            linetype="dashed",
            color="black",
            inherit_aes=False,
            size=0.6,
        )
        + p9.labs(x=x_lab, y=y_lab, color="Portfolio")
        + p9.theme_grey()
        + p9.theme(legend_position="bottom")
    )
    return plot


_DEFAULT_PARETO_METRICS: list[str] = list(_METRIC_LABELS.keys())


def plot_pareto(
    portfolios: dict[str, PerformanceMetrics],
    metrics: list[str] | None = None,
    output_path: Path | None = _FIGURES_DIR / "pareto_grid.png",
) -> Figure:
    """Lower-triangular grid of pairwise Pareto frontier plots.

    The grid has one row and one column per metric.  Cell (row, col) — where
    row > col — shows the ``_plot_pareto()`` panel for
    (col_metric, row_metric).  The diagonal and upper triangle are left blank.
    Shared metric labels run along the bottom edge (x-axis) and left edge
    (y-axis).  A single legend sits below the grid.

    Arguments:
        portfolios: Mapping of portfolio label → PerformanceMetrics.
        metrics: List of PerformanceMetrics field names to compare pairwise.
            Defaults to all metrics in ``_METRIC_LABELS``.
        output_path: Destination PNG path, or None to skip saving.

    Returns:
        A matplotlib Figure containing the assembled panel grid.

    Raises:
        ValueError: If fewer than two metrics are supplied.
    """
    if metrics is None:
        metrics = _DEFAULT_PARETO_METRICS
    if len(metrics) < 2:
        raise ValueError("At least two metrics are required for a pairwise grid.")

    n = len(metrics)
    # Layout constants in inches.  n-1 active columns/rows (lower triangle).
    cell = 3.0   # each subplot cell
    ml = 1.4     # left margin (rotated row labels)
    mb = 1.0     # bottom margin (col labels)
    mt = 0.6     # top margin (title)
    mr = 0.2     # right margin
    lh = 1.2     # legend strip height at bottom
    fs = 18      # shared label / legend font size

    fig_w = ml + cell * (n - 1) + mr
    fig_h = mt + cell * (n - 1) + mb + lh
    fig = plt.figure(figsize=(fig_w, fig_h))

    legend_handles = _build_legend_handles(portfolios, fs)

    for row in range(1, n):
        for col in range(row):
            ax = fig.add_axes([
                (ml + col * cell) / fig_w,
                (lh + mb + (n - 1 - row) * cell) / fig_h,
                cell / fig_w,
                cell / fig_h,
            ])
            panel = _plot_pareto(portfolios, (metrics[col], metrics[row]))
            panel = panel + p9.theme(
                legend_position="none",
                axis_title_x=p9.element_blank(),
                axis_title_y=p9.element_blank(),
                axis_text=p9.element_text(size=18),
                figure_size=(cell, cell),
            )
            buf = io.BytesIO()
            panel.save(buf, format="png", verbose=False, dpi=130)
            buf.seek(0)
            ax.imshow(mpimg.imread(buf), aspect="auto")
            ax.set_axis_off()

    # Row labels - rotated, vertically centred on each cell row
    for row in range(1, n):
        y_mid = (lh + mb + (n - 1 - row) * cell + cell / 2) / fig_h
        fig.text(
            ml * 0.35 / fig_w, y_mid,
            _METRIC_LABELS[metrics[row]],
            va="center", ha="center", fontsize=fs, rotation=90,
        )

    # Column labels - horizontally centred beneath each cell column
    for col in range(n - 1):
        x_mid = (ml + col * cell + cell / 2) / fig_w
        fig.text(
            x_mid, (lh + mb * 0.35) / fig_h,
            _METRIC_LABELS[metrics[col]],
            va="center", ha="center", fontsize=fs,
        )

    # Place legend in the empty upper-right corner (row 0, col n-3 onwards).
    legend_x = (ml + (n - 3) * cell) / fig_w
    legend_y = (lh + mb + cell * (n - 2) + cell * 0.9) / fig_h
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(legend_x, legend_y),
        frameon=True,
        fontsize=fs,
    )

    fig.suptitle("Pairwise Pareto Frontier Grid", fontsize=fs + 4, y=0.99)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150)

    return fig


def _build_legend_handles(
    portfolios: dict[str, PerformanceMetrics],
    fontsize: int,
) -> list[mpatches.Patch]:
    """Build legend Patch handles using mizani's hue palette.

    Uses the same ``hue_pal`` that plotnine's ``scale_color_discrete`` applies,
    so colours match the dots in every panel exactly.

    Arguments:
        portfolios: Mapping of portfolio label → PerformanceMetrics.
        fontsize: Font size passed through (unused here, kept for call-site symmetry).

    Returns:
        List of Patch objects suitable for ``fig.legend()``.
    """
    from mizani.palettes import hue_pal

    labels = list(portfolios.keys())
    colours = hue_pal()(len(labels))
    return [
        mpatches.Patch(color=colour, label=label)
        for label, colour in zip(labels, colours, strict=True)
    ]
