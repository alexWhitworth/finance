"""Visualization layer for portfolio backtest results.

All functions return a plotnine ``ggplot`` object and optionally save to
``figures/<filename>.png``.  Each function is pure modulo filesystem I/O —
pass ``output_path=None`` to suppress saving.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9

from finance.consts import CRISIS_PERIODS, NBER_RECESSION_PERIODS
from finance.metrics import PerformanceMetrics, PerformanceReport
from finance.portfolio import BacktestResult

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

    plot = (
        base
        + p9.geom_line(
            data=data, mapping=p9.aes(x="date", y="nav_millions", color="portfolio"), size=0.8
        )
        + p9.scale_x_datetime(date_labels="%Y", date_minor_breaks="1 year")
        + p9.scale_y_log10()
        + p9.labs(
            title=f"Portfolio NAV Growth\n{subtitle}",
            x="Date",
            y="NAV ($ millions)",
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

    plot = (
        base
        + p9.geom_line(
            data=data, mapping=p9.aes(x="date", y="nav_millions", color="account"), size=0.8
        )
        + p9.scale_x_datetime(date_labels="%Y", date_minor_breaks="1 year")
        + p9.scale_y_log10()
        + p9.labs(
            title=f"LEAPS Tax Drag: Taxable vs. Tax-Sheltered\n{drag_label}  {subtitle}",
            x="Date",
            y="NAV ($ millions)",
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


def plot_pareto(
    portfolios: dict[str, PerformanceMetrics],
    metrics: list[str],
    output_path: Path | None = _FIGURES_DIR / "pareto_grid.png",
) -> p9.ggplot:
    """Grid of pairwise Pareto frontier plots for the requested metrics.

    Produces one panel per unique (x, y) metric pair (lower triangle of the
    pairwise matrix) arranged in a grid that is as square as possible.

    Arguments:
        portfolios: Mapping of portfolio label → PerformanceMetrics.
        metrics: List of PerformanceMetrics field names to compare pairwise.
        output_path: Destination PNG path, or None to skip saving.

    Returns:
        A plotnine ggplot object with ``facet_wrap`` panels.

    Raises:
        ValueError: If fewer than two metrics are supplied.
    """
    if len(metrics) < 2:
        raise ValueError("At least two metrics are required for a pairwise grid.")

    pairs = [
        (x_metric, y_metric)
        for i, x_metric in enumerate(metrics)
        for y_metric in metrics[i + 1:]
    ]

    rows: list[dict] = []
    frontier_rows: list[dict] = []

    for x_metric, y_metric in pairs:
        panel = f"{_METRIC_LABELS[x_metric]} vs {_METRIC_LABELS[y_metric]}"
        for label, pm in portfolios.items():
            rows.append({
                "label": label,
                "x": getattr(pm, x_metric),
                "y": getattr(pm, y_metric),
                "panel": panel,
            })
        sub = pd.DataFrame([
            {"label": label, x_metric: getattr(pm, x_metric), y_metric: getattr(pm, y_metric)}
            for label, pm in portfolios.items()
        ])
        frontier = _pareto_frontier(sub, x_metric, y_metric)
        for _, frow in frontier.iterrows():
            frontier_rows.append({"x": frow[x_metric], "y": frow[y_metric], "panel": panel})

    data = pd.DataFrame(rows)
    frontier_data = pd.DataFrame(frontier_rows)

    n_panels = len(pairs)
    ncol = math.ceil(math.sqrt(n_panels))

    plot = (
        p9.ggplot(data, p9.aes(x="x", y="y", color="label"))
        + p9.geom_point(size=2.5)
        + p9.geom_line(
            data=frontier_data,
            mapping=p9.aes(x="x", y="y"),
            linetype="dashed",
            color="black",
            inherit_aes=False,
            size=0.5,
        )
        + p9.facet_wrap("panel", ncol=ncol, scales="free")
        + p9.labs(x="", y="", color="Portfolio")
        + p9.theme_grey()
        + p9.theme(
            figure_size=(4 * ncol, 4 * math.ceil(n_panels / ncol)),
            legend_position="bottom",
        )
    )

    if output_path is not None:
        _save(plot, output_path)
    return plot
