"""One-time validation harness: model extrinsic vs manual broker quotes (INV-7).

Reads ``validation/manual_quotes.csv`` (one row per manually entered broker
quote), prices each quote's call against the SSVI surface, and compares the
model extrinsic against the observed (bid, ask) extrinsic with the observed
half-spread. It also emits:

* per-quote model-vs-quote extrinsic with half-spread and inside/outside flags;
* (rho, eta) pinning: pinned values (from the CSV, when present) against the
  frozen SSVI priors in ``finance.consts.SSVI_PARAMS`` (D3/U1);
* kappa +/- 0.5 and sigma_inf sensitivity on the 2y ATM band (U2);
* MOVE D_eff one-time check: converted VGIT/MUB IV against listed bond-ETF
  option IVs (U6).

The result is written to ``validation/report.md`` for human review. This is a
one-time, human-reviewed script (registered as ``dev_run`` in the spec); it is
exempt from unit/property/mutation testing and must only use public ``finance``
APIs.

Usage:
    uv run validation/validate_marks.py

Rows missing a bid or ask are excluded from the extrinsic comparison and noted
in the report (spec F-5 edge case).

Extrinsic convention: ``extrinsic = call_price - max(0, S e^{-q t} - K e^{-r t})``
with maturity-matched ``r(T)`` (D8) and the forward dividend yield ``q`` from the
CSV. The model price is the Black-Scholes call at the surface IV
``sigma(k, t)`` from ``finance.vol_surface.surface_implied_vol``.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

from finance.consts import (
    DEFAULT_DIVIDEND_YIELD,
    MOVE_EFFECTIVE_DURATION,
    SSVI_PARAMS,
    SsviParams,
    VolIndexSpec,
)
from finance.leverage import bs_call_price
from finance.vol_surface import surface_implied_vol

_HERE = Path(__file__).resolve().parent
_QUOTES_PATH = _HERE / "manual_quotes.csv"
_REPORT_PATH = _HERE / "report.md"

# MOVE D_eff one-time check (U6): converted bond-ETF IV must land inside the
# listed-option IV range observed on the broker screen for these ETFs.
_LISTED_BOND_ETF_IV_RANGE: dict[str, tuple[float, float]] = {
    "VGIT": (0.03, 0.08),  # listed VGIT option IVs, decimal annualized
    "MUB": (0.02, 0.06),  # listed MUB option IVs, decimal annualized
}
# Observed ^MOVE level (basis points) at validation time, entered manually.
_MOVE_LEVEL_BP: float = 100.0

_EXCLUDED_FIELDS = ("quote_date", "spot", "vol_index_level", "bill_13w_level",
                    "treasury_2y_level", "dividend_yield", "pinned_rho",
                    "pinned_eta", "note")


@dataclass(frozen=True)
class QuoteRow:
    """A single manual broker quote parsed from the CSV."""

    ticker: str
    quote_date: str
    expiry: str
    strike: float
    spot: float
    bid: float | None
    ask: float | None
    vol_index_level: float
    bill_13w_level: float
    treasury_2y_level: float
    dividend_yield: float
    pinned_rho: float | None
    pinned_eta: float | None
    note: str


def _parse_quotes(path: Path) -> list[QuoteRow]:
    """Parse manual_quotes.csv into QuoteRow records (missing bid/ask -> None)."""
    rows: list[QuoteRow] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                QuoteRow(
                    ticker=row["ticker"].strip(),
                    quote_date=row["quote_date"].strip(),
                    expiry=row["expiry"].strip(),
                    strike=float(row["strike"]),
                    spot=float(row["spot"]),
                    bid=_maybe_float(row.get("bid")),
                    ask=_maybe_float(row.get("ask")),
                    vol_index_level=float(row["vol_index_level"]),
                    bill_13w_level=float(row.get("bill_13w_level") or _DEFAULT_BILL),
                    treasury_2y_level=float(row.get("treasury_2y_level") or _DEFAULT_2Y),
                    dividend_yield=float(row.get("dividend_yield") or DEFAULT_DIVIDEND_YIELD),
                    pinned_rho=_maybe_float(row.get("pinned_rho")),
                    pinned_eta=_maybe_float(row.get("pinned_eta")),
                    note=row.get("note", "").strip(),
                )
            )
    return rows


_DEFAULT_BILL = "0.042"
_DEFAULT_2Y = "0.046"


def _maybe_float(value: str | None) -> float | None:
    """Parse an optional float, returning None for empty/invalid cells."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _time_to_expiry_years(quote: QuoteRow) -> float:
    """Years between quote_date and expiry (floored at TIME_FLOOR by the surface)."""
    from datetime import date

    start = date.fromisoformat(quote.quote_date)
    end = date.fromisoformat(quote.expiry)
    return max((end - start).days / 365.0, 1.0 / 365.0)


def _intrinsic(spot: float, strike: float, t: float, r: float, q: float) -> float:
    """Forward-carry intrinsic for a European call: max(0, S e^{-q t} - K e^{-r t})."""
    return max(0.0, spot * math.exp(-q * t) - strike * math.exp(-r * t))


def _model_price(quote: QuoteRow, params: SsviParams, r: float) -> float:
    """Black-Scholes call price at the surface IV sigma(k, t)."""
    t = _time_to_expiry_years(quote)
    sigma = surface_implied_vol(
        spot=quote.spot,
        strike=quote.strike,
        time_to_expiry_years=t,
        vol_index_level=quote.vol_index_level,
        params=params,
        dividend_yield=quote.dividend_yield,
        risk_free_rate=r,
    )
    return bs_call_price(quote.spot, quote.strike, t, sigma, r, quote.dividend_yield)


def _compare_extrinsic(out: TextIO, quotes: list[QuoteRow]) -> None:
    """Emit the per-quote extrinsic-vs-spread table (INV-7)."""
    out.write("## 1. Extrinsic vs. observed spread\n\n")
    out.write(
        "| ticker | expiry | strike | spot | r(T) | q | model ext. | quote mid ext. "
        "| half-spread | inside? |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    inside_count = 0
    compared = 0
    excluded = [q for q in quotes if q.bid is None or q.ask is None]
    for q in quotes:
        if q.bid is None or q.ask is None:
            continue
        t = _time_to_expiry_years(q)
        r = _maturity_matched_r(q, t)
        params = SSVI_PARAMS[q.ticker]
        model_px = _model_price(q, params, r)
        model_ext = model_px - _intrinsic(q.spot, q.strike, t, r, q.dividend_yield)
        mid = (q.bid + q.ask) / 2.0
        quote_ext = mid - _intrinsic(q.spot, q.strike, t, r, q.dividend_yield)
        half = (q.ask - q.bid) / 2.0
        inside = abs(model_ext - quote_ext) <= half
        inside_count += int(inside)
        compared += 1
        out.write(
            f"| {q.ticker} | {q.expiry} | {q.strike:.2f} | {q.spot:.2f} | {r:.4f} "
            f"| {q.dividend_yield:.4f} | {model_ext:.3f} | {quote_ext:.3f} "
            f"| {half:.3f} | {'yes' if inside else 'no'} |\n"
        )
    out.write(f"\nInside the half-spread: **{inside_count}/{compared}**\n\n")
    if compared:
        pct = inside_count / compared
        out.write(f"Acceptance (INV-7): >= 2/3 inside — **{'PASS' if pct >= 2 / 3 else 'FAIL'}** "
                  f"({pct:.0%})\n\n")
    if excluded:
        out.write(
            f"\n**Excluded** (missing bid/ask): {len(excluded)} row(s) — "
            + ", ".join(f"{q.ticker} {q.expiry}@{q.strike}" for q in excluded)
            + "\n\n"
        )


def _maturity_matched_r(q: QuoteRow, t: float) -> float:
    """Maturity-matched r(T) from ^IRX and DGS2 anchors (D8), matching data.py."""
    bill = q.bill_13w_level
    twoy = q.treasury_2y_level
    anchor_short, anchor_2y = 0.25, 2.0
    if t <= anchor_short:
        return bill
    if t >= anchor_2y:
        return twoy
    frac = (t - anchor_short) / (anchor_2y - anchor_short)
    return bill + frac * (twoy - bill)


def _emit_pinning(out: TextIO, quotes: list[QuoteRow]) -> None:
    """Emit (rho, eta) pinning: pinned broker values vs frozen SSVI priors (D3/U1)."""
    out.write("## 2. (rho, eta) pinning vs. frozen priors\n\n")
    out.write("| ticker | pinned_rho | prior_rho | pinned_eta | prior_eta |\n")
    out.write("|---|---|---|---|---|---|\n")
    for q in quotes:
        if q.pinned_rho is None and q.pinned_eta is None:
            continue
        prior = SSVI_PARAMS[q.ticker]
        out.write(
            f"| {q.ticker} | {q.pinned_rho if q.pinned_rho is not None else '—'} "
            f"| {prior.rho:.3f} | {q.pinned_eta if q.pinned_eta is not None else '—'} "
            f"| {prior.eta:.3f} |\n"
        )
    out.write("\n")


def _emit_sensitivity(out: TextIO, quotes: list[QuoteRow]) -> None:
    """Emit kappa +/- 0.5 and sigma_inf sensitivity at the 2y ATM point (U2).

    The 2y ATM sigma is computed per ticker at exactly t = 2.0 with the ticker's
    latest quoted vol-index level, then recomputed with kappa shifted +/- 0.5
    (sigma_inf is held frozen; the residual sigma_inf sensitivity is reported
    as the gap between base ATM sigma and sigma_inf).
    """
    out.write("## 3. kappa / sigma_inf sensitivity on the 2y ATM band\n\n")
    out.write("| ticker | vol index | 2y base ATM sigma | kappa+0.5 | kappa-0.5 | "
              "sigma_inf (frozen) |\n")
    out.write("|---|---|---|---|---|---|\n")
    from finance.vol_surface import atm_total_variance

    # Latest vol-index level per ticker, from the quotes in file order.
    levels: dict[str, float] = {}
    for q in quotes:
        levels[q.ticker] = q.vol_index_level

    t = 2.0
    for ticker, level in levels.items():
        prior = SSVI_PARAMS[ticker]
        base = atm_total_variance(level, t, prior)
        base_sigma = math.sqrt(base / t)
        up = atm_total_variance(level, t, replace(prior, kappa=prior.kappa + 0.5))
        down = atm_total_variance(level, t, replace(prior, kappa=prior.kappa - 0.5))
        out.write(
            f"| {ticker} | {level:.2f} | {base_sigma:.4f} | "
            f"{math.sqrt(up / t):.4f} | {math.sqrt(down / t):.4f} | "
            f"{prior.sigma_inf:.4f} |\n"
        )
    out.write("\n")


def _emit_move_check(out: TextIO) -> None:
    """Emit the MOVE D_eff one-time conversion check (U6)."""
    out.write("## 4. MOVE D_eff conversion check (VGIT/MUB)\n\n")
    out.write(f"Observed ^MOVE level (bp): {_MOVE_LEVEL_BP:.1f}\n\n")
    out.write("| ETF | D_eff | converted IV | listed-option range | inside? |\n")
    out.write("|---|---|---|---|---|\n")
    for etf, d_eff in MOVE_EFFECTIVE_DURATION.items():
        spec = VolIndexSpec(unit="basis_points", effective_duration=d_eff)
        from finance.data import convert_vol_index_level

        iv = convert_vol_index_level(_MOVE_LEVEL_BP, spec)
        lo, hi = _LISTED_BOND_ETF_IV_RANGE.get(etf, (float("-inf"), float("inf")))
        inside = lo <= iv <= hi
        out.write(f"| {etf} | {d_eff} | {iv:.4f} | [{lo:.2f}, {hi:.2f}] | "
                  f"{'yes' if inside else 'no'} |\n")
    out.write("\n")


def main() -> None:
    """Run the validation harness end-to-end and write report.md."""
    quotes = _parse_quotes(_QUOTES_PATH)
    if len(quotes) < 10:
        raise SystemExit("manual_quotes.csv must contain >= 10 quotes (F-5 requirement)")

    with _REPORT_PATH.open("w") as out:
        out.write("# Validation report: model extrinsic vs. manual broker quotes\n\n")
        out.write(f"Quotes read: {len(quotes)} from `{_QUOTES_PATH.name}`\n\n")
        _compare_extrinsic(out, quotes)
        _emit_pinning(out, quotes)
        _emit_sensitivity(out, quotes)
        _emit_move_check(out)

    print(f"Wrote {_REPORT_PATH}")


if __name__ == "__main__":
    main()
