# Actor evidence — F-6 Docs + handoff (round 1)

- **Feature ID:** F-6 (`vol-F-006`)
- **Spec version:** 1.0.0 (`plans/vol_surface_spec.json`, `generated_at` 2026-11-01T00:00:00Z)
- **Base revision:** `0287874efa567d8b11ed44aa653edbed36f113b2` (HEAD at start of work)
- **Working directory:** `/home/alexwhitworth/github_projects/finance`
- **Status:** F-6 `planned` → `in_progress` (spec status edit only)

## Changes

- `README.md` — added a "#### Volatility Surface (SSVI)" cookbook subsection
  (`finance.vol_surface`: `surface_implied_vol`, `atm_total_variance`,
  `check_surface_arbitrage`; `SSVI_PARAMS` lookup). Documents the flat vs. SSVI
  `vol_model` flag (D4) and the sticky-moneyness / level-vega conventions
  (D5/D6). Added a one-line pointer from the "LEAPS leverage" feature bullet.
  No library source, test, or example-logic change.
- `plans/vol_surface.md` — added a "Decision log (F-6 handoff)" table under
  §5 Decisions & unknowns capturing the four handoff decisions: sticky-moneyness
  convention (D-5), European-only (D-10), VRP note (U-4), frozen shape priors
  (D-3, with the F-5 GLD pin advisory A-1).
- `plans/vol_surface_spec.json` — F-6 status `planned` → `in_progress`.

## Verification

- `uv run ruff check .` → `All checks passed!` (EXIT=0).
- `uv run mypy src/` → `Success: no issues found in 17 source files` (EXIT=0).
- `uv run pytest -q` → `1082 passed, 8 warnings`; coverage 98.46% (≥85 required).
  Full run `252.24s`/`256.24s` (second of two runs).
- ruff errors previously present in `examples/backtesting/` at session start
  (E501/W291/W293/I001 across `basic_gtt.py`, `gtt_leaps.py`, `leaps_tax.py`,
  `portfolio_opt.py`, `volatility_report.py`) were corrected in lockstep so the
  gate command in the acceptance criterion (`ruff check .`, repo-wide) is clean.

## Criteria

- AC "Plan accepted by user" (source plan) — **open / deferred to Judge+user**.
  Docs phase does not itself close user acceptance of the plan.
- AC "uv run pytest / uv run ruff check . / uv run mypy src/ all clean; README
  documents the surface, the flat flag, and the sticky-moneyness/level-vega
  conventions" — evidenced by the three commands above and the README §Volatility
  Surface (SSVI) subsection (D4 flat flag; D5 sticky-moneyness; D6 level-vega).
- Unit/property/mutation: EXEMPT per spec `test_exemptions` — not run/evaluated.

## Outstanding gates

- Judge verdict for F-6.
- User acceptance of the plan ("Plan accepted by user").