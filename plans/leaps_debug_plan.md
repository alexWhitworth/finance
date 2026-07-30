# Project: LEAPS Accounting Fix — Implementation Plan

> Binding implementation plan for the defects diagnosed in `analysis/leaps_debug.md`.
> That document is the *investigation*; this is the *build contract*. Read it first
> for symptom evidence and root-cause reasoning. This plan is the input to
> `plan-to-spec`.

---

## 1. System Overview

### Goal

Correct the LEAPS carve-out backtest so that portfolio NAV reflects the true
mark-to-market value of live LEAPS contracts at every date, with **no lookahead**.
Ship the fix behind a test suite that encodes the system properties the current
suite failed to encode: no temporal leakage, conserved accounting, and plausible
output distributions.

### The four defects (from `leaps_debug.md`, re-confirmed against source)

| # | Defect | Location | Confirmed | Severity |
|---|--------|----------|-----------|----------|
| **1** | `_live_contracts` builds `rolled_out` with no date bound → lookahead. Day-1 NAV = $600k (base only); terminal ~5–10x spike as the final unrolled cohort appears. | `leverage.py:603` | ✅ Yes | **Critical** (primary) |
| **2** | `vol_prices` column lookup uses literal `"^VIX"`, but `build_price_data` keys columns by **asset ticker** (`"VTI"`). VIX-dynamic IV silently never engages. | `portfolio.py:555` | ✅ Yes | Medium |
| **3** | `splice` raises when `end_date < splice_date` (empty `post`). No guard. | `data.py:292` | ✅ Latent (not triggered by current example) | Low |
| **4** | No integration test exercises the LEAPS lifecycle. The one real-data test runs `leaps_config=None`. **Also:** it reads `df["IRX"]`/`df["VIX"]`, but the updated parquet ships `^IRX`/`^VIX` → the smoke test is already broken. | `test_integration.py:253-256, 274-281` | ✅ Yes | High (this is why the bug survived) |

### Root cause (one sentence)

State at time `T` depended on `roll_events` generated for times `> T`, because the
pre-computed ledger holds the full roll history at construction time and
`_live_contracts` never filtered it by date.

### Design axiom for the fix

`_live_contracts(ledger, T)` must be a **pure function of events with
`event_date <= T`**. No field of the returned set may depend on any event dated
after `T`. This is the falsifiable property the primary fix and its regression
test must both satisfy.

### Confirmed data facts (grounding for Fix 4)

`data/price_data.parquet` (git-modified, re-inspected):
- Columns: `VTI, VXUS, GLD, MUB, KMLM, VGIT, ^IRX, ^VIX`
- Span: `2018-01-02 → 2024-12-31` (7.0 years, 1761 rows)
- 7 years @ 2-year tenor + ≥366-day min hold ⇒ a day-1 cohort rolls ≈ mid-2019,
  again ≈ 2021, again ≈ 2023 → **full purchase→live→roll→expiry lifecycle is
  covered** ≥ 3 times. This is the temporal realism the mocked `conftest.py`
  fixtures never provided.

---

## 2. Tech Stack & Dependencies

- Python 3.14, `uv run` for all execution.
- `pandas`, `scipy` (existing).
- `pytest`, `pytest-cov` (existing). `hypothesis` for a property test on the
  no-lookahead invariant.
- No new runtime dependencies. Pure-function core preserved; I/O stays in `data.py`.

---

## 3. Data Schema / Type Definitions

No dataclass schema changes. All required fields already exist:

- `LeapsRollEvent.roll_date: pd.Timestamp` (`leverage.py:137`) — the date oracle
  Fix 1 filters on.
- `LeapsGttCloseEvent.close_date` (`:182`), `LeapsPartialCloseEvent.close_date`
  (`:159`) — reviewed for the same lookahead class (see §4.1 note).
- `PriceData.vol_prices: pd.DataFrame` — columns keyed by **asset ticker**; this
  key convention is the contract Fix 2 must honor.

**Schema decision (Fix 2 / Fix 4):** the *production* convention
(`vol_prices` keyed by asset ticker) is authoritative. The test fixture in
`test_integration.py` (which keys by `"^VIX"`) is the side that changes.

---

## 4. Component / Module Breakdown (changes by file)

### 4.1 `leverage.py` — `_live_contracts` (PRIMARY FIX)

**Current (`:603`):**
```python
rolled_out = {event.old_contract for event in ledger.roll_events}
```

**Target:**
```python
rolled_out = {
    event.old_contract
    for event in ledger.roll_events
    if event.roll_date <= current_date      # exclude only AFTER the roll occurs
}
```

Rationale for `<=` (not `<`): on the roll date the old contract closes and its
replacement (`new_contract.purchase_date == roll_date`) opens. `new_contract` is
already in `ledger.contracts` and passes the `purchase_date > current_date`
guard at `<=`, so proceeds carry over with no gap and no double-count. Verified
against `roll_contract` (`:490-541`).

**Lookahead audit of the sibling exclusion sets** (do in the same edit, since the
regression test asserts the general property, not just the roll case):
- `gtt_closed` (`:604`) — built from `gtt_close_events`; each has `close_date`.
  A GTT close at `T_c` must not hide a contract at `T < T_c`. **Add the same
  `event.close_date <= current_date` bound.**
- `partially_closed` (`:605-608`) — maps original→continuation with reduced
  `n_contracts`. This is a *substitution*, not an exclusion. Applying a future
  partial-close's reduced size at an earlier date is the same lookahead class
  (understates size before the close). **Guard with
  `ev.close_date <= current_date`**; before that date, use the original.

> These two are in-scope for the *invariant* (§6 asserts no-lookahead for all
> event-driven state), but they are lower-risk and split into Phase 1b, because:

**Guard scope (trace evidence, `portfolio.py` + `metrics.py`):**
- **Roll guard is the live bug.** In-loop, `_live_contracts` is called on the
  per-window ledger from `run_leaps_simulation` (`portfolio.py:681`), which holds
  all of that window's roll events at construction → future rolls are present at
  query time → lookahead is real. **Phase 1a.**
- **gtt/partial guards fix no active path (today).** `gtt_close_events` and
  `partial_close_events` are attached to the ledger only *post-loop*
  (`portfolio.py:821, 831-834`); during the loop both tuples are empty, so the
  guards are in-loop no-ops. The sole post-hoc caller is `compute_terminal_nav`
  (`metrics.py:383`), which queries at `final_date`, where every close event is
  trivially `<= final_date` — a no-op again. They are correctness insurance, not
  a fix. **Phase 1b.**
- **Partial-close trap:** `partial_close_events` are frozen with a *synthetic*
  `close_date = final_date` (`portfolio.py:809`), not the true close date. A
  naive `ev.close_date <= current_date` guard therefore reverts to the
  full-size original at any date `< final_date`. Harmless only because the query
  date is always `final_date`. See Phase 1b for the required handling.

### 4.2 `portfolio.py` — `vol_prices` lookup (Fix 2, lookup only; activation gated)

**Current (`:555-557`):**
```python
if not price_data.vol_prices.empty and "^VIX" in price_data.vol_prices.columns:
    raw_vix = price_data.vol_prices["^VIX"].reindex(idx, method="ffill")
    mtm_iv_series = raw_vix.rolling(VIX_MTM_WINDOW).mean().ffill()
```

**Target:** key by the LEAPS `underlying` already in scope (`:544`), not a literal:
```python
if not price_data.vol_prices.empty and underlying in price_data.vol_prices.columns:
    raw_vix = price_data.vol_prices[underlying].reindex(idx, method="ffill")
    mtm_iv_series = raw_vix.rolling(VIX_MTM_WINDOW).mean().ffill()
```

**Activation is isolated (per decision).** The primary fix (§4.1) is validated
FIRST with dynamic IV *not* engaging — i.e. the phase-2 validation runs on a
`vol_prices` that does not contain the underlying key (or empty), so behavior is
constant-IV and the NAV correction is attributable solely to Fix 1. Only after
Fix 1 is signed off do we enable dynamic IV (fixture provides the
`underlying`-keyed column) and validate the *incremental* change separately. See
§5 phases 2 vs 4.

### 4.3 `data.py` — `build_price_data` splice guard (Fix 3)

**Current (`:292`):** `if ticker in asset_tickers and start_date < splice_date:`

**Target:** also require the window to actually reach the splice date, so a
window entirely before the splice date does not attempt a splice with empty
`post`:
```python
if ticker in asset_tickers and start_date < splice_date <= end_date:
```
Behavior when `end_date < splice_date`: skip the splice, fetch the primary as-is
(matches the `use_splice=False` path). Keep `splice()` itself raising as the
last-line defense; this guard prevents the caller from ever calling it with an
empty `post`.

### 4.4 `test_integration.py` — fixture schema + LEAPS lifecycle + plausibility (Fix 4)

Three edits to the real-data path:
1. **Un-break the loader:** read `df["^IRX"]` / `df["^VIX"]` (not `IRX`/`VIX`).
2. **Fixture schema:** build `vol_prices` keyed by asset ticker (`{"VTI": vix}`)
   so it matches production and Fix 2 can engage in phase 4. Keep a
   constant-IV variant (empty/underlying-absent `vol_prices`) for the phase-2
   lifecycle assertions.
3. **New LEAPS lifecycle test class** (details in §6).

---

## 5. Step-by-Step Implementation Roadmap

Ordering is deliberate: land the critical fix in isolation, prove it, *then*
layer the secondary behavior change so the two are never conflated in one
validation.

### Phase 0 — Reproduce & baseline (no code change)
- Fix only the parquet column names in the test loader enough to run
  `with_leaps.py` semantics against the 7-yr parquet (or a scratch script).
- Capture the *broken* baseline: day-1 NAV, terminal NAV, skewness, excess
  kurtosis, max 1-day return. This is the "before" evidence for review.

### Phase 1 — Primary fix (`_live_contracts` date-awareness)

The three event guards in §4.1 are NOT peers — they differ sharply in risk and
in whether they fix an active defect. Trace evidence (see §4.1 "Guard scope"):
the roll guard fixes the live bug; the gtt/partial guards are defensive and
currently exercise no lookahead on any call path. Split accordingly.

- **1a — roll guard (the fix, ship this).** Apply ONLY the
  `event.roll_date <= current_date` filter to `rolled_out`. This is the entire
  cause of the day-1 and terminal symptoms. Run `tests/test_leverage.py`,
  `tests/test_gtt_leaps_close.py`, `tests/test_portfolio.py`. Per the §8
  spot-check, no existing test encodes the bug, so expect all green; confirm
  before/after.
- **1b — gtt-close & partial-close guards (defensive, verify-first).** Add the
  `close_date <= current_date` bound to `gtt_closed` and `partially_closed`.
  **Before writing:** confirm the trace claim that both are no-ops on current
  call paths (in-loop tuples are empty; post-hoc query date is always
  `final_date`). **Trap:** `partial_close_events` carry a *synthetic*
  `close_date = final_date` (`portfolio.py:809`), not the true close date — a
  naive guard reverts to the full-size original at any earlier date. Either
  (i) key the partial-close guard on the true close semantics, or (ii) if that
  requires threading real dates through the freeze step, scope 1b to the
  gtt-close guard only and log the partial-close guard as a documented
  no-lookahead follow-up. Decide in-step; do not bundle into 1a.

### Phase 2 — Validate primary fix under CONSTANT IV (dynamic IV NOT engaged)

Decomposed into four independently-committable steps. 2a–2c are parallelizable
(distinct test targets, no shared state); **2d depends on 2c** because the
plausibility bounds cannot be locked until the corrected baseline is observed.
Both suites run in ~4s, so runtime is never the constraint — these splits exist
to keep each implementation step single-purpose and reviewable.

- **2a — `_live_contracts` unit tests [ATOMIC].** The three-guard date-awareness
  cases (roll / gtt-close / partial-close), including the same-day boundary
  (INV-1 unit form). Target: `tests/test_leverage.py`. No fixture rework.
- **2b — no-lookahead property test [ATOMIC].** One `hypothesis` test: the live
  set at `T` is invariant to appending any event dated `> T`. Target:
  `tests/test_leverage.py`. Independent of 2a.
- **2c — LEAPS lifecycle integration test (constant IV).** Un-break the parquet
  loader (`^IRX`/`^VIX`), add the constant-IV fixture variant (`vol_prices`
  WITHOUT the underlying key), assert INV-2 (day-1 NAV ≈ `initial_nav`, ±1%) and
  INV-3 (conserved carve-out). Target: `tests/test_integration.py`. **This
  attributes the entire NAV correction to Fix 1 alone.**
- **2d — calibrate & lock INV-4 plausibility bounds.** Depends on 2c: read the
  corrected baseline (skew, excess kurtosis, ann. return, max 1-day return),
  set the §6 INV-4 numbers with headroom, wire them as hard assertions. This is
  the step the provisional bounds in §6/§7 explicitly defer to.

### Phase 3 — Splice guard (Fix 3)
- Edit `data.py` per §4.3. Add a unit test: a window ending before a ticker's
  splice date returns primary-only, no raise.

### Phase 4 — Enable dynamic IV (Fix 2 activation) + isolate its effect
- Edit `portfolio.py` lookup per §4.2.
- Flip the integration fixture to the asset-ticker-keyed `vol_prices` so dynamic
  IV engages. Re-run the plausibility assertions.
- Add a focused test that dynamic IV actually changes MTM vs constant IV (proves
  the lookup now engages) while both remain within plausibility bounds. This
  isolates Fix 2's incremental effect from Fix 1.

### Phase 5 — Regression gate + cleanup
- Wire the plausibility assertions as hard pytest failures (§6 [INTEGRATION]).
- `uv run pytest` green; `uv run ruff check .`; `uv run mypy src/`.
- Coverage ≥ existing threshold. Update `README.md` / docstrings if signatures
  changed (they do not, per §3). Regenerate `outputs/figures/leaps_tax_drag.png`
  from `with_leaps.py` as after-evidence.

---

## 6. System & Test Invariants

Tags: **[ATOMIC]** = provable at unit level; **[INTEGRATION]** = only provable
when leverage + portfolio + metrics compose over real time.

### INV-1 — No Lookahead (temporal bound) **[ATOMIC]**
> For any `T` and any `ledger`, `_live_contracts(ledger, T)` depends only on
> events with `event_date <= T`.

- **Falsifiable unit test:** construct a ledger with one contract purchased at
  `t0` and a `roll_event` dated `t2`. Assert the contract IS live at `t1`
  (`t0 < t1 < t2`) and is NOT live at `t3 > t2`. Pre-fix this fails at `t1`.
- **Property test (hypothesis):** for random event dates, the live set at `T`
  is invariant to appending any event dated `> T`.
- Repeat the construction for `gtt_close_events` and `partial_close_events`.

### INV-2 — Day-1 accounting (oracle: `initial_nav`) **[INTEGRATION]**
> On the first trading day, `nav_series.iloc[0] ≈ config.initial_nav` (rel 1%),
> including the carved-out LEAPS sleeve.

- Oracle is the *input* `initial_nav`, independent of the NAV derivation logic.
- Pre-fix value: ~$600k (the 40% carve-out was invisible). This test is the
  direct guard against the day-1 symptom.

### INV-3 — Conserved carve-out (accounting) **[INTEGRATION]**
> At every date, `total_nav == sum(base_holdings) + leaps_value +
> defensive_sleeve + leaps_pool`, and no capital is created or destroyed by a
> roll (net proceeds in = net proceeds out, modulo realized tax).

- Assert on the no-GTT, taxable path: cumulative contributions + market P&L −
  realized tax ties to terminal NAV within tolerance.

### INV-4 — Plausibility bounds (hard CI gate) **[INTEGRATION]**
> On the 7-yr real-data LEAPS backtest, derived metrics fall in domain-sane
> ranges. **Failing assertions**, per decision.

Proposed bounds (tune against the corrected baseline in Phase 2 before locking;
these are starting gates, not final numbers):
| Metric | Bound | Rationale |
|--------|-------|-----------|
| max single-day \|return\| | `< 0.50` | A DITM LEAPS sleeve at 40% weight, ~2x delta-leverage, cannot plausibly move NAV >50% in a day. Pre-fix hit 27–59%. |
| annualized return | `-0.20 < r < 0.60` | Leveraged equity over 2018–2024; generous but excludes the 5–10x pathology. |
| skewness | `\|skew\| < 5` | Pre-fix: 32–53. |
| excess kurtosis | `< 50` | Pre-fix: ~1950–3620. |
| terminal NAV | within `[0.5x, 20x]` of contributed capital | excludes the 5–10x-in-18-months spike as a structural break. |

> These bounds are the encoded form of the process-failure lesson: a coarse
> "is this backtest sane?" check that CI runs every time.

### INV-5 — Dynamic IV engages (Fix 2) **[INTEGRATION]**
> With an underlying-keyed `vol_prices`, daily MTM IV differs from `config.iv` on
> at least some dates; with it absent, IV is constant. Both stay within INV-4.

### Realism gate
- INV-2, INV-3, INV-4 **require the 7-yr real parquet** (or equivalent
  multi-roll-cycle data). Static single-contract mocks are insufficient — they
  are what let the bug survive. The lifecycle test MUST span ≥ 2 roll cycles.

---

## 7. Known Assumptions

1. `data/price_data.parquet` (2018–2024, `^IRX`/`^VIX`) is the canonical Fix-4
   fixture; its git-modified state is intentional and will be committed.
2. Production `vol_prices` keys columns by asset ticker; the test fixture is the
   side corrected to match (not vice versa).
3. `<=` is the correct date-comparison operator for all three event guards
   (roll / gtt-close / partial-close), consistent with same-day
   close-and-reopen semantics.
4. Fix-forward is sufficient; no git bisect (per decision). The regression tests,
   not archaeology, are the guarantee going forward.
5. No public dataclass or function signatures change ⇒ no downstream API churn.
6. Plausibility bounds in INV-4 are provisional until calibrated against the
   Phase-2 corrected baseline; final numbers set before Phase 5 locks the gate.

---

## 8. Potential Edge Cases & Pre-Mortem Risks

- **Same-day roll boundary.** A contract rolled exactly on `T`: with `<=`,
  old is excluded and new is live on `T`. Add an explicit unit test at the
  boundary (off-by-one is the highest-risk error in this fix).
- **Existing tests asserting buggy behavior — spot-checked, LOW risk.** Traced
  every lookahead-sensitive assertion in both suites (126 tests, ~4s):
  `test_nav_contribution_excludes_rolled_contracts` (`test_leverage.py:453`)
  evaluates *at* `roll_date`, where `<=` keeps the old contract excluded → passes
  unchanged; the day-1 cost-basis tests (`test_portfolio.py:497,523`) assert
  contract cost basis, never NAV or the live set. **No test pins a terminal NAV
  or an empty live-set, so none encodes the bug** — which is precisely the
  process-failure thesis. Phase 1 is expected to land clean; still confirm by
  running both suites before/after.
- **GTT segmented ledgers.** `run_backtest` concatenates per-window ledgers
  (`portfolio.py:826-835`). The date filter must behave correctly when
  `roll_events` from multiple windows coexist in one merged ledger — dates are
  globally monotonic across windows, so the filter is still correct, but the
  lifecycle test should include at least one GTT-active variant to confirm.
- **Fix 2 changes numbers even when "off".** Ensure the phase-2 constant-IV
  fixture genuinely excludes the underlying key; otherwise dynamic IV leaks into
  the primary-fix validation and defeats the isolation.
- **Plausibility bounds too tight.** Leveraged returns are legitimately
  fat-tailed; over-tight bounds cause flaky CI. Mitigation: calibrate on the
  corrected baseline, leave headroom, prefer bounds that only trip on
  order-of-magnitude structural breaks (the failure class we care about).
- **Splice guard over-correction.** `start_date < splice_date <= end_date` must
  still splice the normal case (window straddles the splice date). Unit-test both
  sides of the boundary.
- **`rolling(VIX_MTM_WINDOW).mean()` warm-up NaNs** at series start once dynamic
  IV engages — code already `.ffill()`s and floors at `config.iv`; confirm the
  first `VIX_MTM_WINDOW` days fall back to the floor, not NaN, in the lifecycle
  test.

---

## 9. Definition of Done (evidence for review)

- [ ] `uv run pytest` green, including new INV-1…INV-5 tests.
- [ ] Before/after table: day-1 NAV, terminal NAV, skewness, excess kurtosis,
      max 1-day return (Phase-0 broken vs Phase-2 corrected vs Phase-4 dynamic-IV).
- [ ] `_live_contracts` no-lookahead property test passing (hypothesis).
- [ ] Regenerated `outputs/figures/leaps_tax_drag.png` showing a smooth NAV path.
- [ ] `uv run ruff check .` and `uv run mypy src/` clean.
- [ ] Coverage ≥ threshold.
