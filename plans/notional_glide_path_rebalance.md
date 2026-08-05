# Notional Glide-Path Rebalancing

## Origin: The DRIFT Bug

During development, a bug in `portfolio.py` set the LEAPS trim target to a fixed dollar
amount at initialization rather than recomputing it as a fixed weight of current NAV:

```python
# Buggy: fixed at $400k forever
target_leaps_value = config.initial_nav * leaps_fraction

# Fix: recomputed at each monthly check
target_leaps_now = total_val * leaps_fraction
```

The bug accidentally implemented **constant-notional rebalancing**: trim LEAPS gains back to
the initial seed dollar amount each month, redirect all proceeds to the diversified base. Over
a 25-year simulation with $10k/month contributions, LEAPS decayed from 40% to ~3% of NAV while
the diversified base compounded the harvested leveraged gains.

### Bug outputs vs. correct DRIFT outputs (LEAPS + base, 2000–2026)

| Metric | Correct DRIFT (fixed weight) | Bug (fixed notional) |
|---|---|---|
| Terminal NAV (sheltered) | ~$60M | ~$182M |
| Ann. Return | 14.6% | 20.1% |
| Sharpe | 0.67 | 1.33 |
| Sortino | 0.58 | 1.72 |
| Max Drawdown | 45% | 23% |
| GFC Ann. Return | −22% | +0.9% |

### Why the bug outperformed

The outperformance is not an artefact — it is the correct output of a different (accidental)
strategy. Three mechanisms drove it:

1. **Harvested leverage compounds in low-correlation assets.** Every month VTI rose, LEAPS
   gains above $400k were liquidated into VXUS/GLD/VGIT/MUB. Those assets have near-zero or
   negative correlation with equities. The proceeds compounded safely without retaining the
   leverage risk.

2. **Avoided variance penalty on geometric compounding.** Geometric mean return ≈ arithmetic
   mean − variance/2. Each avoided drawdown eliminates not just the loss but the compounding
   handicap on all future returns. The bug had negligible LEAPS exposure by 2008 (≈16% weight),
   so the GFC barely registered. That preserved compounding base then grew undisturbed for
   another 18 years.

3. **Implicit momentum / profit-taking.** The cap is one-sided: trim on gains, do nothing on
   losses. This is structurally a trend-following profit-taking rule on a leveraged position.
   In a long bull market it captures levered upside early and locks it in.

### Rebalancing paradigm taxonomy

| Paradigm | Rule | Market bet |
|---|---|---|
| Buy-and-hold | Never rebalance | Trend |
| Constant-mix (DRIFT) | Fixed weight | Mean reversion |
| Constant-notional | Fixed dollar cap | Trend + protection |
| **Glide-path (proposed)** | **Decaying weight cap** | **Trend + lifecycle protection** |

---

## Proposed Strategy: Glide-Path LEAPS Weight

The bug's fixed-notional rule is pathological at long horizons because LEAPS eventually decay
to a rounding error. The proposed strategy formalizes the underlying insight — harvest leveraged
gains into the base, de-lever as real wealth accumulates — as a controlled, parameterized
schedule that converges to a non-zero floor rather than zero.

### Core idea

The LEAPS target weight follows an exponential decay indexed to the NAV multiple of total
contributed capital. As the portfolio accumulates market wealth above what the investor has
put in, the leveraged allocation gradually shrinks toward a floor. Leverage is preserved in
the early accumulation phase when there is little wealth to protect; it declines as the
portfolio grows beyond contributed capital.

### Index variable: NAV multiple of hurdle-adjusted contributed capital

```
Contributed_hurdle(t) = Initial_NAV × (1 + R_f(t))^t
                        + Σ_{τ=1}^{t} Contribution_τ × (1 + R_f(t))^{t-τ}

m(t) = NAV(t) / Contributed_hurdle(t)
```

where `R_f` is the contemporaneous 13-week T-bill yield (`^IRX`), the same series the
backtest already uses for the defensive portfolio (`GTT_RISK_FREE_KEY`).

**Recurrence form** (preferred for implementation — avoids recomputing from scratch each step):

```
Contributed_hurdle(t) = Contributed_hurdle(t-1) × (1 + r_f(t))^(1/12) + Contribution_t
```

where `r_f(t)` is the annualized `^IRX` rate at time `t` and `1/12` converts to a monthly
compounding factor. This is a running scalar updated once per month-end, exactly like the
existing `total_contributed`, extended with a Rf growth factor applied before adding each
new contribution.

This is preferred over calendar time because:
- It is agnostic to *when* wealth accumulates; it responds to *whether* wealth has been
  accumulated and therefore needs protection.
- Drawdowns reduce `m`, partially restoring the target weight. This is behaviorally correct:
  after losing money, the portfolio has less wealth to protect and leverage is again justified.
- Market outperformance accelerates de-levering; underperformance slows it. The schedule
  self-calibrates to the portfolio's actual outcome.
- **Hurdle semantics:** `m = 1.0` means NAV equals what a risk-free investment of the same
  cash flows would have returned, not merely nominal break-even. De-levering only begins once
  the portfolio has cleared the risk-free bar — generating genuine excess returns.

A multiple of 1.0 means the investor has matched the risk-free return on all contributed capital.
A multiple of 2.0 means the portfolio has returned 2× the risk-free-compounded contributions.

**Behavior in prolonged underperformance:** If `NAV(t)` grows slower than `Contributed_hurdle(t)`,
`m(t)` falls below 1.0. The schedule returns `w0` (no de-levering). The investor is
underperforming Rf and should retain full leverage. `m` may drift arbitrarily below 1.0
in extended bear markets; the floor holds LEAPS at `w0` throughout. This is the intended behavior.

**Behavior during GTT defensive windows:** The defensive portfolio earns `≈ R_f` by construction.
`Contributed_hurdle` also grows at `R_f`. Therefore `m(t) ≈ m(t-1)` throughout a defensive
window (modulo new contributions, which are neutral in the same way). The glide-path clock
effectively pauses during defensive regimes — neither crediting nor penalizing the investor
for a regime-driven risk-off period.

### Schedule function: exponential decay

```
w(m) = floor + (w0 − floor) × exp(−λ × max(m − 1, 0))
```

- `w0`: initial LEAPS weight (e.g. 0.40). Matches `leaps_fraction` from `PortfolioConfig`.
- `floor`: minimum LEAPS weight at any wealth level (e.g. 0.05–0.10).
- `λ = ln(2) / half_life_multiple`: decay rate. At `half_life_multiple = 2.0`, the active
  weight above the floor halves each time the NAV multiple doubles.
- `max(m − 1, 0)`: decay only begins once the portfolio is above break-even. Below m=1.0
  (underwater), the full initial weight is maintained.

The `(m − 1)` shift ensures no de-levering occurs until real market gains exist. This avoids
penalizing the investor for contributions made during a drawdown.

### Why exponential over sigmoid

Sigmoid's behavioral feature is a slow start (low de-levering in early years). With
NAV-multiple indexing, that property is already enforced by the index itself: in the first
years `m` is near 1.0 because contributions dominate, so the schedule barely moves regardless
of functional form. The sigmoid's slow start is redundant. Exponential is the cleaner choice:
one interpretable parameter, smooth, no kinks.

### Half-life calibration

At `half_life_multiple = 2.0`:
- Weight halves when NAV has doubled contributed capital.
- At historical US equity returns (~7% real), NAV doubles contributed capital in roughly
  10 years (the market doubles in ~10 years at 7% real CAGR, and contributions are a
  decelerating fraction of NAV over time).
- This produces a ~10-year half-life in calendar time, matching standard lifecycle investing
  intuition ("de-risk over a decade").

The parameter is directly interpretable: "I want to be half as leveraged once I have
doubled my contributed capital."

### Rebalancing rule: one-sided cap

```
if leaps_weight > schedule(m) + drift_band:
    trim LEAPS to schedule(m); proceeds → base at base_target_weights
```

No top-up when `leaps_weight < schedule(m)`. This is a lifecycle protective strategy, not
a mean-reversion strategy. Re-levering into drawdowns conflicts with the protective intent.
The drift band prevents excessive transaction frequency near the boundary.

---

## Implementation Components

The backtest engine has been refactored since this document was first drafted. `portfolio.py`
is now a thin dispatcher; all per-day logic lives in `_backtest_steps.py`, and shared frozen
dataclasses live in `_portfolio_types.py`. All implementation references below reflect that
structure.

### 1. `GlidepathConfig` dataclass (`_portfolio_types.py`)

```python
@dataclass(frozen=True)
class GlidepathConfig:
    half_life_multiple: float = 2.0   # NAV multiple at which active weight halves
    floor: float = 0.05               # minimum LEAPS weight
    drift_band: float = DRIFT_BAND_RELATIVE  # band before trim fires (default 0.10)
```

Placement follows `GttConfig`'s pattern: consumed by the backtest loop and stored on
`PortfolioConfig`, so it lives in `_portfolio_types.py` alongside `GttConfig`. This
is distinct from `LeapsConfig` (in `leverage.py`), which belongs to the LEAPS pricing engine.

Parameters are independent of `LeapsConfig` (which governs BS pricing) and `PortfolioConfig`
(which holds `leaps_fraction` / `w0`). The `w0` comes from the existing `leaps_fraction`
derived from `target_weights`. No `risk_free_rate` field: the hurdle rate is sourced from
`inputs.rfr` (the per-day `^IRX`-derived annualized decimal rate already extracted by
`_extract_day_inputs`), consistent with how `GTT_RISK_FREE_KEY` earns Rf in the defensive
portfolio.

### 2. `RebalanceRule.GLIDE_PATH` enum value (`leverage.py`)

New variant alongside `QUARTERLY` and `DRIFT`. Dispatch is added to `_apply_rebalance`
in `_backtest_steps.py`, in the same block that currently handles `RebalanceRule.DRIFT`.

### 3. Schedule function (pure, `_backtest_steps.py`)

```python
def glide_path_target_weight(
    m: float,
    w0: float,
    floor: float,
    half_life_multiple: float,
) -> float:
    lam = math.log(2) / half_life_multiple
    active_weight = (w0 - floor) * math.exp(-lam * max(m - 1.0, 0.0))
    return floor + active_weight
```

Placed alongside other pure helpers in `_backtest_steps.py` (e.g. `_should_rebalance`).
Key invariants:
- `w(1.0) == w0` (no decay at break-even)
- `w(m) → floor` as `m → ∞`
- Monotone non-increasing in `m` for `m ≥ 1`

### 4. `hurdle_contributed` field on `PortfolioState` (`_portfolio_types.py`)

All mutable loop state is carried forward as frozen `PortfolioState` fields (via
`dataclasses.replace`). `hurdle_contributed` is therefore a new field on `PortfolioState`,
not a local loop variable:

```python
@dataclass(frozen=True)
class PortfolioState:
    ...
    hurdle_contributed: float   # Rf-compounded contribution denominator for m(t)
```

Initialized to `config.initial_nav` in `_build_initial_state`. Updated each month-end
inside `_apply_contribution` (which already fires on `is_month_end`) before `_apply_rebalance`
reads it:

```python
# Inside _apply_contribution, on is_month_end:
monthly_rf = (1.0 + inputs.rfr) ** (1.0 / 12.0)   # inputs.rfr is already annualized decimal
new_hurdle = state.hurdle_contributed * monthly_rf + ctx.config.monthly_contribution
```

`inputs.rfr` is the per-day annualized `^IRX` rate already extracted by `_extract_day_inputs`
from `ctx.rfr_series`. No new data dependency; no raw price indexing required.

**Prerequisite — `rfr_series` population in `_build_context`:** Currently `ctx.rfr_series` is
only populated when `use_leaps` is True. GLIDE_PATH requires it unconditionally. `_build_context`
must populate `rfr_series` whenever `config.rebalance_rule == RebalanceRule.GLIDE_PATH`, regardless
of `use_leaps`. The existing forward-fill already handles weekend gaps; `_build_context`'s
validation block must assert that `rfr_series` has no leading NaNs before the backtest start date.

### 5. Trim logic (`_apply_rebalance`, `_backtest_steps.py`)

At each month-end when `RebalanceRule.GLIDE_PATH` (after `_apply_contribution` has updated
`hurdle_contributed`):
1. Compute `m = total_nav / state.hurdle_contributed`.
2. Compute `target_w = glide_path_target_weight(m, w0, floor, half_life_multiple)`.
3. Compute `current_leaps_weight = leaps_value / total_val`.
4. If `current_leaps_weight > target_w + drift_band`: trim LEAPS to `target_w × total_val`
   using the existing `leaps_scale` mechanism; redirect proceeds to base at `base_target_w`.
5. No action if `current_leaps_weight ≤ target_w + drift_band`.

The existing `leaps_scale` dict and partial-close accumulation pattern from the DRIFT path
in `_apply_rebalance` are reused unchanged.

**GTT re-entry seeding:** `_apply_gtt_reentry` currently seeds the fresh LEAPS simulation
with `total * ctx.leaps_fraction`. Under GLIDE_PATH it must instead seed with
`glide_path_target_weight(m_current) × total`, where `m_current = total / state.hurdle_contributed`
at the re-entry date. Requires a GLIDE_PATH branch inside `_apply_gtt_reentry`.

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Index variable | NAV / Rf-hurdle-adjusted contributed capital | De-levering only begins after clearing the risk-free bar; drawdowns self-correct; defensive clock pauses |
| Rf source | `^IRX` (existing `TBILL_TICKER`) | Consistent with GTT defensive earnings; no new data dependency; time-varying captures ZIRP era correctly |
| Functional form | Exponential decay | One parameter; sigmoid slow-start is redundant with NAV-multiple indexing |
| Half-life | 2× NAV multiple (~10 yr) | Aligns with market doubling time at historical returns |
| Floor | Fixed weight, 5–10% | Prevents strategy from fully abandoning leverage |
| Symmetry | One-sided cap | Lifecycle protection, not mean reversion; re-levering into drawdowns is inconsistent |
| Absolute ceiling | None | Strategy is multigenerational / no terminal date; weight floor is the correct risk control |
| Placement | `RebalanceRule.GLIDE_PATH` | Alongside QUARTERLY/DRIFT; DRIFT logic unchanged |
| Parameters | `GlidepathConfig` dataclass | Decoupled from `LeapsConfig` (pricing) and `PortfolioConfig` (weights) |
| Drift band | Reuse `DRIFT_BAND_RELATIVE` | Consistent with existing trim threshold; avoids churn near boundary |

---


