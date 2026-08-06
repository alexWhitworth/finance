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

### Rebalancing rule: dynamic DRIFT targets

The glide path does not introduce a new rebalancing rule. The portfolio continues to execute
standard two-sided DRIFT rebalancing. What changes is the *target* the DRIFT check measures
against: target weights are a time-varying function of `m(t)`, updated each month-end.

The full target weight schedule at any `m`:

```
w_freed(m)      = w0 − w_LEAPS(m)                                    # weight released from LEAPS

w_LEAPS_target  = w_LEAPS(m)
w_VTI_target    = w_freed(m) × vti_alpha
w_a_target      = (original_base_w[a] + w_freed(m) × (1 − vti_alpha)) × base_target_w_normalized[a]
```

**At `m = 1.0`:** `w_freed = 0`, `w_VTI_target = 0`, all base assets at their original weights.
The initial portfolio is structurally unchanged — no spurious rebalancing on day 1.

**As `m` grows:** LEAPS target shrinks; VTI target grows from zero; base targets expand to absorb
the remaining freed weight.

**Sum invariant:** `w_LEAPS + w_freed × vti_alpha + (original_base + w_freed × (1 − vti_alpha)) = w0 + original_base = 1.0` at all `m`.

DRIFT fires when any weight drifts beyond `DRIFT_BAND_RELATIVE` of its *current* dynamic target.
Because VTI starts with target weight 0 and realized weight 0, it never triggers a spurious
rebalance at `m = 1`. It enters the portfolio naturally as its target weight grows above zero
and DRIFT routes proceeds from overweight assets toward it.

Rebalancing is fully symmetric (two-sided): LEAPS can be topped up or trimmed, VTI can be
bought or sold, base assets rebalance normally. The glide path's de-levering effect comes
from the *changing target*, not from a one-sided rule.

---

## Implementation Components

The backtest engine has been refactored since this document was first drafted. `portfolio.py`
is now a thin dispatcher; all per-day logic lives in `_backtest_steps.py`, and shared frozen
dataclasses live in `_portfolio_types.py`. All implementation references below reflect that
structure.

### Architecture: DRIFT refinement, not a separate rule

The glide path is a *modifier on the existing DRIFT strategy*, not a replacement for it.
`RebalanceRule.DRIFT` is unchanged. The presence of `glide_path_config: GlidepathConfig`
on `PortfolioConfig` activates two additional behaviors:

1. **Monthly target weight update** — on each `is_month_end`, recompute `dynamic_target_weights`
   from `w_LEAPS(m)` and store them on `PortfolioState`. The DRIFT check always uses
   `state.dynamic_target_weights` (not the fixed `config.target_weights`) when a
   `glide_path_config` is present.

2. **VTI as a rebalance target** — VTI enters `PortfolioConfig.target_weights` with its
   initial target weight `0.0`. As `m` grows, its dynamic target increases and DRIFT
   routes capital toward it naturally. No harvest-only special-casing is needed; VTI is
   a fully symmetrically rebalanced asset like any other.

This keeps all DRIFT mechanics — two-sided rebalancing, drift band checks, proceeds
redistribution — entirely unchanged. The only new logic is the schedule that updates targets.

### 1. `GlidepathConfig` dataclass (`_portfolio_types.py`)

```python
@dataclass(frozen=True)
class GlidepathConfig:
    half_life_multiple: float = 2.0   # NAV multiple at which active weight halves
    floor: float = 0.05               # minimum LEAPS weight
    vti_alpha: float = 0.65           # fraction of freed weight routed to VTI (0.50–0.75)
```

Placement follows `GttConfig`'s pattern: stored on `PortfolioConfig`, lives in
`_portfolio_types.py` alongside `GttConfig`. Distinct from `LeapsConfig` (pricing engine).

`drift_band` is not a `GlidepathConfig` field — it remains `DRIFT_BAND_RELATIVE` from
`consts.py`, shared by all DRIFT rebalancing. No `RebalanceRule` change is needed; the
glide path is activated by the presence of `config.glide_path_config` on a
`RebalanceRule.DRIFT` portfolio.

`vti_alpha` governs how freed LEAPS weight is split: fraction goes to VTI, remainder
expands the diversified base proportionally. Reasonable range: 0.50–0.75; default 0.65.

### 2. No new `RebalanceRule` enum value

`RebalanceRule.DRIFT` is reused unchanged. The DRIFT dispatch in `_apply_rebalance`
is augmented — not replaced — when `ctx.glide_path_config is not None`. Specifically,
`_should_rebalance` reads from `state.dynamic_target_weights` instead of `ctx.w` when a
glide path is active.

### 3. `dynamic_target_weights` field on `PortfolioState` (`_portfolio_types.py`)

```python
@dataclass(frozen=True)
class PortfolioState:
    ...
    dynamic_target_weights: pd.Series | None   # current glide-path targets; None when inactive
```

Initialized to `None` in `_build_initial_state` for non-glide-path portfolios. When
`glide_path_config` is present, initialized to the result of `compute_glide_target_weights`
at `m = initial_nav / initial_nav = 1.0` (which equals the original `config.target_weights`
— no change on day 1). Updated each month-end inside `_apply_contribution` before
`_apply_rebalance` reads it.

When `dynamic_target_weights is not None`, `_apply_rebalance` and `_should_rebalance` use
it in place of `ctx.w`. When `None`, behavior is identical to the current implementation.

### 4. Schedule function (pure, `_backtest_steps.py`)

```python
def glide_path_leaps_weight(
    m: float,
    w0: float,
    floor: float,
    half_life_multiple: float,
) -> float:
    lam = math.log(2) / half_life_multiple
    active_weight = (w0 - floor) * math.exp(-lam * max(m - 1.0, 0.0))
    return floor + active_weight
```

And the companion that computes the full weight vector:

```python
def compute_glide_target_weights(
    m: float,
    config: PortfolioConfig,
    glidepath_config: GlidepathConfig,
) -> pd.Series:
    w0 = leaps_fraction(config)           # sum of original LEAPS weights
    w_leaps = glide_path_leaps_weight(m, w0, glidepath_config.floor,
                                      glidepath_config.half_life_multiple)
    w_freed = w0 - w_leaps                # weight released from LEAPS
    alpha = glidepath_config.vti_alpha

    # Distribute freed weight: alpha → VTI, (1-alpha) → base proportionally
    original_base = {k: v for k, v in config.target_weights.items()
                     if not k.endswith(LEAPS_KEY_SUFFIX) and k != "VTI"}
    base_sum = sum(original_base.values())
    new_weights = {}
    for k in config.target_weights:
        if k.endswith(LEAPS_KEY_SUFFIX):
            new_weights[k] = w_leaps * (config.target_weights[k] / w0)
        elif k == "VTI":
            new_weights[k] = w_freed * alpha
        else:
            new_weights[k] = config.target_weights[k] + w_freed * (1 - alpha) * (
                config.target_weights[k] / base_sum
            )
    return pd.Series(new_weights)
```

Key invariants of `glide_path_leaps_weight`:
- `w(1.0) == w0` (no de-levering at break-even)
- `w(m) → floor` as `m → ∞`
- Monotone non-increasing in `m` for `m ≥ 1`

Key invariants of `compute_glide_target_weights`:
- `sum(result) == 1.0` at all `m`
- At `m == 1.0`, result equals `config.target_weights` (with `VTI = 0.0` added)
- VTI weight monotone non-decreasing in `m`
- All base weights monotone non-decreasing in `m`

### 5. `hurdle_contributed` field on `PortfolioState` (`_portfolio_types.py`)

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
reads it. On the same month-end, `dynamic_target_weights` is also updated:

```python
# Inside _apply_contribution, on is_month_end when glide_path_config is present:
monthly_rf = (1.0 + inputs.rfr) ** (1.0 / 12.0)
new_hurdle = state.hurdle_contributed * monthly_rf + ctx.config.monthly_contribution
m = total_nav / new_hurdle
new_targets = compute_glide_target_weights(m, ctx.config, ctx.config.glide_path_config)
# return state with both fields updated
```

Both `hurdle_contributed` and `dynamic_target_weights` are updated atomically in the same
`dataclasses.replace` call before `_apply_rebalance` reads them. Step order is load-bearing.

**Prerequisite — `rfr_series` population in `_build_context`:** Currently `ctx.rfr_series` is
only populated when `use_leaps` is True. The glide path requires it unconditionally when
`config.glide_path_config is not None`. `_build_context` must populate `rfr_series` whenever
`glide_path_config` is present, regardless of `use_leaps`. The existing forward-fill handles
weekend gaps; `_build_context`'s validation block must assert that `rfr_series` has no leading
NaNs before the backtest start date.

**VTI in `target_weights` and `return_data`:** `PortfolioConfig.target_weights` must include
`"VTI": 0.0` as an explicit entry when glide path is active. `_build_context` must assert
`"VTI"` is present in `return_data.returns.columns`. VTI return data is already available
as the LEAPS underlying when `use_leaps` is True — no new data fetch required.

### 6. VTI as a fully rebalanced asset

VTI enters `PortfolioConfig.target_weights` with initial weight `0.0`. `_build_context`
includes it in `base_assets` (it is not a LEAPS key). `_build_initial_state` initializes
`holdings["VTI"] = 0.0`. No special-casing is needed: VTI is a standard holding that
`_apply_returns` compounds daily, and `_apply_rebalance` rebalances toward its current
dynamic target weight like any other asset.

At `m = 1.0`, `dynamic_target_weights["VTI"] = 0.0` — DRIFT never fires on VTI and its
realized weight matches its target. As `m` grows, the VTI target grows above zero and DRIFT
naturally routes overweight assets toward it when the drift band is breached.

The only `_build_context` addition: assert `"VTI"` is present in
`return_data.returns.columns` when `glide_path_config` is set.

### 7. Rebalance logic (`_apply_rebalance`, `_backtest_steps.py`)

`_apply_rebalance` is unchanged structurally. The DRIFT path's only modification is the
source of the target weights used in `_should_rebalance` and in the redistribution step:

```python
# Current (no glide path):
target_w = ctx.w

# With glide path active:
target_w = state.dynamic_target_weights if state.dynamic_target_weights is not None else ctx.w
```

`_should_rebalance` and the redistribution both read from `target_w`. The existing
`leaps_scale` mechanism, partial-close pattern, and all other DRIFT logic are reused
unchanged. LEAPS can be topped up or trimmed as any other asset — the glide path's
de-levering comes entirely from the monthly target update, not from a directional rule.

**GTT re-entry seeding:** `_apply_gtt_reentry` currently seeds the fresh LEAPS simulation
with `total * ctx.leaps_fraction` and reallocates base at `(1 − leaps_fraction) × base_target_w`.
When glide path is active it must instead:

1. Compute `m_current = total / state.hurdle_contributed` at the re-entry date.
2. Call `compute_glide_target_weights(m_current, ...)` to get current dynamic targets.
3. Seed LEAPS with `dynamic_targets["VTI_LEAPS"] × total` (or the LEAPS-key equivalent).
4. Allocate each base asset (including VTI) at `dynamic_targets[a] × total`.

VTI is treated identically to any other base asset here — it receives its dynamic target
allocation from the full NAV, which may be more or less than its pre-defensive balance.
This is correct: the re-entry is a full rebalance to current targets, not a preservation
of prior VTI balances.

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Index variable | NAV / Rf-hurdle-adjusted contributed capital | De-levering only begins after clearing the risk-free bar; drawdowns self-correct; defensive clock pauses |
| Rf source | `^IRX` (existing `TBILL_TICKER`) | Consistent with GTT defensive earnings; no new data dependency; time-varying captures ZIRP era correctly |
| Functional form | Exponential decay | One parameter; sigmoid slow-start is redundant with NAV-multiple indexing |
| Half-life | 2× NAV multiple (~10 yr) | Aligns with market doubling time at historical returns |
| Floor | Fixed weight, 5–10% | Prevents strategy from fully abandoning leverage |
| Rebalance rule | `RebalanceRule.DRIFT`, unmodified | Glide path is a refinement of DRIFT, not a separate strategy; all rebalancing mechanics reused |
| Symmetry | Two-sided DRIFT | De-levering comes from changing targets, not directional rules; LEAPS can be topped up or trimmed |
| Target weight update | Monthly, on `is_month_end` | Same cadence as contributions; keeps `hurdle_contributed` and dynamic targets atomically consistent |
| Dynamic target carrier | `PortfolioState.dynamic_target_weights` | Follows existing frozen-state pattern; `None` when glide path inactive; replaces `ctx.w` in DRIFT check |
| Parameters | `GlidepathConfig` dataclass | Decoupled from `LeapsConfig` (pricing) and `PortfolioConfig` (initial weights) |
| Drift band | Reuse `DRIFT_BAND_RELATIVE` from `consts.py` | Shared by all DRIFT rebalancing; not a `GlidepathConfig` field |
| Proceeds split | Constant `vti_alpha` (default 0.65) | Structural preference; `w(m)` provides all m-adaptive behavior; separation of concerns |
| VTI in portfolio | Standard rebalanced asset, initial weight 0.0 | Clean: no special-casing; DRIFT routes capital to VTI naturally as its target grows |
| Absolute ceiling | None | Strategy is multigenerational / no terminal date; weight floor is the correct risk control |

---


