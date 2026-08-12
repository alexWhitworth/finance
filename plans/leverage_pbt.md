# Project: Hypothesis PBT Assessment — `leverage.py`

---

## 1. System Overview

`leverage.py` is a **pure-function financial computation module** — the ideal target for
property-based testing. Pure functions with well-defined mathematical invariants expose the
maximum surface area for Hypothesis to explore, and the module has a natural three-layer
structure that maps cleanly onto PBT strategies:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 3 — Simulation Orchestrator                                      │
│  run_leaps_simulation → LeapsLedger                                     │
│  [temporal invariants, ledger consistency, accounting closure]          │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 2 — LEAPS Lifecycle                                              │
│  create / price / roll / partial_close / gtt_close / get_live           │
│  [cost-basis conservation, tax accounting, contract state machine]      │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 1 — Black-Scholes Pricing Engine                                 │
│  price / delta / gamma / vega / theta / charm / vanna                  │
│  [mathematical identities, monotonicity, no-arbitrage bounds]           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Current PBT posture:** 6 property tests exist, all concentrated in Layer 1 (4 greek-sign
tests) and the lower half of Layer 2 (`get_live_contracts` expiry invariant + no-lookahead).
**Layer 2 lifecycle accounting and Layer 3 simulation are entirely uncovered by PBT.**

---

## 2. Tech Stack & Dependencies

| Component | Tool | Note |
|---|---|---|
| PBT framework | `hypothesis` | Already in use; `@given` + `@settings` pattern established |
| Strategy primitives | `st.floats`, `st.integers`, `st.builds` | Bounds must avoid degenerate domains |
| Temporal strategies | `st.integers` → `pd.Timestamp` offsets | Pattern already used in test file |
| Composite strategies | `@st.composite` | Required for valid `LeapsContract` generation |
| Numerical oracle | `scipy.stats.norm` | Available; used in `bs_call_price` internals |

---

## 3. Gap Analysis — Uncovered Invariants

### Layer 1 — Black-Scholes (4 gaps)

| Invariant | Function | Currently Tested? | PBT Value |
|---|---|---|---|
| Price positivity: `price >= 0` | `bs_call_price` | Unit test (1 case) | **Low** — trivial given other proofs |
| Delta bounded `(0, 1)` | `bs_call_delta` | Unit test (9 cases) | **Medium** — property confirms no edge-case escape |
| Delta = dPrice/dSpot (FD oracle) | `bs_call_delta` | None | **HIGH** — validates BS implementation correctness |
| Vega = dPrice/dIV (FD oracle) | `bs_call_vega` | None | **HIGH** — validates vega formula directly |
| Put-call parity with `q > 0` | `bs_call_price` | Unit test only at `q=0` | **Medium** — `q > 0` branch is live in production |
| Vanna finite for all inputs | `bs_call_vanna` | None | **Low** — similar to charm; unlikely to break |

### Layer 2 — LEAPS Lifecycle (5 gaps)

| Invariant | Function | Currently Tested? | PBT Value |
|---|---|---|---|
| Cost-basis conservation: `premium * multiplier * n == capital` | `create_leaps_contract` | Unit test (1 case) | **HIGH** — core accounting identity |
| Roll accounting: `net_proceeds == old_value - tax_paid` | `roll_contract` | Unit test (1 case) | **HIGH** — tax branch logic has 3 paths |
| Tax non-negativity: `tax_paid >= 0` always | `roll_contract` | None | **Medium** — `max(0, gain)` can be tricky under float cancellation |
| Partial close contract conservation: `cont.n + n_closed == orig.n` | `partial_close_leaps` | Unit test (1 case) | **HIGH** — floating-point drift risk |
| `price_leaps_contract` positivity | `price_leaps_contract` | Unit test (1 case) | **Low** — follows from BS call price positivity |

### Layer 3 — Simulation (3 gaps)

| Invariant | Function | Currently Tested? | PBT Value |
|---|---|---|---|
| Ledger referential integrity: every roll event's `old_contract ∈ contracts` | `run_leaps_simulation` | 1 deterministic test (fixed seed) | **HIGH** — seed-specific test is not a property |
| TAX_SHELTERED → total_tax = 0 over arbitrary price path | `run_leaps_simulation` | 1 deterministic test (fixed seed) | **Medium** — seed covers one path; property covers all |
| No-lookahead: live contracts at T contain no contracts with `purchase_date > T` | `run_leaps_simulation` | None at simulation level | **Medium** — lower priority given Layer 2 coverage |

---

## 4. Component/Module Breakdown — New Tests API

### Phase 1a — `test_roll_contract_accounting_property`

```python
@given(
    spot_buy=st.floats(50.0, 500.0, allow_nan=False, allow_infinity=False,
                       allow_subnormal=False),
    spot_sell=st.floats(50.0, 500.0, allow_nan=False, allow_infinity=False,
                        allow_subnormal=False),
    capital=st.floats(1_000.0, 500_000.0, allow_nan=False, allow_infinity=False,
                      allow_subnormal=False),
    account_type=st.sampled_from(list(AccountType)),
)
@settings(max_examples=300)
def test_roll_contract_accounting_property(
    spot_buy, spot_sell, capital, account_type
) -> None:
    # Invariants:
    #   event.net_proceeds == price_leaps_contract(old, spot_sell, roll_date) - event.tax_paid
    #   event.tax_paid >= 0
    #   account_type == TAX_SHELTERED → event.tax_paid == 0
```

**Inputs:** Valid (spot_buy, capital) that produce `n_contracts > 0` (use `assume`).
**Outputs:** `LeapsRollEvent` with verified accounting closure.
**Depends on:** `create_leaps_contract`, `roll_contract`, `price_leaps_contract`.

---

### Phase 1b — `test_bs_delta_finite_difference_oracle`

```python
@given(
    spot=st.floats(1.0, 1_000.0, allow_nan=False, allow_infinity=False,
                   allow_subnormal=False),
    strike=st.floats(1.0, 1_000.0, allow_nan=False, allow_infinity=False,
                     allow_subnormal=False),
    t=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False,
                allow_subnormal=False),
    iv=st.floats(min_value=0.01, max_value=1.5, allow_nan=False, allow_infinity=False,
                 allow_subnormal=False),
    q=st.floats(min_value=0.0, max_value=0.10, allow_nan=False, allow_infinity=False,
                allow_subnormal=False),
)
@settings(max_examples=200)
def test_bs_delta_finite_difference_oracle(
    spot, strike, t, iv, q
) -> None:
    # Invariant:
    #   |delta_analytic - (price(S+ε) - price(S-ε)) / 2ε| < 1e-5
    # epsilon = 1e-4 * spot (relative step for numerical stability)
```

**Inputs:** Valid BS domain with `T >= 0.01` (avoids TIME_FLOOR numerical instability).
**Outputs:** Tolerance check against central finite difference.
**Depends on:** `bs_call_delta`, `bs_call_price`.

---

### Phase 1c — `test_partial_close_contract_conservation`

```python
@given(
    spot=st.floats(50.0, 500.0, allow_nan=False, allow_infinity=False,
                   allow_subnormal=False),
    capital=st.floats(1_000.0, 200_000.0, allow_nan=False, allow_infinity=False,
                      allow_subnormal=False),
    scale=st.floats(min_value=0.01, max_value=0.99, allow_nan=False,
                    allow_infinity=False, allow_subnormal=False),
)
@settings(max_examples=200)
def test_partial_close_contract_conservation(
    spot, capital, scale
) -> None:
    # Invariants:
    #   cont.n_contracts + n_contracts_closed == approx(original.n_contracts, rel=1e-9)
    #   cont.n_contracts > 0
    #   n_contracts_closed > 0
```

**Inputs:** Valid contract + `scale` fraction (drives `target_value = current_mtm * scale`).
**Outputs:** `LeapsPartialCloseEvent` with verified contract count conservation.
**Depends on:** `create_leaps_contract`, `price_leaps_contract`, `partial_close_leaps`.

---

### Phase 1d — `test_create_contract_cost_basis_conservation`

```python
@given(
    spot=st.floats(10.0, 1_000.0, allow_nan=False, allow_infinity=False,
                   allow_subnormal=False),
    capital=st.floats(100.0, 1_000_000.0, allow_nan=False, allow_infinity=False,
                      allow_subnormal=False),
    iv=st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False,
                 allow_subnormal=False),
)
@settings(max_examples=200)
def test_create_contract_cost_basis_conservation(
    spot, capital, iv
) -> None:
    # Invariant (when premium above floor):
    #   c.premium_paid * CONTRACT_MULTIPLIER * c.n_contracts == approx(capital, rel=1e-9)
    # Guard:
    #   assume(c.n_contracts > 0)
```

**Inputs:** Valid BS inputs that reliably produce premium above `MIN_PREMIUM_PER_SHARE`.
**Outputs:** Verified that capital deployment is lossless.
**Depends on:** `create_leaps_contract`.

---

### Phase 2 — `test_simulation_ledger_referential_integrity` *(deferred)*

```python
@given(
    returns=st.lists(
        st.floats(min_value=-0.05, max_value=0.10, allow_nan=False,
                  allow_infinity=False, allow_subnormal=False),
        min_size=21 * 6,   # ~6 months
        max_size=21 * 48,  # ~48 months
    ),
    contribution=st.floats(1_000.0, 50_000.0, allow_nan=False, allow_infinity=False,
                            allow_subnormal=False),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_simulation_ledger_referential_integrity(
    returns, contribution
) -> None:
    # Invariant:
    #   {ev.old_contract for ev in ledger.roll_events} ⊆ set(ledger.contracts)
```

**Inputs:** Synthetic price series built from compounded daily returns.
**Outputs:** Ledger with verified referential integrity.
**Depends on:** `run_leaps_simulation`.
**Note:** Runtime ~50–100ms per example; use `max_examples=50`.

---

## 5. Step-by-Step Implementation Roadmap

### Phase 1a — Roll accounting property ✦ highest value, lowest cost

- **Input:** Composite strategy: `(spot_buy, capital, account_type)` → `LeapsContract`; `spot_sell` → roll pricing
- **Guard:** `assume(c.n_contracts > 0)` to skip `MIN_PREMIUM_PER_SHARE` floor cases
- **Assertions:** (1) net_proceeds identity; (2) tax_paid >= 0; (3) sheltered → tax = 0
- **Completion signal:** Passes `max_examples=300` with no failures

### Phase 1b — Delta finite-difference oracle

- **Input:** Valid BS domain; epsilon = `1e-4 * spot` (relative step)
- **Bound T >= 0.01:** Precision degrades near `TIME_FLOOR` even above the floor
- **Bound iv in [0.01, 1.5]:** Avoids overflow in `exp(iv^2 * T)` for large T
- **Assertions:** `|delta_analytic - FD| < 1e-5`
- **Completion signal:** FD error < `1e-5` at `max_examples=200`

### Phase 1c — Partial close conservation

- **Input:** Valid contract + `scale ∈ (0.01, 0.99)`; derive `target_value = current_mtm * scale`
- **Guard:** `assume(current_mtm > 0)` (follows from BS positivity; defensive filter)
- **Assertions:** `cont.n + n_closed ≈ orig.n` with `rel=1e-9`
- **Completion signal:** Passes `max_examples=200`

### Phase 1d — Cost basis conservation

- **Input:** `(spot, capital, iv)` with `iv >= 0.05` to reliably avoid premium floor
- **Guard:** `assume(c.n_contracts > 0)` — still needed for edge-case spot/iv combos
- **Assertions:** `premium_paid * MULTIPLIER * n_contracts ≈ capital` with `rel=1e-9`
- **Completion signal:** Passes `max_examples=200`

### Phase 2 — Simulation ledger integrity *(deferred; implement after Phase 1)*

- **Input:** `st.lists(st.floats(-0.05, 0.10))` → compounded `pd.Series`
- **Runtime control:** `max_examples=50`, `suppress_health_check=[HealthCheck.too_slow]`
- **Assertions:** Referential integrity of `roll_events` vs `contracts`
- **Completion signal:** Passes with no integrity violations across all generated series

---

## 6. System & Test Invariants

### [ATOMIC] Mathematical Invariants (Layer 1)

| Invariant | Oracle | Falsifiability |
|---|---|---|
| `delta ∈ (0, 1)` for all valid inputs | Mathematical proof (N(d1) ∈ (0,1)) | Escape at boundary: very deep OTM near T_FLOOR |
| `gamma >= 0` | N'(d1) >= 0 always | Cannot be negative; float underflow to 0 acceptable |
| `vega >= 0` | S * N'(d1) * sqrt(T) >= 0 | Same underflow case |
| `theta <= 0` at r=q=0 | Time decay theorem | Positive theta at r=q=0 is a bug |
| `delta ≈ FD(price, S)` | `scipy.stats.norm` independent oracle | Tolerance: `1e-5` per share |

### [ATOMIC] Accounting Invariants (Layer 2)

| Invariant | Oracle | Temporal Bound |
|---|---|---|
| `net_proceeds = mtm - tax_paid` | `price_leaps_contract` is the independent oracle | At roll date only |
| `tax_paid = max(0, gain) * ltcg_rate` | Closed-form `max(0, gain) * rate` | TAXABLE only |
| `tax_paid = 0` | Unconditional | TAX_SHELTERED, any date |
| `cont.n + n_closed = orig.n` | Floating-point identity | At partial close execution date |
| `cost_basis = capital` | Input value | At creation date |

### [ATOMIC] State-Machine Invariants (Layer 2)

| Invariant | Oracle | Falsifiability Condition |
|---|---|---|
| Rolled contract not live after roll date | `roll_events` set | `old_contract in get_live_contracts(ledger, roll_date + 1day)` → bug |
| Future roll event does not affect past live set | Temporal filter `<= current_date` | Already covered by existing PBT |
| GTT-closed contract not live at or after close date | `gtt_close_events` set | Already covered by existing PBT |

### [INTEGRATION] Simulation-Level Invariants (Layer 3)

| Invariant | Tag | Note |
|---|---|---|
| Ledger referential integrity | `[INTEGRATION]` | Requires full simulation run |
| TAX_SHELTERED → `sum(tax_paid) = 0` over any price path | `[INTEGRATION]` | Cannot be verified at unit level |
| No future contracts visible before creation date | `[INTEGRATION]` | Requires temporal simulation |

---

## 7. Known Assumptions

1. **`TIME_FLOOR` guards all BS inputs:** The module floors `time_to_expiry` at `TIME_FLOOR`,
   so Hypothesis strategies must align — use `min_value=TIME_FLOOR` (or higher for FD oracle
   tests) rather than `min_value=0.0`.

2. **Floating-point underflow to 0 is valid for greek tests:** `gamma >= 0.0` and `vega >= 0.0`
   correctly pass at `0.0` (deep OTM + low IV); this is not a bug.

3. **`MIN_PREMIUM_PER_SHARE` guard breaks cost-basis conservation:** When the premium is floored,
   `n_contracts = 0` and the identity no longer holds. Phases 1a and 1d must filter these inputs
   with `assume(c.n_contracts > 0)`.

4. **Float arithmetic in partial close:** `n * scale + n * (1 - scale) == n` is not guaranteed
   bit-exact under IEEE 754. The tolerance for Phase 1c must be `rel=1e-9`, not exact equality.

5. **Simulation runtime:** `run_leaps_simulation` over a 48-month series takes ~50–100ms. Phase 2
   must use `settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])`.

6. **`allow_subnormal=False` on all float strategies:** Subnormal floats can trigger spurious
   underflow in `math.exp(-q*T)` and `math.log(S/K)`. All strategies should set
   `allow_subnormal=False`.

---

## 8. Potential Edge Cases & Pre-Mortem Risks

### Anti-patterns in Strategy Design

| Risk | Manifestation | Mitigation |
|---|---|---|
| `iv=0` or near-zero | Division by zero in `iv * math.sqrt(T)` denominator | `min_value=0.01` on all iv strategies |
| `spot / strike` extreme ratio (>100x) | `math.log(S/K)` overflow at `float` limits | Bound both to `[1.0, 10_000.0]`; ratio implicitly bounded |
| `T` near `TIME_FLOOR` | Numerical instability in `2*T*iv*sqrt(T)` denominator in charm | Use `min_value=TIME_FLOOR * 2` for charm/theta FD tests |
| `capital = 0` | `n_contracts = 0`, trivially passing accounting tests | `assume(capital > 100.0)` or floor at `100.0` in strategy |
| Hypothesis generating `float` subnormals | Spurious underflow in `math.exp(-q*T)` | `allow_subnormal=False` on all float strategies |
| Overlapping `purchase_date` and `current_date` | `hold_days = 0` → valid state, `should_roll = False` | Assert correctly; not a degenerate case |
| `pd.DateOffset(years=2)` on Feb 29 | Leap year edge in expiry calculation | Out of scope for Hypothesis; cover with 1 deterministic test |

### Diminishing Returns Analysis

The following areas have **adequate deterministic coverage** where PBT would add minimal value:

| Function | Reason PBT adds little value |
|---|---|
| `should_roll` | Three boolean conditions; all 8 combinations covered deterministically |
| `compute_terminal_nav` | Accounting identity tested across all 4 key scenarios |
| `compute_leaps_tax_summary` | Deterministic aggregation; correctness depends on roll event values |
| `bs_call_vanna` | Finiteness property is low-value given charm coverage |
| `run_leaps_simulation` IV/RFR alignment | Deterministic alignment tests cover the `ffill` logic exhaustively |

---

## Summary Recommendation

**Current state:** The existing 6 property tests are well-targeted. The gap is not quantity —
it is **layer coverage**. Layers 2 and 3 are entirely absent from PBT.

**Recommended action:** Implement Phases 1a–1d. These 4 tests add ~75 lines of test code and
close the highest-value gaps at low implementation cost. Phase 2 is optional — the fixed-seed
deterministic test already provides meaningful coverage for the common simulation case.

**Expected outcome after Phases 1a–1d:** Property test suite grows from 6 → 10 tests, covering
all three layers. Remaining gaps (delta FD oracle, simulation integrity) can be deferred.

> The optimal mix for this module is approximately **8–10 property tests** targeting
> mathematical and accounting identities, supported by **~120+ deterministic tests** for
> behavioral correctness. The current 6:130 ratio underweights Layer 2 PBT relative to the
> accounting risk that layer carries.
