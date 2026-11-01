---
project: "finance"
author: "systems-architect"
last_updated: "2026-11-01"
links:
  - plans/portfolio_management.md
---

## 1. Summary

The library prices 2-year DITM LEAPS (strike = 0.5·spot, k = ln(K/F) ≈ −0.69) with a flat scalar IV — the 30-day vol index level — so live marks fall outside market-maker bid-ask spreads and reported vega overstates true vol sensitivity ~3×. We add a parameterized implied-volatility surface: SSVI (Gatheral & Jacquier 2014) with the ATM total-variance curve θ(t) driven by the vol index through a mean-reverting (OU) variance law, and per-ticker frozen shape parameters (ρ, η, γ) from literature priors, validated one-time against manually entered broker quotes. Carry inputs are fixed in the same pass: r(T) from the zero curve (`^IRX` short end, DGS2 for ~2y, via `fetch_2yr_treasury`), forward dividend yield q, and the `^MOVE` basis-point conversion bug in `data.py`. Biggest open risk: stylized shape parameters with no chain access — mitigated by the smile's small mark impact at DITM k (≤ ~0.4% calm) and a validation harness that pins (ρ, η) from a handful of broker-screen quotes. Reader: approve the parameterization, the back-compat flag, and the sticky-moneyness greeks convention.

## 2. Scope

Profile: library · Layers: python

**Goals**

- Marks, greeks, and roll logic for 2y DITM LEAPS priced from a term-structure-and-smile surface, not a scalar IV.
- Live portfolio management (`portfolio_manager.py`, priority) and backtest (`_backtest_steps.py`, secondary) consume the same surface.
- Extrinsic-to-extrinsic validation against real bid-ask, replacing model-price-vs-mid comparisons.
- Fix `^MOVE` → IV conversion (basis points, not percent) and wire maturity-matched r(T), forward q.

**Non-goals**

- Calibrating to option chains at runtime (no chain access; no solver in the daily loop) — D-3.
- Stochastic-vol model engines (Heston/rough-vol CF pricing) — rejected in Options; Forde & Jacquier (2011) shows the Heston smile flattens at long maturity, the wrong regime for LEAPS.
- American exercise adjustment — D-10.
- Bond-ETF LEAPS trading on the fixed MOVE-derived IV (no `*_LEAPS` positions in scope; fix is for marks/greeks hygiene only).

**Scale & constraints.** 20-year daily backtest (~5,000 days × 6 tickers) must price in closed form; determinism and bit-for-bit back-compat under the flat flag are hard constraints. Primary sources: Gatheral (2004); Gatheral & Jacquier (2014); Gatheral (2006); Bergomi (2016). Complementary: Lee (2004) moment formula; Derman (1999) sticky regimes; Cont & da Fonseca (2002) surface PCA; Demeterfi–Derman–Kamal–Zou (1999) and Carr & Wu (2009) VIX-as-variance-strike; Zeliade (2009) SVI calibration; Jäckel (2015) IV inversion; Gatheral–Jacquier–Rosenbaum (2018) and Bayer–Friz–Gatheral (2016) large-maturity checks. Hagan et al. (2002) SABR is cited only to reject it (expansion error explodes at k ≈ −0.7).

## 3. Design

**Options considered**

| Option | Pros | Cons | Verdict |
| --- | --- | --- | --- |
| Term-structure fix only (θ(t), flat smile) | Smallest change; fixes the 3× vega error | Leaves smile and GLD skew *sign* wrong | ❌ |
| **SSVI surface (Gatheral & Jacquier 2014)** | Term structure *is* the free parameter θ(t); no-arb conditions are closed-form checks; pure closed-form pricing; per-ticker shape handles GLD's opposite skew | Shape params stylized, not fitted | ✅ chosen |
| Chain-calibrated raw SVI (Zeliade 2009) | Fitted, not stylized | Needs chain pipeline; still needs SSVI extrapolation for backtest; over-engineering | ❌ later (D-3) |

**Architecture**

```
data.py ──vol_index_series──▶ vol_surface.py ──σ(k,t)──▶ leverage.py (price_leaps_contract)
   │  ^IRX, DGS2 (r(T))            θ(t) OU curve                    greeks.py (Δ_sm, level-vega)
   │  MOVE × D_eff (fixed)         SSVI w(k,t)                      portfolio_manager.py ◀─ priority
   └─ dividends (q_forward)        no-arb checker + Lee caps         _backtest_steps.py ◀─ secondary (flag)
                                        │
                        validation/ manual_quotes.csv ── extrinsic-vs-extrinsic harness
```

- `vol_surface.py` (new): owns the parameterization. Reads vol index level, spot, strike, t; writes σ(k,t) and diagnostics. Never does I/O. Boundary: no pandas, no network.
- `data.py`: owns fetch + unit conversion (`fetch_2yr_treasury` already at data.py:258). Per-index conversion table replaces uniform ÷100 (D-7).
- `leverage.py` / `greeks.py`: consume `σ(k,t)` where they currently take scalar `iv`; keep scalar fallback.
- `_backtest_steps.py`: per-day θ_t replaces `mtm_iv_series` and `max(raw_vix, ctx.iv)` floors, behind `vol_model` flag (D-4).
- `validation/` harness: one-time scripts + manually entered quotes CSV; not a library module.

**Key rules (D-ids)**

- **D-1 — SSVI in total-variance space, log-moneyness k = ln(K/F).** w(k,t) = (θ_t/2)·{1 + ρ·φ(θ_t)·k + √((φ(θ_t)·k + ρ)² + 1 − ρ²)}, φ(θ) = η·θ^(−γ)·(1+θ)^(γ−1), σ(k,t) = √(w(k,t)/t); θ_t is ATM total variance. Why: variances add across the term structure and the no-arb conditions are stated in w-space; k = ln(K/F) (forward, not spot) makes the surface drift- and carry-consistent. Raw SVI (Gatheral 2004) remains the documented refit target if chains ever appear.
- **D-2 — θ(t) from an OU variance law anchored to the vol index.** v₀ = (vol index level)² (Demeterfi et al. 1999: VIX² ≈ 30-day variance strike; Carr & Wu 2009 for the VRP it embeds); θ(τ) = σ∞²τ + (v₀ − σ∞²)·(1 − e^(−κτ))/κ with frozen (κ, σ∞). Why: fixes the two structure errors — 30-day level ≠ 2-year level (∂θ(2y)/∂v₀ ≈ 0.63 at κ = 1.5 vs ≈ 0.08 at 30d, i.e., current marks overstate vol P&L sensitivity ~3×), and the 30-day smoothing window becomes unnecessary (mean reversion damps it).
- **D-3 — shape parameters are frozen per ticker, from literature priors; no runtime calibration.** (ρ, η, γ) per underlying in `consts.py`; equity indices ρ < 0 (put skew), GLD ρ > 0-ish (call skew — opposite sign). Zeliade-style refit is a documented extension path, never in the daily loop.
- **D-4 — `vol_model: "flat" | "ssvi"` config flag; flat is the default.** Existing backtests reproduce bit-for-bit; SSVI is opt-in per run. The flag selects θ feeding: flat = current scalar logic, ssvi = per-day θ_t.
- **D-5 — greeks report sticky-moneyness delta as headline; sticky-strike (plain BSM) exposed via a `vol_convention` flag.** Δ_sm = Δ_BS − (vega/S)·∂σ/∂k. Why: the surface is parameterized in k, so sticky-moneyness is the internally consistent regime (Derman 1999); Cont & da Fonseca (2002) PCA supports level-dominated moves. Correction is small (~+0.004 at k = −0.69) but it makes the convention explicit rather than implicit.
- **D-6 — headline vega is level-vega, ∂C/∂θ (via ∂C/∂v₀), not ∂C/∂σ(k,t).** The mark moves with the vol index through θ; reporting local-vega at fixed shape is the ~3× overstatement in a single number.
- **D-7 — per-index unit conversion replaces uniform ÷100.** Vol indices quoted in percent points (^VIX etc.) keep ÷100; ^MOVE is quoted in basis points → σ ≈ D_eff · MOVE·10⁻⁴ with per-ETF effective duration (D-8 defaults: VGIT ≈ 5.1, MUB ≈ 5.5). Currently MOVE/100 yields 60–140% "IV", ~6–20× too high.
- **D-8 — r(T) maturity-matched zero curve; q as forward yield.** Short end from `^IRX`, ~2y from DGS2 (`fetch_2yr_treasury`); q grows trailing yield to expiry. Why: both r appearances in BSM are rates *to expiry*; 13-week bill for a 2-year option is a term mismatch worth ≈ $0.92/share per 100bp of 13w−2y spread (ρ ≈ K·T), historically ±100bp → up to ~1.8% of mark. Par-vs-zero gap at 2y is a few bp — accepted (U-5).
- **D-9 — arbitrage enforcement is constructive, not fitted.** Calendar: θ(t) nondecreasing holds by the OU form (both terms ≥ 0 for τ ≥ 0 given v₀, σ∞² > 0) — no daily check. Butterfly: θ_t·φ(θ_t)·(1+|ρ|) < 4 and θ_t·φ(θ_t)²·(1+|ρ|) ≤ 4 asserted at the frozen params and the historical worst-case θ (VIX 80 → v₀ = 0.64). Wings: Lee (2004) caps — total-variance wing slope ≤ 4 — bound the k ≈ −0.7 extrapolation where no quotes exist.
- **D-10 — European exercise only.** r > q today makes the premium ~bp; Merton (1973): q = 0 ⇒ American = European exactly (GLD). Backtest ZIRP-era understatement accepted (U-3).

**Data model.** `SsviParams(rho, eta, gamma, kappa, sigma_inf)` — frozen dataclass, one per underlying ticker, in `consts.py`; keyed lookup, immutable. Per-day θ_t is derived, never persisted. No new stored state.

## 4. Invariants

| ID | Tag | Property | Verification |
| --- | --- | --- | --- |
| INV-1 | ATOMIC | Same inputs ⇒ same σ(k,t), no solver, no randomness in any pricing path | Property tests on pure functions |
| INV-2 | ATOMIC | Butterfly conditions hold for frozen params at historical worst-case θ | pytest fixture (VIX 80 case) |
| INV-3 | ATOMIC | Mark plausibility: parity bounds, C ∈ [max(0, S e^(−qT) − K e^(−r(T)T)), S e^(−qT)] | pytest + harness vs broker quotes |
| INV-4 | ATOMIC | 2y ATM IV plausibility: [12%, 45%]; calm ≈ 16–21%, VIX 40+ crisis ≈ 30–35% (not VIX level) | pytest bounds + harness table |
| INV-5 | INTEGRATION | `vol_model="flat"` reproduces current marks bit-for-bit | Regression: run old vs new on fixture, compare |
| INV-6 | INTEGRATION | No lookahead: day-t backtest mark uses only vol-index/r/q series ≤ t | Backtest integration test |
| INV-7 | INTEGRATION | Extrinsic-to-extrinsic: model extrinsic within observed half-spread on validation quotes when carry inputs are right | One-time harness output, human-reviewed |

## 5. Decisions & unknowns

**Decisions (user-approved): D-1 … D-10 as listed; plus:** no chain access — validation via manually entered broker quotes (m00045); European-only (m00045); MOVE fix in scope (m00045); sticky-moneyness headline + sticky-strike exposed (m00049); r(T) via 2y treasury, `fetch_2yr_treasury` (m00052, m00058); flat-flag default (D-4, m00051).

**Unknowns**

| ID | Statement | Sev | Closes | If wrong |
| --- | --- | --- | --- | --- |
| U-1 | Stylized (ρ, η, γ): marks at DITM k off ≤ ~0.4% calm; greeks inherit shape error in full | H | Phase 5 harness vs manual quotes pins (ρ, η) coarsely | Wrong GLD skew sign → mispriced GLD DITM marks in the *opposite* direction |
| U-2 | κ, σ∞ estimated, not observed; κ off ±0.5 ⇒ ~±1 vol pt at 2y | M | Phase 5 sensitivity table | Term-structure level shift, mostly common across rolls |
| U-3 | European marks understate dealer prices ~1–3% in ZIRP-era backtest window | L | Accepted; documented | Backtest LEAPS returns slightly conservative pre-2022 |
| U-4 | Marks embed variance risk premium (Carr & Wu 2009) | L | Accepted; documented | Backtest vol-drag vs physical expectation — economic fact, not bug |
| U-5 | DGS2 par-coupon vs zero gap at 2y | L | Accepted; few bp | Negligible vs U-2 |
| U-6 | D_eff per bond ETF (VGIT 5.1, MUB 5.5) | L | One-time check vs listed option IVs | Bond-ETF IV marks off; no LEAPS exposure today |

## 6. Roadmap

Feature dependency DAG (F-0 … F-6; chain F-0 → F-1 → F-2 → F-3 → F-5 → F-6 is the critical path, with F-4 forking at F-1 so the backtest branch needs only the core surface module):

```
F-0  Carry wiring + MOVE fix
  └─► F-1  vol_surface.py core
        ├─► F-2  Surface-aware greeks ──► F-3  Live path (priority) ──► F-5  Validation ──► F-6  Docs
        └─► F-4  Backtest wiring (secondary)
```

| Phase | Depends on | Delivers | Done |
| --- | --- | --- | --- |
| 0. Carry wiring + MOVE fix | — | r(T) curve (`^IRX` + `fetch_2yr_treasury`), forward q, per-index conversion in `data.py` | INV-3 partial; MOVE unit test (60→6% scale) |
| 1. `vol_surface.py` core | 0 | θ(t), SSVI w(k,t), σ(k,t), no-arb checker, Lee caps | INV-1, INV-2, INV-4 |
| 2. Surface-aware pricing/greeks | 1 | `price_leaps_contract` on surface; Δ_sm / sticky-strike flag; level-vega | INV-3 |
| 3. Live portfolio path (priority) | 2 | `portfolio_manager.py` per-ticker surface config; NAV/greeks/rebalance consume it | INV-7 dry-run on current quotes |
| 4. Backtest wiring (secondary) | 1 | `vol_model` flag; per-day θ_t in `_backtest_steps.py`; updated tests | INV-5, INV-6 |
| 5. Validation | 3 | Extrinsic-vs-spread harness (`validation/`), manual-quote pinning, κ/σ∞ sensitivity table | INV-7 report, human review |
| 6. Docs + handoff | 5 | README, decision log, `plan-to-spec` input | Plan accepted by user |

Later (no phase): Zeliade-style SVI refit if chain access appears; American CRR control-variate wrapper if bond-ETF or low-rate LEAPS trading.