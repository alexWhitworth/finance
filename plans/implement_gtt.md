# Project: GTT Market Timing Strategy — Core Library Integration

## 1. System Overview

Integrate the GTT (Growth Trend Timing) market timing strategy from `analyses/gtt_eda.py` into
the core backtest library as an optional, opt-in overlay on top of existing `run_backtest` logic.

When active, GTT monitors two signals daily and, when recession risk is detected and the equity
price trend confirms, moves VTI (and VTI_LEAPS) capital into a configurable defensive allocation.
VTI shares are sold and LEAPS contracts are **closed** (realizing gains/tax); on re-entry the
capital re-buys VTI shares and **fresh** 2-year LEAPS contracts at prevailing prices. All other
assets are unaffected. Existing backtests with no GTT config are completely unchanged.

### Decision Rule (uniform t+1 execution lag on all signals)

```
recession_risk_t = UE_12M_t OR VIX_5D_t

position_t+1 =
    if not recession_risk_t              → Long (normal target weights)
    if recession_risk_t AND price_t >= SMA200_t  → Long (normal target weights)
    if recession_risk_t AND price_t <  SMA200_t  → Defensive (defensive_weights)
```

**Signal timing (both signals are stamped at the date the information is available
at close t, then executed uniformly at open t+1):**

- **UE_12M:** The monthly UNRATE print is published by the BLS Employment Situation
  release on the **first Friday of the month *following* the reference month** (e.g. the
  Jan-2026 rate prints Fri 2026-02-06). FRED indexes UNRATE at the *reference-month start*
  (`2026-01-01`), which is ~5 weeks before the print exists. `compute_ue_signal` therefore
  **re-stamps each observation to its true publication date (first Friday of the following
  month)** before the daily forward-fill, so the January value is never visible before early
  February. This closes a ~1-month look-ahead leak.
- **VIX_5D:** Uses the window `[t-4, t]`; the signal is known at close t.
- **Execution:** the shared **1-day lag inside `compute_position_mask`** carries *both*
  signals from close t → open t+1. For UE this maps the Friday publication close → the next
  trading day (Monday); for VIX it maps close t → t+1. There is a single execution lag, not a
  separate per-signal shift.

### Component Flow

```
I/O Boundary                        Pure Core
─────────────────────────────────   ────────────────────────────────────
fetch_gtt_signal_data()         →   GttSignalData (frozen)
  ├── fetch UNRATE (FRED)                │
  ├── fetch ^VIX (yfinance)             ▼
  ├── compute UE_12M signal       run_backtest(
  ├── compute VIX_5D signal         return_data,
  └── return daily position mask     price_data,
                                     config,           ← GttConfig optional
                                     gtt_signal=...    ← GttSignalData optional
                                   )
                                        │
                                        ▼
                                   BacktestResult (unchanged shape)
```

---

## 2. Tech Stack & Dependencies

- **Python 3.13**, existing library stack unchanged
- `fredapi` — already in `pyproject.toml` as a hard dependency; no change needed
- `yfinance` — already present; VIX already fetchable via `fetch_prices` / `fetch_volatility_index`
- No new external dependencies required

---

## 3. Data Schema / Type Definitions

### 3.1 New: `GttConfig` (in `portfolio.py`)

```python
@dataclass(frozen=True)
class GttConfig:
    vix_p90_threshold: float          # Fixed P90 threshold (e.g. 0.272). Caller's responsibility
                                      # to compute from desired history to avoid look-ahead.
    sma_window: int = 200             # Rolling window for equity price SMA (trading days)
    vix_consecutive_days: int = 5     # N consecutive days VIX >= threshold to fire VIX_5D
    unrate_trade_lag_days: int = 1    # Trading-day execution lag from the UNRATE publication
                                      # date (1st Friday of the month AFTER the reference month)
                                      # to the trade. 1 = trade the next trading day (Mon after
                                      # the Fri print). The ~1-month reference→publication lag is
                                      # handled inside compute_ue_signal, NOT by this field.
    defensive_weights: dict[str, float] = field(
        default_factory=lambda: {
            "R_f": 0.25,
            "KMLM": 0.25,
            "VGIT": 0.25,
            "GLD": 0.25,
        }
    )
    # Reserved for future signals on other equity tickers (e.g. VXUS).
    # Currently only VTI and VTI_LEAPS are governed by GTT.
    # When a VXUS-specific signal is designed, add its config fields here.
```

**Notes:**
- `defensive_weights` must sum to 1.0; validated in `__post_init__`
- `"R_f"` is a sentinel key meaning T-bill cash. Its daily gross return is drawn from the
  existing **`return_data.risk_free_rate` Series** (the daily annualized decimal from
  `data.fetch_risk_free_rate`), converted per-date: `daily_R_f_return_t = rfr_series[t] / 252`.
  It is **not** a scalar — the same date-varying series already used elsewhere in the backtest
  and by `run_leaps_simulation` (reindexed to the return index with `ffill`, `fillna(0.0)`).
- Other keys must be valid tickers present in `return_data`
- `vix_p90_threshold` is a decimal (e.g. `0.272`), not a percentage

### 3.2 New: `GttSignalData` (in new module `src/finance/gtt.py`)

```python
@dataclass(frozen=True)
class GttSignalData:
    position_mask: pd.Series      # DatetimeIndex → int (1 = Long, 0 = Defensive)
                                  # Already lag-adjusted; directly consumable by run_backtest
    ue_signal: pd.Series          # DatetimeIndex → int (1 = UE_12M active)
    vix_signal: pd.Series         # DatetimeIndex → int (1 = VIX_5D active)
    vix_p90_threshold: float      # Threshold used (stored for reproducibility)
    unrate_start: pd.Timestamp    # Earliest date in UNRATE series used
    vix_start: pd.Timestamp       # Earliest date in VIX series used
```

### 3.3 Updated: `PortfolioConfig` (in `portfolio.py`)

Add one optional field:

```python
gtt_config: GttConfig | None = None   # None = GTT disabled, existing behavior unchanged
```

`__post_init__` adds one validation: if `gtt_config` is not None and `defensive_weights`
contains non-R_f keys, those keys must exist in `target_weights`.

### 3.4 New constants (in `consts.py`)

```python
# Tickers governed by the GTT timing signal.
# VTI_LEAPS is matched via the LEAPS_KEY_SUFFIX suffix ("VTI_LEAPS").
# Extend this set when a VXUS GTT signal is designed and validated.
GTT_EQUITY_TICKERS: frozenset[str] = frozenset({"VTI"})

GTT_UNRATE_TRADE_LAG_DAYS: int = 1      # Trading-day execution lag from the UNRATE publication
                                        # date to the trade. The reference→publication (~1-month)
                                        # lag is handled inside compute_ue_signal via first-Friday
                                        # re-stamping, NOT by this constant.
GTT_VIX_CONSECUTIVE_DAYS: int = 5       # Default persistence window
GTT_SMA_WINDOW: int = 200               # Default equity price SMA window

GTT_DEFENSIVE_WEIGHTS_DEFAULT: dict[str, float] = {
    "R_f": 0.25,
    "KMLM": 0.25,
    "VGIT": 0.25,
    "GLD": 0.25,
}
```

---

## 4. Component / Module Breakdown

### 4.1 New module: `src/finance/gtt.py`

**Responsibility:** All GTT signal computation and FRED I/O. Keeps `portfolio.py` pure and
keeps FRED fetching isolated from the existing yfinance-based `data.py`.

**Public API:**

```python
# I/O function (side effects: FRED + yfinance network calls)
def fetch_gtt_signal_data(
    start_date: str,
    end_date: str,
    vix_p90_threshold: float,
    vix_consecutive_days: int = GTT_VIX_CONSECUTIVE_DAYS,
    unrate_trade_lag_days: int = GTT_UNRATE_TRADE_LAG_DAYS,
    sma_window: int = GTT_SMA_WINDOW,
    equity_prices: pd.Series | None = None,   # VTI price series for SMA; if None, fetched internally
) -> GttSignalData:
    """Fetch UNRATE from FRED and VIX from yfinance, compute all signals, and
    return a lag-adjusted daily position mask. UNRATE is publication-dated inside
    compute_ue_signal (1st Friday of the following month); the single close→open
    execution lag is applied in compute_position_mask.
    pragma: no cover (I/O boundary)
    """

# Pure signal computation functions (testable, no I/O)
def compute_ue_signal(
    unrate: pd.Series,
    rolling_window_months: int = 12,
) -> pd.Series:
    """Return daily int Series (0/1): 1 where UNRATE >= trailing 12-month MA.

    Publication-date alignment (Option B — deterministic BLS cadence):
      1. FRED indexes UNRATE at the reference-month start (e.g. 2026-01-01 for the
         January rate). That value is NOT public until the Employment Situation
         release on the first Friday of the FOLLOWING month.
      2. Re-stamp each monthly observation from its reference-month index to the
         first Friday of the following month (its true publication date).
      3. Resample publication-dated series to daily via forward-fill.
      4. Compute the trailing 12-month MA and the 0/1 flag on the publication-dated
         series.

    The result is a daily series that only "knows" a month's UNRATE from its actual
    publication date onward. The final close t → open t+1 execution lag is applied
    once, downstream, in compute_position_mask (shared with VIX_5D). No per-signal
    day-count shift is applied here.

    Note: on the rare month the BLS deviates from the first-Friday cadence (holiday
    weeks, shutdowns) this is an approximation of ≤ a few days; documented, not
    corrected. Empirically checked against live FRED release dates in Phase 3.
    """

def compute_vix_signal(
    vix: pd.Series,
    threshold: float,
    consecutive_days: int = GTT_VIX_CONSECUTIVE_DAYS,
) -> pd.Series:
    """Return daily int Series (0/1): VIX >= threshold for N consecutive days."""

def compute_position_mask(
    ue_signal: pd.Series,
    vix_signal: pd.Series,
    equity_prices: pd.Series,
    sma_window: int = GTT_SMA_WINDOW,
) -> pd.Series:
    """Combine signals with 200d SMA filter. Returns 1 (Long) or 0 (Defensive).
    Output is already 1-day lagged (signal at close t → position at open t+1).

    This 1-day shift is the SINGLE execution lag for both signals: UE_12M is already
    publication-dated (first Friday of the following month) by compute_ue_signal, so
    the shift maps the Friday publication close → the next trading day (Monday); VIX_5D
    is known at close t, so the same shift maps close t → t+1. Both are computed on
    daily indices, so the shift is 1 *trading* day, not 1 calendar day.
    """
```

### 4.2 Modified: `src/finance/portfolio.py`

**Changes to `run_backtest`:**

The loop gains a GTT branch. On each day, after computing normal holdings:

```
if gtt_signal is not None and date is governed by GTT:
    if position_mask[date] == 0:   # Defensive
        override equity holdings → 0
        allocate freed capital → defensive_weights
        (R_f portion earns rfr/252 that day; other tickers use their day_ret)
    else:
        normal behavior
```

**Rebalance interaction (Option C):**
On a quarterly rebalance date, rebalance runs as normal across all assets. After rebalancing,
GTT override is applied: if position_mask[date] == 0, VTI (and VTI_LEAPS) holdings are
zeroed and the freed capital is redistributed to defensive_weights.

**Monthly contribution diversion:**
On month-end, if GTT is in defensive mode for the equity leg, the VTI (and LEAPS) share of
`monthly_contribution` is allocated to `defensive_weights` instead.

**LEAPS interaction (close-and-reopen with fresh contracts):**

When GTT goes defensive, LEAPS contracts are **actually closed** — marked to market, taxed
(taxable accounts), and the proceeds parked in the defensive sleeve. When GTT re-enters, the
parked capital buys **fresh** DITM 2-year contracts at the then-current spot and IV. This is the
accurate treatment: the closed contracts stop carrying delta during the defensive period, and
re-entry pays the prevailing option price (correctly capturing elevated IV after a vol spike).

*Why this changes the architecture.* Today `run_leaps_simulation` is pre-computed **once**, up
front, and the daily loop only prices the resulting ledger. GTT-driven close/reopen cannot use a
single pre-computed ledger because the re-entry capital depends on how the parked proceeds grow
inside the defensive sleeve. However, the defensive-sleeve return path is itself independent of
LEAPS (it is a fixed-weight blend of `defensive_weights` assets + `R_f`), so it **can** be
pre-computed. This lets us keep the "pre-compute the ledger, then price it in the loop" pattern
via a **segmented simulation**:

```
Given position_mask, split the timeline into alternating Long / Defensive windows.
Pre-compute defensive_gross_return_t  (pure: defensive_weights · asset_returns, R_f → rfr/252).

pool = 0.0                                   # parked LEAPS-origin capital (dollars)
for each window in chronological order:
    if window is LONG:
        segment_ledger = run_leaps_simulation(
            price_series = prices[window],
            monthly_contribution_to_leaps = leaps_monthly,
            initial_capital = pool,          # deploy parked capital as a fresh contract on day 1
            ... iv_series, risk_free_series ...
        )
        # On the window's final day, force-close every live contract:
        #   value  = price_leaps_contract(c, spot, last_day, iv, rfr)   for each live c
        #   tax    = max(0, value - cost_basis) * ltcg_rate    (0 if TAX_SHELTERED or loss)
        #   record a LeapsGttCloseEvent(close_date, contract, value, tax)
        pool = sum(value - tax over live contracts)
        append segment_ledger.contracts / roll_events / close_events to the combined ledger
    else:  # DEFENSIVE window
        # Parked LEAPS capital rides the defensive sleeve alongside diverted contributions.
        for day in window:
            pool *= (1 + defensive_gross_return_day)
        pool += diverted_leaps_contributions over the window   # leaps share of monthly_contribution
        # No live LEAPS contracts exist during a defensive window → LEAPS MTM = 0.
```

The combined `LeapsLedger` is assembled from all segments. During a defensive window the loop
finds **no** live contracts (`_live_contracts` returns empty), so equity/LEAPS attribution is
correctly zero and the parked `pool` shows up under the defensive sleeve — matching the real
economics with no notional hand-waving.

*New event type (in `leverage.py`):*

```python
@dataclass(frozen=True)
class LeapsGttCloseEvent:
    """A forced full close of a LEAPS contract triggered by a GTT defensive signal.

    Distinct from a roll (no replacement opened) and a partial close (full, and taxed).
    """
    close_date: pd.Timestamp
    contract: LeapsContract
    mtm_value: float          # Black-Scholes value at close
    gain_realized: float      # mtm_value - cost_basis (may be negative)
    tax_paid: float           # max(0, gain_realized) * ltcg_rate; 0 if TAX_SHELTERED/loss
    net_proceeds: float       # mtm_value - tax_paid, parked in the defensive sleeve
```

`LeapsLedger` gains one field: `gtt_close_events: tuple[LeapsGttCloseEvent, ...] = ()`.

**Monthly contributions during a defensive period:** the VTI and LEAPS shares of
`monthly_contribution` are diverted to the defensive sleeve. The base-equity share and the
LEAPS share are tracked as separate parked pools so that on re-entry each returns to its own
destination (VTI → shares at current price; LEAPS → a fresh contract sized by its parked pool).

**Re-entry timing:** re-entry deploys the parked LEAPS pool on the first Long day via
`create_leaps_contract` at that day's spot/IV — it does not wait for a month-end. The normal
month-end roll/purchase cadence resumes thereafter.

**No changes to `BacktestResult` shape.** `weight_history` will reflect 0 weight for VTI
and non-zero for defensive assets during GTT-active periods, which is the correct
observable behavior.

### 4.3 Modified: `src/finance/data.py`

No changes required. VIX is already fetchable via `fetch_prices` or `fetch_volatility_index`.
FRED fetching belongs in `gtt.py`.

### 4.4 Modified: `src/finance/consts.py`

Add the five constants listed in §3.4.

---

## 5. Step-by-Step Implementation Roadmap

### Phase 1 — Constants & Config (low risk, no behavior change)

1. Add `GTT_EQUITY_TICKERS`, `GTT_UNRATE_TRADE_LAG_DAYS`, `GTT_VIX_CONSECUTIVE_DAYS`,
   `GTT_SMA_WINDOW`, `GTT_DEFENSIVE_WEIGHTS_DEFAULT` to `consts.py`
2. Add `GttConfig` dataclass to `portfolio.py` (with `__post_init__` validation)
3. Add `gtt_config: GttConfig | None = None` to `PortfolioConfig`
4. Add `__post_init__` validation: defensive weight keys exist in target_weights (excluding R_f)
5. Tests: `GttConfig` validation (sum != 1.0, unknown ticker keys)

### Phase 2 — `gtt.py` Pure Signal Functions

1. Create `src/finance/gtt.py`
2. Implement `compute_ue_signal` — re-stamp each monthly UNRATE obs to the first Friday of the
   *following* month (publication date), daily ffill, 12M rolling mean, 0/1 flag. No day-count
   shift here; the execution lag lives in `compute_position_mask`.
3. Implement `compute_vix_signal` — rolling N-day sum >= N (consecutive days logic)
4. Implement `compute_position_mask` — OR logic, 200d SMA filter, 1-day lag
5. Add `GttSignalData` dataclass
6. Tests (pure functions, no I/O): unit tests for each signal function with synthetic series,
   property-based tests (Hypothesis) verifying position_mask is always 0 or 1

### Phase 3 — `fetch_gtt_signal_data` (I/O boundary)

1. Implement `fetch_gtt_signal_data` in `gtt.py` (`# pragma: no cover`)
2. Fetch UNRATE via `fredapi.Fred.get_series('UNRATE')`
3. **Empirical FRED-indexing check (blocking, one-time):** confirm the assumption that
   `Fred.get_series('UNRATE')` indexes each observation at the *reference-month start*
   (e.g. `2026-01-01` for the January rate). Print the last ~6 index dates and eyeball
   against known Employment Situation release dates; assert the index is month-start
   (`idx.day == 1` for all observations) and that the first-Friday-of-following-month
   re-stamp lands on or after the true release date. If FRED's convention differs, the
   Option B re-stamp offset in `compute_ue_signal` must be corrected before proceeding.
4. Fetch VIX via existing `fetch_prices` (yfinance) or `fetch_volatility_index`
5. Wire through to pure functions; return `GttSignalData`
6. Integration test (optional, marked slow): live fetch for a short date range

### Phase 4 — Segmented LEAPS Simulation (close-and-reopen)

1. Add `LeapsGttCloseEvent` dataclass and `gtt_close_events` field to `LeapsLedger` in
   `leverage.py`
2. Add a pure helper `close_leaps_contract(contract, date, spot, iv, ltcg_rate, rfr) ->
   LeapsGttCloseEvent` (mirrors `roll_contract` but opens no replacement)
3. Add `run_segmented_leaps_simulation(price_series, position_mask, defensive_gross_return,
   leaps_monthly, config, ...)` that walks alternating Long/Defensive windows, calling
   `run_leaps_simulation` per Long window with `initial_capital = parked_pool`, force-closing
   at each Long→Defensive boundary, and compounding the pool through Defensive windows
4. Tests (pure): single Long window == current behavior; a Long→Defensive→Long sequence closes
   and reopens with the expected proceeds; TAX_SHELTERED path pays zero close tax

### Phase 5 — `run_backtest` GTT Branch

1. Add `gtt_signal: GttSignalData | None = None` parameter to `run_backtest`
2. Pre-compute (pure, outside the loop):
   - Identify GTT-governed tickers: `GTT_EQUITY_TICKERS` ∪ all `*_LEAPS` keys derived from them
   - Reindex `gtt_signal.position_mask` to `returns.index`
   - `defensive_gross_return_t` = `defensive_weights · asset_returns` (with `R_f` → `rfr/252`)
   - If LEAPS present, build the segmented ledger via `run_segmented_leaps_simulation`
3. Inside the loop, after step (a) apply returns:
   - If `position_mask[date] == 0`: base-equity holdings ride the defensive sleeve; LEAPS MTM
     is naturally 0 because no contracts are live (segmented ledger already reflects the close)
4. Monthly contribution diversion: inside month-end block, branch on `position_mask`; base and
   LEAPS shares parked in the defensive sleeve, each returned to its destination on re-entry
5. Rebalance interaction: rebalance runs first (Option C), then GTT override applied
6. Tests: unit tests with synthetic return/price data; verify NAV, weight_history,
   return_series match expected values when GTT fires and when it does not; verify a
   no-LEAPS GTT backtest and a no-GTT LEAPS backtest are both unchanged

### Phase 6 — Validation Against EDA

1. Write `analyses/gtt_library_validation.py` that runs the library backtest with GTT enabled
   and compares Table 5.1 metrics against `outputs/gtt_findings.md` reference numbers
2. Accept ±5bp tolerance on CAGR (due to contribution timing differences vs EDA daily loop)
3. Flag any metric deviating beyond tolerance

---

## 6. Known Assumptions

| # | Assumption | Implication |
|---|-----------|-------------|
| A1 | `vix_p90_threshold` is caller-supplied; no look-ahead protection in library | Caller must compute threshold from appropriate history window |
| A2 | FRED indexes UNRATE at the reference-month start; the print is public on the 1st Friday of the *following* month. `compute_ue_signal` re-stamps each obs to that first Friday (Option B) so no reference-month look-ahead exists. The final close→open execution lag is a single trading-day shift in `compute_position_mask` | If FRED changes its indexing convention or BLS shifts off the first-Friday cadence, the re-stamp offset must be revised (empirically checked in Phase 3, step 3) |
| A3 | VIX_5D = 5 *consecutive* days (rolling sum == N, not just >= N in window) | Matches EDA; stricter than sliding-window variants |
| A4 | LEAPS are **closed** (marked to market, taxed on gains) when GTT goes defensive and **reopened as fresh 2-year contracts** on re-entry | Accurate treatment: delta exposure ceases during defensive windows and re-entry pays prevailing IV. Requires segmented `run_leaps_simulation` and a new `LeapsGttCloseEvent` |
| A4a | A GTT close realizes LTCG tax on gains in taxable accounts (like a roll), even if the contract has been held < 366 days | Conservative/correct: a forced close is a real disposal. Short-hold closes would be taxed at the same `ltcg_rate` — a known simplification vs. true STCG rates |
| A5 | GTT governs only `GTT_EQUITY_TICKERS` (VTI) and their `_LEAPS` variants | VXUS held through all regimes until a VXUS-specific signal is designed |
| A6 | `R_f` sentinel in `defensive_weights` earns `return_data.risk_free_rate[t] / 252` on each date t (a date-varying Series, not a scalar) | Uses the same `return_data.risk_free_rate` Series (`data.fetch_risk_free_rate`, daily annualized decimal) as the rest of the backtest; `defensive_gross_return` is computed per-date |
| A7 | Defensive allocation is held at fixed weights; not rebalanced within a GTT-active period | Simplest correct behavior; drift within a defensive period is acceptable |
| A8 | `fredapi` is a hard dep (already in `pyproject.toml`); no optional-extra needed | Confirmed from `pyproject.toml` inspection |

---

## 7. Potential Edge Cases & Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| GTT fires on day 1 (before 200d SMA is computable) | Medium | `sma_window` period returns NaN SMA → treat as "above SMA" (stay long) until SMA is available; document this warm-up behavior |
| UNRATE series gap / FRED API down | Medium | `fetch_gtt_signal_data` raises `ValueError` with clear message; never silently defaults to "no recession" |
| `defensive_weights` ticker missing from `return_data` on a given day | Low | Caught at `__post_init__` validation time, not at runtime; fail fast |
| GTT deactivates mid-quarter: re-entry timing vs. next rebalance date | Low | On re-entry, parked LEAPS pool buys a fresh contract that day; base-equity pool re-buys shares at current price; next rebalance corrects any drift |
| Whipsaw: rapid Long→Defensive→Long flips churn LEAPS (close tax + fresh premium each cycle) | Medium | Real cost of the strategy, not a bug — surfaced via `gtt_close_events` count and realized tax in the ledger so the drag is measurable. VIX_5D's 5-consecutive-day rule already damps flip frequency |
| GTT close on a contract with `n_contracts == 0` (premium floored) or a defensive window with zero live contracts | Low | Skip close (nothing to sell); pool simply compounds through the defensive sleeve |
| Re-entry premium too small to buy a contract (`create_leaps_contract` floors `n_contracts` to 0) | Low | Parked pool stays in cash within the sleeve until the next month-end purchase can deploy it; matches existing `create_leaps_contract` behavior |
| `position_mask` index does not align with `returns.index` (holiday mismatches) | Medium | Reindex with `ffill` in the backtest loop; document max acceptable gap |
| Backtests starting before 1993 (pre-VIX availability) | Medium | `fetch_gtt_signal_data` raises `ValueError` if `start_date < 1993-01-01`; VIX is the binding constraint |
| Look-ahead in `vix_p90_threshold` when caller passes a full-sample P90 | Low | Documented in `GttConfig` docstring; responsibility is explicitly caller's |
| UNRATE reference-month look-ahead: using the Jan rate before its early-Feb publication would leak ~1 month, concentrated exactly at recession-onset crossings and inflating F-11's 2001/2008 excess | High | `compute_ue_signal` re-stamps each obs to the 1st Friday of the following month before ffill (Option B); F-03 acceptance test asserts the signal does NOT fire in the reference month; FRED indexing verified empirically in Phase 3 |
| `defensive_weights` sum validation with floating-point | Low | Use `abs(sum - 1.0) > 1e-6` tolerance, same pattern as `PortfolioConfig` |
