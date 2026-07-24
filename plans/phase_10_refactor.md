# Phase 10 — Refactor, Redesign, & Realism Improvements

## 1. Overview

Phase 10 is a broad, breaking refactor that corrects fundamental design flaws
introduced in Phases 2–9. The changes span every module. Nothing in the public
API is preserved unchanged where it is incorrect. All existing tests must be
audited and updated; coverage must remain ≥ 80%.

### Goals

1. Make data fetching generic (not hardcoded to one portfolio composition).
2. Separate volatility index data from asset price data in the type system.
3. Replace all scalar risk-free rate usages with time-varying `pd.Series`.
4. Fix the LEAPS accounting model: capital carved out of NAV, not an external overlay.
5. Thread dynamic IV (VIX-based) through contract creation, rolling, and daily MTM.
6. Add drift-based rebalancing with LTCG tax awareness.
7. Add pro-rata partial LEAPS close for rebalancing overshoot.
8. Consolidate all shared constants into `consts.py`.

---

## 2. New File: `consts.py`

All constants currently scattered across modules move here. Every module imports
from `consts.py` rather than defining its own.

```python
# src/finance/consts.py

# --- Asset universe ---
TICKERS: tuple[str, ...] = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")

# --- Splice map: ticker → (proxy_ticker, splice_date) ---
# Used by build_price_data to splice proxy series before inception date.
SPLICE_MAP: dict[str, tuple[str, str]] = {
    "KMLM": ("AQMIX", "2021-01-01"),
    "VXUS": ("VGTSX", "2011-01-25"),   # VXUS inception 2011-01-25
    "MUB":  ("VWITX", "2007-09-10"),   # MUB inception 2007-09-10
}

# --- Vol indices ---
# Excluded from return calculations and volatility model (not investable).
VOL_INDEX_TICKERS: frozenset[str] = frozenset({
    "^VIX", "V2TX.DE", "VXEEM", "^GVZ", "^OVX", "^MOVE",
})

# VXUS composite vol blend: developed_weight * V2TX.DE + (1 - developed_weight) * VXEEM
VXUS_VOL_BLEND: dict[str, float] = {"V2TX.DE": 0.75, "VXEEM": 0.25}
VXUS_VOL_DEVELOPED_WEIGHT: float = 0.75

# Per-asset vol index mapping (None = no vol index available)
ASSET_VOL_INDEX: dict[str, str | None] = {
    "VTI":  "^VIX",
    "VXUS": "VXUS_COMPOSITE",   # special key → triggers VXUS_VOL_BLEND logic
    "GLD":  "^GVZ",
    "MUB":  "^MOVE",
    "KMLM": None,
    "VGIT": "^MOVE",
}

# --- T-bill / risk-free ---
TBILL_TICKER: str = "^IRX"

# --- Returns ---
NIIT_RATE: float = 0.408

# --- Volatility model ---
EWMA_LAMBDA: float = 0.95
ROLLING_CORR_WINDOW_WEEKS: int = 156   # 36 months
TRADING_DAYS_PER_YEAR: int = 252
COV_RIDGE: float = 1e-8

# --- LEAPS ---
LEAPS_STRIKE_RATIO: float = 0.50
DEFAULT_IV: float = 0.18
LTCG_RATE: float = 0.238
MIN_HOLD_DAYS: int = 366
SIX_MONTHS_DAYS: int = 182
CONTRACT_MULTIPLIER: int = 100
TIME_FLOOR: float = 1.0 / 365
DEFAULT_RISK_FREE_RATE: float = 0.0
DEFAULT_DIVIDEND_YIELD: float = 0.013
MIN_PREMIUM_PER_SHARE: float = 0.01   # guard against n_contracts explosion

# LEAPS weight key convention in PortfolioConfig.target_weights
LEAPS_KEY_SUFFIX: str = "_LEAPS"      # e.g. "VTI_LEAPS"

# --- Drift rebalancing ---
DRIFT_BAND_RELATIVE: float = 0.10    # ±10% relative band around target weight

# --- Portfolio ---
MIN_CRISIS_OBSERVATIONS: int = 20

# --- Metrics ---
RISK_FREE_RATE_DEFAULT: float = 0.0   # only used when IRX data is unavailable

# --- Crisis periods ---
CRISIS_PERIODS: dict[str, tuple[str, str]] = {
    "GFC": ("2007-10-01", "2009-03-31"),
    "COVID": ("2020-02-01", "2020-04-30"),
    "2022 Rate Hike": ("2022-01-01", "2022-10-31"),
}
```

---

## 3. Updated Data Models

### 3.1 `PriceData` — add `vol_prices` field

```python
@dataclass(frozen=True)
class PriceData:
    prices: pd.DataFrame       # DatetimeIndex × asset columns (investable only)
    dividends: pd.DataFrame    # DatetimeIndex × asset columns
    vol_prices: pd.DataFrame   # DatetimeIndex × vol index columns (VIX, etc.)
                               # Empty DataFrame if no vol indices were fetched.
    tickers: tuple[str, ...]
    start_date: str
    end_date: str
    spliced: bool
```

`vol_prices` is kept separate so that:
- `returns.py` never sees vol index columns — no exclusion logic needed
- `portfolio.py` and `leverage.py` read `price_data.vol_prices["^VIX"]` directly
- `volatility.py` only ever sees investable asset returns

### 3.2 `ReturnData` — unchanged except field sourcing

`risk_free_rate: pd.Series` remains. It is always populated (zero-filled if IRX
unavailable). The field is annualized decimal (e.g. `0.05` for 5%); callers
divide by 252 internally where daily conversion is needed.

### 3.3 `LeapsLedger` — add `partial_close_events`

```python
@dataclass(frozen=True)
class LeapsLedger:
    contracts: tuple[LeapsContract, ...]
    roll_events: tuple[LeapsRollEvent, ...]
    partial_close_events: tuple[LeapsPartialCloseEvent, ...]
    account_type: AccountType
```

### 3.4 New: `LeapsPartialCloseEvent`

```python
@dataclass(frozen=True)
class LeapsPartialCloseEvent:
    close_date: pd.Timestamp
    original_contract: LeapsContract      # contract before reduction
    continuation_contract: LeapsContract  # same contract, reduced n_contracts
    n_contracts_closed: float
    gain_realized: float                  # MTM of closed portion - cost basis
    tax_paid: float                       # gain * ltcg_rate if TAXABLE and gain > 0
    net_proceeds: float                   # returned to base portfolio holdings
```

When a partial close occurs:
- `original_contract` is removed from the live set
- `continuation_contract` (identical except `n_contracts`) is added
- `net_proceeds` are added back to the base holdings dict in `run_backtest`

### 3.5 `PortfolioConfig` — add `account_type` for LEAPS tax

```python
@dataclass(frozen=True)
class PortfolioConfig:
    target_weights: dict[str, float]      # "VTI_LEAPS": 0.30 routes through LEAPS
    initial_nav: float
    monthly_contribution: float
    rebalance_rule: RebalanceRule         # QUARTERLY | DRIFT
    weight_strategy: WeightStrategy
    leaps_config: LeapsConfig | None      # None = no LEAPS overlay
```

No separate `account_type` needed on `PortfolioConfig`; `LeapsConfig.account_type`
already carries it.

---

## 4. Module API Changes

### 4.1 `consts.py` (new)

Pure constants module. No functions, no imports from `finance.*`.

---

### 4.2 `data.py`

#### Changed: `build_price_data`

```python
def build_price_data(
    start_date: str,
    end_date: str,
    tickers: list[str] = list(TICKERS),
    use_splice: bool = True,
    fetch_vol_indices: bool = False,
) -> PriceData:
    """
    Top-level I/O entry point.

    Arguments:
        start_date: Inclusive start in YYYY-MM-DD format.
        end_date: Inclusive end in YYYY-MM-DD format.
        tickers: Asset tickers to fetch. Defaults to TICKERS from consts.py.
        use_splice: If True, apply SPLICE_MAP entries for tickers that have a
            proxy defined and whose splice_date is after start_date.
        fetch_vol_indices: If True, fetch vol indices for tickers that have an
            entry in ASSET_VOL_INDEX and populate PriceData.vol_prices.
            Defaults to False so basic backtests have no overhead.

    Returns:
        PriceData with prices, dividends, vol_prices, tickers, dates, spliced.
    """
```

Splice logic is generalized: for each ticker in `tickers`, if `SPLICE_MAP` has
an entry and `start_date < splice_date`, fetch the proxy and call `splice()`.

#### Changed: `splice_kmlm` → `splice`

```python
def splice(
    primary_prices: pd.Series,
    proxy_prices: pd.Series,
    splice_date: str,
) -> pd.Series:
    """
    Pure. Concatenate proxy (pre splice_date) with primary (post splice_date).
    Proxy prices are level-adjusted so the seam return is exactly zero.
    Result series name matches primary_prices.name.
    """
```

#### New: `fetch_volatility_index`

```python
def fetch_volatility_index(
    asset_ticker: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Fetch the vol index Series for a given asset ticker.

    For most assets, downloads the corresponding index from ASSET_VOL_INDEX.
    For "VXUS", blends V2TX.DE and VXEEM using VXUS_VOL_BLEND weights.

    Arguments:
        asset_ticker: Asset ticker key (e.g. "VTI", "VXUS", "GLD").
        start_date: Inclusive start in YYYY-MM-DD format.
        end_date: Inclusive end in YYYY-MM-DD format.

    Returns:
        Daily vol index Series (decimal, e.g. 0.20 for 20%). Forward-filled.
        Returns a zero-filled Series if no vol index is defined for the ticker
        or if the download fails.

    Notes:
        VXUS composite = 0.75 * V2TX.DE + 0.25 * VXEEM, re-indexed to the
        intersection of both series, then forward-filled to cover gaps.
    """
```

#### Unchanged

- `fetch_prices` — I/O boundary; signature unchanged
- `fetch_dividends` — unchanged
- `fetch_risk_free_rate` — unchanged
- `_forward_fill_prices` — unchanged (private)

---

### 4.3 `returns.py`

#### Changed: `_decompose_tax_exempt_return` (was `_decompose_mub_return`)

```python
def _decompose_tax_exempt_return(
    prices: pd.Series,
    dividends: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Split a tax-exempt bond fund's total return into price and income components."""
```

#### Changed: `adjust_tey`

```python
def adjust_tey(
    prices: pd.Series,
    dividends: pd.Series,
    marginal_rate: float = NIIT_RATE,
) -> pd.Series:
    """
    Adjust a tax-exempt bond fund's return series for tax-equivalent yield.
    Parameter names are now generic (not MUB-specific).
    """
```

#### Changed: `build_return_data`

```python
def build_return_data(
    price_data: PriceData,
    marginal_rate: float = NIIT_RATE,
    apply_tey: bool = True,
    tey_tickers: list[str] = ["MUB"],
    risk_free_series: pd.Series | None = None,
) -> ReturnData:
    """
    Arguments:
        tey_tickers: Tickers to apply TEY adjustment to. Each must have a
            corresponding dividends column in price_data.dividends.
            Defaults to ["MUB"].
    """
```

The TEY loop now iterates over `tey_tickers` instead of hardcoding `"MUB"`.

---

### 4.4 `volatility.py`

#### Changed: `build_volatility_model`

```python
def build_volatility_model(
    return_data: ReturnData,
    as_of_date: pd.Timestamp | None = None,
) -> VolatilityModel:
    """
    Before computing EWMA vols and correlations, drops any columns whose
    ticker is in VOL_INDEX_TICKERS (from consts.py). This prevents vol index
    series from corrupting the covariance matrix.
    """
```

The exclusion happens once at entry: `returns = returns[[c for c in returns.columns if c not in VOL_INDEX_TICKERS]]`. All downstream helpers (`compute_ewma_vol`, `compute_rolling_weekly_corr`, etc.) receive already-filtered data — no changes to their signatures.

---

### 4.5 `metrics.py`

#### Changed: `sharpe_ratio`

```python
def sharpe_ratio(returns: pd.Series, risk_free_rate: pd.Series) -> float:
    """
    Annualized Sharpe ratio using time-varying risk-free rate.

    Excess return: r_e(t) = r_p(t) - R_f(t) / 252
    Sharpe = mean(r_e) / std(r_e) * sqrt(252)

    Arguments:
        returns: Daily simple return Series.
        risk_free_rate: Daily annualized risk-free rate Series (decimal).
            Aligned to returns.index before subtraction. Divided by
            TRADING_DAYS_PER_YEAR internally to produce daily rate.
    """
```

#### Changed: `sortino_ratio`

```python
def sortino_ratio(returns: pd.Series, risk_free_rate: pd.Series) -> float:
    """
    Sortino ratio using time-varying risk-free rate.

    Excess return: r_e(t) = r_p(t) - R_f(t) / 252
    Downside deviation: sigma_d = sqrt(mean(min(r_e(t), 0)^2)) * sqrt(252)
    Sortino = annualized_return(r_e) / sigma_d

    Notes:
        annualized_return(r_e) is the geometric annualization of excess returns,
        not (ann_return - mean_rfr). This is consistent with the Sharpe formula.
    """
```

#### Changed: `omega_ratio`

```python
def omega_ratio(returns: pd.Series, risk_free_rate: pd.Series) -> float:
    """
    Omega ratio evaluated on excess returns.

    Omega = sum(max(r_e(t), 0)) / (sum(max(-r_e(t), 0)) + eps)
    where r_e(t) = r_p(t) - R_f(t) / 252.

    The threshold is implicitly 0 on the excess return series.
    """
```

#### Changed: `compute_metrics`

```python
def compute_metrics(
    returns: pd.Series,
    nav_series: pd.Series,
    period_label: str,
    risk_free_rate: pd.Series,
) -> PerformanceMetrics:
    """
    risk_free_rate must be a pd.Series aligned (or alignable) to returns.index.
    It is reindexed with forward-fill before being passed to ratio functions.
    """
```

#### Changed: `build_performance_report`

```python
def build_performance_report(
    backtest_result: BacktestResult,
    return_data: ReturnData,
    vol_model: VolatilityModel,
    crisis_periods: dict[str, tuple[str, str]] = CRISIS_PERIODS,
) -> PerformanceReport:
    """
    Full period: passes return_data.risk_free_rate (full Series) to compute_metrics.
    Crisis slices: slices return_data.risk_free_rate to the crisis window and
        passes the sliced Series to compute_metrics.
    No scalar mean is used anywhere.
    """
```

The `risk_free_rate: float = RISK_FREE_RATE` constant and the `RISK_FREE_RATE`
module-level default are removed.

---

### 4.6 `leverage.py`

#### Changed: `LeapsConfig`

```python
@dataclass(frozen=True)
class LeapsConfig:
    iv: float = DEFAULT_IV             # fallback floor when no iv_series available
    ltcg_rate: float = LTCG_RATE
    account_type: AccountType = AccountType.TAXABLE
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE   # fallback scalar
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD
```

#### Changed: `create_leaps_contract` — add guard

If `premium_per_share < MIN_PREMIUM_PER_SHARE` (0.01), return a contract with
`n_contracts = 0.0`. Callers must check `contract.n_contracts > 0` before
adding to the ledger.

#### New: `LeapsPartialCloseEvent` dataclass

Defined in Section 3.4 above. Lives in `leverage.py`.

#### New: `partial_close_leaps`

```python
def partial_close_leaps(
    contract: LeapsContract,
    current_date: pd.Timestamp,
    current_spot: float,
    target_value: float,
    iv: float = DEFAULT_IV,
    ltcg_rate: float = LTCG_RATE,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> LeapsPartialCloseEvent:
    """
    Reduce a LEAPS position pro-rata to hit target_value in MTM.

    Only callable on LTCG-qualified contracts (hold_days >= MIN_HOLD_DAYS).
    Callers are responsible for checking LTCG eligibility before calling.

    Steps:
      1. Mark full position to market.
      2. scale = target_value / current_mtm  (must be in (0, 1))
      3. n_contracts_closed = contract.n_contracts * (1 - scale)
      4. Compute gain on closed portion; apply tax if TAXABLE.
      5. Return LeapsPartialCloseEvent with continuation_contract
         (n_contracts = contract.n_contracts * scale).

    Arguments:
        contract: The contract to partially close.
        current_date: Execution date (must be >= contract.purchase_date + MIN_HOLD_DAYS).
        current_spot: VTI spot price.
        target_value: Desired total MTM value after the close (in dollars).
        iv: Implied volatility for pricing.
        ltcg_rate: LTCG + NIIT rate applied on taxable gains.
        risk_free_rate: Risk-free rate for Black-Scholes.

    Returns:
        LeapsPartialCloseEvent with original, continuation, gain, tax, and
        net_proceeds (the dollars returned to the base portfolio).

    Raises:
        ValueError: If target_value >= current_mtm (no reduction needed).
        ValueError: If contract has not been held MIN_HOLD_DAYS.
    """
```

#### Changed: `compute_leaps_nav_contribution`

Updated to account for `partial_close_events` in the ledger:
- Build a `partially_closed` dict: `original_contract → continuation_contract`
  from `ledger.partial_close_events`
- When iterating live contracts, replace any partially-closed original with
  its continuation contract

```python
def compute_leaps_nav_contribution(
    ledger: LeapsLedger,
    current_date: pd.Timestamp,
    current_spot: float,
    iv: float = DEFAULT_IV,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """
    Computes net P&L (MTM - cost_basis) for all live contracts.
    Rolled-out originals and partially-closed originals are excluded;
    continuation contracts are used in their place.
    """
```

#### Changed: `run_leaps_simulation` — add `iv_series`

```python
def run_leaps_simulation(
    price_series: pd.Series,
    monthly_contribution_to_leaps: float,
    config: LeapsConfig,
    risk_free_series: pd.Series | None = None,
    iv_series: pd.Series | None = None,
) -> LeapsLedger:
    """
    iv_series: Optional daily VIX series (decimal, e.g. 0.20 for 20%).
        When supplied, the value on each month-end date is used for
        contract creation and roll pricing, overriding config.iv.
        config.iv is used as a floor: iv = max(iv_series[date], config.iv).
    """
```

---

### 4.7 `portfolio.py`

#### New enum value: `RebalanceRule.DRIFT`

```python
class RebalanceRule(enum.Enum):
    QUARTERLY = "quarterly"
    DRIFT = "drift"
```

#### New helper: `should_rebalance`

```python
def should_rebalance(
    current_weights: pd.Series,
    target_weights: pd.Series,
    rule: RebalanceRule,
    band: float = DRIFT_BAND_RELATIVE,
) -> bool:
    """
    Pure. Returns True if any asset weight has drifted outside its band.

    QUARTERLY: always returns False (rebalance schedule handled by get_rebalance_dates).
    DRIFT: returns True if any |w_i - t_i| / t_i > band for any asset i.

    Arguments:
        current_weights: Realized weights at the check date.
        target_weights: Target weights from PortfolioConfig.
        rule: RebalanceRule controlling the rebalancing schedule.
        band: Relative drift threshold. Default 0.10 (±10%).
    """
```

#### Changed: `run_backtest` — new signature

```python
def run_backtest(
    return_data: ReturnData,
    price_data: PriceData,
    config: PortfolioConfig,
    leaps_ledger: LeapsLedger | None = None,
) -> BacktestResult:
```

**Key changes to the backtest loop:**

**LEAPS weight routing (Model B — carved-out):**
- On initialization, identify LEAPS keys: `leaps_keys = [k for k in target_weights if k.endswith(LEAPS_KEY_SUFFIX)]`
- The underlying asset for each LEAPS key is `k.removesuffix(LEAPS_KEY_SUFFIX)` (e.g. `"VTI_LEAPS"` → `"VTI"`)
- LEAPS capital is initialized from `initial_nav * target_weights["VTI_LEAPS"]`; this runs a fresh `run_leaps_simulation` internally using `price_data.prices["VTI"]`
- The base holdings dict contains only non-LEAPS keys

**Monthly contribution split:**
- `leaps_fraction = sum(target_weights[k] for k in leaps_keys)`
- `leaps_contribution = monthly_contribution * leaps_fraction` → new contracts via `create_leaps_contract`
- `base_contribution = monthly_contribution * (1 - leaps_fraction)` → allocated to base assets

**VIX-based IV:**
- Daily VIX series read from `price_data.vol_prices` if `"^VIX"` is present
- 30-day rolling mean of VIX applied for daily MTM to reduce noise:
  `iv_smooth = price_data.vol_prices["^VIX"].rolling(30).mean().ffill()`
- For contract creation (month-end), use the raw VIX value (not smoothed)
- `config.leaps_config.iv` is the floor: `iv = max(vix_value, config.leaps_config.iv)`

**Drift rebalancing:**
- For `DRIFT` rule: check monthly (at `month_end_dates`) whether any weight
  is outside its relative band using `should_rebalance()`
- For LEAPS positions: only trigger a partial close if the contract is LTCG-eligible
  (`hold_days >= MIN_HOLD_DAYS`). If not eligible, skip the LEAPS rebalance and
  allow drift for that month.
- Tax on partial close (`net_proceeds`) is added back to base holdings
- For `QUARTERLY` rule: `get_rebalance_dates()` behavior unchanged

**VTI spot price — no reconstruction:**
- `price_data.prices["VTI"]` provides absolute prices directly.
- The spot reconstruction hack in the old `run_backtest` is deleted.

---

## 5. Implementation Roadmap

### Sub-phase A — `consts.py` + imports (Quick, no logic changes)

- [ ] Create `src/finance/consts.py` with all constants
- [ ] Update all modules to import from `consts.py` (remove duplicate definitions)
- [ ] `uv run ruff check .` and `uv run mypy src/` clean
- [ ] All 211 existing tests still pass (no logic changes)

### Sub-phase B — `data.py` refactor

- [ ] Rename `splice_kmlm` → `splice`; generalize parameter names
- [ ] Update `build_price_data` signature: `tickers`, `use_splice`, `fetch_vol_indices`
- [ ] Implement generalized splice loop via `SPLICE_MAP`
- [ ] Add `PriceData.vol_prices` field (empty DataFrame default)
- [ ] Implement `fetch_volatility_index` with VXUS composite blend logic
- [ ] Update tests: new `PriceData` construction requires `vol_prices=pd.DataFrame()`
- [ ] Verify parquet fixture at `data/price_data.parquet` loads cleanly with new schema

### Sub-phase C — `returns.py` refactor

- [ ] Rename `_decompose_mub_return` → `_decompose_tax_exempt_return`
- [ ] Update `adjust_tey` parameter names to generic (`prices`, `dividends`)
- [ ] Add `tey_tickers: list[str]` parameter to `build_return_data`
- [ ] Update `build_return_data` loop
- [ ] Update tests: replace `"MUB"` references where MUB-specific to generic

### Sub-phase D — `volatility.py` — vol exclusion

- [ ] Add VOL_INDEX_TICKERS filter at entry of `build_volatility_model`
- [ ] Update tests: no interface changes, but add a test that confirms VIX
      in `return_data.returns` is silently excluded

### Sub-phase E — `metrics.py` — dynamic RFR

- [ ] Remove `RISK_FREE_RATE: float` constant; import from `consts.py` as fallback only
- [ ] Update `sharpe_ratio`, `sortino_ratio`, `omega_ratio` to `pd.Series` only
- [ ] Update `compute_metrics` signature
- [ ] Update `build_performance_report` to slice and pass Series, not scalar mean
- [ ] Full test audit of `test_metrics.py`:
  - Replace all `sharpe_ratio(r, float)` calls with `sharpe_ratio(r, pd.Series(...))`
  - Add tests verifying dynamic RFR produces correct excess return math
  - Verify constant-Series gives same result as the old scalar path would have

### Sub-phase F — `leverage.py` — partial close + iv_series

- [ ] Add `LeapsPartialCloseEvent` dataclass
- [ ] Add `partial_close_events` field to `LeapsLedger`
- [ ] Implement `partial_close_leaps()`
- [ ] Update `compute_leaps_nav_contribution` for partial close accounting
- [ ] Add `iv_series` parameter to `run_leaps_simulation`; implement floor logic
- [ ] Add `MIN_PREMIUM_PER_SHARE` guard in `create_leaps_contract`
- [ ] Tests:
  - `partial_close_leaps` reduces n_contracts correctly, tax computed correctly
  - n_contracts guard: zero-premium contract is skipped
  - `iv_series` overrides config.iv at each month-end

### Sub-phase G — `portfolio.py` — full rewrite of `run_backtest`

This is the largest change. Do it last after all dependencies are stable.

- [ ] Add `RebalanceRule.DRIFT` enum value
- [ ] Implement `should_rebalance()` helper
- [ ] Update `run_backtest` signature: add `price_data: PriceData`
- [ ] Implement LEAPS key detection and capital routing (Model B)
- [ ] Implement monthly contribution split (LEAPS vs. base)
- [ ] Implement VIX-based dynamic IV with 30-day smoothing for MTM
- [ ] Implement drift rebalancing with LTCG eligibility guard for LEAPS partial close
- [ ] Delete VTI spot reconstruction logic
- [ ] Update all 26 existing `run_backtest` tests for new signature
- [ ] Add tests:
  - LEAPS capital correctly carved out of initial NAV
  - Monthly contribution correctly split
  - Drift rebalancing triggers at ±10% relative band
  - Partial close returns net_proceeds to base holdings
  - LTCG guard: contract < 366 days old is not partially closed

### Sub-phase H — Integration, examples, coverage

- [ ] Update `tests/test_integration.py` for new full pipeline
- [ ] Update `examples/` scripts for new APIs
- [ ] Update `data/price_data.parquet` fixture: split VIX/IRX into `vol_prices`
  (or adjust test loading logic to match new `PriceData` schema)
- [ ] `uv run pytest --cov=src --cov-report=term-missing` → ≥ 80% coverage
- [ ] `uv run ruff check .` → clean
- [ ] `uv run mypy src/` → clean under strict mode

---

## 6. Known Assumptions & Simplifications

| Assumption | Justification |
|---|---|
| Non-LEAPS rebalancing is tax-free | ETF turnover tax drag is small; LEAPS is the dominant effect |
| LTCG + NIIT (23.8%) applied uniformly on all taxable LEAPS gains | Avoids per-lot STCG/LTCG tracking; clearly labeled in docs |
| LEAPS strike is always 50% of spot | Configurable via `LEAPS_STRIKE_RATIO` in consts.py |
| 30-day rolling VIX mean for daily MTM | Reduces mark-to-market noise from VIX spikes; documented |
| `config.leaps_config.iv` is used as a floor, not primary | Prevents absurdly cheap contracts during low-vol regimes |
| Fractional contracts are valid (`n_contracts: float`) | Allows proportional sizing without minimum-lot constraints |
| Drift rebalancing checked monthly, not daily | Daily checking adds complexity without material accuracy gain |

---

## 7. Edge Cases & Risks

| Risk | Mitigation |
|---|---|
| LEAPS key (`VTI_LEAPS`) with no corresponding price column (`VTI`) | `run_backtest` raises `ValueError` on startup if the underlying ticker is absent from `price_data.prices` |
| All live LEAPS contracts are STCG-ineligible at drift check | Skip LEAPS partial close; log drift as unresolved; do not force a taxable close |
| `partial_close_leaps` called with `target_value >= current_mtm` | Raises `ValueError`; caller responsible for checking before calling |
| VIX data unavailable for a date range | `price_data.vol_prices` is empty DataFrame; `run_backtest` falls back to `config.leaps_config.iv` |
| VXUS composite vol: V2TX.DE and VXEEM have different trading calendars | Blend on intersection; forward-fill to fill gaps (max 5 days) |
| Splice proxy unavailable for full pre-splice date range | `build_price_data` raises `ValueError` with clear message identifying the ticker |
| MUB dividend data unavailable or misaligned | `build_return_data` falls back to zero dividends for that ticker; TEY adjustment produces price-return only (documented behavior) |
| Drift band set too tight (e.g. 1%) causes continuous rebalancing | Band is a named constant (`DRIFT_BAND_RELATIVE`) in consts.py; user can override |
| `compute_leaps_nav_contribution` with expired contracts | Expiry check `c.expiry_date > current_date` filters them out; no BS blow-up |
