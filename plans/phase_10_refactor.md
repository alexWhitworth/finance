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
9. Add return distribution shape metrics (skewness, excess kurtosis) to support levered vs. unlevered comparison.
10. Add LEAPS tax drag summary to quantify the taxable vs. tax-sheltered cost.
11. Add multi-strategy side-by-side comparison table for allocation decisions.

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

### 3.3 New: `LeapsTaxSummary`

```python
@dataclass(frozen=True)
class LeapsTaxSummary:
    """Aggregate LEAPS tax drag over the full backtest period.

    Attributes:
        total_roll_tax: Sum of tax_paid across all LeapsRollEvents.
        n_rolls: Number of roll events executed.
        terminal_tax: Terminal liquidation tax (from TerminalNav.terminal_tax).
            0 for TAX_SHELTERED accounts.
        total_tax: total_roll_tax + terminal_tax.
        tax_drag_pct: total_tax as a fraction of final pre-tax NAV.
        annualized_tax_drag: tax_drag_pct annualized over the backtest years.
        account_type: AccountType governing tax treatment.
    """

    total_roll_tax: float
    n_rolls: int
    terminal_tax: float
    total_tax: float
    tax_drag_pct: float
    annualized_tax_drag: float
    account_type: AccountType
```

### 3.5 New: `TerminalNav`

```python
@dataclass(frozen=True)
class TerminalNav:
    """Pre- and post-tax terminal portfolio value for a LEAPS backtest.

    Attributes:
        pre_tax_nav: Final portfolio NAV including open LEAPS MTM gains, before
            any terminal liquidation tax.
        post_tax_nav: pre_tax_nav minus terminal LTCG + NIIT on all open LEAPS
            gains. Equals pre_tax_nav when account_type is TAX_SHELTERED.
        terminal_tax: Dollar tax applied. Always 0 for TAX_SHELTERED accounts.
        open_gain: Total unrealized gain across all live contracts at end date
            (MTM - cost_basis). Can be negative.
        ltcg_rate: Rate used for terminal tax calculation.
        account_type: AccountType governing whether tax was applied.
    """

    pre_tax_nav: float
    post_tax_nav: float
    terminal_tax: float
    open_gain: float
    ltcg_rate: float
    account_type: AccountType
```

### 3.6 `LeapsLedger` — add `partial_close_events`

```python
@dataclass(frozen=True)
class LeapsLedger:
    contracts: tuple[LeapsContract, ...]
    roll_events: tuple[LeapsRollEvent, ...]
    partial_close_events: tuple[LeapsPartialCloseEvent, ...]
    account_type: AccountType
```

### 3.7 New: `LeapsPartialCloseEvent`

```python
@dataclass(frozen=True)
class LeapsPartialCloseEvent:
    close_date: pd.Timestamp
    original_contract: LeapsContract      # contract before reduction
    continuation_contract: LeapsContract  # same contract, reduced n_contracts
    n_contracts_closed: float
    net_proceeds: float                   # MTM of closed portion, returned to base holdings
                                          # No tax applied: rebalancing is tax-free for all assets
```

When a partial close occurs:
- `original_contract` is removed from the live set
- `continuation_contract` (identical except `n_contracts`) is added
- `net_proceeds` are added back to the base holdings dict in `run_backtest`

### 3.8 `PortfolioConfig`

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

#### Changed: `PerformanceMetrics` — add distribution shape fields

```python
@dataclass(frozen=True)
class PerformanceMetrics:
    annualized_return: float
    annualized_std: float
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    omega: float
    skewness: float         # third standardized moment of daily excess returns
    excess_kurtosis: float  # fourth standardized moment minus 3 (0 = normal)
    period_label: str
```

`skewness` and `excess_kurtosis` are computed on the **excess return series**
`r_e(t)` (same series used for Sharpe/Sortino/Omega), not raw returns, so all
distribution metrics are on a consistent basis. A positive `skewness` confirms
right-tail benefit from the levered portfolio; a negative `excess_kurtosis`
relative to the unlevered run indicates lighter left tail.

#### New: `skewness` and `excess_kurtosis` functions

```python
def return_skewness(excess_returns: pd.Series) -> float:
    """Pure. Third standardized moment of the excess return series."""

def return_excess_kurtosis(excess_returns: pd.Series) -> float:
    """Pure. Fourth standardized moment minus 3. Zero implies normal tails."""
```

Both return `0.0` for series with fewer than 4 observations.

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
    Excess returns r_e(t) are computed once and reused for all ratio functions
    and for skewness / excess_kurtosis.
    """
```

#### Changed: `PerformanceReport` — add `terminal_nav` and `tax_summary`

```python
@dataclass(frozen=True)
class PerformanceReport:
    full_period: PerformanceMetrics
    crisis_periods: tuple[PerformanceMetrics, ...]
    vol_contribution_table: pd.DataFrame
    forward_vol_forecast: float
    terminal_nav: TerminalNav | None        # None when no LEAPS overlay present
    tax_summary: LeapsTaxSummary | None     # None when no LEAPS overlay present
```

#### Changed: `build_performance_report`

```python
def build_performance_report(
    backtest_result: BacktestResult,
    price_data: PriceData,
    return_data: ReturnData,
    vol_model: VolatilityModel,
    crisis_periods: dict[str, tuple[str, str]] = CRISIS_PERIODS,
) -> PerformanceReport:
    """
    Full period: passes return_data.risk_free_rate (full Series) to compute_metrics.
    Crisis slices: slices return_data.risk_free_rate to the crisis window and
        passes the sliced Series to compute_metrics.
    No scalar mean is used anywhere.

    price_data: Required to obtain the final-date VTI spot for compute_terminal_nav.
        Uses price_data.prices["VTI"].iloc[-1] aligned to the backtest end date.

    terminal_nav: populated via compute_terminal_nav when backtest_result has a
        non-None leaps_ledger. Uses config.leaps_config.iv as the IV floor.
        Set to None when leaps_ledger is None.

    tax_summary: populated via compute_leaps_tax_summary when leaps_ledger present.
        Set to None when leaps_ledger is None.
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
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> LeapsPartialCloseEvent:
    """
    Reduce a LEAPS position pro-rata to hit target_value in MTM.

    No tax is applied on the close. Rebalancing is tax-free for all assets
    (simplifying assumption: equalize treatment across asset classes).

    Steps:
      1. Mark full position to market.
      2. scale = target_value / current_mtm  (must be in (0, 1))
      3. n_contracts_closed = contract.n_contracts * (1 - scale)
      4. net_proceeds = MTM of closed portion (no tax deduction).
      5. Return LeapsPartialCloseEvent with continuation_contract
         (n_contracts = contract.n_contracts * scale).

    Arguments:
        contract: The contract to partially close.
        current_date: Execution date.
        current_spot: VTI spot price.
        target_value: Desired total MTM value after the close (in dollars).
        iv: Implied volatility for pricing.
        risk_free_rate: Risk-free rate for Black-Scholes.

    Returns:
        LeapsPartialCloseEvent with original, continuation, and
        net_proceeds (the dollars returned to the base portfolio).

    Raises:
        ValueError: If target_value >= current_mtm (no reduction needed).
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

#### New: `compute_leaps_tax_summary`

```python
def compute_leaps_tax_summary(
    ledger: LeapsLedger,
    terminal_nav: TerminalNav,
    final_nav: float,
    years: float,
) -> LeapsTaxSummary:
    """
    Aggregate LEAPS tax drag over the full backtest period.

    Arguments:
        ledger: Complete LeapsLedger with all roll events.
        terminal_nav: TerminalNav from compute_terminal_nav (provides terminal_tax).
        final_nav: Pre-tax terminal portfolio NAV.
        years: Backtest duration in years (used to annualize drag).

    Returns:
        LeapsTaxSummary with total_roll_tax, n_rolls, terminal_tax, total_tax,
        tax_drag_pct, annualized_tax_drag, and account_type.

    Notes:
        tax_drag_pct = total_tax / final_nav.
        annualized_tax_drag = (1 - (1 - tax_drag_pct) ^ (1 / years)) expressed
        as a positive fraction. Returns 0.0 for TAX_SHELTERED accounts.
    """
```

#### New: `compute_terminal_nav`

```python
def compute_terminal_nav(
    ledger: LeapsLedger,
    final_nav: float,
    final_date: pd.Timestamp,
    final_spot: float,
    iv: float = DEFAULT_IV,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ltcg_rate: float = LTCG_RATE,
) -> TerminalNav:
    """
    Compute pre- and post-tax terminal NAV assuming full liquidation of all
    live LEAPS contracts at final_date.

    Terminal tax applies LTCG + NIIT to all open gains regardless of individual
    contract hold durations — a simplifying assumption that is conservative
    (understates, not overstates, LEAPS benefit in taxable accounts).
    TAX_SHELTERED accounts always produce terminal_tax = 0.

    Live contracts are identified using the same logic as
    compute_leaps_nav_contribution (excludes rolled-out and partially-closed
    originals; uses continuation contracts in their place).

    Arguments:
        ledger: Complete LeapsLedger from run_backtest or run_leaps_simulation.
        final_nav: Portfolio NAV at the final backtest date (pre-tax).
        final_date: Last date of the backtest.
        final_spot: VTI spot price at final_date.
        iv: Implied volatility for terminal MTM pricing.
        risk_free_rate: Risk-free rate for Black-Scholes terminal pricing.
        ltcg_rate: Combined LTCG + NIIT rate. Applied to positive open_gain only.

    Returns:
        TerminalNav with pre_tax_nav, post_tax_nav, terminal_tax, open_gain,
        ltcg_rate, and account_type.
    """
```

**Algorithm:**
1. Identify live contracts (same rolled-out + partial-close exclusion logic as `compute_leaps_nav_contribution`)
2. `total_mtm = sum(price_leaps_contract(c, final_spot, final_date, iv, risk_free_rate) for c in live)`
3. `total_cost_basis = sum(c.premium_paid * CONTRACT_MULTIPLIER * c.n_contracts for c in live)`
4. `open_gain = total_mtm - total_cost_basis`
5. `terminal_tax = max(0.0, open_gain) * ltcg_rate` if `TAXABLE`, else `0.0`
6. `post_tax_nav = final_nav - terminal_tax`

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

The external `leaps_ledger` parameter is removed. LEAPS is always initiated
internally by `run_backtest` when LEAPS keys are present in `target_weights`.
There is no bypass path for pre-built ledgers.

```python
def run_backtest(
    return_data: ReturnData,
    price_data: PriceData,
    config: PortfolioConfig,
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
- All assets, including LEAPS positions, are rebalanced without tax — no
  LTCG eligibility check needed, no MIN_HOLD_DAYS guard
- `net_proceeds` from any LEAPS partial close are added back to base holdings
- For `QUARTERLY` rule: `get_rebalance_dates()` behavior unchanged

**`partial_close_events` accumulation pattern:**
`LeapsLedger` remains `frozen=True`. Partial close events generated during the
drift-rebalance loop are accumulated in a local mutable `list[LeapsPartialCloseEvent]`
(a function-scoped construction buffer, invisible to callers). At the return boundary,
the final ledger is produced once via `replace(ledger, partial_close_events=tuple(partial_close_list))`.
This is identical to how `run_leaps_simulation` builds `contracts` and `roll_events`:
local mutable lists → frozen tuples at return. It is O(n) and consistent with the
FP/OOP hybrid pattern throughout the codebase. Do **not** use `dataclasses.replace()`
inside the loop — that would produce O(n²) throwaway allocations.

**VTI spot price — no reconstruction:**
- `price_data.prices["VTI"]` provides absolute prices directly.
- The spot reconstruction hack in the old `run_backtest` is deleted.

---

### 4.8 `figures.py`

#### Changed: import source for `CRISIS_PERIODS`

`figures.py` currently imports `CRISIS_PERIODS` from `finance.metrics`. After
Sub-phase A this must move to `from finance.consts import CRISIS_PERIODS`.

#### New: `compare_performance_table`

```python
def compare_performance_table(
    reports: list[tuple[str, PerformanceReport]],
) -> str:
    """
    Format multiple PerformanceReports into a single aligned side-by-side table.

    Each column is one strategy (label from the tuple). Rows are metrics:
    Ann. Return, Ann. Std, Max DD, Sharpe, Sortino, Calmar, Omega,
    Skewness, Excess Kurtosis.
    A separator row divides full-period metrics from any LEAPS-specific rows.
    LEAPS rows (Terminal NAV pre/post-tax, Total Tax Drag, Ann. Tax Drag) are
    included only when at least one report has a non-None terminal_nav.

    Arguments:
        reports: List of (label, PerformanceReport) tuples. Labels become
            column headers. Order is preserved.

    Returns:
        Formatted string suitable for console output.
    """
```

---

## 5. Implementation Roadmap

### Sub-phase A — `consts.py` + imports (Quick, no logic changes) ✅ DONE

- [x] Create `src/finance/consts.py` with all constants
- [x] Update all modules to import from `consts.py` (remove duplicate definitions)
- [x] `uv run ruff check .` and `uv run mypy src/` clean
- [x] All tests still pass (219 total, 98% coverage)

### Sub-phase B — `data.py` refactor ✅ DONE

- [x] Rename `splice_kmlm` → `splice`; generalize parameter names
- [x] Update `build_price_data` signature: `tickers`, `use_splice`, `fetch_vol_indices`
- [x] Implement generalized splice loop via `SPLICE_MAP`
- [x] Add `PriceData.vol_prices` field (empty DataFrame default)
- [x] Implement `fetch_volatility_index` with VXUS composite blend logic
- [x] Update tests: new `PriceData` construction requires `vol_prices=pd.DataFrame()`
- [x] 8 new tests added to `test_data.py` (custom tickers, vol_prices empty, etc.)

### Sub-phase C — `returns.py` refactor ✅ DONE

- [x] Rename `_decompose_mub_return` → `_decompose_tax_exempt_return`; alias preserved for backward compat
- [x] Update `adjust_tey` parameter names to generic (`prices`, `dividends`); result named after `prices.name`
- [x] Add `tey_tickers: list[str]` parameter to `build_return_data` (defaults to `["MUB"]`)
- [x] Update `build_return_data` loop to iterate `tey_tickers`
- [x] 4 new tests: alias match, custom tey_tickers, absent ticker skipped, result name matches input

### Sub-phase D — `volatility.py` — vol exclusion ✅ DONE

- [x] Add `VOL_INDEX_TICKERS` filter at entry of `build_volatility_model`
- [x] Test: `^VIX` column in `return_data.returns` is excluded from `ewma_vols` and `cov_matrix`

### Sub-phase E — `metrics.py` — dynamic RFR + distribution shape ✅ DONE

- [x] Remove `RISK_FREE_RATE: float` constant; import from `consts.py` as fallback only
- [x] Update `sharpe_ratio`, `sortino_ratio`, `omega_ratio` to `pd.Series` only
- [x] Add `return_skewness()` and `return_excess_kurtosis()` pure functions (via `scipy.stats`)
- [x] Add `skewness` and `excess_kurtosis` fields to `PerformanceMetrics`
- [x] Update `compute_metrics`: compute excess returns once, pass to all ratio functions and shape functions
- [x] Add `price_data: PriceData` parameter to `build_performance_report`
- [x] Add `terminal_nav: None` and `tax_summary: None` placeholder fields to `PerformanceReport`
- [x] Update `build_performance_report` to slice and pass RFR Series, not scalar mean
- [x] Full test audit of `test_metrics.py`:
  - [x] Replace all `sharpe_ratio(r, float)` calls with `sharpe_ratio(r, pd.Series(...))`
  - [x] Added `test_sharpe_ratio_nonzero_rfr_reduces_value` — dynamic RFR effect verified
  - [x] `return_skewness`: 3 tests (short obs, symmetric→0, right-skewed→positive)
  - [x] `return_excess_kurtosis`: 3 tests (short obs, normal→≈0, fat-tails→positive)
  - [x] `compute_metrics` populates `skewness` and `excess_kurtosis` fields with consistency check
  - [x] `build_performance_report` tests updated: add `price_data` arg; `terminal_nav`/`tax_summary` are None
- [x] 235 tests pass, 98.33% coverage, ruff clean, mypy clean

### Sub-phase F — `leverage.py` — partial close + terminal nav + tax summary ✅ DONE

- [x] Add `LeapsTaxSummary` dataclass
- [x] Add `TerminalNav` dataclass
- [x] Add `LeapsPartialCloseEvent` dataclass
- [x] Add `partial_close_events: tuple[LeapsPartialCloseEvent, ...] = ()` field to `LeapsLedger`
- [x] Implement `partial_close_leaps()` (no tax; raises ValueError when target >= MTM)
- [x] Implement `compute_terminal_nav()` with TAXABLE / TAX_SHELTERED branching
- [x] Implement `compute_leaps_tax_summary()`
- [x] Extract `_live_contracts()` helper; update `compute_leaps_nav_contribution` for partial close
- [x] Add `MIN_PREMIUM_PER_SHARE` guard in `create_leaps_contract` (n_contracts = 0.0)
- [x] Update `PerformanceReport.terminal_nav` and `tax_summary` types to `TerminalNav | None` / `LeapsTaxSummary | None`
- [x] Update `build_performance_report` to populate both fields when ledger present
- [x] Tests (all in `test_leverage.py` + `test_metrics.py`):
  - [x] `partial_close_leaps` reduces n_contracts correctly, no tax deducted
  - [x] `partial_close_leaps` raises ValueError when target >= MTM
  - [x] `compute_terminal_nav` TAXABLE: applies LTCG to positive open_gain only
  - [x] `compute_terminal_nav` TAXABLE: terminal_tax = 0 for underwater contracts
  - [x] `compute_terminal_nav` TAX_SHELTERED: terminal_tax = 0, post_tax_nav == pre_tax_nav
  - [x] `compute_terminal_nav` empty ledger: terminal_tax = 0
  - [x] `compute_leaps_tax_summary` TAXABLE: total_tax = roll_tax + terminal_tax
  - [x] `compute_leaps_tax_summary` TAX_SHELTERED: all zeros
  - [x] `compute_leaps_tax_summary` annualized_drag > 0 for taxable with positive tax
  - [x] `build_performance_report` terminal_nav/tax_summary None without LEAPS ledger
  - [x] `build_performance_report` terminal_nav/tax_summary populated with LEAPS ledger
  - [x] n_contracts guard: zero-premium contract produces n_contracts = 0.0
- [x] 254 tests pass, 98.24% coverage, ruff clean, mypy clean
- Note: `iv_series` parameter for `run_leaps_simulation` deferred to Sub-phase G (portfolio rewrite)

### Sub-phase G — `portfolio.py` — full rewrite of `run_backtest`

This is the largest change. Do it last after all dependencies are stable.

Design decisions locked in:
- External `leaps_ledger` parameter removed; LEAPS always initiated internally via LEAPS keys in `target_weights`.
- `partial_close_events` accumulated in a local `list` during the loop; frozen onto the ledger once at the return boundary (mirrors `run_leaps_simulation` pattern).
- `"VTI_LEAPS"` and `"VTI"` may coexist in `target_weights`; standard usage does not mix them.

- [ ] Add `RebalanceRule.DRIFT` enum value
- [ ] Implement `should_rebalance()` helper
- [ ] Update `run_backtest` signature: `(return_data, price_data, config)` — remove `leaps_ledger`
- [ ] Implement LEAPS key detection and capital routing (Model B)
- [ ] Add `iv_series` parameter to `run_leaps_simulation`; pass raw VIX series for contract creation/rolling
- [ ] Implement monthly contribution split (LEAPS vs. base)
- [ ] Implement VIX-based dynamic IV with 30-day smoothing for daily MTM
- [ ] Implement drift rebalancing: local `partial_close_list`, freeze onto ledger at return
- [ ] Delete VTI spot reconstruction logic
- [ ] Update all existing `run_backtest` tests for new signature (no `leaps_ledger` arg; add `price_data`)
- [ ] Add tests:
  - LEAPS capital correctly carved out of initial NAV
  - Monthly contribution correctly split between LEAPS and base
  - Drift rebalancing triggers at ±10% relative band
  - Partial close returns net_proceeds to base holdings (no tax deduction)
  - `partial_close_events` frozen correctly onto final ledger

### Sub-phase H — `figures.py` + integration + coverage

- [x] Fix `figures.py` import: `CRISIS_PERIODS` from `finance.consts` not `finance.metrics` (done in Sub-phase A)
- [ ] Implement `compare_performance_table(reports)` in `figures.py`
- [ ] Update `format_performance_table` to include `skewness`, `excess_kurtosis` rows
- [ ] Update `format_performance_table` to include LEAPS tax rows when `terminal_nav` present
- [ ] Tests for `compare_performance_table`: single-report case, multi-report alignment,
      LEAPS rows present/absent based on terminal_nav
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
| Rebalancing is tax-free for all assets, including LEAPS partial closes | Equalizes treatment across asset classes; avoids per-lot STCG/LTCG tracking |
| LTCG + NIIT (23.8%) applied on LEAPS roll events and terminal liquidation (not rebalancing) | Roll tax and terminal tax are the economically significant events; rebalancing closes are smaller and infrequent |
| Terminal tax applies LTCG rate to all open gains regardless of individual contract hold durations | Conservative simplification: slightly understates LEAPS benefit in taxable accounts; avoids per-lot hold tracking |
| Terminal tax = 0 for TAX_SHELTERED accounts | Correct: no tax realization in IRA/401k |
| Both pre-tax and post-tax terminal NAV reported | Allows comparison of gross and net-of-tax performance |
| Skewness and excess kurtosis computed on excess returns, not raw returns | Consistent basis with Sharpe/Sortino/Omega; isolates distributional effect of leverage net of the risk-free rate |
| `annualized_tax_drag` uses geometric annualization `1 - (1 - drag)^(1/years)` | Correct for compounding; avoids linear approximation error on long backtests |
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
| All live LEAPS contracts at drift check | Any live contract can be partially closed — no LTCG eligibility check required |
| `partial_close_leaps` called with `target_value >= current_mtm` | Raises `ValueError`; caller responsible for checking before calling |
| VIX data unavailable for a date range | `price_data.vol_prices` is empty DataFrame; `run_backtest` falls back to `config.leaps_config.iv` |
| VXUS composite vol: V2TX.DE and VXEEM have different trading calendars | Blend on intersection; forward-fill to fill gaps (max 5 days) |
| Splice proxy unavailable for full pre-splice date range | `build_price_data` raises `ValueError` with clear message identifying the ticker |
| MUB dividend data unavailable or misaligned | `build_return_data` falls back to zero dividends for that ticker; TEY adjustment produces price-return only (documented behavior) |
| Drift band set too tight (e.g. 1%) causes continuous rebalancing | Band is a named constant (`DRIFT_BAND_RELATIVE`) in consts.py; user can override |
| `compute_leaps_nav_contribution` with expired contracts | Expiry check `c.expiry_date > current_date` filters them out; no BS blow-up |
