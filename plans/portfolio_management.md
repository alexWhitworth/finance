# Project: Finance Live Portfolio Management API

## 1. System Overview

This adds a **live portfolio management layer** alongside the existing backtest engine. The two
flows are distinct:

```
[Backtest flow — existing]
run_backtest(ReturnData, PriceData, PortfolioConfig) → BacktestResult

[Live management flow — new]
LivePortfolio (user's current brokerage state)
       │
       ├──► compute_rebalance_plan()    → RebalancePlan
       ├──► compute_portfolio_greeks()  → PortfolioGreeks
       ├──► compute_volatility_report() → VolatilityReport
       └──► compute_gtt_status()        → GttStatus
```

The user's weekly/monthly workflow:

1. **Input**: Describe actual brokerage holdings as a `LivePortfolio`.
2. **Rebalance**: Given target weights, what trades are needed?
3. **LEAPS greeks**: For each live LEAPS contract, compute delta/gamma/vega/theta/charm/vanna.
4. **Volatility**: EWMA vols, covariance, contributions via existing `volatility.py`.
5. **GTT signal**: Is the market currently Long or Defensive?

A secondary flow bridges from a completed backtest to the live layer:

```
BacktestResult.final_state → as_live_portfolio() → LivePortfolio
```

Typical weekly usage:

```python
portfolio = LivePortfolio(as_of_date=today, holdings={...}, target_weights={...}, ...)
nav       = compute_nav_breakdown(portfolio, leaps_mtm=current_leaps_value)
greeks    = compute_portfolio_greeks(portfolio, spot=vti_price, iv=current_iv)
rebal     = compute_rebalance_plan(portfolio, nav, RebalanceRule.QUARTERLY, ...)
vol       = compute_volatility_report(portfolio, return_data)
gtt       = compute_gtt_status(today, vix_p90_threshold=0.272, start_date="2020-01-01")
```

---

## 2. Tech Stack & Dependencies

No new dependencies. All from existing stack: `scipy.stats` (BS greeks, already used in
`leverage.py`), `pandas`, `numpy`, `yfinance`, `fredapi` (already in `gtt.py`).

---

## 3. Data Schema / Type Definitions

### 3a. `LivePortfolio` — primary user input type (`finance/portfolio_manager.py`)

`PortfolioState` is purpose-built for the backtest loop. It carries bookkeeping fields
(`all_window_ledgers`, `all_gtt_closes`, `prev_total_nav`) that have no meaning when a user
is entering their brokerage positions. `LivePortfolio` is self-contained and user-constructible
without understanding backtest internals.

```python
@dataclass(frozen=True)
class LivePortfolio:
    """User's current brokerage portfolio state as of a specific date.

    Attributes:
        as_of_date: The date these holdings reflect (typically today or last close).
        holdings: Base asset ticker → current dollar value (e.g. {"VTI": 50000.0}).
        target_weights: Target allocation. Must sum to 1.0 within 1e-6.
        leaps_contracts: Active LEAPS contracts with surviving fraction.
            Each entry is (contract, scale) where scale ∈ (0, 1].
            Empty tuple for portfolios with no LEAPS overlay.
        gtt_regime: Current GTT regime: 1=Long, 0=Defensive, None=GTT inactive.
        defensive_sleeve: Dollar value in GTT defensive allocation. 0.0 if inactive.
        leaps_pool: Force-closed LEAPS proceeds parked during defensive window. 0.0 if inactive.

    Raises:
        ValueError: If target_weights does not sum to 1.0 within 1e-6.
        ValueError: If any leaps_scale value is outside (0, 1].
        ValueError: If any contract has expiry_date <= as_of_date.
    """
    as_of_date: pd.Timestamp
    holdings: dict[str, float]
    target_weights: dict[str, float]
    leaps_contracts: tuple[tuple[LeapsContract, float], ...]
    gtt_regime: int | None
    defensive_sleeve: float = 0.0
    leaps_pool: float = 0.0

    def __post_init__(self) -> None:
        total = sum(self.target_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"target_weights must sum to 1.0; got {total:.6f}")
        for contract, scale in self.leaps_contracts:
            if contract.expiry_date <= self.as_of_date:
                raise ValueError(
                    f"Contract with expiry {contract.expiry_date.date()} is expired "
                    f"as of {self.as_of_date.date()}. Remove it before constructing LivePortfolio."
                )
            if not (0.0 < scale <= 1.0):
                raise ValueError(f"leaps_scale must be in (0, 1]; got {scale}")
```

### 3b. Output types — `finance/portfolio_manager.py`

```python
@dataclass(frozen=True)
class NavBreakdown:
    """NAV decomposition for a LivePortfolio.

    Attributes:
        base_nav: Sum of base asset holdings.
        leaps_nav: Mark-to-market value of all active LEAPS contracts (caller-supplied).
        defensive_sleeve: GTT-swept capital.
        leaps_pool: Parked force-closed LEAPS proceeds.
        total_nav: base_nav + leaps_nav + defensive_sleeve + leaps_pool.
    """
    base_nav: float
    leaps_nav: float
    defensive_sleeve: float
    leaps_pool: float
    total_nav: float


@dataclass(frozen=True)
class HoldingView:
    """Single asset row in a portfolio weight drift analysis.

    Attributes:
        ticker: Asset identifier.
        dollar_value: Current dollar value.
        actual_weight: dollar_value / total_nav.
        target_weight: From LivePortfolio.target_weights (0.0 if absent).
        weight_drift: actual_weight - target_weight (signed).
        relative_drift: weight_drift / target_weight; None when target_weight == 0.
    """
    ticker: str
    dollar_value: float
    actual_weight: float
    target_weight: float
    weight_drift: float
    relative_drift: float | None


@dataclass(frozen=True)
class TradeOrder:
    """Single buy/sell instruction from a rebalance simulation.

    Attributes:
        ticker: Asset to trade.
        current_value: Dollar value before rebalance.
        target_value: Dollar value after rebalance.
        trade_amount: target_value - current_value. Positive = buy, negative = sell.
        current_weight: Realized weight before rebalance.
        target_weight: Target weight from LivePortfolio.
    """
    ticker: str
    current_value: float
    target_value: float
    trade_amount: float
    current_weight: float
    target_weight: float


@dataclass(frozen=True)
class RebalancePlan:
    """Simulated rebalance outcome for a LivePortfolio.

    Attributes:
        as_of_date: Date of the simulation.
        would_trigger: Whether the rebalance rule fires at this date.
        trigger_reason: "quarterly_scheduled" | "drift_threshold" | "not_triggered".
        trades: Per-asset buy/sell instructions. Empty tuple if not triggered.
        leaps_trim: Dollar reduction to LEAPS (positive = LEAPS partially closed).
            Non-zero only when DRIFT rule fires and LEAPS are overweight.
        holdings_view: Per-asset drift breakdown before rebalance.
    """
    as_of_date: pd.Timestamp
    would_trigger: bool
    trigger_reason: str
    trades: tuple[TradeOrder, ...]
    leaps_trim: float
    holdings_view: tuple[HoldingView, ...]


@dataclass(frozen=True)
class VolatilityReport:
    """Portfolio volatility analysis snapshot.

    Attributes:
        as_of_date: Snapshot date.
        vol_model: Full VolatilityModel (ewma_vols, rolling_corr, cov_matrix).
        portfolio_vol: Forecasted annualized portfolio volatility (sigma_hat_p).
        contribution_table: DataFrame from build_vol_contribution_table().
            Columns: sigma_tilde, sigma_hat, rho_VTI, contrib. Index: asset.
        weights_used: Realized weights used in the contribution computation.
    """
    as_of_date: pd.Timestamp
    vol_model: VolatilityModel
    portfolio_vol: float
    contribution_table: pd.DataFrame
    weights_used: pd.Series


@dataclass(frozen=True)
class GttStatus:
    """Current GTT signal state.

    Attributes:
        as_of_date: Date of evaluation.
        regime: 1=Long, 0=Defensive.
        ue_signal: UE_12M signal value at as_of_date (0 or 1).
        vix_signal: VIX_5D signal value at as_of_date (0 or 1).
        vix_current: Raw VIX value at as_of_date (decimal, e.g. 0.21).
        vix_threshold: P90 threshold used.
        price_vs_sma200: "above" | "below" | "warming_up" (SMA not yet available).
        signal_data: Full GttSignalData for audit/reproducibility.
    """
    as_of_date: pd.Timestamp
    regime: int
    ue_signal: int
    vix_signal: int
    vix_current: float
    vix_threshold: float
    price_vs_sma200: str
    signal_data: GttSignalData
```

### 3c. LEAPS greeks output types — `finance/greeks.py`

```python
@dataclass(frozen=True)
class ContractGreeks:
    """Black-Scholes greeks for a single LeapsContract.

    All position-level fields scaled by (n_contracts * CONTRACT_MULTIPLIER * leaps_scale).

    Attributes:
        contract: Source LeapsContract.
        as_of_date: Evaluation date.
        spot: Spot price used.
        iv: Implied volatility used.
        risk_free_rate: Risk-free rate used.
        time_to_expiry: Years to expiry (floored at TIME_FLOOR).
        price: BS call price per share.
        delta: dV/dS. In (0, 1) for calls.
        gamma: d²V/dS². Always positive for long calls.
        vega: dV/dσ per unit IV move.
        theta: dV/dt in dollars per calendar day (negative for long calls).
        vanna: dDelta/dVol. Sensitivity of delta to implied vol.
        charm: dDelta/dt. Delta decay per calendar day.
        leaps_scale: Surviving fraction applied.
        position_delta: delta * n_contracts * CONTRACT_MULTIPLIER * leaps_scale.
        position_vega: vega * n_contracts * CONTRACT_MULTIPLIER * leaps_scale.
        position_theta: theta * n_contracts * CONTRACT_MULTIPLIER * leaps_scale.
    """
    contract: LeapsContract
    as_of_date: pd.Timestamp
    spot: float
    iv: float
    risk_free_rate: float
    time_to_expiry: float
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    vanna: float
    charm: float
    leaps_scale: float
    position_delta: float
    position_vega: float
    position_theta: float


@dataclass(frozen=True)
class PortfolioGreeks:
    """Aggregate greeks across all active LEAPS contracts in a LivePortfolio.

    Attributes:
        as_of_date: Evaluation date.
        contracts: Per-contract greeks.
        net_delta: Sum of position_delta.
        net_vega: Sum of position_vega.
        net_gamma: Sum of per-position scaled gamma.
        net_theta: Total dollar theta per calendar day.
        net_vanna: Sum of per-position scaled vanna.
        net_charm: Sum of per-position scaled charm.
    """
    as_of_date: pd.Timestamp
    contracts: tuple[ContractGreeks, ...]
    net_delta: float
    net_vega: float
    net_gamma: float
    net_theta: float
    net_vanna: float
    net_charm: float
```

### 3d. Change to existing type — `BacktestResult`

```python
@dataclass(frozen=True)
class BacktestResult:
    nav_series: pd.Series
    weight_history: pd.DataFrame
    return_series: pd.Series
    leaps_ledger: LeapsLedger | None
    config: PortfolioConfig
    final_state: PortfolioState      # NEW: enables as_live_portfolio() bridge
```

Required field (no default). `BacktestResult` is only ever constructed inside `run_backtest`.

### 3e. LEAPS DCA entry signal — `finance/dca_signal.py`

The composite entry score governs **when and how aggressively** to deploy capital into DITM
LEAPS. It combines four factors into a score in [0, 100], with a MACD-based trend gate that
dampens the whole score when the trend is broken.

```python
@dataclass(frozen=True)
class LeapsDcaSignal:
    """Multi-factor composite entry score for DITM LEAPS DCA timing.

    Attributes:
        as_of_date: Evaluation date.
        ticker: Underlying ticker evaluated (e.g. "VTI").
        entry_score: Composite score in [0, 100]. Higher = more favorable entry.
        score_percentile: entry_score rank within the lookback window (0–100).
        alpha_t: Tranche allocation fraction in [0, 1].
            0.0 when score_percentile < hold_pctile (hold in cash).
            1.0 when score_percentile >= aggressive_pctile (full sweep).
            Linear interpolation between the two thresholds otherwise.
        dca_action: "HOLD" | "TRANCHE" | "AGGRESSIVE_SWEEP".
        rsi: 14-day RSI at as_of_date.
        stoch_d: 5/3 Stochastic %D at as_of_date.
        iv_percentile: 252-day IV percentile rank (0–100). Low = cheap vol.
        iv_current: Raw IV (decimal) at as_of_date.
        macd_hist: MACD histogram value at as_of_date.
        macd_bearish_confirmed: True if MACD histogram has been negative for ≥ 3
            consecutive sessions (debounced bearish flag).
        macd_gate: Gate multiplier applied (1.0 or macd_gate_floor).
    """
    as_of_date: pd.Timestamp
    ticker: str
    entry_score: float
    score_percentile: float
    alpha_t: float
    dca_action: str
    rsi: float
    stoch_d: float
    iv_percentile: float
    iv_current: float
    macd_hist: float
    macd_bearish_confirmed: bool
    macd_gate: float
```

---

## 4. Component / Module Breakdown

### 4a. `finance/leverage.py` — add missing BS greeks (pure functions)

Four new functions alongside existing `bs_call_price`, `bs_call_delta`, `bs_call_vanna`.
All use the existing `_bs_d1` helper and share the same scalar-float signature pattern.

```python
def bs_call_gamma(spot, strike, time_to_expiry, iv, risk_free_rate, dividend_yield) -> float:
    # N'(d1) * e^(-q*T) / (S * sigma * sqrt(T))

def bs_call_vega(spot, strike, time_to_expiry, iv, risk_free_rate, dividend_yield) -> float:
    # S * e^(-q*T) * N'(d1) * sqrt(T)  [per unit of vol]

def bs_call_theta(spot, strike, time_to_expiry, iv, risk_free_rate, dividend_yield) -> float:
    # Annualized form divided by 365 for per-calendar-day value

def bs_call_charm(spot, strike, time_to_expiry, iv, risk_free_rate, dividend_yield) -> float:
    # -e^(-q*T) * N'(d1) * [2*(r-q)*T - d2*sigma*sqrt(T)] / (2*T*sigma*sqrt(T))
```

### 4b. `finance/rebalance.py` (new module, extracted from `_backtest_steps.py`)

Receives `_should_rebalance` (renamed `should_rebalance`) relocated from `_backtest_steps.py`.
`_backtest_steps.py` imports from here after the move. No behavior change.

```python
def should_rebalance(
    current_weights: pd.Series,
    target_weights: pd.Series,
    rule: RebalanceRule,
    band: float = DRIFT_BAND_RELATIVE,
) -> bool: ...
```

### 4c. `finance/leverage.py` — `get_live_contracts` (extracted from `_backtest_steps.py`)

Receives `_live_contracts` (renamed `get_live_contracts`) relocated from `_backtest_steps.py`.
Belongs here: it only depends on `LeapsLedger` and `pd.Timestamp`. `_backtest_steps.py` imports
from `leverage` after the move.

```python
def get_live_contracts(
    ledger: LeapsLedger,
    current_date: pd.Timestamp,
) -> list[LeapsContract]: ...
```

### 4d. `finance/greeks.py` (new module)

```python
def compute_contract_greeks(
    contract: LeapsContract,
    spot: float,
    iv: float,
    as_of_date: pd.Timestamp,
    risk_free_rate: float = 0.0,
    leaps_scale: float = 1.0,
) -> ContractGreeks:
    """Compute all BS greeks for one LeapsContract at caller-supplied market conditions."""

def compute_portfolio_greeks(
    portfolio: LivePortfolio,
    spot: float,
    iv: float,
    risk_free_rate: float = 0.0,
) -> PortfolioGreeks:
    """Compute greeks for all active contracts in a LivePortfolio.

    Returns PortfolioGreeks with empty contracts tuple and all-zero net greeks
    when portfolio.leaps_contracts is empty.
    """
```

`leaps_scale` per contract is pulled directly from each `(contract, scale)` pair in
`portfolio.leaps_contracts`. No need for `PriceData` — callers supply today's market
scalars directly.

### 4e. `finance/portfolio_manager.py` (new module)

All functions are pure except `compute_gtt_status` (delegates network I/O to
`fetch_gtt_signal_data`).

```python
def as_live_portfolio(
    result: BacktestResult,
    gtt_active: bool = False,
) -> LivePortfolio:
    """Bridge from a completed BacktestResult to a LivePortfolio.

    Uses get_live_contracts(result.final_state.leaps_ledger, final_date) to filter
    active contracts. Pulls leaps_scale from result.final_state.leaps_scale.
    """

def compute_nav_breakdown(
    portfolio: LivePortfolio,
    leaps_mtm: float = 0.0,
) -> NavBreakdown:
    """Decompose LivePortfolio NAV into components.

    leaps_mtm: caller-supplied current LEAPS mark-to-market (e.g. from bs_call_price).
    """

def compute_holdings_view(
    portfolio: LivePortfolio,
    nav: NavBreakdown,
) -> tuple[HoldingView, ...]:
    """Compute per-asset weight drift against target_weights."""

def compute_rebalance_plan(
    portfolio: LivePortfolio,
    nav: NavBreakdown,
    rebalance_rule: RebalanceRule,
    is_rebal_date: bool,
    is_month_end: bool,
) -> RebalancePlan:
    """Simulate rebalance trades without executing them.

    Pure. Uses should_rebalance() from rebalance.py. Does not mutate portfolio.
    Note: assumes Long regime. Callers should inspect portfolio.gtt_regime first
    and handle Defensive portfolios separately.
    """

def compute_volatility_report(
    portfolio: LivePortfolio,
    return_data: ReturnData,
) -> VolatilityReport:
    """Run the full vol stack on current portfolio weights.

    Calls build_volatility_model(return_data, as_of_date=portfolio.as_of_date)
    then build_vol_contribution_table(). Weights derived from portfolio.holdings / nav.base_nav.
    """

def compute_gtt_status(  # I/O boundary — not pure
    as_of_date: pd.Timestamp,
    vix_p90_threshold: float,
    start_date: str,
    equity_prices: pd.Series | None = None,
) -> GttStatus:
    """Fetch current GTT signal and return structured status.

    Delegates to fetch_gtt_signal_data (two network calls: FRED + yfinance).
    vix_p90_threshold must be pre-computed by the caller to avoid look-ahead.
    Callers running multiple management functions on the same date should call
    this once and reuse the GttStatus.signal_data.
    """
```

### 4f. `finance/dca_signal.py` (new module)

Implements the multi-factor LEAPS DCA entry score. The signal is computed from OHLC prices
and an IV series. Both are already available from `PriceData` with one extension (see §4g).

**Factor decomposition:**

| Factor | Source | Score direction | Weight |
|---|---|---|---|
| RSI (14-day) | close prices | Inverted: high RSI → low score | 20% |
| Stochastic %D (5/3) | high/low/close | Inverted: overbought → low score | 15% |
| IV Percentile (252-day) | IV series | Inverted: expensive vol → low score | 35% |
| MACD Histogram regime | close prices | Bullish histogram → 100, else 0 | 30% |

MACD additionally acts as a **trend gate**: when the histogram is negative the entire composite
is multiplied by `macd_gate_floor` (default 0.5), not just its 30% slice. A debounced bearish
flag (`macd_bearish_confirmed`) requires 3 consecutive negative sessions before treating the
trend as broken.

**Data inputs and library integration:**

The existing `PriceData` stores only adjusted close; Stochastic %D requires high and low prices.
`build_price_data` already fetches these from yfinance but does not store them. `PriceData` needs
a new optional field `ohlcv: pd.DataFrame` (DatetimeIndex × [open, high, low, close, volume]).
This is opt-in via a `fetch_ohlcv: bool = False` parameter on `build_price_data`, keeping the
existing interface unchanged.

IV is sourced from `PriceData.vol_prices` (populated when `fetch_vol_indices=True`). The
`fetch_market_data` function from the original script is replaced entirely by existing
`build_price_data(fetch_vol_indices=True, fetch_ohlcv=True)`.

```python
def compute_leaps_dca_signal(
    price_data: PriceData,
    ticker: str,
    as_of_date: pd.Timestamp,
    hold_pctile: float = 25.0,
    aggressive_pctile: float = 75.0,
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal_window: int = 9,
    stoch_k: int = 5,
    stoch_d: int = 3,
    iv_window: int = 252,
    lookback: int = 504,
    min_lookback: int = 252,
    w_rsi: float = 0.20,
    w_stoch: float = 0.15,
    w_iv: float = 0.35,
    w_macd: float = 0.30,
    use_macd_gate: bool = True,
    macd_gate_floor: float = 0.5,
) -> LeapsDcaSignal:
    """Compute the multi-factor DITM LEAPS DCA entry signal for a single ticker.

    Arguments:
        price_data: PriceData built with fetch_vol_indices=True and fetch_ohlcv=True.
        ticker: Underlying ticker to evaluate (e.g. "VTI").
        as_of_date: Evaluation date. Signal uses only data up to and including this date.
        hold_pctile: Score percentile below which alpha_t = 0.0 ("HOLD"). Default 25.
        aggressive_pctile: Score percentile at or above which alpha_t = 1.0
            ("AGGRESSIVE_SWEEP"). Default 75.
        ... (remaining params mirror the original compute_entry_score signature)

    Returns:
        LeapsDcaSignal with entry score, percentile, alpha_t, DCA action, and
        all intermediate factor values.

    Raises:
        ValueError: If ticker is not in price_data.prices.
        ValueError: If price_data.ohlcv is empty (fetch_ohlcv was not requested).
        ValueError: If price_data.vol_prices has no column for ticker's IV proxy
            (fetch_vol_indices was not requested, or no vol index is mapped).
        ValueError: If w_rsi + w_stoch + w_iv + w_macd does not sum to 1.0 within 1e-6.
        ValueError: If fewer than min_lookback trading days of data are available
            up to as_of_date.
    """
```

**No-lookahead guarantee:** `compute_leaps_dca_signal` slices `price_data` to
`[:as_of_date]` before all computations. The rolling IV percentile, MACD, RSI, and Stochastic
windows are computed on that truncated series. This matches the library's existing T1 invariant.

### 4g. `finance/data.py` — add OHLCV support

Minor additive change: two new items in `PriceData` and `build_price_data`.

```python
@dataclass(frozen=True)
class PriceData:
    prices: pd.DataFrame
    dividends: pd.DataFrame
    vol_prices: pd.DataFrame
    tickers: tuple[str, ...]
    start_date: str
    end_date: str
    spliced: bool
    ohlcv: pd.DataFrame  # NEW: DatetimeIndex × MultiIndex(ticker, field)
                         #      fields: open, high, low, close, volume
                         #      Empty DataFrame when fetch_ohlcv=False (default)
```

`build_price_data` gains `fetch_ohlcv: bool = False`. When True, fetches OHLCV for all
`asset_tickers` via a single `yf.download(..., auto_adjust=True)` call (the full response
already contains OHLCV; current code discards everything except `Close`).

### 4h. `finance/__init__.py` — public API surface

```python
# Existing
from finance.portfolio import run_backtest

# Existing types now explicitly re-exported
from finance._portfolio_types import BacktestResult, PortfolioConfig, PortfolioState, GttConfig

# New: live management
from finance.portfolio_manager import (
    LivePortfolio,
    NavBreakdown,
    HoldingView,
    RebalancePlan,
    TradeOrder,
    VolatilityReport,
    GttStatus,
    as_live_portfolio,
    compute_nav_breakdown,
    compute_holdings_view,
    compute_rebalance_plan,
    compute_volatility_report,
    compute_gtt_status,
)

# New: LEAPS greeks
from finance.greeks import (
    ContractGreeks,
    PortfolioGreeks,
    compute_contract_greeks,
    compute_portfolio_greeks,
)

# New: LEAPS DCA entry signal
from finance.dca_signal import (
    LeapsDcaSignal,
    compute_leaps_dca_signal,
)
```

---

## 5. Step-by-Step Implementation Roadmap

### Phase 0: Refactor private helpers (prerequisite for all phases)

**0a.** Move `_live_contracts` from `_backtest_steps.py` to `leverage.py`; rename to
`get_live_contracts`. Update all call sites in `_backtest_steps.py` to import from `leverage`.

**0b.** Create `finance/rebalance.py`. Move `_should_rebalance` there; rename to
`should_rebalance`. Update its call site in `_backtest_steps.py`.

- Completion signal: `uv run pytest` passes unchanged; `uv run mypy src/` clean.
- No behavior change — pure refactor.

### Phase 1: `BacktestResult.final_state` and bridge

**1a.** Add `final_state: PortfolioState` (required field) to `BacktestResult` in
`_portfolio_types.py`.

**1b.** Pass loop-end `state` into `BacktestResult(...)` in `run_backtest`.

**1c.** Implement `as_live_portfolio(result, gtt_active) -> LivePortfolio` in
`portfolio_manager.py`. Uses `get_live_contracts(result.final_state.leaps_ledger, final_date)`.

- Completion signal: existing tests pass with no behavioral change.

### Phase 2: Missing BS greeks in `leverage.py`

**2a.** Add `bs_call_gamma`, `bs_call_vega`, `bs_call_theta`, `bs_call_charm` as pure functions.

- Completion signal: unit tests against analytic reference values (ATM call, T=1yr, σ=0.20).

### Phase 3: `finance/greeks.py`

**3a.** Define `ContractGreeks`, `PortfolioGreeks` frozen dataclasses.

**3b.** Implement `compute_contract_greeks(contract, spot, iv, as_of_date, risk_free_rate,
leaps_scale)`.

**3c.** Implement `compute_portfolio_greeks(portfolio, spot, iv, risk_free_rate)`.
Returns all-zero `PortfolioGreeks` for empty `portfolio.leaps_contracts`.

- Completion signal: unit tests with known inputs verifying per-contract and aggregate greeks.

### Phase 4: `finance/portfolio_manager.py`

**4a.** Define `NavBreakdown`, `HoldingView`, `TradeOrder`, `RebalancePlan`, `VolatilityReport`,
`GttStatus`, `LivePortfolio` (with `__post_init__` validation).

**4b.** Implement `compute_nav_breakdown()` and `compute_holdings_view()`.

**4c.** Implement `compute_rebalance_plan()`. Uses `should_rebalance()` from `rebalance.py`.

**4d.** Implement `compute_volatility_report()`. Calls `build_volatility_model()` and
`build_vol_contribution_table()` from `volatility.py`.

**4e.** Implement `compute_gtt_status()`. Thin wrapper over `fetch_gtt_signal_data()`.
Marked `# pragma: no cover` (network I/O).

- Completion signal: unit tests for 4b–4d with synthetic `LivePortfolio`. 4e excluded from coverage.

### Phase 5: `finance/dca_signal.py` and `data.py` OHLCV extension

**5a.** Add `ohlcv: pd.DataFrame` field to `PriceData` (default empty `pd.DataFrame`).
Add `fetch_ohlcv: bool = False` parameter to `build_price_data`. When True, retain the full
OHLCV response from `yf.download` instead of discarding everything except `Close`.

- Completion signal: `build_price_data(..., fetch_ohlcv=True)` returns non-empty `ohlcv`;
  existing tests pass unchanged (default `fetch_ohlcv=False`).

**5b.** Define `LeapsDcaSignal` frozen dataclass in `dca_signal.py`.

**5c.** Implement `compute_leaps_dca_signal()`.
- Slices `price_data` to `[:as_of_date]` before all computations (T1 invariant).
- IV sourced from `price_data.vol_prices[ticker_iv_column]` where the column key is the
  ticker's mapped vol index (e.g. `"VTI"` → `"VTI_IV"` from `ASSET_VOL_INDEX`).
- High/low for Stochastic sourced from `price_data.ohlcv[ticker][["high", "low"]]`.
- Raises `ValueError` if `ohlcv` or the required `vol_prices` column is empty.

- Completion signal: unit tests with 3 years of synthetic OHLCV + IV data verify score
  bounds [0, 100], MACD gate behavior, alpha_t clipping, and action classification.

### Phase 6: Public surface and docs

**6a.** Update `finance/__init__.py` with all re-exports.

**6b.** Docstrings on all new public functions and types; `uv run mypy src/` strict clean.

---

## 6. System & Test Invariants

| ID | Type | Invariant | Oracle / Bound |
|----|------|-----------|----------------|
| I1 | [ATOMIC] | `NavBreakdown.total_nav == base_nav + leaps_nav + defensive_sleeve + leaps_pool` | Arithmetic; assert within 1e-9 |
| I2 | [ATOMIC] | `sum(h.actual_weight for h in holdings_view) <= 1.0 + 1e-9` (can be < 1 with defensive sleeve) | Arithmetic |
| I3 | [ATOMIC] | `sum(t.trade_amount for t in plan.trades) ≈ 0.0` (no cash created/destroyed) | Conservation; assert within 1e-6 |
| I4 | [ATOMIC] | `ContractGreeks.delta ∈ (0, 1)` for any LEAPS call | Domain: calls have delta in (0, 1) |
| I5 | [ATOMIC] | `ContractGreeks.gamma > 0` for long call | Domain |
| I6 | [ATOMIC] | `ContractGreeks.theta < 0` for long call with t > TIME_FLOOR | Time decay always negative |
| I7 | [ATOMIC] | `ContractGreeks.price` matches `bs_call_price()` with identical inputs | Internal consistency; assert within 1e-9 |
| I8 | [ATOMIC] | `RebalancePlan.would_trigger == False` when QUARTERLY and `is_rebal_date == False` | `should_rebalance` contract |
| I9 | [ATOMIC] | `LivePortfolio.__post_init__` raises `ValueError` for any contract with `expiry_date <= as_of_date` | Validated with a known-expired contract |
| I10 | [ATOMIC] | `LivePortfolio.__post_init__` raises `ValueError` for `target_weights` not summing to 1.0 ± 1e-6 | Validation |
| I11 | [INTEGRATION] | `as_live_portfolio(result)` produces `NavBreakdown.total_nav ≈ result.nav_series.iloc[-1]` | End-of-backtest NAV identity; requires full backtest run |
| I12 | [INTEGRATION] | `compute_portfolio_greeks(portfolio, ...).net_delta ∈ (0, n_contracts_total * 100)` | Domain bounds; requires live LEAPS |
| I13 | [INTEGRATION] | Existing tests produce identical `nav_series`, `weight_history`, `return_series` after Phase 0–1 | Non-regression |
| I14 | [ATOMIC] | `LeapsDcaSignal.entry_score ∈ [0, 100]` | Score is a weighted sum of bounded components; clipping applied |
| I15 | [ATOMIC] | `LeapsDcaSignal.alpha_t ∈ [0, 1]` | `np.clip` applied after linear interpolation |
| I16 | [ATOMIC] | `LeapsDcaSignal.dca_action` is one of `"HOLD"`, `"TRANCHE"`, `"AGGRESSIVE_SWEEP"` | Exhaustive case coverage |
| I17 | [ATOMIC] | `compute_leaps_dca_signal` raises `ValueError` when `price_data.ohlcv` is empty | Validated before any computation |
| I18 | [ATOMIC] | `compute_leaps_dca_signal` with `as_of_date=T` uses no data after T | Slice `[:as_of_date]` applied before all rolling computations (T1) |
| I19 | [ATOMIC] | `w_rsi + w_stoch + w_iv + w_macd == 1.0 ± 1e-6`; raises `ValueError` otherwise | Validated in `__post_init__`-style guard at function entry |
| I20 | [INTEGRATION] | `compute_leaps_dca_signal` score percentile shifts monotonically as new favorable data is added | Requires multi-date sweep test over synthetic series |

**Temporal bound (T1):** All pure management functions accept `as_of_date` as an explicit
argument. They must never infer the date from internal state fields. This preserves the
library's no-lookahead invariant. `compute_leaps_dca_signal` enforces this by slicing
`price_data` to `[:as_of_date]` before computing any rolling window.

**Realism gate:** I11 and I12 require a backtest run against real price data, not synthetic
mock Series. A minimum 3-year slice of real VTI prices should anchor the integration tests.

---

## 7. Known Assumptions

1. **`get_live_contracts` and `should_rebalance` are moved before any new code is written.**
   Phase 0 is a strict prerequisite.

2. **`BacktestResult` is only ever constructed inside `run_backtest`.** Adding `final_state`
   as a required field is non-breaking.

3. **`leaps_mtm` is caller-supplied in `compute_nav_breakdown`.** The user knows their
   LEAPS value from their brokerage or from calling `bs_call_price` directly. The library
   does not fetch it automatically.

4. **`compute_gtt_status` makes two network calls (FRED + yfinance).** Callers running
   multiple management functions on the same date should call this once and reuse the result.

5. **`compute_volatility_report` uses realized holdings weights,** not target weights. This
   reflects actual risk exposure and is the correct input for risk decomposition.

6. **`compute_rebalance_plan` assumes Long regime.** Defensive-regime rebalance behavior is
   more complex (GTT re-sweep logic) and is deferred. Callers should inspect
   `portfolio.gtt_regime` before calling.

7. **`PriceData.ohlcv` is opt-in and empty by default.** All existing code that does not
   request OHLCV is unaffected. `compute_leaps_dca_signal` will raise `ValueError` if
   called with a `PriceData` built without `fetch_ohlcv=True`.

8. **IV for the DCA signal comes from `PriceData.vol_prices`, not a live option chain.**
   The original script merged live NTM option chain IV for the most-recent observation.
   This is not replicated — the library uses the vol index proxy (VIX for VTI) throughout
   for consistency and reproducibility. Live IV from an option chain can be substituted by
   the caller by patching the last row of `vol_prices` before calling
   `compute_leaps_dca_signal`.

9. **MACD parameter names:** The `signal` window parameter is named `macd_signal_window`
   in the library (not `macd_signal`) to avoid shadowing Python's built-in `signal` module.

---

## 8. Edge Cases & Pre-Mortem Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `compute_volatility_report` called with `return_data` not covering `portfolio.as_of_date` | Medium | `build_volatility_model` already raises `ValueError` for out-of-range dates; propagates naturally |
| `compute_rebalance_plan` called with Defensive portfolio | Medium | Documented limitation in 4c; returns plan assuming Long; caller checks `portfolio.gtt_regime` |
| LEAPS near expiry: theta and charm diverge as `time_to_expiry → 0` | Medium | `TIME_FLOOR` applied in all BS functions; document the floor |
| `RebalancePlan.leaps_trim` for DRIFT rule cannot replicate `leaps_scale` update without executing it | Medium | `leaps_trim` reports the dollar amount only; scale update is not simulated. Documented limitation. |
| `as_live_portfolio` when backtest had no LEAPS: `final_state.leaps_ledger` is None | Low | Returns empty `leaps_contracts` tuple; tested explicitly |
| `compute_leaps_dca_signal` called for a ticker with no `ASSET_VOL_INDEX` mapping (e.g. IWM) | Medium | Falls back to VIX column if present; raises `ValueError` if vol_prices has no usable column. Document the fallback behavior explicitly. |
| `PriceData.ohlcv` MultiIndex column structure differs across yfinance versions | Low | Pin yfinance version; add a parsing guard in `build_price_data` that normalizes the MultiIndex to `(ticker, field)` before storing |
| Stochastic %D produces NaN for the first `stoch_k + stoch_d - 1` days; score percentile window requires `min_lookback` days | Medium | `ValueError` raised if fewer than `min_lookback` days available after slicing to `as_of_date`; NaN rows are dropped before percentile computation |
| MACD `macd_bearish_confirmed` uses `groupby(...cumsum())` pattern which has O(n²) worst case on long series | Low | Acceptable for 5-year daily series (~1250 rows); document as a known performance characteristic |
