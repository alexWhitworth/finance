"""Live portfolio management — bridge from backtest result to live portfolio state.

Provides the LivePortfolio, NavBreakdown, HoldingView, TradeOrder, RebalancePlan,
VolatilityReport, and GttStatus dataclasses, and the pure functions
as_live_portfolio(), compute_nav_breakdown(), compute_holdings_view(),
compute_rebalance_plan(), and compute_volatility_report(), plus the I/O
boundary compute_gtt_status(). All functions except compute_gtt_status are pure.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from finance._portfolio_types import BacktestResult
from finance.consts import DRIFT_BAND_RELATIVE, LEAPS_KEY_SUFFIX
from finance.gtt import GttSignalData, fetch_gtt_signal_data
from finance.leverage import LeapsContract, RebalanceRule, get_live_contracts
from finance.rebalance import should_rebalance
from finance.returns import ReturnData
from finance.volatility import (
    VolatilityModel,
    build_vol_contribution_table,
    build_volatility_model,
    forecast_portfolio_vol,
)


@dataclass(frozen=True)
class LivePortfolio:
    """User's current brokerage portfolio state as of a specific date.

    Self-contained and user-constructible without understanding backtest internals.
    All fields are immutable. Validates target_weights sum, leaps_scale range, and
    contract expiry on construction.

    Attributes:
        as_of_date: Date these holdings reflect.
        holdings: Base asset ticker → current dollar value.
        target_weights: Target allocation. Must sum to 1.0 within 1e-6.
        leaps_contracts: Active LEAPS contracts with surviving fraction
            (scale ∈ (0, 1]). Empty tuple if no LEAPS.
        gtt_regime: Current GTT regime: 1=Long, 0=Defensive, None=inactive.
        defensive_sleeve: Dollar value in GTT defensive allocation. Default 0.0.
        leaps_pool: Force-closed LEAPS proceeds parked during defensive window.
            Default 0.0.

    Raises:
        ValueError: If target_weights does not sum to 1.0 within 1e-6.
        ValueError: If any leaps_scale is not in (0, 1].
        ValueError: If any contract.expiry_date <= as_of_date.
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
            raise ValueError(
                f"target_weights must sum to 1.0; got {total:.8f}"
            )
        for contract, scale in self.leaps_contracts:
            if not (0.0 < scale <= 1.0):
                raise ValueError(
                    f"leaps_scale must be in (0, 1]; got {scale} for contract "
                    f"purchased {contract.purchase_date.date()}"
                )
            if contract.expiry_date <= self.as_of_date:
                raise ValueError(
                    f"contract.expiry_date ({contract.expiry_date.date()}) must be "
                    f"> as_of_date ({self.as_of_date.date()})"
                )


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


def compute_nav_breakdown(
    portfolio: LivePortfolio,
    leaps_mtm: float = 0.0,
) -> NavBreakdown:
    """Decompose LivePortfolio NAV into base, LEAPS, defensive, and pool components.

    Arguments:
        portfolio: LivePortfolio whose holdings to decompose.
        leaps_mtm: Current mark-to-market value of all active LEAPS contracts
            (caller-supplied; not computed here). Defaults to 0.0.

    Returns:
        NavBreakdown with total_nav == base_nav + leaps_nav + defensive_sleeve
        + leaps_pool within 1e-9 (I1).
    """
    base_nav = sum(portfolio.holdings.values())
    total_nav = base_nav + leaps_mtm + portfolio.defensive_sleeve + portfolio.leaps_pool
    return NavBreakdown(
        base_nav=base_nav,
        leaps_nav=leaps_mtm,
        defensive_sleeve=portfolio.defensive_sleeve,
        leaps_pool=portfolio.leaps_pool,
        total_nav=total_nav,
    )


def compute_holdings_view(
    portfolio: LivePortfolio,
    nav: NavBreakdown,
) -> tuple[HoldingView, ...]:
    """Compute per-asset weight drift against target_weights.

    Arguments:
        portfolio: LivePortfolio with base asset holdings and target weights.
        nav: NavBreakdown providing total_nav for weight normalization.

    Returns:
        Tuple of HoldingView — one per ticker in portfolio.holdings. The sum of
        actual_weight values is <= 1.0 + 1e-9 (I2) since holdings cover only base
        assets; defensive_sleeve and leaps components are excluded.
    """
    total = nav.total_nav
    views = []
    for ticker, value in portfolio.holdings.items():
        actual_w = value / total if total > 0.0 else 0.0
        target_w = portfolio.target_weights.get(ticker, 0.0)
        drift = actual_w - target_w
        rel = drift / target_w if target_w != 0.0 else None
        views.append(
            HoldingView(
                ticker=ticker,
                dollar_value=value,
                actual_weight=actual_w,
                target_weight=target_w,
                weight_drift=drift,
                relative_drift=rel,
            )
        )
    return tuple(views)


def compute_leaps_holdings_view(
    portfolio: LivePortfolio,
    nav: NavBreakdown,
) -> tuple[HoldingView, ...]:
    """Compute weight-drift HoldingView rows for the LEAPS sleeve(s) in target_weights.

    compute_holdings_view() only covers portfolio.holdings (base assets) — the
    LEAPS sleeve is tracked separately via leaps_contracts/nav.leaps_nav and is
    otherwise absent from a holdings-drift view. This uses the same key
    detection (LEAPS_KEY_SUFFIX) and proportional-split logic
    compute_rebalance_plan() applies internally for its own DRIFT check, so
    callers can display LEAPS weight drift alongside base assets from
    compute_holdings_view().

    Arguments:
        portfolio: LivePortfolio providing target_weights.
        nav: NavBreakdown providing leaps_nav and total_nav.

    Returns:
        Tuple of HoldingView, one per key in target_weights ending with
        LEAPS_KEY_SUFFIX. Empty tuple if no such key is present.
    """
    leaps_keys = [k for k in portfolio.target_weights if k.endswith(LEAPS_KEY_SUFFIX)]
    if not leaps_keys:
        return ()

    total_nav = nav.total_nav
    leaps_weight = nav.leaps_nav / total_nav if total_nav > 0.0 else 0.0
    key_target_sum = sum(portfolio.target_weights[k] for k in leaps_keys)

    views = []
    for k in leaps_keys:
        share = portfolio.target_weights[k] / key_target_sum if key_target_sum > 0.0 else 0.0
        actual_w = leaps_weight * share
        target_w = portfolio.target_weights[k]
        drift = actual_w - target_w
        views.append(
            HoldingView(
                ticker=k,
                dollar_value=nav.leaps_nav * share,
                actual_weight=actual_w,
                target_weight=target_w,
                weight_drift=drift,
                relative_drift=drift / target_w if target_w != 0.0 else None,
            )
        )
    return tuple(views)


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
    """Simulated rebalance outcome for a LivePortfolio. Pure: does not mutate portfolio.

    Attributes:
        as_of_date: Date of the simulation.
        would_trigger: Whether the rebalance rule fires at this date.
        trigger_reason: One of: quarterly_scheduled | drift_threshold | not_triggered.
        trades: Per-asset buy/sell instructions. Empty tuple if not triggered.
        leaps_trim: Dollar reduction to LEAPS (positive = LEAPS partially closed).
            Non-zero only when DRIFT rule fires and LEAPS overweight.
        holdings_view: Per-asset drift breakdown before rebalance.
    """

    as_of_date: pd.Timestamp
    would_trigger: bool
    trigger_reason: str
    trades: tuple[TradeOrder, ...]
    leaps_trim: float
    holdings_view: tuple[HoldingView, ...]


def compute_rebalance_plan(
    portfolio: LivePortfolio,
    nav: NavBreakdown,
    rebalance_rule: RebalanceRule,
    is_rebal_date: bool,
    is_month_end: bool,
) -> RebalancePlan:
    """Simulate rebalance trades without executing them.

    Determines whether the rebalance rule fires, then computes trade orders to
    bring each base asset from its current realized weight to its target weight.
    Assumes Long regime — callers must guard on portfolio.gtt_regime before
    calling when defensive allocation may be active (R-001).

    Trade conservation invariant (I3): sum(t.trade_amount) ≈ 0.0 within 1e-6
    when trades are non-empty. Each asset is reallocated from its current dollar
    value to target_weight * base_nav; since sum(target_weights) == 1.0 and
    the base_nav is fixed, the sum of target_values equals the sum of current
    values, so trades cancel exactly.

    leaps_trim is the dollar overshoot of the LEAPS sleeve relative to its
    target fraction of total_nav. Non-zero whenever the rebalance actually
    fires (QUARTERLY or DRIFT) and the LEAPS sleeve is overweight — QUARTERLY
    has no tolerance band, so it trims LEAPS back to target on every scheduled
    date, while DRIFT only fires (and therefore only trims) once drift exceeds
    the band.

    Arguments:
        portfolio: LivePortfolio providing holdings, target_weights, and
            leaps_contracts.
        nav: NavBreakdown from compute_nav_breakdown(), providing base_nav and
            total_nav.
        rebalance_rule: RebalanceRule.QUARTERLY or RebalanceRule.DRIFT.
        is_rebal_date: True when the caller has determined the date is a
            scheduled quarterly rebalance date. Ignored for DRIFT rule.
        is_month_end: True when the caller has determined the date is a
            month-end. Used by the DRIFT rule's check cadence.

    Returns:
        RebalancePlan with would_trigger, trigger_reason, trades, leaps_trim,
        and holdings_view populated.

    Notes:
        LEAPS weight in the drift check is computed as leaps_nav / total_nav,
        and the target LEAPS fraction is the sum of target_weights for any key
        ending with '_LEAPS'. leaps_trim reports the dollar overshoot only;
        no scale update is simulated (R-007).
    """
    holdings_view = compute_holdings_view(portfolio, nav)
    base_nav = nav.base_nav
    total_nav = nav.total_nav

    # Determine trigger
    would_trigger = False
    trigger_reason = "not_triggered"

    if rebalance_rule == RebalanceRule.QUARTERLY:
        if is_rebal_date:
            would_trigger = True
            trigger_reason = "quarterly_scheduled"
    else:  # DRIFT
        if is_month_end:
            # Build current and target weight Series over base + LEAPS
            weights_now: dict[str, float] = {
                h.ticker: h.actual_weight for h in holdings_view
            }
            if total_nav > 0.0:
                leaps_weight = nav.leaps_nav / total_nav
            else:
                leaps_weight = 0.0
            # Add a synthetic LEAPS key if LEAPS present
            leaps_keys = [
                k for k in portfolio.target_weights if k.endswith(LEAPS_KEY_SUFFIX)
            ]
            for k in leaps_keys:
                weights_now[k] = (
                    leaps_weight * (portfolio.target_weights[k] / sum(
                        portfolio.target_weights[lk] for lk in leaps_keys
                    ))
                    if leaps_keys and sum(portfolio.target_weights[lk] for lk in leaps_keys) > 0
                    else 0.0
                )
            current_w = pd.Series(weights_now)
            target_w = pd.Series(portfolio.target_weights)
            if should_rebalance(current_w, target_w, RebalanceRule.DRIFT, DRIFT_BAND_RELATIVE):
                would_trigger = True
                trigger_reason = "drift_threshold"

    if not would_trigger:
        return RebalancePlan(
            as_of_date=portfolio.as_of_date,
            would_trigger=False,
            trigger_reason="not_triggered",
            trades=(),
            leaps_trim=0.0,
            holdings_view=holdings_view,
        )

    # Build trade orders: reallocate base_nav by target_weights (base assets only)
    base_target_keys = [
        k for k in portfolio.target_weights if not k.endswith(LEAPS_KEY_SUFFIX)
    ]
    leaps_target_fraction = sum(
        v for k, v in portfolio.target_weights.items() if k.endswith(LEAPS_KEY_SUFFIX)
    )
    # Base target weights normalized over base assets only
    base_target_sum = sum(portfolio.target_weights[k] for k in base_target_keys)
    if base_target_sum > 0.0:
        base_target_norm = {
            k: portfolio.target_weights[k] / base_target_sum for k in base_target_keys
        }
    else:
        n = len(base_target_keys)
        base_target_norm = dict.fromkeys(base_target_keys, 1.0 / n) if n > 0 else {}

    holding_map = {h.ticker: h for h in holdings_view}
    orders = []
    for ticker in base_target_keys:
        cur_val = holding_map[ticker].dollar_value if ticker in holding_map else 0.0
        tgt_weight = portfolio.target_weights.get(ticker, 0.0)
        tgt_val = base_nav * base_target_norm.get(ticker, 0.0)
        cur_weight = cur_val / total_nav if total_nav > 0.0 else 0.0
        orders.append(
            TradeOrder(
                ticker=ticker,
                current_value=cur_val,
                target_value=tgt_val,
                trade_amount=tgt_val - cur_val,
                current_weight=cur_weight,
                target_weight=tgt_weight,
            )
        )

    # Zero-target sell orders for stray holdings absent from target_weights (I3).
    base_target_set = set(base_target_keys)
    for ticker, cur_val in portfolio.holdings.items():
        if ticker in base_target_set or ticker.endswith(LEAPS_KEY_SUFFIX):
            continue
        cur_weight = cur_val / total_nav if total_nav > 0.0 else 0.0
        orders.append(
            TradeOrder(
                ticker=ticker,
                current_value=cur_val,
                target_value=0.0,
                trade_amount=-cur_val,
                current_weight=cur_weight,
                target_weight=0.0,
            )
        )

    # leaps_trim: applies whenever the rebalance actually fires — QUARTERLY or
    # DRIFT — when the LEAPS sleeve is overweight relative to its target
    # fraction. would_trigger is unconditionally True past this point.
    leaps_trim = 0.0
    leaps_nav = nav.leaps_nav
    target_leaps_nav = total_nav * leaps_target_fraction
    if leaps_nav > target_leaps_nav:
        leaps_trim = leaps_nav - target_leaps_nav

    return RebalancePlan(
        as_of_date=portfolio.as_of_date,
        would_trigger=True,
        trigger_reason=trigger_reason,
        trades=tuple(orders),
        leaps_trim=leaps_trim,
        holdings_view=holdings_view,
    )


def leaps_trim_as_trade_order(
    leaps_view: HoldingView,
    leaps_trim: float,
) -> TradeOrder | None:
    """Represent a RebalancePlan's leaps_trim as a TradeOrder for display.

    compute_rebalance_plan() reports LEAPS overweight only as a dollar
    overshoot (RebalancePlan.leaps_trim) — not as an entry in plan.trades —
    because a partial LEAPS close isn't a simple buy/sell of a base asset and
    the freed proceeds are not redistributed to other trades (R-007). This is
    a read-only convenience for combining the LEAPS sleeve with base-asset
    trades in a single display table; the result is not part of plan.trades
    and does not participate in the trade-conservation invariant (I3).

    Arguments:
        leaps_view: HoldingView for the LEAPS sleeve, from
            compute_leaps_holdings_view().
        leaps_trim: RebalancePlan.leaps_trim from the same evaluation.

    Returns:
        TradeOrder representing the LEAPS partial close, or None when
        leaps_trim is 0.0.
    """
    if leaps_trim <= 0.0:
        return None
    return TradeOrder(
        ticker=leaps_view.ticker,
        current_value=leaps_view.dollar_value,
        target_value=leaps_view.dollar_value - leaps_trim,
        trade_amount=-leaps_trim,
        current_weight=leaps_view.actual_weight,
        target_weight=leaps_view.target_weight,
    )


@dataclass(frozen=True)
class VolatilityReport:
    """Portfolio volatility analysis snapshot.

    Attributes:
        as_of_date: Snapshot date.
        vol_model: Full VolatilityModel (ewma_vols, rolling_corr, cov_matrix).
        portfolio_vol: Forecasted annualized portfolio volatility (sigma_hat_p).
        contribution_table: DataFrame from build_vol_contribution_table().
            Columns: sigma_tilde, sigma_hat, rho_VTI, contrib.
        weights_used: Realized weights used in contribution computation.
    """

    as_of_date: pd.Timestamp
    vol_model: VolatilityModel
    portfolio_vol: float
    contribution_table: pd.DataFrame
    weights_used: pd.Series


def compute_volatility_report(
    portfolio: LivePortfolio,
    return_data: ReturnData,
) -> VolatilityReport:
    """Run the full vol stack on current portfolio weights.

    Calls build_volatility_model() and build_vol_contribution_table() from
    volatility.py. Weights are derived from realized holdings normalized to
    sum to 1.0 over base assets only (LEAPS excluded from weights_used).

    Arguments:
        portfolio: LivePortfolio whose holdings determine realized weights.
        return_data: ReturnData providing daily returns for the vol model.
            Must cover portfolio.as_of_date.

    Returns:
        VolatilityReport with portfolio_vol > 0 for any non-trivial portfolio.

    Raises:
        ValueError: If return_data does not cover portfolio.as_of_date
            (propagated from build_volatility_model).
    """
    vol_model = build_volatility_model(return_data, as_of_date=portfolio.as_of_date)

    # Build realized weights from holdings (base assets only)
    base_nav = sum(portfolio.holdings.values())
    if base_nav > 0.0:
        weights_dict = {t: v / base_nav for t, v in portfolio.holdings.items()}
    else:
        n = len(portfolio.holdings)
        weights_dict = dict.fromkeys(portfolio.holdings, 1.0 / n) if n > 0 else {}
    weights_used = pd.Series(weights_dict)

    portfolio_vol = forecast_portfolio_vol(weights_used, vol_model)
    contribution_table = build_vol_contribution_table(weights_used, return_data, vol_model)

    return VolatilityReport(
        as_of_date=portfolio.as_of_date,
        vol_model=vol_model,
        portfolio_vol=portfolio_vol,
        contribution_table=contribution_table,
        weights_used=weights_used,
    )


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
        price_vs_sma200: above | below | warming_up.
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


def compute_gtt_status(  # pragma: no cover
    as_of_date: pd.Timestamp,
    vix_p90_threshold: float,
    start_date: str,
    equity_prices: pd.Series | None = None,
) -> GttStatus:
    """Fetch current GTT signal and return structured status.

    Delegates to fetch_gtt_signal_data for FRED + yfinance I/O. Evaluates
    signal values at as_of_date. price_vs_sma200 is derived from the equity
    prices relative to the 200-day SMA at as_of_date. Not pure — I/O boundary.

    Arguments:
        as_of_date: Date of evaluation. Never inferred from system clock.
        vix_p90_threshold: Fixed P90 VIX threshold as a decimal (e.g. 0.272).
        start_date: ISO start date for data fetch (YYYY-MM-DD).
        equity_prices: Optional pre-fetched VTI price series. If None, fetched
            internally via yfinance.

    Returns:
        GttStatus with regime ∈ {0, 1} and price_vs_sma200 ∈
        {'above', 'below', 'warming_up'}.
    """
    from finance.consts import GTT_SMA_WINDOW

    end_date = str(as_of_date.date())
    signal_data = fetch_gtt_signal_data(
        start_date=start_date,
        end_date=end_date,
        vix_p90_threshold=vix_p90_threshold,
        equity_prices=equity_prices,
    )

    # Extract signal values at as_of_date (forward-fill to handle non-trading days)
    def _at(series: pd.Series, date: pd.Timestamp) -> int:
        aligned = series.reindex([date], method="ffill")
        val = aligned.iloc[0] if not aligned.empty else 0
        return int(val) if not pd.isna(val) else 0

    regime = _at(signal_data.position_mask, as_of_date)
    ue_sig = _at(signal_data.ue_signal, as_of_date)
    vix_sig = _at(signal_data.vix_signal, as_of_date)

    # Fetch raw VIX at as_of_date for vix_current
    vix_raw = yf.download(
        "^VIX", start=start_date, end=end_date, auto_adjust=True, progress=False
    )
    vix_series: pd.Series = (vix_raw["Close"].squeeze() / 100.0).rename("VIX")
    vix_aligned = vix_series.reindex([as_of_date], method="ffill")
    vix_current = float(vix_aligned.iloc[0]) if not vix_aligned.empty else float("nan")

    # price_vs_sma200 from equity prices
    if equity_prices is None:
        vti_raw = yf.download(
            "VTI", start=start_date, end=end_date, auto_adjust=True, progress=False
        )
        equity_prices = vti_raw["Close"].squeeze().rename("VTI")

    sma = equity_prices.rolling(window=GTT_SMA_WINDOW, min_periods=GTT_SMA_WINDOW).mean()
    price_at = equity_prices.reindex([as_of_date], method="ffill")
    sma_at = sma.reindex([as_of_date], method="ffill")

    if sma_at.empty or pd.isna(sma_at.iloc[0]):
        price_vs_sma200 = "warming_up"
    elif float(price_at.iloc[0]) >= float(sma_at.iloc[0]):
        price_vs_sma200 = "above"
    else:
        price_vs_sma200 = "below"

    return GttStatus(
        as_of_date=as_of_date,
        regime=regime,
        ue_signal=ue_sig,
        vix_signal=vix_sig,
        vix_current=vix_current,
        vix_threshold=vix_p90_threshold,
        price_vs_sma200=price_vs_sma200,
        signal_data=signal_data,
    )


def as_live_portfolio(
    result: BacktestResult,
    gtt_active: bool = False,
) -> LivePortfolio:
    """Convert a completed BacktestResult into a LivePortfolio.

    Uses get_live_contracts() to filter contracts from final_state.leaps_ledger
    to those still active as of the last backtest date. Pulls leaps_scale from
    final_state.leaps_scale. When the backtest had no LEAPS overlay or all
    contracts have expired, leaps_contracts is an empty tuple.

    Arguments:
        result: Completed BacktestResult from run_backtest. Must have
            final_state populated (requires F-003/F-004).
        gtt_active: Whether the GTT overlay was active. When True, gtt_regime
            is set from final_state.prev_regime. When False, gtt_regime is None.
            Pass True only when the backtest was configured with a gtt_signal;
            on a non-GTT backtest, prev_regime is always 1 (Long) and the
            returned gtt_regime is semantically meaningless.

    Returns:
        LivePortfolio reflecting the portfolio state at the last backtest date.
    """
    state = result.final_state
    if state.prev_date_ts is None:
        raise ValueError(
            "BacktestResult.final_state.prev_date_ts is None — "
            "run_backtest produced an empty date range."
        )
    as_of_date: pd.Timestamp = state.prev_date_ts

    # Build leaps_contracts: (contract, scale) pairs for live contracts only
    leaps_pairs: tuple[tuple[LeapsContract, float], ...]
    if state.leaps_ledger is None:
        leaps_pairs = ()
    else:
        live = get_live_contracts(state.leaps_ledger, as_of_date)
        leaps_pairs = tuple(
            (c, state.leaps_scale.get(c, 1.0)) for c in live
        )

    gtt_regime: int | None = state.prev_regime if gtt_active else None

    return LivePortfolio(
        as_of_date=as_of_date,
        holdings=dict(state.holdings),
        target_weights=dict(result.config.target_weights),
        leaps_contracts=leaps_pairs,
        gtt_regime=gtt_regime,
        defensive_sleeve=state.defensive_sleeve,
        leaps_pool=state.leaps_pool,
    )
