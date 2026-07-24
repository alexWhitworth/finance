"""LEAPS contract lifecycle — Black-Scholes pricing, contract creation, roll, and simulation.

All business logic is pure (no I/O). The module also owns the enumeration types
(AccountType, RebalanceRule, WeightStrategy) shared with portfolio.py.
"""

import enum
import math
from dataclasses import dataclass

import pandas as pd
from scipy import stats

from finance.consts import (
    CONTRACT_MULTIPLIER,
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_IV,
    DEFAULT_RISK_FREE_RATE,
    LEAPS_STRIKE_RATIO,
    LTCG_RATE,
    MIN_HOLD_DAYS,
    SIX_MONTHS_DAYS,
    TIME_FLOOR,
)

# ---------------------------------------------------------------------------
# Enumeration types (shared with portfolio.py)
# ---------------------------------------------------------------------------


class AccountType(enum.Enum):
    """Tax treatment applied to LEAPS gains at roll.

    Attributes:
        TAXABLE: Gains taxed at long-term capital gains rates on each roll.
        TAX_SHELTERED: No tax due on roll (IRA, 401k, etc.).
    """

    TAXABLE = "taxable"
    TAX_SHELTERED = "tax_sheltered"


class RebalanceRule(enum.Enum):
    """Determines when portfolio rebalancing is triggered.

    Attributes:
        QUARTERLY: Rebalance on the last business day of Mar/Jun/Sep/Dec.
    """

    QUARTERLY = "quarterly"


class WeightStrategy(enum.Enum):
    """Determines how target portfolio weights are computed.

    Attributes:
        USER_SPECIFIED: Weights supplied directly via PortfolioConfig.target_weights.
    """

    USER_SPECIFIED = "user_specified"


# ---------------------------------------------------------------------------
# Dataclass types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeapsConfig:
    """Configuration for a LEAPS overlay simulation.

    Attributes:
        iv: Constant implied volatility for Black-Scholes pricing. Default 0.18.
        ltcg_rate: Combined LTCG + NIIT rate applied on taxable rolls. Default 0.238.
        account_type: Governs whether tax is applied on roll.
        risk_free_rate: Continuously compounded risk-free rate used in BS pricing.
            Pass a scalar (constant) or use the per-date override in run_leaps_simulation.
            Default 0.0.
        dividend_yield: Continuously compounded annual dividend yield of the underlying.
            Default 0.0.
    """

    iv: float = DEFAULT_IV
    ltcg_rate: float = LTCG_RATE
    account_type: AccountType = AccountType.TAXABLE
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD


@dataclass(frozen=True)
class LeapsContract:
    """A single DITM VTI LEAPS call contract position.

    Attributes:
        purchase_date: Trade date.
        expiry_date: Option expiry (approximately 2 years from purchase_date).
        strike: Strike price; set at LEAPS_STRIKE_RATIO * spot_at_purchase.
        spot_at_purchase: VTI price on purchase_date.
        premium_paid: Per-share Black-Scholes call price at purchase.
        notional: spot_at_purchase * CONTRACT_MULTIPLIER (100-share multiplier).
        n_contracts: Number of contracts; float to allow fractional allocation.
        account_type: Tax treatment applied at roll.
        dividend_yield: Dividend yield used when this contract was priced.

    Notes:
        Total cost basis = premium_paid * CONTRACT_MULTIPLIER * n_contracts.
    """

    purchase_date: pd.Timestamp
    expiry_date: pd.Timestamp
    strike: float
    spot_at_purchase: float
    premium_paid: float
    notional: float
    n_contracts: float
    account_type: AccountType
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD


@dataclass(frozen=True)
class LeapsRollEvent:
    """Record of a single LEAPS roll transaction.

    Attributes:
        roll_date: Date the roll was executed.
        old_contract: Contract being closed.
        new_contract: Replacement contract opened with net proceeds.
        gain_realized: Mark-to-market value minus original total cost basis.
        tax_paid: LTCG tax on positive gain; 0.0 for TAX_SHELTERED accounts.
        net_proceeds: old_value - tax_paid, reinvested into new_contract.
    """

    roll_date: pd.Timestamp
    old_contract: LeapsContract
    new_contract: LeapsContract
    gain_realized: float
    tax_paid: float
    net_proceeds: float


@dataclass(frozen=True)
class LeapsLedger:
    """Complete transaction history for one LEAPS simulation.

    Attributes:
        contracts: All contracts ever created (includes both live and rolled-out).
        roll_events: All roll transactions executed during the simulation.
        account_type: Account type governing all contracts in this ledger.
    """

    contracts: tuple[LeapsContract, ...]
    roll_events: tuple[LeapsRollEvent, ...]
    account_type: AccountType


# ---------------------------------------------------------------------------
# Black-Scholes pure functions
# ---------------------------------------------------------------------------


def _bs_d1(spot: float, strike: float, t_years: float, iv: float, r: float, q: float) -> float:
    """Compute Black-Scholes d1 term.

    Arguments:
        spot: Current asset price.
        strike: Option strike price.
        t_years: Time to expiry in years (must be > 0).
        iv: Implied volatility (annualized).
        r: Continuously compounded risk-free rate.
        q: Continuously compounded dividend yield.

    Returns:
        d1 = (log(S/K) + (r - q + 0.5*sigma^2)*t) / (sigma*sqrt(t)).
    """
    return (math.log(spot / strike) + (r - q + 0.5 * iv**2) * t_years) / (iv * math.sqrt(t_years))


def bs_call_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    iv: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    """Compute Black-Scholes European call option price with continuous dividend yield.

    Arguments:
        spot: Current asset price (S).
        strike: Strike price (K).
        time_to_expiry: Time to expiry in years. Floored at TIME_FLOOR.
        iv: Implied volatility (annualized, e.g. 0.18 for 18%).
        risk_free_rate: Continuously compounded risk-free rate. Default 0.0.
        dividend_yield: Continuously compounded dividend yield (q). Default 0.0.

    Returns:
        Call option price per share.

    References:
        @article{black1973pricing,
            title={The Pricing of Options and Corporate Liabilities},
            author={Black, Fischer and Scholes, Myron},
            journal={Journal of Political Economy},
            volume={81},
            number={3},
            pages={637--654},
            year={1973}
        }
    """
    t_years = max(time_to_expiry, TIME_FLOOR)
    d1 = _bs_d1(spot, strike, t_years, iv, risk_free_rate, dividend_yield)
    d2 = d1 - iv * math.sqrt(t_years)
    return float(
        spot * math.exp(-dividend_yield * t_years) * stats.norm.cdf(d1)
        - strike * math.exp(-risk_free_rate * t_years) * stats.norm.cdf(d2)
    )


def bs_call_delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    iv: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    """Compute Black-Scholes delta of a European call option.

    Arguments:
        spot: Current asset price.
        strike: Strike price.
        time_to_expiry: Time to expiry in years. Floored at TIME_FLOOR.
        iv: Implied volatility (annualized).
        risk_free_rate: Continuously compounded risk-free rate. Default 0.0.
        dividend_yield: Continuously compounded dividend yield (q). Default 0.0.

    Returns:
        Delta in (0, 1); approaches 1.0 for deep in-the-money options.
    """
    t_years = max(time_to_expiry, TIME_FLOOR)
    d1 = _bs_d1(spot, strike, t_years, iv, risk_free_rate, dividend_yield)
    return float(math.exp(-dividend_yield * t_years) * stats.norm.cdf(d1))


def bs_call_vanna(
    spot: float,
    strike: float,
    time_to_expiry: float,
    iv: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    """Compute Black-Scholes vanna of a European call option.

    Vanna measures the sensitivity of delta to changes in implied volatility
    (equivalently, the sensitivity of vega to spot price moves).

    vanna = -e^{-qT} * N'(d1) * (d2 / sigma)

    Arguments:
        spot: Current asset price.
        strike: Strike price.
        time_to_expiry: Time to expiry in years. Floored at TIME_FLOOR.
        iv: Implied volatility (annualized).
        risk_free_rate: Continuously compounded risk-free rate. Default 0.0.
        dividend_yield: Continuously compounded dividend yield (q). Default 0.0.

    Returns:
        Vanna (dDelta/dVol = dVega/dSpot).
    """
    t_years = max(time_to_expiry, TIME_FLOOR)
    d1 = _bs_d1(spot, strike, t_years, iv, risk_free_rate, dividend_yield)
    d2 = d1 - iv * math.sqrt(t_years)
    return float(-math.exp(-dividend_yield * t_years) * stats.norm.pdf(d1) * (d2 / iv))


# ---------------------------------------------------------------------------
# LEAPS contract lifecycle — pure functions
# ---------------------------------------------------------------------------


def create_leaps_contract(
    purchase_date: pd.Timestamp,
    spot: float,
    capital_to_deploy: float,
    iv: float = DEFAULT_IV,
    account_type: AccountType = AccountType.TAXABLE,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
) -> LeapsContract:
    """Create a LEAPS call contract sized by available capital.

    Strike is set at LEAPS_STRIKE_RATIO * spot (deep in the money).
    Expiry is purchase_date + 2 years.
    n_contracts = capital_to_deploy / (premium_per_share * CONTRACT_MULTIPLIER).

    Arguments:
        purchase_date: Trade date.
        spot: VTI price at purchase.
        capital_to_deploy: Dollar amount to invest in the position.
        iv: Implied volatility used for Black-Scholes pricing. Default 0.18.
        account_type: Tax treatment applied at future roll. Default TAXABLE.
        risk_free_rate: Continuously compounded risk-free rate. Default 0.0.
        dividend_yield: Continuously compounded dividend yield. Default 0.0.

    Returns:
        LeapsContract with all fields populated.
    """
    strike = LEAPS_STRIKE_RATIO * spot
    expiry: pd.Timestamp = pd.Timestamp(purchase_date + pd.DateOffset(years=2))
    t_years = (expiry - purchase_date).days / 365.0
    premium_per_share = bs_call_price(spot, strike, t_years, iv, risk_free_rate, dividend_yield)
    n_contracts = capital_to_deploy / (premium_per_share * CONTRACT_MULTIPLIER)
    return LeapsContract(
        purchase_date=purchase_date,
        expiry_date=expiry,
        strike=strike,
        spot_at_purchase=spot,
        premium_paid=premium_per_share,
        notional=spot * CONTRACT_MULTIPLIER,
        n_contracts=n_contracts,
        account_type=account_type,
        dividend_yield=dividend_yield,
    )


def price_leaps_contract(
    contract: LeapsContract,
    current_spot: float,
    current_date: pd.Timestamp,
    iv: float = DEFAULT_IV,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """Mark a LEAPS contract to market using Black-Scholes.

    Uses the dividend yield stored on the contract at creation time.

    Arguments:
        contract: The LeapsContract to price.
        current_spot: Current VTI spot price.
        current_date: Valuation date.
        iv: Implied volatility for pricing. Default 0.18.
        risk_free_rate: Continuously compounded risk-free rate. Default 0.0.

    Returns:
        Total mark-to-market value of the position (all contracts, all shares).
    """
    t_years = (contract.expiry_date - current_date).days / 365.0
    per_share = bs_call_price(
        current_spot, contract.strike, t_years, iv, risk_free_rate, contract.dividend_yield
    )
    return float(per_share * CONTRACT_MULTIPLIER * contract.n_contracts)


def should_roll(
    contract: LeapsContract,
    current_date: pd.Timestamp,
    new_expiry_available: pd.Timestamp,
) -> bool:
    """Determine whether a LEAPS contract should be rolled.

    Returns True only if all three conditions hold:
      1. A new 2-year expiry exists beyond the current contract's expiry.
      2. The current contract expires within SIX_MONTHS_DAYS (< 182 days).
      3. The contract has been held at least MIN_HOLD_DAYS (>= 366) for LTCG treatment.

    Arguments:
        contract: The contract being evaluated.
        current_date: Today's date.
        new_expiry_available: The new expiry date that would be used for the replacement.

    Returns:
        True if the contract should be rolled today.
    """
    hold_days = (current_date - contract.purchase_date).days
    days_to_expiry = (contract.expiry_date - current_date).days
    return (
        new_expiry_available > contract.expiry_date
        and days_to_expiry < SIX_MONTHS_DAYS
        and hold_days >= MIN_HOLD_DAYS
    )


def roll_contract(
    old_contract: LeapsContract,
    current_date: pd.Timestamp,
    current_spot: float,
    iv: float = DEFAULT_IV,
    ltcg_rate: float = LTCG_RATE,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
) -> LeapsRollEvent:
    """Execute a LEAPS roll: close old contract and open a new 2-year contract.

    Steps:
      1. Mark old contract to market.
      2. Compute realized gain = current_value - original_cost_basis.
      3. Apply LTCG tax on positive gains (0 if TAX_SHELTERED or gain <= 0).
      4. Use net proceeds to buy a new DITM 2-year contract.

    Arguments:
        old_contract: Contract to close.
        current_date: Roll execution date.
        current_spot: VTI price at roll date.
        iv: Implied volatility for both pricing and new contract. Default 0.18.
        ltcg_rate: Combined LTCG + NIIT rate for taxable gains. Default 0.238.
        risk_free_rate: Continuously compounded risk-free rate. Default 0.0.
        dividend_yield: Continuously compounded dividend yield for the new contract.
            Default 0.0.

    Returns:
        LeapsRollEvent with the full transaction record.
    """
    old_value = price_leaps_contract(old_contract, current_spot, current_date, iv, risk_free_rate)
    cost_basis = old_contract.premium_paid * CONTRACT_MULTIPLIER * old_contract.n_contracts
    gain_realized = old_value - cost_basis

    if old_contract.account_type == AccountType.TAX_SHELTERED:
        tax_paid = 0.0
    else:
        tax_paid = max(0.0, gain_realized) * ltcg_rate

    net_proceeds = old_value - tax_paid
    new_contract = create_leaps_contract(
        current_date, current_spot, net_proceeds, iv, old_contract.account_type,
        risk_free_rate, dividend_yield,
    )
    return LeapsRollEvent(
        roll_date=current_date,
        old_contract=old_contract,
        new_contract=new_contract,
        gain_realized=gain_realized,
        tax_paid=tax_paid,
        net_proceeds=net_proceeds,
    )


def compute_leaps_nav_contribution(
    ledger: LeapsLedger,
    current_date: pd.Timestamp,
    current_spot: float,
    iv: float = DEFAULT_IV,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """Compute total net P&L contribution of live LEAPS contracts to portfolio NAV.

    Live contracts are those not yet rolled out and not yet expired.
    NAV contribution = sum(mark_to_market) - sum(total_cost_basis).

    Arguments:
        ledger: The LeapsLedger containing all contract and roll history.
        current_date: Valuation date.
        current_spot: Current VTI spot price.
        iv: Implied volatility for mark-to-market pricing. Default 0.18.
        risk_free_rate: Continuously compounded risk-free rate for BS pricing. Default 0.0.

    Returns:
        Net P&L contribution in dollars. Can be negative if contracts are underwater.
    """
    if not ledger.contracts:
        return 0.0
    rolled_out = {event.old_contract for event in ledger.roll_events}
    live = [
        c for c in ledger.contracts
        if c not in rolled_out and c.expiry_date > current_date
    ]
    if not live:
        return 0.0
    total_mtm = sum(
        price_leaps_contract(c, current_spot, current_date, iv, risk_free_rate) for c in live
    )
    total_cost = sum(c.premium_paid * CONTRACT_MULTIPLIER * c.n_contracts for c in live)
    return float(total_mtm - total_cost)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_leaps_simulation(
    price_series: pd.Series,
    monthly_contribution_to_leaps: float,
    config: LeapsConfig,
    risk_free_series: pd.Series | None = None,
) -> LeapsLedger:
    """Run the full LEAPS accumulation and roll simulation over a price history.

    On each month-end trading day:
      1. Check all live contracts for roll conditions; execute rolls if triggered.
      2. Deploy monthly_contribution_to_leaps into a new DITM 2-year contract.

    The risk-free rate used for Black-Scholes at each month-end is taken from
    risk_free_series (if provided) or from config.risk_free_rate (scalar).

    Arguments:
        price_series: Daily VTI price Series (DatetimeIndex, chronological).
        monthly_contribution_to_leaps: Dollar amount allocated to LEAPS each month.
        config: LeapsConfig governing IV, LTCG rate, account type, and defaults for
            risk_free_rate and dividend_yield.
        risk_free_series: Optional daily annualized risk-free rate Series (decimal).
            When supplied, the rate on each month-end date is used for BS pricing,
            overriding config.risk_free_rate.

    Returns:
        LeapsLedger with the complete history of all contracts and roll events.
    """
    if price_series.empty:
        return LeapsLedger(contracts=(), roll_events=(), account_type=config.account_type)

    # Last trading day of each calendar month in the price series
    dt_index = pd.DatetimeIndex(price_series.index)
    gb = price_series.groupby(dt_index.to_period("M"))
    month_end_dates = pd.DatetimeIndex([grp.index[-1] for _, grp in gb])

    # Pre-align risk-free series to month-end dates if supplied
    rfr_aligned: pd.Series | None = None
    if risk_free_series is not None and not risk_free_series.empty:
        rfr_aligned = risk_free_series.reindex(month_end_dates, method="ffill").fillna(0.0)

    all_contracts: list[LeapsContract] = []
    live_contracts: list[LeapsContract] = []
    roll_events_list: list[LeapsRollEvent] = []

    for date in month_end_dates:
        spot = float(price_series.loc[date])
        new_expiry: pd.Timestamp = pd.Timestamp(date + pd.DateOffset(years=2))
        rfr = float(rfr_aligned.loc[date]) if rfr_aligned is not None else config.risk_free_rate

        # Check roll conditions on every live contract
        still_live: list[LeapsContract] = []
        for contract in live_contracts:
            if should_roll(contract, date, new_expiry):
                event = roll_contract(
                    contract, date, spot, config.iv, config.ltcg_rate, rfr, config.dividend_yield
                )
                roll_events_list.append(event)
                all_contracts.append(event.new_contract)
                still_live.append(event.new_contract)
            else:
                still_live.append(contract)
        live_contracts = still_live

        # Monthly purchase
        if monthly_contribution_to_leaps > 0:
            new_c = create_leaps_contract(
                date, spot, monthly_contribution_to_leaps, config.iv, config.account_type,
                rfr, config.dividend_yield,
            )
            if new_c.n_contracts > 0:
                all_contracts.append(new_c)
                live_contracts.append(new_c)

    return LeapsLedger(
        contracts=tuple(all_contracts),
        roll_events=tuple(roll_events_list),
        account_type=config.account_type,
    )
