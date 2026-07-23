"""LEAPS option contract types and portfolio enumeration types.

This module contains only dataclass and enum definitions. Business logic
(option pricing, contract creation, roll simulation) will be added in Phase 5.
"""

import enum
from dataclasses import dataclass

import pandas as pd


class AccountType(enum.Enum):
    """Tax treatment applied to LEAPS gains at roll.

    Attributes:
        TAXABLE: Gains are taxable at long-term capital gains rates.
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


@dataclass(frozen=True)
class LeapsConfig:
    """Configuration for a LEAPS overlay simulation.

    Attributes:
        iv: Constant implied volatility for Black-Scholes pricing. Default 0.18.
        ltcg_rate: Combined LTCG + NIIT rate applied on taxable rolls. Default 0.238.
        account_type: Governs whether tax is applied on roll.
    """

    iv: float = 0.18
    ltcg_rate: float = 0.238
    account_type: AccountType = AccountType.TAXABLE


@dataclass(frozen=True)
class LeapsContract:
    """A single VTI LEAPS call contract position.

    Attributes:
        purchase_date: Trade date.
        expiry_date: Option expiry (typically 2 years from purchase).
        strike: Strike price; set at 50% of spot_at_purchase.
        spot_at_purchase: VTI price on purchase_date.
        premium_paid: Black-Scholes call price per contract at purchase.
        notional: spot_at_purchase * 100 (standard 100-share multiplier).
        n_contracts: Number of contracts; float to support fractional allocation.
        account_type: Tax treatment applied at roll.
    """

    purchase_date: pd.Timestamp
    expiry_date: pd.Timestamp
    strike: float
    spot_at_purchase: float
    premium_paid: float
    notional: float
    n_contracts: float
    account_type: AccountType


@dataclass(frozen=True)
class LeapsRollEvent:
    """Record of a single LEAPS roll transaction.

    Attributes:
        roll_date: Date the roll was executed.
        old_contract: Contract being closed.
        new_contract: Replacement contract opened with net proceeds.
        gain_realized: Mark-to-market value minus original premium paid.
        tax_paid: LTCG tax on gain; 0.0 for TAX_SHELTERED accounts.
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
        contracts: All contracts ever created (including rolled-out positions).
        roll_events: All roll transactions executed during the simulation.
        account_type: Account type governing all contracts in this ledger.
    """

    contracts: tuple[LeapsContract, ...]
    roll_events: tuple[LeapsRollEvent, ...]
    account_type: AccountType
