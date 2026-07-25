"""Project-wide constants.

All modules import from here rather than defining constants locally.
"""

TICKERS: tuple[str, ...] = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")

SPLICE_MAP: dict[str, tuple[str, str]] = {
    "KMLM": ("AQMIX", "2021-01-01"),
    "VXUS": ("VGTSX", "2011-01-25"),
    "MUB": ("VWITX", "2007-09-10"),
}

VOL_INDEX_TICKERS: frozenset[str] = frozenset({"^VIX", "V2TX.DE", "VXEEM", "^GVZ", "^OVX", "^MOVE"})

VXUS_VOL_BLEND: dict[str, float] = {"V2TX.DE": 0.75, "VXEEM": 0.25}
VXUS_VOL_DEVELOPED_WEIGHT: float = 0.75

ASSET_VOL_INDEX: dict[str, str | None] = {
    "VTI": "^VIX",
    "VXUS": "VXUS_COMPOSITE",
    "GLD": "^GVZ",
    "MUB": "^MOVE",
    "KMLM": None,
    "VGIT": "^MOVE",
}

TBILL_TICKER: str = "^IRX"

NIIT_RATE: float = 0.408

EWMA_LAMBDA: float = 0.95
ROLLING_CORR_WINDOW_WEEKS: int = 156  # 36 months ≈ 156 weeks
TRADING_DAYS_PER_YEAR: int = 252
COV_RIDGE: float = 1e-8  # added to diagonal to guarantee positive definiteness

LEAPS_STRIKE_RATIO: float = 0.50
DEFAULT_IV: float = 0.18
LTCG_RATE: float = 0.238
MIN_HOLD_DAYS: int = 366  # hold at least 1 year + 1 day for LTCG treatment
SIX_MONTHS_DAYS: int = 182  # roll trigger: < 6 months to expiry
CONTRACT_MULTIPLIER: int = 100  # standard 100-share option multiplier
TIME_FLOOR: float = 1.0 / 365  # minimum T to prevent BS blow-up near expiry
DEFAULT_RISK_FREE_RATE: float = 0.0
DEFAULT_DIVIDEND_YIELD: float = 0.013
MIN_PREMIUM_PER_SHARE: float = 0.01
LEAPS_KEY_SUFFIX: str = "_LEAPS"
DRIFT_BAND_RELATIVE: float = 0.10
VIX_MTM_WINDOW: int = 30  # rolling-mean window (days) for VIX-smoothed daily MTM IV

MIN_CRISIS_OBSERVATIONS: int = 20
RISK_FREE_RATE_DEFAULT: float = 0.0

CRISIS_PERIODS: dict[str, tuple[str, str]] = {
    "GFC": ("2007-10-01", "2009-03-31"),
    "COVID": ("2020-02-01", "2020-04-30"),
    "2022 Rate Hike": ("2022-01-01", "2022-10-31"),
}
