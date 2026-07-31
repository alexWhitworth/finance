"""Project-wide constants.

All modules import from here rather than defining constants locally.
"""

TICKERS: tuple[str, ...] = ("VTI", "VXUS", "GLD", "MUB", "KMLM", "VGIT")

SPLICE_MAP: dict[str, tuple[str, str]] = {
    # KMLM: proxy is a parquet file of MLMI total-return index history (1988-2020-12-01).
    # Splice date = KMLM ETF inception. See data/kmlm_mlmi_pre.parquet.
    "KMLM": ("file:data/kmlm_mlmi_pre.parquet", "2020-12-02"),
    "VXUS": ("VGTSX", "2011-01-28"),
    "VTI": ("VTSMX", "2001-06-15"),
    "VGIT": ("VFITX", "2009-11-23"),
    "MUB": ("VWITX", "2007-09-10"),
    "GLD": ("GC=F", "2004-11-18"),  # only goes back to 2000-08-30 :(
    # Unable to find a splice source further back. Still trying to get.
    # -------------------------------------------
    # "GLD": inception on yfinance = 2004-11-18.
        # Removed from FRED (blogpost: https://shorturl.at/Bq28j)
        # NASDAQ: not freely available
        # WorldBank: not downloading
}

VOL_INDEX_TICKERS: frozenset[str] = frozenset({"^VIX", "^GVZ", "^MOVE"})

ASSET_VOL_INDEX: dict[str, str | None] = {
    "VTI": "^VIX",
    # VXUS: preferred blend was V2TX.DE (VSTOXX, 75%) + ^VXEEM (25%); both delisted on yfinance.
    # Fallback: ^VIX scaled by VOL_INDEX_SCALAR["VXUS"] = 1.15.
    "VXUS": "^VIX",
    "GLD": "^GVZ",
    "MUB": "^MOVE",
    "KMLM": None,
    "VGIT": "^MOVE",
}

# Scalar applied to the fetched vol index after unit conversion (÷100).
# Add an entry here whenever a proxy ticker systematically under- or over-states
# the true vol level for a given asset.
#
# VXUS note: the preferred approach was a blended IV from V2TX.DE (VSTOXX,
# developed-market weight 0.75) and ^VXEEM (EM weight 0.25). Both tickers are
# delisted / unavailable on yfinance as of mid-2025. ^VIX * 1.15 is a fallback
# proxy; international equity vol runs ~15% higher than US vol historically.
# Revisit if a reliable developed + EM blend becomes available again.
VOL_INDEX_SCALAR: dict[str, float] = {
    "VXUS": 1.15,
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

NBER_RECESSION_PERIODS: list[tuple[str, str]] = [
    ("1969-12-31", "1970-11-30"),
    ("1973-11-30", "1975-03-31"),
    ("1980-01-31", "1980-07-31"),
    ("1981-07-31", "1982-11-30"),
    ("1990-07-31", "1991-03-31"),
    ("2001-02-28", "2001-11-30"),
    ("2007-12-31", "2009-06-30"),
    ("2020-02-28", "2020-04-30"),
]

# ---------------------------------------------------------------------------
# GTT (Growth Trend Timing) market-timing overlay
# ---------------------------------------------------------------------------

# Tickers governed by the GTT timing signal.
# VTI_LEAPS is matched via the LEAPS_KEY_SUFFIX suffix ("VTI_LEAPS").
# Extend this set when a VXUS GTT signal is designed and validated.
GTT_EQUITY_TICKERS: frozenset[str] = frozenset({"VTI"})

# Trading-day execution lag from the UNRATE publication date to the trade. The
# reference→publication (~1-month) lag is handled inside compute_ue_signal via
# first-Friday re-stamping, NOT by this constant.
GTT_UNRATE_TRADE_LAG_DAYS: int = 1
GTT_VIX_CONSECUTIVE_DAYS: int = 5  # Default persistence window
GTT_SMA_WINDOW: int = 200  # Default equity price SMA window

GTT_DEFENSIVE_WEIGHTS_DEFAULT: dict[str, float] = {
    "R_f": 0.25,
    "KMLM": 0.5,
    "VGIT": 0.25,
    "GLD": 0.0,
}
