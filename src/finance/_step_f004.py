"""F-004: _build_context — pre-loop validation and precomputed series.

Extracts all input validation and series precomputation that currently lives at
the top of run_backtest (lines 583–723 of portfolio.py) into a single pure
function returning an immutable BacktestContext.

No I/O is performed here. Side effects (the run_leaps_simulation call) remain
in _build_initial_state (F-005).
"""

from __future__ import annotations

import pandas as pd

from finance.consts import DEFAULT_IV, LEAPS_KEY_SUFFIX, VIX_MTM_WINDOW
from finance.data import PriceData
from finance.gtt import GttSignalData
from finance.portfolio import (
    BacktestContext,
    GTT_RISK_FREE_KEY,
    PortfolioConfig,
    _defensive_gross_return,
    _gtt_governed_keys,
    _long_windows,
    _reindex_position_mask,
    get_rebalance_dates,
)
from finance.returns import ReturnData


def _build_context(
    return_data: ReturnData,
    price_data: PriceData,
    config: PortfolioConfig,
    gtt_signal: GttSignalData | None,
) -> BacktestContext:
    """Validate all run_backtest inputs and precompute all series and index sets.

    Extracts the pre-loop setup from run_backtest: GTT validation, asset split,
    index set construction, GTT series alignment, and LEAPS series precomputation
    (excluding the run_leaps_simulation call, which belongs in _build_initial_state).

    Arguments:
        return_data: Daily simple returns and risk-free rate series.
        price_data: Adjusted close prices and optional vol-index prices.
        config: Full portfolio configuration, including optional LEAPS and GTT config.
        gtt_signal: GTT position mask (1=Long / 0=Defensive), or None to disable.

    Returns:
        Immutable BacktestContext populated with all 22 fields.

    Raises:
        ValueError: If exactly one of gtt_signal / config.gtt_config is set.
        ValueError: If non-R_f defensive_weights tickers are absent from return_data.
        ValueError: If any base asset in config.target_weights is absent from return_data.
        ValueError: If LEAPS keys are present but config.leaps_config is None.
        ValueError: If more than one distinct LEAPS underlying is requested.
        ValueError: If LEAPS underlying is absent from price_data.prices.
    """
    # ------------------------------------------------------------------
    # GTT validation: gtt_signal and config.gtt_config must be paired.
    # ------------------------------------------------------------------
    if (gtt_signal is None) != (config.gtt_config is None):
        raise ValueError(
            "gtt_signal and config.gtt_config must both be set or both be None; got "
            f"gtt_signal={'set' if gtt_signal is not None else 'None'}, "
            f"config.gtt_config={'set' if config.gtt_config is not None else 'None'}"
        )

    returns = return_data.returns

    if gtt_signal is not None:
        assert config.gtt_config is not None  # guaranteed by paired check above
        missing_def = [
            k
            for k in config.gtt_config.defensive_weights
            if k != GTT_RISK_FREE_KEY and k not in returns.columns
        ]
        if missing_def:
            raise ValueError(
                f"defensive_weights tickers absent from return_data: {missing_def}"
            )

    # ------------------------------------------------------------------
    # Asset split: base assets vs LEAPS carve-outs.
    # ------------------------------------------------------------------
    leaps_keys = tuple(k for k in config.target_weights if k.endswith(LEAPS_KEY_SUFFIX))
    base_assets = tuple(k for k in config.target_weights if k not in set(leaps_keys))

    missing = [a for a in base_assets if a not in returns.columns]
    if missing:
        raise ValueError(f"Assets missing from return_data: {missing}")

    use_leaps = len(leaps_keys) > 0
    if use_leaps and config.leaps_config is None:
        raise ValueError("LEAPS keys present in target_weights but leaps_config is None")

    w = pd.Series(config.target_weights)
    leaps_fraction = float(w[list(leaps_keys)].sum()) if leaps_keys else 0.0

    base_target_w = w[list(base_assets)]
    if len(base_assets) > 0 and base_target_w.sum() > 0:
        base_target_w = base_target_w / base_target_w.sum()

    # ------------------------------------------------------------------
    # Index sets: rebalance dates and month-end dates.
    # ------------------------------------------------------------------
    idx = pd.DatetimeIndex(returns.index)

    rebal_dates: frozenset[pd.Timestamp] = frozenset(
        get_rebalance_dates(idx, config.rebalance_rule)
    )

    month_end_dates: frozenset[pd.Timestamp] = frozenset(
        pd.Timestamp(grp.index[-1])
        for _, grp in returns.groupby(idx.to_period("M"))
    )

    # ------------------------------------------------------------------
    # GTT series: position mask, defensive gross return, long window map.
    # ------------------------------------------------------------------
    governed_base: tuple[str, ...] = ()
    defensive_weights: dict[str, float] = {}
    mask_aligned: pd.Series | None = None
    def_gross: pd.Series | None = None
    gtt_active = False
    long_window_end: dict[pd.Timestamp, pd.Timestamp] = {}

    if gtt_signal is not None:
        assert config.gtt_config is not None  # paired above
        governed = _gtt_governed_keys(config.target_weights)
        governed_base = tuple(k for k in governed if k in set(base_assets))
        gtt_active = len(governed) > 0

    if gtt_active:
        assert gtt_signal is not None and config.gtt_config is not None
        defensive_weights = config.gtt_config.defensive_weights
        mask_aligned = _reindex_position_mask(gtt_signal.position_mask, idx)
        def_gross = _defensive_gross_return(
            returns, return_data.risk_free_rate, defensive_weights
        )
        long_window_end = dict(_long_windows(mask_aligned))

    # ------------------------------------------------------------------
    # LEAPS series: validate underlying, build price/IV/RFR series.
    # (run_leaps_simulation belongs in _build_initial_state — F-005.)
    # ------------------------------------------------------------------
    underlying_prices: pd.Series | None = None
    iv: float = DEFAULT_IV
    rfr_series: pd.Series | None = None
    raw_vix: pd.Series | None = None
    mtm_iv_series: pd.Series | None = None
    leaps_monthly = 0.0

    if use_leaps:
        underlyings = {k.removesuffix(LEAPS_KEY_SUFFIX) for k in leaps_keys}
        if len(underlyings) > 1:
            raise ValueError(
                f"Only one LEAPS underlying is supported; got {sorted(underlyings)}"
            )
        underlying = next(iter(underlyings))
        if underlying not in price_data.prices.columns:
            raise ValueError(
                f"LEAPS underlying '{underlying}' absent from price_data.prices"
            )

        assert config.leaps_config is not None  # guarded above
        iv = config.leaps_config.iv
        underlying_prices = price_data.prices[underlying].reindex(idx, method="ffill")
        rfr_series = return_data.risk_free_rate.reindex(idx, method="ffill").fillna(0.0)

        if (
            not price_data.vol_prices.empty
            and underlying in price_data.vol_prices.columns
        ):
            raw_vix = price_data.vol_prices[underlying].reindex(idx, method="ffill")
            mtm_iv_series = raw_vix.rolling(VIX_MTM_WINDOW).mean().ffill()

        leaps_monthly = config.monthly_contribution * leaps_fraction

    base_contribution = config.monthly_contribution * (1.0 - leaps_fraction)

    return BacktestContext(
        base_assets=base_assets,
        leaps_keys=leaps_keys,
        leaps_fraction=leaps_fraction,
        base_target_w=base_target_w,
        governed_base=governed_base,
        gtt_active=gtt_active,
        defensive_weights=defensive_weights,
        use_leaps=use_leaps,
        iv=iv,
        leaps_monthly=leaps_monthly,
        base_contribution=base_contribution,
        config=config,
        return_data=return_data,
        underlying_prices=underlying_prices,
        raw_vix=raw_vix,
        mtm_iv_series=mtm_iv_series,
        rfr_series=rfr_series,
        mask_aligned=mask_aligned,
        def_gross=def_gross,
        rebal_dates=rebal_dates,
        month_end_dates=month_end_dates,
        long_window_end=long_window_end,
        w=w,
    )
