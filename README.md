# finance

A Python library for backtesting multi-asset portfolios. Includes optional synthetic leverage 
via DITM VTI LEAPS and a macro-economic based market timing signal. 

## tl;dr (for most investors):

From various tests and analyses when building this library, the following guidance applies to most
investors:

1. Prefer `RebalanceRule.DRIFT` to `GttConfig`. GTT (described below) does a great job with
macro-economic market drawdowns (eg. GFC), but not monetary driven ones (eg. 2022).
    - I expect Glide path rebalancing, once the feature is built, will be even better.
2. Use LEAPS leverage, in `AccountType.TAX_SHELTERED` accounts. There is major tax drag if in
`AccountType.TAXABLE`.
3. Have a diverse, multi-asset portfolio as described in the below Motivation.

## Motivation

The library is motivated by the desire to rigorously evaluate multi-asset portfolios within 
a "Lifecycle Investing" framework. Optionally supports market timing signal for US equities. 

Multi-asset portfolio construction follows standard Modern Portfolio Theory, All-Weather, 
and/or Risk Parity frameworks:

```                       
              RISING GROWTH                 FALLING GROWTH
        ┌────────────────────────────┬──────────────────────────────┐
        │                            │                              │
 RISING │  • GLD (Commodities/Gold)  │  • GLD (Gold/Real Assets)    │
 INFL   │  • KMLM / DBMF (Short Bonds│  • KMLM / DBMF (Short Fixed  │
        │    / Long Commodities)     │    Income / Long Commodities)│
        ├────────────────────────────┼──────────────────────────────┤
        │                            │  • VGIT (Intermediate Gov)   │
 FALL   │  • VTI (US Equity Delta)   │  • TLT (Long Duration)       │
 INFL   │  • VXUS (Ex-US Equity      │  • MUB (Muni TEY Yield/Cash  │
        │          Delta)            │       Taxable Accounts)      │
        └────────────────────────────┴──────────────────────────────┘
```

## Features

- **Performance metrics:** annualized return, max drawdown, Sharpe, Sortino, Calmar, Omega ratios,
skew, and excess-kurtosis over full period and pre-defined crisis windows (GFC, COVID, 2022 rate 
hike)
- **TEY adjustment:** muni returns scaled to tax-equivalent yield at 40.8% NIIT
- **LEAPS leverage:** DITM VTI LEAPS (50% strike, Black-Scholes pricing), with tax-aware biannual
roll with LTCG preservation, and taxable vs. tax-sheltered scenarios
- **Rebalancing** — Quarterly to user-specified weights or monthly threshold-drift (extensible to
risk parity)
- **Portfolio Volatility Forecasting/Attribution:** EWMA vol (λ=0.95), 36-month rolling weekly
correlations, per-asset contribution table summing to 1
- **Asset Splicing** - For commonly used ETFs (eg VTI, MUB), automatic splicing the the oldest 
available time series of the asset (e.g. VGTSX, VWITX)
- **Growth Trend Timing (GTT):** Empirically and theoretically justifiable market timing signal
for US Equities. 
    - Requires a FRED API Key. [Request one here](https://fred.stlouisfed.org/docs/api/api_key.html).
    - Save your API key in a `.env` file `FRED_API_KEY=<your key>` so that `dotenv` can load it


## Setup + Development

```bash
# Install:
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"

# Develop:
uv run pytest                  # run tests with coverage
uv run ruff check src/ tests/  # lint
uv run mypy src/               # type-check
```

- **598 tests · 97.66% line coverage · ruff clean · mypy strict clean**
- **Note:** No `mutmut` or `hypothesis` tests. Contributions welcomed!

## Usage

### Quickstart

```python
from pathlib import Path

from finance.data import build_price_data, fetch_risk_free_rate
from finance.returns import build_return_data
from finance.volatility import build_volatility_model
from finance.portfolio import PortfolioConfig, run_backtest
from finance.metrics import build_performance_report
from finance.figures import format_performance_table, plot_nav_growth

price_data = build_price_data(START, END, use_splice=True)
return_data = build_return_data(price_data, risk_free_series=fetch_risk_free_rate(START, END))

config = PortfolioConfig(target_weights={...}, initial_nav=1_000_000.0, ...)
result = run_backtest(return_data, price_data, config)

report = build_performance_report(result, price_data, return_data, build_volatility_model(return_data))
print(format_performance_table(report))
plot_nav_growth({"My Portfolio": result}, output_path=Path("outputs/figures/nav.png"))
```

**Runnable examples** (recommended: pipe to `tee`):

```bash
uv run examples/basic_backtest.py 2>&1 | tee basic_backtest.log

uv run examples/basic_backtest.py           # full backtest + performance table + NAV chart
uv run examples/basic_gtt.py                # GTT signal vs. buy-and-hold comparison
uv run examples/leaps_drift.py              # LEAPS overlay, taxable vs. tax-sheltered
uv run examples/volatility_report.py        # vol contribution table + forward vol forecast
uv run examples/crisis_analysis.py          # GFC / COVID / 2022 per-period metrics
uv run examples/gtt_leaps.py                # GTT Leaps vs non-GTT Leaps

```

### Extensions

#### 1. Lifecycle Investing and LEAPS overlay

Lifecycle Investing (_Ayers + Nalebuff (2010)_) show that it's prudent to make leveraged investments
when you're young. By smoothing dollar denominated stock exposure over your lifetime, investors
reduce lifetime risk and increase lifetime returns. The authors aim to elevate the role of time 
diversification to the same conversation as asset diversification.

The suggested implementation is via [DITM LEAPs](https://www.strasmore.com/blog/deep-itm-leaps). 
`finance` enables this via adding a `VTI_LEAPS` allocation key and a `LeapsConfig` to run 
the LEAPS simulation internally:

```python
from finance.leverage import AccountType, LeapsConfig
from finance.portfolio import PortfolioConfig, run_backtest

# fetch_vol_indices=True required for VIX-based dynamic IV
price_data = build_price_data(START, END, use_splice=True, fetch_vol_indices=True)

# VTI_LEAPS does not replace VTI — the portfolio can hold both
config = PortfolioConfig(
    target_weights={..., "VTI_LEAPS": 0.4},
    leaps_config=LeapsConfig(iv=0.18, ltcg_rate=0.238, account_type=AccountType.TAXABLE),
    ...
)
result = run_backtest(return_data, price_data, config)
```

See [`examples/leaps_drift_rebalance.py`](examples/leaps_drift_rebalance.py) for taxable vs. 
tax-sheltered comparison.

#### 2. Growth and Trend Timing (GTT)

GTT (_Philosophical Economics (2016)_) introduces a hybrid market-timing framework which uses
the labor market as a lagging-turned-coincident indicator of broad economic contraction. Our
implementation adds a ^VIX signal to capture sharp contractions such as the COVID market crash.

GTT requires a **dual-confirmation** to act: both a rising unemployment rate and a technical price
signal (200d SMA crossover). The dual-confirmation improves the sensitivity and specificity of the
signal. Analyses show this signal performs very well for macro-economic driven market declines 
(eg. Dot-com crash, GFC, etc); it performs less well with monetary driven declines (eg. 2022)
and sharp crashes (eg. COVID).

- **Note:** Bug with GTT x LEAPS. See [Issue #1](https://github.com/alexWhitworth/finance/issues/1). 
GTT w/o LEAPS works fine
- **Note #2:** GTT is generally out-performed by `RebalanceRule.DRIFT`. 

```python
from finance.gtt import fetch_gtt_signal_data
from finance.portfolio import GttConfig, PortfolioConfig, run_backtest

# fetch_vol_indices=True required; FRED_API_KEY must be set in .env
price_data = build_price_data(START, END, use_splice=True, fetch_vol_indices=True)

gtt_signal = fetch_gtt_signal_data(
    START, END,
    vix_p90_threshold=0.272,  # P90 of VIX 1993-2026
    equity_prices=price_data.prices["VTI"].rename("VTI"),
)

config = PortfolioConfig(
    target_weights={...},
    gtt_config=GttConfig(vix_p90_threshold=0.272, defensive_weights={...}),
    ...
)
result = run_backtest(return_data, price_data, config, gtt_signal=gtt_signal)
```

See [`examples/basic_gtt.py`](examples/basic_gtt.py) for the full runnable version.

## References

```bibtex
@book{ayres2010lifecycle,
  title = {Lifecycle Investing: A New, Safe, and Audacious Way to Improve the Performance of 
  Your Retirement Portfolio},
  author = {Ayres, Ian and Nalebuff, Barry},
  year = {2010},
  publisher = {Basic Books}
}

@online{philosophicalecon2016uetrend,
    author       = {{Philosophical Economics}},
    title        = {In Search of the Perfect Recession Indicator},
    year         = {2016},
    month        = {February},
    day          = {21},
    url          = {https://www.philosophicaleconomics.com/2016/02/uetrend/},
    urldate      = {2026-07-26},
    note         = {Blog post}
}
```
