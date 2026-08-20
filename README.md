# finance

[![CI](https://github.com/alexWhitworth/finance/actions/workflows/ci.yml/badge.svg)](https://github.com/alexWhitworth/finance/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)

A Python library for backtesting and managing multi-asset portfolios. It includes optional
synthetic leverage via DITM VTI LEAPS and a macroeconomic market-timing signal. A live portfolio
management layer (NAV breakdown, weight-drift, rebalance simulation, LEAPS greeks) and a LEAPS
DCA entry-timing signal round out the toolkit for deploying and managing capital over time.

## Contents

- [tl;dr (for most investors)](#tldr-for-most-investors)
- [Motivation](#motivation)
- [Features](#features)
- [Setup + Development](#setup--development)
- [Usage](#usage)
  - [Quickstart](#quickstart)
  - [Runnable examples](#runnable-examples)
  - [Cookbook](#cookbook)
  - [Extensions](#extensions)
- [References](#references)
- [License](#license)

## tl;dr (for most investors):

From extensive backtests and analyses when building this library, the following guidance
applies to most investors. When paired with the multi-asset portfolios defined in `examples/`,
the result is **handily beating** standard institutional frameworks (eg. 60/40, 80/20, 
All-Weather, etc):

1. `RebalanceRule.DRIFT` with or without a `GlidepathConfig`, dominates.
    - No glide path: better risk adjusted returns (Sharpe, Sortino) across all regimes.
    - `GlidepathConfig` has better terminal NAV and late-stage wealth preservation. Preferred
    for dynamic lifetime de-leveraging (multi-decade wealth generation), but suffers from
    sequence-of-returns risk
2. Use LEAPS leverage. But solely in `AccountType.TAX_SHELTERED` accounts. There is major tax
drag if in `AccountType.TAXABLE`.
3. GTT (described below) should be discarded. GTT does a great job with macro-economic market 
drawdowns (eg. GFC), but not monetary driven ones (eg. 2022). As a result, `GttConfig` is
dominated by a multi-asset portfolio with leveraged equity.
    - **WHY?** Timing the market requires being correct twice — both exit and entry timing. 
    GTT does a decent, but imperfect, job. Secondly, properly constructed multi-asset portfolios
    do a _better job._ 
    - The core aim of diversification is finding **multiple uncorrelated assets.** If achieved,
    this outperforms market timing, especially imperfect market timing.
4. For the typical investor with high equity exposure (e.g. VTI, VXUS), a multi-asset portfolio
should extend beyond "stocks and bonds." See [Motivation](#motivation) for more details. Backtests
strongly support adding Gold and Managed Futures.
5. Better construction of the "defensive sleeve" of a multi-asset portfolio (i.e. beyond bonds)
**enables** more equity risk exposure, via leverage. Succinctly, you can achieve similar or better
risk-adjusted return but with higher CAGRs by combining leverage and better "defensive sleeve"
construction.

**Evidence:** see [`outputs/backtest_summary.md`](outputs/backtest_summary.md) and
[`outputs/gtt_findings.md`](outputs/gtt_findings.md) for the analysis behind all conclusions in
the tl;dr. These are working notes from the most recent research pass, not a continuously
maintained backtest archive — rerun the [examples](#runnable-examples) against current data to
reproduce.

## Motivation

The library is motivated by the desire to rigorously evaluate multi-asset portfolios within 
a "Lifecycle Investing" framework. It optionally supports a market-timing signal for US equities. 

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

### Backtesting & Analysis

- **Performance metrics:** annualized return, max drawdown, Sharpe, Sortino, Calmar, Omega ratios,
skew, and excess-kurtosis over full period and pre-defined crisis windows (GFC, COVID, 2022 rate 
hike)
- **TEY adjustment:** muni returns scaled to tax-equivalent yield at 40.8% NIIT
- **LEAPS leverage:** DITM VTI LEAPS (50% strike, Black-Scholes pricing), with tax-aware biannual
roll with LTCG preservation, and taxable vs. tax-sheltered scenarios
- **Rebalancing** — Three options currently supported (extensible to risk parity):
    1. Quarterly: to exact user-specified weights
    2. Monthly DRIFT: rebalances on threshold-drift vs user-specified weights, allowing
    assets to sit within a target range vs a fixed point target weight: `10% +/- 2% vs 10%`.
    3. Glide-path: deleverage the portfolio overtime, by converting realized gains the base
    multi-asset, lower risk, unlevered portfolio. 
        - Technically: uses exponential decay proportional to wealth accumulation over time. 
        - (_see [glide_path_rebalance](./plans/glide_path_rebalance_spec.json)
    for details_)
- **Portfolio Volatility Forecasting/Attribution:** EWMA vol (λ=0.95), 36-month rolling weekly
correlations, per-asset contribution table summing to 1
- **Asset Splicing** — For commonly used ETFs (eg. VTI, MUB), automatic splicing the oldest
available time series of the asset (eg. VGTSX, VWITX)
- **Portfolio Weight Optimization:** grid-search across target-weight combinations through the
same `run_backtest` pipeline, with side-by-side performance comparison and a risk/return
Pareto-frontier chart
- **Growth Trend Timing (GTT):** Empirically and theoretically justifiable market timing signal
for US Equities. 
    - Requires a FRED API Key. [Request one here](https://fred.stlouisfed.org/docs/api/api_key.html).
    - Save your API key in a `.env` file `FRED_API_KEY=<your key>` so that `dotenv` can load it

### Live Portfolio Management

- **Live Portfolio Management:** bridge a completed backtest into a `LivePortfolio` — or
hand-enter brokerage holdings and LEAPS contracts directly, no backtest required — for NAV
breakdown, per-asset weight-drift, rebalance-plan simulation (QUARTERLY/DRIFT), and LEAPS
position greeks
- **LEAPS DCA Entry Signal:** multi-factor composite score (RSI, Stochastic %D, IV percentile,
MACD) for timing new DITM LEAPS deployment, producing a tranche allocation (`alpha_t`) and a
HOLD / TRANCHE / AGGRESSIVE_SWEEP action

## Setup + Development

**Requires Python 3.13+.**

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

- Tests, lint, and type-checking run in CI on every push/PR (badge above); coverage floor is
enforced at 80% (`fail_under` in `pyproject.toml`) — current coverage runs well above that.
- **Note:** No `mutmut` tests; limited `hypothesis`. Contributions welcomed!

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

### Runnable examples

**Strongly recommended:** pipe to `tee`

```bash
uv run examples/backtesting/basic_backtest.py 2>&1 | tee basic_backtest.log

uv run examples/backtesting/basic_backtest.py           # full backtest + performance table + NAV chart
uv run examples/backtesting/basic_gtt.py                # GTT signal vs. buy-and-hold comparison
uv run examples/backtesting/leaps_tax.py                # LEAPS overlay, taxable vs. tax-sheltered
uv run examples/backtesting/volatility_report.py        # vol contribution table + forward vol forecast
uv run examples/backtesting/crisis_analysis.py          # GFC / COVID / 2022 per-period metrics
uv run examples/backtesting/gtt_leaps.py                # GTT with Leaps. All rebalance rules
uv run examples/backtesting/portfolio_opt.py            # Optimization across multiple portfolio choices

uv run examples/portfolio_manage/backtest_bridge.py     # backtest -> LivePortfolio: NAV, drift, rebalance, greeks, vol
uv run examples/portfolio_manage/manual_leaps_review.py # hand-entered holdings + LEAPS contracts, no backtest
uv run examples/portfolio_manage/leaps_dca_signal.py    # LEAPS DCA entry signal + 12-month sweep
```

### Cookbook

API walkthroughs for features that are ready to use as-is — no extended rationale, just the call
sequence. (For the research-backed design choices behind LEAPS leverage and market timing, see
[Extensions](#extensions) below.)

#### Live Portfolio Management

`finance.portfolio_manager` bridges the backtest engine into a lightweight, user-constructible
live-management layer. Two entry points into the same `LivePortfolio`:

- `as_live_portfolio(result)` — convert a completed `BacktestResult` into a `LivePortfolio`
(holdings, target weights, live LEAPS contracts, GTT regime) as of the last backtest date.
- Construct `LivePortfolio(...)` directly from hand-entered brokerage holdings and
`build_leaps_contract(...)` — a "cold start" with no backtest involved.

From either path, the same pure-function pipeline applies:

```python
from finance import (
    as_live_portfolio, compute_nav_breakdown, compute_holdings_view,
    compute_rebalance_plan, compute_portfolio_greeks, compute_volatility_report,
)
from finance.portfolio_manager import compute_leaps_holdings_view
from finance.leverage import RebalanceRule

portfolio = as_live_portfolio(result)                 # or LivePortfolio(...) by hand
nav = compute_nav_breakdown(portfolio, leaps_mtm=...)  # caller supplies LEAPS mark-to-market
holdings = (*compute_holdings_view(portfolio, nav), *compute_leaps_holdings_view(portfolio, nav))
plan = compute_rebalance_plan(
    portfolio, nav, RebalanceRule.QUARTERLY, is_rebal_date=True, is_month_end=True
)
greeks = compute_portfolio_greeks(portfolio, spot=..., iv=...)
vol_report = compute_volatility_report(portfolio, return_data)
```

`finance.figures` adds matching formatters: `format_nav_breakdown_table`, `format_holdings_table`,
`format_trade_orders_table`, `format_contract_greeks_table`.

- **Note:** `compute_rebalance_plan`'s `leaps_trim` fires whenever the rebalance actually
triggers — QUARTERLY or DRIFT — not DRIFT alone, since QUARTERLY has no tolerance band and should
always trim an overweight LEAPS sleeve back to target on its scheduled date.

See [`examples/portfolio_manage/backtest_bridge.py`](examples/portfolio_manage/backtest_bridge.py)
(backtest → live) and
[`examples/portfolio_manage/manual_leaps_review.py`](examples/portfolio_manage/manual_leaps_review.py)
(hand-entered holdings/contracts) for full runnable versions.

#### LEAPS DCA Entry Signal

`compute_leaps_dca_signal` scores whether now is a favorable time to deploy new capital into
DITM LEAPS, combining four technical factors (RSI, Stochastic %D, IV percentile, MACD) into a
single `entry_score` (0-100). The score is ranked against its own trailing history to produce a
tranche allocation `alpha_t` and a `dca_action` of `HOLD` / `TRANCHE` / `AGGRESSIVE_SWEEP`.

```python
from finance import LeapsDcaSignal, compute_leaps_dca_signal
from finance.data import build_price_data
from finance.figures import format_leaps_dca_signal_table

price_data = build_price_data(START, END, tickers=["VTI"], fetch_vol_indices=True, fetch_ohlcv=True)
signal: LeapsDcaSignal = compute_leaps_dca_signal(price_data, "VTI", price_data.prices.index[-1])
print(format_leaps_dca_signal_table(signal))
```

See [`examples/portfolio_manage/leaps_dca_signal.py`](examples/portfolio_manage/leaps_dca_signal.py)
for the full runnable version, including a 12-month signal sweep.

#### Portfolio Weight Optimization

No dedicated optimizer — a grid-search sweep over target-weight combinations through the same
`run_backtest` pipeline used everywhere else, compared side-by-side and plotted as a risk/return
Pareto frontier.

```python
from finance.figures import compare_performance_table, plot_pareto

reports = {}
for name, weights in combinations.items():          # e.g. VXUS/GLD/KMLM/VGIT weight sweep
    config = PortfolioConfig(target_weights=weights, initial_nav=INITIAL_NAV, ...)
    result = run_backtest(return_data, price_data, config)
    reports[name] = build_performance_report(result, price_data, return_data, vol_model)

print(compare_performance_table(list(reports.items())))
plot_pareto({k: r.full_period for k, r in reports.items()}, output_path=Path("outputs/figures/portfolio_opt_pareto.png"))
```

See [`examples/backtesting/portfolio_opt.py`](examples/backtesting/portfolio_opt.py) for the full
grid-search sweep.

### Extensions

Research-backed design choices — why each feature exists, not just how to call it.

#### 1. Lifecycle Investing and LEAPS overlay

Lifecycle Investing (_Ayres + Nalebuff (2010)_) show that it's prudent to make leveraged investments
when you're young. By smoothing dollar denominated stock exposure over your lifetime, investors
reduce lifetime risk and increase lifetime returns. The authors argue diversification across time
deserves the same weight as diversification across assets.

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

See [`examples/backtesting/leaps_tax.py`](examples/backtesting/leaps_tax.py) for taxable vs. 
tax-sheltered comparison.

#### 2. Growth and Trend Timing (GTT)

GTT (_Philosophical Economics (2016)_) introduces a hybrid market-timing framework that uses
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

See [`examples/backtesting/basic_gtt.py`](examples/backtesting/basic_gtt.py) for the full runnable version.

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

## License

MIT — see [LICENSE](LICENSE).
