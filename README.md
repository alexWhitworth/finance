# finance

A Python library for backtesting multi-asset portfolios with optional synthetic leverage via DITM VTI LEAPS, tax-aware roll modeling, and EWMA volatility forecasting.

## Features

- **Performance metrics** — annualized return, max drawdown, Sharpe, Sortino, Calmar, Omega ratios, skew, and excess-kurtosis over full period and pre-defined crisis windows (GFC, COVID, 2022 rate hike)
- **Volatility attribution** — EWMA vol (λ=0.95), 36-month rolling weekly correlations, per-asset contribution table summing to 1
- **TEY adjustment** — muni returns scaled to tax-equivalent yield at 40.8% NIIT
- **LEAPS leverage** — DITM VTI LEAPS (50% strike, Black-Scholes pricing), biannual roll with LTCG preservation, taxable vs. tax-sheltered scenarios
- **Rebalancing** — Quarterly to user-specified weights or monthly threshold-drift (extensible to risk parity)

## Setup

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

```python
from finance.data import build_price_data
from finance.returns import build_return_data
from finance.volatility import build_volatility_model
from finance.leverage import RebalanceRule, WeightStrategy
from finance.portfolio import PortfolioConfig, run_backtest
from finance.metrics import build_performance_report
from finance.figures import (
    plot_nav_growth, plot_drawdown, plot_vol_contributions,
    format_performance_table, compare_performance_table,
)

# 1. Fetch and prepare data
price_data = build_price_data("2015-01-01", "2026-06-30", use_splice=True)
return_data = build_return_data(price_data)

# 2. Run backtest
config = PortfolioConfig(
    target_weights={"VTI": 0.40, "VXUS": 0.20, "GLD": 0.10,
                    "MUB": 0.10, "KMLM": 0.10, "VGIT": 0.10},
    initial_nav=1_000_000.0,
    monthly_contribution=10_000.0,
    rebalance_rule=RebalanceRule.QUARTERLY,
    weight_strategy=WeightStrategy.USER_SPECIFIED,
    leaps_config=None,
)
result = run_backtest(return_data, price_data, config)

# 3. Build report
vol_model = build_volatility_model(return_data)
report = build_performance_report(result, price_data, return_data, vol_model)
print(format_performance_table(report))

# 4. Save charts to figures/
plot_nav_growth({"My Portfolio": result})
plot_drawdown({"My Portfolio": result})
plot_vol_contributions(report)

# 5. Compare multiple portfolios side-by-side
print(compare_performance_table([("Conservative", report), ("Aggressive", report2)]))
```

### LEAPS overlay

Add a `VTI_LEAPS` allocation key and a `LeapsConfig` to run the LEAPS simulation internally:

```python
from finance.leverage import AccountType, LeapsConfig

leaps_config = PortfolioConfig(
    target_weights={"VTI": 0.35, ..., "VTI_LEAPS": 0.05},
    leaps_config=LeapsConfig(iv=0.18, ltcg_rate=0.238, account_type=AccountType.TAXABLE),
    ...
)
result = run_backtest(return_data, price_data, leaps_config)
```

## Development

```bash
uv run pytest                  # run tests with coverage
uv run ruff check src/ tests/  # lint
uv run mypy src/               # type-check
```

**449 tests · 98.46% line coverage · ruff clean · mypy strict clean**

## Examples

**Recommended:** pipe outputs to `tee` (e.g. `uv run file.py 2>&1 | tee logfile.log`)

```bash
# pipe to tee usage:
uv run examples/basic_backtest.py 2>&1 | tee basic_backtest.log

# Examples:
uv run examples/basic_backtest.py         # full backtest + performance table + NAV chart
uv run examples/leaps_drift_rebalance.py  # LEAPS, drift rebalancing, + taxable vs. tax-sheltered
uv run examples/volatility_report.py      # vol contribution table + forward vol forecast
uv run examples/crisis_analysis.py        # GFC / COVID / 2022 per-period metrics
uv run examples/basic_gtt.py              # Basic backtest, with the GTT signal and comparison
```

## Growth and Trend Timing

- **to-write**
- Requres a FRED API Key. [Request one here](https://fred.stlouisfed.org/docs/api/api_key.html).
    - Save in a `.env` file `FRED_API_KEY=<your key>` so that `dotenv` can load it

#### Reference:

```bibtex
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