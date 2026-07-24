# finance

A Python library for backtesting multi-asset portfolios with optional synthetic leverage via DITM VTI LEAPS, tax-aware roll modeling, and EWMA volatility forecasting.

## Features

- **Performance metrics** — annualized return, max drawdown, Sharpe, Sortino, Calmar, Omega ratios over full period and pre-defined crisis windows (GFC, COVID, 2022 rate hike)
- **Volatility attribution** — EWMA vol (λ=0.95), 36-month rolling weekly correlations, per-asset contribution table summing to 1
- **TEY adjustment** — MUB returns scaled to tax-equivalent yield at 40.8% NIIT
- **LEAPS leverage** — DITM VTI LEAPS (50% strike, Black-Scholes pricing), biannual roll with LTCG preservation, taxable vs. tax-sheltered scenarios
- **Quarterly rebalancing** — user-specified weights (extensible to risk parity and threshold-drift)

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
from finance.figures import plot_nav_growth, plot_drawdown, plot_vol_contributions, format_performance_table

# 1. Fetch and prepare data
price_data = build_price_data("2015-01-01", "2024-12-31", use_aqmix_splice=True)
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
result = run_backtest(return_data, config)

# 3. Build report
vol_model = build_volatility_model(return_data)
report = build_performance_report(result, return_data, vol_model)
print(format_performance_table(report))

# 4. Save charts to figures/
plot_nav_growth({"My Portfolio": result})
plot_drawdown({"My Portfolio": result})
plot_vol_contributions(report)
```

See `implementation_plan.md` for full architecture, API contracts, and LEAPS leverage usage.

## Development

```bash
uv run pytest                  # run tests with coverage
uv run ruff check src/ tests/  # lint
uv run mypy src/               # type-check
```

## Implementation Status

| Phase | Module(s) | Status | Tests | Coverage |
|---|---|---|---|---|
| 1 | Scaffolding | ✅ Complete | — | — |
| 2 | `data.py`, `returns.py` | ✅ Complete | 37 | 98% |
| 3 | `volatility.py` | ✅ Complete | 34 | 98% |
| 4 | `metrics.py` | ✅ Complete | 37 | 99% |
| 5 | `leverage.py` | ✅ Complete | 43 | 100% |
| 6 | `portfolio.py` | ✅ Complete | 26 | 100% |
| 7 | `figures.py` | ✅ Complete | 24 | 99% |
| 8 | Integration & coverage | ✅ Complete | 16 | 99% overall |
| 9 | `examples/` | ✅ Complete | — | — |

**211 tests · 98.97% line coverage · ruff clean · mypy strict clean**

## Examples

```bash
uv run examples/basic_backtest.py      # full backtest + performance table + NAV chart
uv run examples/with_leaps.py          # LEAPS overlay, taxable vs. tax-sheltered
uv run examples/volatility_report.py   # vol contribution table + forward vol forecast
uv run examples/crisis_analysis.py     # GFC / COVID / 2022 per-period metrics
```
