# finance

A Python library for backtesting multi-asset portfolios with optional synthetic leverage via DITM VTI LEAPS, tax-aware roll modeling, and EWMA volatility forecasting.

## Asset Universe

VTI, VXUS, GLD, VTEB, KMLM, VGIT — with AQMIX as a KMLM proxy prior to Jan 2021.

## Features

- **Performance metrics** — annualized return, max drawdown, Sharpe, Sortino, Calmar, Omega ratios over full period and pre-defined crisis windows (GFC, COVID, 2022 rate hike)
- **Volatility attribution** — EWMA vol (λ=0.95), 36-month rolling weekly correlations, per-asset contribution table summing to 1
- **TEY adjustment** — VTEB returns scaled to tax-equivalent yield at 40.8% NIIT
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

price_data = build_price_data("2015-01-01", "2024-12-31", use_aqmix_splice=True)
return_data = build_return_data(price_data)
```

## Development

```bash
uv run pytest                  # run tests with coverage
uv run ruff check src/ tests/  # lint
uv run mypy src/               # type-check
```

## Implementation Status

| Phase | Module(s) | Status |
|---|---|---|
| 1 | Scaffolding | ✅ Complete |
| 2 | `data.py`, `returns.py` | ✅ Complete |
| 3 | `volatility.py` | ✅ Complete |
| 4 | `metrics.py` | ✅ Complete |
| 5 | `leverage.py` | ✅ Complete |
| 6 | `portfolio.py` | Pending |
| 7 | Reporting & visualization | Pending |
| 8 | Integration & coverage | Pending |

See `implementation_plan.md` for full architecture and API contracts.
