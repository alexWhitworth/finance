# Extension: Multi-Underlying LEAPS in the Live Portfolio Layer

## 0. Relationship to `plans/portfolio_management.md`

- `plans/portfolio_management.md` is shipped and tested; its invariants (I1-I20) describe 
implemented behavior and stay untouched.
- This is an additive extension, not a revision.  

**Trigger:** `examples/portfolio_manage/manual_leaps_review.py` and the live layer it exercises
(`LivePortfolio`, `greeks.py`, `portfolio_manager.py`) assume one LEAPS underlying. A user
holding DITM LEAPS on VTI *and* IWM *and* GLD can't represent that today — nothing records
which underlying a `LeapsContract` is on.

**Scope (confirmed with user):**
- **In:** `LeapsContract`, `greeks.py`, `portfolio_manager.py`, both example scripts,
  `figures.py` table formatting.
- **Out:** `run_backtest` / `_backtest_steps.py`. It enforces exactly one LEAPS underlying
  (`_build_context` raises if `*_LEAPS` keys span >1 ticker), and the GTT overlay only governs
  one ticker (`GTT_EQUITY_TICKERS = {"VTI"}`). Lifting that is a much larger project touching
  the day loop, `PortfolioState`, `LeapsLedger`, and GTT. Original invariants I11/I13 hold
  unmodified.

---

## 1. Problem Framing

### 1.1 What's broken

`LeapsContract` (`finance/leverage.py`) has no field for its underlying — docstring says *"A
single DITM VTI LEAPS call contract position."* `LivePortfolio.leaps_contracts` is a flat,
untagged tuple. Two consequences:

1. **No routing.** `compute_portfolio_greeks(portfolio, spot: float, iv: float, ...)` applies
   one scalar to every contract — only correct when all contracts share an underlying.
2. **A latent bug, already present but unreachable.** `target_weights` has supported multiple
   `*_LEAPS` keys since the original plan shipped, but `compute_leaps_holdings_view` and
   `compute_rebalance_plan` split the *single* `NavBreakdown.leaps_nav` across those keys
   **proportionally to target weight**, not actual exposure. If VTI LEAPS ran up 30% while IWM
   LEAPS stayed flat, drift would still be reported at the blended target ratio. Never fired
   because no contract could carry a second underlying; fixing (1) forces fixing this too.

### 1.2 Non-goals

No change to `run_backtest`, `PortfolioState`, `LeapsLedger`, GTT semantics, or `dca_signal.py`
(already per-ticker). No new dependency.

---

## 2. Architecture: Steel-Manned Options

**tag the contract, route by dict (chosen).** Add `ticker: str = "VTI"` to
`LeapsContract`. Keep `leaps_contracts` flat. Replace scalar `spot`/`iv` with `dict[str,
float]` keyed by ticker. Grouping for display/rebalance still comes from the existing
`*_LEAPS` suffix convention — no new concept needed.

**Decision:** Default `ticker="VTI"` keeps every existing construction site, fixture, and
backtest call unaffected — purely additive.

---

## 3. Data Schema Changes

### 3a. `leverage.py::LeapsContract` — add `ticker`

```python
@dataclass(frozen=True)
class LeapsContract:
    """A single DITM LEAPS call contract on one underlying.

    Attributes:
        ticker: Underlying ticker (e.g. "VTI", "IWM", "GLD"). Defaults to
            "VTI" so every existing construction site is unaffected.
        ... (existing attributes unchanged)
    """
    purchase_date: pd.Timestamp
    ...
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD
    ticker: str = "VTI"   # NEW
```

`build_leaps_contract(...)` gains a matching `ticker: str = "VTI"` kwarg. `create_leaps_contract`
(backtest-only) is unchanged — see Known Assumption 3.

### 3b. `greeks.py` — spot/IV become per-ticker maps

```python
def compute_portfolio_greeks(
    portfolio: LivePortfolio,
    spot: dict[str, float],      # CHANGED: was float
    iv: dict[str, float],        # CHANGED: was float
    risk_free_rate: float = 0.0,
) -> PortfolioGreeks:
    """Routes each contract by contract.ticker.

    Raises:
        ValueError: Lists every ticker missing from spot or iv.
    """

def leaps_mtm_by_ticker(greeks: PortfolioGreeks) -> dict[str, float]:
    """Sums position MTM (price * n_contracts * CONTRACT_MULTIPLIER *
    leaps_scale) grouped by contract.ticker. Feeds compute_nav_breakdown."""
```

`compute_contract_greeks` is unchanged — it already prices one contract at a time;
`compute_portfolio_greeks` is the only fan-out point needing ticker routing.

### 3c. `portfolio_manager.py` — NAV and rebalance types

```python
@dataclass(frozen=True)
class NavBreakdown:
    """
    Attributes:
        leaps_nav_by_ticker: MTM per LEAPS underlying (NEW).
        leaps_nav: sum(leaps_nav_by_ticker.values()). Unchanged meaning.
        ... (base_nav, defensive_sleeve, leaps_pool, total_nav unchanged)
    """
    base_nav: float
    leaps_nav: float
    leaps_nav_by_ticker: dict[str, float]   # NEW
    defensive_sleeve: float
    leaps_pool: float
    total_nav: float


def compute_nav_breakdown(
    portfolio: LivePortfolio,
    leaps_mtm_by_ticker: dict[str, float] | None = None,   # CHANGED: was leaps_mtm: float = 0.0
) -> NavBreakdown:
    """None sentinel avoids a mutable default; normalizes to {}."""


@dataclass(frozen=True)
class RebalancePlan:
    """
    Attributes:
        leaps_trim: dict[str, float]   # CHANGED: was float
            Per-underlying overshoot, keyed by ticker (not the "<ticker>_LEAPS"
            target_weights key). Empty when no sleeve is overweight.
        ... (as_of_date, would_trigger, trigger_reason, trades, holdings_view unchanged)
    """
    as_of_date: pd.Timestamp
    would_trigger: bool
    trigger_reason: str
    trades: tuple[TradeOrder, ...]
    leaps_trim: dict[str, float]   # CHANGED
    holdings_view: tuple[HoldingView, ...]


def leaps_trims_as_trade_orders(   # RENAMED from leaps_trim_as_trade_order (singular)
    leaps_views: tuple[HoldingView, ...],
    leaps_trim: dict[str, float],
) -> tuple[TradeOrder, ...]:
    """One TradeOrder per ticker in leaps_trim. Empty tuple if leaps_trim is empty."""
```

No change to `LivePortfolio`, `HoldingView`, `TradeOrder`, `GttStatus` — `HoldingView.ticker`
for a LEAPS sleeve already carries `"<ticker>_LEAPS"`, which already disambiguates by
underlying; only the computation feeding it (proportional split) was wrong.

---

## 4. Component / Module Breakdown

**`leverage.py`** — add `ticker: str = "VTI"` to `LeapsContract` and `build_leaps_contract`. No
change to `create_leaps_contract`, `roll_contract`, `run_leaps_simulation` (backtest-only).

**`greeks.py`** — `compute_portfolio_greeks`: `spot`/`iv` become `dict[str, float]`, routed by
`contract.ticker`; collect all missing tickers before raising one `ValueError`. Add
`leaps_mtm_by_ticker`. `compute_contract_greeks`/dataclasses unchanged.

**`portfolio_manager.py`** — add `leaps_nav_by_ticker` to `NavBreakdown`; swap
`compute_nav_breakdown`'s `leaps_mtm` for `leaps_mtm_by_ticker`. Extract a shared private
`_leaps_actual_weights_by_key(portfolio, nav) -> dict[str, float]` (maps each `"<ticker>_LEAPS"`
key to `nav.leaps_nav_by_ticker.get(ticker, 0.0) / total_nav`), used by both
`compute_leaps_holdings_view` and the DRIFT branch of `compute_rebalance_plan` instead of the
old proportional split. Compute `leaps_trim` per ticker. Rename
`leaps_trim_as_trade_order` -> `leaps_trims_as_trade_orders`. `as_live_portfolio` unchanged —
contracts already carry whatever `ticker` the backtest tagged (default "VTI").

**`figures.py`** — `format_contract_greeks_table` gains a `"ticker"` row entry. Other table
functions unchanged — already keyed by ticker string, agnostic to how many appear.

**`examples/portfolio_manage/manual_leaps_review.py`** — fetch price/vol data for all
underlyings in use (e.g. `["VTI", "IWM", "GLD"]`); build `spot_map`/`iv_map` dicts; pass
`ticker=...` to each `build_leaps_contract`; add one `*_LEAPS` key per underlying to
`TARGET_WEIGHTS`; replace the manual MTM sum with `leaps_mtm_by_ticker(greeks)`; replace the
single trim call with `leaps_trims_as_trade_orders(leaps_views, plan.leaps_trim)`.

**`examples/portfolio_manage/backtest_bridge.py`** — wrap existing scalars into single-key
dicts (`{"VTI": ...}`) at both call sites. Backtest stays VTI-only; pure call-site update.

---

## 5. Step-by-Step Implementation Roadmap

**Phase 0 — Tag the contract.** Add `ticker: str = "VTI"` to `LeapsContract` and
`build_leaps_contract`.
*Done when:* `uv run pytest` passes unchanged; `uv run mypy src/` clean.

**Phase 1 — Route greeks by ticker.** `compute_portfolio_greeks` takes `spot`/`iv` dicts, routes
by `contract.ticker`, raises with all missing tickers listed. Add `leaps_mtm_by_ticker`.
*Done when:* a 2-ticker test prices each contract against its own spot/IV; a missing-ticker
case lists all missing names.

**Phase 2 — Fix per-ticker NAV and drift.** Add `leaps_nav_by_ticker`; change
`compute_nav_breakdown`'s input; extract `_leaps_actual_weights_by_key`; wire it into
`compute_leaps_holdings_view` and `compute_rebalance_plan`'s DRIFT branch; make `leaps_trim`
per-ticker.
*Done when:* a regression test (two sleeves, divergent MTM) shows drift from actual exposure,
not target-weight share.

**Phase 3 — Rename the trim helper.** `leaps_trim_as_trade_order` -> `leaps_trims_as_trade_orders`,
taking the full `leaps_views` tuple and the `leaps_trim` dict.
*Done when:* two overweight sleeves return two orders; zero overweight returns `()`.

**Phase 4 — Display.** Add `ticker` column to `format_contract_greeks_table`.
*Done when:* visual check against `manual_leaps_review.py` output.

**Phase 5 — Update examples.** Multi-underlying demo in `manual_leaps_review.py`; dict-wrapping
call-site updates in `backtest_bridge.py`.
*Done when:* both scripts run end-to-end with no errors; `manual_leaps_review.py` shows a
sensible multi-sleeve table.

**Phase 6 — Tests and docs.** Update `test_greeks.py`, `test_portfolio_manager.py`,
`test_figures.py` for new signatures. Docstrings on all changed signatures.
*Done when:* `uv run mypy src/` strict clean; `uv run ruff check .` clean.

---

## 6. System & Test Invariants

| ID | Type | Invariant | Oracle / Bound |
|----|------|-----------|----------------|
| M1 | ATOMIC | `leaps_nav == sum(leaps_nav_by_ticker.values())` within 1e-9 | Arithmetic |
| M2 | ATOMIC | `compute_portfolio_greeks` lists every missing ticker, not just the first | 2 contracts, both tickers missing from `iv` |
| M3 | ATOMIC | Contract `c` is priced with `spot[c.ticker]`/`iv[c.ticker]` only | 2-ticker portfolio, distinct spot/IV per ticker |
| M4 | REGRESSION | Drift reflects actual per-ticker exposure, not target-weight split | VTI sleeve +30% MTM, IWM flat, equal target weights → unequal actual_weight |
| M5 | ATOMIC | `leaps_trim` contains only tickers actually overweight | One overweight, one underweight sleeve |
| M6 | ATOMIC | `leaps_trims_as_trade_orders` returns exactly `len(leaps_trim)` orders | Count check |
| M7 | NON-REGRESSION | Original I1-I20 hold; backtest still rejects >1 LEAPS underlying | Existing suite passes unchanged |
| M8 | ATOMIC | Omitting `ticker` defaults to `"VTI"` on both `LeapsContract` and `build_leaps_contract` | Construction without the kwarg |

---

## 7. Known Assumptions

1. Grouping still comes from the `*_LEAPS` suffix in `target_weights`, not a new container.
   `target_weights["GLD_LEAPS"]` and `LeapsContract.ticker == "GLD"` are not cross-validated —
   see Edge Cases.
2. `leaps_trim` keys are bare tickers (`"IWM"`), not `_LEAPS`-suffixed. Only direct callers of
   `compute_rebalance_plan` need to know; `leaps_trims_as_trade_orders` handles the mapping.
3. Backtest-created contracts keep `ticker="VTI"` — not threaded with the real underlying.
   Correct today (every config uses `"VTI_LEAPS"`). A future single-underlying, non-VTI backtest
   would mislabel bridged contracts via `as_live_portfolio()`; fixing that is a small,
   mechanical change to 3 backtest call sites, deferred since nothing exercises it yet.
4. `leaps_mtm_by_ticker` is still fully caller-supplied, same as the original plan's handling of
   `leaps_mtm`. The new helper is a convenience, not a requirement.

---

## 8. Edge Cases & Pre-Mortem Risks

| Risk | Likelihood | Mitigation |
|------|-----------|-------------|
| `target_weights` has `"IWM_LEAPS"` but no contract tagged `ticker="IWM"` | Medium | `.get(ticker, 0.0)` shows 100% underweight, not an error — same as any untracked target key. |
| A contract is tagged `ticker="GLD"` but no `"GLD_LEAPS"` key exists | Medium | MTM still counted in `total_nav`/`leaps_nav_by_ticker`, but omitted from the drift table (iterates `target_weights` keys only). Same gap class as an untracked base holding. |
| `spot`/`iv` dicts contain extra, unused tickers | Low | Harmless — only `contract.ticker` lookups happen; no validation needed. |
| Same ticker across multiple contracts (e.g. post-GTT-close survivor) | Low | Already handled — scale applied per-contract before summing in `leaps_mtm_by_ticker`. |
| Mutable default on `leaps_mtm_by_ticker` | Medium | Use `None` sentinel, normalize to `{}` inside the function — never a mutable default argument. |
