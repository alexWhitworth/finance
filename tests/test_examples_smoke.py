"""Smoke tests for the portfolio-management example scripts.

These execute the examples end-to-end in a subprocess and assert a clean exit,
capturing the console output for a light content check. They perform live
network I/O (yfinance) and are therefore marked ``slow``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_example(script: str, *args: str, timeout: int = 600) -> str:
    """Run an example script with the current interpreter and return stdout.

    Arguments:
        script: Path to the script relative to the repository root.
        args: Extra CLI arguments passed after the script path.
        timeout: Maximum wall-clock seconds for the subprocess.

    Returns:
        The combined stdout string of the subprocess.
    """
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / script), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, f"{script} failed:\n{result.stderr}"
    return result.stdout


@pytest.mark.slow
def test_manual_leaps_review_exits_clean_and_prints_three_sleeves() -> None:
    """F-006: manual_leaps_review.py runs clean and shows all three underlyings."""
    out = _run_example("examples/portfolio_manage/manual_leaps_review.py")
    assert "VTI_LEAPS" in out
    assert "IWM_LEAPS" in out
    assert "GLD_LEAPS" in out


@pytest.mark.slow
def test_manual_leaps_review_fixtures_prints_surface_columns() -> None:
    """F-3: --fixtures runs offline and prints per-contract surface fields (INV-7)."""
    out = _run_example("examples/portfolio_manage/manual_leaps_review.py", "--fixtures")
    # Per-contract surface output: sigma(k,t), mark, headline delta, level_vega.
    assert "sigma(k,t)=" in out
    assert "headline_delta=" in out
    assert "level_vega=" in out
    for ticker in ("VTI", "IWM", "GLD"):
        assert f"{ticker}: sigma(k,t)=" in out


@pytest.mark.slow
def test_backtest_bridge_exits_clean() -> None:
    """F-007: backtest_bridge.py runs clean end-to-end (VTI-only bridge)."""
    out = _run_example("examples/portfolio_manage/backtest_bridge.py")
    assert "would_trigger=" in out
