# tests/test_overfitting_tax.py

"""
Unit tests for scripts/overfitting_tax.py's pure helper, summarize_tax.

summarize_tax takes plain scalars (NOT a WalkForwardResult), so these tests are
pure arithmetic with hand-built numbers — no DuckDB, no Optuna, no walk-forward
run.  They pin the two behaviours that matter: the tax formula (in-sample minus
fitted OOS) and the empty-in-sample guard (mean defaults to 0.0 instead of
raising on statistics.mean([])).

Run with:
    uv run pytest tests/test_overfitting_tax.py -v
"""

# pytest.approx tolerates floating-point noise in the computed tax/mean fields.
import pytest

# The helper under test and its frozen return type.  scripts/ is importable as a
# top-level package path in this project's test config (same as how other tests
# import from src/...); the file lives at scripts/overfitting_tax.py.
from scripts.overfitting_tax import TaxRow, summarize_tax


def test_summarize_tax_basic():
    """Mean in-sample is averaged and tax = mean_in_sample − fitted_oos.

    in_sample_sharpes=[1.0, 2.0] → mean 1.5; fitted_oos=0.3 → tax 1.2.  Every
    TaxRow field is asserted so a regression in any one (e.g. swapping fixed and
    fitted, or mis-signing the tax) fails loudly rather than silently shipping a
    wrong summary into the basket report.
    """
    row = summarize_tax(
        symbol="SPY",
        n_folds=4,
        fixed_oos=0.10,
        fitted_oos=0.30,
        bh=0.50,
        in_sample_sharpes=[1.0, 2.0],
    )

    assert isinstance(row, TaxRow)
    assert row.symbol == "SPY"
    assert row.n_folds == 4
    assert row.fixed_oos_sharpe == pytest.approx(0.10)
    assert row.fitted_oos_sharpe == pytest.approx(0.30)
    assert row.bh_sharpe == pytest.approx(0.50)
    assert row.mean_in_sample_sharpe == pytest.approx(1.5)
    # tax = 1.5 − 0.3 = 1.2 — the gap the optimizer fooled itself by.
    assert row.tax == pytest.approx(1.2)


def test_summarize_tax_empty_in_sample():
    """Empty in_sample_sharpes → mean_in_sample 0.0 and tax == −fitted_oos.

    statistics.mean([]) raises; the helper must guard it so a symbol that
    produced no folds reports a 0.0 in-sample rather than crashing the sweep.
    With mean_in_sample forced to 0.0, the tax collapses to 0.0 − fitted_oos.
    """
    row = summarize_tax(
        symbol="ZZZ",
        n_folds=0,
        fixed_oos=0.0,
        fitted_oos=0.40,
        bh=0.0,
        in_sample_sharpes=[],
    )

    assert row.mean_in_sample_sharpe == pytest.approx(0.0)
    # tax = 0.0 − 0.40 = −0.40 (the empty guard, no ZeroDivision / ValueError).
    assert row.tax == pytest.approx(-0.40)
