"""The KDE climatology matches onset_blending's probabilities and keeps its row sums and leave-one-year-out."""

import sys

import numpy as np
import pandas as pd
import pytest

from momp.onset import climatology_kde as kde

ONSET_BLENDING = "/home/hcwf/code/onset_blending"


@pytest.fixture(scope="module")
def reference():
    """onset_blending's climatology module, or a skip when it or its dependencies are missing."""
    if ONSET_BLENDING not in sys.path:
        sys.path.insert(0, ONSET_BLENDING)
    try:
        import onset_blending.core.climatology as module
    except Exception as error:
        pytest.skip(f"onset_blending is not importable from {ONSET_BLENDING}: {error!r}")
    return module


def _onsets(rng, n_cells, n_years):
    """Random onset days per cell as Series indexed by year, with some missing years."""
    years = np.arange(2000, 2000 + n_years)
    cells = {}
    for cell in range(n_cells):
        days = np.round(rng.normal(150 + 6 * cell, 9, n_years))
        days[rng.random(n_years) < 0.15] = np.nan
        cells[f"c{cell}"] = pd.Series(days, index=years)
    return cells


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_probabilities_equal_onset_blendings(reference, seed):
    """Conditional and unconditional probabilities equal onset_blending's for random onsets and issue grids."""
    rng = np.random.default_rng(seed)
    cells = _onsets(rng, 4, 20)
    issue = np.sort(rng.choice(np.arange(100, 230), size=25, replace=False))
    horizon = int(rng.integers(5, 31))
    conditional = kde.conditional_probabilities(cells, issue, horizon)
    unconditional = kde.unconditional_probabilities(cells, issue, horizon)
    assert conditional.dims == ("cell", "issue", "lead")
    assert list(conditional["lead"].values) == list(range(1, horizon + 1))
    assert list(unconditional["lead"].values) == list(range(0, horizon + 1))

    keys = pd.DataFrame({"d0": issue, "window": horizon, "year": 0})
    context = {"keys": keys, "max_h": horizon, "cv_by_year": False}
    days = [s.dropna().to_numpy() for s in cells.values()]
    years = [s.dropna().index.to_numpy() for s in cells.values()]
    ref_cond, ref_uncond = reference._climatology_task((years, days), context)
    np.testing.assert_array_equal(conditional.values, ref_cond)
    np.testing.assert_array_equal(unconditional.values, ref_uncond)


def test_cv_equals_onset_blendings_and_leaves_the_test_year_out(reference):
    """With cv_years each test year's row equals onset_blending's refit without that year, and differs from the full fit."""
    rng = np.random.default_rng(5)
    cells = _onsets(rng, 3, 22)
    issue = np.arange(120, 200, 7)
    horizon = 15
    test_years = [2003, 2010, 2021]
    cv = kde.conditional_probabilities(cells, issue, horizon, cv_years=test_years)
    full = kde.conditional_probabilities(cells, issue, horizon)
    assert cv.dims == ("cell", "year", "issue", "lead")

    keys = pd.DataFrame([(y, d, horizon) for y in test_years for d in issue], columns=["year", "d0", "window"])
    context = {"keys": keys, "max_h": horizon, "cv_by_year": True}
    days = [s.dropna().to_numpy() for s in cells.values()]
    years = [s.dropna().index.to_numpy() for s in cells.values()]
    ref_cond, _ = reference._climatology_task((years, days), context)
    mine = cv.values.reshape(len(cells), len(test_years) * len(issue), horizon)
    np.testing.assert_array_equal(mine, ref_cond)

    for m, year in enumerate(test_years):
        without = {c: s[s.index != year] for c, s in cells.items()}
        expected = kde.conditional_probabilities(without, issue, horizon)
        np.testing.assert_array_equal(cv.isel(year=m).values, expected.values)
        assert not np.array_equal(cv.isel(year=m).values, full.values)


def test_conditional_rows_sum_to_the_conditional_mass():
    """A conditional row sums to P(onset within the horizon | none by the issue day) and keeps every lead above the floor."""
    rng = np.random.default_rng(3)
    onsets = rng.normal(160, 8, 40)
    horizon, issue = 20, np.array([130, 150, 160, 170])
    probs = kde.conditional_probabilities({"a": onsets}, issue, horizon).values[0]
    cdf = kde.onset_cdf(kde.fit_kde(onsets))
    mass = (cdf(issue + horizon) - cdf(issue)) / (1 - cdf(issue))
    np.testing.assert_allclose(probs.sum(axis=1), mass, rtol=1e-6)
    assert (probs >= kde.MIN_PROB).all()

    unconditional = kde.unconditional_probabilities({"a": onsets}, issue, horizon).values[0]
    np.testing.assert_allclose(unconditional[:, 0], cdf(issue))
    np.testing.assert_allclose(unconditional[:, 1:].sum(axis=1), cdf(issue + horizon) - cdf(issue), rtol=1e-6)


def test_floor_keeps_row_sums():
    """The floor lifts zero entries to MIN_PROB while the row keeps its target sum."""
    p = np.array([[0.0, 0.3, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0]])
    target = np.array([1.0, 0.0])
    out = kde.floor_probabilities(p, target)
    np.testing.assert_allclose(out.sum(axis=1), target)
    assert (out[0] >= kde.MIN_PROB).all()
    assert (out[1] == 0).all()


def test_too_few_onsets_give_nan_and_cv_needs_years():
    """A cell with fewer than ten onsets is NaN, and cv_years needs Series indexed by year."""
    few = {"a": np.arange(140.0, 148.0)}
    assert np.isnan(kde.conditional_probabilities(few, [150], 5).values).all()
    with pytest.raises(ValueError, match="indexed by year"):
        kde.conditional_probabilities(few, [150], 5, cv_years=[2000])
    years = pd.Series(np.arange(140.0, 152.0), index=np.arange(2000, 2012))
    left_out = kde.conditional_probabilities({"a": years.iloc[:10]}, [150], 5, cv_years=[2000]).values
    assert np.isnan(left_out).all()
