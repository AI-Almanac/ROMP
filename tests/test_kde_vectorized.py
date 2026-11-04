"""The vectorised KDE climatology equals the per-unit scipy computation (`onset_probabilities` on `fit_kde`)."""

import numpy as np
import pandas as pd
import pytest

from hcwf.references import kde


def _per_unit(years_by_unit, days_by_unit, keys, max_h, cv):
    """Compute the probabilities with one scipy KDE per unit (and per key year with ``cv``), the pre-vectorised way."""
    d0 = keys["d0"].to_numpy(dtype=int)
    windows = keys["window"].to_numpy(dtype=int)
    key_years = keys["year"].to_numpy(dtype=int)
    conditional = np.full((len(days_by_unit), len(keys), max_h), np.nan)
    unconditional = np.full((len(days_by_unit), len(keys), max_h + 1), np.nan)
    for u, (years, days) in enumerate(zip(years_by_unit, days_by_unit)):
        if not cv:
            conditional[u], unconditional[u] = kde.onset_probabilities(kde.fit_kde(days), d0, max_h, windows)
            continue
        for year in np.unique(key_years):
            rows = np.flatnonzero(key_years == year)
            fitted = kde.fit_kde(days[years != year])
            conditional[u, rows], unconditional[u, rows] = kde.onset_probabilities(fitted, d0[rows], max_h,
                                                                                    windows[rows])
    return conditional, unconditional


def _units(rng):
    """Units with varied onset counts, NaN days, 0/1/2 onsets, constant days and the leap year 2000 among the years."""
    years = np.arange(1998, 2026)
    units = []
    for i in range(40):
        n = int(rng.integers(9, len(years) + 1))
        chosen = np.sort(rng.choice(years, size=n, replace=False))
        days = np.round(rng.normal(150 + 3 * i, rng.choice([0.5, 4, 12]), n))
        days[rng.random(n) < 0.1] = np.nan
        units.append((chosen, days))
    for n in (0, 1, 2, 9, 10):
        units.append((years[:n], np.round(rng.normal(160, 8, n))))
    units.append((years[:15], np.full(15, 170.0)))           # zero variance
    units.append((years[:12], np.full(12, 170.5) + 0.0))     # zero variance, fractional
    units.append((years[:20], np.array([100.0, 100.0] + [101.0] * 18)))   # tiny variance
    units.append((years, np.round(rng.normal(30, 2, len(years)))))        # onsets near the grid's early end
    units.append((years, np.round(rng.normal(300, 3, len(years)))))       # onsets after every issue day
    units.append((np.tile(years, 4)[:99], np.round(rng.normal(150, 5, 99))))   # many onsets (several per year)
    return [u[0] for u in units], [u[1] for u in units]


def _keys(rng, by_year):
    """Keys across month ends and a leap year, with per-key horizons including 0, -1 and the maximum."""
    issue = pd.date_range("2000-02-20", "2000-04-10").dayofyear.to_numpy() + np.array([0])
    issue = np.concatenate([issue, np.arange(100, 330, 9)])
    windows = rng.choice([-1, 0, 5, 14, 30, 45], size=len(issue))
    windows[:3] = 45
    if not by_year:
        return pd.DataFrame({"year": 0, "d0": issue, "window": windows})
    years = np.array([1999, 2000, 2004, 2024])
    return pd.DataFrame({"year": np.repeat(years, len(issue)), "d0": np.tile(issue, len(years)),
                         "window": np.tile(windows, len(years))})


@pytest.mark.parametrize("cv", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_vectorised_equals_per_unit_scipy(seed, cv):
    """Every probability is within 1e-9 relative of the per-unit scipy result, with the same NaN pattern."""
    rng = np.random.default_rng(seed)
    years, days = _units(rng)
    keys = _keys(rng, by_year=cv)
    new = kde.unit_probabilities(years, days, keys, 45, cv=cv)
    old = _per_unit(years, days, keys, 45, cv)
    for got, want in zip(new, old):
        assert got.shape == want.shape
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=0)
    fitted = ~np.isnan(new[1][:, :, 0]).all(axis=1)   # a unit without a fit is NaN everywhere
    assert fitted.sum() > 30 and not fitted[40:42].any()


def test_blocks_of_units_give_the_same_result(monkeypatch):
    """Splitting the units into small blocks and fit chunks changes nothing."""
    rng = np.random.default_rng(3)
    years, days = _units(rng)
    keys = _keys(rng, by_year=True)
    whole = kde.unit_probabilities(years, days, keys, 45, cv=True)
    monkeypatch.setattr(kde, "BLOCK_ELEMENTS", 1)
    monkeypatch.setattr(kde, "FIT_CHUNK", 7)
    split = kde.unit_probabilities(years, days, keys, 45, cv=True)
    for a, b in zip(whole, split):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)
