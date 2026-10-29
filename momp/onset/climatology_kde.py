"""Kernel-density climatology of onset days: conditional and unconditional onset probabilities.

A cell's historical onset days are smoothed with a Gaussian kernel density
estimate (Scott's bandwidth). Its CDF is built the way R's ``density()`` plus
``approxfun(rule = 2)`` build one: the density on 512 grid points spanning the
data +/- 3 bandwidths, cumulated and normalised, with the end values held
outside the grid. From that CDF F, for an issue day d and lead k (1..horizon):

* unconditional: P(onset on day d + k) = F(d + k) - F(d + k - 1), and lead 0
  is P(onset already happened by d) = F(d);
* conditional: P(onset on day d + k | no onset by d) = (F(d + k) - F(d + k - 1)) / (1 - F(d)).

Every probability is floored at MIN_PROB while each row keeps its sum (the
excess comes off the entries above the floor in proportion), so no lead has
probability exactly zero. Optionally the KDE is refitted without each test
year (leave-one-year-out), so a year is never forecast from its own onset.

The numerics are a port of the onset_blending package's climatology, operation
for operation, so the two give identical probabilities for the same inputs.
Days are plain numbers on any common axis (for ROMP, the day of the year).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

MIN_PROB = 5e-7
EPS = 1e-12
MIN_ONSETS = 10


def fit_kde(onset_days):
    """A Gaussian KDE (Scott's bandwidth) of the onset days, or None with fewer than MIN_ONSETS of them.

    NaN days (years without an onset) are dropped first. A degenerate sample
    that scipy cannot fit (for example all days equal) also gives None.
    """
    x = np.asarray(onset_days, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < MIN_ONSETS:
        return None
    try:
        return gaussian_kde(x, bw_method="scott")
    except Exception:
        return None


def onset_cdf(kde):
    """The KDE's CDF as a callable, built like R's density() + approxfun(rule = 2).

    The density is evaluated on 512 points from the smallest day minus three
    bandwidths to the largest plus three, cumulated and normalised; outside
    that range the CDF holds its first and last value.
    """
    data = kde.dataset.flatten()
    bandwidth = kde.factor * np.std(data, ddof=1)
    grid = np.linspace(data.min() - 3 * bandwidth, data.max() + 3 * bandwidth, 512)
    pdf = kde.evaluate(grid)
    cdf = np.cumsum(pdf) / np.sum(pdf)
    return interp1d(grid, cdf, bounds_error=False, fill_value=(cdf[0], cdf[-1]))


def floor_probabilities(p_raw, target_sum, lb: float = MIN_PROB, eps: float = EPS):
    """Each row of p_raw at least lb, rescaled to keep its target sum (clipped to [0, 1]).

    The excess over the target comes off what lies above the floor, in
    proportion. Rows whose target is zero become zero; rows whose target is
    too small for every entry to reach the floor become uniform.
    """
    p_raw = np.asarray(p_raw, dtype=float)
    target_sum = np.asarray(target_sum, dtype=float).copy()
    p_raw = np.maximum(p_raw, 0.0)
    p_raw = np.where(np.isfinite(p_raw), p_raw, 0.0)
    target_sum = np.where(np.isfinite(target_sum) & (target_sum >= 0.0), target_sum, 0.0)
    target_sum = np.minimum(target_sum, 1.0)

    out = np.empty_like(p_raw)
    zero = target_sum <= eps
    small = (~zero) & (target_sum < p_raw.shape[1] * lb)
    normal = ~(zero | small)
    out[zero] = 0.0
    out[small] = target_sum[small, None] / p_raw.shape[1]
    if normal.any():
        p = np.maximum(p_raw[normal], lb)
        excess = p.sum(axis=1) - target_sum[normal]
        no_reduction = excess <= eps
        adjusted = np.empty_like(p)
        adjusted[no_reduction] = p[no_reduction]
        reduce = ~no_reduction
        if reduce.any():
            p_reduce = p[reduce]
            reducible = p_reduce - lb
            reducible_sum = reducible.sum(axis=1)
            degenerate = reducible_sum <= eps
            reduced = np.empty_like(p_reduce)
            targets = target_sum[normal][reduce]
            reduced[degenerate] = targets[degenerate, None] / p_raw.shape[1]
            regular = ~degenerate
            reduced[regular] = p_reduce[regular] - (
                excess[reduce][regular, None] * reducible[regular] / reducible_sum[regular, None]
            )
            adjusted[reduce] = np.maximum(reduced, lb)
        out[normal] = adjusted
    return out


def onset_probabilities(kde, issue_days, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Conditional (rows, horizon) and unconditional (rows, horizon + 1) probabilities from one KDE.

    One row per issue day. Conditional column k - 1 is P(onset on issue day +
    k | none by the issue day); unconditional column 0 is P(onset by the issue
    day) and column k is P(onset on issue day + k). Both are NaN when kde is
    None. A conditional row sums to P(onset within the horizon | none yet),
    except when the KDE has put (almost) all mass before the issue day, where
    it falls back to the unconditional shape.
    """
    d0 = np.asarray(issue_days, dtype=int)
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    conditional = np.full((len(d0), horizon), np.nan)
    unconditional = np.full((len(d0), horizon + 1), np.nan)
    if kde is None or len(d0) == 0:
        return conditional, unconditional
    cdf = onset_cdf(kde)
    days = np.arange(1, horizon + 1)
    num = cdf(d0[:, None] + days) - cdf(d0[:, None] + days - 1)
    num = np.maximum(num, 0.0)
    num = np.where(np.isfinite(num), num, 0.0)

    base_prob = cdf(d0)
    unconditional[:, 0] = base_prob
    unconditional[:, 1:] = floor_probabilities(num, num.sum(axis=1))

    denom = 1.0 - base_prob
    cond = np.empty_like(num)
    fallback = (~np.isfinite(denom)) | (denom <= EPS)
    if fallback.any():
        cond[fallback] = floor_probabilities(num[fallback], num[fallback].sum(axis=1))
    regular = ~fallback
    if regular.any():
        p_raw = num[regular] / denom[regular, None]
        p_raw = np.maximum(p_raw, 0.0)
        p_raw = np.where(np.isfinite(p_raw), p_raw, 0.0)
        target_sum = p_raw.sum(axis=1)
        empty = (~np.isfinite(target_sum)) | (target_sum <= EPS)
        cond_regular = np.empty_like(p_raw)
        cond_regular[empty] = 0.0
        if (~empty).any():
            cond_regular[~empty] = floor_probabilities(p_raw[~empty], target_sum[~empty])
        cond[regular] = cond_regular
    conditional[:] = cond
    return conditional, unconditional


def _cell_onsets(value, cv: bool):
    """(years or None, onset days) of one cell, from a Series indexed by year or a plain array of days."""
    if isinstance(value, pd.Series):
        return value.index.to_numpy(dtype=int), value.to_numpy(dtype=float)
    if cv:
        raise ValueError("cv_years needs each cell's onsets as a pandas Series indexed by year")
    return None, np.asarray(value, dtype=float)


def _probabilities(onset_days_by_cell, issue_days, horizon, cv_years, which: int) -> xr.DataArray:
    """The conditional (which = 0) or unconditional (which = 1) array over (cell, [year,] issue, lead)."""
    issue = np.asarray(issue_days, dtype=int)
    cells = list(onset_days_by_cell)
    cv = cv_years is not None
    test_years = np.asarray(list(cv_years), dtype=int) if cv else None
    n_lead = int(horizon) + which
    shape = (len(cells), len(test_years), len(issue), n_lead) if cv else (len(cells), len(issue), n_lead)
    out = np.full(shape, np.nan)
    for n, cell in enumerate(cells):
        years, days = _cell_onsets(onset_days_by_cell[cell], cv)
        if not cv:
            out[n] = onset_probabilities(fit_kde(days), issue, horizon)[which]
            continue
        for m, year in enumerate(test_years):
            out[n, m] = onset_probabilities(fit_kde(days[years != year]), issue, horizon)[which]
    leads = np.arange(1 - which, int(horizon) + 1)
    coords = {"cell": pd.Index(cells, dtype=object, tupleize_cols=False)}
    dims = ["cell"]
    if cv:
        coords["year"] = test_years
        dims.append("year")
    coords["issue"] = issue
    coords["lead"] = leads
    return xr.DataArray(out, dims=dims + ["issue", "lead"], coords=coords)


def conditional_probabilities(onset_days_by_cell: Mapping, issue_days: Sequence[int], horizon: int, *,
                              cv_years: Sequence[int] | None = None) -> xr.DataArray:
    """P(onset on lead k | no onset by the issue day) for leads 1..horizon, per cell, issue day and optionally test year.

    onset_days_by_cell maps each cell to its historical onset days: a pandas
    Series indexed by year, or, without cv_years, any array of days (NaN for a
    year without onset is fine). issue_days are the issue days on the same day
    axis as the onsets. The result has dims (cell, issue, lead), or (cell,
    year, issue, lead) when cv_years is given, in which case each test year's
    KDE is fitted without that year's onset. A cell with fewer than
    MIN_ONSETS onsets (after any leave-out) is NaN.
    """
    return _probabilities(onset_days_by_cell, issue_days, horizon, cv_years, 0)


def unconditional_probabilities(onset_days_by_cell: Mapping, issue_days: Sequence[int], horizon: int, *,
                                cv_years: Sequence[int] | None = None) -> xr.DataArray:
    """P(onset already happened by the issue day) at lead 0, then P(onset on lead k) for k = 1..horizon.

    Arguments and dims as conditional_probabilities, with lead running 0..horizon.
    """
    return _probabilities(onset_days_by_cell, issue_days, horizon, cv_years, 1)
