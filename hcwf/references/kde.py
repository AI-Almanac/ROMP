"""Kernel-density climatology of onset days: conditional and unconditional onset probabilities.

A cell's historical onset days are smoothed with a Gaussian kernel density
estimate (Scott's bandwidth). Its CDF is built the way R's ``density()`` plus
``approxfun(rule = 2)`` build one: the density on 512 grid points spanning the
data +/- 3 bandwidths, cumulated and normalised, with the end values held
outside the grid. From that CDF F, for an issue day d and lead k (1..horizon):

* unconditional: P(onset on day d + k) = F(d + k) - F(d + k - 1), and lead 0
  is P(onset already happened by d) = F(d);
* conditional: P(onset on day d + k | no onset by d) = (F(d + k) - F(d + k - 1)) / (1 - F(d)).

Every probability is floored at `MIN_PROB` while each row keeps its sum (the
excess comes off the entries above the floor in proportion), so no lead has
probability exactly zero. Optionally the KDE is refitted without each test
year (leave-one-year-out), so a year is never forecast from its own onset.

The numerics are a port of the onset_blending package's climatology, operation
for operation, so the two give identical probabilities for the same inputs
(for seasons inside one calendar year; see below).
Days are plain numbers on any common axis: the day of the year for HCWF's reference
forecast, the days since the season start for onset_blending (`season_days` makes either). Each date is counted
from the season start of its season's year (the year the onset was detected for, the target year for an init), so
a season crossing New Year (Kenya's short rains, October to January) counts a January onset after the autumn
inits; onset_blending counted from the calendar year of the date, so parity with it holds for seasons inside one
calendar year.

Around the core sits onset_blending's caller logic. `issue_grid` lists the issue dates of the
test years with each one's day of the season and its horizon, `climatology_years` picks the
training and test windows, `eligible_onsets` keeps the units with enough onset years, and
`probabilities_for_issues` computes the probabilities once per distinct (season day, horizon),
or per (year, season day, horizon) when the KDE is refitted without each test year, in tasks of
units over the run's pool (`hcwf.parallel`). `conditional_probabilities` and
`unconditional_probabilities` are the same computation on HCWF's cells and one horizon.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

MIN_PROB = 5e-7
"""The floor of every probability."""
EPS = 1e-12
"""The tolerance below which a sum or an excess counts as zero."""
MIN_ONSETS = 10
"""The fewest onset days a KDE is fitted on."""


def fit_kde(onset_days):
    """Fit a Gaussian KDE (Scott's bandwidth) of the onset days, or None with fewer than `MIN_ONSETS` of them.

    NaN days (years without an onset) are dropped first. A degenerate sample
    that scipy cannot fit (for example all days equal) also gives None.

    Parameters
    ----------
    onset_days : array_like of float
        The onset days of one cell, NaN for a year without onset.

    Returns
    -------
    scipy.stats.gaussian_kde or None
        The fitted estimate, or None when it cannot be fitted.
    """
    x = np.asarray(onset_days, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < MIN_ONSETS:
        return None
    from scipy.stats import gaussian_kde   # scipy is slow to import; only fit_kde and onset_cdf need it

    try:
        return gaussian_kde(x, bw_method="scott")
    except Exception:
        return None


def onset_cdf(kde):
    """Build the KDE's CDF as a callable, like R's ``density()`` + ``approxfun(rule = 2)``.

    The density is evaluated on 512 points from the smallest day minus three
    bandwidths to the largest plus three, cumulated and normalised; outside
    that range the CDF holds its first and last value.

    Parameters
    ----------
    kde : scipy.stats.gaussian_kde
        The estimate, as `fit_kde` returns it.

    Returns
    -------
    callable
        ``cdf(days)`` returns the CDF at the given days (a ``scipy.interpolate.interp1d``).
    """
    from scipy.interpolate import interp1d

    data = kde.dataset.flatten()
    bandwidth = kde.factor * np.std(data, ddof=1)
    grid = np.linspace(data.min() - 3 * bandwidth, data.max() + 3 * bandwidth, 512)
    pdf = kde.evaluate(grid)
    cdf = np.cumsum(pdf) / np.sum(pdf)
    return interp1d(grid, cdf, bounds_error=False, fill_value=(cdf[0], cdf[-1]))


def floor_probabilities(p_raw, target_sum, lb: float = MIN_PROB, eps: float = EPS):
    """Floor each row of ``p_raw`` at ``lb``, rescaled to keep its target sum (clipped to [0, 1]).

    The excess over the target comes off what lies above the floor, in
    proportion. Rows whose target is zero become zero; rows whose target is
    too small for every entry to reach the floor become uniform.

    Parameters
    ----------
    p_raw : array_like of float, shape (row, lead)
        The raw probabilities; negative and non-finite entries count as 0.
    target_sum : array_like of float, shape (row,)
        The sum each row must keep; non-finite or negative counts as 0 and it is capped at 1.
    lb : float, default MIN_PROB
        The floor.
    eps : float, default EPS
        The tolerance below which a sum or an excess counts as zero.

    Returns
    -------
    numpy.ndarray of float64, shape (row, lead)
        The floored probabilities.
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


def lead_probabilities(cdf_values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn the CDF at the issue day and at each of the next ``length`` days into the floored probabilities.

    Column 0 of ``cdf_values`` is F(d), column k is F(d + k). The unconditional probability of lead k is
    F(d + k) - F(d + k - 1) and the conditional one that over 1 - F(d), each floored by `floor_probabilities` to keep
    its row sum; where (almost) all the mass lies before the issue day the conditional probabilities fall back to the
    unconditional shape.

    Parameters
    ----------
    cdf_values : array_like of float, shape (row, length + 1)
        The CDF at the issue day and the ``length`` days after it.

    Returns
    -------
    conditional : numpy.ndarray of float64, shape (row, length)
        The conditional probabilities of the leads 1 to ``length``.
    base : numpy.ndarray of float64, shape (row,)
        P(onset by the issue day), the unconditional probability of lead 0.
    unconditional : numpy.ndarray of float64, shape (row, length)
        The unconditional probabilities of the leads 1 to ``length``.
    """
    cdf_values = np.asarray(cdf_values, dtype=float)
    num = cdf_values[:, 1:] - cdf_values[:, :-1]
    num = np.maximum(num, 0.0)
    num = np.where(np.isfinite(num), num, 0.0)

    base_prob = cdf_values[:, 0]
    unconditional = floor_probabilities(num, num.sum(axis=1))

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
    return cond, base_prob, unconditional


def onset_probabilities(kde, issue_days, horizon: int, horizons=None) -> tuple[np.ndarray, np.ndarray]:
    """Compute the conditional and unconditional onset probabilities from one KDE.

    One row per issue day. Conditional column k - 1 is P(onset on issue day +
    k | none by the issue day); unconditional column 0 is P(onset by the issue
    day) and column k is P(onset on issue day + k). Both are NaN when ``kde`` is
    None. A conditional row sums to P(onset within the horizon | none yet),
    except when the KDE has put (almost) all mass before the issue day, where
    it falls back to the unconditional shape.

    ``horizons`` gives each row its own horizon (at most ``horizon``, which sets the width): a row's leads
    past its horizon stay NaN, and a row whose horizon is not positive is all NaN (an issue date outside
    every configured horizon).

    Parameters
    ----------
    kde : scipy.stats.gaussian_kde or None
        The estimate of the cell's onset days.
    issue_days : array_like of int, shape (row,)
        The issue days, on the same day axis as the onsets.
    horizon : int
        The number of leads of the result (its width).
    horizons : array_like of int, shape (row,), optional
        Each row's own horizon; by default ``horizon`` for all.

    Returns
    -------
    conditional : numpy.ndarray of float64, shape (row, horizon)
        The conditional probabilities of the leads 1 to ``horizon``.
    unconditional : numpy.ndarray of float64, shape (row, horizon + 1)
        The unconditional probabilities of the leads 0 to ``horizon``.

    Raises
    ------
    ValueError
        ``horizon`` is less than 1.
    """
    d0 = np.asarray(issue_days, dtype=int)
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    windows = np.full(len(d0), horizon) if horizons is None else np.asarray(horizons, dtype=int)
    conditional = np.full((len(d0), horizon), np.nan)
    unconditional = np.full((len(d0), horizon + 1), np.nan)
    if kde is None or len(d0) == 0:
        return conditional, unconditional
    cdf = onset_cdf(kde)
    for length in np.unique(windows[windows > 0]):
        rows = np.flatnonzero(windows == length)
        length = int(length)
        cond, base_prob, uncond = lead_probabilities(cdf(d0[rows, None] + np.arange(length + 1)))
        unconditional[rows, 0] = base_prob
        unconditional[rows, 1:length + 1] = uncond
        conditional[rows, :length] = cond
    return conditional, unconditional


UNITS_PER_TASK = 200
"""Units per process task: enough to amortise a task, few enough to spread."""


GRID_POINTS = 512
"""The number of points of the grid a KDE's CDF is built on."""
BLOCK_ELEMENTS = 2_000_000
"""The most (unit, key, lead) elements a block of units computes at once (about 16 MB per float64 array)."""
FIT_CHUNK = 2048
"""The most KDE fits evaluated on their grids at once (2048 x 512 float64 is 8 MB per temporary)."""


def _cdf_on_days(data: np.ndarray, valid: np.ndarray, first_day: int, n_days: int) -> np.ndarray:
    """Evaluate the KDE CDFs of many samples at consecutive integer days, like `fit_kde` with `onset_cdf`.

    Each sample is fitted as ``gaussian_kde(x, bw_method="scott")`` and its CDF built as `onset_cdf` does, then
    interpolated as ``interp1d`` does, with the same operations in the same order as scipy and numpy
    (samples of equal size are processed together so that every reduction runs over the same elements), so the
    results equal the per-sample computation bit for bit (with this scipy and numpy). A sample that `fit_kde` would
    not fit (fewer than `MIN_ONSETS` days, or a singular covariance) gives NaN. The few samples whose variance is
    so small that rounding could decide whether scipy fits them are left to `fit_kde`.

    Parameters
    ----------
    data : numpy.ndarray of float64, shape (fit, day)
        The sample days of each fit; the entries where ``valid`` is false are ignored.
    valid : numpy.ndarray of bool, shape (fit, day)
        Which entries belong to the sample.
    first_day : int
        The first day to evaluate.
    n_days : int
        The number of consecutive days to evaluate.

    Returns
    -------
    numpy.ndarray of float64, shape (fit, n_days)
        The CDF at ``first_day`` to ``first_day + n_days - 1``, NaN for a fit that fails.
    """
    n_fit = len(data)
    days = first_day + np.arange(n_days, dtype=float)
    out = np.full((n_fit, n_days), np.nan)
    count = valid.sum(axis=1)
    # each sample's days first, in their order (scipy's dataset), the rest padding
    packed = np.take_along_axis(data, np.argsort(~valid, axis=1, kind="stable"), axis=1)
    for n in np.unique(count[count >= MIN_ONSETS]):
        members = np.flatnonzero(count == n)
        weights = np.ones(n) / n
        factor = np.power(1 / np.vecdot(weights, weights), -1.0 / 5.0)   # Scott's factor, neff ** (-1 / (d + 4))
        weight_sum = weights.sum()
        fact = weight_sum - 1 * np.sum(weights * weights) / weight_sum      # numpy.cov's normalisation, ddof = 1
        for start in range(0, len(members), FIT_CHUNK):
            rows = members[start:start + FIT_CHUNK]
            sample = np.ascontiguousarray(packed[rows, :n])
            # per sample, as numpy.cov(aweights=weights) and numpy.std do (numpy's 2-D row sums add in another order)
            variance, spread = np.empty(len(rows)), np.empty(len(rows))
            for i in range(len(rows)):
                one = sample[i:i + 1]
                centred = one - (np.multiply(one, weights).sum(axis=1) / weight_sum)[:, None]
                variance[i] = np.dot(centred, (centred * weights).T)[0, 0]
                spread[i] = np.std(sample[i], ddof=1)
            variance *= np.true_divide(1, fact)
            scale = np.abs(sample).max(axis=1)
            fine = variance > 1e-12 * (scale ** 2 + 1.0)
            for i in np.flatnonzero(~fine):      # rounding may decide whether scipy fits it: ask scipy
                kde = fit_kde(sample[i])
                if kde is not None:
                    out[rows[i]] = onset_cdf(kde)(days)
            rows, sample, variance, spread = rows[fine], sample[fine], variance[fine], spread[fine]
            if not len(rows):
                continue
            cho = np.sqrt(variance) * factor
            bandwidth = factor * spread
            low = sample.min(axis=1) - 3 * bandwidth
            high = sample.max(axis=1) + 3 * bandwidth
            grid = np.linspace(low, high, GRID_POINTS, axis=1)
            whitening = 1.0 / cho
            norm = np.power(2 * np.pi, -0.5) / cho
            whitened_grid = grid * whitening[:, None]
            whitened = sample * whitening[:, None]
            pdf = np.zeros_like(grid)
            for j in range(n):
                residual = whitened[:, j, None] - whitened_grid
                pdf += weights[j] * (np.exp(-(residual * residual) / 2.0) * norm[:, None])
            total = np.array([row.sum() for row in pdf])   # one 1-D sum per row: numpy's 2-D row sums add in another order
            cdf = np.cumsum(pdf, axis=1) / total[:, None]
            out[rows] = _interpolate(grid, cdf, days)
    return out


def _interpolate(grid: np.ndarray, cdf: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Interpolate each row's CDF on its grid at ``days``, as ``interp1d`` does with the end values held outside.

    Parameters
    ----------
    grid : numpy.ndarray of float64, shape (fit, point)
        Each fit's increasing grid.
    cdf : numpy.ndarray of float64, shape (fit, point)
        The CDF on each grid.
    days : numpy.ndarray of float64, shape (day,)
        The points to evaluate at, the same for every fit.

    Returns
    -------
    numpy.ndarray of float64, shape (fit, day)
        The interpolated values.
    """
    last = grid.shape[1] - 1
    span = (grid[:, -1] - grid[:, 0])[:, None] / last
    # numpy.searchsorted(grid, day) (the first point not below the day), found by estimate and correction
    above = np.clip(np.ceil((days[None, :] - grid[:, :1]) / span), 1, last).astype(np.int64)
    for _ in range(2):
        above -= (above > 1) & (np.take_along_axis(grid, above - 1, axis=1) >= days[None, :])
        above += (above < last) & (np.take_along_axis(grid, above, axis=1) < days[None, :])
    x_lo, x_hi = np.take_along_axis(grid, above - 1, axis=1), np.take_along_axis(grid, above, axis=1)
    y_lo, y_hi = np.take_along_axis(cdf, above - 1, axis=1), np.take_along_axis(cdf, above, axis=1)
    inside = (y_hi - y_lo) / (x_hi - x_lo) * (days[None, :] - x_lo) + y_lo
    inside = np.where(days[None, :] < grid[:, :1], cdf[:, :1], inside)
    return np.where(days[None, :] > grid[:, -1:], cdf[:, -1:], inside)


def _unit_block(years_by_unit: list, days_by_unit: list, d0: np.ndarray, windows: np.ndarray,
                key_years: np.ndarray, max_h: int, cv: bool) -> tuple[np.ndarray, np.ndarray]:
    """Compute the probabilities of some units at every key, all at once.

    Equivalent to `onset_probabilities` on each unit's `fit_kde` (refitted per key year with ``cv``), as arrays over
    (unit, key year, day) for the CDFs and (unit, key, lead) for the rest.

    Parameters
    ----------
    years_by_unit : list of numpy.ndarray of int or None
        The onset years of each unit (only read with ``cv``).
    days_by_unit : list of numpy.ndarray of float
        The onset days of each unit.
    d0 : numpy.ndarray of int, shape (key,)
        The issue day of each key.
    windows : numpy.ndarray of int, shape (key,)
        The horizon of each key.
    key_years : numpy.ndarray of int, shape (key,)
        The year of each key.
    max_h : int
        The width of the lead axis.
    cv : bool
        Fit each key year's KDE without that year's onset.

    Returns
    -------
    conditional : numpy.ndarray of float64, shape (unit, key, max_h)
        The conditional probabilities.
    unconditional : numpy.ndarray of float64, shape (unit, key, max_h + 1)
        The unconditional probabilities.
    """
    n_unit, n_key = len(days_by_unit), len(d0)
    width = max(1, *(len(days) for days in days_by_unit))
    data = np.full((n_unit, width), np.nan)
    unit_years = np.zeros((n_unit, width), dtype=np.int64)
    for u, days in enumerate(days_by_unit):
        data[u, :len(days)] = days
        if cv:
            unit_years[u, :len(days)] = years_by_unit[u]
    present = ~np.isnan(data)
    if cv:
        distinct, year_index = np.unique(key_years, return_inverse=True)
        valid = present[:, None, :] & (unit_years[:, None, :] != distinct[None, :, None])
    else:
        distinct, year_index = np.zeros(1, dtype=np.int64), np.zeros(n_key, dtype=np.int64)
        valid = present[:, None, :]
    n_year = len(distinct)
    first = int(d0.min())
    n_days = int(d0.max()) - first + max_h + 1
    fits = _cdf_on_days(np.broadcast_to(data[:, None, :], valid.shape).reshape(-1, width), valid.reshape(-1, width),
                        first, n_days).reshape(n_unit, n_year, n_days)

    at = (d0 - first)[:, None] + np.arange(max_h + 1)[None, :]
    cdf = fits[np.arange(n_unit)[:, None, None], year_index[None, :, None], at[None, :, :]]
    cdf = cdf.reshape(n_unit * n_key, max_h + 1)
    length = np.tile(windows, n_unit)
    live = (length > 0) & np.isfinite(cdf[:, 0])
    conditional = np.full((n_unit * n_key, max_h), np.nan)
    unconditional = np.full((n_unit * n_key, max_h + 1), np.nan)
    for size in np.unique(length[live]):
        rows = np.flatnonzero(live & (length == size))
        cond, base, uncond = lead_probabilities(cdf[rows, :size + 1])
        unconditional[rows, 0] = base
        unconditional[rows, 1:size + 1] = uncond
        conditional[rows, :size] = cond
    return conditional.reshape(n_unit, n_key, max_h), unconditional.reshape(n_unit, n_key, max_h + 1)


def _unit_task(task: tuple, keys: pd.DataFrame, max_h: int, cv: bool) -> tuple[np.ndarray, np.ndarray]:
    """Compute the probabilities of a run of units at each key (year, d0, window).

    Module-level so that the process pool can pickle it. The units go through `_unit_block` in blocks that keep
    the arrays within `BLOCK_ELEMENTS`.

    Parameters
    ----------
    task : tuple of (list, list)
        The onset years and the onset days of each unit of the run (`unit_probabilities` builds it).
    keys : pandas.DataFrame
        The keys, with the integer columns ``year``, ``d0`` (issue day) and ``window`` (horizon).
    max_h : int
        The width of the lead axis.
    cv : bool
        Fit each key year's KDE without that year's onset.

    Returns
    -------
    conditional : numpy.ndarray of float64, shape (unit, key, max_h)
        The conditional probabilities.
    unconditional : numpy.ndarray of float64, shape (unit, key, max_h + 1)
        The unconditional probabilities.
    """
    years_by_unit, days_by_unit = task
    n_unit, n_key = len(days_by_unit), len(keys)
    conditional = np.full((n_unit, n_key, max_h), np.nan)
    unconditional = np.full((n_unit, n_key, max_h + 1), np.nan)
    if n_unit == 0 or n_key == 0:
        return conditional, unconditional
    d0 = keys["d0"].to_numpy(dtype=int)
    windows = keys["window"].to_numpy(dtype=int)
    key_years = keys["year"].to_numpy(dtype=int)
    block = max(1, BLOCK_ELEMENTS // (n_key * (max_h + 1)))
    for start in range(0, n_unit, block):
        stop = min(start + block, n_unit)
        conditional[start:stop], unconditional[start:stop] = _unit_block(
            years_by_unit[start:stop], [np.asarray(days, dtype=float) for days in days_by_unit[start:stop]],
            d0, windows, key_years, max_h, cv)
    return conditional, unconditional


def unit_probabilities(years_by_unit: Sequence, days_by_unit: Sequence, keys: pd.DataFrame, max_h: int, *,
                       cv: bool = False, workers: int = 1, units_per_task: int = UNITS_PER_TASK):
    """Compute the conditional and unconditional probabilities of every unit at every key.

    Each unit has its onset years and onset days (NaN for none; the years are only read with ``cv``); ``keys`` is a
    table with ``year``, ``d0`` (issue day on the onsets' axis) and ``window`` (the key's horizon, at most ``max_h``; not
    positive for none). With ``cv`` each key year's KDE is fitted without that year's onset. Units go in tasks of
    ``units_per_task`` to the run's pool when ``workers`` > 1 (results are in unit order whatever the workers).

    Parameters
    ----------
    years_by_unit : sequence of numpy.ndarray of int
        The onset years of each unit.
    days_by_unit : sequence of numpy.ndarray of float
        The onset days of each unit, NaN for a year without onset.
    keys : pandas.DataFrame
        The keys, with the columns ``year``, ``d0`` and ``window``.
    max_h : int
        The width of the lead axis.
    cv : bool, default False
        Fit each key year's KDE without that year's onset.
    workers : int, default 1
        The number of pool workers.
    units_per_task : int, default UNITS_PER_TASK
        The number of units in one task.

    Returns
    -------
    conditional : numpy.ndarray of float64, shape (unit, key, max_h)
        The conditional probabilities of the leads 1 to ``max_h``.
    unconditional : numpy.ndarray of float64, shape (unit, key, max_h + 1)
        The unconditional probabilities of the leads 0 to ``max_h``.
    """
    from hcwf import parallel

    n = len(days_by_unit)
    tasks = [(list(years_by_unit[start:start + units_per_task]), list(days_by_unit[start:start + units_per_task]))
             for start in range(0, n, units_per_task)]
    results = parallel.map_items(_unit_task, tasks, (keys, max_h, cv), workers=workers)
    if not results:
        return np.full((0, len(keys), max_h), np.nan), np.full((0, len(keys), max_h + 1), np.nan)
    return np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results])


def _cell_onsets(value, cv: bool):
    """Split one cell's onsets into years and days.

    Parameters
    ----------
    value : pandas.Series or array_like of float
        A Series of onset days indexed by year, or a plain array of days.
    cv : bool
        Whether the years are needed (leave-one-year-out).

    Returns
    -------
    years : numpy.ndarray of int64 or None
        The years, None for a plain array.
    days : numpy.ndarray of float64
        The onset days.

    Raises
    ------
    ValueError
        ``cv`` is true and ``value`` is not a Series.
    """
    if isinstance(value, pd.Series):
        return value.index.to_numpy(dtype=int), value.to_numpy(dtype=float)
    if cv:
        raise ValueError("cv_years needs each cell's onsets as a pandas Series indexed by year")
    return None, np.asarray(value, dtype=float)


def _probabilities(onset_days_by_cell, issue_days, horizon, cv_years, which: int) -> xr.DataArray:
    """Compute the conditional (``which = 0``) or unconditional (``which = 1``) probabilities per cell.

    Parameters
    ----------
    onset_days_by_cell : mapping
        Each cell's onsets (a Series indexed by year, or an array of days without ``cv_years``).
    issue_days : array_like of int
        The issue days.
    horizon : int
        The last lead.
    cv_years : iterable of int or None
        The test years for leave-one-year-out, or None.
    which : int
        0 for the conditional, 1 for the unconditional probabilities.

    Returns
    -------
    xarray.DataArray
        The probabilities over ``(cell, [year,] issue, lead)``, with the leads 1 to ``horizon`` (0 to ``horizon`` for
        ``which = 1``).
    """
    issue = np.asarray(issue_days, dtype=int)
    cells = list(onset_days_by_cell)
    cv = cv_years is not None
    test_years = np.asarray(list(cv_years), dtype=int) if cv else None
    onsets = [_cell_onsets(onset_days_by_cell[cell], cv) for cell in cells]
    rows = np.tile(issue, len(test_years)) if cv else issue
    keys = pd.DataFrame({"year": np.repeat(test_years, len(issue)) if cv else np.zeros(len(rows), dtype=int),
                         "d0": rows, "window": np.full(len(rows), int(horizon))})
    out = unit_probabilities([years for years, _ in onsets], [days for _, days in onsets], keys, int(horizon),
                             cv=cv)[which]
    out = out.reshape((len(cells), len(test_years), len(issue), -1) if cv else (len(cells), len(issue), -1))
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
    """Compute P(onset on lead k | no onset by the issue day) for leads 1 to ``horizon``, per cell and issue day.

    ``onset_days_by_cell`` maps each cell to its historical onset days: a pandas
    Series indexed by year, or, without ``cv_years``, any array of days (NaN for a
    year without onset is fine). ``issue_days`` are the issue days on the same day
    axis as the onsets. The result has dims ``(cell, issue, lead)``, or ``(cell,
    year, issue, lead)`` when ``cv_years`` is given, in which case each test year's
    KDE is fitted without that year's onset. A cell with fewer than
    `MIN_ONSETS` onsets (after any leave-out) is NaN.

    `ValueError` is raised by a called function when ``horizon`` is less than 1, or ``cv_years`` is given and a cell's onsets are not a Series indexed by year.

    Parameters
    ----------
    onset_days_by_cell : mapping
        Each cell's historical onset days.
    issue_days : sequence of int
        The issue days on the same axis as the onsets.
    horizon : int
        The last lead.
    cv_years : sequence of int, optional
        The test years to leave out one at a time; None for no cross-validation.

    Returns
    -------
    xarray.DataArray
        The probabilities, float64 over ``(cell, [year,] issue, lead)``, ``lead`` running 1 to ``horizon``.
    """
    return _probabilities(onset_days_by_cell, issue_days, horizon, cv_years, 0)


def unconditional_probabilities(onset_days_by_cell: Mapping, issue_days: Sequence[int], horizon: int, *,
                                cv_years: Sequence[int] | None = None) -> xr.DataArray:
    """Compute P(onset already happened by the issue day) at lead 0, then P(onset on lead k) for k = 1 to ``horizon``.

    Arguments and dims as `conditional_probabilities`, with ``lead`` running 0 to ``horizon``.

    `ValueError` is raised by a called function when ``horizon`` is less than 1, or ``cv_years`` is given and a cell's onsets are not a Series indexed by year.

    Parameters
    ----------
    onset_days_by_cell : mapping
        Each cell's historical onset days.
    issue_days : sequence of int
        The issue days on the same axis as the onsets.
    horizon : int
        The last lead.
    cv_years : sequence of int, optional
        The test years to leave out one at a time; None for no cross-validation.

    Returns
    -------
    xarray.DataArray
        The probabilities, float64 over ``(cell, [year,] issue, lead)``, ``lead`` running 0 to ``horizon``.
    """
    return _probabilities(onset_days_by_cell, issue_days, horizon, cv_years, 1)


# ---------------------------------------------------------------------------
# The issue grid, the training window and the probabilities of a whole season (onset_blending's climatology)
# ---------------------------------------------------------------------------

def _month_day_of(years: np.ndarray, month: int, day: int) -> np.ndarray:
    """Return the date ``month``-``day`` of each year, as ``datetime64[D]``.

    Parameters
    ----------
    years : numpy.ndarray of int
        The years.
    month : int
        The month, 1 to 12.
    day : int
        The day of the month.

    Returns
    -------
    numpy.ndarray of datetime64[D]
        The dates, one per year.
    """
    first = (years - 1970).astype("datetime64[Y]").astype("datetime64[M]") + np.timedelta64(month - 1, "M")
    return first.astype("datetime64[D]") + np.timedelta64(day - 1, "D")


def season_days(dates, season_start: tuple[int, int] = (1, 1), leap_shift: bool = False,
                label_years=None) -> np.ndarray:
    """Compute the days from the season start (month, day) of each date's season year, as floats; NaN for NaT.

    The default start is 1 January. onset_blending counts from its season start, in real calendar days. A date is
    counted from the season start of its season's year, ``label_years`` (default: the year of the date itself), so a
    season that crosses New Year counts a January date after the December ones: with a 1 October start and label year
    2020, 10 January 2021 is day 101, not -264. With ``leap_shift``, 29 February shares the day of 1 March: dates from
    1 March on count one less for each leap year's 29 February passed since the start (when the start is before
    March), so a calendar date has the same number every year and the KDE's day axis is not blurred by 29 February
    (HCWF's reference forecast).

    Parameters
    ----------
    dates : array_like of datetime64
        The dates, any shape (the result is flattened).
    season_start : tuple of (int, int), default (1, 1)
        The month and day the count starts from.
    leap_shift : bool, default False
        Count 29 February with 1 March (a start before March).
    label_years : int or array_like of int, optional
        The year of the season each date belongs to, broadcast against ``dates``; each date's own year when None.

    Returns
    -------
    numpy.ndarray of float64, shape (date,)
        The day counts, NaN for ``NaT``.
    """
    dates = np.asarray(dates)
    index = pd.DatetimeIndex(dates.ravel())
    valid = ~index.isna()
    own = np.where(valid, index.year, 1970).astype(np.int64)
    label = own if label_years is None else np.broadcast_to(np.asarray(label_years, dtype=np.int64),
                                                            dates.shape).ravel()
    month, day = season_start
    date = index.values.astype("datetime64[D]")
    days = (date - _month_day_of(label, month, day)).astype(np.int64).astype(float)
    if leap_shift and month < 3:
        for year in (label, label + 1):
            leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
            days = days - (leap & valid & (date >= _month_day_of(year, 3, 1)))
    return np.where(valid, days, np.nan)


def climatology_years(observed_years: Sequence[int], forecast_years: Sequence[int], *,
                      train_years: tuple[int, int] | None = None,
                      test_years: tuple[int, int] | None = None) -> tuple[tuple[int, int], tuple[int, int]]:
    """Choose the (train, test) year windows.

    These are the given ones, else the test window spans the forecast years and training runs
    from the first observed year to the year before the first test year (to the last observed year when that
    leaves nothing).

    Parameters
    ----------
    observed_years : sequence of int
        The years with observed onsets.
    forecast_years : sequence of int
        The forecast years.
    train_years : tuple of (int, int), optional
        The first and last training year; chosen when None.
    test_years : tuple of (int, int), optional
        The first and last test year; the span of ``forecast_years`` when None.

    Returns
    -------
    train : tuple of (int, int)
        The first and last training year.
    test : tuple of (int, int)
        The first and last test year.

    Raises
    ------
    ValueError
        ``observed_years`` or ``forecast_years`` is empty.
    """
    if not len(observed_years):
        raise ValueError("Cannot build a climatology without observed years")
    if not len(forecast_years):
        raise ValueError("Cannot build a climatology without forecast years")
    test = test_years or (min(forecast_years), max(forecast_years))
    if train_years is not None:
        return tuple(train_years), tuple(test)
    first = min(observed_years)
    last = min(test[0] - 1, max(observed_years))
    if last < first:
        last = max(observed_years)
    return (first, last), tuple(test)


def eligible_onsets(onsets: pd.DataFrame, window: tuple[int, int], min_onset_years: int) -> tuple[pd.DataFrame, int]:
    """Select the onsets found in the window's years, of the units with at least ``min_onset_years`` of them.

    Parameters
    ----------
    onsets : pandas.DataFrame
        The onsets, with the columns ``id``, ``year`` and ``onset_day`` (NaN for none).
    window : tuple of (int, int)
        The first and last year to keep.
    min_onset_years : int
        The fewest onsets a unit must have in the window.

    Returns
    -------
    pandas.DataFrame
        The onsets of the eligible units in the window.
    int
        The number of eligible units.

    Raises
    ------
    ValueError
        No unit is eligible.
    """
    onsets = onsets[onsets["onset_day"].notna() & onsets["year"].between(*window)].copy()
    counts = onsets.groupby("id")["onset_day"].size()
    eligible = counts[counts >= int(min_onset_years)].index
    onsets = onsets[onsets["id"].isin(eligible)].copy()
    if onsets.empty:
        raise ValueError(
            "No units have enough observed onset years for a climatology "
            f"(min_onset_years={min_onset_years}, years {window[0]}:{window[1]})"
        )
    return onsets, len(eligible)


def _season_date(year: int, month_day: str, season_start: str) -> pd.Timestamp:
    """Return the date of a month-day in the season of ``year``: the next calendar year when it is before the start.

    Parameters
    ----------
    year : int
        The season's year (that of its start).
    month_day : str
        The month-day, ``"MM-DD"``.
    season_start : str
        The season start, ``"MM-DD"``.

    Returns
    -------
    pandas.Timestamp
        The date, ``NaT`` when it is not a month-day.
    """
    after = isinstance(month_day, str) and isinstance(season_start, str) and month_day < season_start
    return pd.Timestamp(f"{year + int(after)}-{month_day}")


def issue_grid(test: tuple[int, int], season_start: str, issue_end: str, forecast_window, horizons) -> pd.DataFrame:
    """List every issue date of the test years, with its day of the season (``d0``) and horizon (``window``).

    The issue dates run from ``season_start`` to ``issue_end`` (``"MM-DD"``) of each test year's season; ``d0`` counts days
    since the season start. A month-day before ``season_start`` falls in the next calendar year, so a season may cross
    New Year (start ``"10-01"``, end ``"01-31"``); ``year`` is the season's year. ``horizons`` is None (every date has ``forecast_window``) or a list of dicts with the keys ``start_md``,
    ``end_md`` and ``forecast_window`` (a date has the first horizon that contains it; -1 outside every one).

    Parameters
    ----------
    test : tuple of (int, int)
        The first and last test year.
    season_start : str
        The first issue date of each year, ``"MM-DD"``.
    issue_end : str
        The last issue date of each year, ``"MM-DD"``.
    forecast_window : int
        The horizon of every date when ``horizons`` is None.
    horizons : list of dict or None
        The horizons by issue period.

    Returns
    -------
    pandas.DataFrame
        One row per issue date with the columns ``time`` (date), ``year``, ``d0`` and ``window`` (int).

    Raises
    ------
    ValueError
        The season dates are not ``"MM-DD"``.
    """
    times, years, d0, windows = [], [], [], []
    for year in range(int(test[0]), int(test[1]) + 1):
        first, last = pd.Timestamp(f"{year}-{season_start}"), _season_date(year, issue_end, season_start)
        if pd.isna(first) or pd.isna(last):
            raise ValueError(f"Bad season dates; expected 'MM-DD'. Got: {season_start} / {issue_end}")
        days = pd.date_range(first, last, freq="D")
        times.append(days.date)
        years.append(np.full(len(days), year))
        d0.append((days - first).days.to_numpy())
        if horizons is None:
            windows.append(np.full(len(days), int(forecast_window)))
            continue
        window = np.full(len(days), -1)
        unset = np.ones(len(days), dtype=bool)
        for h in horizons:
            inside = unset & (days >= _season_date(year, h["start_md"], season_start)) & (
                days <= _season_date(year, h["end_md"], season_start))
            window[inside] = int(h["forecast_window"])
            unset &= ~inside
        windows.append(window)
    return pd.DataFrame({
        "time": np.concatenate(times),
        "year": np.concatenate(years).astype(int),
        "d0": np.concatenate(d0).astype(int),
        "window": np.concatenate(windows).astype(int),
    })


def plan_issues(test: tuple[int, int], season_start: str, issue_end: str, forecast_window, horizons,
                cv_by_year: bool) -> tuple[pd.DataFrame, bool, int]:
    """Plan the issues: the test years' issue grid, whether the probabilities vary by year, and the longest horizon.

    They vary by year when the KDE is refitted without each test year, or when a season day has different horizons in
    different years. ``horizons`` as for `issue_grid`; without it ``forecast_window`` is every date's horizon.

    `ValueError` is raised by a called function when the season dates are bad (see `issue_grid`).

    Parameters
    ----------
    test : tuple of (int, int)
        The first and last test year.
    season_start : str
        The first issue date of each year, ``"MM-DD"``.
    issue_end : str
        The last issue date of each year, ``"MM-DD"``.
    forecast_window : int
        The horizon of every date when ``horizons`` is empty.
    horizons : list of dict or None
        The horizons by issue period (see `issue_grid`).
    cv_by_year : bool
        Whether the KDE is refitted without each test year.

    Returns
    -------
    grid : pandas.DataFrame
        The issue grid, as `issue_grid` returns it.
    by_year : bool
        Whether the probabilities vary by year.
    max_h : int
        The longest horizon.
    """
    grid = issue_grid(test, season_start, issue_end, None if horizons else forecast_window, horizons or None)
    max_h = max(h["forecast_window"] for h in horizons) if horizons else int(forecast_window)
    by_year = cv_by_year or bool((grid.groupby("d0")["window"].nunique() > 1).any())
    return grid, by_year, max_h


def probabilities_for_issues(onsets: pd.DataFrame, grid: pd.DataFrame, by_year: bool, max_h: int, cv_by_year: bool,
                             workers: int = 1) -> xr.Dataset:
    """Compute the conditional and unconditional probabilities over (unit, [year,] season_day, lead).

    ``onsets`` has ``id``, ``year`` and ``onset_day`` (days on the same axis as the grid's ``d0``, e.g. since the season
    start); the units keep their order of appearance. Probabilities are computed once per distinct (season day,
    horizon), or per (year, season day, horizon) with ``by_year``; ``cv_by_year`` refits each unit's KDE without the key
    year. Conditional has leads 1 to ``max_h`` (NaN at lead 0), unconditional 0 to ``max_h``.

    Parameters
    ----------
    onsets : pandas.DataFrame
        The training onsets of every unit.
    grid : pandas.DataFrame
        The issue grid (`issue_grid`).
    by_year : bool
        Compute per year as well.
    max_h : int
        The longest horizon.
    cv_by_year : bool
        Refit each unit's KDE without the key year.
    workers : int, default 1
        The number of pool workers.

    Returns
    -------
    xarray.Dataset
        The variables ``conditional`` and ``unconditional`` (float64) over ``(unit, [year,] season_day, lead)``.
    """
    columns = ["year", "d0", "window"] if by_year else ["d0", "window"]
    keys = grid[columns].drop_duplicates().reset_index(drop=True)
    if not by_year:
        keys["year"] = 0
    codes, units = pd.factorize(onsets["id"])   # integer unit codes in order of appearance
    order = np.argsort(codes, kind="stable")    # rows grouped by unit, each unit's rows in their original order
    edges = np.searchsorted(codes[order], np.arange(len(units) + 1))
    all_years = onsets["year"].to_numpy(dtype=int)[order]
    all_days = onsets["onset_day"].to_numpy(dtype=float)[order]
    conditional, unconditional = unit_probabilities(
        [all_years[a:b] for a, b in zip(edges[:-1], edges[1:])],
        [all_days[a:b] for a, b in zip(edges[:-1], edges[1:])], keys, max_h, cv=cv_by_year, workers=workers)
    units = np.asarray(units)

    season_day = np.unique(keys["d0"].to_numpy())
    day = np.searchsorted(season_day, keys["d0"].to_numpy())
    leads = np.arange(max_h + 1)
    if by_year:
        years = np.unique(keys["year"].to_numpy())
        year = np.searchsorted(years, keys["year"].to_numpy())
        shape = (len(units), len(years), len(season_day), len(leads))
        index = (slice(None), year, day)
        dims, coords = ("unit", "year", "season_day", "lead"), {"year": years}
    else:
        shape = (len(units), len(season_day), len(leads))
        index = (slice(None), day)
        dims, coords = ("unit", "season_day", "lead"), {}
    dense_conditional = np.full(shape, np.nan)
    dense_unconditional = np.full(shape, np.nan)
    dense_conditional[index + (slice(1, None),)] = conditional
    dense_unconditional[index] = unconditional
    return xr.Dataset(
        {"conditional": (dims, dense_conditional), "unconditional": (dims, dense_unconditional)},
        coords={"unit": np.asarray(units, dtype=object), "season_day": season_day, "lead": leads, **coords},
    )


@dataclass
class Climatology:
    """Onset probabilities by unit, season day and lead day."""

    data: xr.Dataset | None
    """``conditional`` (leads 1 to H, NaN at lead 0) and ``unconditional`` (leads 0 to H) over ``(unit, season_day,
    lead)``, with a ``year`` dimension when the KDE is refitted without each test year (or a season day's horizon
    differs by year); its ``issue`` coordinate lists the test years' issue dates with their year and season day. None
    after `release`."""
    train_years: tuple[int, int]
    """The first and last year of the training window."""
    test_years: tuple[int, int]
    """The first and last year of the test window."""
    eligible_units: int
    """The number of units with enough onset years in the training window."""
    windows: dict[tuple[int, int], xr.DataArray] = field(default_factory=dict)
    """Conditional probabilities trained on other year windows, each over its own units, by window."""

    @property
    def by_year(self) -> bool:
        """Whether the probabilities have a ``year`` dimension."""
        return "year" in self.data["conditional"].dims

    def release(self) -> None:
        """Drop the arrays, keeping the year windows."""
        self.data = None
        self.windows = {}

    def at(self, name: str, unit, year, season_day, leads: slice | None = None) -> np.ndarray:
        """Return the probabilities (row, lead) for rows given as unit positions, years and season days.

        Parameters
        ----------
        name : {"conditional", "unconditional"}
            The variable of `data`.
        unit : array_like of int
            The position of each row's unit.
        year : array_like of int
            Each row's year (ignored when the data has no year dimension).
        season_day : array_like of int
            Each row's season day.
        leads : slice, optional
            The lead positions to keep; all when None.

        Returns
        -------
        numpy.ndarray of float64, shape (row, lead)
            The probabilities.
        """
        array = self.data[name] if leads is None else self.data[name].isel(lead=leads)
        return _gather(array, unit, year, season_day)


def _gather(array: xr.DataArray, unit, year, season_day) -> np.ndarray:
    """Gather the values of `array` at rows given as unit positions, years and season days.

    Parameters
    ----------
    array : xarray.DataArray
        Probabilities over ``(unit, [year,] season_day, lead)``.
    unit : array_like of int
        The position of each row's unit.
    year : array_like of int
        Each row's year (ignored without a ``year`` dimension).
    season_day : array_like of int
        Each row's season day.

    Returns
    -------
    numpy.ndarray of float64, shape (row, lead)
        The values.
    """
    day = np.searchsorted(array["season_day"].values, season_day)
    if "year" in array.dims:
        return array.values[unit, np.searchsorted(array["year"].values, year), day]
    return array.values[unit, day]


def build_climatology(onsets: pd.DataFrame, grid: pd.DataFrame, by_year: bool, max_h: int, *,
                      train: tuple[int, int], test: tuple[int, int], min_onset_years: int, cv_by_year: bool,
                      windows: Sequence[tuple[int, int]] = (), workers: int = 1) -> Climatology:
    """Build the climatologies for the issue dates of the grid: the training window's, and one per extra year window.

    ``onsets`` are all the observed onsets (``id``, ``year``, ``onset_day``); each window keeps the units with at least
    ``min_onset_years`` onsets in its years (`eligible_onsets`). The grid, ``by_year`` and ``max_h`` come from `plan_issues`.

    `ValueError` is raised by a called function when no unit has enough onset years in a window (see `eligible_onsets`).

    Parameters
    ----------
    onsets : pandas.DataFrame
        All the observed onsets.
    grid : pandas.DataFrame
        The issue grid.
    by_year : bool
        Whether the probabilities vary by year.
    max_h : int
        The longest horizon.
    train : tuple of (int, int)
        The first and last training year.
    test : tuple of (int, int)
        The first and last test year.
    min_onset_years : int
        The fewest onsets a unit needs in a window.
    cv_by_year : bool
        Refit each unit's KDE without the key year.
    windows : sequence of tuple of (int, int), default ()
        Extra year windows to train conditional probabilities on.
    workers : int, default 1
        The number of pool workers.

    Returns
    -------
    Climatology
        The probabilities of the training window and of each extra window.
    """
    train_onsets, eligible = eligible_onsets(onsets, train, min_onset_years)
    data = probabilities_for_issues(train_onsets, grid, by_year, max_h, cv_by_year, workers)
    data = data.assign_coords(
        issue=pd.to_datetime(grid["time"]).to_numpy(),
        issue_year=("issue", grid["year"].to_numpy(dtype=np.int64)),
        issue_season_day=("issue", grid["d0"].to_numpy(dtype=np.int64)),
    )
    data.attrs.update(cv_by_year=int(cv_by_year))
    by_window = {}
    for window in windows:
        window_onsets, _ = eligible_onsets(onsets, window, min_onset_years)
        by_window[window] = probabilities_for_issues(window_onsets, grid, by_year, max_h, cv_by_year, workers)[
            "conditional"]
    return Climatology(data=data, train_years=train, test_years=test, eligible_units=eligible, windows=by_window)
