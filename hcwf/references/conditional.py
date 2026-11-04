"""Conditional climatology as a reference forecast: onset probabilities given no onset yet at the issue date.

HCWF scores only inits before the observed onset, so the fair climatological
reference is P(onset on lead k | no onset by the issue date), not the
unconditional distribution of onset dates. The probabilities come from a KDE of
the cell's observed onsets over the climatology years (`hcwf.references.kde`),
optionally refitted without the evaluated year (``climatology_cv``).

Probabilistic use gives pairs (`hcwf.metrics.bins.PairArrays`) of the shape the climatology scoring
consumes (per init, cell and day bin, with Before and After bins); the
probabilities are exact, so the pairs carry infinite member counts and the
fair-score corrections vanish. Deterministic use gives the conditional median
onset date per init and cell, in the layout of
`hcwf.references.climatology.compute_climatology_as_forecast`.

`conditional_pairs` takes the typed inputs of `hcwf.config.model` (run, case, selection).
"""

import numpy as np
import pandas as pd

from hcwf.config.model import CONDITIONAL_CLIMATOLOGY, Selection
from hcwf.metrics.bins import LeadBins, PairArrays, bin_fields, onset_cases
from hcwf.references.kde import conditional_probabilities, season_days
from hcwf.spatial import axes
import logging

log = logging.getLogger(__name__)

CV_MODES = ("none", "leave_one_year_out")
"""The values of ``climatology_cv`` (both climatology references: the conditional and the unconditional)."""
DEFAULT_CV = "leave_one_year_out"
"""The ``climatology_cv`` used when none is given."""
_GRID_TOLERANCE = 1e-3
"""How far (in degrees) a climatology cell may be from a target cell and still be matched to it."""


def is_conditional_climatology(name) -> bool:
    """Tell whether ``name`` is the conditional climatology reference.

    Parameters
    ----------
    name : str
        A reference name.

    Returns
    -------
    bool
        True when it equals ``CONDITIONAL_CLIMATOLOGY``.
    """
    return name == CONDITIONAL_CLIMATOLOGY


def cv_mode(climatology_cv) -> str:
    """Return the ``climatology_cv`` setting (default ``"leave_one_year_out"``), checked against `CV_MODES`.

    Parameters
    ----------
    climatology_cv : str or None
        The setting; None or empty means the default.

    Returns
    -------
    str
        The mode, one of `CV_MODES`.

    Raises
    ------
    ValueError
        The mode is not one of `CV_MODES`.
    """
    mode = climatology_cv or DEFAULT_CV
    if mode not in CV_MODES:
        raise ValueError(f"climatology_cv must be one of {CV_MODES}, got {mode!r}")
    return mode


def season_day(dates) -> np.ndarray:
    """Compute the days since 1 January of each date's year, without leap-day shifts; NaN for NaT.

    From 1 March on a leap year's dates count one less, so a calendar date has
    the same number every year and the KDE's day axis is not blurred by 29 February.
    (`hcwf.references.kde.season_days` with ``leap_shift``: the one day counter, with onset_blending's start.)

    Parameters
    ----------
    dates : numpy.ndarray of datetime64
        The dates, any shape.

    Returns
    -------
    numpy.ndarray of float64
        The day counts, in the shape of ``dates``.
    """
    return season_days(dates, (1, 1), leap_shift=True)


def _conditional_by_case(clim_onset, target_year, cells, issue_dates, horizon, cv):
    """Compute the conditional lead probabilities of cells given by grid indices, NaN where no KDE fits.

    Parameters
    ----------
    clim_onset : xarray.DataArray of datetime64
        The observed onset dates ``(year, lat, lon)`` of the climatology years (or ``(year, unit)``).
    target_year : int
        The evaluated year; left out of the KDE fit when ``cv`` is true.
    cells : tuple of numpy.ndarray of int
        One index array per spatial axis of ``clim_onset`` (``lat`` and ``lon``, or ``unit``), one entry per requested
        cell.
    issue_dates : pandas.DatetimeIndex
        The issue dates.
    horizon : int
        The last lead, in days; leads 1 to ``horizon`` are returned.
    cv : bool
        Refit each KDE without ``target_year``.

    Returns
    -------
    values : numpy.ndarray of float64, shape (key, unique_issue, lead)
        The probabilities of each distinct requested cell (``key``) and each distinct issue day; NaN where no KDE fits.
    position : numpy.ndarray of int64, shape (cell,)
        For each requested cell, its row in ``values``.
    back : numpy.ndarray of int64, shape (issue,)
        For each issue date, its column in ``values``.
    """
    spatial = axes.spatial_dims(clim_onset.dims)
    years = clim_onset.year.values.astype(int)
    dates = clim_onset.transpose("year", *spatial).values.astype("datetime64[ns]")
    days = season_day(dates).reshape(dates.shape)
    cell_at = list(zip(*(c.tolist() for c in cells)))
    keys = sorted(set(cell_at))
    onset_by_cell = {key: pd.Series(days[(slice(None), *key)], index=years) for key in keys}
    issue = season_day(issue_dates)
    unique_issue, back = np.unique(issue, return_inverse=True)
    probs = conditional_probabilities(onset_by_cell, unique_issue, horizon,
                                      cv_years=[int(target_year)] if cv else None)
    values = probs.values[:, 0] if cv else probs.values
    slot = {key: n for n, key in enumerate(keys)}
    position = np.array([slot[key] for key in cell_at], dtype=int)
    return values, position, back


def _align(clim_onset, like):
    """Reindex the climatology onsets onto the spatial axes of ``like`` (nearest cells within tolerance, or units by id).

    Parameters
    ----------
    clim_onset : xarray.DataArray of datetime64
        The climatology onsets, over ``(year, lat, lon)`` or ``(year, unit)``.
    like : xarray.DataArray
        An array whose ``lat`` and ``lon`` (or ``unit``) coordinates give the target axes.

    Returns
    -------
    xarray.DataArray of datetime64
        ``clim_onset`` on the axes of ``like``; ``NaT`` where there is no cell within `_GRID_TOLERANCE` (or no such unit).
    """
    if "unit" in like.dims:
        return clim_onset.reindex(unit=like["unit"].values)
    return clim_onset.reindex(lat=like["lat"].values, lon=like["lon"].values, method="nearest",
                              tolerance=_GRID_TOLERANCE)


def conditional_lead_probabilities(clim_onset, year, init_dates, like, horizon, climatology_cv=None):
    """Compute the conditional onset probabilities of every init and cell for leads 1 to ``horizon``.

    The same probabilities the scoring functions of this module use (the KDE of ``clim_onset`` refitted without
    ``year`` when ``climatology_cv`` is ``"leave_one_year_out"``); NaN for cells without a KDE. ``like`` is a DataArray over
    the spatial axes of the observed onsets (``clim_onset`` is aligned to them as the scoring does).

    `ValueError` is raised by a called function when ``climatology_cv`` is not a valid mode.

    Parameters
    ----------
    clim_onset : xarray.DataArray of datetime64
        The observed onset dates of the climatology years, ``(year, lat, lon)`` or ``(year, unit)``.
    year : int
        The evaluated year (left out of the fit with leave-one-year-out).
    init_dates : iterable of datetime-like
        The issue dates.
    like : xarray.DataArray
        An array over the spatial axes of the observed onsets.
    horizon : int
        The last lead, in days.
    climatology_cv : str, optional
        The cross-validation mode (see `cv_mode`); None means the default.

    Returns
    -------
    numpy.ndarray of float64, shape (init, lat, lon, lead) or (init, unit, lead)
        The probability of the onset on each lead 1 to ``horizon``, NaN where the cell has no KDE.
    """
    cv = cv_mode(climatology_cv) == "leave_one_year_out"
    clim = _align(clim_onset, like)
    spatial = axes.spatial_dims(like.dims)
    shape = tuple(like.sizes[d] for d in spatial)
    inits = pd.DatetimeIndex([pd.to_datetime(t) for t in init_dates])
    cells = tuple(a.ravel() for a in np.indices(shape))
    values, position, back = _conditional_by_case(clim, year, cells, inits, horizon, cv)
    prob = values[position][:, back, :]
    return prob.reshape(*shape, len(inits), horizon).transpose(len(shape), *range(len(shape)), len(shape) + 1)


def create_conditional_climatological_forecast_obs_pairs(clim_onset, target_year, init_dates, observed_onset_da=None, *,
                                                         day_bins, climatology_cv=None):
    """Make the forecast-observation pairs of the conditional climatology for one year, per init, cell and day bin.

    A case is a cell with an observed onset and an init before it, as for the other forecasts. Its
    probabilities are the conditional KDE probabilities of the leads in each
    bin; the Before bin holds the leads before the first bin (0 when the bins
    start at lead 1) and the After bin the rest of the mass. Cases whose cell
    has no KDE (too few onset years) are left out. Cases are ordered by init,
    lat and lon; their member counts are infinite.

    `ValueError` is raised by a called function when ``climatology_cv`` is not a valid mode.

    Parameters
    ----------
    clim_onset : xarray.DataArray of datetime64
        The ``(year, lat, lon)`` observed onset dates of the climatology years.
    target_year : int
        The year to evaluate.
    init_dates : iterable of datetime-like
        The initialization dates.
    observed_onset_da : xarray.DataArray of datetime64, optional
        The target year's observed onsets ``(lat, lon)``; by default the target year's slice of ``clim_onset``.
    day_bins : sequence
        The day bins, as `hcwf.metrics.bins.LeadBins.from_day_bins` takes them.
    climatology_cv : str, optional
        The cross-validation mode (see `cv_mode`); None means the default.

    Returns
    -------
    hcwf.metrics.bins.PairArrays
        The pairs, empty when there is no case.
    """
    cv = cv_mode(climatology_cv) == "leave_one_year_out"
    bins = LeadBins.from_day_bins(day_bins)
    max_day, min_day = bins.last_day, bins.first_day
    extended_bins = bins.extended
    obs_da = clim_onset.sel(year=target_year) if observed_onset_da is None else observed_onset_da
    spatial = axes.spatial_dims(obs_da.dims)
    coords = axes.coords_of(obs_da, spatial)
    clim = _align(clim_onset, obs_da)
    obs = obs_da.transpose(*spatial).values.astype("datetime64[ns]")
    inits = pd.DatetimeIndex([pd.to_datetime(t) for t in init_dates])
    init_case, *where = onset_cases(inits, obs)
    if not len(init_case):
        return PairArrays.empty(bins, spatial)

    values, position, back = _conditional_by_case(clim, target_year, tuple(where), inits, max_day, cv)
    prob = values[position, back[init_case], :]
    kept = np.flatnonzero(~np.isnan(prob).any(axis=1))
    if not len(kept):
        return PairArrays.empty(bins, spatial)
    prob = prob[kept]
    init_case, where = init_case[kept], tuple(w[kept] for w in where)

    obs_days = ((obs[where] - inits.values.astype("datetime64[ns]")[init_case])
                // np.timedelta64(1, "D")).astype(np.int64)
    bin_prob, observed = [], []
    for k, (start, end) in enumerate(extended_bins):
        if k == 0:
            bin_prob.append(prob[:, :min_day - 1].sum(axis=1))
            observed.append(obs_days <= min_day - 1)
        elif k == bins.n_inner + 1:
            bin_prob.append(np.maximum(1.0 - prob.sum(axis=1), 0.0))
            observed.append(obs_days > max_day)
        else:
            bin_prob.append(prob[:, start - 1:end].sum(axis=1))
            observed.append((start <= obs_days) & (obs_days <= end))

    return PairArrays(prob=np.stack(bin_prob, axis=1), observed=np.stack(observed, axis=1).astype(np.int8),
                      members=np.full(len(init_case), np.inf), **bin_fields(bins),
                      init=inits.values.astype("datetime64[D]")[init_case],
                      place=axes.columns_of(spatial, coords, where),
                      members_with_onset=np.full(len(init_case), np.inf))


def conditional_pairs(clim_onset, rc, case, select=Selection(), *, years, day_bins, observed=None):
    """Make the conditional climatology pairs of all ``years``, concatenated.

    A year's observed onsets (the truth) are those of ``observed`` (the run's own, which the models are scored
    against); a year missing there comes from ``clim_onset`` when it holds that year, otherwise it is detected from the
    year's rainfall as for the forecasts (`observed_of_year`).
    The run ``rc`` carries the init dates, the onset definition and ``climatology_cv``.

    Parameters
    ----------
    clim_onset : xarray.DataArray of datetime64
        The observed onset dates of the climatology years.
    rc : hcwf.config.model.RunConfig
        The run.
    case : hcwf.config.model.Case
        The case whose model the thresholds are loaded for.
    select : hcwf.config.model.Selection, default Selection()
        The call's choices.
    years : iterable of int
        The target years.
    day_bins : sequence
        The day bins.
    observed : dict of int to xarray.DataArray, optional
        The observed onsets of the run's years.

    Returns
    -------
    hcwf.metrics.bins.PairArrays
        The pairs of all years, in the order of ``years``.

    Raises
    ------
    ValueError
        No year gave a case, or the ``climatology_cv`` setting is not a valid mode.
    """
    from hcwf.io.forecasts import initialization_dates
    cv = cv_mode(rc.reference.climatology_cv)
    tables = []
    for year in years:
        log.debug(f"Conditional climatology pairs for target year {year}")
        table = create_conditional_climatological_forecast_obs_pairs(
            clim_onset, year, initialization_dates(year, rc.evaluation),
            observed_of_year(year, clim_onset, rc, case.model, select, observed), day_bins=day_bins, climatology_cv=cv)
        if table.n_cases:
            tables.append(table)
    if not tables:
        raise ValueError("No data was successfully processed for any target year")
    return PairArrays.concat(tables)


def observed_of_year(year, clim_onset, rc, model, select, observed=None):
    """Return the observed onsets of a target year, the truth of a climatology reference.

    ``observed`` holds the run's own onsets of the evaluated years, the ones the models are scored against (made once
    by `hcwf.scoring.pairs.forecast_obs_pairs`); a year missing from it is read from ``clim_onset`` when that holds the
    year (the same values), else detected like the model's (`hcwf.onset.detect.detect_observed`, which reuses the
    observed-onset cache).

    Parameters
    ----------
    year : int
        The target year.
    clim_onset : xarray.DataArray of datetime64
        The onsets of the climatology years.
    rc : hcwf.config.model.RunConfig
        The run.
    model : hcwf.config.model.ModelSpec
        The model the onset thresholds are loaded for (read only when the year has to be detected).
    select : hcwf.config.model.Selection
        The call's choices.
    observed : dict of int to xarray.DataArray, optional
        The run's observed onsets by year.

    Returns
    -------
    xarray.DataArray of datetime64
        The observed onset dates over ``(lat, lon)`` (or ``unit``).
    """
    if observed is not None and year in observed:
        return observed[year]
    if year in clim_onset.year.values:
        return clim_onset.sel(year=year)
    from hcwf.io.thresholds import thresholds_of
    return _detect_observed(year, rc, select, thresholds_of(rc, model, select))


def _detect_observed(year, rc, select, thresholds):
    """Detect the observed onsets of a year outside the climatology years.

    Parameters
    ----------
    year : int
        The year.
    rc : hcwf.config.model.RunConfig
        The run.
    select : hcwf.config.model.Selection
        The call's choices.
    thresholds : xarray.DataArray
        The onset thresholds.

    Returns
    -------
    xarray.DataArray of datetime64
        The observed onset dates over ``(lat, lon)`` (or ``unit``).
    """
    from hcwf.io.observations import observations_of
    from hcwf.onset.detect import detect_observed

    return detect_observed(observations_of(year, rc, select), thresholds, year, rc, select)


def compute_conditional_climatology_as_forecast(clim_onset, year, init_dates, observed_onset_da, *,
                                                max_forecast_day, climatology_cv=None):
    """Compute the conditional median onset as a deterministic forecast, one row per (init, lat, lon).

    The layout is that of `hcwf.references.climatology.compute_climatology_as_forecast`.
    For each init before a cell's observed onset, the forecast onset is the
    first lead (1 to ``max_forecast_day``) where the conditional CDF, the running
    sum of the conditional probabilities, reaches 0.5. There is no forecast
    onset (NaN day, no date) when it does not within ``max_forecast_day``, or when the
    cell has no KDE.

    `ValueError` is raised by a called function when ``climatology_cv`` is not a valid mode.

    Parameters
    ----------
    clim_onset : xarray.DataArray of datetime64
        The observed onset dates of the climatology years.
    year : int
        The evaluated year.
    init_dates : iterable of datetime-like
        The initialization dates.
    observed_onset_da : xarray.DataArray of datetime64
        The observed onsets of the year; they select the cases and give the grid.
    max_forecast_day : int
        The last lead, in days.
    climatology_cv : str, optional
        The cross-validation mode (see `cv_mode`); None means the default.

    Returns
    -------
    pandas.DataFrame
        One row per case (init, lat, lon) or (init, unit) with ``init_time``, the spatial columns, ``onset_day``,
        ``onset_date`` and ``obs_onset_date``. Empty when there is no case.
    """
    cv = cv_mode(climatology_cv) == "leave_one_year_out"
    spatial = axes.spatial_dims(observed_onset_da.dims)
    coords = axes.coords_of(observed_onset_da, spatial)
    clim = _align(clim_onset, observed_onset_da)
    obs = observed_onset_da.transpose(*spatial).values.astype("datetime64[ns]")
    inits = pd.DatetimeIndex([pd.to_datetime(t) for t in init_dates])
    init_ns = inits.values.astype("datetime64[ns]")
    init_case, *where = onset_cases(inits, obs)
    where = tuple(where)
    if not len(init_case):
        return pd.DataFrame([])

    values, position, back = _conditional_by_case(clim, year, where, inits, max_forecast_day, cv)
    cdf = np.cumsum(values[position, back[init_case], :], axis=1)
    reached = np.nan_to_num(cdf, nan=0.0) >= 0.5
    lead = np.argmax(reached, axis=1) + 1
    has_onset = reached.any(axis=1)
    init_at = init_ns[init_case]
    onset_at = init_at + lead.astype("timedelta64[D]").astype("timedelta64[ns]")

    return pd.DataFrame({
        "init_time": [init_dates[n] for n in init_case],
        **axes.columns_of(spatial, coords, where),
        "onset_day": np.where(has_onset, lead, np.nan),
        "onset_date": np.where(has_onset, axes.date_strings(onset_at), None),
        "obs_onset_date": axes.date_strings(obs[where]),
    })
