"""Conditional climatology as a reference forecast: onset probabilities given no onset yet at the issue date.

HCWF scores only inits before the observed onset, so the fair climatological
reference is P(onset on lead k | no onset by the issue date), not the
unconditional distribution of onset dates. The probabilities come from a KDE of
the cell's observed onsets over the climatology years (hcwf.references.kde),
optionally refitted without the evaluated year (climatology_cv).

Probabilistic use gives pairs (PairArrays) of the shape the climatology scoring
consumes (per init, cell and day bin, with Before and After bins); the
probabilities are exact, so the pairs carry infinite member counts and the
fair-score corrections vanish. Deterministic use gives the conditional median
onset date per init and cell, in the layout of compute_climatology_as_forecast.

`conditional_pairs` takes the typed inputs of hcwf.config.model (run, case, selection).
"""

import numpy as np
import pandas as pd

from hcwf.config.model import Selection
from hcwf.metrics.bins import LeadBins, PairArrays, bin_fields
from hcwf.references.kde import conditional_probabilities, season_days
from hcwf.spatial import axes
import logging

log = logging.getLogger(__name__)

CONDITIONAL_CLIMATOLOGY = "conditional_climatology"
CLIMATOLOGY_REFERENCES = ("climatology", CONDITIONAL_CLIMATOLOGY)
CV_MODES = ("none", "leave_one_year_out")
DEFAULT_CV = "leave_one_year_out"
_GRID_TOLERANCE = 1e-3


def is_conditional_climatology(name) -> bool:
    """Whether name is the conditional climatology reference."""
    return name == CONDITIONAL_CLIMATOLOGY


def cv_mode(climatology_cv) -> str:
    """The climatology_cv setting (default leave_one_year_out), checked against CV_MODES."""
    mode = climatology_cv or DEFAULT_CV
    if mode not in CV_MODES:
        raise ValueError(f"climatology_cv must be one of {CV_MODES}, got {mode!r}")
    return mode


def season_day(dates) -> np.ndarray:
    """Days since 1 January of each date's year, without leap-day shifts; NaN for NaT.

    From 1 March on a leap year's dates count one less, so a calendar date has
    the same number every year and the KDE's day axis is not blurred by 29 February.
    (hcwf.references.kde.season_days with leap_shift: the one day counter, with onset_blending's start.)
    """
    return season_days(dates, (1, 1), leap_shift=True)


def _conditional_by_case(clim_onset, target_year, cells, issue_dates, horizon, cv):
    """Conditional lead probabilities (case, issue, lead) of cells given by grid indices, NaN where no KDE fits.

    clim_onset is the observed onset dates (year, lat, lon) of the climatology years (or (year, unit)); `cells` is
    a tuple of index arrays into its spatial axes (lat and lon, or unit), once per requested cell.
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
    """The climatology onsets on the spatial axes of `like` (nearest cells within tolerance, or units by id; others NaT)."""
    if "unit" in like.dims:
        return clim_onset.reindex(unit=like["unit"].values)
    return clim_onset.reindex(lat=like["lat"].values, lon=like["lon"].values, method="nearest",
                              tolerance=_GRID_TOLERANCE)


def _cases(inits, obs):
    """Index arrays (init, *spatial) of the cells with an observed onset and the inits before it, init-major."""
    has_obs = ~np.isnat(obs)
    return np.nonzero(has_obs[None] & (axes.broadcast_rows(inits.values.astype("datetime64[ns]"), obs.ndim + 1) < obs[None]))


def conditional_lead_probabilities(clim_onset, year, init_dates, like, horizon, climatology_cv=None):
    """Conditional onset probabilities of every init and cell, shaped (init, lat, lon, lead) (or (init, unit, lead))
    for leads 1..horizon.

    The same probabilities the scoring functions of this module use (the KDE of clim_onset refitted without
    `year` when climatology_cv is leave_one_year_out); NaN for cells without a KDE. `like` is a DataArray over the
    spatial axes of the observed onsets (clim_onset is aligned to them as the scoring does) and init_dates the issue
    dates.
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
    """Forecast-observation pairs (PairArrays) of the conditional climatology for one year, per init, cell and day bin.

    clim_onset is the (year, lat, lon) observed onset dates of the climatology
    years. observed_onset_da is the target year's observed onsets (lat, lon);
    by default the target year's slice of clim_onset. A case is a cell with an
    observed onset and an init before it, as for the other forecasts. Its
    probabilities are the conditional KDE probabilities of the leads in each
    bin; the Before bin holds the leads before the first bin (0 when the bins
    start at lead 1) and the After bin the rest of the mass. Cases whose cell
    has no KDE (too few onset years) are left out. Cases are ordered by init,
    lat and lon; their member counts are infinite.
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
    init_case, *where = _cases(inits, obs)
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


def conditional_pairs(clim_onset, rc, case, select=Selection(), *, years, day_bins):
    """Conditional climatology pairs (PairArrays) of all years in years, concatenated.

    A year's observed onsets come from clim_onset when it holds that year,
    otherwise they are detected from the year's rainfall as for the forecasts.
    The run `rc` carries the init dates, the onset definition and climatology_cv.
    """
    from hcwf.io.forecasts import initialization_dates
    from hcwf.io.thresholds import thresholds_of

    thresholds = thresholds_of(rc, case.model, select)
    cv = cv_mode(rc.reference.climatology_cv)
    tables = []
    for year in years:
        log.debug(f"Conditional climatology pairs for target year {year}")
        observed = None if year in clim_onset.year.values else _detect_observed(year, rc, select, thresholds)
        table = create_conditional_climatological_forecast_obs_pairs(
            clim_onset, year, initialization_dates(year, rc.evaluation), observed, day_bins=day_bins,
            climatology_cv=cv)
        if table.n_cases:
            tables.append(table)
    if not tables:
        raise ValueError("No data was successfully processed for any target year")
    return PairArrays.concat(tables)


def _detect_observed(year, rc, select, thresholds):
    """The observed onsets (lat, lon) of a year outside the climatology years."""
    from hcwf.io.observations import observations_of
    from hcwf.onset.detect import detect_observed

    return detect_observed(observations_of(year, rc, select), thresholds, year, rc, select)


def compute_conditional_climatology_as_forecast(clim_onset, year, init_dates, observed_onset_da, *,
                                                max_forecast_day, climatology_cv=None):
    """The conditional median onset as a deterministic forecast, one row per (init, lat, lon), as compute_climatology_as_forecast.

    For each init before a cell's observed onset, the forecast onset is the
    first lead (1..max_forecast_day) where the conditional CDF, the running
    sum of the conditional probabilities, reaches 0.5. There is no forecast
    onset (NaN day, no date) when it does not within max_forecast_day, when the
    cell has no KDE.
    """
    cv = cv_mode(climatology_cv) == "leave_one_year_out"
    spatial = axes.spatial_dims(observed_onset_da.dims)
    coords = axes.coords_of(observed_onset_da, spatial)
    clim = _align(clim_onset, observed_onset_da)
    obs = observed_onset_da.transpose(*spatial).values.astype("datetime64[ns]")
    inits = pd.DatetimeIndex([pd.to_datetime(t) for t in init_dates])
    init_ns = inits.values.astype("datetime64[ns]")
    init_case, *where = _cases(inits, obs)
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
