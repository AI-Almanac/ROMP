"""Conditional climatology as a reference forecast: onset probabilities given no onset yet at the issue date.

ROMP scores only inits before the observed onset, so the fair climatological
reference is P(onset on lead k | no onset by the issue date), not the
unconditional distribution of onset dates. The probabilities come from a KDE of
the cell's observed onsets over the climatology years (momp.onset.climatology_kde),
optionally refitted without the evaluated year (climatology_cv).

Probabilistic use gives a pair table of the shape the climatology scoring
consumes (per init, cell and day bin, with Before and After bins); the
probabilities are exact, so the table carries infinite member counts and the
fair-score corrections vanish. Deterministic use gives the conditional median
onset date per init and cell, in the layout of compute_climatology_as_forecast.
"""

import numpy as np
import pandas as pd

from momp.onset.climatology_kde import conditional_probabilities
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


def climatology_cv_mode(kwargs) -> str:
    """The climatology_cv setting of a config dict (default leave_one_year_out), checked against CV_MODES."""
    mode = kwargs.get("climatology_cv") or DEFAULT_CV
    if mode not in CV_MODES:
        raise ValueError(f"climatology_cv must be one of {CV_MODES}, got {mode!r}")
    return mode


def season_day(dates) -> np.ndarray:
    """Days since 1 January of each date's year, without leap-day shifts; NaN for NaT.

    From 1 March on a leap year's dates count one less, so a calendar date has
    the same number every year and the KDE's day axis is not blurred by 29 February.
    """
    index = pd.DatetimeIndex(np.asarray(dates).ravel())
    day = index.dayofyear.to_numpy() - 1
    leap_shift = index.is_leap_year & (index.month >= 3)
    return (day - leap_shift).astype(float)


def _conditional_by_case(clim_onset, target_year, cell_lat, cell_lon, issue_dates, horizon, cv):
    """Conditional lead probabilities (case, issue, lead) of cells given by grid indices, NaN where no KDE fits.

    clim_onset is the observed onset dates (year, lat, lon) of the climatology
    years; cell_lat and cell_lon index its grid, once per requested cell.
    """
    years = clim_onset.year.values.astype(int)
    dates = clim_onset.transpose("year", "lat", "lon").values.astype("datetime64[ns]")
    days = season_day(dates).reshape(dates.shape)
    keys = sorted(set(zip(cell_lat.tolist(), cell_lon.tolist())))
    onset_by_cell = {key: pd.Series(days[:, key[0], key[1]], index=years) for key in keys}
    issue = season_day(issue_dates)
    unique_issue, back = np.unique(issue, return_inverse=True)
    probs = conditional_probabilities(onset_by_cell, unique_issue, horizon,
                                      cv_years=[int(target_year)] if cv else None)
    values = probs.values[:, 0] if cv else probs.values
    slot = {key: n for n, key in enumerate(keys)}
    position = np.array([slot[key] for key in zip(cell_lat.tolist(), cell_lon.tolist())], dtype=int)
    return values, position, back


def _align(clim_onset, lat, lon):
    """The climatology onsets on the given lat/lon grid (nearest cells within tolerance; others NaT)."""
    return clim_onset.reindex(lat=lat, lon=lon, method="nearest", tolerance=_GRID_TOLERANCE)


def _cases(inits, obs):
    """Index arrays (init, lat, lon) of the cells with an observed onset and the inits before it, init-major."""
    has_obs = ~np.isnat(obs)
    return np.nonzero(has_obs[None] & (inits.values.astype("datetime64[ns]")[:, None, None] < obs[None]))


def conditional_lead_probabilities(clim_onset, year, init_dates, lats, lons, horizon, climatology_cv=None):
    """Conditional onset probabilities of every init and cell, shaped (init, lat, lon, lead) for leads 1..horizon.

    The same probabilities the scoring functions of this module use (the KDE of clim_onset refitted without
    `year` when climatology_cv is leave_one_year_out); NaN for cells without a KDE. `lats` and `lons` are the
    grid of the observed onsets (clim_onset is aligned to it as the scoring does) and init_dates the issue dates.
    """
    cv = climatology_cv_mode({"climatology_cv": climatology_cv}) == "leave_one_year_out"
    clim = _align(clim_onset, lats, lons)
    inits = pd.DatetimeIndex([pd.to_datetime(t) for t in init_dates])
    lat_cell, lon_cell = (a.ravel() for a in np.indices((len(lats), len(lons))))
    values, position, back = _conditional_by_case(clim, year, lat_cell, lon_cell, inits, horizon, cv)
    prob = values[position][:, back, :]
    return prob.reshape(len(lats), len(lons), len(inits), horizon).transpose(2, 0, 1, 3)


def create_conditional_climatological_forecast_obs_pairs(clim_onset, target_year, init_dates, observed_onset_da=None, *,
                                                         day_bins, climatology_cv=None, **kwargs):
    """Forecast-observation pairs of the conditional climatology for one year, per init, cell and day bin.

    clim_onset is the (year, lat, lon) observed onset dates of the climatology
    years. observed_onset_da is the target year's observed onsets (lat, lon);
    by default the target year's slice of clim_onset. A case is a cell with an
    observed onset and an init before it, as for the other forecasts. Its
    probabilities are the conditional KDE probabilities of the leads in each
    bin; the Before bin holds the leads before the first bin (0 when the bins
    start at lead 1) and the After bin the rest of the mass. Cases whose cell
    has no KDE (too few onset years) are left out. Rows are ordered by init,
    lat, lon and bin, with the columns of the climatology pair table.
    """
    cv = climatology_cv_mode({"climatology_cv": climatology_cv}) == "leave_one_year_out"
    max_day, min_day = max(day_bins)[1], min(day_bins)[0]
    extended_bins = ((-float("inf"), min_day - 1),) + tuple(day_bins) + ((max_day + 1, float("inf")),)
    obs_da = clim_onset.sel(year=target_year) if observed_onset_da is None else observed_onset_da
    lats, lons = obs_da.lat.values, obs_da.lon.values
    clim = _align(clim_onset, lats, lons)
    obs = obs_da.transpose("lat", "lon").values.astype("datetime64[ns]")
    inits = pd.DatetimeIndex([pd.to_datetime(t) for t in init_dates])
    init_case, lat_case, lon_case = _cases(inits, obs)
    if not len(init_case):
        return pd.DataFrame([])

    values, position, back = _conditional_by_case(clim, target_year, lat_case, lon_case, inits, max_day, cv)
    prob = values[position, back[init_case], :]
    kept = np.flatnonzero(~np.isnan(prob).any(axis=1))
    if not len(kept):
        return pd.DataFrame([])
    prob = prob[kept]
    init_case, lat_case, lon_case = init_case[kept], lat_case[kept], lon_case[kept]

    obs_days = ((obs[lat_case, lon_case] - inits.values.astype("datetime64[ns]")[init_case])
                // np.timedelta64(1, "D")).astype(np.int64)
    labels, bin_prob, observed = [], [], []
    for start, end in extended_bins:
        if start == -float("inf"):
            labels.append(f"Before day {min_day}")
            bin_prob.append(prob[:, :min_day - 1].sum(axis=1))
            observed.append(obs_days <= min_day - 1)
        elif start > max_day:
            labels.append(f"After day {max_day}")
            bin_prob.append(np.maximum(1.0 - prob.sum(axis=1), 0.0))
            observed.append(obs_days > max_day)
        else:
            labels.append(f"Days {start}-{end}")
            bin_prob.append(prob[:, start - 1:end].sum(axis=1))
            observed.append((start <= obs_days) & (obs_days <= end))
    n_bins = len(extended_bins)
    n_case = len(init_case)

    def per_case(values):
        """Repeat each per-case value once for every bin."""
        return np.repeat(values, n_bins)

    return pd.DataFrame({
        "init_time": per_case(inits.strftime("%Y-%m-%d").to_numpy(dtype=object)[init_case]),
        "lat": per_case(lats[lat_case]),
        "lon": per_case(lons[lon_case]),
        "bin_start": np.tile(np.array([b[0] for b in extended_bins], dtype=float), n_case),
        "bin_end": np.tile(np.array([b[1] for b in extended_bins], dtype=float), n_case),
        "bin_label": np.tile(np.array(labels, dtype=object), n_case),
        "predicted_prob": np.stack(bin_prob, axis=1).ravel(),
        "observed_onset": np.stack(observed, axis=1).astype(np.int64).ravel(),
        "total_members": np.inf,
        "total_members_with_onset": np.inf,
        "year": target_year,
        "obs_onset_date": per_case(pd.DatetimeIndex(obs[lat_case, lon_case]).strftime("%Y-%m-%d").to_numpy(dtype=object)),
        "obs_days_from_init": per_case(obs_days),
        "bin_index": np.tile(np.arange(n_bins), n_case),
        "forecast_type": "conditional_climatological",
    })


def multi_year_conditional_climatological_forecast_obs_pairs(clim_onset, *, years, day_bins, **kwargs):
    """Conditional climatology pair tables of all years in years, concatenated.

    A year's observed onsets come from clim_onset when it holds that year,
    otherwise they are detected from the year's rainfall as for the forecasts.
    kwargs carries the run configuration (init dates, onset definition,
    climatology_cv).
    """
    from momp.io.input import get_initialization_dates

    tables = []
    for year in years:
        log.debug(f"Conditional climatology pairs for target year {year}")
        observed = None if year in clim_onset.year.values else _detect_observed_onset(year, kwargs)
        table = create_conditional_climatological_forecast_obs_pairs(
            clim_onset, year, get_initialization_dates(year, **kwargs), observed, day_bins=day_bins, **_cv(kwargs))
        if len(table):
            tables.append(table)
    if not tables:
        raise ValueError("No data was successfully processed for any target year")
    return pd.concat(tables, ignore_index=True)


def _cv(kwargs) -> dict:
    """The climatology_cv entry of a config dict, as keyword arguments."""
    return {"climatology_cv": climatology_cv_mode(kwargs)}


def _detect_observed_onset(year, kwargs):
    """The observed onsets (lat, lon) of a year outside the climatology years."""
    from momp.io.input import load_imd_rainfall, load_thresholds
    from momp.stats.detect import detect_observed_onset

    return detect_observed_onset(load_imd_rainfall(year, **kwargs), load_thresholds(**kwargs), year, **kwargs)


def compute_conditional_climatology_as_forecast(clim_onset, year, init_dates, observed_onset_da, *,
                                                max_forecast_day, earliest_onset_date, climatology_cv=None, **kwargs):
    """The conditional median onset as a deterministic forecast, one row per (init, lat, lon), as compute_climatology_as_forecast.

    For each init before a cell's observed onset, the forecast onset is the
    first lead (1..max_forecast_day) where the conditional CDF, the running
    sum of the conditional probabilities, reaches 0.5. There is no forecast
    onset (NaN day, no date) when it does not within max_forecast_day, when the
    cell has no KDE, or when the date is before earliest_onset_date (a
    (month, day) tuple; falsy for no filter).
    """
    cv = climatology_cv_mode({"climatology_cv": climatology_cv}) == "leave_one_year_out"
    lats, lons = observed_onset_da.lat.values, observed_onset_da.lon.values
    clim = _align(clim_onset, lats, lons)
    obs = observed_onset_da.transpose("lat", "lon").values.astype("datetime64[ns]")
    inits = pd.DatetimeIndex([pd.to_datetime(t) for t in init_dates])
    init_ns = inits.values.astype("datetime64[ns]")
    init_case, lat_case, lon_case = _cases(inits, obs)
    if not len(init_case):
        return pd.DataFrame([])

    values, position, back = _conditional_by_case(clim, year, lat_case, lon_case, inits, max_forecast_day, cv)
    cdf = np.cumsum(values[position, back[init_case], :], axis=1)
    reached = np.nan_to_num(cdf, nan=0.0) >= 0.5
    lead = np.argmax(reached, axis=1) + 1
    has_onset = reached.any(axis=1)
    init_at = init_ns[init_case]
    onset_at = init_at + lead.astype("timedelta64[D]").astype("timedelta64[ns]")
    if earliest_onset_date:
        init_year = init_at.astype("datetime64[Y]")
        earliest = ((init_year.astype("datetime64[M]") + np.timedelta64(earliest_onset_date[0] - 1, "M")
                     ).astype("datetime64[D]") + np.timedelta64(earliest_onset_date[1] - 1, "D"))
        has_onset &= onset_at.astype("datetime64[D]") >= earliest

    def dates(values):
        """Format datetime values as an object array of YYYY-MM-DD strings."""
        return pd.DatetimeIndex(values).strftime("%Y-%m-%d").to_numpy(dtype=object)

    return pd.DataFrame({
        "init_time": [init_dates[n] for n in init_case],
        "lat": lats[lat_case],
        "lon": lons[lon_case],
        "onset_day": np.where(has_onset, lead, np.nan),
        "onset_date": np.where(has_onset, dates(onset_at), None),
        "obs_onset_date": dates(obs[lat_case, lon_case]),
    })
