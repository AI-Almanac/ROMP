"""Onset detection on observed rainfall and on deterministic or ensemble model forecasts.

A thin adapter between HCWF's pipeline and hcwf.onset: it settles units, day
boundaries and thresholds, runs hcwf.onset.engine, and shapes the results into
the arrays and DataFrames the metrics code expects.

The functions take the typed inputs of hcwf.config.model: the run (`rc`: RunConfig), the case (source and window) and the
per-call `select` (Selection).
"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.config.model import Selection
from hcwf.io.input import ingest_forecast
from hcwf.lib.onset_config import forecast_reach
from hcwf.onset import engine
from hcwf.stats import observed_cache
from hcwf.onset.alignment import check_forecast_alignment
from hcwf.onset.ingest import ingest
from hcwf.onset.model import DailyRain, Status
from hcwf.onset.thresholds import unit_thresholds
from hcwf.spatial import axes
import logging

log = logging.getLogger(__name__)

_GRID_TOLERANCE = 1e-3
_ALIGNMENT_MAX_INITS = 40
_ALIGNMENT_MAX_MEMBERS = 10
_alignment_done = set()


def engine_options(rc, forecast=False) -> dict:
    """The run options of the engine a run holds (horizon and missing-rain policy); none given is HCWF's own."""
    return rc.onset_options.engine(rc.observations.units, forecast)


def _on_grid(da, lat, lon):
    """Put a per-cell DataArray (or dataset) on the lat/lon grid of the rainfall; cells it lacks become NaN."""
    if "lat" not in da.dims or "lon" not in da.dims:
        return da
    return da.reindex(lat=lat, lon=lon, method="nearest", tolerance=_GRID_TOLERANCE)


def _like(da, coords):
    """A per-cell or per-unit DataArray on the spatial axes `coords` (cells within tolerance, or units by id); the rest NaN."""
    if "unit" in coords:
        return da.reindex(unit=np.asarray(coords["unit"]).astype(str))
    return _on_grid(da, coords["lat"], coords["lon"])


def _on_units(value, ids, convention=None):
    """A threshold over the units of the rainfall, joined by unit id: a number of mm, a table of (id, onset_thresh)
    or an array over "unit"; a table must cover the units (see hcwf.onset.thresholds.unit_thresholds)."""
    return xr.DataArray(unit_thresholds(value, np.asarray(ids).astype(str), convention), dims=("unit",),
                        coords={"unit": np.asarray(ids).astype(str)})


def _field(value, coords, convention=None):
    """One threshold on the rainfall's own spatial axes (cells of the grid, or units)."""
    if "unit" in coords:
        return _on_units(value, coords["unit"], convention)
    return _on_grid(value, coords["lat"], coords["lon"]) if isinstance(value, xr.DataArray) else value


def _thresholds(setup, thresh_slice, coords, per_window=True, convention=None):
    """The thresholds for the engine: named arrays on the rainfall's cells or units, or None for a fixed amount.

    coords are the rainfall's spatial coordinates (lat and lon, or unit). Model-climatology thresholds carry a "window"
    dim (one per verification window). Forecast detection keeps it; per_window=False (observed onsets, which have no
    lead) keeps the first window.
    """
    out = {name: _field(value, coords, convention) for name, value in dict(thresh_slice or {}).items()}
    if not per_window:
        out = {name: (value.isel(window=0, drop=True) if "window" in getattr(value, "dims", ()) else value)
               for name, value in out.items()}
    return out


def _season_climatology(setup, thresh_slice, coords):
    """The harmonic season analysis on the rainfall grid, for derived seasons only."""
    if not setup.derived_seasons:
        return None
    climatology = getattr(thresh_slice, "season_climatology", None)
    if climatology is None:
        raise ValueError("this definition has derived seasons but no season climatology was loaded; "
                         "pass the thresholds returned by hcwf.io.input.load_thresholds")
    if "lat" not in coords:
        raise ValueError("derived seasons are analysed on a grid; evaluation units do not support them")
    return _on_grid(climatology, coords["lat"], coords["lon"])


def _convention(rc):
    """The grid-id convention of the run's units (None: established or inferred)."""
    spec = rc.observations.units
    return None if spec is None else spec.convention


def detect_observed_dataset(rain_slice, thresh_slice, year, rc, select=Selection()):
    """Run the engine on one year of observed rainfall; return (engine Dataset, OnsetSetup).

    The Dataset has dims (year, season, *spatial) (lat and lon, or unit) with one year and the selected season (both kept
    as length-1 dims) and the variables onset_date, onset_index and status of hcwf.onset.engine.detect_observed; its
    "calendar" attribute is the calendar of the dates. detect_observed and the NetCDF export (hcwf.export) both
    read their values from here, so they cannot disagree. A `select.reference_onset` (a ReferenceOnset, for
    evaluation units) starts each search no earlier than that date.
    """
    setup = rc.onset
    rain = rain_slice if isinstance(rain_slice, DailyRain) else ingest(
        rain_slice, day_boundary=setup.obs_day_boundary, units="mm", source="observations",
        **rc.onset_options.ingest(False, rc.observations.units))
    coords = {d: rain.coords[d] for d in rain.dims[:-1]}
    reference = select.reference_onset
    search = None if reference is None else {"reference": reference.search_start([int(year)], coords["unit"])}
    result = engine.detect_observed(
        setup.definition, rain, thresholds=_thresholds(setup, thresh_slice, coords, per_window=False,
                                                       convention=_convention(rc)), years=[year],
        season_climatology=_season_climatology(setup, thresh_slice, coords), backend=setup.backend,
        search_start=search, **engine_options(rc))
    if search:
        result = result.isel(start=1)
    result = result.sel(year=[year], season=[setup.season])
    result.attrs["calendar"] = rain.calendar
    return result, setup


def detect_observed(rain_slice, thresh_slice, year, rc, select=Selection()):
    """Detect observed onset dates for a given year.

    rain_slice is daily rainfall in mm with dims (time, lat, lon), or (time, unit) for evaluation units (the reader's
    `present` coordinate says which units have a series), or that rainfall already ingested as a DailyRain (the
    climatology years the thresholds were computed from); for units it may also be a function that reads it, called only
    when the onsets are not cached. thresh_slice is the threshold from hcwf.io.input.load_thresholds (the definition's
    named thresholds).
    Returns a (lat, lon) or (unit) DataArray of datetime64 onset dates, NaT where there is no onset. For units it carries
    the coordinates `present` and `onset_index` (the 1-based day of the onset in the year's series, NaN without one).
    While a run has the observed-onset cache active (hcwf.stats.observed_cache) the dates of a year already detected
    under the same fingerprint are returned without detecting again.
    """
    reader = rain_slice if callable(rain_slice) else None
    cached = observed_cache.lookup(None if reader else rain_slice, thresh_slice, year, rc, select)
    if cached is not None:
        return cached
    if reader is not None:
        rain_slice = reader()
    result, setup = detect_observed_dataset(rain_slice, thresh_slice, year, rc, select)
    spatial = ("lat", "lon") if "lat" in result["onset_date"].dims else tuple(result["onset_date"].dims[2:])
    onset = result["onset_date"].isel(year=0, season=0).transpose(*spatial)
    status = result["status"].isel(year=0, season=0).transpose(*spatial)
    counts = {s.name: int((status == s).sum()) for s in Status}
    log.debug(f"Observed onset {year}, cells by status: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    values = onset.values.astype("datetime64[ns]")
    out = xr.DataArray(values, coords=[(d, result[d].values) for d in spatial], name="onset_date")
    if isinstance(rain_slice, xr.DataArray) and "present" in rain_slice.coords:
        present = np.asarray(rain_slice["present"].values, dtype=bool)
        found = (status.values == int(Status.ONSET)) & present
        index = result["onset_index"].isel(year=0, season=0).transpose(*spatial).values
        out = out.copy(data=np.where(found, values, np.datetime64("NaT"))).assign_coords(
            present=(spatial, present), onset_index=(spatial, np.where(found, index + 1, np.nan)))
    observed_cache.store(rain_slice, thresh_slice, year, rc, select, out)
    return out


def detect_forecast(p_model, thresh_slice, onset_da, rc, case, select=Selection(), *, members=None, history=None,
                    search_start=None, restrict_to_season=True, onset_count=False):
    """Run the engine on a forecast cube and line it up with the observed onsets.

    Returns a dict of numpy arrays shaped (init_time, [member], lat, lon): lead day
    after init of the onset (NaN none; 0 or below for an onset back-dated by the observed history), date, status; plus the valid-case mask
    (init before the observed onset) and the observed onset dates (init_time, lat, lon). "result" is the engine
    Dataset these arrays come from (season kept as a length-1 dim, with the ensemble summary for members) and
    "calendar" the calendar of its dates; the NetCDF export (hcwf.export) reads them.

    The cube may be over evaluation units instead of lat and lon ((init_time, member, unit, step)). `search_start`
    (variants that start the search later, see hcwf.onset.engine.detect_forecast), `restrict_to_season` (False: one
    season covering every lead) and `onset_count` (the member onset counts of `count_members`) may then be given; with
    onset_count there is no observed onset to line up with (onset_da is None) and the dict holds "counts" (init_time,
    start, unit, lead) instead of the day, date and status arrays.
    """
    setup = rc.onset
    max_forecast_day = rc.evaluation.max_forecast_day
    steps_needed = max_forecast_day + forecast_reach(setup) - 1 if isinstance(max_forecast_day, int) else None
    cube = p_model.sel(step=slice(1, steps_needed))
    if "member" in cube.dims and members and list(cube["member"].values) != list(members):
        cube = cube.sel(member=list(members))  # the loader already took them; selecting again would copy the cube
    spatial = axes.spatial_dims(cube.dims)
    order = ("init_time",) + (("member",) if "member" in cube.dims else ()) + spatial + ("step",)
    forecast = ingest_forecast(cube.transpose(*order), setup, case.model, rc.onset_options, select, "forecast")
    coords = {d: forecast.coords[d] for d in spatial}
    stitched = None
    if history is not None and setup.forecast_history == "observed":
        on_grid = _like(history, coords).transpose(*spatial, ...)
        stitched = ingest(on_grid, day_boundary=setup.obs_day_boundary, units="mm", source="observed history")
    count = bool(onset_count)
    result = engine.detect_forecast(
        setup.definition, forecast, thresholds=_thresholds(setup, thresh_slice, coords, convention=_convention(rc)),
        lead_window=(1, max_forecast_day), backend=setup.backend, history=stitched,
        season_climatology=_season_climatology(setup, thresh_slice, coords),
        restrict_to_season=restrict_to_season, search_start=search_start,
        onset_count=count, **engine_options(rc, forecast=True))
    season = setup.season
    if not restrict_to_season:   # one season covering every lead, named for itself
        season = result["season"].values[0]
    export_result = result.sel(season=[season])
    result = result.sel(season=season)
    if not count and bool((result["status"] == Status.CENSORED).any()):
        warnings.warn("some forecast searches are CENSORED although the lead window was shortened to fit the rule; "
                      "they are reported as no onset. Is max_forecast_day larger than the loaded leads allow?",
                      UserWarning)
    init = np.asarray(forecast.coords["init_time"])
    if count:
        counts = result["onset_count"].transpose("init_time", "start", *spatial, "lead")
        return {"setup": setup, "result": export_result, "calendar": forecast.calendar, "init": init,
                "counts": counts, "start": [str(s) for s in counts["start"].values]}
    lead_dims = (("init_time", "member") if "member" in forecast.dims else ("init_time",)) + spatial
    dates = result["onset_date"].transpose(*lead_dims).values
    status = result["status"].transpose(*lead_dims).values
    obs = (onset_da.reindex(unit=np.asarray(coords["unit"]).astype(str)) if "unit" in coords else
           onset_da.reindex(lat=coords["lat"], lon=coords["lon"], method="nearest", tolerance=_GRID_TOLERANCE)
           ).transpose(*spatial)
    obs_values = obs.values.astype("datetime64[ns]")
    valid = ~np.isnat(obs_values)[None] & (axes.broadcast_rows(init.astype("datetime64[ns]"), obs_values.ndim + 1)
                                           < obs_values[None])
    found = status == Status.ONSET
    init_day = init.astype("datetime64[D]").reshape((-1,) + (1,) * (dates.ndim - 1))
    days_after_init = (dates.astype("datetime64[D]") - init_day) / np.timedelta64(1, "D")
    return {"setup": setup, "result": export_result, "calendar": forecast.calendar, "spatial": spatial,
            "init": init, "axes": {d: np.asarray(coords[d]) for d in spatial},
            "member": np.asarray(forecast.coords["member"]) if "member" in forecast.dims else None,
            "day": np.where(found, days_after_init, np.nan), "date": dates, "status": status,
            "valid": valid, "obs": obs_values, "totals": (int(obs.size * len(init)), int((~np.isnat(obs_values)).sum()))}


def count_members(cube, thresh_slice, rc, case, select=Selection(), *, search_start=None):
    """For an ensemble forecast on evaluation units: (counts, start names, init times).

    cube has dims (init_time, member, unit, step) in mm, its steps 1..n, with n the run's max_forecast_day. counts is an
    (init_time, start, unit, lead) DataArray: the number of members whose onset falls on each lead day 0..n, from the
    search that starts at lead 1 ("base" first) and from each variant of `search_start` (the fixed cutoff date, the
    reference onset); "start" names them in that order.
    """
    out = detect_forecast(cube, thresh_slice, None, rc, case, select, search_start=search_start, onset_count=True,
                          restrict_to_season=False)
    return out["counts"], out["start"], out["init"]


def _date_strings(values):
    """'%Y-%m-%d' text of datetime64 values, as a list."""
    return pd.DatetimeIndex(np.ravel(values)).strftime("%Y-%m-%d").tolist()


def _print_back_dated(day, valid):
    """Print how many scored forecast onsets lie on or before the init (back-dated by the observed history)."""
    mask = np.broadcast_to(valid[:, None] if day.ndim == valid.ndim + 1 else valid, day.shape)
    count = int((day[mask] <= 0).sum())
    if count:
        log.debug(f"Back-dated forecast onsets (onset_day <= 0, scored as documented in docs/onset_in_hcwf.md): {count}")


def _print_status_counts(status, valid):
    """Log how many forecast searches ended in each status."""
    mask = np.broadcast_to(valid[:, None] if status.ndim == valid.ndim + 1 else valid, status.shape)
    counts = np.bincount(status[mask].astype(np.int64), minlength=len(Status))
    log.debug("Forecast onset searches by status: "
          + ", ".join(f"{s.name} {int(counts[s])}" for s in Status if counts[s]))
    if counts[Status.PRIOR]:
        warnings.warn(f"{int(counts[Status.PRIOR])} forecast searches before the observed onset are PRIOR; the "
                      "observed history confirmed an onset that the observed onset date does not show. They are "
                      "reported as no onset.", UserWarning)


def _names(status):
    """Status names for an array of codes."""
    labels = np.array([s.name for s in Status])
    return labels[np.asarray(status, dtype=np.int64)]


def detect_deterministic(p_model, thresh_slice, onset_da, rc, case, select=Selection(), history=None):
    """Compute onset dates for deterministic model forecast.

    p_model has dims (init_time, lat, lon, step), or (init_time, unit, step) on evaluation units. Returns one row per
    (init, cell or unit) in that order for those with an observed onset and an init before it, with
    columns init_time, lat and lon (or unit), onset_day (lead day, None if no onset), onset_date
    ('YYYY-MM-DD' or None) and obs_onset_date, plus onset_status for definitions.
    """
    out = detect_forecast(p_model, thresh_slice, onset_da, rc, case, select, history=history)
    setup, valid = out["setup"], out["valid"]
    spatial = out["spatial"]
    log.debug(f"Processing {len(out['init'])} init times x {valid[0].size} {'/'.join(spatial)} points...")
    i, *where = np.nonzero(valid)
    at = (i, *where)
    day = out["day"][at]
    has = ~np.isnan(day)
    onset_date = np.where(has, _date_strings(out["date"][at]), None).tolist() if len(i) else []
    columns = {
        "init_time": out["init"][i], **axes.columns_of(spatial, out["axes"], where),
        "onset_day": [int(d) if h else None for d, h in zip(day, has)],
        "onset_date": onset_date,
        "obs_onset_date": _date_strings(out["obs"][tuple(where)]) if len(i) else [],
    }
    columns["onset_status"] = _names(out["status"][at]).tolist()
    onset_df = pd.DataFrame(columns) if len(i) else pd.DataFrame([])
    total, with_obs = out["totals"]
    log.debug(f"\nProcessing Summary:")
    log.debug(f"Total potential initializations: {total}")
    log.debug(f"Skipped (no observed onset): {total - with_obs * len(out['init'])}")
    log.debug(f"Skipped (initialized after observed onset): {with_obs * len(out['init']) - len(i)}")
    log.debug(f"Valid initializations processed: {len(i)}")
    log.debug(f"Onsets found: {int(has.sum())}")
    log.debug(f"Onset rate: {has.sum() / len(i):.3f}" if len(i) else "Onset rate: 0.000")
    _print_status_counts(out["status"], valid)
    _print_back_dated(out["day"], valid)
    return onset_df


def _ensemble_status(status, threshold, has_onset):
    """Status of an ensemble onset: ONSET if it has one, else the most informative way the members fell short."""
    names = np.full(has_onset.shape, Status.NOT_FOUND.name, dtype=object)
    total = status.shape[1]
    for code in (Status.INSUFFICIENT_DATA, Status.CENSORED):
        reach = ((status == Status.ONSET).sum(axis=1) + (status == code).sum(axis=1)) / total >= threshold
        names[reach] = code.name
    names[has_onset] = Status.ONSET.name
    return names


def ensemble_onset(day, init, threshold):
    """The ensemble onset HCWF scores from member onset days (init_time, member, lat, lon or unit), NaN for none.

    Returns (count of members with an onset, their fraction, ensemble onset day, ensemble onset date), each
    shaped (init_time, lat, lon) or (init_time, unit). The ensemble has an onset when the fraction is at least threshold; its day
    (days after the init) is the mean member day rounded half to even. init is the array of init dates.
    """
    n_member = day.shape[1]
    count = (~np.isnan(day)).sum(axis=1)
    fraction = count / n_member if n_member > 0 else np.zeros(count.shape)
    with np.errstate(invalid="ignore"):
        mean_day = np.nanmean(day, axis=1) if n_member else np.full(count.shape, np.nan)
    ensemble_day = np.where(fraction >= threshold, np.round(mean_day), np.nan)
    ensemble_date = axes.broadcast_rows(pd.DatetimeIndex(init).values, ensemble_day.ndim) + (
        np.nan_to_num(ensemble_day, nan=0).astype("int64").astype("timedelta64[D]"))
    return count, fraction, ensemble_day, ensemble_date


def detect_ensemble(p_model, thresh_slice, onset_da, rc, case, select=Selection(), history=None):
    """Compute per-member and ensemble onsets.

    p_model has dims (init_time, member, lat, lon, step), or (init_time, member, unit, step) on evaluation units. The ensemble has an onset
    when the fraction of members with one is at least onset_percentage_threshold;
    its day is the mean member onset day rounded half to even.
    Returns (member onset DataFrame, ensemble onset DataFrame).
    """
    members = rc.evaluation.members
    threshold = rc.onset_options.percentage_threshold
    out = detect_forecast(p_model, thresh_slice, onset_da, rc, case, select, members=members, history=history)
    setup, valid, day = out["setup"], out["valid"], out["day"]
    n_init, n_member = day.shape[0], day.shape[1]
    spatial = out["spatial"]
    log.debug(f"Processing {n_init} init times x {valid[0].size} unique locations x {n_member} members...")

    member_day = day
    count, fraction, ensemble_day, ensemble_date = ensemble_onset(day, out["init"], threshold)

    grid = valid.shape
    flat_valid = np.broadcast_to(valid[:, None], (grid[0], n_member) + grid[1:])
    member_rows = tuple(np.nonzero(flat_valid))
    member_index = pd.Index(np.flatnonzero(flat_valid.ravel()))
    ens_rows = tuple(np.nonzero(valid))
    ens_index = pd.Index(np.flatnonzero(valid.ravel()))
    mi, mm, *member_where = member_rows
    ei, *ens_where = ens_rows
    member_at, ens_at = (mi, mm, *member_where), (ei, *ens_where)

    member_values = member_day[member_at]
    onset_df = pd.DataFrame({
        "init_time": out["init"][mi], **axes.columns_of(spatial, out["axes"], member_where),
        "member": out["member"][mm],
        "onset_day": member_values,
        "obs_onset_date": pd.Series(pd.to_datetime(out["obs"][tuple(member_where)])).dt.strftime("%Y-%m-%d").to_numpy()
        if len(mi) else [],
    }, index=member_index)
    chosen = ensemble_day[ens_at]
    has_onset = ~np.isnan(chosen)
    ens_dates = np.where(has_onset, _date_strings(ensemble_date[ens_at]) if len(ei) else [], None)
    ensemble_df = pd.DataFrame({
        "init_time": out["init"][ei], **axes.columns_of(spatial, out["axes"], ens_where),
        "onset_day": chosen,
        "member_onset_count": count[ens_at], "total_members": n_member,
        "onset_percentage": fraction[ens_at],
        "obs_onset_date": pd.Series(pd.to_datetime(out["obs"][tuple(ens_where)])).dt.strftime("%Y-%m-%d").to_numpy()
        if len(ei) else [],
    }, index=ens_index)
    ensemble_df.insert(4 if len(spatial) == 2 else 3, "onset_date", None)
    if has_onset.any():
        ensemble_df.loc[has_onset, "onset_date"] = ens_dates[has_onset]
    onset_df["onset_status"] = _names(out["status"][member_at])
    ensemble_df["onset_status"] = _ensemble_status(out["status"][(ei, slice(None), *ens_where)], threshold, has_onset)

    log.debug(f"\nProcessing Summary:")
    log.debug(f"Generated {len(onset_df)} member-forecast combinations")
    log.debug(f"Found onset in {int(np.sum(~np.isnan(member_values)))} cases")
    log.debug(f"Onset rate: {np.mean(~np.isnan(member_values)):.3f}" if len(member_values) else "Onset rate: nan")
    log.debug(f"Ensemble onsets found (>= {threshold:.0%} of members): {int(has_onset.sum())}")
    log.debug(f"Ensemble onset rate: {has_onset.mean():.3f}" if len(has_onset) else "Ensemble onset rate: 0.000")
    _print_status_counts(out["status"], valid)
    _print_back_dated(member_day, valid)
    return onset_df, ensemble_df


def alignment_report(rain_slice, p_model, rc, case):
    """Compare observations with forecast leads 1-3; return (report, note).

    report is the AlignmentReport (None when the check could not run) and note says why it did not.
    Subsamples the inits and members (hcwf.onset.alignment is expensive on whole years).
    """
    setup = rc.onset
    try:
        cube = p_model.sel(step=slice(1, 3))
        stride = max(1, -(-cube.sizes["init_time"] // _ALIGNMENT_MAX_INITS))
        cube = cube.isel(init_time=slice(None, None, stride))
        if "member" in cube.dims:
            cube = cube.isel(member=slice(0, _ALIGNMENT_MAX_MEMBERS))
        obs = ingest(rain_slice, day_boundary=setup.obs_day_boundary, units="mm", source="observations")
        forecast = ingest_forecast(cube, setup, case.model, rc.onset_options, Selection(),
                                   str(case.model.name or "forecast"))
        return check_forecast_alignment(obs, forecast, leads=(1, 2, 3)), ""
    except Exception as exc:  # noqa: BLE001 - a failed check is reported, not raised
        return None, f"could not run: {exc}"


def check_alignment(rain_slice, p_model, rc, case):
    """Compare observations with forecast leads 1-3 once per model and run, and warn if they look shifted.

    Skipped (with a note) when check_alignment is off or the observation day boundary is unknown.
    Never raises: a failed check is reported and the run goes on.
    """
    setup = rc.onset
    key = (case.model.name, rc.observations.dir, setup.lead_label)
    if key in _alignment_done or not setup.check_alignment:
        return
    _alignment_done.add(key)
    report, note = alignment_report(rain_slice, p_model, rc, case)
    if report is None:
        log.info(f"Alignment check skipped: {note}")
        return
    log.info(f"Alignment check, {case.model.name} against observations:")
    log.info(report.summary())
    if report.verdict.startswith("likely shifted"):
        warnings.warn(f"{case.model.name} forecasts look shifted against the observations "
                      f"({report.verdict}); check obs_day_boundary and model_lead_label.", UserWarning)

