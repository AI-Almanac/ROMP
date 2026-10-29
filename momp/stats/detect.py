"""Onset detection on observed rainfall and on deterministic or ensemble model forecasts.

A thin adapter between ROMP's pipeline and momp.onset: it settles units, day
boundaries and thresholds, runs momp.onset.engine, and shapes the results into
the arrays and DataFrames the metrics code expects.
"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from momp.io.input import onset_steps_needed
from momp.lib.onset_config import onset_setup
from momp.onset import engine
from momp.onset.alignment import check_forecast_alignment
from momp.onset.ingest import ingest, ingest_forecast
from momp.onset.legacy import LEGACY_THRESHOLD
from momp.onset.model import Status

_GRID_TOLERANCE = 1e-3
_ALIGNMENT_MAX_INITS = 40
_ALIGNMENT_MAX_MEMBERS = 10
_alignment_done = set()


def _on_grid(da, lat, lon):
    """Put a per-cell DataArray (or dataset) on the lat/lon grid of the rainfall; cells it lacks become NaN."""
    if "lat" not in da.dims or "lon" not in da.dims:
        return da
    return da.reindex(lat=lat, lon=lon, method="nearest", tolerance=_GRID_TOLERANCE)


def _thresholds(setup, thresh_slice, lat, lon):
    """The thresholds for the engine: named per-cell DataArrays on the rainfall grid, or None for a fixed amount."""
    if setup.legacy:
        if isinstance(thresh_slice, xr.DataArray) and thresh_slice.ndim:
            return {LEGACY_THRESHOLD: _on_grid(thresh_slice, lat, lon)}
        return None
    return {name: _on_grid(value, lat, lon) for name, value in dict(thresh_slice or {}).items()}


def _season_climatology(setup, thresh_slice, lat, lon):
    """The harmonic season analysis on the rainfall grid, for derived seasons only."""
    if not setup.derived_seasons:
        return None
    climatology = getattr(thresh_slice, "season_climatology", None)
    if climatology is None:
        raise ValueError("this definition has derived seasons but no season climatology was loaded; "
                         "pass the thresholds returned by momp.io.input.load_thresholds")
    return _on_grid(climatology, lat, lon)


def detect_observed_onset(rain_slice, thresh_slice, year, **kwargs):
    """Detect observed onset dates for a given year.

    rain_slice is daily rainfall in mm with dims (time, lat, lon); thresh_slice is
    the threshold from momp.io.input.load_thresholds (a number of mm or a 2-D
    DataArray for the old flat keys, else the definition's named thresholds).
    Returns a (lat, lon) DataArray of datetime64 onset dates, NaT where the cell
    has no onset.
    """
    setup = onset_setup(kwargs, thresh_slice)
    rain = ingest(rain_slice, day_boundary=setup.ingest_boundary, units="mm", source="observations")
    lat, lon = rain.coords["lat"], rain.coords["lon"]
    result = engine.detect_observed(
        setup.definition, rain, thresholds=_thresholds(setup, thresh_slice, lat, lon), years=[year],
        season_climatology=_season_climatology(setup, thresh_slice, lat, lon), backend=setup.backend)
    onset = result["onset_date"].sel(year=year, season=setup.season).transpose("lat", "lon")
    if not setup.legacy:
        counts = {s.name: int((result["status"].sel(year=year, season=setup.season) == s).sum()) for s in Status}
        print(f"Observed onset {year}, cells by status: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    return xr.DataArray(onset.values.astype("datetime64[ns]"), coords=[("lat", lat), ("lon", lon)], name="onset_date")


def _check_steps(p_model, steps_needed):
    """Raise if the forecast has fewer lead days than the old flat-key detection read.

    With a definition, a candidate whose days run past the last lead is CENSORED instead.
    """
    if steps_needed is None:
        raise ValueError("cannot tell how many forecast time steps the onset detection needs; "
                         "max_forecast_day, wet_spell and dry_extent must be whole numbers")
    full_steps = p_model.sizes["step"]
    if full_steps < steps_needed:
        raise ValueError(f"Not enough forecast time steps: model steps {full_steps} \
                < min steps required {steps_needed}, consider decrese dry_extent value")


def _detect_forecast(p_model, thresh_slice, onset_da, kwargs, *, members=None):
    """Run the engine on a forecast cube and line it up with the observed onsets.

    Returns a dict of numpy arrays shaped (init_time, [member], lat, lon): lead day
    after init of the onset (-1 none), date, status; plus the valid-case mask
    (init before the observed onset) and the observed onset dates (init_time, lat, lon).
    """
    setup = onset_setup(kwargs, thresh_slice)
    max_forecast_day = kwargs["max_forecast_day"]
    steps_needed = onset_steps_needed(**kwargs)
    if setup.legacy:
        _check_steps(p_model, steps_needed)
    cube = p_model.sel(step=slice(1, steps_needed))
    if "member" in cube.dims and members:
        cube = cube.sel(member=list(members))
    forecast = ingest_forecast(cube, day_boundary=setup.forecast_boundary, lead_label=setup.lead_label,
                               units="mm", source="forecast")
    lat, lon = forecast.coords["lat"], forecast.coords["lon"]
    result = engine.detect_forecast(
        setup.definition, forecast, thresholds=_thresholds(setup, thresh_slice, lat, lon),
        lead_window=(1, max_forecast_day), restrict_to_season=not setup.derived_seasons, backend=setup.backend)
    season = "window" if setup.derived_seasons else setup.season
    result = result.sel(season=season)
    if not setup.legacy and bool((result["status"] == Status.CENSORED).any()):
        warnings.warn("some forecast searches are CENSORED although the lead window was shortened to fit the rule; "
                      "they are reported as no onset. Is max_forecast_day larger than the loaded leads allow?",
                      UserWarning)
    lead_dims = ("init_time", "member", "lat", "lon") if "member" in forecast.dims else ("init_time", "lat", "lon")
    lead = result["onset_lead"].transpose(*lead_dims).values
    dates = result["onset_date"].transpose(*lead_dims).values
    status = result["status"].transpose(*lead_dims).values
    init = np.asarray(forecast.coords["init_time"])
    obs = onset_da.reindex(lat=lat, lon=lon, method="nearest", tolerance=_GRID_TOLERANCE).transpose("lat", "lon")
    obs_values = obs.values.astype("datetime64[ns]")
    valid = ~np.isnat(obs_values)[None] & (init.astype("datetime64[ns]")[:, None, None] < obs_values[None])
    offset = 1 if setup.lead_label == "start" else 0
    found = status == Status.ONSET
    return {"setup": setup, "init": init, "lat": np.asarray(lat), "lon": np.asarray(lon),
            "member": np.asarray(forecast.coords["member"]) if "member" in forecast.dims else None,
            "day": np.where(found, np.nan_to_num(lead, nan=-1) - offset, -1).astype(np.int64), "date": dates, "status": status,
            "valid": valid, "obs": obs_values, "totals": (int(obs.size * len(init)), int((~np.isnat(obs_values)).sum()))}


def _date_strings(values):
    """'%Y-%m-%d' text of datetime64 values, as a list."""
    return pd.DatetimeIndex(np.ravel(values)).strftime("%Y-%m-%d").tolist()


def _print_status_counts(setup, status, valid):
    """Print how many forecast searches ended in each status (definition mode only)."""
    if setup.legacy:
        return
    mask = np.broadcast_to(valid[:, None] if status.ndim == 4 else valid, status.shape)
    counts = np.bincount(status[mask].astype(np.int64), minlength=len(Status))
    print("Forecast onset searches by status: "
          + ", ".join(f"{s.name} {int(counts[s])}" for s in Status if counts[s])
          + "")


def _names(status):
    """Status names for an array of codes."""
    labels = np.array([s.name for s in Status])
    return labels[np.asarray(status, dtype=np.int64)]


def compute_onset_for_deterministic_model(p_model, thresh_slice, onset_da, **kwargs):
    """Compute onset dates for deterministic model forecast.

    p_model has dims (init_time, lat, lon, step). Returns one row per (init, cell)
    in that order for cells with an observed onset and an init before it, with
    columns init_time, lat, lon, onset_day (lead day, None if no onset), onset_date
    ('YYYY-MM-DD' or None) and obs_onset_date, plus onset_status for definitions.
    """
    out = _detect_forecast(p_model, thresh_slice, onset_da, kwargs)
    setup, valid = out["setup"], out["valid"]
    print(f"Processing {len(out['init'])} init times x {len(out['lat'])} lats x {len(out['lon'])} lons...")
    i, j, k = np.nonzero(valid)
    day = out["day"][i, j, k]
    has = day >= 0
    onset_date = np.where(has, _date_strings(out["date"][i, j, k]), None).tolist() if len(i) else []
    columns = {
        "init_time": out["init"][i], "lat": out["lat"][j], "lon": out["lon"][k],
        "onset_day": [int(d) if h else None for d, h in zip(day, has)],
        "onset_date": onset_date,
        "obs_onset_date": _date_strings(out["obs"][j, k]) if len(i) else [],
    }
    if not setup.legacy:
        columns["onset_status"] = _names(out["status"][i, j, k]).tolist()
    onset_df = pd.DataFrame(columns) if len(i) else pd.DataFrame([])
    total, with_obs = out["totals"]
    print(f"\nProcessing Summary:")
    print(f"Total potential initializations: {total}")
    print(f"Skipped (no observed onset): {total - with_obs * len(out['init'])}")
    print(f"Skipped (initialized after observed onset): {with_obs * len(out['init']) - len(i)}")
    print(f"Valid initializations processed: {len(i)}")
    print(f"Onsets found: {int(has.sum())}")
    print(f"Onset rate: {has.sum() / len(i):.3f}" if len(i) else "Onset rate: 0.000")
    _print_status_counts(setup, out["status"], valid)
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


def compute_onset_for_all_members(p_model, thresh_slice, onset_da, **kwargs):
    """Compute per-member and ensemble onsets.

    p_model has dims (init_time, member, lat, lon, step). The ensemble has an onset
    when the fraction of members with one is at least onset_percentage_threshold;
    its day is the mean member onset day rounded half to even.
    Returns (member onset DataFrame, ensemble onset DataFrame).
    """
    members = kwargs.get("members")
    threshold = kwargs["onset_percentage_threshold"]
    out = _detect_forecast(p_model, thresh_slice, onset_da, kwargs, members=members)
    setup, valid, day = out["setup"], out["valid"], out["day"]
    n_init, n_member = day.shape[0], day.shape[1]
    print(f"Processing {n_init} init times x {len(out['lat']) * len(out['lon'])} unique locations x {n_member} members...")

    member_day = np.where(day >= 0, day, np.nan).astype(float)
    count = (day >= 0).sum(axis=1)
    fraction = count / n_member if n_member > 0 else np.zeros(count.shape)
    with np.errstate(invalid="ignore"):
        mean_day = np.nanmean(member_day, axis=1) if n_member else np.full(count.shape, np.nan)
    ensemble_day = np.where(fraction >= threshold, np.round(mean_day), np.nan)
    ensemble_date = pd.DatetimeIndex(out["init"]).values[:, None, None] + (
        np.nan_to_num(ensemble_day, nan=0).astype("int64").astype("timedelta64[D]"))
    onset_loop = setup.legacy and kwargs["dry_extent"] > kwargs["wet_spell"]

    grid = valid.shape
    if onset_loop:
        order = np.array(np.nonzero(np.transpose(valid, (0, 2, 1)))).T
        ii, kk, jj = order[:, 0], order[:, 1], order[:, 2]
        member_rows = (np.repeat(ii, n_member), np.tile(np.arange(n_member), len(ii)),
                       np.repeat(jj, n_member), np.repeat(kk, n_member))
        member_index = None
        ens_rows = (ii, jj, kk)
        ens_index = None
    else:
        flat_valid = np.broadcast_to(valid[:, None], (grid[0], n_member) + grid[1:])
        member_rows = tuple(np.nonzero(flat_valid))
        member_index = pd.Index(np.flatnonzero(flat_valid.ravel()))
        ens_rows = tuple(np.nonzero(valid))
        ens_index = pd.Index(np.flatnonzero(valid.ravel()))
    mi, mm, mj, mk = member_rows
    ei, ej, ek = ens_rows

    member_values = member_day[mi, mm, mj, mk]
    onset_df = pd.DataFrame({
        "init_time": out["init"][mi], "lat": out["lat"][mj], "lon": out["lon"][mk],
        "member": out["member"][mm],
        "onset_day": member_values if not onset_loop else [None if np.isnan(v) else int(v) for v in member_values],
        "obs_onset_date": pd.Series(pd.to_datetime(out["obs"][mj, mk])).dt.strftime("%Y-%m-%d").to_numpy()
        if len(mi) else [],
    }, index=member_index)
    chosen = ensemble_day[ei, ej, ek]
    has_onset = ~np.isnan(chosen)
    ens_dates = np.where(has_onset, _date_strings(ensemble_date[ei, ej, ek]) if len(ei) else [], None)
    ensemble_df = pd.DataFrame({
        "init_time": out["init"][ei], "lat": out["lat"][ej], "lon": out["lon"][ek],
        "onset_day": chosen if not onset_loop else [int(v) if h else None for v, h in zip(chosen, has_onset)],
        "member_onset_count": count[ei, ej, ek], "total_members": n_member,
        "onset_percentage": fraction[ei, ej, ek],
        "obs_onset_date": pd.Series(pd.to_datetime(out["obs"][ej, ek])).dt.strftime("%Y-%m-%d").to_numpy()
        if len(ei) else [],
    }, index=ens_index)
    ensemble_df.insert(4, "onset_date", None)
    if onset_loop:
        ensemble_df["onset_date"] = ens_dates.tolist()
    elif has_onset.any():
        ensemble_df.loc[has_onset, "onset_date"] = ens_dates[has_onset]
    if not setup.legacy:
        onset_df["onset_status"] = _names(out["status"][mi, mm, mj, mk])
        ensemble_df["onset_status"] = _ensemble_status(out["status"][ei, :, ej, ek], threshold, has_onset)

    print(f"\nProcessing Summary:")
    print(f"Generated {len(onset_df)} member-forecast combinations")
    print(f"Found onset in {int(np.sum(member_values >= 0))} cases")
    print(f"Onset rate: {np.mean(member_values >= 0):.3f}" if len(member_values) else "Onset rate: nan")
    print(f"Ensemble onsets found (>= {threshold:.0%} of members): {int(has_onset.sum())}")
    print(f"Ensemble onset rate: {has_onset.mean():.3f}" if len(has_onset) else "Ensemble onset rate: 0.000")
    _print_status_counts(setup, out["status"], valid)
    return onset_df, ensemble_df


def check_alignment_once(rain_slice, p_model, **kwargs):
    """Compare observations with forecast leads 1-3 once per model and run, and warn if they look shifted.

    Skipped (with a note) when check_alignment is off or the observation day boundary is unknown.
    Never raises: a failed check is reported and the run goes on.
    """
    setup = onset_setup(kwargs)
    key = (kwargs.get("model"), kwargs.get("obs_dir"), setup.lead_label)
    if key in _alignment_done or not setup.check_alignment:
        return
    _alignment_done.add(key)
    if setup.obs_day_boundary is None:
        print("Alignment check skipped: the observation day boundary is unknown "
              "(set obs_day_boundary to enable it).")
        return
    try:
        cube = p_model.sel(step=slice(1, 3))
        stride = max(1, -(-cube.sizes["init_time"] // _ALIGNMENT_MAX_INITS))
        cube = cube.isel(init_time=slice(None, None, stride))
        if "member" in cube.dims:
            cube = cube.isel(member=slice(0, _ALIGNMENT_MAX_MEMBERS))
        obs = ingest(rain_slice, day_boundary=setup.obs_day_boundary, units="mm", source="observations")
        forecast = ingest_forecast(cube, day_boundary=setup.forecast_boundary, lead_label=setup.lead_label,
                                   units="mm", source=str(kwargs.get("model", "forecast")))
        report = check_forecast_alignment(obs, forecast, leads=(1, 2, 3))
    except Exception as exc:
        print(f"Alignment check could not run: {exc}")
        return
    print(f"Alignment check, {kwargs.get('model')} against observations:")
    print(report.summary())
    if report.verdict.startswith("likely shifted"):
        warnings.warn(f"{kwargs.get('model')} forecasts look shifted against the observations "
                      f"({report.verdict}); check obs_day_boundary and model_lead_label.", UserWarning)
