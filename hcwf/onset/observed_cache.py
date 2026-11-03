"""A cache of observed onset dates, shared by every model, window and reference of a run and kept across runs.

Observed onsets depend only on the observations, how they are read, the evaluated cells, the onset definition,
the thresholds (and season climatology) and the years, so one detection serves all models, verification windows,
the reference and later runs. The entry is keyed by a *fingerprint* of those six parts (see hcwf.cache.Store, which
keeps the entries and reports hits and misses); the backend, models, windows and reference are not in it. Evaluation
units replace the cells part by a units part (the units specification, the files it names, the units kept), and run
options that change the numbers (missing-rain policy, resolution, horizon, the reference onset a search starts at) add
an options part when they are set; without them the fingerprint is the one a grid run always had. While a run is active (``activate``), ``hcwf.onset.detect.detect_observed``
asks ``lookup`` first and calls ``store`` after detecting, so each year is detected once per fingerprint.

The fingerprint is computed from typed inputs (``fingerprint_parts(rain, thresh, obs, area, onset, options, select,
years)``; ``parts_for`` takes them from a RunConfig). Its parts are the same hashes the kwargs form gave, so entries
written by earlier runs are still hits.

On disk (``<cache_dir>/observed_onsets/<fingerprint>/``): ``parts.json`` (the hash of each part) and ``onsets.nc``
(``onset_date`` over year and the spatial dimensions, lat and lon or unit, plus any per-unit arrays the onsets carry as
coordinates, such as ``present``). Without a cache folder the cache is in memory for the run only. A failure of the cache never stops a run: the onsets are then simply detected.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.cache import Codec, Store, file_identity, part_hash
from hcwf.config.model import Selection

log = logging.getLogger(__name__)

SUBFOLDER = "observed_onsets"
_state = {"on": False, "dir": None, "worker": False, "store": None}
_tables: dict = {}
_new: list = []
_memo: dict = {}


def _empty() -> dict:
    return {"years": {}, "dims": None, "coords": None, "lat": None, "lon": None, "extras": {}}


def _read(path: Path) -> dict | None:
    """Read a cached entry: {"years": {year: values}, "dims", "coords": {dim: values}, "extras": {year: {name: values}}}
    (with "lat" and "lon" too for a grid)."""
    with xr.open_dataset(path) as ds:
        dims = tuple(ds["onset_date"].dims[1:])
        years = {int(y): ds["onset_date"].sel(year=y).values.astype("datetime64[ns]") for y in ds["year"].values}
        extra = [n for n in ds.data_vars if n != "onset_date"]
        extras = {int(y): {n: ds[n].sel(year=y).values for n in extra} for y in ds["year"].values}
        coords = {d: ds[d].values.copy() for d in dims}
        return {"years": years, "dims": dims, "coords": coords, "lat": coords.get("lat"), "lon": coords.get("lon"),
                "extras": extras}


def _write(path: Path, table: dict) -> None:
    """Write an entry: onset_date over year and the spatial dims, and each extra array over the same."""
    years, dims = sorted(table["years"]), table["dims"]
    variables = {"onset_date": (("year", *dims), np.stack([table["years"][y] for y in years]))}
    for name in sorted({n for y in years for n in table["extras"].get(y, {})}):
        variables[name] = (("year", *dims), np.stack([table["extras"][y][name] for y in years]))
    ds = xr.Dataset(variables, coords={"year": years, **{d: table["coords"][d] for d in dims}})
    ds.to_netcdf(path)


def _new_store(root) -> Store:
    """The observed-onset store of a cache folder (None: in memory)."""
    return Store(SUBFOLDER, root, codec=Codec("onsets.nc", _read, _write), prefix="obs", label="Observed onsets",
                 describe=lambda table: f"{len(table['years'])} years cached",
                 miss_note="; detecting once for all models, windows and references")


def activate(cache_dir=None) -> None:
    """Start using the cache for this run; cache_dir None keeps it in memory. Clears what an earlier run left."""
    reset()
    _state.update(on=True, dir=Path(cache_dir) if cache_dir else None, store=_new_store(cache_dir))


def reset() -> None:
    """Stop using the cache and forget the in-memory tables, announcements and memoised parts."""
    _state.update(on=False, dir=None, worker=False, store=None)
    _new.clear()
    _tables.clear()
    _memo.clear()
    from hcwf.io.reference import clear_obs_cache

    clear_obs_cache()


def threshold_hash(thresh) -> str:
    """Hash of the threshold values: the named arrays or the season climatology."""
    cached = _memo.get(("thresh", id(thresh)))
    if cached is not None and cached[0] is thresh:
        return cached[1]
    if isinstance(thresh, dict):
        season = getattr(thresh, "season_climatology", None)
        named = {name: value.to_dict("list") if isinstance(value, pd.DataFrame) else value
                 for name, value in thresh.items()}   # a table is hashed by its contents
        result = part_hash(named, None if season is None else {n: season[n] for n in season.data_vars})
    elif isinstance(thresh, xr.DataArray):   # the values and the grid they sit on
        result = part_hash(thresh, {str(name): thresh.coords[name] for name in thresh.coords})
    else:
        result = part_hash(thresh)
    _memo[("thresh", id(thresh))] = (thresh, result)
    return result


def observed_years(ev) -> list[int]:
    """The years observed onsets are needed for: the evaluated years and the climatology years of an EvaluationSpec."""
    return sorted({int(y) for y in (*ev.years, *ev.climatology_years)})


def observation_files(obs, files_years, years) -> list:
    """The identity (name, size, mtime) of the observation file of every year (or of the zarr store, with the years).

    obs is the ObsSpec; files_years is `Selection.obs_files_years` (the years of the listed files to keep)."""
    from hcwf.config.datasets import is_zarr
    from hcwf.io.stores import year_files

    if obs.paths:   # listed files, any number per year
        found = []
        for year in years:
            files = year_files(obs.paths, obs.dir or None, files_years, year)
            found.append([file_identity(f) for f in files] or [f"missing {year}"])
        return found
    if is_zarr(obs.dir):
        return [file_identity(obs.dir), list(years)]
    found = []
    for year in years:
        for pattern in obs.file_pattern or ():
            path = Path(str(obs.dir)) / pattern.format(year)
            if path.exists():
                found.append(file_identity(path))
                break
        else:
            found.append([f"missing {year}"])
    return found


def _obs_files_hash(obs, files_years, years) -> str:
    """Hash of observation_files, memoised for the run."""
    key = ("files", str(obs.dir), str(obs.file_pattern), tuple(years), str(obs.paths), str(files_years))
    if key not in _memo:
        _memo[key] = part_hash(observation_files(obs, files_years, years))
    return _memo[key]


def _mask_files(area) -> list:
    """Identity of the files the cell selection reads (mask file, condition file, nc_mask, shapefiles)."""
    paths = []
    if area.mask is not None:
        paths += [area.mask.file, (area.mask.condition or {}).get("file")]
    paths += [area.nc_mask, area.shpfile_dir]
    return [file_identity(p) for p in paths if p]


_MASK_GIVEN = ("land", "country", "file", "variable", "keep_value", "condition", "cell_rule", "overlap_fraction",
               "resolution")
"""The keys of a config's `mask:` section (the other MaskSpec fields are set by the implicit masks only)."""


def _mask_mapping(area):
    """The `mask:` mapping of the config as it was given (`AreaSpec.mask_given`, so a key written with its default value
    still hashes as written, as before). An area built by hand has none: the section's keys that differ from their defaults."""
    if area.mask_given is not None:
        return area.mask_given
    mask = area.mask
    if mask is None:
        return None
    default = type(mask)()
    return {key: list(value) if key == "country" else value for key in _MASK_GIVEN
            if (value := getattr(mask, key)) != getattr(default, key)}


def _reference_part(reference):
    """What identifies a reference onset: its month-day, table or computed field."""
    if reference is None:
        return None
    table = None if reference.table is None else reference.table.to_dict("list")
    return (reference.month_day, table, reference.computed, reference.statistic)


def _options(options, units, select) -> dict:
    """The run options that change observed onsets and are set: reading policies, the horizon, the reference onset."""
    out = {**options.ingest(False, units)}
    if options.horizon:
        out["horizon"] = options.horizon
    if select.reference_onset is not None:
        out["reference_onset"] = _reference_part(select.reference_onset)
    return out


def fingerprint_parts(rain, thresh, obs, area, onset, options, select, years) -> dict:
    """The hash of each fingerprint part for one observed-onset detection (see the module text).

    `obs` (ObsSpec), `area` (AreaSpec), `onset` (OnsetSetup) and `options` (OnsetOptions) are the run's, `select` (Selection)
    the call's, and `years` the years observed onsets are needed for (`observed_years`). `rain` supplies the grid of the
    cells part (a DataArray, or an ingested DailyRain); with evaluation units it is not needed (None is fine): the files
    and the units specification say which units there are."""
    from hcwf.onset import definition as definition_mod

    spec = obs.units
    parts = {
        "obs_files": _obs_files_hash(obs, select.obs_files_years, years),
        "obs_ingest": part_hash(obs.var, obs.unit_cvt, onset.obs_day_boundary, obs.file_pattern),
        "definition": part_hash(definition_mod.definition_hash(onset.definition), onset.season, onset.dry_check),
        "thresholds": threshold_hash(thresh),
        "years": part_hash(list(years)),
    }
    if spec is None:
        lat, lon = (np.asarray(rain.coords[name] if isinstance(rain.coords, dict) else rain[name].values)
                    for name in ("lat", "lon"))   # rain is a DataArray, or an ingested DailyRain
        parts["cells"] = part_hash(area.region, json.dumps(_mask_mapping(area), sort_keys=True, default=str),
                                   _mask_files(area), bool(area.polygon), bool(select.grid_point),
                                   select.lat if select.grid_point else None,
                                   select.lon if select.grid_point else None, lat, lon)
    else:
        parts["units"] = part_hash(repr(spec), [file_identity(p) for p in spec.files()], obs.dims_map(),
                                   sorted(select.keep_unit_ids or ()), onset.lead_label)
    opts = _options(options, spec, select)
    if opts:
        parts["options"] = part_hash(opts)
    return parts


def parts_for(rain, thresh, rc, select=Selection()) -> dict:
    """fingerprint_parts of a RunConfig's observations, area, onset and options."""
    return fingerprint_parts(rain, thresh, rc.observations, rc.area, rc.onset, rc.onset_options, select,
                             observed_years(rc.evaluation))


def fingerprint(parts: dict) -> str:
    """The fingerprint (``obs-`` and 12 hex characters) of a dict of part hashes."""
    return _new_store(None).fingerprint(parts)


def _table(parts: dict) -> dict:
    """The in-memory table of these parts, read from the cache the first time (which logs a hit or a miss)."""
    fp = _state["store"].fingerprint(parts)
    if fp not in _tables:
        _tables[fp] = _state["store"].get(parts) or _empty()
    return _tables[fp]


def lookup(rain, thresh, year, rc, select=Selection()):
    """The cached observed onsets of a year as a DataArray over the spatial dims (a per-unit array the onsets carried
    comes back as a coordinate), or None (inactive, or not cached yet)."""
    if not _state["on"]:
        return None
    try:
        parts = parts_for(rain, thresh, rc, select)
        table = _table(parts)
        values = table["years"].get(int(year))
        if values is None:
            return None
        dims = table["dims"]
        out = xr.DataArray(values.copy(), coords=[(d, table["coords"][d]) for d in dims], name="onset_date")
        extras = table["extras"].get(int(year), {})
        return out.assign_coords({name: (dims, value.copy()) for name, value in extras.items()}) if extras else out
    except Exception as exc:  # noqa: BLE001 - the cache must never break a run
        log.debug("observed-onset cache lookup failed (%s)", exc)
        return None


def _entry(onset) -> tuple:
    """(dims, coords, extras) of an onset DataArray: its index coordinates and the arrays it carries as coordinates."""
    dims = tuple(onset.dims)
    return (dims, {d: np.asarray(onset[d].values) for d in dims},
            {name: np.asarray(onset.coords[name].values) for name in onset.coords if name not in onset.dims})


def store(rain, thresh, year, rc, select, onset) -> None:
    """Keep the detected onsets of a year (memory, and disk when a cache folder is set)."""
    if not _state["on"]:
        return
    try:
        parts = parts_for(rain, thresh, rc, select)
        fp = _state["store"].fingerprint(parts)
        table = _table(parts)
        dims, coords, extras = _entry(onset)
        table["years"][int(year)] = np.asarray(onset.values).astype("datetime64[ns]")
        table.update(dims=dims, coords=coords, lat=coords.get("lat"), lon=coords.get("lon"))
        if extras:
            table["extras"][int(year)] = extras
        if _state["worker"]:   # a worker process: the parent writes the entry (see absorb)
            _new.append((fp, parts, int(year), table["years"][int(year)], dims, coords, extras))
        else:
            _state["store"].put(parts, table)
    except Exception as exc:  # noqa: BLE001 - the cache must never break a run
        log.debug("observed-onset cache store failed (%s)", exc)


def current() -> dict:
    """The in-memory tables of this run's detections, {fingerprint: {"years": {year: values}, "dims", "coords"}}
    (with "lat" and "lon" too for a grid), over lat and lon or over the units."""
    return {fp: table for fp, table in _tables.items() if table["dims"] is not None}


def export_state() -> dict:
    """The cache as worker processes need it: its folder, the tables read or detected so far and which were announced."""
    return {"dir": _state["dir"], "tables": dict(_tables), "announced": set(_state["store"].announced if _state["store"] else ())}


def install_state(state: dict) -> None:
    """Start a worker process's cache from the parent's state: reads the same folder, never writes to it."""
    reset()
    _state.update(on=True, dir=state["dir"], worker=True, store=_new_store(state["dir"]))
    _tables.update(state["tables"])
    _state["store"].announced.update(state["announced"])


def drain_new() -> list:
    """The onset tables a worker detected since the last call, [(fingerprint, parts, year, values, dims, coords, extras)];
    clears them."""
    out = list(_new)
    _new.clear()
    return out


def absorb(records: list) -> None:
    """Take onsets a worker detected into this process's cache (memory, and the folder when one is set), in order."""
    if not _state["on"]:
        return
    for fp, parts, year, values, dims, coords, extras in records:
        table = _table(parts)
        table["years"][year] = values
        table.update(dims=dims, coords=coords, lat=coords.get("lat"), lon=coords.get("lon"))
        if extras:
            table["extras"][year] = extras
        _state["store"].put(parts, table)
