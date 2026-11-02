"""A cache of observed onset dates, shared by every model, window and reference of a run and kept across runs.

Observed onsets depend only on the observations, how they are read, the evaluated cells, the onset definition,
the thresholds (and season climatology) and the years, so one detection serves all models, verification windows,
the reference and later runs. The entry is keyed by a *fingerprint* of those six parts (see momp.cache.Store, which
keeps the entries and reports hits and misses); the backend, models, windows and reference are not in it. While a run is active (``activate``), ``momp.stats.detect.detect_observed_onset``
asks ``lookup`` first and calls ``store`` after detecting, so each year is detected once per fingerprint.

On disk (``<cache_dir>/observed_onsets/<fingerprint>/``): ``parts.json`` (the hash of each part) and ``onsets.nc``
(``onset_date`` over year, lat, lon). Without a cache folder the cache is in memory for the run only. A failure of the cache never stops a run: the onsets are then simply detected.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from momp.cache import Codec, Store, file_identity, part_hash

log = logging.getLogger(__name__)

SUBFOLDER = "observed_onsets"
_state = {"on": False, "dir": None, "worker": False, "store": None}
_tables: dict = {}
_new: list = []
_memo: dict = {}


def _read(path: Path) -> dict | None:
    """Read a cached entry: {"years": {year: values}, "lat": ..., "lon": ...}."""
    with xr.open_dataset(path) as ds:
        years = {int(y): ds["onset_date"].sel(year=y).values.astype("datetime64[ns]") for y in ds["year"].values}
        return {"years": years, "lat": ds["lat"].values.copy(), "lon": ds["lon"].values.copy()}


def _write(path: Path, table: dict) -> None:
    """Write an entry: onset_date over year, lat, lon."""
    years = sorted(table["years"])
    ds = xr.Dataset({"onset_date": (("year", "lat", "lon"), np.stack([table["years"][y] for y in years]))},
                    coords={"year": years, "lat": table["lat"], "lon": table["lon"]})
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
    from momp.io.input import clear_obs_cache

    clear_obs_cache()


def _threshold_hash(thresh) -> str:
    """Hash of the threshold values: the named arrays, the season climatology, or the legacy scalar or map."""
    cached = _memo.get(("thresh", id(thresh)))
    if cached is not None and cached[0] is thresh:
        return cached[1]
    if isinstance(thresh, dict):
        season = getattr(thresh, "season_climatology", None)
        result = part_hash(dict(thresh), None if season is None else {n: season[n] for n in season.data_vars})
    else:
        result = part_hash(thresh)
    _memo[("thresh", id(thresh))] = (thresh, result)
    return result


def _years(kwargs) -> list[int]:
    """The years observed onsets are needed for: the evaluated years and the climatology years."""
    years = set()
    for key in ("years", "years_clim"):
        value = kwargs.get(key)
        if value and not isinstance(value, str):
            years.update(int(y) for y in value)
    return sorted(years)


def obs_files(kwargs, years) -> list:
    """The identity (name, size, mtime) of the observation file of every year (or of the zarr store, with the years)."""
    from momp.io.input import is_zarr

    if is_zarr(kwargs["obs_dir"]):
        return [file_identity(kwargs["obs_dir"]), list(years)]
    found = []
    for year in years:
        for pattern in kwargs.get("obs_file_pattern") or ():
            path = Path(str(kwargs["obs_dir"])) / pattern.format(year)
            if path.exists():
                found.append(file_identity(path))
                break
        else:
            found.append([f"missing {year}"])
    return found


def _obs_files_hash(kwargs, years) -> str:
    """Hash of obs_files, memoised for the run."""
    key = ("files", str(kwargs.get("obs_dir")), str(kwargs.get("obs_file_pattern")), tuple(years))
    if key not in _memo:
        _memo[key] = part_hash(obs_files(kwargs, years))
    return _memo[key]


def _mask_files(kwargs) -> list:
    """Identity of the files the cell selection reads (mask file, condition file, nc_mask, shapefiles)."""
    paths = []
    mask = kwargs.get("mask")
    if isinstance(mask, dict):
        paths += [mask.get("file"), (mask.get("condition") or {}).get("file")]
    paths += [kwargs.get("nc_mask"), kwargs.get("shpfile_dir")]
    return [file_identity(p) for p in paths if p]


def fingerprint_parts(rain, thresh, kwargs) -> dict:
    """The hash of each fingerprint part for one observed-onset detection (see the module text)."""
    from momp.lib.onset_config import onset_setup
    from momp.onset import definition as definition_mod

    setup = onset_setup(kwargs, thresh)
    years = _years(kwargs)
    lat, lon = (np.asarray(rain.coords[name] if isinstance(rain.coords, dict) else rain[name].values)
                for name in ("lat", "lon"))   # rain is a DataArray, or an ingested DailyRain
    return {
        "obs_files": _obs_files_hash(kwargs, years),
        "obs_ingest": part_hash(kwargs.get("obs_var"), kwargs.get("obs_unit_cvt"), setup.ingest_boundary,
                                kwargs.get("obs_file_pattern")),
        "cells": part_hash(kwargs.get("region"), json.dumps(kwargs.get("mask"), sort_keys=True, default=str),
                           _mask_files(kwargs), bool(kwargs.get("polygon")), bool(kwargs.get("grid_point")),
                           kwargs.get("lat_select") if kwargs.get("grid_point") else None,
                           kwargs.get("lon_select") if kwargs.get("grid_point") else None, lat, lon),
        "definition": part_hash(definition_mod.definition_hash(setup.definition), setup.season, setup.dry_check,
                                setup.legacy),
        "thresholds": _threshold_hash(thresh),
        "years": part_hash(years),
    }


def fingerprint(parts: dict) -> str:
    """The fingerprint (``obs-`` and 12 hex characters) of a dict of part hashes."""
    return _new_store(None).fingerprint(parts)


def _table(parts: dict) -> dict:
    """The in-memory table of these parts, read from the cache the first time (which logs a hit or a miss)."""
    fp = _state["store"].fingerprint(parts)
    if fp not in _tables:
        _tables[fp] = _state["store"].get(parts) or {"years": {}, "lat": None, "lon": None}
    return _tables[fp]


def lookup(rain, thresh, year, kwargs):
    """The cached observed onsets of a year as a (lat, lon) DataArray, or None (inactive, or not cached yet)."""
    if not _state["on"]:
        return None
    try:
        parts = fingerprint_parts(rain, thresh, kwargs)
        table = _table(parts)
        values = table["years"].get(int(year))
        if values is None:
            return None
        return xr.DataArray(values.copy(), coords=[("lat", table["lat"]), ("lon", table["lon"])], name="onset_date")
    except Exception as exc:  # noqa: BLE001 - the cache must never break a run
        log.debug("observed-onset cache lookup failed (%s)", exc)
        return None


def store(rain, thresh, year, kwargs, onset) -> None:
    """Keep the detected onsets of a year (memory, and disk when a cache folder is set)."""
    if not _state["on"]:
        return
    try:
        parts = fingerprint_parts(rain, thresh, kwargs)
        fp = _state["store"].fingerprint(parts)
        table = _table(parts)
        table["years"][int(year)] = np.asarray(onset.values).astype("datetime64[ns]")
        table["lat"], table["lon"] = np.asarray(onset["lat"].values), np.asarray(onset["lon"].values)
        if _state["worker"]:   # a worker process: the parent writes the entry (see absorb)
            _new.append((fp, parts, int(year), table["years"][int(year)], table["lat"], table["lon"]))
        else:
            _state["store"].put(parts, table)
    except Exception as exc:  # noqa: BLE001 - the cache must never break a run
        log.debug("observed-onset cache store failed (%s)", exc)


def current() -> dict:
    """The in-memory tables of this run, {fingerprint: {"years": {year: values}, "lat", "lon"}} (for results)."""
    return _tables


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
    """The onset tables a worker detected since the last call, [(fingerprint, parts, year, values, lat, lon)]; clears them."""
    out = list(_new)
    _new.clear()
    return out


def absorb(records: list) -> None:
    """Take onsets a worker detected into this process's cache (memory, and the folder when one is set), in order."""
    if not _state["on"]:
        return
    for fp, parts, year, values, lat, lon in records:
        table = _table(parts)
        table["years"][year] = values
        table["lat"], table["lon"] = lat, lon
        _state["store"].put(parts, table)
