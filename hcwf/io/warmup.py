"""What every year of a run shares, built once in the parent process before the years go to workers: the thresholds and the cell masks.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import xarray as xr

from hcwf.config.datasets import is_zarr
from hcwf.config.model import NO_SELECTION
from hcwf.io.dims import dim_fmt, dim_fmt_model, dim_fmt_model_ensemble
from hcwf.io.stores import observation_source, open_store, year_files
from hcwf.io.thresholds import load_thresholds

log = logging.getLogger(__name__)

def _warm_masks(lazy, var, dataset_name, area, select, dir_fig):
    """Build the cell masks of a lazily opened dataset's grid (what its first read would build), without reading data.

    `lazy` is the (dim-standardised) Dataset; a zero field on its region box is masked in place with the run's
    settings, which fills the same mask caches (and logs the same mask text) as the real read.
    """
    from hcwf.spatial.select import mask_region_loaded, region_box

    box = region_box(lazy[[var]], region=area.region)
    lat, lon = np.asarray(box["lat"].values), np.asarray(box["lon"].values)
    field = xr.DataArray(np.zeros((len(lat), len(lon)), dtype="float32"), dims=("lat", "lon"),
                         coords={"lat": lat, "lon": lon})
    mask_region_loaded(xr.Dataset({var: field}), area=area, select=select, dataset_name=dataset_name, dir_fig=dir_fig)


def warm_up(onset, obs, area, options, ev, output, *, model=None, select=NO_SELECTION, parallel=None,
            cache_dir=None):
    """Read or build in this process what every year of a run shares: the thresholds and the cell masks of both grids.

    Done once before the years go to worker processes, which then get these (see hcwf.parallel.snapshot)
    instead of each building them, and whose messages are logged once. The first of the evaluation years picks the
    NetCDF files whose grids are looked at. Thresholds are loaded as the years would load them; the masks are best effort
    (a grid that cannot be looked at is left to the workers, which raise the real error if there is one).
    """
    load_thresholds(onset, obs, area, options, ev, output, model=model, select=select, parallel=parallel,
                    cache_dir=cache_dir)
    years = ev.years
    if not years or obs.units is not None:   # units are read without a cell mask
        return
    year = int(list(years)[0])
    dir_fig = output.figure_dir
    try:
        store, path = observation_source(year, obs.dir, obs.file_pattern, obs.paths, select.obs_files_years)
        path = path[0] if isinstance(path, tuple) else path
        lazy = dim_fmt(open_store(store) if store is not None else xr.open_dataset(path))
        try:
            _warm_masks(lazy, obs.var, str(obs.name or "obs"), area, select, dir_fig)
        finally:
            if path is not None:
                lazy.close()
    except Exception as exc:  # noqa: BLE001 - only a warm-up
        log.debug(f"obs mask pre-warm skipped: {exc}")
    try:
        fix = dim_fmt_model_ensemble if ev.probabilistic else dim_fmt_model
        if select.model_paths:
            path = year_files(select.model_paths, model.dir, select.model_files_years, year)[0]
            lazy = fix(xr.open_dataset(path), model.dims_map())
        elif is_zarr(model.dir):
            lazy = fix(open_store(model.dir), model.dims_map())
            path = None
        else:
            path = os.path.join(model.dir, model.file_pattern.format(year))
            lazy = fix(xr.open_dataset(path), model.dims_map())
        try:
            _warm_masks(lazy, model.var, str(model.name or "model"), area, select, dir_fig)
        finally:
            if path is not None:
                lazy.close()
    except Exception as exc:  # noqa: BLE001 - only a warm-up (a climatology source has no forecast files)
        log.debug(f"model mask pre-warm skipped: {exc}")


def warm_up_of(rc, model, select=NO_SELECTION):
    """`warm_up` of a run for the source `model`."""
    return warm_up(rc.onset, rc.observations, rc.area, rc.onset_options, rc.evaluation, rc.output, model=model,
                   select=select, parallel=rc.parallel, cache_dir=rc.cache_dir)
