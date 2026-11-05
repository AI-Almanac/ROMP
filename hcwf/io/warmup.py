"""What every year of a run shares, built once in the parent process before the years go to workers.

This is the thresholds (`hcwf.io.thresholds.load_thresholds`) and the cell masks of the observation and model grids
(`hcwf.spatial.select.mask_region_loaded`). `warm_up` is called by the run entry points before the years are
scattered; the workers then receive these through the shared state of `hcwf.cache` instead of each building them.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import xarray as xr

from hcwf.config.datasets import is_zarr
from hcwf.config.model import NO_SELECTION
from hcwf.io.dims import dim_fmt, dim_fmt_model, dim_fmt_model_ensemble
from hcwf.io.stores import observation_source, open_data, open_store, year_files
from hcwf.io.thresholds import load_thresholds

__all__ = ["warm_up_of"]


log = logging.getLogger(__name__)


def _warm_masks(lazy, var, dataset_name, area, select, dir_fig, source=None):
    """Build the cell masks of a lazily opened dataset's grid (what its first read would build), without reading data.

    A zero field on the dataset's region box is masked in place with the run's settings, which fills the same mask
    caches (and logs the same mask text) as the real read. A source that is averaged onto the evaluation grid is read on
    that grid, so the mask of that grid is built instead (`hcwf.spatial.evalgrid.keep_cells`).

    Parameters
    ----------
    lazy : xarray.Dataset
        The lazily opened, dim-standardised dataset, with ``lat`` and ``lon`` coordinates.
    var : str
        The data variable whose grid is looked at.
    dataset_name : str
        The dataset's name, used in the mask log text and cache keys.
    area : hcwf.config.model.AreaSpec
        The region and mask settings of the run.
    select : hcwf.config.model.Selection
        The run's selection settings (``NO_SELECTION`` when none).
    dir_fig : str or pathlib.Path
        The folder where mask figures are written.
    source : str, optional
        ``"observations"`` or the model's name, which says whether it is averaged onto the evaluation grid.
    """
    from hcwf.spatial.evalgrid import keep_cells
    from hcwf.spatial.select import mask_region_loaded, region_box

    grid = area.evaluation_grid
    if grid is not None and source is not None and grid.regrids(source):
        keep_cells(grid, area, select, dataset_name, dir_fig)
    else:
        box = region_box(lazy[[var]], region=area.region)
        lat, lon = np.asarray(box["lat"].values), np.asarray(box["lon"].values)
        field = xr.DataArray(np.zeros((len(lat), len(lon)), dtype="float32"), dims=("lat", "lon"),
                             coords={"lat": lat, "lon": lon})
        mask_region_loaded(xr.Dataset({var: field}), area=area, select=select, dataset_name=dataset_name,
                           dir_fig=dir_fig)


def warm_up(onset, obs, area, options, ev, output, *, model, select=NO_SELECTION, parallel=None,
            cache_dir=None):
    """Read or build in this process what every year of a run shares: the thresholds and the cell masks of both grids.

    Done once before the years go to worker processes, which then get these (see `hcwf.parallel.snapshot`)
    instead of each building them, and whose messages are logged once. The first of the evaluation years picks the
    NetCDF files whose grids are looked at. Thresholds are loaded as the years would load them; the masks are best
    effort (a grid that cannot be looked at is left to the workers, which raise the real error if there is one).

    Parameters
    ----------
    onset : hcwf.config.onset.OnsetSetup
        The onset definition.
    obs : hcwf.config.model.ObsSpec
        The observation dataset settings; with ``obs.units`` set no mask is built.
    area : hcwf.config.model.AreaSpec
        The region and mask settings.
    options : hcwf.config.model.OnsetOptions
        The onset options.
    ev : hcwf.config.model.EvaluationSpec
        The evaluation settings (years, probabilistic or not).
    output : hcwf.config.model.OutputSpec
        The output settings (``figure_dir``).
    model : hcwf.config.model.ModelSpec
        The forecast source whose grid gets a mask (and whose forecasts the thresholds may come from).
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.
    parallel : hcwf.config.model.ParallelSpec, optional
        The parallel settings, passed on to the threshold loading.
    cache_dir : str or pathlib.Path, optional
        The cache folder, passed on to the threshold loading.

    Returns
    -------
    None
        Nothing; the thresholds and masks are left in the caches of this process.
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
        lazy = dim_fmt(open_store(store) if store is not None else open_data(path))
        try:
            _warm_masks(lazy, obs.var, str(obs.name or "obs"), area, select, dir_fig, "observations")
        finally:
            if path is not None:
                lazy.close()
    except Exception as exc:  # noqa: BLE001 - only a warm-up
        log.debug(f"obs mask pre-warm skipped: {exc}")
    try:
        fix = dim_fmt_model_ensemble if ev.probabilistic else dim_fmt_model
        if select.model_paths:
            path = year_files(select.model_paths, model.dir, select.model_files_years, year)[0]
            lazy = fix(open_data(path), model.dims_map())
        elif is_zarr(model.dir):
            lazy = fix(open_store(model.dir), model.dims_map())
            path = None
        else:
            path = os.path.join(model.dir, model.file_pattern.format(year))
            lazy = fix(open_data(path), model.dims_map())
        try:
            _warm_masks(lazy, model.var, str(model.name or "model"), area, select, dir_fig, model.name)
        finally:
            if path is not None:
                lazy.close()
    except Exception as exc:  # noqa: BLE001 - only a warm-up (a climatology source has no forecast files)
        log.debug(f"model mask pre-warm skipped: {exc}")


def warm_up_of(rc, model, select=NO_SELECTION):
    """Call `warm_up` with the settings of a run for the source ``model``.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run configuration.
    model : hcwf.config.model.ModelSpec
        The forecast source.
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.

    Returns
    -------
    None
        Nothing, as `warm_up`.
    """
    return warm_up(rc.onset, rc.observations, rc.area, rc.onset_options, rc.evaluation, rc.output, model=model,
                   select=select, parallel=rc.parallel, cache_dir=rc.cache_dir)
