"""Where the data is: the zarr stores, the NetCDF files of a year, and how a forecast file is opened.

The files of a year come from `file_pattern` (one per year, "{}" for the year), a zarr store of all years, or an explicit list of
files (`paths`, any number per year, optionally filtered by `years`): see `year_files`. Nothing here decodes data.
"""
from __future__ import annotations

import logging
import os

import pandas as pd
import xarray as xr

from hcwf.cache import memo
from hcwf.config.datasets import file_year, is_zarr, resolve_paths
from hcwf.io.dims import dim_fmt_model, dim_fmt_model_ensemble, find_dim
from hcwf.errors import DatasetError

log = logging.getLogger(__name__)

_stores = memo("io.stores.open", on_clear=lambda dataset: dataset.close())
"""The open zarr stores of the run, by path; closed when the run's state is reset."""


def open_store(path):
    """The lazy Dataset of a zarr store (chunks=None: nothing is decoded until a selection is loaded), kept open per path
    for the run."""
    key = str(path)
    return _stores.get_or_make(key, lambda: xr.open_zarr(key, chunks=None))


def year_files(paths, base, years_filter, year) -> list[str]:
    """The files of `year` among those a dataset lists in `paths` (names or globs, relative ones under `base`).

    The year of a file is the first 19xx or 20xx in its name (the rule of onset_blending); `years_filter` keeps only
    the files of those years first. Any number of files may fall in a year (live files hold one init each); they
    come in the order resolve_paths gives, by year and name.
    """
    try:
        files = resolve_paths(paths, base, years_filter)
    except DatasetError as exc:
        raise FileNotFoundError(str(exc)) from None
    return [str(f) for f in files if file_year(f) == year]


def observation_source(year, obs_dir, obs_file_pattern, paths=None, years_filter=None):
    """Where the year's observations are: (zarr store path, None) or (None, NetCDF file path); FileNotFoundError if absent.

    With `paths` (see year_files) the year's listed files are the source: one path, or a tuple of paths when
    several files hold the year.
    """
    if paths:
        found = year_files(paths, obs_dir or None, years_filter, year)
        if not found:
            raise FileNotFoundError(f"No observation file found for year {year} among the listed paths {list(paths)}")
        return None, found[0] if len(found) == 1 else tuple(found)
    if is_zarr(obs_dir):
        return str(obs_dir), None
    file_patterns = [p.format(year) for p in obs_file_pattern]

    for pattern in file_patterns:
        test_path = f"{obs_dir}/{pattern}"
        if os.path.exists(test_path):
            return None, test_path

    available_files = [f for f in os.listdir(obs_dir) if f.endswith('.nc')]
    raise FileNotFoundError(
        f"No observation file found for year {year} in {obs_dir}. "
        f"Tried patterns: {file_patterns}. "
        f"Available files: {available_files}"
    )


def forecast_sources(year, model_dir, file_pattern, paths=None, years_filter=None) -> list:
    """What holds the year's forecasts: [None] for a zarr store (all years), else the file paths.

    One file, file_pattern.format(year) in model_dir, or the listed `paths` (see year_files) of the year, one init
    or many in each. Raises FileNotFoundError when the year is absent.
    """
    if paths:
        found = year_files(paths, model_dir or None, years_filter, year)
        if not found:
            raise FileNotFoundError(f"No forecast file found for year {year} among the listed paths {list(paths)}")
        return found
    if is_zarr(model_dir):
        return [None]
    file_path = os.path.join(model_dir, file_pattern.format(year))
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return [file_path]


def fix_dims_for(ensemble):
    """The function that standardises a forecast's dimension names: members required (True), absent (False) or optional
    (None, as the units pipeline reads files with or without them)."""
    if ensemble is None:
        def optional_member(ds, model_dims=None):
            ds = dim_fmt_model(ds, model_dims)
            member = find_dim(list(ds.coords), "member", fuzzy=True)
            return ds if member in (None, "member") else ds.rename({member: "member"})
        return optional_member
    return dim_fmt_model_ensemble if ensemble else dim_fmt_model


def open_forecast(year, model_dir, file_pattern, ensemble, model_dims, filtered_dates_yr, file_path=None, multi=False):
    """Open the year's forecasts lazily and select its inits; returns (Dataset, close function).

    filtered_dates_yr None keeps every init of the file (the units pipeline reads whole files).

    A model_dir that is a zarr store holds all years: the inits of the year are picked by time. Otherwise the
    year's NetCDF file file_pattern.format(year), or `file_path` when the caller names it, is opened. Raises
    FileNotFoundError when the year is absent. With `multi` (one of several files of the year) a file without a
    wanted init gives None instead of an error.
    """
    fix = fix_dims_for(ensemble)
    if file_path is None and is_zarr(model_dir):
        ds = fix(open_store(model_dir), model_dims)
        close = lambda: None  # noqa: E731 - the store handle is shared
    else:
        file_path = file_path or os.path.join(model_dir, file_pattern.format(year))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ds = fix(xr.open_dataset(file_path), model_dims)
        close = ds.close
    if filtered_dates_yr is None:
        return ds, close
    available_init_times = pd.to_datetime(ds.init_time.values)
    matching_times = available_init_times[available_init_times.isin(filtered_dates_yr)]
    if len(matching_times) == 0:
        if multi:
            close()
            return None, None
        if is_zarr(model_dir) and file_path is None and not (available_init_times.year == year).any():
            raise FileNotFoundError(f"No forecasts of {year} in {model_dir}")
        raise ValueError(f"No matching initialization times found for year {year}")
    return ds.sel(init_time=matching_times), close
