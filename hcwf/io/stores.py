"""Where the data is: the zarr stores, the NetCDF files of a year, and how a forecast file is opened.

The files of a year come from ``file_pattern`` (one per year, ``"{}"`` for the year), a zarr store of all years, or an
explicit list of files (``paths``, any number per year, optionally filtered by ``years``): see `year_files`. Nothing here
decodes data. The readers `hcwf.io.observations` and `hcwf.io.forecasts` use these functions to find what to read,
and `hcwf.io.dims` supplies the dimension standardisation applied when a forecast is opened.
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
    """Open a zarr store lazily and keep it open for the run.

    The store is opened with ``chunks=None``: nothing is decoded until a selection is loaded. The handle is kept per
    path in a run table and closed when the run's state is reset.

    Parameters
    ----------
    path : str or pathlib.Path
        The zarr store folder.

    Returns
    -------
    xarray.Dataset
        The lazy dataset of the store, shared by every caller with the same path.
    """
    key = str(path)
    return _stores.get_or_make(key, lambda: xr.open_zarr(key, chunks=None))


def year_files(paths, base, years_filter, year) -> list[str]:
    """Return the files of ``year`` among those a dataset lists in ``paths``.

    The year of a file is the first 19xx or 20xx in its name (the rule of onset_blending); ``years_filter`` keeps only
    the files of those years first. Any number of files may fall in a year (live files hold one init each); they
    come in the order `hcwf.config.datasets.resolve_paths` gives, by year and name.

    Parameters
    ----------
    paths : list of str
        File names or globs; relative ones are taken under ``base``.
    base : str or pathlib.Path, optional
        The folder relative entries of ``paths`` are under; None for the current directory.
    years_filter : collection of int, optional
        Keep only the files of these years before picking; None keeps all.
    year : int
        The year to select.

    Returns
    -------
    list of str
        The paths of the files whose year is ``year``, possibly empty.

    Raises
    ------
    FileNotFoundError
        ``paths`` cannot be resolved (the message of the underlying `hcwf.errors.DatasetError`).
    """
    try:
        files = resolve_paths(paths, base, years_filter)
    except DatasetError as exc:
        raise FileNotFoundError(str(exc)) from None
    return [str(f) for f in files if file_year(f) == year]


def observation_source(year, obs_dir, obs_file_pattern, paths=None, years_filter=None):
    """Find where the year's observations are.

    With ``paths`` (see `year_files`) the year's listed files are the source: one path, or a tuple of paths when
    several files hold the year. Otherwise a zarr store ``obs_dir`` holds all years, else the first of the
    patterns that exists in ``obs_dir`` is used.

    Parameters
    ----------
    year : int
        The year to find.
    obs_dir : str or pathlib.Path
        The observation folder, or a zarr store.
    obs_file_pattern : list of str
        File name patterns with ``{}`` for the year, tried in order.
    paths : list of str, optional
        An explicit list of files or globs, which replaces the patterns.
    years_filter : collection of int, optional
        The years kept among ``paths``.

    Returns
    -------
    tuple of (str or None, str or tuple of str or None)
        ``(store path, None)`` for a zarr store, or ``(None, file path)`` for NetCDF (a tuple of paths when
        several files of ``paths`` hold the year).

    Raises
    ------
    FileNotFoundError
        No file holds the year (the message lists what was tried or what is available).
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
    """Find what holds the year's forecasts: ``[None]`` for a zarr store (all years), else the file paths.

    One file, ``file_pattern.format(year)`` in ``model_dir``, or the listed ``paths`` (see `year_files`) of the year, one
    init or many in each.

    Parameters
    ----------
    year : int
        The year to find.
    model_dir : str or pathlib.Path
        The forecast folder, or a zarr store.
    file_pattern : str
        The file name with ``{}`` for the year.
    paths : list of str, optional
        An explicit list of files or globs, which replaces the pattern.
    years_filter : collection of int, optional
        The years kept among ``paths``.

    Returns
    -------
    list
        ``[None]`` for a zarr store, else the file paths of the year.

    Raises
    ------
    FileNotFoundError
        The year is absent.
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
    """Return the function that standardises a forecast's dimension names.

    Parameters
    ----------
    ensemble : bool, optional
        True when members are required (`hcwf.io.dims.dim_fmt_model_ensemble`), False when absent
        (`hcwf.io.dims.dim_fmt_model`), None when optional (the units pipeline reads files with or without them).

    Returns
    -------
    callable
        ``fix(ds, model_dims)`` returning the dataset with canonical dimension names.
    """
    if ensemble is None:
        def optional_member(ds, model_dims=None):
            """Standardise the names and rename a member dimension if there is one."""
            ds = dim_fmt_model(ds, model_dims)
            member = find_dim(list(ds.coords), "member", fuzzy=True)
            return ds if member in (None, "member") else ds.rename({member: "member"})
        return optional_member
    return dim_fmt_model_ensemble if ensemble else dim_fmt_model


def open_forecast(year, model_dir, file_pattern, ensemble, model_dims, filtered_dates_yr, file_path=None, multi=False):
    """Open the year's forecasts lazily and select its inits.

    A ``model_dir`` that is a zarr store holds all years: the inits of the year are picked by time. Otherwise the
    year's NetCDF file ``file_pattern.format(year)``, or ``file_path`` when the caller names it, is opened. With
    ``multi`` (one of several files of the year) a file without a wanted init gives ``(None, None)`` instead of an
    error.

    Parameters
    ----------
    year : int
        The year to open.
    model_dir : str or pathlib.Path
        The forecast folder, or a zarr store.
    file_pattern : str
        The file name with ``{}`` for the year.
    ensemble : bool, optional
        Whether members are required, absent or optional (see `fix_dims_for`).
    model_dims : dict of str to str, optional
        Names used in the file for canonical dimensions, e.g. ``{"init_time": "time"}``.
    filtered_dates_yr : collection of datetime-like, optional
        The init times wanted; None keeps every init of the file (the units pipeline reads whole files).
    file_path : str, optional
        The NetCDF file to open instead of the pattern's.
    multi : bool, default False
        Whether the file is one of several of the year.

    Returns
    -------
    tuple of (xarray.Dataset or None, callable or None)
        The dataset (standardised dimension names, restricted to the wanted inits) and a function that closes it
        (a no-op for the shared zarr store). Both None for ``multi`` without a wanted init.

    Raises
    ------
    FileNotFoundError
        The year's file is absent, or a zarr store has no forecasts of the year.
    ValueError
        The file has no init time among ``filtered_dates_yr``.
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
