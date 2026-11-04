"""Where the data is: the zarr stores, the NetCDF files of a year, and how a forecast file is opened.

The files of a year come from ``file_pattern`` (one per year, ``"{}"`` for the year), a zarr store of all years, or an
explicit list of files (``paths``, any number per year, optionally filtered by ``years``): see `year_files`. Nothing here
decodes data. The readers `hcwf.io.observations` and `hcwf.io.forecasts` use these functions to find what to read,
and `hcwf.io.dims` supplies the dimension standardisation applied when a forecast is opened.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path

import pandas as pd
import xarray as xr

from hcwf.cache import memo
from hcwf.config.datasets import file_year, is_zarr, resolve_paths
from hcwf.io.dims import dim_fmt, dim_fmt_model, dim_fmt_model_ensemble, find_dim
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


def is_data_file(name) -> bool:
    """Tell whether a file name is a NetCDF file or a zarr store (the one test of what counts as a data file).

    Parameters
    ----------
    name : str or pathlib.Path
        The file name; only its text is read.

    Returns
    -------
    bool
        True for a name holding ``.nc`` or ending in ``.zarr``.
    """
    name = str(name)
    return ".nc" in name or name.endswith(".zarr")


def data_files(folder) -> list[Path]:
    """Return the NetCDF files and zarr stores directly in a folder, sorted by name.

    Parameters
    ----------
    folder : str or pathlib.Path
        The folder to list.

    Returns
    -------
    list of pathlib.Path
        The data files (see `is_data_file`); an unreadable folder raises `OSError`.
    """
    return sorted(p for p in Path(folder).iterdir() if is_data_file(p.name))


def pattern_files(folder, pattern: str) -> list[tuple]:
    """Return the files of a file pattern with the year each holds: the one listing of a ``{}`` pattern.

    ``{}`` stands for the year, four digits, and the rest of the name is as written, so ``{}.nc`` matches ``2010.nc``
    and not ``2010_old.nc``. A pattern without ``{}`` matches its own name (or glob) and holds no year.

    Parameters
    ----------
    folder : str or pathlib.Path
        The folder the pattern is relative to.
    pattern : str
        The file name with ``{}`` for the year.

    Returns
    -------
    list of tuple of (int or None, str)
        ``(year, path)`` of each file found, by year and then path.
    """
    base = glob.escape(str(folder))
    if "{}" not in pattern:
        return [(None, f) for f in sorted(glob.glob(os.path.join(base, pattern)))]
    named = re.compile(re.escape(pattern).replace(re.escape("{}"), r"(\d{4})", 1))
    found = []
    for path in glob.glob(os.path.join(base, pattern.replace("{}", "[0-9][0-9][0-9][0-9]"))):
        match = named.fullmatch(os.path.relpath(path, str(folder)))
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def years_in_folder(folder, pattern: str) -> tuple:
    """Return the years of the files in a folder that match a file pattern, sorted.

    Parameters
    ----------
    folder : str or pathlib.Path
        The folder to look in.
    pattern : str
        The file pattern, ``{}`` standing for the year (see `pattern_files`).

    Returns
    -------
    tuple of int
        The years found, increasing.
    """
    return tuple(sorted({year for year, _ in pattern_files(folder, pattern) if year is not None}))


def listed_years(paths, base, years_filter) -> tuple:
    """Return the years of the files a dataset lists in ``paths``, sorted.

    Parameters
    ----------
    paths : sequence of str
        The listed files or globs, as in the dataset specification.
    base : str, optional
        The folder relative paths are resolved against.
    years_filter : sequence of int, optional
        Keep only the files of these years.

    Returns
    -------
    tuple of int
        The years found, increasing.

    Raises
    ------
    ValueError
        The files cannot be resolved (the message of the `hcwf.errors.DatasetError`).
    """
    try:
        files = resolve_paths(paths, base, years_filter)
    except DatasetError as exc:
        raise ValueError(str(exc)) from None
    return tuple(sorted({y for y in map(file_year, files) if y is not None}))


def store_years(path, forecast: bool = False, dims=None) -> tuple:
    """Return the years a zarr store holds, from its time coordinate (observations) or its init times (forecasts).

    Parameters
    ----------
    path : str or pathlib.Path
        The zarr store.
    forecast : bool, default False
        Read the ``init_time`` of a forecast store (named as `hcwf.io.dims.dim_fmt_model` does) instead of the ``time``
        of an observation store.
    dims : dict of str to str, optional
        Names used in the file for canonical dimensions (see `hcwf.io.dims.rename_model_dims`); forecasts only.

    Returns
    -------
    tuple of int
        The distinct years, increasing.

    Raises
    ------
    ValueError
        The store has no time coordinate (no init time for ``forecast``).
    """
    coord = "init_time" if forecast else "time"
    try:
        ds = dim_fmt_model(open_store(path), dims) if forecast else dim_fmt(open_store(path))
    except (IndexError, KeyError, ValueError):
        ds = None
    if ds is None or coord not in ds.coords:
        raise ValueError(f"{path}: the zarr store has no {'init time' if forecast else 'time'} coordinate")
    return tuple(sorted({int(y) for y in pd.DatetimeIndex(ds[coord].values).year.unique()}))


def dataset_years(folder, paths=None, years_filter=None, file_pattern=None, *, strict: bool = False) -> tuple:
    """Return the years a dataset holds, however it is given: listed files, a zarr store or a folder of yearly files.

    Parameters
    ----------
    folder : str or pathlib.Path
        The folder of the files (the base of relative ``paths``), or a zarr store.
    paths : sequence of str, optional
        The listed files or globs (see `year_files`); they replace the store and the pattern.
    years_filter : collection of int, optional
        The years kept among ``paths``.
    file_pattern : str or sequence of str, optional
        The file name pattern(s), ``{}`` for the year; the years found under any of them are returned. None means
        ``"{}.nc"``.
    strict : bool, default False
        Raise when a listed file has no year in its name, instead of leaving it out.

    Returns
    -------
    tuple of int
        The years, increasing.

    Raises
    ------
    ValueError
        A listed file cannot be resolved, or (``strict``) has no year in its name, or a store has no time coordinate.
    """
    if paths:
        years = listed_years(paths, folder or None, years_filter)
        if strict:
            nameless = [str(f) for f in resolve_paths(paths, folder or None, years_filter) if file_year(f) is None]
            if nameless:
                raise ValueError("a file name has no year (19xx or 20xx); files are read a year at a time: "
                                 + ", ".join(nameless))
        return years
    if is_zarr(folder):
        return store_years(folder)
    patterns = (file_pattern or "{}.nc",) if file_pattern is None or isinstance(file_pattern, str) else tuple(file_pattern)
    return tuple(sorted({y for pattern in patterns for y in years_in_folder(folder, pattern)}))


def model_sources(folder, paths=None, file_pattern=None) -> list[str]:
    """Return everything a model's forecasts are read from: the listed files or stores, a zarr store, or the pattern's files.

    A source is a NetCDF file or a zarr store (`hcwf.io.forecasts.UnitForecast` opens either); a store is one source
    holding every year.

    Parameters
    ----------
    folder : str or pathlib.Path
        The folder of the yearly files, or a zarr store.
    paths : sequence of str, optional
        The listed files (or stores), already resolved; they replace the store and the pattern.
    file_pattern : str, optional
        The file name with ``{}`` for the year; None means ``"{}.nc"``.

    Returns
    -------
    list of str
        The sources, by year in the name and then path.

    Raises
    ------
    FileNotFoundError
        Nothing is found in the folder.
    """
    if paths:
        return [str(p) for p in paths]
    if is_zarr(folder):
        return [str(folder)]
    pattern = file_pattern or "{}.nc"
    files = [path for year, path in pattern_files(folder, pattern) if year is not None]
    if not files:
        raise FileNotFoundError(f"No forecast file of the pattern {pattern!r} in {folder}")
    return files


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
