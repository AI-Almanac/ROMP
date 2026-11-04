"""The one table of dimension names, and the helpers that rename a dataset's coordinates to HCWF's canonical names.

Canonical names: lat, lon, time (observations), init_time, step, member (forecasts) and unit (a string dimension of
pre-aggregated units such as admin units). `ROLE_ALIASES` is the only place the names a file may use for each role are
listed; `hcwf.onset.ingest`, `hcwf.config.detect` and the ``dim_fmt`` functions below all read it. The readers in
`hcwf.io.observations` and `hcwf.io.forecasts` apply the ``dim_fmt`` functions to every dataset they open.
"""
import pandas as pd

ROLE_ALIASES = {
    "time": ("time", "valid_time", "date"),
    "init_time": ("init_time", "time", "forecast_reference_time", "initial_time"),
    "step": ("lead", "step", "prediction_timedelta", "prediction_timedelta_daily", "day", "lead_day", "lead_days"),
    "member": ("member", "number", "ensemble", "realization", "sample", "members", "ensemble_member"),
    "lat": ("lat", "latitude", "y", "nav_lat", "lats", "center_lat"),
    "lon": ("lon", "longitude", "x", "nav_lon", "lons", "long", "center_lon"),
    "unit": ("adm3_name", "adm3"),
}
"""Names (compared in lower case, in this order of preference) a dimension may have for each role."""
ROLE_KEYWORDS = {
    "time": ("time", "date"),
    "step": ("day", "prediction_timedelta", "lead", "step", "timedelta"),
    "member": ("number", "sample"),
    "lat": ("lat",),
    "lon": ("lon",),
}
"""Name fragments tried, in the order the file lists its names, when no alias of the role is present."""
_DAY_UNITS = ("day", "days")


def matches(names, role, *, exclude=(), fuzzy=False, coords=None):
    """Return the names that can play ``role``: an alias, else (if ``fuzzy``) one holding a keyword of the role.

    Exact aliases are compared in lower case and returned in the table's order of preference. ``exclude`` lists names
    taken by other roles. For the step role a coordinate whose units attribute is days also counts, when no name
    does.

    Parameters
    ----------
    names : iterable of str
        The candidate names, e.g. the coordinates of a dataset.
    role : {"time", "init_time", "step", "member", "lat", "lon", "unit"}
        The role to find a name for (a key of `ROLE_ALIASES`).
    exclude : iterable of str, default ()
        Names taken by other roles, left out.
    fuzzy : bool, default False
        Also accept a name holding a keyword of the role (`ROLE_KEYWORDS`) when no alias is present.
    coords : mapping, optional
        Maps a name to an object with an ``attrs`` dict, e.g. ``ds.coords``; used for the step role to accept a
        coordinate whose ``units`` attribute is days.

    Returns
    -------
    list of str
        The matching names, best first; empty if none.
    """
    names = [n for n in names if n not in exclude]
    lowered = {str(n).lower(): n for n in reversed(names)}
    hits = [lowered[a] for a in ROLE_ALIASES[role] if a in lowered]
    if not hits and fuzzy:
        hits = [n for n in names if any(k in str(n).lower() for k in ROLE_KEYWORDS.get(role, ()))]
    if not hits and role == "step" and coords is not None:
        hits = [n for n in names if n in coords
                and str(getattr(coords[n], "attrs", {}).get("units", "")).strip().lower() in _DAY_UNITS]
    return hits


def find_dim(names, role, *, exclude=(), fuzzy=False, coords=None):
    """Return the first name that can play ``role``, or None.

    Parameters
    ----------
    names : iterable of str
        The candidate names.
    role : str
        The role to find (see `matches`).
    exclude : iterable of str, default ()
        Names taken by other roles.
    fuzzy : bool, default False
        Also accept keyword matches (see `matches`).
    coords : mapping, optional
        Coordinate attributes for the step role (see `matches`).

    Returns
    -------
    str or None
        The best match, or None when no name fits.
    """
    hits = matches(names, role, exclude=exclude, fuzzy=fuzzy, coords=coords)
    return hits[0] if hits else None


def _rename_unit(ds):
    """Rename a string unit dimension (``adm3_name``, ``adm3``) to ``unit`` and give it clean string labels.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset; returned unchanged when it has no unit dimension.

    Returns
    -------
    xarray.Dataset
        The dataset with a ``unit`` dimension whose coordinate holds the stripped, decoded labels (taken from an
        ``adm3_name`` or ``adm3`` variable if the dimension had no coordinate).
    """
    name = find_dim(list(ds.dims), "unit")
    if name is None:
        return ds
    if name != "unit":
        ds = ds.rename({name: "unit"})
    if "unit" not in ds.coords:
        for var in ("adm3_name", "ADM3_NAME", "Adm3_Name", "adm3"):
            if var in ds.variables and ds[var].dims == ("unit",):
                ds = ds.assign_coords(unit=ds[var].values)
                break
    if "unit" in ds.coords:
        labels = ds["unit"].values
        decoded = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in labels.tolist()]
        ds = ds.assign_coords(unit=[x.strip() for x in decoded])
    return ds


def rename_spatial(obj, *, fuzzy=False, required=False):
    """Rename the latitude and longitude names of a dataset or array to ``lat`` and ``lon``: the one implementation.

    Each of the two is renamed when it is not already present, from the aliases of `ROLE_ALIASES` (the dimensions and
    coordinates are searched). A ``unit`` dimension stands in for both and nothing is renamed.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The data.
    fuzzy : bool, default False
        Also accept a name holding ``lat`` or ``lon`` when no alias is present (`ROLE_KEYWORDS`).
    required : bool, default False
        Raise when no latitude or no longitude name is found (unless there is a ``unit`` dimension).

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The data with ``lat`` and ``lon`` names; ``obj`` itself when there is nothing to rename.

    Raises
    ------
    IndexError
        ``required`` and no latitude or no longitude name is found.
    """
    if "unit" in obj.dims:
        return obj
    names = list(dict.fromkeys([*obj.dims, *obj.coords]))
    renames = {}
    for target in ("lat", "lon"):
        if target in names:
            continue
        name = find_dim(names, target, fuzzy=fuzzy)
        if name is None:
            if required:
                raise IndexError(f"no {target} name among {sorted(map(str, names))}")
            continue
        renames[name] = target
    return obj.rename(renames) if renames else obj


def dim_fmt(ds):
    """Standardise the dimension names of an observation dataset.

    Units and spatial names are renamed; a dataset that is not a plain lat/lon grid also gets a ``time`` coordinate.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset as read from a file.

    Returns
    -------
    xarray.Dataset
        The dataset with ``lat``, ``lon`` and ``time`` (or ``unit``) names.
    """
    ds = rename_spatial(_rename_unit(ds), fuzzy=True, required=True)
    coord_list = list(ds.coords.keys())

    if set(ds.dims) == {"lat", "lon"} and len(ds.dims) == 2:
        return ds

    if "time" not in coord_list:
        ds = ds.rename({find_dim(coord_list, "time", fuzzy=True): "time"})

    return ds


def rename_model_dims(ds, model_dims=None):
    """Rename coordinates the caller named explicitly to HCWF's canonical names.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset.
    model_dims : dict of str to str, optional
        Maps canonical names (``"init_time"``, ``"step"``, ``"member"``, ``"lat"``, ``"lon"``) to the names used in
        the file, e.g. ``{"init_time": "time"}``; None or empty renames nothing.

    Returns
    -------
    xarray.Dataset
        The renamed dataset.

    Raises
    ------
    ValueError
        A name in ``model_dims`` is not a variable of the dataset.
    """
    if not model_dims:
        return ds
    rename = {source: canonical for canonical, source in model_dims.items() if source != canonical}
    missing = sorted(source for source in rename if source not in ds.variables)
    if missing:
        raise ValueError(
            f"model_dims names {missing} that are not in the dataset; "
            f"available: {sorted(map(str, ds.variables))}"
        )
    return ds.rename(rename)


def _init_time_coord(ds, coord_list):
    """Pick the coordinate to call ``init_time``, preferring a datetime one.

    Lead-time names like ``prediction_timedelta`` also contain ``time``, so datetime coordinates come first.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset.
    coord_list : list of str
        The coordinate names of ``ds``.

    Returns
    -------
    str
        The first datetime coordinate whose name contains ``time``, else the first name containing it (an
        `IndexError` follows if no name contains ``time``).
    """
    time_coords = [variable for variable in coord_list if "time" in variable.lower()]
    datetime_coords = [
        variable for variable in time_coords if pd.api.types.is_datetime64_any_dtype(ds[variable])
    ]
    return (datetime_coords or time_coords)[0]


def dim_fmt_model(ds, model_dims=None):
    """Standardise the dimension names of deterministic reforecast model data.

    Renames to ``lat``, ``lon``, ``init_time`` and ``step``; a timedelta step is converted to integer days.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset as read from a file.
    model_dims : dict of str to str, optional
        Explicit names used in the file (see `rename_model_dims`).

    Returns
    -------
    xarray.Dataset
        The dataset with canonical dimension names; ``step`` holds integer days.
    """
    ds = rename_spatial(_rename_unit(rename_model_dims(ds, model_dims)), fuzzy=True, required=True)
    coord_list = list(ds.coords.keys())

    if "init_time" not in coord_list:
        ds = ds.rename({_init_time_coord(ds, coord_list): "init_time"})
        coord_list = list(ds.coords.keys())

    if "step" not in coord_list:
        step = find_dim(coord_list, "step", fuzzy=True, coords=ds.coords)
        ds = ds.rename({step: "step"})

    # convert TimedeltaIndex to integer (days)
    if isinstance(ds.indexes["step"], pd.TimedeltaIndex):
        ds = ds.assign_coords(step=ds.step.dt.days)

    return ds


def dim_fmt_model_ensemble(ds, model_dims=None):
    """Standardise the dimension names of probabilistic reforecast model data.

    As `dim_fmt_model`, and a ``member`` dimension is required.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset as read from a file.
    model_dims : dict of str to str, optional
        Explicit names used in the file (see `rename_model_dims`).

    Returns
    -------
    xarray.Dataset
        The dataset with canonical dimension names including ``member``.
    """
    ds = dim_fmt_model(ds, model_dims)

    coord_list = list(ds.coords.keys())

    if "member" not in coord_list:
        member = find_dim(coord_list, "member", fuzzy=True)
        ds = ds.rename({member: "member"})

    return ds
