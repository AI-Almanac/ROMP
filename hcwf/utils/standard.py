"""The one table of dimension names, and the helpers that rename a dataset's coordinates to HCWF's canonical names.

Canonical names: lat, lon, time (observations), init_time, step, member (forecasts) and unit (a string dimension of
pre-aggregated units such as admin units). `ROLE_ALIASES` is the only place the names a file may use for each role are
listed; hcwf.onset.ingest, hcwf.lib.dataset_detect and the dim_fmt functions below all read it.
"""
import pandas as pd

ROLE_ALIASES = {
    "time": ("time", "valid_time", "date"),
    "init_time": ("init_time", "time", "forecast_reference_time", "initial_time"),
    "step": ("lead", "step", "prediction_timedelta", "prediction_timedelta_daily", "day", "lead_day", "lead_days"),
    "member": ("member", "number", "ensemble", "realization", "sample", "members", "ensemble_member"),
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "longitude", "x"),
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
    """The names (in the order given) that can play `role`: an alias, else (fuzzy) one holding a keyword of the role.

    Exact aliases are compared in lower case and returned in the table's order of preference. `exclude` lists names
    taken by other roles. For the step role a coordinate whose units attribute is days also counts, when no name
    does (`coords` maps a name to its attributes, e.g. ds.coords).
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
    """The first name that can play `role` (see `matches`), or None."""
    hits = matches(names, role, exclude=exclude, fuzzy=fuzzy, coords=coords)
    return hits[0] if hits else None


def _rename_unit(ds):
    """Rename a string unit dimension (adm3_name, adm3) to "unit" and give it its labels as a clean string coordinate."""
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


def _rename_spatial(ds):
    """Rename the latitude and longitude coordinates to lat and lon (not needed for a unit dimension)."""
    coord_list = list(ds.coords.keys())
    if "lon" not in coord_list and "unit" not in ds.dims:
        lat = find_dim(coord_list, "lat", fuzzy=True)
        lon = find_dim(coord_list, "lon", fuzzy=True)
        if lat is None or lon is None:
            raise IndexError("list index out of range")
        ds = ds.rename({lat: "lat", lon: "lon"})
    return ds


def dim_fmt(ds):
    """Standardize dimension names"""
    ds = _rename_spatial(_rename_unit(ds))
    coord_list = list(ds.coords.keys())

    if set(ds.dims) == {"lat", "lon"} and len(ds.dims) == 2:
        return ds

    if "time" not in coord_list:
        ds = ds.rename({find_dim(coord_list, "time", fuzzy=True): "time"})

    return ds


def rename_model_dims(ds, model_dims=None):
    """Rename coordinates the caller named explicitly to HCWF's canonical names.

    `model_dims` maps canonical names ("init_time", "step", "member", "lat",
    "lon") to the names used in the file, e.g. {"init_time": "time"}.
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
    """Prefer a datetime coordinate: lead-time names like "prediction_timedelta" also contain "time"."""
    time_coords = [variable for variable in coord_list if "time" in variable.lower()]
    datetime_coords = [
        variable for variable in time_coords if pd.api.types.is_datetime64_any_dtype(ds[variable])
    ]
    return (datetime_coords or time_coords)[0]


def dim_fmt_model(ds, model_dims=None):
    """Standardize dimension names for deterministic reforecast model data"""
    ds = _rename_spatial(_rename_unit(rename_model_dims(ds, model_dims)))
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
    """Standardize dimension names for probabilistic reforecast model data"""

    ds = dim_fmt_model(ds, model_dims)

    coord_list = list(ds.coords.keys())

    if "member" not in coord_list:
        member = find_dim(coord_list, "member", fuzzy=True)
        ds = ds.rename({member: "member"})

    return ds
