import pandas as pd

def dim_fmt(ds):
    """Standardize dimension names"""
    coord_list = list(ds.coords.keys())

    if "lon" not in coord_list:
        #print("lon NOT in coords  --> ")  # , model_name)
        lat_coords = [variable for variable in coord_list if "lat" in variable.lower()][0]
        lon_coords = [variable for variable in coord_list if "lon" in variable.lower()][0]

        ds = ds.rename({lat_coords: "lat", lon_coords: "lon"})

#    if "time" not in coord_list:
#        print("time NOT in coords --> ")  # , model_name)
#        time_coords = [variable for variable in coord_list if "TIME" in variable][0]
#        ds = ds.rename({time_coords: "time"})

    if set(ds.dims) == {"lat", "lon"} and len(ds.dims) == 2:
        return ds

    if "time" not in coord_list:
        keywords = ["time", 'date']
        time_coords = [
            variable
            for variable in coord_list
            if any(keyword in variable.lower() for keyword in keywords)
        ][0]
        ds = ds.rename({time_coords: "time"})

    return ds


def rename_model_dims(ds, model_dims=None):
    """Rename coordinates the caller named explicitly to ROMP's canonical names.

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
    ds = rename_model_dims(ds, model_dims)
    coord_list = list(ds.coords.keys())

    if "lon" not in coord_list:
        #print("lon NOT in coords  --> ")  # , model_name)
        lat_coords = [variable for variable in coord_list if "lat" in variable.lower()][0]
        lon_coords = [variable for variable in coord_list if "lon" in variable.lower()][0]

        ds = ds.rename({lat_coords: "lat", lon_coords: "lon"})
        coord_list = list(ds.coords.keys())

    if "init_time" not in coord_list:
        #print("init_time NOT in coords --> ")  # , model_name)
        ds = ds.rename({_init_time_coord(ds, coord_list): "init_time"})
        coord_list = list(ds.coords.keys())

    if "step" not in coord_list:
        keywords = ["day", "prediction_timedelta"]
        step_coords = [
            variable
            for variable in coord_list
            if any(keyword in variable.lower() for keyword in keywords)
        ][0]
        ds = ds.rename({step_coords: "step"})

    # convert TimedeltaIndex to integer (days)
    if isinstance(ds.indexes["step"], pd.TimedeltaIndex):
        ds = ds.assign_coords(step=ds.step.dt.days)

    return ds


def dim_fmt_model_ensemble(ds, model_dims=None):
    """Standardize dimension names for probabilistic reforecast model data"""

    ds = dim_fmt_model(ds, model_dims)

    coord_list = list(ds.coords.keys())

    if "member" not in coord_list:
        keywords = ["number", "sample"]
        ensemble_coords = [
            variable
            for variable in coord_list
            if any(keyword in variable.lower() for keyword in keywords)
        ][0]
        ds = ds.rename({ensemble_coords: "member"})

    return ds



