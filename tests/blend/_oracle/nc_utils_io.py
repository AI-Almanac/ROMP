"""TEST-ONLY ORACLE: functions copied unchanged from onset_blending@1e6d696 tests/_legacy/nc_utils.py (the legacy
table readers of observations and forecasts and the cell transforms of their tables), with only the imports
changed to the oracle copies of rain_horizon_utils, spatial_id_utils and sparse_transform_utils.
Never import this from hcwf."""
import os
import re
import warnings
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd

from .rain_horizon_utils import resolve_rain_day_max, validate_day_coordinate
from .spatial_id_utils import GridIdConvention, validate_expected_source_ids
from .sparse_transform_utils import (
    compile_sparse_cell_transform,
    sparse_observed_weighted_mean,
    sparse_target_support,
)


def rename_dimensions(df, rename_map):
    """Case-insensitive rename of column names according to rename_map (old->new)."""
    if not rename_map:
        return df
    rename = {}
    for old, new in rename_map.items():
        for col in df.columns:
            if col.lower() == old.lower():
                rename[col] = new
                break
    return df.rename(columns=rename)


def get_nc_adm3_names(ds):
    """
    Read the adm3_name string coordinate from a netCDF4 Dataset.
    Tries common capitalisation variants. Returns a list of strings.
    """
    import netCDF4 as nc4
    variants = ["adm3_name", "ADM3_NAME", "Adm3_Name", "adm3"]
    for v in variants:
        if v in ds.variables:
            var = ds.variables[v]
            # netCDF4 string variables (dtype=object / vlen str) must be
            # accessed via [:].tolist() — iterating the masked array directly
            # raises "memoryview: format O not supported".
            try:
                items = var[:].tolist()
            except (NotImplementedError, TypeError):
                # chararray fallback (fixed-length char dim)
                import netCDF4 as nc4
                items = nc4.chartostring(var[:]).tolist()
            
            names = []
            for item in items:
                if isinstance(item, (bytes, bytearray)):
                    names.append(item.decode("utf-8").strip())
                else:
                    names.append(str(item).strip())
            return names
    raise ValueError(
        "Could not find adm3_name variable/dimension in NetCDF file. "
        f"Available variables: {list(ds.variables.keys())}"
    )


def get_nc_time(ds):
    """
    Robust time getter from a netCDF4 Dataset.

    Returns dict: values, units, name, source
    """
    variants = ["TIME", "time", "Time", "t", "T"]
    for v in variants:
        if v in ds.variables:
            vals = np.array(ds[v][:]).flatten()
            units = getattr(ds[v], "units", None)
            return {"values": vals, "units": units, "name": v, "source": "var"}
    for v in variants:
        vl = v.lower()
        for name in ds.variables:
            if name.lower() == vl:
                vals = np.array(ds[name][:]).flatten()
                units = getattr(ds[name], "units", None)
                return {"values": vals, "units": units, "name": name, "source": "var_ci"}
    for v in variants:
        if v in ds.dimensions:
            if v in ds.variables:
                vals = np.array(ds[v][:]).flatten()
                units = getattr(ds[v], "units", None)
            else:
                vals = np.arange(ds.dimensions[v].size, dtype=float)
                units = None
            return {"values": vals, "units": units, "name": v, "source": "dim"}
    raise ValueError("Could not find time variable/dimension in NetCDF file")


def nc_time_to_dates(time_num, time_units):
    """
    Convert numeric NetCDF time values to pandas DatetimeSeries using CF units.
    """
    m = re.match(
        r"^\s*(seconds?|minutes?|hours?|days?)\s+since\s+(.+)\s*$",
        str(time_units),
        re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"Unrecognized NetCDF time units: {time_units}")
    unit = m.group(1).lower().rstrip("s")  # second/minute/hour/day
    origin_str = m.group(2).strip().replace("T", " ")

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            origin = datetime.strptime(origin_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Could not parse NetCDF time origin: {origin_str}")

    mult = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}[unit]
    timestamps = [origin + timedelta(seconds=float(t) * mult) for t in time_num]
    return pd.DatetimeIndex(timestamps)


def _filled_with_nan(raw):
    """``np.ma.filled(raw, np.nan)`` for an array just read from a file,
    filled in place rather than copied: a forecast variable runs to GBs, and
    nothing else holds the buffer read for it. Anything but a float masked
    array that owns its data takes the copying path."""
    if not np.ma.isMaskedArray(raw):
        return np.asarray(raw)
    data = raw.data
    if not np.issubdtype(data.dtype, np.floating) or not data.flags.writeable or not data.flags.owndata:
        return np.ma.filled(raw, np.nan)
    if raw.mask is not np.ma.nomask:
        data[np.ma.getmaskarray(raw)] = np.nan
    return data


def _row_dates(time_num, time_units):
    """``nc_time_to_dates(time_num, time_units)`` as (``.date`` objects, ``.year``)
    arrays, converting each distinct time value once: a long or wide table
    repeats each time for every cell (and member)."""
    codes, distinct = pd.factorize(np.asarray(time_num), use_na_sentinel=False)
    dates = nc_time_to_dates(distinct, time_units)
    return dates.date[codes], dates.year.to_numpy()[codes]


def _product_frame(names, values):
    """``pd.DataFrame(list(itertools.product(*values)), columns=names)`` built
    from array repeats rather than one Python tuple per row.

    Each column takes the dtype pandas infers for it from the tuples: that
    inference is per column, so it is reproduced by inferring a one-column
    frame of the dimension's own values, which is then repeated by position.
    """
    sizes = [len(v) for v in values]
    if not sizes or 0 in sizes:
        return pd.DataFrame(list(product(*values)), columns=names)
    n_rows = int(np.prod(sizes))
    columns = {}
    for i, (name, vals) in enumerate(zip(names, values)):
        inner = int(np.prod(sizes[i + 1:]))
        outer = n_rows // (sizes[i] * inner)
        column = pd.DataFrame([(v,) for v in vals]).iloc[:, 0]
        positions = np.tile(np.repeat(np.arange(sizes[i]), inner), outer)
        columns[name] = column.take(positions).reset_index(drop=True)
    return pd.DataFrame(columns)


def _prepare_transform_source_ids(df, weights_df, spec, context):
    convention_data = weights_df.attrs.get("source_id_convention")
    convention = (
        GridIdConvention.from_dict(convention_data)
        if convention_data else None
    )
    df, extra_ids, _ = validate_expected_source_ids(
        df,
        weights_df["source_id"].drop_duplicates(),
        convention=convention,
        spec=spec,
        context=context,
    )
    if extra_ids and not weights_df.attrs.get("reported_extra_source_ids"):
        print(
            f"  cell-transform source coverage: {len(extra_ids)} raw grid "
            "IDs are outside the supplied weights and will be ignored."
        )
        weights_df.attrs["reported_extra_source_ids"] = True
    return df


def _observed_weighted_mean(df, value_col, group_cols):
    """Aggregate over finite source values and renormalize observed weights."""
    values = pd.to_numeric(df[value_col], errors="coerce")
    valid = np.isfinite(values) & np.isfinite(df["weight"])
    work = df.assign(
        _weighted_value=np.where(valid, values * df["weight"], 0.0),
        _observed_weight=np.where(valid, df["weight"], 0.0),
    )
    out = work.groupby(group_cols, as_index=False, dropna=False)[
        ["_weighted_value", "_observed_weight"]
    ].sum()
    out[value_col] = out["_weighted_value"].div(out["_observed_weight"])
    out.loc[out["_observed_weight"] <= 0, value_col] = np.nan
    return out[group_cols + [value_col]]


def _get_sparse_transform(weights_df):
    transform = weights_df.attrs.get("sparse_transform")
    if transform is None:
        transform = compile_sparse_cell_transform(
            weights_df,
            source_ids=tuple(sorted(weights_df["source_id"].unique())),
            target_ids=tuple(sorted(weights_df["target_id"].unique())),
        )
        weights_df.attrs["sparse_transform"] = transform
    return transform


def _transform_forecast_wide_pandas(df, weights_df, day_cols, meta_cols):
    """Trusted pandas reference path for forecast cell transforms."""
    long = df.melt(id_vars=["id"] + meta_cols, value_vars=day_cols,
                   var_name="day_col", value_name="rain")
    long = long.merge(weights_df.rename(columns={"source_id": "id"}), on="id", how="inner")
    trans = _observed_weighted_mean(
        long, "rain", meta_cols + ["target_id", "day_col"]
    )
    trans = trans.rename(columns={"target_id": "id"})
    wide = trans.pivot(
        index=meta_cols + ["id"], columns="day_col", values="rain"
    ).reset_index()
    wide.columns.name = None
    return wide


def _transform_groundtruth_long_pandas(df, weights_df, value_col, meta_cols):
    """Trusted pandas reference path for ground-truth cell transforms."""
    x = df.merge(weights_df.rename(columns={"source_id": "id"}), on="id", how="inner")
    trans = _observed_weighted_mean(
        x, value_col, meta_cols + ["target_id"]
    )
    trans = trans.rename(columns={"target_id": "id"})
    return trans


def _transform_values_sparse(df, weights_df, value_cols, meta_cols,
                             group_chunk_size=16):
    """Transform one or more rainfall columns in bounded metadata-group chunks."""
    transform = _get_sparse_transform(weights_df)
    work = df[df["id"].isin(transform.source_index)].copy()
    work["id"] = work["id"].astype(str)
    group_id_col = "_transform_group_id"
    if meta_cols:
        work[group_id_col] = work.groupby(
            meta_cols, sort=True, dropna=False
        ).ngroup()
        group_meta = work[[group_id_col] + meta_cols].drop_duplicates(group_id_col)
    else:
        work[group_id_col] = 0
        group_meta = pd.DataFrame({group_id_col: [0]})
    group_meta = group_meta.sort_values(group_id_col).reset_index(drop=True)

    chunks = []
    group_ids = group_meta[group_id_col].to_numpy()
    n_source = len(transform.source_ids)
    n_target = len(transform.target_ids)
    n_value = len(value_cols)
    for start in range(0, len(group_ids), int(group_chunk_size)):
        chunk_ids = group_ids[start:start + int(group_chunk_size)]
        rows = work[work[group_id_col].isin(chunk_ids)]
        group_pos = rows[group_id_col].map(
            {group_id: pos for pos, group_id in enumerate(chunk_ids)}
        ).to_numpy(int)
        source_pos = rows["id"].map(transform.source_index).to_numpy(int)
        values = rows[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

        source_values = np.full((n_source, len(chunk_ids) * n_value), np.nan)
        for value_pos in range(n_value):
            output_pos = group_pos * n_value + value_pos
            source_values[source_pos, output_pos] = values[:, value_pos]
        transformed = sparse_observed_weighted_mean(transform, source_values)

        present_sources = np.zeros((n_source, len(chunk_ids)), dtype=bool)
        present_sources[source_pos, group_pos] = True
        supported = sparse_target_support(transform, present_sources)

        chunk_meta = group_meta.set_index(group_id_col).loc[chunk_ids].reset_index()
        if meta_cols:
            out = chunk_meta.loc[
                chunk_meta.index.repeat(n_target), meta_cols
            ].reset_index(drop=True)
        else:
            out = pd.DataFrame(index=np.arange(len(chunk_ids) * n_target))
        out["id"] = np.tile(transform.target_ids, len(chunk_ids))
        for value_pos, value_col in enumerate(value_cols):
            out[value_col] = transformed[:, value_pos::n_value].T.reshape(-1)
        chunks.append(out.loc[supported.T.reshape(-1)].reset_index(drop=True))

    if not chunks:
        return pd.DataFrame(columns=meta_cols + ["id"] + value_cols)
    return pd.concat(chunks, ignore_index=True)[meta_cols + ["id"] + value_cols]


def transform_forecast_wide(df, weights_df, spec=None):
    """Apply a sparse cell transform to a forecast wide-by-day DataFrame."""
    if weights_df is None:
        return df
    df = _prepare_transform_source_ids(
        df, weights_df, spec, "forecast cell transform"
    )
    day_cols = [c for c in df.columns if re.search(r"_day_\d+$", c)]
    if not day_cols:
        raise ValueError("No forecast day columns found to transform.")
    spatial_cols = {"id", "adm3_name", "lat", "lon", "latitude", "longitude"}
    meta_cols = [c for c in df.columns if c not in day_cols and c not in spatial_cols]
    relevant = df[df["id"].isin(set(weights_df["source_id"]))]
    if relevant.duplicated(meta_cols + ["id"]).any():
        warnings.warn(
            "Duplicate source IDs within a forecast metadata group; using the "
            "pandas cell-transform reference path.",
            RuntimeWarning,
        )
        return _transform_forecast_wide_pandas(df, weights_df, day_cols, meta_cols)
    out = _transform_values_sparse(df, weights_df, day_cols, meta_cols)
    return out.sort_values(
        meta_cols + ["id"], na_position="first"
    ).reset_index(drop=True)


def transform_groundtruth_long(df, weights_df, value_col, spec=None):
    """Apply a sparse cell transform to a ground-truth long DataFrame."""
    if weights_df is None:
        return df
    df = _prepare_transform_source_ids(
        df, weights_df, spec, "ground-truth cell transform"
    )
    if value_col not in df.columns:
        raise ValueError(f"Value col not found: {value_col}")
    spatial_cols = {"id", "adm3_name", "lat", "lon", "latitude", "longitude"}
    meta_cols = [c for c in df.columns if c != value_col and c not in spatial_cols]
    relevant = df[df["id"].isin(set(weights_df["source_id"]))]
    if relevant.duplicated(meta_cols + ["id"]).any():
        warnings.warn(
            "Duplicate source IDs within a ground-truth metadata group; using "
            "the pandas cell-transform reference path.",
            RuntimeWarning,
        )
        return _transform_groundtruth_long_pandas(
            df, weights_df, value_col, meta_cols
        )
    return _transform_values_sparse(df, weights_df, [value_col], meta_cols)


def _select_forecast_days(day_values, spec, nc_path):
    """Return selected day positions, labels, and integer lead values."""
    day_values = np.asarray(day_values).reshape(-1)
    day_numbers = []
    for value in day_values:
        try:
            day_numbers.append(int(value))
        except (TypeError, ValueError):
            day_numbers.append(None)

    options = spec.get("options", {})
    min_day = options.get("min_day")
    max_day = options.get("max_day")
    strict_rain_day_max = options.get("rain_day_max")
    if strict_rain_day_max is not None:
        _, horizon_policy = resolve_rain_day_max(
            {
                "rain_day_max": strict_rain_day_max,
                "rain_horizon_policy": options.get("rain_horizon_policy"),
            },
            probability_day_max=1,
        )
        validation_values = [
            value for value, day in zip(day_values, day_numbers)
            if day is not None and (min_day is None or day >= int(min_day))
        ]
        validate_day_coordinate(
            validation_values,
            strict_rain_day_max,
            context=f"NetCDF file {os.path.basename(nc_path)}",
            allow_extra=horizon_policy == "truncate",
        )

    keep = np.ones(len(day_values), dtype=bool)
    if min_day is not None:
        keep &= np.array([
            day is not None and day >= int(min_day) for day in day_numbers
        ])
    if max_day is not None:
        keep &= np.array([
            day is not None and day <= int(max_day) for day in day_numbers
        ])
    if not np.any(keep):
        raise ValueError(
            f"Day filtering removed all columns for: {os.path.basename(nc_path)}"
        )

    positions = np.flatnonzero(keep)
    return (
        positions,
        day_values[positions].tolist(),
        [day_numbers[position] for position in positions],
    )


def nc_read_forecast_wide(nc_path, var_name, dim_rename_map, spec,
                          day_dim="day", prefix=None, add_year=True):
    """
    Read a forecast NetCDF (adm3_name-indexed) into a wide-by-lead-day DataFrame.

    The adm3_name dimension is read as strings and stored in an `id` column.
    Returns DataFrame or None if variable not found.
    """
    import netCDF4 as nc4
    prefix = prefix or var_name

    ds = nc4.Dataset(nc_path)
    try:
        if var_name not in ds.variables:
            return None

        v = ds.variables[var_name]
        dim_names = list(v.dimensions)

        # Build coordinate arrays; adm3_name handled specially as strings
        dim_vals = {}
        adm3_dim_name = None
        for d in dim_names:
            dl = d.lower()
            if dl in ("adm3_name", "adm3"):
                adm3_dim_name = d
                dim_vals[d] = get_nc_adm3_names(ds)
            elif dl == "time":
                tinfo = get_nc_time(ds)
                dim_vals[d] = tinfo["values"]
            else:
                if d in ds.variables:
                    dim_vals[d] = np.array(ds[d][:]).flatten()
                elif d in ds.dimensions:
                    dim_vals[d] = np.arange(ds.dimensions[d].size)
                else:
                    dim_vals[d] = np.arange(len(v))

        # Apply renames to identify day dimension
        dummy = pd.DataFrame({d: [dim_vals[d][0]] for d in dim_names})
        dummy_renamed = rename_dimensions(dummy, dim_rename_map or {})
        renamed_names = [c.lower() for c in dummy_renamed.columns]

        try:
            day_idx = renamed_names.index(day_dim.lower())
        except ValueError:
            raise ValueError(
                f"Could not identify day dimension '{day_dim}' (after rename). "
                f"Available: {renamed_names}"
            )

        day_positions, day_vals, day_vals_num = _select_forecast_days(
            dim_vals[dim_names[day_idx]], spec, nc_path
        )

        # Read and permute array so day is last
        arr = _filled_with_nan(v[:])
        other_idx = [i for i in range(len(dim_names)) if i != day_idx]
        perm = other_idx + [day_idx]
        arr = np.transpose(arr, perm)
        arr = arr[..., day_positions]

        # Build grid of other dims
        other_dim_names = [dim_names[i] for i in other_idx]
        other_dim_vals = [dim_vals[dim_names[i]] for i in other_idx]
        grid = _product_frame(other_dim_names, other_dim_vals)
        grid = rename_dimensions(grid, dim_rename_map or {})
        grid.columns = grid.columns.str.lower()

        # Rename the adm3 column to `id`
        adm3_col_renamed = adm3_dim_name.lower() if adm3_dim_name else None
        if adm3_col_renamed and adm3_col_renamed in grid.columns:
            grid = grid.rename(columns={adm3_col_renamed: "id"})
        # Also catch after rename_map has been applied
        if "adm3_name" in grid.columns:
            grid = grid.rename(columns={"adm3_name": "id"})

        n_rows = len(grid)
        n_days = len(day_vals)
        mat = arr.reshape(n_rows, n_days)

        day_labels = [
            str(dv_num) if dv_num is not None else str(dv)
            for dv, dv_num in zip(day_vals, day_vals_num)
        ]
        day_col_names = [f"{prefix}_day_{lab}" for lab in day_labels]
        # mat is this function's own array (the day selection copied it), so
        # the frame can wrap it instead of copying it again.
        day_df = pd.DataFrame(mat, columns=day_col_names, copy=False)

        out = pd.concat([grid.reset_index(drop=True), day_df], axis=1)

        if add_year and "time" in out.columns:
            tinfo = get_nc_time(ds)
            if pd.api.types.is_float_dtype(out["time"]) or pd.api.types.is_integer_dtype(out["time"]):
                out["time"], out["year"] = _row_dates(out["time"].values, tinfo["units"])
            else:
                dates = pd.to_datetime(out["time"])
                out["time"] = dates.date
                out["year"] = pd.DatetimeIndex(dates).year

        return out
    finally:
        ds.close()


def nc_read_groundtruth_long(nc_path, var_name, dim_rename_map, add_year=True,
                             missing_rain_policy="keep"):
    """
    Read a ground-truth NetCDF (adm3_name-indexed) into a long (tidy) DataFrame.

    Under ``drop``, exclude an entire non-time series containing missing
    rainfall rather than deleting individual dates and compressing the
    calendar used by positional onset windows.

    Returns DataFrame or None if variable not found.
    """
    import netCDF4 as nc4

    ds = nc4.Dataset(nc_path)
    try:
        if var_name not in ds.variables:
            return None

        v = ds.variables[var_name]
        dim_names = list(v.dimensions)

        dim_vals = {}
        adm3_dim_name = None
        for d in dim_names:
            dl = d.lower()
            if dl in ("adm3_name", "adm3"):
                adm3_dim_name = d
                dim_vals[d] = get_nc_adm3_names(ds)
            elif dl == "time":
                tinfo = get_nc_time(ds)
                dim_vals[d] = tinfo["values"]
            else:
                if d in ds.variables:
                    dim_vals[d] = np.array(ds[d][:]).flatten()
                elif d in ds.dimensions:
                    dim_vals[d] = np.arange(ds.dimensions[d].size)
                else:
                    dim_vals[d] = np.arange(len(v))

        arr = _filled_with_nan(v[:])
        grid = _product_frame(dim_names, [dim_vals[d] for d in dim_names])
        grid = rename_dimensions(grid, dim_rename_map or {})
        grid.columns = grid.columns.str.lower()

        # Rename adm3 column to `id`
        adm3_col_renamed = adm3_dim_name.lower() if adm3_dim_name else None
        if adm3_col_renamed and adm3_col_renamed in grid.columns:
            grid = grid.rename(columns={adm3_col_renamed: "id"})
        if "adm3_name" in grid.columns:
            grid = grid.rename(columns={"adm3_name": "id"})

        vcol = var_name.lower()
        grid[vcol] = arr.flatten()
        missing_rain_policy = str(missing_rain_policy).lower()
        if missing_rain_policy not in ("keep", "drop", "zero"):
            raise ValueError(
                "missing_rain_policy must be 'keep', 'drop', or 'zero', "
                f"got '{missing_rain_policy}'"
            )

        if add_year and "time" in grid.columns:
            tinfo = get_nc_time(ds)
            if pd.api.types.is_float_dtype(grid["time"]) or pd.api.types.is_integer_dtype(grid["time"]):
                grid["time"], years = _row_dates(grid["time"].values, tinfo["units"])
                # What pd.to_datetime(grid["time"]).dt.year gives: int32.
                grid["year"] = years.astype(np.int32)
                calendar = pd.Series(pd.to_datetime(nc_time_to_dates(
                    tinfo["values"], tinfo["units"]
                )))
            else:
                grid["time"] = pd.to_datetime(grid["time"]).dt.date
                calendar = pd.Series(pd.to_datetime(tinfo["values"]))
                grid["year"] = pd.to_datetime(grid["time"]).dt.year

            calendar = calendar.sort_values().reset_index(drop=True)
            if calendar.isna().any() or calendar.duplicated().any():
                raise ValueError(
                    f"{os.path.basename(nc_path)}: ground-truth time coordinate "
                    "must contain one valid date per day."
                )
            day_steps = calendar.groupby(calendar.dt.year).diff().dt.days
            gaps = day_steps.notna() & day_steps.ne(1)
            if gaps.any():
                gap_pos = int(np.flatnonzero(gaps.to_numpy())[0])
                raise ValueError(
                    f"{os.path.basename(nc_path)}: ground-truth time coordinate "
                    "is not daily-contiguous; first gap is "
                    f"{calendar.iloc[gap_pos - 1].date()} to "
                    f"{calendar.iloc[gap_pos].date()}."
                )

        if missing_rain_policy == "drop":
            missing = grid[vcol].isna()
            if missing.any():
                series_cols = [
                    column for column in grid.columns
                    if column not in ("time", vcol)
                ]
                if series_cols:
                    incomplete = missing.groupby(
                        [grid[column] for column in series_cols], dropna=False
                    ).transform("any")
                else:
                    incomplete = pd.Series(missing.any(), index=grid.index)
                grid = grid.loc[~incomplete].reset_index(drop=True)
        elif missing_rain_policy == "zero":
            grid[vcol] = grid[vcol].fillna(0.0)

        return grid
    finally:
        ds.close()
