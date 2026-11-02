"""TEST-ONLY ORACLE: functions copied unchanged from onset_blending@1e6d696 tests/_legacy/nc_utils.py
(ensure_id_col, _resolve_unit_latlon, filter_by_dissemination_cells, read_cell_transform), with only the imports
changed to the oracle copies of spatial_id_utils and sparse_transform_utils. Never import this from momp."""
import os

import numpy as np
import pandas as pd

from .spatial_id_utils import (
    GridIdConvention,
    ensure_spatial_id_col,
    normalize_id_series,
    resolve_grid_id_convention,
    validate_id_coordinate_consistency,
)
from .sparse_transform_utils import compile_sparse_cell_transform


def ensure_id_col(df, id_col="id", spec=None, convention=None,
                  force_latlon=False, context="spatial data"):
    """
    Ensure the DataFrame has an `id` column.
    For adm3-based data, rename `adm3_name` to `id`. For legacy gridded data,
    construct the same `<lat>_<lon>` key used by the R pipeline.
    """
    return ensure_spatial_id_col(
        df,
        id_col=id_col,
        spec=spec,
        convention=convention,
        force_latlon=force_latlon,
        context=context,
    )


# Keep the old name as an alias so callers that import it directly still work.


def _resolve_unit_latlon(df, filt):
    """
    Return (lat, lon) arrays aligned to df rows for the domain bbox filter, or
    (None, None) if they cannot be determined. Sources, in order:
      1. explicit 'lat'/'lon' columns on df (gridded/legacy data),
      2. an id of the form '<lat>_<lon>' (grid-cell units),
      3. filter.centroids_file: CSV with target_id + lat/lon (or center_lat/
         center_lon) giving a representative point per unit (admin units).
         Legacy adm3_name centroid files remain accepted.
    """
    if "lat" in df.columns and "lon" in df.columns:
        return df["lat"].astype(float).values, df["lon"].astype(float).values

    ids = df["id"].astype(str)
    parsed = ids.str.extract(r"^(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$")
    if len(df) and parsed.notna().all(axis=None):
        return parsed[0].astype(float).values, parsed[1].astype(float).values

    cf = (filt or {}).get("centroids_file")
    if cf:
        if not os.path.exists(cf):
            raise FileNotFoundError(f"filter.centroids_file not found: {cf}")
        c = pd.read_csv(cf, dtype=str)
        keyc = filt.get("centroids_id_col")
        if keyc and keyc not in c.columns:
            raise ValueError(
                f"filter.centroids_id_col '{keyc}' not found. "
                f"Found: {c.columns.tolist()}"
            )
        if not keyc:
            keyc = next(
                (x for x in ("target_id", "id", "adm3_name") if x in c.columns),
                c.columns[0],
            )
        latc = next((x for x in ("lat", "center_lat", "latitude") if x in c.columns), None)
        lonc = next((x for x in ("lon", "center_lon", "longitude") if x in c.columns), None)
        if not latc or not lonc:
            raise ValueError("filter.centroids_file must have lat/lon (or center_lat/center_lon).")
        c = c.rename(columns={keyc: "id", latc: "_lat", lonc: "_lon"})
        c["id"] = normalize_id_series(c["id"], context="centroid IDs")
        m = df[["id"]].merge(c[["id", "_lat", "_lon"]].drop_duplicates("id"), on="id", how="left")
        return m["_lat"].astype(float).values, m["_lon"].astype(float).values

    return None, None


def filter_by_dissemination_cells(df, spec):
    """
    Restrict the domain to the modelling units of interest. Two independent,
    composable restrictions read from spec['filter']:

      - dissemination_cells_file : keep only ids listed in a configured `id_col`,
        canonical `id`, legacy `adm3_name`, or `lat`/`lon` columns.
      - bbox : an optional FURTHER restriction {lat_min, lat_max, lon_min,
        lon_max} (any subset of keys), applied on top of the dissemination set.
        Defaults to no bbox restriction (i.e. all dissemination cells).

    Returns df unchanged if neither is configured.
    """
    filt = spec.get("filter") or {}
    df = ensure_id_col(df, spec=spec, context="rainfall spatial IDs")

    dc_file = filt.get("dissemination_cells_file")
    if dc_file:
        if not os.path.exists(dc_file):
            raise FileNotFoundError(f"dissemination_cells_file not found: {dc_file}")
        dc = pd.read_csv(dc_file, dtype=str)
        configured_id_col = filt.get("id_col") or filt.get(
            "dissemination_id_col"
        )
        if configured_id_col:
            if configured_id_col not in dc.columns:
                raise ValueError(
                    f"Configured dissemination id_col '{configured_id_col}' "
                    f"not found. Found: {dc.columns.tolist()}"
                )
            dc = dc.rename(columns={configured_id_col: "id"})
        elif not any(
            column in dc.columns for column in ("id", "adm3_name")
        ) and not {"lat", "lon"}.issubset(dc.columns):
            fallback_col = dc.columns[0]
            print(
                "  WARNING: dissemination_cells_file has no explicit spatial "
                f"key; using first column '{fallback_col}' as a legacy fallback."
            )
            dc = dc.rename(columns={fallback_col: "id"})
        convention = resolve_grid_id_convention(
            spec=spec,
            authoritative_ids=df["id"],
            context="rainfall/dissemination ID handoff",
        )
        dc = ensure_id_col(
            dc,
            spec=spec,
            convention=convention,
            context="dissemination cell IDs",
        )
        validate_id_coordinate_consistency(dc, context="dissemination cell IDs")
        dc = dc.drop_duplicates("id")
        valid_ids = set(dc["id"])
        before = len(df)
        df = df[df["id"].isin(valid_ids)]
        print(f"  dissemination_cells filter: {before} -> {len(df)} rows "
              f"({before - len(df)} removed)")

    bbox = filt.get("bbox")
    if bbox:
        lat, lon = _resolve_unit_latlon(df, filt)
        if lat is None:
            raise ValueError(
                "filter.bbox requested but unit lat/lon could not be determined: "
                "df has no 'lat'/'lon' columns, ids are not '<lat>_<lon>', and no "
                "filter.centroids_file was provided."
            )
        mask = np.ones(len(df), dtype=bool)
        if bbox.get("lat_min") is not None:
            mask &= lat >= float(bbox["lat_min"])
        if bbox.get("lat_max") is not None:
            mask &= lat <= float(bbox["lat_max"])
        if bbox.get("lon_min") is not None:
            mask &= lon >= float(bbox["lon_min"])
        if bbox.get("lon_max") is not None:
            mask &= lon <= float(bbox["lon_max"])
        before = len(df)
        df = df[mask]
        print(f"  bbox filter {bbox}: {before} -> {len(df)} rows ({before - len(df)} removed)")

    return df


def read_cell_transform(spec):
    """Read weights file for linear cell regridding. Returns DataFrame or None."""
    if not spec.get("options", {}).get("cell_transform_enabled", False):
        return None
    f = spec["options"].get("cell_transform_file")
    if not f:
        raise ValueError("cell_transform_enabled=True but options.cell_transform_file is empty.")
    if not os.path.exists(f):
        raise FileNotFoundError(f"cell_transform_file not found: {f}")
    w = pd.read_csv(f, dtype={"source_id": str, "target_id": str})
    for col in ("target_id", "source_id", "weight"):
        if col not in w.columns:
            raise ValueError(f"Transform file must have: target_id, source_id, weight")
    if w.empty:
        raise ValueError("Cell-transform weights file must not be empty.")
    w["source_id"] = normalize_id_series(
        w["source_id"], context="cell-transform source IDs"
    )
    w["target_id"] = normalize_id_series(
        w["target_id"], context="cell-transform target IDs"
    )
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce")
    if not np.isfinite(w["weight"]).all() or (w["weight"] < 0).any():
        raise ValueError("Cell-transform weights must be finite and non-negative.")
    if w.duplicated(["source_id", "target_id"]).any():
        raise ValueError("Cell-transform source_id/target_id pairs must be unique.")
    zero_targets = w.groupby("target_id")["weight"].sum()
    zero_targets = zero_targets[zero_targets <= 0]
    if len(zero_targets):
        sample = ", ".join(zero_targets.index.astype(str)[:10])
        raise ValueError(f"Cell-transform targets have zero total weight: {sample}")
    w = w.loc[w["weight"] > 0].copy()
    convention = resolve_grid_id_convention(
        spec=spec,
        authoritative_ids=w["source_id"].drop_duplicates(),
        context=f"cell-transform file {os.path.basename(f)}",
    )
    if convention is not None:
        w.attrs["source_id_convention"] = convention.as_dict()
        print(
            "  cell-transform source ID convention: "
            f"digits={convention.decimal_digits}, "
            f"format={convention.number_format} "
            f"({convention.source})"
        )
    w.attrs["sparse_transform"] = compile_sparse_cell_transform(
        w,
        source_ids=tuple(sorted(w["source_id"].unique())),
        target_ids=tuple(sorted(w["target_id"].unique())),
    )
    return w
