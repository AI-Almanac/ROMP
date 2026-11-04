"""TEST-ONLY ORACLE: functions copied unchanged from onset_blending@1e6d696 tests/_legacy/nc_utils.py
(add_id_from_latlon, prep_thresholds_id, attach_thresholds_id, _by_distinct, attach_ref_onset), with only the imports
changed to the oracle copies of spatial_id_utils and nc_utils_filters. Never import this from hcwf."""
import numpy as np
import pandas as pd

from .nc_utils_filters import ensure_id_col
from .spatial_id_utils import (
    ensure_spatial_id_col,
    normalize_id_series,
    resolve_grid_id_convention,
    working_copy,
)


def add_id_from_latlon(df, lat_col="lat", lon_col="lon", id_col="id",
                       spec=None, convention=None):
    """Backward-compatible explicit lat/lon-to-id helper."""
    if id_col in df.columns:
        return df.copy()
    if lat_col not in df.columns or lon_col not in df.columns:
        return ensure_id_col(df, id_col=id_col, spec=spec,
                             convention=convention)
    out = df.copy()
    coords = out[[lat_col, lon_col]].rename(
        columns={lat_col: "lat", lon_col: "lon"}
    )
    coords = ensure_spatial_id_col(
        coords,
        id_col=id_col,
        spec=spec,
        convention=convention,
        force_latlon=True,
        context="latitude/longitude IDs",
    )
    out[id_col] = coords[id_col].to_numpy()
    return out


def prep_thresholds_id(thr_df, spec=None, convention=None):
    """
    Normalize thresholds DataFrame to be keyed by id (adm3_name).

    Accepts:
      - DataFrame with 'id' column
      - DataFrame with 'adm3_name' column
      - DataFrame with legacy 'lat' and 'lon' columns
      - Scalar float / int  (single global threshold)
    """
    if thr_df is None:
        return None
    # Scalar threshold — return as-is; callers handle it
    if isinstance(thr_df, (int, float, np.floating, np.integer)):
        return thr_df
    thr_df = thr_df.copy()
    thr_df.columns = thr_df.columns.str.lower()
    if "id" not in thr_df.columns:
        if "adm3_name" in thr_df.columns:
            thr_df["id"] = thr_df["adm3_name"]
        elif "lat" in thr_df.columns and "lon" in thr_df.columns:
            thr_df = add_id_from_latlon(
                thr_df, spec=spec, convention=convention
            )
        else:
            raise ValueError(
                "Thresholds table must have 'id', 'adm3_name', or ('lat', 'lon')."
            )
    thr_df["id"] = normalize_id_series(thr_df["id"], context="threshold IDs")
    thr_df["onset_thresh"] = thr_df["onset_thresh"].astype(float)
    thresholds = thr_df[["id", "onset_thresh"]].drop_duplicates()
    conflicting_ids = thresholds["id"].duplicated(keep=False)
    if conflicting_ids.any():
        sample = ", ".join(thresholds.loc[conflicting_ids, "id"].unique()[:10])
        raise ValueError(f"Threshold IDs have conflicting values: {sample}")
    return thresholds.set_index("id")


def attach_thresholds_id(df, thr_df, spec=None, convention=None):
    """Left-join onset_thresh from thr_df by id."""
    df = ensure_id_col(df, spec=spec, convention=convention)
    convention = convention or resolve_grid_id_convention(
        spec=spec,
        authoritative_ids=df["id"],
        context="rainfall/threshold ID handoff",
    )
    if thr_df is None:
        df["onset_thresh"] = np.nan
        return df
    # Scalar threshold
    if isinstance(thr_df, (int, float, np.floating, np.integer)):
        df["onset_thresh"] = float(thr_df)
        return df
    thr_idx = prep_thresholds_id(thr_df, spec=spec, convention=convention)
    if isinstance(thr_idx, (int, float)):
        df["onset_thresh"] = float(thr_idx)
        return df
    rainfall_ids = pd.Index(df["id"].drop_duplicates())
    missing_ids = rainfall_ids[~rainfall_ids.isin(thr_idx.index)]
    if len(missing_ids):
        sample = ", ".join(map(str, missing_ids[:10]))
        raise ValueError(
            f"Threshold table is missing {len(missing_ids)} of "
            f"{len(rainfall_ids)} rainfall IDs after domain filtering. "
            f"Examples: {sample}"
        )
    df = df.merge(thr_idx.reset_index(), on="id", how="left")
    return df


def _by_distinct(values, fn):
    """``fn(values)`` for an elementwise ``fn``, computed once per distinct value.

    ``fn`` takes and returns a Series; the result is aligned to ``values``.
    Only for values of one Python type (dates, years): factorize would merge
    equal-hashing values of different types.
    """
    codes, distinct = pd.factorize(values, use_na_sentinel=False)
    mapped = fn(pd.Series(distinct))
    out = mapped.take(codes)
    out.index = values.index
    return out


def attach_ref_onset(df, ref):
    """
    Attach a `ref_onset_date` column to df from the value returned by
    read_ref_onset_dates: None (NaT), a constant-month-day dict (a fixed date
    each year), or a DataFrame keyed by 'year' and/or 'id' (per-year and/or
    per-unit dates). Merges on whichever keys are present in both.
    """
    df = working_copy(df)
    if ref is None:
        df["ref_onset_date"] = pd.NaT
        return df
    if isinstance(ref, dict) and ref.get("mode") == "constant_month_day":
        md = ref["month_day"]
        if "year" in df.columns:
            df["ref_onset_date"] = _by_distinct(
                df["year"],
                lambda year: pd.to_datetime(
                    year.astype(int).astype(str) + "-" + md, errors="coerce"
                ).dt.date,
            )
        else:
            df["ref_onset_date"] = pd.NaT
        return df
    # DataFrame keyed by year and/or id
    if "id" in ref.columns:
        df = ensure_id_col(df)
    keys = [k for k in ("id", "year") if k in ref.columns and k in df.columns]
    if not keys:
        df["ref_onset_date"] = pd.NaT
        return df
    return df.merge(ref, on=keys, how="left")
