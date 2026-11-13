"""Onset detection for observations and forecasts.

All detection is delegated to an :class:`~momp.stats.onset_rule.OnsetRule`
(resolved from the run configuration via :func:`resolve_rule`), so the observed
and forecast sides of a benchmark always apply the identical definition. This
module owns everything *around* the rule: search-window subsetting, MOK
date filtering, valid-case selection (forecasts initialised before the observed
onset), ensemble aggregation, and DataFrame assembly.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from momp.stats.onset_rule import OnsetRule, first_onset_index, resolve_rule
from momp.utils.practical import restore_args

_LEGACY_KEYS = ("wet_init", "wet_spell", "dry_spell", "dry_extent")


def _rule(onset_rule, onset_rule_params, kwargs) -> OnsetRule:
    """Resolve the rule for a call from explicit args + the flat ROMP config in kwargs."""
    flat = {k: kwargs[k] for k in _LEGACY_KEYS if kwargs.get(k) is not None}
    return resolve_rule(onset_rule, onset_rule_params, **flat)


def find_first_true(arr):
    """Index of the first truthy value along a 1-D array (NaN counts as False), or -1."""
    arr_bool = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0).astype(bool)
    return int(np.argmax(arr_bool)) if arr_bool.any() else -1


# --------------------------------------------------------------------------- #
# Scalar compatibility shim
# --------------------------------------------------------------------------- #
def detect_onset(day, forecast_series, thresh, *, onset_rule=None, onset_rule_params=None, **kwargs):
    """Is forecast day ``day`` (1-based) an onset for this single series?

    Kept for backward compatibility with the original per-day API; new code
    should apply the rule to a whole array and use :func:`first_onset_index`.
    """
    rule = _rule(onset_rule, onset_rule_params, kwargs)
    series = xr.DataArray(np.asarray(forecast_series, dtype=float), dims=("step",))
    mask = rule(series, thresh, dim="step").values
    idx = day - 1
    return bool(mask[idx]) if 0 <= idx < len(mask) else False


# --------------------------------------------------------------------------- #
# Observed onset
# --------------------------------------------------------------------------- #
def detect_observed_onset(
    rain_slice,
    thresh_slice,
    year,
    *,
    start_date,
    end_date,
    fallback_date,
    mok,
    extend_end_day=47,
    onset_rule=None,
    onset_rule_params=None,
    **kwargs,
):
    """Observed onset date per grid cell for ``year``.

    The search window starts at ``mok`` (MM, DD) if given, else ``start_date``'s
    month/day, and ends ``extend_end_day`` days after ``end_date``'s month/day
    (extended further if the rule's lookahead needs it). Returns a
    ``datetime64`` DataArray over the non-time dims of ``rain_slice`` (NaT where
    no onset).
    """
    rule = _rule(onset_rule, onset_rule_params, kwargs)

    start_MMDD = start_date[1:]
    end_MMDD = end_date[1:]

    search_end = datetime(year, *end_MMDD)
    if extend_end_day:
        search_end = search_end + timedelta(days=max(int(extend_end_day), rule.lookahead + 1))

    if mok:
        search_start = datetime(year, *mok)
        date_label = f"{mok[0]:02d}-{mok[1]:02d}"
    else:
        search_start = datetime(year, *start_MMDD)
        date_label = f"{start_MMDD[0]:02d}-{start_MMDD[1]:02d}"

    time_dates = pd.to_datetime(rain_slice.time.values)
    start_idx_candidates = np.where(time_dates >= search_start)[0]
    if len(start_idx_candidates) == 0 and fallback_date:
        print(f"Warning: {date_label} not found in data for year {year}")
        fb = datetime(year, *fallback_date)
        start_idx = np.where(time_dates >= fb)[0][0]
        print(f"Using fallback date: {fb:%m-%d}")
    else:
        start_idx = start_idx_candidates[0]

    rain_subset = rain_slice.isel(time=slice(start_idx, None)).sel(time=slice(None, search_end))

    mask = rule(rain_subset, thresh_slice, dim="time")
    idx = first_onset_index(mask, "time")

    time_coords = rain_subset.time.values.astype("datetime64[ns]")
    dates = np.where(idx.values >= 0, time_coords[np.clip(idx.values, 0, None)], np.datetime64("NaT"))
    onset_da = xr.DataArray(dates.astype("datetime64[ns]"), dims=idx.dims, coords=idx.coords, name="onset_date")
    return onset_da


# --------------------------------------------------------------------------- #
# Shared forecast machinery
# --------------------------------------------------------------------------- #
def _forecast_onset_days(p_model, thresh_slice, rule, *, max_forecast_day, mok):
    """Rule-based onset day per forecast (all non-step dims kept).

    Candidate days are ``1..max_forecast_day``; with ``mok`` only forecast dates
    on/after the MOK date count. Returns a float DataArray (NaN = no onset).
    """
    max_steps_needed = max_forecast_day + rule.lookahead
    full_steps = p_model.sizes["step"]
    if full_steps < max_steps_needed:
        raise ValueError(
            f"Not enough forecast time steps: model steps {full_steps} < min steps required "
            f"{max_steps_needed} (max_forecast_day={max_forecast_day} + rule lookahead={rule.lookahead}); "
            f"reduce max_forecast_day or use a rule with a shorter lookahead"
        )

    rain = p_model.sel(step=slice(1, max_steps_needed))
    mask = rule(rain, thresh_slice, dim="step")
    mask = mask.where(mask.step <= max_forecast_day, False)

    if mok:
        init_times = pd.to_datetime(mask.init_time.values)
        steps = mask.step.values.astype(int)
        valid = np.zeros((len(init_times), len(steps)), dtype=bool)
        for t_idx, init_date in enumerate(init_times):
            mok_date = pd.Timestamp(datetime(init_date.year, *mok))
            valid[t_idx, :] = (init_date + pd.to_timedelta(steps, unit="D")).date >= mok_date.date()
        mask = mask & xr.DataArray(
            valid, dims=("init_time", "step"), coords={"init_time": mask.init_time, "step": mask.step}
        )

    idx = first_onset_index(mask, "step")
    steps = mask.step.values.astype(float)
    days = np.where(idx.values >= 0, steps[np.clip(idx.values, 0, None)], np.nan)
    return xr.DataArray(days, dims=idx.dims, coords=idx.coords, name="onset_day")


def _valid_cases(onset_da, init_time_coord):
    """(init_time, lat, lon) bool: observed onset exists and the forecast started before it."""
    obs = onset_da.transpose("lat", "lon")
    obs_values = obs.values.astype("datetime64[ns]")
    init_values = init_time_coord.values.astype("datetime64[ns]")
    valid = (~np.isnat(obs_values))[None, :, :] & (init_values[:, None, None] < obs_values[None, :, :])
    return xr.DataArray(
        valid,
        dims=("init_time", "lat", "lon"),
        coords={"init_time": init_time_coord, "lat": obs.lat, "lon": obs.lon},
    ), obs


def _require_dims(p_model, dims, what):
    missing = set(dims) - set(p_model.dims)
    if missing:
        raise ValueError(f"{what}: p_model is missing dims {sorted(missing)}; has {p_model.dims}")


# --------------------------------------------------------------------------- #
# Deterministic forecast
# --------------------------------------------------------------------------- #
def compute_onset_for_deterministic_model(
    p_model,
    thresh_slice,
    onset_da,
    *,
    max_forecast_day,
    mok,
    end_date,
    onset_rule=None,
    onset_rule_params=None,
    **kwargs,
):
    """Onset day/date per (init_time, lat, lon) for a deterministic forecast.

    Only forecasts initialised before the observed onset at a cell with an
    observed onset are returned (as in the original implementation).
    """
    _require_dims(p_model, ("init_time", "step", "lat", "lon"), "compute_onset_for_deterministic_model")
    rule = _rule(onset_rule, onset_rule_params, kwargs)

    print(
        f"Processing {p_model.sizes['init_time']} init times x {p_model.sizes['lat']} lats x "
        f"{p_model.sizes['lon']} lons with rule {rule.label()}..."
    )

    onset_day = _forecast_onset_days(
        p_model.transpose("init_time", "lat", "lon", "step"), thresh_slice, rule,
        max_forecast_day=max_forecast_day, mok=mok,
    )
    valid_case, obs = _valid_cases(onset_da, onset_day.init_time)
    onset_day = onset_day.where(valid_case)

    df = onset_day.to_dataframe().reset_index()
    obs_b = obs.broadcast_like(valid_case)
    df["obs_onset_date"] = obs_b.to_dataframe(name="obs_onset_date").reset_index()["obs_onset_date"]
    df["valid"] = valid_case.to_dataframe(name="valid").reset_index()["valid"]
    df = df[df["valid"]].drop(columns="valid").copy()

    has = df["onset_day"].notna()
    df["onset_date"] = None
    if has.any():
        dates = pd.to_datetime(df.loc[has, "init_time"]) + pd.to_timedelta(df.loc[has, "onset_day"].astype(int), unit="D")
        df.loc[has, "onset_date"] = dates.dt.strftime("%Y-%m-%d")
    df["obs_onset_date"] = pd.to_datetime(df["obs_onset_date"]).dt.strftime("%Y-%m-%d")
    df["onset_day"] = df["onset_day"].astype(object).where(has, None)
    df.loc[has, "onset_day"] = df.loc[has, "onset_day"].astype(int)

    onset_df = df[["init_time", "lat", "lon", "onset_day", "onset_date", "obs_onset_date"]].reset_index(drop=True)

    n_valid = len(onset_df)
    n_found = int(has.sum())
    print("\nProcessing Summary:")
    print(f"Valid initializations processed: {n_valid}")
    print(f"Onsets found: {n_found}")
    print(f"Onset rate: {n_found / n_valid:.3f}" if n_valid else "Onset rate: 0.000")
    return onset_df


# --------------------------------------------------------------------------- #
# Ensemble forecast
# --------------------------------------------------------------------------- #
def compute_onset_for_all_members_vectorized(
    p_model,
    thresh_slice,
    onset_da,
    *,
    members,
    onset_percentage_threshold,
    max_forecast_day,
    mok,
    end_date,
    onset_rule=None,
    onset_rule_params=None,
    **kwargs,
):
    """Per-member onset days and the ensemble consensus onset.

    Returns ``(onset_df, onset_mean_df)``: one row per (init_time, lat, lon,
    member) and one per (init_time, lat, lon) with the mean onset day of members
    that found onset, kept only when at least ``onset_percentage_threshold`` of
    members did.
    """
    _require_dims(p_model, ("init_time", "member", "step", "lat", "lon"), "compute_onset_for_all_members")
    rule = _rule(onset_rule, onset_rule_params, kwargs)

    if not members:
        members = tuple(p_model.member.values.tolist())
    else:
        members = tuple(members)
        missing = set(members) - set(p_model.member.values.tolist())
        if missing:
            raise ValueError(f"members {sorted(missing)} not present in p_model.member")

    p_model = p_model.sel(member=list(members)).transpose("init_time", "member", "lat", "lon", "step")

    onset_da_members = _forecast_onset_days(p_model, thresh_slice, rule, max_forecast_day=max_forecast_day, mok=mok)
    valid_case, obs_onset = _valid_cases(onset_da, onset_da_members.init_time)
    onset_da_members = onset_da_members.where(valid_case)

    obs_onset_for_cases = obs_onset.broadcast_like(valid_case).where(valid_case)
    obs_onset_for_members = obs_onset_for_cases.broadcast_like(onset_da_members)

    onset_df = onset_da_members.to_dataframe().reset_index()
    onset_df["obs_onset_date"] = (
        obs_onset_for_members.to_dataframe(name="obs_onset_date").reset_index()["obs_onset_date"]
    )
    onset_df = onset_df[onset_df["obs_onset_date"].notna()].copy()
    onset_df["obs_onset_date"] = pd.to_datetime(onset_df["obs_onset_date"]).dt.strftime("%Y-%m-%d")
    onset_df["onset_day"] = onset_df["onset_day"].where(onset_df["onset_day"].notna(), None)

    onset_count = onset_da_members.notnull().sum("member")
    total_members = len(members)
    onset_percentage = onset_count / total_members if total_members > 0 else 0
    ensemble_onset_day = onset_da_members.mean("member", skipna=True).round()
    ensemble_onset_day = ensemble_onset_day.where(onset_percentage >= onset_percentage_threshold)
    ensemble_onset_day = ensemble_onset_day.where(valid_case)

    onset_mean_df = ensemble_onset_day.to_dataframe(name="onset_day").reset_index()
    onset_mean_df["member_onset_count"] = (
        onset_count.to_dataframe(name="member_onset_count").reset_index()["member_onset_count"]
    )
    onset_mean_df["total_members"] = total_members
    onset_mean_df["onset_percentage"] = (
        onset_percentage.to_dataframe(name="onset_percentage").reset_index()["onset_percentage"]
    )
    onset_mean_df["obs_onset_date"] = (
        obs_onset_for_cases.to_dataframe(name="obs_onset_date").reset_index()["obs_onset_date"]
    )
    onset_mean_df = onset_mean_df[onset_mean_df["obs_onset_date"].notna()].copy()
    onset_mean_df["onset_date"] = None

    has_ensemble_onset = onset_mean_df["onset_day"].notna()
    if has_ensemble_onset.any():
        init_dates = pd.to_datetime(onset_mean_df.loc[has_ensemble_onset, "init_time"])
        onset_days = onset_mean_df.loc[has_ensemble_onset, "onset_day"].astype(int)
        onset_dates = init_dates + pd.to_timedelta(onset_days, unit="D")
        onset_mean_df.loc[has_ensemble_onset, "onset_date"] = onset_dates.dt.strftime("%Y-%m-%d")

    onset_mean_df["obs_onset_date"] = pd.to_datetime(onset_mean_df["obs_onset_date"]).dt.strftime("%Y-%m-%d")
    onset_mean_df["onset_day"] = onset_mean_df["onset_day"].where(onset_mean_df["onset_day"].notna(), None)

    onset_df = onset_df[["init_time", "lat", "lon", "member", "onset_day", "obs_onset_date"]]
    onset_mean_df = onset_mean_df[
        [
            "init_time",
            "lat",
            "lon",
            "onset_day",
            "onset_date",
            "member_onset_count",
            "total_members",
            "onset_percentage",
            "obs_onset_date",
        ]
    ]

    print("\nProcessing Summary:")
    print(f"Rule: {rule.label()} (lookahead {rule.lookahead} d)")
    print(f"Generated {len(onset_df)} member-forecast combinations")
    print(f"Found onset in {onset_df['onset_day'].notna().sum()} cases")
    print(f"Onset rate: {onset_df['onset_day'].notna().mean():.3f}" if len(onset_df) else "Onset rate: 0.000")

    return onset_df, onset_mean_df


def compute_onset_for_all_members(p_model, thresh_slice, onset_da, *, members, onset_percentage_threshold,
                                  max_forecast_day, mok, end_date, **kwargs):
    """Public entry point; see :func:`compute_onset_for_all_members_vectorized`."""
    kwargs = restore_args(compute_onset_for_all_members, kwargs, locals())
    return compute_onset_for_all_members_vectorized(p_model, thresh_slice, onset_da, **kwargs)
