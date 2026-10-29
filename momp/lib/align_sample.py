"""A fast, sampled check of the day boundary and lead label between observations and a forecast model.

``momp-run --check --alignment`` and the lead-label detection of zarr datasets use this instead of correlating a
whole year. It reads a small, deterministic sample once and gets every lag from one vectorised step:

* inits: a few dozen, evenly spread over the evaluated period (the model's inits inside the run's dates);
* cells: up to a few thousand, a regular subgrid of the cells the observations and forecasts share inside the
  region box, without the cells the observations do not cover (the sea);
* the forecast leads 1-3 (members averaged on sqrt rain) and, for each init, the observed days from three days
  before the init to the last lead plus three days, all read in one selection per file or store;
* ``momp.onset.alignment.lagged_correlations`` pairs the forecast with a stacked (lag, ...) view of the
  observations and gives the pooled correlation of every lag and lead at once.

The verdicts are those of ``momp.onset.alignment.check_forecast_alignment`` (spec section 8). The result is
cached next to the detection cache (``<cache>/alignment``) under a fingerprint of both datasets, the lead label,
the day boundaries and the sample specification; ``check_models`` runs several models in parallel threads.

The keyword arguments are those of ``momp.export.case_kwargs`` (obs_dir, obs_file_pattern, obs_var,
obs_unit_cvt, model_dir, model_var, unit_cvt, file_pattern, model_dims, probabilistic, members, accumulation,
daily_boundary, region, start_date, end_date, init_days, years).
"""
from __future__ import annotations

import hashlib
import os
import pickle
import threading
import warnings
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from momp.onset import ingest as ing
from momp.onset.alignment import (ForecastAlignmentReport, _match_axis, _sqrt_field,
                                  forecast_alignment_from_samples)
from momp.onset.calendar import label_keys

SAMPLE_INITS = 40
"""Most inits in the sample."""
SAMPLE_CELLS = 3000
"""Most cells in the sample (a regular subgrid of the shared cells)."""
SAMPLE_MEMBERS = 10
"""Most ensemble members averaged."""
LEADS = (1, 2, 3)
"""Forecast leads compared."""
MAX_LAG = 3
"""Lags run from -MAX_LAG to +MAX_LAG days."""
_MIN_COVER = 0.5
"""A cell is kept when at least this share of its sampled values are present in both datasets."""
_NETCDF_LOCK = threading.Lock()
"""Held while NetCDF files are read: the NetCDF/HDF5 libraries are not safe to use from several threads."""
_CACHE_VERSION = 1
"""Bump when the check changes what it reports, to drop old cached verdicts."""


@dataclass
class Sample:
    """The arrays of one sampled read: observations (init, day, cell), forecast (init, lead, cell), in sqrt(mm)."""

    obs_days: np.ndarray
    forecast: np.ndarray
    leads: tuple
    pad: int
    n_inits: int
    n_cells: int
    notes: list = field(default_factory=list)


def _patterns(value) -> tuple:
    """A file pattern or a sequence of them as a tuple."""
    if value is None:
        return ("{}.nc",)
    return (value,) if isinstance(value, str) else tuple(value)


def _years_in(folder, pattern: str) -> tuple:
    """The years of the files in a folder that match a file pattern ({} is the year), sorted."""
    import glob
    import re

    years = set()
    for path in glob.glob(os.path.join(str(folder), pattern.replace("{}", "*"))):
        match = re.search(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)", os.path.basename(path))
        if match:
            years.add(int(match.group(1)))
    return tuple(sorted(years))


def _is_zarr(path) -> bool:
    """True if path is a zarr store (see momp.io.input.is_zarr)."""
    from momp.io.input import is_zarr

    return is_zarr(path)


def _open_store(path) -> xr.Dataset:
    """The lazy Dataset of a zarr store, shared per path (momp.io.input.open_store)."""
    from momp.io.input import open_store

    return open_store(path)


def _box_indices(values: np.ndarray, lo: float, hi: float, wrap: bool = False) -> np.ndarray:
    """Indices of coordinate values inside [lo, hi]; longitudes are compared modulo 360 when wrap is set."""
    values = np.asarray(values, dtype=float)
    if not wrap:
        return np.nonzero((values >= lo) & (values <= hi))[0]
    v, lo, hi = values % 360, lo % 360, hi % 360
    keep = (v >= lo) & (v <= hi) if lo <= hi else (v >= lo) | (v <= hi)
    return np.nonzero(keep)[0]


def _region_box(region):
    """(lats, latn, lonw, lone) of the region, or None when it is unknown."""
    from momp.params.region_def import domain

    try:
        return domain(region)
    except Exception:  # noqa: BLE001 - an unknown region just means no box
        return None


def _scale(da: xr.DataArray, factor) -> xr.DataArray:
    """The loaded array multiplied by a unit factor (as the run's loaders do, in the array's own dtype)."""
    if factor:
        da = da * np.asarray(factor, dtype=da.dtype)
    return da


def _standardise(ds: xr.Dataset, kw: dict) -> xr.Dataset:
    """A forecast Dataset with ROMP's names (init_time, step, member, lat, lon), as the run reads it."""
    from momp.utils.standard import dim_fmt_model, dim_fmt_model_ensemble

    fix = dim_fmt_model_ensemble if kw.get("probabilistic") else dim_fmt_model
    return fix(ds, kw.get("model_dims"))


def _forecast_sources(kw: dict) -> list:
    """The standardised forecast Datasets to read: the zarr store, or the yearly files of the run's years."""
    folder = kw["model_dir"]
    if _is_zarr(folder):
        return [(None, _standardise(_open_store(folder), kw), lambda: None)]
    out = []
    pattern = _patterns(kw.get("file_pattern"))[0]
    for year in kw.get("years") or _years_in(folder, pattern):
        path = os.path.join(folder, pattern.format(year))
        if os.path.exists(path):
            raw = xr.open_dataset(path)
            out.append((year, _standardise(raw, kw), raw.close))
    return out


def _candidate_inits(times, kw: dict) -> np.ndarray:
    """Indices of the init times inside the run's years, month-day window and init weekdays (else all of them)."""
    try:
        stamps = pd.to_datetime(np.asarray(times))
    except (TypeError, ValueError):
        return np.arange(len(times))
    keep = np.ones(len(stamps), dtype=bool)
    start, end = kw.get("start_date"), kw.get("end_date")
    years = kw.get("years")
    if years:
        keep &= stamps.year.isin(list(years))
    if start and end:
        md = stamps.month * 100 + stamps.day
        keep &= (md >= start[1] * 100 + start[2]) & (md <= end[1] * 100 + end[2])
    if kw.get("init_days") is not None:
        keep &= stamps.weekday.isin(list(kw["init_days"]))
    return np.nonzero(keep)[0] if keep.any() else np.arange(len(stamps))


def _spread(n_available: int, n_wanted: int) -> np.ndarray:
    """n_wanted indices spread evenly over range(n_available) (all of them when fewer are available)."""
    if n_available <= n_wanted:
        return np.arange(n_available)
    return np.unique(np.round(np.linspace(0, n_available - 1, n_wanted)).astype(int))


def _stride(n: int, step: int) -> np.ndarray:
    """Every step-th index of range(n), starting half a step in so the subgrid is centred."""
    return np.arange(step // 2 if step > 1 else 0, n, step)


def _shared_grid(obs_ds: xr.Dataset, fc_ds: xr.Dataset, kw: dict) -> tuple:
    """Matched, subsampled grid indices (obs_lat, fc_lat, obs_lon, fc_lon) inside the region box.

    The subgrid takes every s-th shared latitude and longitude, s the smallest stride that leaves about
    SAMPLE_CELLS cells (a little more, as the sea is dropped after reading).
    """
    obs_lat, obs_lon = (np.asarray(obs_ds[c].values, float) for c in ("lat", "lon"))
    fc_lat, fc_lon = (np.asarray(fc_ds[c].values, float) for c in ("lat", "lon"))
    box = _region_box(kw.get("region"))
    lat_in = _box_indices(obs_lat, box[0], box[1]) if box else np.arange(len(obs_lat))
    lon_in = _box_indices(obs_lon, box[2], box[3], wrap=True) if box else np.arange(len(obs_lon))
    ra, fa = _match_axis(obs_lat[lat_in], fc_lat)
    rb, fb = _match_axis(obs_lon[lon_in], fc_lon)
    ol, fl, olon, flon = lat_in[ra], fa, lon_in[rb], fb
    if len(ol) == 0 or len(olon) == 0:
        raise ValueError("no shared cells inside the region box")
    step = max(1, int(np.ceil(np.sqrt(len(ol) * len(olon) / (SAMPLE_CELLS * 1.3)))))
    pick_lat, pick_lon = _stride(len(ol), step), _stride(len(olon), step)
    return ol[pick_lat], fl[pick_lat], olon[pick_lon], flon[pick_lon]


def _member_count(da: xr.DataArray) -> int:
    """How many members to average: up to SAMPLE_MEMBERS, but no more than one file chunk along the member dim.

    Reading a second chunk of members multiplies the decoding of compressed NetCDF files several times over, and
    a few members already give most of the benefit of averaging; at least 4 are kept.
    """
    chunks = da.encoding.get("chunksizes") or da.encoding.get("chunks")
    if "member" not in da.dims or not chunks or len(chunks) != da.ndim:
        return SAMPLE_MEMBERS
    return int(max(min(SAMPLE_MEMBERS, chunks[da.dims.index("member")]), min(4, SAMPLE_MEMBERS)))


def _read_forecast(kw: dict, grid: tuple, sources: list):
    """The sampled forecast as (ForecastRain, init index list): spread-out inits, leads 1-3, subgrid, members cut."""
    fl, flon = grid[1], grid[3]
    chosen = []
    for k, (_, ds, _) in enumerate(sources):
        for i in _candidate_inits(ds["init_time"].values, kw):
            chosen.append((k, int(i), ds["init_time"].values[i]))
    if not chosen:
        raise ValueError("the forecasts have no init in the run's period")
    chosen.sort(key=lambda c: c[2])
    chosen = [chosen[i] for i in _spread(len(chosen), SAMPLE_INITS)]
    span_lat, span_lon = slice(int(fl.min()), int(fl.max()) + 1), slice(int(flon.min()), int(flon.max()) + 1)
    parts = []
    for k, (_, ds, _) in enumerate(sources):
        picks = [i for kk, i, _ in chosen if kk == k]
        if not picks:
            continue
        sub = ds
        if float(sub["step"][0]) == 0:
            sub = sub.sel(step=slice(1, None))
        sub = sub.sel(step=slice(None, max(LEADS)))
        da = sub[kw["model_var"]]
        n_members = _member_count(da)
        da = da.isel(init_time=picks)
        if "member" in da.dims:
            if kw.get("probabilistic") and kw.get("members"):
                da = da.sel(member=list(kw["members"])[:n_members])
            else:
                da = da.isel(member=slice(0, n_members))
        da = da.isel(lat=span_lat, lon=span_lon).load()
        da = _scale(da, kw.get("unit_cvt"))
        da = da.isel(lat=fl - span_lat.start, lon=flon - span_lon.start)
        parts.append(da)
    cube = parts[0] if len(parts) == 1 else xr.concat(parts, dim="init_time")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast, _ = ing.ingest_forecast_info(
            cube, day_boundary="00 UTC, labelled by start", lead_label="start", units="mm", source="forecast",
            init_dim="init_time", lead_dim="step", member_dim="member" if "member" in cube.dims else None,
            accumulation=kw.get("accumulation") or "daily", daily_boundary=kw.get("daily_boundary") or None)
    return forecast


def _forecast_field(forecast, leads) -> np.ndarray:
    """Member-mean sqrt-rain of the given leads as (init, lead, cell), cells in (lat, lon) row-major order."""
    lead_axis = forecast.dims.index("lead")
    values = np.asarray(forecast.coords["lead"]).tolist()
    for k in leads:
        if k not in values:
            raise ValueError(f"lead {k} is not in the forecast leads")
    pick = [values.index(k) for k in leads]
    f = _sqrt_field(np.take(forecast.tenths, pick, axis=lead_axis), np.take(forecast.missing, pick, axis=lead_axis))
    dims = forecast.dims
    if "member" in dims:
        axis = dims.index("member")
        with np.errstate(all="ignore"):
            f = np.nanmean(f, axis=axis)
        dims = tuple(d for d in dims if d != "member")
    f = np.transpose(f, [dims.index(d) for d in ("init_time", "lat", "lon", "lead")])
    return f.reshape(f.shape[0], -1, f.shape[-1]).transpose(0, 2, 1)


def _obs_segments(kw: dict, years) -> list:
    """The standardised observation Datasets to read: the zarr store, or the yearly files with data for the years."""
    from momp.utils.standard import dim_fmt

    folder = kw["obs_dir"]
    if _is_zarr(folder):
        return [(dim_fmt(_open_store(folder)), lambda: None)]
    out = []
    if not years:
        years = _years_in(folder, _patterns(kw.get("obs_file_pattern"))[0])[:1]
    for year in sorted(years):
        for pattern in _patterns(kw.get("obs_file_pattern")):
            path = os.path.join(folder, pattern.format(year))
            if os.path.exists(path):
                raw = xr.open_dataset(path)
                out.append((dim_fmt(raw), raw.close))
                break
    return out


def _read_obs(kw: dict, grid: tuple, forecast, pad: int, n_days: int):
    """Observed sqrt-rain for each init's days -pad..: (init, day, cell) and the notes; one read per file or store."""
    ol, _, olon, _ = grid
    init_keys = label_keys(forecast.init_year, forecast.init_month, forecast.init_day)
    year_span = sorted({int(y) for y in forecast.init_year} | {int(y) - 1 for y in forecast.init_year}
                       | {int(y) + 1 for y in forecast.init_year})
    segments = _obs_segments(kw, year_span)
    if not segments:
        raise ValueError("no observation file covers the forecast inits")
    try:
        times = []
        calendar = None
        for ds, _ in segments:
            cal, ords = ing._read_time(ds["time"].values, constant_time_of_day=True)
            if calendar not in (None, cal):
                raise ValueError("the observation files use different calendars")
            calendar = cal
            times.append(ords)
        if calendar != forecast.calendar:
            raise ValueError(f"calendars differ ({calendar} vs {forecast.calendar}); cannot compare")
        low, high = min(int(t.min()) for t in times), max(int(t.max()) for t in times)
        full = np.arange(low, high + 1)
        _, y, m, d = ing._labels(full, calendar)
        full_keys = label_keys(y, m, d)
        pos = np.searchsorted(full_keys, init_keys)
        found = (pos < len(full_keys)) & (full_keys[np.minimum(pos, len(full_keys) - 1)] == init_keys)
        if not found.any():
            raise ValueError("no forecast init date is in the observations")
        init_ord = full[np.minimum(pos, len(full) - 1)]
        need = init_ord[:, None] + (np.arange(n_days) - pad)[None, :]
        wanted = np.unique(need[found])
        values = np.full((len(wanted), len(ol) * len(olon)), np.nan, dtype=np.float32)
        span_lat, span_lon = slice(int(ol.min()), int(ol.max()) + 1), slice(int(olon.min()), int(olon.max()) + 1)
        for (ds, _), ords in zip(segments, times):
            slot = np.searchsorted(ords, wanted)
            hit = (slot < len(ords)) & (ords[np.minimum(slot, len(ords) - 1)] == wanted)
            if not hit.any():
                continue
            order = np.argsort(slot[hit])
            idx = slot[hit][order]
            da = ds[kw["obs_var"]].isel(time=idx, lat=span_lat, lon=span_lon).load()
            da = _scale(da, kw.get("obs_unit_cvt")).transpose("time", "lat", "lon")
            block = da.values[:, ol - span_lat.start][:, :, olon - span_lon.start]
            values[np.nonzero(hit)[0][order]] = block.reshape(len(idx), -1)
    finally:
        for _, close in segments:
            close()
    tenths = np.round(values.astype(np.float64) * 10.0) / 10.0
    rain = np.where(np.isfinite(tenths) & (tenths >= -0.05), np.sqrt(np.maximum(tenths, 0.0)), np.nan)
    lookup = np.searchsorted(wanted, np.where(found[:, None], need, wanted[0]))
    obs = rain[lookup]
    obs[~found] = np.nan
    return obs.astype(np.float32), found


def read_sample(kw: dict) -> Sample:
    """Read the sample of one model against the observations (see the module text); raises ValueError on trouble.

    Reading NetCDF files is serialised across threads (zarr stores are read concurrently).
    """
    both_zarr = _is_zarr(kw["obs_dir"]) and _is_zarr(kw["model_dir"])
    with nullcontext() if both_zarr else _NETCDF_LOCK:
        return _read_sample(kw)


def _read_sample(kw: dict) -> Sample:
    """The body of read_sample, run under the NetCDF lock when files are read."""
    sources = _forecast_sources(kw)
    if not sources:
        raise ValueError("no forecast file or store found")
    try:
        obs_segments = _obs_segments(kw, kw.get("years") or ())
        if not obs_segments:
            raise ValueError("no observation file found")
        grid = _shared_grid(obs_segments[0][0], sources[0][1], kw)
        for _, close in obs_segments:
            close()
        forecast = _read_forecast(kw, grid, sources)
    finally:
        for _, _, close in sources:
            close()
    pad = MAX_LAG
    n_days = pad + max(LEADS) + MAX_LAG + 1
    obs_days, found = _read_obs(kw, grid, forecast, pad, n_days)
    fc = _forecast_field(forecast, LEADS)
    notes = []
    obs_days, fc = obs_days[found], fc[found]
    if not len(fc):
        raise ValueError("no forecast init date is in the observations")
    cover_obs = np.isfinite(obs_days).mean(axis=(0, 1))
    cover_fc = np.isfinite(fc).mean(axis=(0, 1))
    keep = np.nonzero((cover_obs >= _MIN_COVER) & (cover_fc >= _MIN_COVER))[0]
    if len(keep) == 0:
        raise ValueError("no sampled cell has data in both the observations and the forecasts")
    if len(keep) > SAMPLE_CELLS:
        keep = keep[_spread(len(keep), SAMPLE_CELLS)]
    return Sample(obs_days[:, :, keep], fc[:, :, keep], LEADS, pad, int(len(fc)), int(len(keep)), notes)


def sample_report(sample: Sample, lead_label: str, *, obs_boundary: str = "", forecast_boundary: str = "",
                  source: str = "forecast") -> ForecastAlignmentReport:
    """The alignment report of a sample under one lead label (cheap: no reading)."""
    report = forecast_alignment_from_samples(
        sample.obs_days, sample.forecast, leads=sample.leads, pad=sample.pad, lead_label=lead_label,
        obs="observations", source=source, n_inits=sample.n_inits, obs_boundary=obs_boundary,
        forecast_boundary=forecast_boundary, max_lag=MAX_LAG)
    report.notes = sample.notes + report.notes
    for lead_report in report.per_lead.values():
        lead_report.other = source
    return report


# --- the cache ---------------------------------------------------------------------------------------------------

def cache_dir() -> Path:
    """The folder of the cached verdicts: next to the detection cache, as <cache>/alignment."""
    from momp.lib.dataset_detect import cache_dir as detect_dir

    return detect_dir().parent / "alignment"


def _signature(path, patterns) -> list:
    """What identifies a dataset's content: sizes and mtimes of its files, or the metadata of a zarr store."""
    folder = Path(str(path))
    out = []
    try:
        if _is_zarr(folder):
            for meta in ("zarr.json", ".zmetadata", ".zgroup", ".zattrs"):
                if (folder / meta).exists():
                    st = (folder / meta).stat()
                    out.append((meta, st.st_size, st.st_mtime_ns, (folder / meta).read_bytes()[:200000]))
            for child in sorted(folder.iterdir()):
                if child.is_dir():
                    for meta in ("zarr.json", ".zarray"):
                        if (child / meta).exists():
                            st = (child / meta).stat()
                            out.append((f"{child.name}/{meta}", st.st_size, st.st_mtime_ns))
        else:
            names = {p.name for p in folder.iterdir() if ".nc" in p.name}
            for name in sorted(names):
                st = (folder / name).stat()
                out.append((name, st.st_size, st.st_mtime_ns))
    except OSError:
        return [("unreadable", str(path))]
    return out


def fingerprint(kw: dict, lead_label: str, obs_boundary: str, forecast_boundary: str) -> str:
    """A hash of both datasets, the lead label, the day boundaries and the sample specification."""
    keys = ("obs_dir", "obs_var", "obs_unit_cvt", "model_dir", "model_var", "unit_cvt", "model_dims", "region",
            "probabilistic", "members", "accumulation", "daily_boundary", "start_date", "end_date", "init_days",
            "years")
    text = repr((_CACHE_VERSION, lead_label, obs_boundary, forecast_boundary,
                 [(k, repr(kw.get(k))) for k in keys], _patterns(kw.get("obs_file_pattern")),
                 _patterns(kw.get("file_pattern")), _signature(kw["obs_dir"], None),
                 _signature(kw["model_dir"], None),
                 (SAMPLE_INITS, SAMPLE_CELLS, SAMPLE_MEMBERS, LEADS, MAX_LAG)))
    return hashlib.sha256(text.encode()).hexdigest()


def _load(key: str):
    """The cached report for a fingerprint, or None."""
    try:
        with open(cache_dir() / f"{key}.pkl", "rb") as fh:
            out = pickle.load(fh)
    except Exception:  # noqa: BLE001 - a missing or corrupt entry is a miss
        return None
    return out if isinstance(out, ForecastAlignmentReport) else None


def _store(key: str, report: ForecastAlignmentReport) -> None:
    """Write a report to the cache; failures (read-only home, full disk) are ignored."""
    try:
        folder = cache_dir()
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f"{key}.{os.getpid()}.{id(report)}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(report, fh)
        os.replace(tmp, folder / f"{key}.pkl")
    except Exception:  # noqa: BLE001
        pass


def check_labels(kw: dict, labels=("start",), *, obs_boundary: str = "", forecast_boundary_format: str = "{label}",
                 source: str = "forecast", use_cache: bool = True) -> dict:
    """Sampled alignment reports of one model under each lead label: {label: (report, from_cache)}.

    The sample is read once, and not at all when every label is cached. forecast_boundary_format turns a label
    into the forecast's declared day boundary text (used only for the note on declared boundaries). Raises
    ValueError (or the reader's error) when the sample cannot be made.
    """
    out, todo = {}, []
    keys = {label: fingerprint(kw, label, obs_boundary, forecast_boundary_format.format(label=label))
            for label in labels} if use_cache else {}
    for label in labels:
        hit = _load(keys[label]) if use_cache else None
        if hit is not None:
            out[label] = (hit, True)
        else:
            todo.append(label)
    if todo:
        sample = read_sample(kw)
        for label in todo:
            report = sample_report(sample, label, obs_boundary=obs_boundary, source=source,
                                   forecast_boundary=forecast_boundary_format.format(label=label))
            if use_cache:
                _store(keys[label], report)
            out[label] = (report, False)
    return out


def check_models(jobs: dict, *, workers: int = 8, use_cache: bool = True) -> dict:
    """Run check_labels for several models in parallel threads.

    jobs maps a name to (kw, labels, options) where options are check_labels' keyword arguments. Returns
    {name: {label: (report, cached)}} or {name: exception} for a model whose sample could not be made.
    """
    def run(item):
        """One model's check; a failure is returned, not raised."""
        name, (kw, labels, options) = item
        try:
            return name, check_labels(kw, labels, use_cache=use_cache, **options)
        except Exception as exc:  # noqa: BLE001 - reported per model
            return name, exc

    if len(jobs) <= 1:
        return dict(run(item) for item in jobs.items())
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return dict(pool.map(run, jobs.items()))
