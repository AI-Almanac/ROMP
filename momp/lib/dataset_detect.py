"""Complete a dataset description from its files, and check what the user wrote against them.

``resolve_entry`` takes the keys a user gave for one dataset (path, file_pattern, variable,
units (mm or m), dims, day_boundary, lead_label, members) and returns every key filled in.
A key the user gave always wins; a missing key is detected by opening the first matching
file lazily and is reported with its evidence, never silently. A key that cannot be detected
reliably (several candidate variables, units missing from the file, day boundary, a lead label
the observations do not settle) is a *problem* naming the key to set. A key the user gave that
the files contradict (units "mm" with values that look like metres, a dim that does not exist)
is an error for names and a warning for units.

What is detected, and how:
  file_pattern  the year-like number in the file names in ``path``, replaced by {}
  kind          forecast if the data has a lead/step dimension, else observations (checked, not set)
  variable      the only gridded variable, else the only precipitation-like one
  units         the file's ``units`` attribute (mm or m), sanity-checked against the values
  dims          known dimension names (as momp.onset.ingest), role by role
  members       the size of the member dimension
  accumulation  daily or since_init, from how the values grow along the lead (momp.onset.ingest.detect_accumulation)
  lead_label    forecast against observations at day lags -3..+3 (momp.onset.alignment)
  day_boundary  only an explicit ``day_boundary`` attribute in the file
  region        the lat/lon extent against the domains in momp/onset/regions
"""
from __future__ import annotations

import glob
import hashlib
import os
import pickle
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from momp.lib.dataset_schema import DatasetError, expand_path, parse_dataset
from momp.onset import ingest

YEAR_RE = re.compile(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)")
_PRECIP_RE = re.compile(r"(?i)precip|rain|prcp|(^|_)(tp|pr)($|_)")
_ROLE_ALIASES = {
    "time": ("time", "valid_time", "date"),
    "init_time": ("init_time", "time", "forecast_reference_time"),
    "step": tuple(a.lower() for a in ingest._LEAD_DIMS),
    "member": ("member", "number", "ensemble", "realization", "sample"),
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "longitude", "x"),
}
_ORDER = ("path", "file_pattern", "variable", "units", "dims", "day_boundary", "lead_label", "accumulation",
          "daily_boundary", "members")
_ALIGN_INITS = 40
_NULLABLE = ("day_boundary", "lead_label")
"""Keys where an explicit null is a choice (the region's boundary; derive the label)."""


@dataclass
class Line:
    """One resolved key: its value, how it was obtained ('specified', 'detected', 'from dataset ...'), evidence."""

    key: str
    value: Any
    how: str
    evidence: str = ""

    def text(self) -> str:
        """The report line, e.g. ``variable: tp  [detected: only gridded variable in 2020.nc]``."""
        note = self.how + (f": {self.evidence}" if self.evidence else "")
        return f"{self.key}: {self.value}  [{note}]"


@dataclass
class Resolution:
    """The outcome for one dataset: completed values, a report line per key, problems and warnings."""

    values: dict
    lines: list[Line] = field(default_factory=list)
    problems: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    extent: tuple | None = None
    sample_file: str | None = None
    checked: bool = False

    def report(self) -> list[str]:
        """Printable lines: every key with its provenance, then warnings and problems."""
        out = [line.text() for line in self.lines]
        out += [f"warning: {w}" for w in self.warnings]
        out += [f"PROBLEM {k}: {m}" for k, m in self.problems.items()]
        return out


def find_pattern(names: list[str]) -> tuple[str | None, str]:
    """The file pattern shared by names (year replaced by {}) and the evidence, or (None, reason)."""
    patterns = Counter()
    for name in names:
        match = YEAR_RE.search(name)
        if match:
            patterns[name[:match.start()] + "{}" + name[match.end():]] += 1
    if not patterns:
        return None, "no file name contains a year such as 2020"
    if len(patterns) > 1:
        return None, "several name patterns: " + ", ".join(f"'{p}' ({n} files)" for p, n in patterns.items())
    (pattern, count), = patterns.items()
    return pattern, f"{count} file(s) named like {pattern!r}, e.g. {next(n for n in names if YEAR_RE.search(n))}"


def _roles(da: xr.DataArray, kind: str, problems: dict) -> dict:
    """Map each dim role of the data to the file's dim name; ambiguous or missing roles go to problems."""
    wanted = ("time", "lat", "lon") if kind == "observations" else ("init_time", "step", "member", "lat", "lon")
    found = {}
    for role in wanted:
        hits = [d for d in da.dims if str(d).lower() in _ROLE_ALIASES[role]]
        if role == "step" and not hits:
            hits = [d for d in da.dims if any(s in str(d).lower() for s in ("lead", "step", "timedelta"))]
        if role == "init_time":
            hits = [d for d in hits if d not in [x for x in da.dims if str(x).lower() in _ROLE_ALIASES["step"]]]
        if len(hits) > 1:
            problems[f"dims.{role}"] = f"several dims could be {role}: {', '.join(map(str, hits))}; set dims: {{{role}: <name>}}"
        elif hits:
            found[role] = str(hits[0])
        elif role != "member":
            problems[f"dims.{role}"] = f"no {role} dim among {', '.join(map(str, da.dims))}; set dims: {{{role}: <name>}}"
    return found


_SAMPLE_BLOCKS = 3
"""Contiguous blocks read from a dataset's file: its start, middle and end along the time-like dim."""
_SAMPLE_BLOCK_LEN = 2
"""Most consecutive time-like indices in one block."""
_SAMPLE_BUDGET = 3_000_000
"""Most values read for the detection sample of one variable (a few MB)."""
_SAMPLE_CELLS = 96
"""Longest lat or lon stretch read: a centred block of this many cells when the grid is larger."""
_SAMPLES: dict = {}
"""Loaded detection samples, per (file, modification time, variable), for the life of the process."""


def open_data(path: str) -> xr.Dataset:
    """Open a NetCDF file lazily, or a zarr store (a ``.zarr`` directory) with xr.open_zarr."""
    if str(path).rstrip("/").endswith(".zarr") or os.path.isdir(path):
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def _block_indices(n: int, per_index: int) -> list[np.ndarray]:
    """Index blocks (start, middle, end of a length-n dim), each a run of consecutive positions.

    The run length is at most _SAMPLE_BLOCK_LEN and shrinks so the blocks stay within _SAMPLE_BUDGET values
    (per_index values per position), but is never below 1.
    """
    length = int(max(1, min(_SAMPLE_BLOCK_LEN, _SAMPLE_BUDGET // max(1, _SAMPLE_BLOCKS * per_index))))
    if n <= _SAMPLE_BLOCKS * length:
        return [np.arange(n)]
    starts = sorted({0, (n - length) // 2, n - length})
    return [np.arange(s0, s0 + length) for s0 in starts]


def load_sample(da: xr.DataArray, roles: dict, key=None) -> xr.DataArray:
    """A tiny in-memory copy of da, read once: a few whole consecutive time steps from its start, middle and end.

    The time-like dim is the init_time (forecasts) or time (observations) role, else the first dim. Lead is read
    in full (accumulation needs the whole lead series); the member dim is cut to its first member and a grid
    larger than _SAMPLE_CELLS is cut to a centred block, so the read stays within _SAMPLE_BUDGET values (a few MB)
    and decodes each file chunk once. key (e.g. (path, variable)) caches the result in this process.
    """
    if key is not None and key in _SAMPLES:
        return _SAMPLES[key]
    dim = next((roles[r] for r in ("init_time", "time") if roles.get(r) in da.dims), None) or da.dims[0]
    cut = {}
    for role in ("lat", "lon"):
        name = roles.get(role)
        if name in da.dims and da.sizes[name] > _SAMPLE_CELLS:
            start = (da.sizes[name] - _SAMPLE_CELLS) // 2
            cut[name] = slice(start, start + _SAMPLE_CELLS)
    if roles.get("member") in da.dims:
        cut[roles["member"]] = slice(0, 1)
    small = da.isel(cut)
    per_index = int(np.prod([small.sizes[d] for d in small.dims if d != dim], dtype=np.int64))
    blocks = [small.isel({dim: idx}).load() for idx in _block_indices(small.sizes[dim], per_index)]
    out = blocks[0] if len(blocks) == 1 else xr.concat(blocks, dim=dim)
    out.attrs = dict(da.attrs)
    if key is not None:
        _SAMPLES[key] = out
    return out


def clear_samples() -> None:
    """Forget the cached detection samples (for tests, or after the files changed)."""
    _SAMPLES.clear()


def _sample(da: xr.DataArray, roles: dict) -> np.ndarray:
    """A small spread-out sample of the data values as a flat array, with fill values removed.

    Cheap on an in-memory DataArray (see load_sample); on lazy data it would decode chunks repeatedly.
    """
    index = {}
    for dim in da.dims:
        n = da.sizes[dim]
        cap = 40 if dim in (roles.get("lat"), roles.get("lon")) else 12
        index[dim] = np.unique(np.linspace(0, n - 1, min(n, cap)).astype(int))
    values = np.asarray(da.isel(index).values, dtype="float64").ravel()
    return values[np.isfinite(values) & (np.abs(values) < 1e5)]


def _units_from_attr(text) -> str | None:
    """'mm' or 'm' for a units attribute ROMP can convert, else None."""
    key = " ".join(str(text).lower().split())
    if key in ingest._UNITS_MM:
        return "mm"
    if key in ingest._UNITS_M:
        return "m"
    return None


def _check_magnitude(values: np.ndarray, implied: str | None) -> str | None:
    """A warning if the values contradict the units (mm or m), else None."""
    if values.size == 0:
        return "the sampled values are all missing; units could not be checked against them"
    peak = float(values.max())
    if implied == "mm" and 0 < peak < 0.5:
        return f"units are mm but the sampled values peak at {peak:.3g}, which looks like metres (check units)"
    if implied == "m" and peak > 5:
        return f"units are m but the sampled values reach {peak:.3g}, which looks like mm (check units)"
    return None


def detect_region(extent: tuple) -> tuple[str, str]:
    """The region whose domain best overlaps the data extent (lat0, lat1, lon0, lon1), with the evidence.

    Returns ("unknown", reason) when no domain overlaps by at least 30% (intersection over union).
    """
    from momp.onset import regions

    la0, la1, lo0, lo1 = extent
    best, best_iou = None, 0.0
    for ident in regions.region_ids():
        region = regions.load_region(ident)
        if region.lat is None or region.lon is None:
            continue
        inter = max(0.0, min(la1, region.lat[1]) - max(la0, region.lat[0])) * \
            max(0.0, min(lo1, region.lon[1]) - max(lo0, region.lon[0]))
        union = (la1 - la0) * (lo1 - lo0) + (region.lat[1] - region.lat[0]) * (region.lon[1] - region.lon[0]) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = region, iou
    box = f"data cover lat {la0:g}..{la1:g}, lon {lo0:g}..{lo1:g}"
    if best is None or best_iou < 0.3:
        return "unknown", f"{box}; no region domain overlaps enough"
    return best.id, f"{box}; overlaps the {best.title} domain (intersection/union {best_iou:.2f})"


def _lead_label(values: dict, sample: str, obs: dict, roles: dict) -> tuple[str | None, str]:
    """Detect lead_label by aligning the forecast with the observations; returns (label or None, evidence/reason)."""
    from momp.onset.alignment import check_forecast_alignment

    if obs is None or not obs.get("variable") or not obs.get("day_boundary"):
        return None, "needs the observations (with variable and day_boundary) to compare against"
    year = YEAR_RE.search(os.path.basename(sample))
    obs_dir = expand_path(obs["path"])
    files = sorted(glob.glob(os.path.join(obs_dir, obs["file_pattern"].replace("{}", year.group(1) if year else "*"))))
    if not files:
        return None, f"no observation file for year {year.group(1) if year else '?'} in {obs_dir}"
    obs_units, fc_units = obs.get("units"), values.get("units")
    try:
        with open_data(files[0]) as ods:
            obs_rain = ingest.ingest(ods[obs["variable"]].load(), day_boundary=obs["day_boundary"], units=obs_units)
        with open_data(sample) as fds:
            da = fds[values["variable"]]
            pick = {roles["init_time"]: slice(0, _ALIGN_INITS), roles["step"]: slice(0, 3)}
            if "member" in roles:
                pick[roles["member"]] = slice(0, 5)
            da = da.isel(pick).load()
        verdicts = {}
        for label in ("start", "end"):
            forecast = ingest.ingest_forecast(da, day_boundary=f"00 UTC, labelled by {label}", lead_label=label,
                                              units=fc_units, init_dim=roles["init_time"], lead_dim=roles["step"],
                                              member_dim=roles.get("member"))
            verdicts[label] = check_forecast_alignment(obs_rain, forecast).verdict
    except (ValueError, KeyError) as exc:
        return None, f"the alignment check could not run: {exc}"
    aligned = [label for label, verdict in verdicts.items() if verdict == "aligned"]
    detail = "; ".join(f"{label}: {verdict}" for label, verdict in verdicts.items())
    if len(aligned) == 1:
        return aligned[0], f'forecast lines up with observations at lag 0 under "{aligned[0]}" ({detail})'
    return None, f"the alignment check is not conclusive ({detail})"


_CACHE_VERSION = 1
"""Bump when detection changes what it reports, to drop old cache entries."""


def cache_dir() -> Path:
    """The folder of the on-disk detection cache: $MOMP_CACHE_DIR/detect, else ~/.cache/momp/detect."""
    base = os.environ.get("MOMP_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "momp")
    return Path(base) / "detect"


def _fingerprint(entry: dict, kind: str, lead_label_default: bool, sources: dict | None = None) -> str | None:
    """A hash of what detection depends on: the user's keys, and the paths, sizes and mtimes of the dataset's files.

    None when the folder cannot be listed (nothing to cache against).
    """
    try:
        folder = Path(expand_path(entry["path"]))
        files = sorted(p for p in folder.iterdir() if ".nc" in p.name or p.name.endswith(".zarr"))
    except (KeyError, DatasetError, OSError):
        return None
    stats = []
    for path in files:
        try:
            st = path.stat()
            stats.append((str(path), st.st_size, st.st_mtime_ns))
        except OSError:
            return None
    text = repr((_CACHE_VERSION, kind, lead_label_default, sorted((k, repr(v)) for k, v in entry.items()),
                 sorted((k, repr(v)) for k, v in (sources or {}).items()), stats))
    return hashlib.sha256(text.encode()).hexdigest()


def _cache_load(key: str) -> Resolution | None:
    """The cached Resolution for a fingerprint, or None (also when the cache file is unreadable)."""
    try:
        with open(cache_dir() / f"{key}.pkl", "rb") as fh:
            res = pickle.load(fh)
    except Exception:  # noqa: BLE001 - a missing or corrupt cache entry is just a miss
        return None
    return res if isinstance(res, Resolution) else None


def _cache_store(key: str, res: Resolution) -> None:
    """Write a Resolution to the cache; failures (read-only home, full disk) are ignored."""
    try:
        folder = cache_dir()
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f"{key}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(res, fh)
        os.replace(tmp, folder / f"{key}.pkl")
    except Exception:  # noqa: BLE001
        pass


def resolve_entry(entry: dict, kind: str, *, sources: dict | None = None, obs: dict | None = None,
                  detect_lead_label: bool = True, lead_label_default: bool = False, use_cache: bool = True) -> Resolution:
    """Resolve one dataset, from the on-disk cache when the dataset's files and keys are unchanged.

    The cache key is a fingerprint of the user's keys and the file paths, sizes and mtimes; a hit reads no data and
    marks every detected line "detected (cached)" with the same evidence. A resolution that ran the lead-label
    alignment (it depends on the observations) is not cached. See resolve_entry_uncached for the keys.
    """
    runs_alignment = kind == "forecast" and detect_lead_label and "lead_label" not in entry
    key = _fingerprint(entry, kind, lead_label_default, sources) if use_cache and not runs_alignment else None
    if key:
        hit = _cache_load(key)
        if hit is not None:
            for line in hit.lines:
                if line.how == "detected":
                    line.how = "detected (cached)"
            hit.values = {**hit.values}
            return hit
    res = resolve_entry_uncached(entry, kind, sources=sources, obs=obs, detect_lead_label=detect_lead_label,
                                 lead_label_default=lead_label_default)
    if key and res.checked:
        _cache_store(key, res)
    return res


def resolve_entry_uncached(entry: dict, kind: str, *, sources: dict | None = None, obs: dict | None = None,
                           detect_lead_label: bool = True, lead_label_default: bool = False) -> Resolution:
    """Resolve one dataset: validate what is given, detect what is missing, report every key.

    lead_label_default: when the lead_label is not given and not detected (detect_lead_label False), leave it unset
    ("derived from the day boundary") instead of making it a problem.
    entry holds the keys the user gave (raw, as in a dataset file); kind is 'observations' or 'forecast';
    sources says where a key came from if not the user ("dataset 'x'"); obs is the resolved observations
    entry, needed to detect a forecast's lead_label. Never raises for problems with the data: see Resolution.
    """
    sources = sources or {}
    res = Resolution(values={k: v for k, v in entry.items() if v is not None or k in _NULLABLE})
    values, problems = res.values, res.problems

    def specified(key):
        """Record a key the user (or a dataset file) gave."""
        if key in values:
            res.lines.append(Line(key, values[key], sources.get(key, "specified")))

    def detected(key, value, evidence):
        """Record a detected key."""
        values[key] = value
        res.lines.append(Line(key, value, "detected", evidence))

    if not values.get("path"):
        problems["path"] = "missing. Fix: set path to the folder of the yearly files"
        return res
    try:
        folder = Path(expand_path(values["path"]))
    except DatasetError as exc:
        problems["path"] = str(exc)
        return res
    specified("path")
    names = sorted(p.name for p in folder.iterdir() if ".nc" in p.name or p.name.endswith(".zarr")) \
        if folder.is_dir() else []
    needed = [k for k in ("file_pattern", "variable", "day_boundary", "lead_label", "units") if k not in values]
    if kind == "forecast" and "day_boundary" in needed:
        needed.remove("day_boundary")
    if kind == "observations" and "lead_label" in needed:
        needed.remove("lead_label")
    if not names:
        why = "folder not found" if not folder.is_dir() else "no NetCDF files in it"
        if needed:
            problems["path"] = f"{why}: {folder}; the files are needed to detect {', '.join(needed)}. " \
                               "Fix: correct path, or set those keys"
        else:
            res.warnings.append(f"path {folder}: {why}; the files were not checked")
        for key in _ORDER[1:]:
            specified(key)
        return res

    if "file_pattern" in values:
        specified("file_pattern")
    else:
        pattern, evidence = find_pattern(names)
        if pattern is None:
            problems["file_pattern"] = f"cannot be detected ({evidence}). Fix: set file_pattern, e.g. '{{}}.nc'"
        else:
            detected("file_pattern", pattern, evidence)
    files = sorted(glob.glob(os.path.join(str(folder), values.get("file_pattern", "*").replace("{}", "*"))))
    if not files:
        problems["file_pattern"] = f"no file in {folder} matches {values.get('file_pattern')!r}"
        return res
    sample = files[0]
    res.sample_file, res.checked = sample, True
    name = os.path.basename(sample)
    try:
        ds = open_data(sample)
    except Exception as exc:  # noqa: BLE001 - any reader error is reported against the path
        problems["path"] = f"cannot open {sample}: {exc}"
        return res
    with ds:
        _resolve_from_file(ds, name, sample, kind, res, specified, detected, obs, detect_lead_label,
                           lead_label_default)
    return res


def _resolve_from_file(ds, name, sample, kind, res, specified, detected, obs, detect_lead_label,
                       lead_label_default=False):
    """The detections that need the opened sample file (variable, kind, dims, units, members, labels)."""
    values, problems = res.values, res.problems
    gridded = [v for v in ds.data_vars if ds[v].ndim >= 3]
    if "variable" in values:
        specified("variable")
        if values["variable"] not in ds.data_vars:
            problems["variable"] = (f"{values['variable']!r} is not in {name}; variables: "
                                    f"{', '.join(map(str, ds.data_vars))}")
            return
    elif len(gridded) == 1:
        detected("variable", str(gridded[0]), f"only gridded variable in {name}")
    else:
        rainy = [v for v in gridded if _PRECIP_RE.search(str(v)) or _PRECIP_RE.search(
            str(ds[v].attrs.get("long_name", "")) + str(ds[v].attrs.get("standard_name", "")))]
        if len(rainy) == 1:
            detected("variable", str(rainy[0]), f"only precipitation-like variable in {name} "
                                                f"(of {', '.join(map(str, gridded))})")
        else:
            problems["variable"] = (f"cannot be detected: {name} has {len(gridded)} gridded variables "
                                    f"({', '.join(map(str, gridded))}). Fix: set variable")
            return
    da = ds[values["variable"]]

    probe = {}
    lead_dims = [d for d in da.dims if str(d).lower() in _ROLE_ALIASES["step"]
                 or any(s in str(d).lower() for s in ("lead", "step", "timedelta"))]
    looks = "forecast" if lead_dims else "observations"
    if looks != kind:
        problems["kind"] = (f"the dataset is described as {kind} but {name} looks like {looks} data "
                            f"(dims: {', '.join(map(str, da.dims))}). Fix: put it under "
                            f"{'models' if looks == 'forecast' else 'observations'}")
        return
    roles = _roles(da, kind, probe)
    if "dims" in values:
        specified("dims")
        for role, dim in values["dims"].items():
            if dim not in da.dims:
                problems[f"dims.{role}"] = (f"{dim!r} is not a dim of {values['variable']} in {name} "
                                            f"(dims: {', '.join(map(str, da.dims))})")
        roles.update({r: d for r, d in values["dims"].items() if d in da.dims})
        probe = {k: m for k, m in probe.items() if k.split(".")[1] not in values["dims"]}
        values["dims"] = {**roles, **values["dims"]} if not problems else values["dims"]
    else:
        problems.update(probe)
        if not probe:
            detected("dims", dict(roles), f"known dimension names in {name}: " +
                     ", ".join(f"{r}={d}" for r, d in roles.items()))
            values["dims"] = dict(roles)
    if probe and "dims" in values:
        problems.update(probe)

    lat, lon = roles.get("lat"), roles.get("lon")
    if lat in ds.coords and lon in ds.coords and ds[lat].size and ds[lon].size:
        res.extent = (float(ds[lat].min()), float(ds[lat].max()), float(ds[lon].min()), float(ds[lon].max()))

    if "member" in roles and kind == "forecast":
        size = int(da.sizes[roles["member"]])
        if "members" in values:
            specified("members")
            if values["members"] != size:
                res.warnings.append(f"members is {values['members']} but {name} has {size} values on dim "
                                    f"'{roles['member']}'")
        else:
            detected("members", size, f"member dim '{roles['member']}' has {size} values in {name}")
    elif kind == "forecast" and "members" not in values:
        res.lines.append(Line("members", 1, "detected", f"no member dimension in {name}: deterministic"))

    probe_da = load_sample(da, roles, key=(sample, _mtime(sample), str(values["variable"])))
    guess = _forecast_accumulation(probe_da, values, res, specified, detected, roles) if kind == "forecast" else None
    _units(da, name, values, res, specified, detected, roles, guess, probe_da)

    if kind == "observations":
        if "day_boundary" in values:
            specified("day_boundary")
        else:
            text = da.attrs.get("day_boundary") or ds.attrs.get("day_boundary")
            if text:
                detected("day_boundary", str(text), f"'day_boundary' attribute in {name}")
            else:
                problems["day_boundary"] = ("cannot be read from the file. Fix: set day_boundary to when the "
                                            "observation day starts and which date labels it, e.g. "
                                            "'00 UTC, labelled by start' (see the dataset's documentation)")
    else:
        if "daily_boundary" in values:
            specified("daily_boundary")
        if "lead_label" in values:
            specified("lead_label")
        elif lead_label_default and not detect_lead_label and not problems:
            res.lines.append(Line("lead_label", None, "default", "derived from the observation day boundary; set "
                                  "onset.check_alignment: true to check it against the observations"))
        elif not detect_lead_label or problems:
            problems["lead_label"] = ("not set. Fix: set lead_label to 'start' (lead k is dated init + k - 1) "
                                      "or 'end' (init + k)")
        else:
            label, evidence = _lead_label(values, sample, obs, roles)
            if label:
                detected("lead_label", label, evidence)
            else:
                problems["lead_label"] = (f"cannot be detected: {evidence}. Fix: set lead_label to 'start' "
                                          "(lead k is dated init + k - 1) or 'end' (init + k)")


def _forecast_accumulation(da, values, res, specified, detected, roles):
    """Resolve a forecast's accumulation (daily or since_init) and return the units guess (units, evidence) or None.

    A given accumulation is kept; otherwise it is detected with ingest.detect_accumulation (an ambiguous series is a
    problem naming the key). The units guess is ingest.detect_units on the daily totals of that accumulation.
    """
    lead = roles.get("step")
    kind = values.get("accumulation")
    if "accumulation" in values:
        specified("accumulation")
    elif lead is None:
        values["accumulation"] = kind = "auto"
        res.lines.append(Line("accumulation", "auto", "default", "no lead dimension to look at"))
    else:
        try:
            kind, evidence = ingest.detect_accumulation(da, lead)
            detected("accumulation", kind, evidence)
        except ValueError as exc:
            text = str(exc)
            if text.startswith("Ambiguous"):
                res.problems["accumulation"] = text.replace("; set accumulation='daily' or 'since_init'", "") + \
                    ". Fix: set accumulation to 'daily' or 'since_init'"
                return None
            values["accumulation"] = kind = "auto"
            res.lines.append(Line("accumulation", "auto", "default", f"could not be detected ({text}); worked out when loaded"))
    if lead is None or kind not in ("daily", "since_init"):
        return None
    try:
        return ingest.detect_units(da, kind, lead)
    except (ValueError, KeyError):
        return None


def _mtime(path) -> int:
    """Modification time of a file (or zarr directory) in ns, 0 if it cannot be read; part of a sample's cache key."""
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return 0


def _units(da, name, values, res, specified, detected, roles, guess=None, loaded=None):
    """Units (mm or m): given or read from the attribute; contradictions and unknowns are problems naming units.

    guess is ingest.detect_units' (units, evidence) for forecasts: it names the likely units when none can be read,
    and a given or attribute unit that disagrees with it is a warning. loaded is the in-memory sample of da
    (load_sample); the magnitudes are read from it once, whichever checks need them.
    """
    attr = da.attrs.get("units")
    from_attr = _units_from_attr(attr) if attr is not None else None
    sampled = _sample(da if loaded is None else loaded, roles)
    peak = sampled
    hint = ""
    if guess:
        hint = f" Daily totals look like {guess[0]} ({guess[1]})."
    elif peak.size:
        hint = (f" The sampled values peak at {peak.max():.3g}, which looks like "
                f"{'metres (units: m)' if peak.max() < 0.5 else 'mm (units: mm)'}.")
    if "units" in values:
        specified("units")
        if values["units"] not in ("mm", "m"):
            res.problems["units"] = f"{values['units']!r} is not supported. Fix: use units: mm or units: m"
            return
        if from_attr and from_attr != values["units"]:
            res.problems["units"] = (f"is {values['units']!r} but the 'units' attribute of {da.name} in {name} says "
                                     f"{attr!r}. Fix: set units to the right one, or remove it to use the attribute")
            return
        implied = values["units"]
    elif from_attr:
        detected("units", from_attr, f"'units' attribute {attr!r} of {da.name} in {name}")
        implied = from_attr
    else:
        why = f"unrecognised units attribute {attr!r}" if attr is not None else "the file has no units attribute"
        res.problems["units"] = f"cannot be detected: {why}.{hint} Fix: set units to mm or m"
        return
    message = _check_magnitude(sampled, implied)
    if message:
        res.problems["units"] = message + ". Fix: set the right units"
    elif guess and guess[0] != implied:
        res.warnings.append(f"units are {implied} but the daily totals look like {guess[0]} ({guess[1]}); "
                            "check units")


def finalize_units(values: dict) -> dict:
    """Add unit_cvt (the multiplier to mm the pipeline needs) derived from units; returns values."""
    from momp.lib.dataset_schema import unit_factor

    if "units" in values:
        values["unit_cvt"] = unit_factor(values["units"], "units")
    return values


def check_entry(values: dict, kind: str) -> None:
    """Validate completed values against the dataset schema (raises DatasetError)."""
    parse_dataset({k: v for k, v in values.items() if k not in ("name",) or v}, "dataset", kind, require_id=False)


def warn_all(res: Resolution) -> None:
    """Issue the resolution's warnings as UserWarnings too (so they show in logs and tests)."""
    for message in res.warnings:
        warnings.warn(message, UserWarning, stacklevel=3)
