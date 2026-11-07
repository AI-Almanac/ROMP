"""Complete a dataset description from its files, and check what the user wrote against them.

`resolve_entry` takes the keys a user gave for one dataset (``path``, ``file_pattern``, ``variable``,
``units`` (``"mm"`` or ``"m"``), ``dims``, ``day_boundary``, ``lead_label``, ``members``) and returns every key
filled in. A key the user gave always wins; a missing key is detected by opening the first matching file lazily and
is reported with its evidence, never silently. A key that cannot be detected reliably (several candidate variables,
units missing from the file, day boundary, a lead label the observations do not settle) is a *problem* naming the key
to set. A key the user gave that the files contradict (``units: mm`` with values that look like metres, a dim that
does not exist) is an error for names and a warning for units.

What is detected, and how:

* zarr store: a ``.zarr`` path is one dataset holding every year: no ``file_pattern``; the years come from the
  time/init coordinate, the rest from the store's metadata and a tiny slice.
* ``file_pattern``: the year-like number in the file names in ``path``, replaced by ``{}``.
* kind: forecast if the data has a lead/step dimension, else observations (checked, not set).
* ``variable``: the only gridded variable, else the only precipitation-like one.
* ``units``: the file's ``units`` attribute (mm or m), sanity-checked against the values.
* ``dims``: known dimension names (as `hcwf.onset.ingest`), role by role.
* ``members``: the size of the member dimension.
* ``accumulation``: daily or since_init, from how the values grow along the lead
  (`hcwf.onset.ingest.detect_accumulation`).
* ``lead_label``: forecast against observations at day lags -3..+3 (`hcwf.onset.alignment`; the sampled
  `hcwf.onset.sample_alignment` when a dataset is a zarr store).
* ``day_boundary``: only an explicit ``day_boundary`` attribute in the file.
* region: the lat/lon extent against the domains in ``hcwf/onset/regions`` (`detect_region`).

The second half of the module (forecast signatures: `read_forecast_file`, `forecast_signature`,
`signature_differences`) describes what a forecast file holds, so that a trained blend can compare it with every live
file. The datasets' file handling lives in `hcwf.config.datasets`; the result of `resolve_entry` is cached on disk
through `hcwf.cache.Store`.
"""
from __future__ import annotations

import os
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.cache import Store, file_identity, global_root, memo
from hcwf.config.datasets import expand_path, file_year, is_zarr, resolve_paths, year_match
from hcwf.onset import ingest
from hcwf.io.dims import matches
from hcwf.io.stores import data_files, open_data, pattern_files
from hcwf.errors import DatasetError

__all__ = ["cache_dir", "forecast_signature", "open_data", "read_forecast_file", "signature_differences"]


_PRECIP_RE = re.compile(r"(?i)precip|rain|prcp|(^|_)(tp|pr)($|_)")
"""Variable (or long/standard name) patterns that mark a precipitation-like variable."""
_ORDER = ("path", "paths", "years", "file_pattern", "variable", "units", "unit_cvt", "dims", "day_boundary",
          "lead_label", "accumulation", "daily_boundary", "members")
"""The order in which the keys of a dataset are reported."""
_ALIGN_INITS = 40
"""The most issue dates read for the lead-label alignment check."""
_NULLABLE = ("day_boundary", "lead_label")
"""Keys where an explicit null is a choice (the region's boundary; derive the label)."""


@dataclass
class Line:
    """One resolved key: its value, how it was obtained, and the evidence."""

    key: str
    """The config key, e.g. ``"variable"``."""
    value: Any
    """The resolved value."""
    how: str
    """How it was obtained: ``"specified"``, ``"detected"``, ``"detected (cached)"``, ``"default"`` or a dataset-file
    source such as ``"dataset 'x'"``."""
    evidence: str = ""
    """What the detection saw (empty when the key was specified)."""

    def text(self) -> str:
        """Return the report line, e.g. ``variable: tp  [detected: only gridded variable in 2020.nc]``.

        Returns
        -------
        str
            ``"<key>: <value>  [<how>[: <evidence>]]"``.
        """
        note = self.how + (f": {self.evidence}" if self.evidence else "")
        return f"{self.key}: {self.value}  [{note}]"


@dataclass
class Resolution:
    """The outcome for one dataset: completed values, a report line per key, problems and warnings."""

    values: dict
    """The dataset's keys, as given and as detected."""
    lines: list[Line] = field(default_factory=list)
    """One `Line` per resolved key, in the order they were resolved."""
    problems: dict = field(default_factory=dict)
    """Key (or ``dims.<role>``) to the message of a key that could not be resolved; empty when all is well."""
    warnings: list[str] = field(default_factory=list)
    """Messages about values that look wrong but do not stop the run."""
    extent: tuple | None = None
    """``(lat0, lat1, lon0, lon1)`` of the sampled file's coordinates (floats), or None when not read."""
    checked: bool = False
    """Whether a data file was found and opened (only such a resolution, without problems, is cached)."""
    region: str | None = None
    """The region the sample is taken in when the config names one (None: where the data are, see `load_sample`)."""
    box: tuple | None = None
    """The ``(south, north, west, east)`` of the run's area when it is a box (the config's ``box`` or a mask's bounds): the
    sample is taken around its centre, in place of the region's."""

    def report(self) -> list[str]:
        """Return the printable lines: every key with its provenance, then warnings and problems.

        Returns
        -------
        list of str
            ``Line.text`` for each line, then ``warning: ...`` for each warning, then ``PROBLEM <key>: ...`` for each
            problem.
        """
        out = [line.text() for line in self.lines]
        out += [f"warning: {w}" for w in self.warnings]
        out += [f"PROBLEM {k}: {m}" for k, m in self.problems.items()]
        return out


def find_pattern(names: list[str]) -> tuple[str | None, str]:
    """Find the file pattern shared by file names, with the year replaced by ``{}``.

    Parameters
    ----------
    names : list of str
        The file names (not paths) of a dataset's folder.

    Returns
    -------
    tuple of (str or None, str)
        ``(pattern, evidence)``; ``(None, reason)`` when no name holds a year or the names follow several patterns.
    """
    patterns = Counter()
    for name in names:
        match = year_match(name, whole_number=True)
        if match:
            patterns[name[:match.start()] + "{}" + name[match.end():]] += 1
    if not patterns:
        return None, "no file name contains a year such as 2020"
    if len(patterns) > 1:
        return None, "several name patterns: " + ", ".join(f"'{p}' ({n} files)" for p, n in patterns.items())
    (pattern, count), = patterns.items()
    return pattern, f"{count} file(s) named like {pattern!r}, e.g. {next(n for n in names if year_match(n, whole_number=True))}"


def _roles(da: xr.DataArray, kind: str, problems: dict) -> dict:
    """Map each dim role of the data to the file's dim name; ambiguous or missing roles go to problems.

    A string unit dimension (admin units, role ``"unit"``) stands in for lat and lon.

    Parameters
    ----------
    da : xarray.DataArray
        The variable whose dims are classified.
    kind : {"observations", "forecast"}
        Which roles are wanted: ``time`` for observations; ``init_time``, ``step`` and ``member`` for forecasts
        (``member`` is optional), plus ``lat`` and ``lon`` (or ``unit``).
    problems : dict
        Filled in place with ``dims.<role>`` to message for a role that is ambiguous or missing.

    Returns
    -------
    dict of str to str
        The role to the file's dim name, for each role found once.
    """
    dims = [str(d) for d in da.dims]
    unit = matches(dims, "unit")
    spatial = ("unit",) if unit else ("lat", "lon")
    wanted = (("time",) if kind == "observations" else ("init_time", "step", "member")) + spatial
    found = {}
    for role in wanted:
        hits = matches(dims, role, exclude=matches(dims, "step") if role == "init_time" else (), fuzzy=role == "step",
                       coords=da.coords if role == "step" else None)
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
_SAMPLE_MIN_CELLS = 32
"""Fewest lat or lon cells of the block a chunked file's sample reads: smaller stored chunks are joined to reach it."""
_SAMPLES = memo("config.detect.samples", maxsize=32)
"""Loaded detection samples, per (file, modification time, variable), for the run."""


def _block_indices(n: int, per_index: int) -> list[np.ndarray]:
    """Return index blocks (start, middle, end of a length-n dim), each a run of consecutive positions.

    The run length is at most ``_SAMPLE_BLOCK_LEN`` and shrinks so the blocks stay within ``_SAMPLE_BUDGET`` values
    (``per_index`` values per position), but is never below 1.

    Parameters
    ----------
    n : int
        The length of the dimension.
    per_index : int
        The number of values one position of the dimension holds.

    Returns
    -------
    list of numpy.ndarray of int64
        The positions of each block; a single block of every position when the dim is short.
    """
    length = int(max(1, min(_SAMPLE_BLOCK_LEN, _SAMPLE_BUDGET // max(1, _SAMPLE_BLOCKS * per_index))))
    if n <= _SAMPLE_BLOCKS * length:
        return [np.arange(n)]
    starts = sorted({0, (n - length) // 2, n - length})
    return [np.arange(s0, s0 + length) for s0 in starts]


def _cell_positions(n: int, runs, centre: int | None = None) -> np.ndarray:
    """Return the positions of the block a sample reads along a lat or lon axis.

    A chunked axis (any length) is cut to the stored chunk that holds its centre, joined with neighbouring chunks until
    it holds ``_SAMPLE_MIN_CELLS`` cells and cut to the ``_SAMPLE_CELLS`` nearest the centre if that is longer. An axis
    stored contiguously is read whole when it has at most ``_SAMPLE_CELLS`` cells, else cut to a centred block of
    ``_SAMPLE_CELLS``. The block is centred on ``centre`` (the middle of the axis when None).

    Parameters
    ----------
    n : int
        The length of the axis.
    runs : list of tuple of (int, int) or None
        The stored chunks along the axis (see `hcwf.spatial.select.chunk_runs`); None when it is not chunked.
    centre : int, optional
        The position the block is taken around; the middle of the axis when None.

    Returns
    -------
    numpy.ndarray of int64
        The positions, consecutive and increasing.
    """
    centre = n // 2 if centre is None else int(min(max(centre, 0), n - 1))
    if runs is None:
        if n <= _SAMPLE_CELLS:
            return np.arange(n)
        start = min(max(centre - _SAMPLE_CELLS // 2, 0), n - _SAMPLE_CELLS)
        return np.arange(start, start + _SAMPLE_CELLS)
    first = next(i for i, (lo, hi) in enumerate(runs) if lo <= centre < hi)
    lo = hi = first
    while runs[hi][1] - runs[lo][0] < _SAMPLE_MIN_CELLS and (lo > 0 or hi < len(runs) - 1):
        if hi < len(runs) - 1 and (lo == 0 or hi - first <= first - lo):
            hi += 1
        else:
            lo -= 1
    start, stop = runs[lo][0], runs[hi][1]
    if stop - start > _SAMPLE_CELLS:
        start = min(max(centre - _SAMPLE_CELLS // 2, start), stop - _SAMPLE_CELLS)
        stop = start + _SAMPLE_CELLS
    return np.arange(start, stop)


def _sample_pieces(wanted: np.ndarray, runs, how: str) -> list[np.ndarray]:
    """Split the wanted positions of one axis into the pieces that are each read with one range.

    Parameters
    ----------
    wanted : numpy.ndarray of int64
        The sorted positions the sample holds along the axis.
    runs : list of tuple of (int, int) or None
        The stored chunks along the axis; None when it is not chunked.
    how : {"merged", "runs", "groups"}
        ``"groups"``: a piece is a run of consecutive wanted positions. ``"merged"``: those, joined when two share a
        stored chunk (so that the chunk is decoded once). ``"runs"``: a piece is the wanted positions of one stored
        chunk. An axis that is not chunked is always cut into groups.

    Returns
    -------
    list of numpy.ndarray of int64
        The wanted positions of each piece.
    """
    groups = np.split(wanted, np.flatnonzero(np.diff(wanted) > 1) + 1)
    if runs is None or how == "groups":
        return groups
    starts = np.array([lo for lo, _ in runs])
    if how == "runs":
        return [piece for piece in np.split(wanted, np.searchsorted(wanted, starts[1:])) if len(piece)]
    merged = [groups[0]]
    for group in groups[1:]:
        if np.searchsorted(starts, merged[-1][-1], side="right") == np.searchsorted(starts, group[0], side="right"):
            merged[-1] = np.concatenate([merged[-1], group])
        else:
            merged.append(group)
    return merged


def _sample_plan(da, wanted: dict, runs: dict, dim: str) -> list[list[np.ndarray]]:
    """Choose the pieces of every axis so that each stored chunk is decoded once and no read is too big.

    The most joined cut is used unless its biggest read exceeds the block size of `hcwf.spatial.select.load_blocks`
    (a time chunk longer than the whole file's time steps, say); then the axes other than ``dim`` are cut at the chunk
    edges, and last ``dim`` is cut into its blocks of time steps (a chunk shared by two of them is then decoded for
    each).

    Parameters
    ----------
    da : xarray.DataArray
        The variable.
    wanted : dict of str to numpy.ndarray
        The positions the sample holds along each dim.
    runs : dict of str to list or None
        The stored chunks along each dim (see `hcwf.spatial.select.chunk_runs`).
    dim : str
        The time-like dim.

    Returns
    -------
    list of list of numpy.ndarray
        For each dim of ``da``, the wanted positions of its pieces.
    """
    from hcwf.spatial import select

    plan = []
    for how_dim, how_other in (("merged", "merged"), ("merged", "runs"), ("groups", "runs")):
        plan = [_sample_pieces(wanted[d], runs[d], how_dim if d == dim else how_other) for d in da.dims]
        biggest = da.dtype.itemsize * int(np.prod([max(int(p[-1]) + 1 - int(p[0]) for p in pieces)
                                                   for pieces in plan], dtype=np.int64))
        if biggest <= select._BLOCK_BYTES:
            break
    return plan


_CENTRES = ((0.5, 0.5), (0.5, 0.35), (0.5, 0.65), (0.35, 0.5), (0.65, 0.5), (0.35, 0.35), (0.35, 0.65), (0.65, 0.35),
            (0.65, 0.65), (0.5, 0.2), (0.5, 0.8), (0.2, 0.5), (0.8, 0.5), (0.2, 0.2), (0.2, 0.8), (0.8, 0.2),
            (0.8, 0.8))
"""The places (as fractions along latitude and longitude) a sample is looked for, in order, when the first has no values."""


def _region_centre(da: xr.DataArray, roles: dict, region: str | None, box: tuple | None = None) -> dict:
    """Find the grid positions of the centre of the area's box (a given box, else a region's) along lat and lon.

    Parameters
    ----------
    da : xarray.DataArray
        The variable, with its ``lat`` and ``lon`` coordinates.
    roles : dict of str to str
        The role to dim name map of ``da``.
    region : str or None
        The region name; None (or one that is unknown, or a grid without coordinates) gives no centre.
    box : tuple of float, optional
        The ``(south, north, west, east)`` of the run's area; it is used instead of the region's box.

    Returns
    -------
    dict of str to int
        The position along each of the lat and lon dims; empty when there is no centre.
    """
    lat, lon = roles.get("lat"), roles.get("lon")
    if (region is None and box is None) or lat not in da.coords or lon not in da.coords:
        return {}
    if box is not None:
        south, north, west, east = box
    else:
        from hcwf.onset.regions import box as region_domain

        try:
            south, north, west, east = region_domain(region)
        except Exception:  # noqa: BLE001 - an unknown region is reported by the region checks
            return {}
    east = east + 360.0 if east < west else east
    lon_c = (west + east) / 2
    lons = np.asarray(da[lon].values, dtype=float)
    lon_c = lon_c % 360 if lons.min() >= 0 else ((lon_c + 180) % 360) - 180
    lats = np.asarray(da[lat].values, dtype=float)
    return {lat: int(np.abs(lats - (south + north) / 2).argmin()), lon: int(np.abs(lons - lon_c).argmin())}


def load_sample(da: xr.DataArray, roles: dict, key=None, region: str | None = None,
                box: tuple | None = None) -> xr.DataArray:
    """Load a tiny in-memory copy of a variable: a few whole consecutive time steps from its start, middle and end.

    The grid is cut to a block around the centre of the ``region``'s box when one is given, else around the middle of
    the grid. A block with no finite value (the ocean of a land-only product, a global grid's centre) is not a sample:
    the next places of `_CENTRES` are tried, up to the first with values.

    The time-like dim is the ``init_time`` (forecasts) or ``time`` (observations) role, else the first dim. Lead is
    read in full (accumulation needs the whole lead series); the member dim is cut to its first member and a grid
    larger than ``_SAMPLE_CELLS`` is cut to a block (`_cell_positions`), so the read stays within
    ``_SAMPLE_BUDGET`` values (a few MB).

    The read follows the file's chunk grid (`hcwf.spatial.select.chunk_runs`): the block is one stored chunk of lat and
    lon (more where a chunk is small), and every stored chunk that holds a wanted value, whole, is read once by
    `hcwf.spatial.select.read_box` (the time steps and the member wanted are then taken in memory). A variable that is
    stored contiguously is read as ranges of the wanted positions.

    Parameters
    ----------
    da : xarray.DataArray
        The (lazy) variable.
    roles : dict of str to str
        The role to dim name map of ``da`` (see ``_roles``).
    key : hashable, optional
        A cache key, e.g. ``(path, mtime, variable)``; when given the result is kept in this process for the run.
    region : str, optional
        The region whose box centre the sample is taken at.
    box : tuple of float, optional
        The ``(south, north, west, east)`` of the run's area, whose centre the sample is taken at instead.

    Returns
    -------
    xarray.DataArray
        The loaded sample, with the attributes of ``da``.
    """
    if key is not None and key in _SAMPLES:
        return _SAMPLES[key]
    grid = [roles.get("lat"), roles.get("lon")]
    places = [_region_centre(da, roles, region, box)] if region or box else []
    if all(d in da.dims for d in grid):
        places += [{}] + [{d: int(f * (da.sizes[d] - 1)) for d, f in zip(grid, fractions)}
                          for fractions in _CENTRES[1:]]
    out = None
    for centre in places or [{}]:
        candidate = _load_sample_at(da, roles, centre)
        out = candidate if out is None else out
        if not np.issubdtype(candidate.dtype, np.floating) or np.isfinite(candidate.values).any():
            out = candidate
            break
    if key is not None:
        _SAMPLES[key] = out
    return out


def _load_sample_at(da: xr.DataArray, roles: dict, centre: dict) -> xr.DataArray:
    """Load the sample of `load_sample` with the grid block around the given positions.

    Parameters
    ----------
    da : xarray.DataArray
        The (lazy) variable.
    roles : dict of str to str
        The role to dim name map of ``da``.
    centre : dict of str to int
        The position along the lat and lon dims the block is centred on; a dim that is left out is centred in the middle.

    Returns
    -------
    xarray.DataArray
        The loaded sample, with the attributes of ``da``.
    """
    from hcwf.spatial import select

    dim = next((roles[r] for r in ("init_time", "time") if roles.get(r) in da.dims), None) or da.dims[0]
    runs = {d: select.chunk_runs(da, axis) for axis, d in enumerate(da.dims)}
    wanted, per_index = {}, 1
    for d in da.dims:
        n = da.sizes[d]
        if d == roles.get("member"):
            wanted[d] = np.arange(1)
        elif d in (roles.get("lat"), roles.get("lon")):
            wanted[d] = _cell_positions(n, runs[d], centre.get(d))
        elif d != dim:
            wanted[d] = np.arange(n)
        if d != dim:
            per_index *= len(wanted[d]) if d not in (roles.get("lat"), roles.get("lon")) else min(n, _SAMPLE_CELLS)
    wanted[dim] = np.concatenate(_block_indices(da.sizes[dim], per_index))
    pieces = _sample_plan(da, wanted, runs, dim)   # one range per piece of every axis
    data = None
    for corner in np.ndindex(*[len(p) for p in pieces]):
        parts = [pieces[a][c] for a, c in enumerate(corner)]
        box = [(int(p[0]), int(p[-1]) + 1) for p in parts]
        block = select.read_box(da, box)
        if data is None:
            data = np.empty(tuple(len(wanted[d]) for d in da.dims), dtype=block.dtype)
        here = [p - a for (a, _), p in zip(box, parts)]
        place = [np.searchsorted(wanted[d], p) for d, p in zip(da.dims, parts)]
        data[np.ix_(*place)] = block[np.ix_(*here)]
    template = da.isel({d: wanted[d] for d in da.dims})
    out = xr.DataArray(data, dims=da.dims, coords=template.coords, name=da.name, attrs=dict(da.attrs))
    out.encoding = dict(da.encoding)
    return out


def _sample(da: xr.DataArray, roles: dict) -> np.ndarray:
    """Sample the data values, spread out, as a flat array with fill values removed.

    Cheap on an in-memory DataArray (see `load_sample`); on lazy data it would decode chunks repeatedly. At most 40
    positions along lat and lon and 12 along any other dim are taken.

    Parameters
    ----------
    da : xarray.DataArray
        The variable, preferably loaded.
    roles : dict of str to str
        The role to dim name map of ``da``.

    Returns
    -------
    numpy.ndarray of float64, shape (value,)
        The finite values with magnitude below ``1e5``.
    """
    index = {}
    for dim in da.dims:
        n = da.sizes[dim]
        cap = 40 if dim in (roles.get("lat"), roles.get("lon")) else 12
        index[dim] = np.unique(np.linspace(0, n - 1, min(n, cap)).astype(int))
    values = np.asarray(da.isel(index).values, dtype="float64").ravel()
    return values[np.isfinite(values) & (np.abs(values) < 1e5)]


def _units_from_attr(text) -> str | None:
    """Return ``"mm"`` or ``"m"`` for a units attribute HCWF can convert, else None.

    Parameters
    ----------
    text : object
        The ``units`` attribute of the file; it is lower-cased and whitespace-normalised before the lookup in
        ``ingest.MM_PER_UNIT``.

    Returns
    -------
    str or None
        ``"mm"`` for a factor of 1, ``"m"`` for any other known factor, None for an unknown unit.
    """
    key = " ".join(str(text).lower().split())
    factor = ingest.MM_PER_UNIT.get(key)
    return None if factor is None else ("mm" if factor == 1.0 else "m")


def _check_magnitude(values: np.ndarray, implied: str | None) -> str | None:
    """Return a warning if the values contradict the units (mm or m), else None.

    Parameters
    ----------
    values : numpy.ndarray of float64, shape (value,)
        The sampled values (see ``_sample``).
    implied : {"mm", "m"} or None
        The units the file is read in.

    Returns
    -------
    str or None
        The warning: no finite values; ``"mm"`` with a peak below 0.5; ``"m"`` with a peak above 5. None when they agree.
    """
    if values.size == 0:
        return "the sampled values are all missing; units could not be checked against them"
    peak = float(values.max())
    if implied == "mm" and 0 < peak < 0.5:
        return f"units are mm but the sampled values peak at {peak:.3g}, which looks like metres (check units)"
    if implied == "m" and peak > 5:
        return f"units are m but the sampled values reach {peak:.3g}, which looks like mm (check units)"
    return None


def detect_region(extent: tuple) -> tuple[str, str]:
    """Find the region whose domain best overlaps the data extent, with the evidence.

    Parameters
    ----------
    extent : tuple of (float, float, float, float)
        The data extent ``(lat0, lat1, lon0, lon1)`` in degrees.

    Returns
    -------
    tuple of (str, str)
        ``(region id, evidence)``; ``("unknown", reason)`` when no domain overlaps by at least 30% (intersection
        over union).
    """
    from hcwf.onset import regions

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
    """Detect ``lead_label`` by aligning the forecast with the observations.

    Reads the first observation file of the sample's year and the first issue dates of the sample forecast file,
    ingests the forecast under both labels and keeps the label whose alignment check says ``aligned``. Datasets given
    as zarr stores go to ``_lead_label_sampled``; datasets given by ``paths`` are not checked.

    Parameters
    ----------
    values : dict
        The forecast's resolved keys so far (``variable``, ``units``, ``paths``, ...).
    sample : str
        The forecast file read.
    obs : dict or None
        The resolved observations entry (needs ``variable`` and ``day_boundary``).
    roles : dict of str to str
        The forecast's role to dim name map (``init_time`` and ``step``, optionally ``member``).

    Returns
    -------
    tuple of (str or None, str)
        ``(label, evidence)``, or ``(None, reason)`` when the check cannot run or is not conclusive.
    """
    from hcwf.onset.alignment import check_forecast_alignment

    if obs is None or not obs.get("variable") or not obs.get("day_boundary"):
        return None, "needs the observations (with variable and day_boundary) to compare against"
    if obs.get("paths") or values.get("paths"):
        year = file_year(sample, whole_number=not values.get("paths"))
        if is_zarr(sample) or not obs.get("paths") and is_zarr(expand_path(obs["path"])):
            return None, "the sampled check does not read datasets given by paths. Fix: set lead_label"
        try:
            listed = dataset_files(obs) if obs.get("paths") else [path for found, path in pattern_files(
                expand_path(obs["path"]), obs["file_pattern"]) if not year or found == year]
        except DatasetError as exc:
            return None, str(exc)
        files = [str(f) for f in listed if year is None or file_year(f) == year or not obs.get("paths")]
        if not files:
            return None, f"no observation file for year {year if year else '?'}"
    else:
        obs_dir = expand_path(obs["path"])
        if is_zarr(sample) or is_zarr(obs_dir):
            return _lead_label_sampled(values, sample, obs, roles, obs_dir)
        year = file_year(sample, whole_number=True)
        files = [path for found, path in pattern_files(obs_dir, obs["file_pattern"]) if not year or found == year]
        if not files:
            return None, f"no observation file for year {year if year else '?'} in {obs_dir}"
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


def _lead_label_sampled(values: dict, sample: str, obs: dict, roles: dict, obs_dir: str) -> tuple[str | None, str]:
    """Detect ``lead_label`` with the sampled alignment check (`hcwf.onset.sample_alignment`), for zarr stores.

    The sample (a few dozen inits and a few thousand cells, all lags at once) is read once and judged under
    both labels; exactly one aligned label settles it, anything else is inconclusive.

    Parameters
    ----------
    values : dict
        The forecast's resolved keys so far.
    sample : str
        The forecast file or zarr store.
    obs : dict
        The resolved observations entry.
    roles : dict of str to str
        The forecast's role to dim name map; only the presence of ``member`` is used.
    obs_dir : str
        The expanded observations folder.

    Returns
    -------
    tuple of (str or None, str)
        ``(label, evidence)``, or ``(None, reason)`` when the check cannot run or is not conclusive.
    """
    from hcwf.onset import sample_alignment as align_sample
    from hcwf.config.datasets import unit_factor

    try:
        obs_units, fc_units = obs.get("units"), values.get("units")
        obs_pattern = obs.get("file_pattern") or "{}.nc"
        model_pattern = values.get("file_pattern") or "{}.nc"
        model_dir = sample if is_zarr(sample) else os.path.dirname(sample)
        kw = {"obs_dir": obs_dir, "obs_file_pattern": [obs_pattern], "obs_var": obs["variable"],
              "obs_unit_cvt": unit_factor(obs_units, "units") if obs_units else None, "model_dir": model_dir,
              "model_var": values["variable"], "unit_cvt": unit_factor(fc_units, "units") if fc_units else None,
              "file_pattern": model_pattern, "model_dims": values.get("dims"), "probabilistic": "member" in roles,
              "accumulation": values.get("accumulation"), "daily_boundary": values.get("daily_boundary"),
              "region": None, "years": None}
        done = align_sample.check_labels(kw, ("start", "end"), obs_boundary=obs["day_boundary"], source="forecast")
    except (ValueError, KeyError, OSError, DatasetError) as exc:
        return None, f"the alignment check could not run: {exc}"
    verdicts = {label: report.verdict for label, (report, _) in done.items()}
    aligned = [label for label, verdict in verdicts.items() if verdict == "aligned"]
    detail = "; ".join(f"{label}: {verdict}" for label, verdict in verdicts.items())
    if len(aligned) == 1:
        return aligned[0], f'forecast lines up with observations at lag 0 under "{aligned[0]}" (sampled check; {detail})'
    return None, f"the alignment check is not conclusive ({detail})"


def cache_dir() -> Path:
    """Return the folder of the on-disk detection cache: ``$HCWF_CACHE_DIR/detect``, else ``~/.cache/hcwf/detect``.

    Returns
    -------
    pathlib.Path
        The ``detect`` folder under `hcwf.cache.global_root`.
    """
    return global_root() / "detect"


def _store() -> Store:
    """Return the detection cache: a `hcwf.cache.Store` of kind ``"detect"`` in `cache_dir` (nothing kept in memory).

    Returns
    -------
    hcwf.cache.Store
        The store; every read gives a fresh copy.
    """
    return Store("detect", cache_dir().parent, label="Dataset detection", memory=False)


def dataset_files(entry: dict) -> list[Path]:
    """Return the files a dataset entry reads: its ``paths`` (see `~hcwf.config.datasets.resolve_paths`), the zarr store, or the NetCDF files.

    Parameters
    ----------
    entry : dict
        The dataset keys: ``path`` and/or ``paths`` and ``years``.

    Returns
    -------
    list of pathlib.Path
        The listed files, or the zarr store, or the sorted ``.nc`` files and zarr stores in the folder.

    Notes
    -----
    A missing ``path`` without ``paths`` surfaces as the `TypeError` of ``Path(None)``; callers check it first.
    """
    base = expand_path(entry["path"]) if entry.get("path") else None
    if entry.get("paths"):
        return resolve_paths(entry["paths"], base, entry.get("years"))
    folder = Path(base)
    return [folder] if is_zarr(folder) else data_files(folder)


def _parts(entry: dict, kind: str, lead_label_default: bool, sources: dict | None = None,
           region: str | None = None, box: tuple | None = None) -> dict | None:
    """Return what detection depends on: the user's keys, and the identity (path, size, mtime) of the dataset's files.

    Parameters
    ----------
    entry : dict
        The keys the user gave for the dataset.
    kind : {"observations", "forecast"}
        The kind of the dataset.
    lead_label_default : bool
        Whether an unset ``lead_label`` is left to be derived from the day boundary.
    sources : dict, optional
        Where a key came from if not the user.
    region : str, optional
        The region the sample is taken in (part of the fingerprint when given).
    box : tuple of float, optional
        The area's box the sample is taken in (part of the fingerprint when given): the sampled place depends on it.

    Returns
    -------
    dict or None
        The parts of the cache fingerprint; None when the folder cannot be listed or a file read (nothing to cache
        against).
    """
    try:
        files = dataset_files(entry)
    except (KeyError, DatasetError, OSError):
        return None
    identities = [file_identity(path) for path in files]
    if not identities or any(i[1] is None for i in identities):
        return None
    parts = {"kind": kind, "lead_label_default": lead_label_default,
             "entry": {k: repr(v) for k, v in entry.items()}, "sources": {k: repr(v) for k, v in (sources or {}).items()},
             "files": identities}
    if region is not None:
        parts["region"] = region
    if box is not None:
        parts["box"] = [float(v) for v in box]
    return parts


def _fingerprint(entry: dict, kind: str, lead_label_default: bool, sources: dict | None = None) -> str | None:
    """Return the cache fingerprint of a dataset's detection (see ``_parts``), or None when it cannot be cached.

    Parameters
    ----------
    entry : dict
        The keys the user gave for the dataset.
    kind : {"observations", "forecast"}
        The kind of the dataset.
    lead_label_default : bool
        Whether an unset ``lead_label`` is left to be derived from the day boundary.
    sources : dict, optional
        Where a key came from if not the user.

    Returns
    -------
    str or None
        The fingerprint, or None when ``_parts`` is None.
    """
    parts = _parts(entry, kind, lead_label_default, sources)
    return None if parts is None else _store().fingerprint(parts)


def resolve_entry(entry: dict, kind: str, *, sources: dict | None = None, obs: dict | None = None,
                  detect_lead_label: bool = True, lead_label_default: bool = False, use_cache: bool = True,
                  region: str | None = None, box: tuple | None = None) -> Resolution:
    """Resolve one dataset, from the on-disk cache when the dataset's files and keys are unchanged.

    The cache key is a fingerprint of the user's keys, the file paths, sizes and mtimes, and the place the data are
    sampled (the region or area box); a hit reads no data and marks every detected line "detected (cached)" with the same
    evidence. Only a resolution without problems is stored (its warnings are kept with it): a failed or partial detection
    is recomputed the next time. A resolution that ran the lead-label alignment (it depends on the observations) is not
    cached. See `resolve_entry_uncached` for the keys.

    Parameters
    ----------
    entry : dict
        The keys the user gave (raw, as in a dataset file).
    kind : {"observations", "forecast"}
        The kind of the dataset.
    sources : dict, optional
        Where a key came from if not the user (``"dataset 'x'"``).
    obs : dict, optional
        The resolved observations entry, needed to detect a forecast's ``lead_label``.
    detect_lead_label : bool, default True
        Whether to detect ``lead_label`` by the alignment check when it is not given.
    lead_label_default : bool, default False
        Leave an unset ``lead_label`` to be derived from the day boundary instead of making it a problem.
    use_cache : bool, default True
        Whether to read and write the on-disk cache.
    region : str, optional
        The region the config names: the data are sampled around the centre of its box (see `load_sample`).
    box : tuple of float, optional
        The ``(south, north, west, east)`` of the run's area (a ``box`` or a mask's bounds): the sample is taken around
        its centre, and it is part of the cache key.

    Returns
    -------
    Resolution
        The completed values, report lines, problems and warnings.
    """
    runs_alignment = kind == "forecast" and detect_lead_label and "lead_label" not in entry
    parts = _parts(entry, kind, lead_label_default, sources, region, box) if use_cache and not runs_alignment else None
    if parts:
        hit = _store().get(parts, quiet=True)
        if isinstance(hit, Resolution):
            for line in hit.lines:
                if line.how == "detected":
                    line.how = "detected (cached)"
            hit.values = {**hit.values}
            return hit
    res = resolve_entry_uncached(entry, kind, sources=sources, obs=obs, detect_lead_label=detect_lead_label,
                                 lead_label_default=lead_label_default, region=region, box=box)
    if parts and res.checked and not res.problems:   # a failed or partial detection is recomputed, never served again
        _store().put(parts, res)
    return res


def resolve_entry_uncached(entry: dict, kind: str, *, sources: dict | None = None, obs: dict | None = None,
                           detect_lead_label: bool = True, lead_label_default: bool = False,
                           region: str | None = None, box: tuple | None = None) -> Resolution:
    """Resolve one dataset: validate what is given, detect what is missing, report every key.

    Never raises for problems with the data: they are collected in `Resolution.problems`.

    Parameters
    ----------
    entry : dict
        The keys the user gave (raw, as in a dataset file).
    kind : {"observations", "forecast"}
        The kind of the dataset.
    sources : dict, optional
        Where a key came from if not the user (``"dataset 'x'"``).
    obs : dict, optional
        The resolved observations entry, needed to detect a forecast's ``lead_label``.
    detect_lead_label : bool, default True
        Whether to detect ``lead_label`` by the alignment check when it is not given.
    lead_label_default : bool, default False
        When ``lead_label`` is not given and not detected (``detect_lead_label`` false), leave it unset ("derived from
        the day boundary") instead of making it a problem.
    region : str, optional
        The region the config names: the data are sampled around the centre of its box (see `load_sample`).
    box : tuple of float, optional
        The ``(south, north, west, east)`` of the run's area, whose centre the data are sampled around instead.

    Returns
    -------
    Resolution
        The completed values, report lines, problems and warnings.
    """
    sources = sources or {}
    res = Resolution(values={k: v for k, v in entry.items() if v is not None or k in _NULLABLE}, region=region,
                     box=box)
    values, problems = res.values, res.problems

    def specified(key):
        """Record a key the user (or a dataset file) gave."""
        if key in values:
            res.lines.append(Line(key, values[key], sources.get(key, "specified")))

    def detected(key, value, evidence):
        """Record a detected key."""
        values[key] = value
        res.lines.append(Line(key, value, "detected", evidence))

    paths = values.get("paths")
    if not values.get("path") and not paths:
        problems["path"] = "missing. Fix: set path to the folder of the yearly files, or paths to the files"
        return res
    try:
        folder = Path(expand_path(values["path"])) if values.get("path") else None
    except DatasetError as exc:
        problems["path"] = str(exc)
        return res
    if folder is not None:
        specified("path")
    if paths:
        return _resolve_listed(values, kind, res, specified, detected, obs, detect_lead_label, lead_label_default)
    if is_zarr(folder):
        return _resolve_zarr(folder, kind, res, specified, detected, obs, detect_lead_label, lead_label_default)
    names = [p.name for p in data_files(folder)] if folder.is_dir() else []
    needed = [k for k in ("file_pattern", "variable", "day_boundary", "lead_label", "units") if k not in values]
    if kind == "forecast" and "day_boundary" in needed:
        needed.remove("day_boundary")
    if kind == "observations" and "lead_label" in needed:
        needed.remove("lead_label")
    if "unit_cvt" in values and "units" in needed:
        needed.remove("units")
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
    files = [path for _, path in pattern_files(folder, values["file_pattern"])] if "file_pattern" in values \
        else [str(p) for p in data_files(folder)]
    if not files:
        problems["file_pattern"] = f"no file in {folder} matches {values.get('file_pattern')!r}"
        return res
    sample = files[0]
    res.checked = True
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


def _resolve_listed(values, kind, res, specified, detected, obs, detect_lead_label, lead_label_default):
    """Resolve a dataset that names its files in ``paths``: no ``file_pattern``; everything from the first file.

    The files are the listed ones (`~hcwf.config.datasets.resolve_paths`), several per year if they are; the first holds the layout.

    Parameters
    ----------
    values : dict
        The dataset keys (``res.values``).
    kind : {"observations", "forecast"}
        The kind of the dataset.
    res : Resolution
        The resolution being built; filled in place.
    specified : callable
        ``specified(key)`` records a key the user gave.
    detected : callable
        ``detected(key, value, evidence)`` records a detected key.
    obs : dict or None
        The resolved observations entry.
    detect_lead_label : bool
        Whether to detect ``lead_label`` by the alignment check.
    lead_label_default : bool
        Whether an unset ``lead_label`` is left to be derived from the day boundary.

    Returns
    -------
    Resolution
        ``res``.
    """
    problems = res.problems
    for key in ("paths", "years"):
        specified(key)
    try:
        files = dataset_files(values)
    except DatasetError as exc:
        problems["paths"] = str(exc)
        return res
    sample = str(files[0])
    res.checked = True
    res.lines.append(Line("file_pattern", "not needed", "detected", f"{len(files)} file(s) listed in paths"))
    try:
        ds = open_data(sample)
    except Exception as exc:  # noqa: BLE001 - any reader error is reported against the path
        problems["paths"] = f"cannot open {sample}: {exc}"
        return res
    with ds:
        _resolve_from_file(ds, os.path.basename(sample), sample, kind, res, specified, detected, obs,
                           detect_lead_label, lead_label_default)
    return res


def _store_years(ds: xr.Dataset) -> tuple[str, int, int] | None:
    """Return the name and year range of the first 1-D datetime coordinate of a store.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened store.

    Returns
    -------
    tuple of (str, int, int) or None
        ``(coordinate name, first year, last year)``, or None when no such coordinate exists.
    """
    for name, coord in ds.coords.items():
        if coord.ndim == 1 and np.issubdtype(coord.dtype, np.datetime64) and coord.size:
            years = coord.dt.year
            return str(name), int(years.min()), int(years.max())
    return None


def _resolve_zarr(folder, kind, res, specified, detected, obs, detect_lead_label, lead_label_default):
    """Resolve a dataset that is one zarr store holding every year: no ``file_pattern``, everything from the metadata.

    The years come from the time (or init) coordinate, the variable, units, dims and members from the store's
    metadata and a tiny slice for the value checks; ``day_boundary`` and ``lead_label`` are as for NetCDF files.

    Parameters
    ----------
    folder : pathlib.Path
        The zarr store.
    kind : {"observations", "forecast"}
        The kind of the dataset.
    res : Resolution
        The resolution being built; filled in place.
    specified : callable
        ``specified(key)`` records a key the user gave.
    detected : callable
        ``detected(key, value, evidence)`` records a detected key.
    obs : dict or None
        The resolved observations entry.
    detect_lead_label : bool
        Whether to detect ``lead_label`` by the alignment check.
    lead_label_default : bool
        Whether an unset ``lead_label`` is left to be derived from the day boundary.

    Returns
    -------
    Resolution
        ``res``.
    """
    values, problems = res.values, res.problems
    try:
        ds = open_data(str(folder))
    except Exception as exc:  # noqa: BLE001 - any reader error is reported against the path
        problems["path"] = f"cannot open the zarr store {folder}: {exc}"
        return res
    res.checked = True
    span = _store_years(ds)
    evidence = "zarr store: one dataset holding every year" + (
        f" ({span[1]}-{span[2]} from coordinate '{span[0]}')" if span else "")
    if "file_pattern" in values:
        specified("file_pattern")
    else:
        res.lines.append(Line("file_pattern", "not needed", "detected", evidence))
    with ds:
        _resolve_from_file(ds, Path(folder).name, str(folder), kind, res, specified, detected, obs,
                           detect_lead_label, lead_label_default)
    return res


def _resolve_from_file(ds, name, sample, kind, res, specified, detected, obs, detect_lead_label,
                       lead_label_default=False):
    """Run the detections that need the opened sample file (variable, kind, dims, units, members, labels).

    Everything is recorded in ``res`` (values, lines, problems, warnings, extent); a problem that makes the rest
    meaningless (unknown variable, wrong kind, ambiguous variable) ends the detection early.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened sample file.
    name : str
        The file name, for messages.
    sample : str
        The path of the sample file.
    kind : {"observations", "forecast"}
        The kind of the dataset.
    res : Resolution
        The resolution being built; filled in place.
    specified : callable
        ``specified(key)`` records a key the user gave.
    detected : callable
        ``detected(key, value, evidence)`` records a detected key.
    obs : dict or None
        The resolved observations entry.
    detect_lead_label : bool
        Whether to detect ``lead_label`` by the alignment check.
    lead_label_default : bool, default False
        Whether an unset ``lead_label`` is left to be derived from the day boundary.

    Returns
    -------
    None
        Nothing; the outcome is in ``res``.
    """
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
    lead_dims = matches([str(d) for d in da.dims], "step", fuzzy=True, coords=da.coords)
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
    if lat is not None and lat in ds.coords and lon in ds.coords and ds[lat].size and ds[lon].size:
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

    probe_da = load_sample(da, roles, key=(sample, _mtime(sample), str(values["variable"]), res.region, res.box),
                           region=res.region, box=res.box)
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
    """Resolve a forecast's accumulation (daily or since_init) and return the units guess.

    A given accumulation is kept; otherwise it is detected with ``ingest.detect_accumulation`` (an ambiguous series is
    a problem naming the key; one that cannot be read is set to ``"auto"``, to be worked out when loaded). The units
    guess is ``ingest.detect_units`` on the daily totals of that accumulation.

    Parameters
    ----------
    da : xarray.DataArray
        The loaded sample of the variable.
    values : dict
        The dataset keys; ``accumulation`` is set here.
    res : Resolution
        The resolution being built; filled in place.
    specified : callable
        ``specified(key)`` records a key the user gave.
    detected : callable
        ``detected(key, value, evidence)`` records a detected key.
    roles : dict of str to str
        The role to dim name map (``step`` is the lead dim).

    Returns
    -------
    tuple or None
        The ``(units, evidence)`` guess, or None when there is no lead dim, the accumulation is ambiguous or
        ``"auto"``, or the units cannot be guessed.
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
    """Return the modification time of a file (or zarr directory), part of a sample's cache key.

    Parameters
    ----------
    path : str or os.PathLike
        The file or directory.

    Returns
    -------
    int
        ``st_mtime_ns`` in nanoseconds, 0 if it cannot be read.
    """
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return 0


def _units(da, name, values, res, specified, detected, roles, guess=None, loaded=None):
    """Resolve the units (mm or m): given or read from the attribute; contradictions and unknowns are problems.

    ``guess`` is the ``(units, evidence)`` of ``ingest.detect_units`` for forecasts: it names the likely units when none
    can be read, and a given or attribute unit that disagrees with it is a warning. ``loaded`` is the in-memory sample
    of ``da`` (`load_sample`); the magnitudes are read from it once, whichever checks need them.

    Parameters
    ----------
    da : xarray.DataArray
        The variable (its ``units`` attribute is read).
    name : str
        The file name, for messages.
    values : dict
        The dataset keys; ``unit_cvt`` is dropped when the file declares its units.
    res : Resolution
        The resolution being built; filled in place.
    specified : callable
        ``specified(key)`` records a key the user gave.
    detected : callable
        ``detected(key, value, evidence)`` records a detected key.
    roles : dict of str to str
        The role to dim name map of ``da``.
    guess : tuple, optional
        The ``(units, evidence)`` guess from the daily totals (forecasts only).
    loaded : xarray.DataArray, optional
        The loaded sample of ``da``; ``da`` itself is sampled when None.

    Returns
    -------
    None
        Nothing; the outcome is in ``res`` and ``values``.
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
    if "unit_cvt" in values and "units" not in values:
        if from_attr:  # declared units win: unit_cvt is the factor of files that declare none
            values.pop("unit_cvt")
            detected("units", from_attr, f"'units' attribute {attr!r} of {da.name} in {name} (unit_cvt is only for "
                                         "files that declare no units)")
            implied = from_attr
        else:
            specified("unit_cvt")
            return
    elif "units" in values:
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
    if message and not sampled.size:   # nothing to check against: the units are given or declared, so only a warning
        res.warnings.append(message + f"; units are taken as {implied} (as given or declared in the file)")
    elif message:
        res.problems["units"] = message + ". Fix: set the right units"
    elif guess and guess[0] != implied:
        res.warnings.append(f"units are {implied} but the daily totals look like {guess[0]} ({guess[1]}); "
                            "check units")


def finalize_units(values: dict) -> dict:
    """Add ``unit_cvt`` (the multiplier to mm the pipeline needs), derived from ``units``.

    Parameters
    ----------
    values : dict
        The resolved dataset keys; changed in place.

    Returns
    -------
    dict
        ``values``, with ``unit_cvt`` set when ``units`` is present.
    """
    from hcwf.config.datasets import unit_factor

    if "units" in values:
        values["unit_cvt"] = unit_factor(values["units"], "units")
    return values


def warn_all(res: Resolution) -> None:
    """Issue the resolution's warnings as ``UserWarning`` too (so they show in logs and tests).

    Parameters
    ----------
    res : Resolution
        The resolution whose ``warnings`` are issued.
    """
    for message in res.warnings:
        warnings.warn(message, UserWarning, stacklevel=3)


# ---------------------------------------------------------------------------------------------------------------
# Forecast signatures: what a forecast file holds, apart from its issue dates
# ---------------------------------------------------------------------------------------------------------------
#
# A trained blend stores the signature of its training files and compares it with every live forecast file before
# anything is read. The signature is made of the file's resolved dataset entry (`resolve_entry`: the variable, the
# rainfall units it is read in, the accumulation, the dimension roles) plus what the entry does not say: the lead days,
# the member ids, the grid (or unit ids), the issue hours of the day, other dimension sizes and how the file says its
# daily values were aggregated (its cell methods). Comparing two signatures is `signature_differences`; the stored
# signature of an older blend may hold fewer fields, and only the fields it holds are compared.

_AGGREGATION_PAIR = re.compile(r"([^\s:]+):\s*([^\s:]+)")


def _lead_numbers(values) -> list[int]:
    """Return the lead days of a step coordinate: whole days, from numbers or timedeltas.

    Parameters
    ----------
    values : array_like
        The step coordinate values, numbers (days) or ``timedelta64``.

    Returns
    -------
    list of int
        The lead days.
    """
    values = np.asarray(values).reshape(-1)
    if values.dtype.kind == "m":
        values = values / np.timedelta64(1, "D")
    return [int(v) for v in values]


def _issue_stamps(values) -> tuple[list[int], list]:
    """Return the issue hours and dates of an init-time coordinate.

    Parameters
    ----------
    values : array_like
        The init-time coordinate values (decoded datetimes or cftime objects).

    Returns
    -------
    tuple of (list of int, list of datetime.date)
        The sorted distinct hours of day, and the sorted distinct dates.
    """
    import datetime as dt

    values = np.asarray(values).reshape(-1)
    if values.dtype.kind == "M":
        stamps = [t.to_pydatetime() for t in pd.DatetimeIndex(values)]
    else:
        stamps = list(values)
    return sorted({int(t.hour) for t in stamps}), sorted({dt.date(t.year, t.month, t.day) for t in stamps})


def _aggregation(ds, variable, lead_dim: str | None) -> dict:
    """Return how the file says its daily values were made.

    That is the cell method over its lead dimension (e.g. ``sum``) and the window its lead coordinate states (e.g. the
    UTC day ending at the issue time plus the lead); None where it does not say.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened file.
    variable : xarray.DataArray
        The rainfall variable (its ``cell_methods`` attribute is read).
    lead_dim : str or None
        The file's lead dimension, or None when there is none.

    Returns
    -------
    dict of str to str or None
        ``{"method": ..., "window": ...}``, lower-cased.
    """
    method = None
    if lead_dim is not None:
        for name, how in _AGGREGATION_PAIR.findall(str(variable.attrs.get("cell_methods", "") or "")):
            if name in (lead_dim, "day") or lead_dim.startswith(name):
                method = how.lower()
    window = None
    if lead_dim is not None and lead_dim in ds.variables:
        attrs = ds[lead_dim].attrs
        text = attrs.get("description") or attrs.get("long_name")
        window = " ".join(str(text).lower().split()) if text else None
    return {"method": method, "window": window}


def read_forecast_file(path, entry: dict, *, lead_days=None, max_member=None, selected: bool = True,
                       year: int | None = None) -> tuple[dict, list]:
    """Read the signature and issue dates of a forecast file or zarr store.

    The units, accumulation, members and dimension roles are detected from the file. A zarr store holds every year in
    one array; ``year`` describes the slice of the issues of that year, so that the years of a store are compared as
    the files of a model are.

    Parameters
    ----------
    path : str or os.PathLike
        The forecast file, or the zarr store.
    entry : dict
        The dataset keys the file is resolved with: ``variable``, and optionally ``dims`` (role to file dimension) and
        ``unit_cvt``, as `resolve_entry` takes them.
    lead_days : tuple of (int, int), optional
        The model's lead-day window (first, last), applied when ``selected``.
    max_member : int, optional
        The model's highest member id, applied when ``selected``.
    selected : bool, default True
        Give the signature after ``lead_days`` and ``max_member``, as training reads the file; else everything the
        (live) file holds.
    year : int, optional
        Describe only the issues of this year (``issue_hours`` and the dates); None describes all the issues.

    Returns
    -------
    tuple of (dict, list)
        The signature (``variable``, ``units``, ``accumulation``, ``dims``, ``aggregation`` and, where the file has
        them, ``issue_hours``, ``lead_days``, ``members``, ``lat``, ``lon``, ``unit_ids``) and the sorted issue
        dates.

    Raises
    ------
    ValueError
        The file cannot be described: it cannot be opened, has no such variable, its units cannot be read, or its
        dimensions cannot be told apart.
    """
    path, name = str(path), Path(str(path)).name
    given = {"path": path, "variable": entry["variable"]} if is_zarr(path) else {"paths": [path], "variable": entry["variable"]}
    if entry.get("unit_cvt") is not None:
        given["unit_cvt"] = entry["unit_cvt"]
    if entry.get("dims"):
        try:
            with open_data(path) as ds:
                present = set(map(str, ds[entry["variable"]].dims)) if entry["variable"] in ds.variables else None
        except Exception as exc:  # noqa: BLE001 - any reader error is reported against the file
            raise ValueError(f"{name}: cannot open it: {exc}") from None
        # A role the file has no dimension for (a deterministic file of an ensemble blend) is simply absent.
        dims = {role: dim for role, dim in entry["dims"].items() if present is None or dim in present}
        if dims:
            given["dims"] = dims
    res = resolve_entry(given, "forecast", detect_lead_label=False, lead_label_default=True)
    values = res.values
    problems = dict(res.problems)
    problems.pop("lead_label", None)    # the live file cannot say it; the run's own is used
    if problems.pop("accumulation", None) is not None or "accumulation" not in values:
        values["accumulation"] = "auto"      # a series that does not say is left to be worked out when read
    if problems:
        raise ValueError(f"{name}: " + "; ".join(f"{key}: {message}" for key, message in problems.items()))
    roles = values["dims"]
    units = "mm" if "units" in values else f"undeclared (x{float(values.get('unit_cvt', 1.0)):g})"
    signature: dict = {"variable": entry["variable"], "units": units, "accumulation": values["accumulation"],
                       "dims": {}}
    dates: list = []
    with open_data(path) as ds:
        variable = ds[entry["variable"]]
        by_name = {file_dim: role for role, file_dim in roles.items()}
        signature["aggregation"] = _aggregation(ds, variable, roles.get("step"))
        for dim in variable.dims:
            role = by_name.get(str(dim))
            if dim in ds.variables:
                found = np.asarray(ds[dim].values)
            else:
                found = np.arange(ds.sizes[dim])
            if role == "init_time":
                if year is not None and found.dtype.kind == "M":
                    found = found[pd.DatetimeIndex(found).year == year]
                signature["issue_hours"], dates = _issue_stamps(found)
            elif role == "step":
                days = _lead_numbers(found)
                if selected and lead_days is not None:
                    days = [d for d in days if lead_days[0] <= d <= lead_days[1]]
                signature["lead_days"] = days
            elif role == "member":
                members = [int(v) for v in found]
                if selected and max_member is not None:
                    members = [m for m in members if m <= max_member]
                signature["members"] = members
            elif role in ("lat", "lon"):
                signature[role] = sorted(round(float(v), 6) for v in found)
            elif role == "unit":
                signature["unit_ids"] = sorted(str(v) for v in found)
            else:
                signature["dims"][str(dim)] = len(found)
    return signature, dates


def forecast_signature(path, entry: dict, **options) -> dict:
    """Return what a forecast file holds, apart from its issue dates (see `read_forecast_file`).

    Parameters
    ----------
    path : str or os.PathLike
        The forecast file.
    entry : dict
        The dataset keys, as for `read_forecast_file`.
    **options : dict
        Passed to `read_forecast_file`: ``lead_days``, ``max_member``, ``selected``.

    Returns
    -------
    dict
        The signature.

    """
    return read_forecast_file(path, entry, **options)[0]


def signature_differences(trained: dict, live: dict, *, fewer_members: bool = False, grid: bool = True) -> list[str]:
    """Describe how a live file fails to cover the trained signature, one reason each (empty when it covers it).

    It may hold more (longer leads, more members, a larger domain such as the globe): a blend reads only what it was
    trained on. It may not hold less, nor another grid spacing, units, accumulation, daily aggregation, issue hour or
    variable. Only the fields the trained signature holds are compared.

    Parameters
    ----------
    trained : dict
        The signature of the training files (see `read_forecast_file`); an older blend may hold fewer fields.
    live : dict
        The signature of the live file.
    fewer_members : bool, default False
        A live ensemble may hold only some of the trained members (any one will do); its onset probabilities are then
        shares of those.
    grid : bool, default True
        Compare the grids' latitudes and longitudes. A blend that stored the grid it was trained on leaves this to
        `hcwf.blend.trained.TrainedGrid.problems`, which lets a finer live grid be averaged onto the stored one; the
        presence of a grid is still compared.

    Returns
    -------
    list of str
        One message per difference; empty when the live file covers the trained signature.
    """
    out = []
    for key, label in (
        ("variable", "variable"), ("units", "rainfall units"), ("accumulation", "accumulation"),
        ("aggregation", "daily aggregation"), ("issue_hours", "issue hour (UTC)"), ("dims", "other dimensions"),
    ):
        if key in trained and trained.get(key) != live.get(key):
            out.append(f"{label} {_brief(live.get(key))}, trained on {_brief(trained.get(key))}")
    for key, label in (("lead_days", "lead days"), ("members", "members"), ("unit_ids", "unit ids")):
        if key not in trained and key not in live:
            continue
        have, need = live.get(key), trained.get(key)
        if (have is None) != (need is None):
            out.append(f"{label} {_brief(have)}, trained on {_brief(need)}")
        elif key == "members" and fewer_members and need is not None and set(need) & set(have):
            continue
        elif need is not None and set(need) - set(have):
            missing = sorted(set(need) - set(have))
            out.append(f"{label} {_brief(have)}, trained on {_brief(need)} (missing {_brief(missing)})")
    for key, label in (("lat", "latitudes"), ("lon", "longitudes")):
        if key not in trained and key not in live:
            continue
        have, need = live.get(key), trained.get(key)
        if (have is None) != (need is None):
            out.append(f"{label} {_brief(have)}, trained on {_brief(need)}")
        elif need is not None and not grid:
            continue
        elif need is not None:
            have_step, need_step = _grid_step(have), _grid_step(need)
            if None not in (have_step, need_step) and have_step != need_step:
                out.append(f"{label} every {_degrees(have_step)}, trained on every {_degrees(need_step)} "
                           "(another resolution)")
            elif set(need) - set(have):
                out.append(f"{label} {_span(have)}, trained on {_span(need)}: the trained domain is not covered")
    return out


def _grid_step(values: list) -> float | None:
    """Return a grid's spacing, the median gap between its distinct values.

    Parameters
    ----------
    values : list of float
        The coordinate values of one axis.

    Returns
    -------
    float or None
        The spacing in degrees rounded to 6 decimals; None for fewer than two distinct values.
    """
    distinct = np.unique(values)
    return round(float(np.median(np.diff(distinct))), 6) if len(distinct) > 1 else None


def _degrees(step: float | None) -> str:
    """Format a grid step in degrees for a message.

    Parameters
    ----------
    step : float or None
        The spacing in degrees.

    Returns
    -------
    str
        For example ``"0.25 degrees"`` (with the degree sign); ``"?"`` for None.
    """
    return "?" if step is None else f"{step:g}\N{DEGREE SIGN}"


def _span(values: list) -> str:
    """Format the extent of a list of coordinate values for a message.

    Parameters
    ----------
    values : list of float
        The coordinate values.

    Returns
    -------
    str
        ``"min-max"`` with an en dash, or ``"none"`` for an empty list.
    """
    return f"{min(values):g}\N{EN DASH}{max(values):g}" if values else "none"


def _brief(value) -> str:
    """Abbreviate a signature value for a message: a long list becomes a range or a count with its ends.

    Parameters
    ----------
    value : object
        A signature field.

    Returns
    -------
    str
        ``"a-b"`` for a run of consecutive integers longer than 6, ``"n values a...b"`` for another long list,
        else ``repr(value)``.
    """
    if isinstance(value, list) and len(value) > 6:
        if all(isinstance(v, int) for v in value) and value == list(range(value[0], value[-1] + 1)):
            return f"{value[0]}\N{EN DASH}{value[-1]}"
        return f"{len(value)} values {value[0]}\N{HORIZONTAL ELLIPSIS}{value[-1]}"
    return repr(value)
