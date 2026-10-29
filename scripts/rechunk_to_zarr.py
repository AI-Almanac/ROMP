#!/usr/bin/env python
"""Rechunk yearly NetCDF rainfall datasets into one zarr (v3) store per dataset.

The source tree is mirrored: every directory under ``--src`` that contains
``*.nc`` files (one file per year) becomes ``<dst>/<same relative path>.zarr``,
e.g. ``india/obs`` -> ``data/india/obs.zarr`` and
``india/models/AIFS_v2`` -> ``data/india/models/AIFS_v2.zarr``.

The conversion is a lossless rechunk: variable names, attrs, coordinates and
dimension order are kept (no unit conversion, no latitude flip), the data
variable is stored as float32 (a float64 source is only down-cast after
checking that every value survives the round trip exactly, otherwise the task
fails), and the files, their sizes and mtimes plus the chunking are recorded
in the group attribute ``rechunk_to_zarr``.  Codec: Blosc zstd, byte shuffle.
Metadata is consolidated when a store is finished.

Chunking (target ``--target-mb`` of float32 per chunk, default 64 MB)

* Observations ``(time, lat, lon)``: the FULL time axis in every chunk, with
  near-square spatial tiles.  Reading the whole history of a pixel or tile
  touches the minimum number of chunks.
* Forecasts ``(init, [member], lead, lat, lon)``: all members and the full
  lead axis in every chunk.  The access pattern is "all leads of many inits
  for a spatial tile", so a chunk keeps the whole lead/member extent together
  and varies only init (1-``--max-init-chunk``, default 8) and space.  When one
  init over the whole grid fits the budget the grid is kept whole and several
  inits are stacked (deterministic models); otherwise one init per chunk and
  the grid is cut into the fewest, most square tiles that fit (large
  ensembles).  Chunks smaller than the target occur when the 8-init cap is hit
  on small grids.

Parallelism: a single process pool (spawn) works on all datasets at once.  Each
task assembles a disjoint, chunk-aligned region of one store in memory (an
init-time block for forecasts, a latitude band of chunks for observations) and
writes it with one assignment, so no chunk is ever written by two tasks.
Finished tasks leave a marker in ``<store>.progress/``; finished stores carry
``complete: true`` in their attrs.  Re-running skips both, and resumes an
interrupted store (its chunk plan must be unchanged, else use ``--overwrite``).

Examples::

    python scripts/rechunk_to_zarr.py --src ../ai-almanac/datasets --dst data
    python scripts/rechunk_to_zarr.py --only 'india/models/*' --verify --benchmark
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import fnmatch
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

ATTR_KEY = "rechunk_to_zarr"
STATS_FILE = "rechunk_stats.json"
BLOCK_BYTES = 256 * 2**20  # in-memory size of one write task


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _jsonable(x):
    """Convert numpy scalars/arrays (nested) to JSON-compatible python."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def _dir_bytes(path: Path) -> int:
    """Total apparent size of all files below ``path``."""
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def _balanced_tiles(h: int, w: int, max_px: int) -> tuple[int, int]:
    """Fewest, most square tiles ``(th, tw)`` of a ``h x w`` grid with ``th*tw <= max_px``."""
    best, best_key = (1, 1), None
    for nl in range(1, h + 1):
        th = -(-h // nl)
        if th > max_px:
            continue
        tw = min(w, max_px // th)
        nw = -(-w // tw)
        tw = -(-w // nw)
        key = (-(-h // th) * nw, abs(math.log(th / tw)))
        if best_key is None or key < best_key:
            best, best_key = (th, tw), key
    return best


def choose_chunks(shape, dims, itemsize, time_dim, target_bytes, max_init=8):
    """Chunk shape for a ``(time, [member...], lead, lat, lon)`` variable (see module doc)."""
    nt, (h, w) = shape[0], shape[-2:]
    middle = shape[1:-2]
    if not middle:  # observations: full time axis, spatial tiles
        th, tw = _balanced_tiles(h, w, max(1, target_bytes // (nt * itemsize)))
        return (nt, th, tw)
    per_px = int(np.prod(middle)) * itemsize  # bytes per pixel per init
    if per_px * h * w > target_bytes:
        th, tw = _balanced_tiles(h, w, max(1, target_bytes // per_px))
        return (1, *middle, th, tw)
    n_init = int(min(max_init, nt, max(1, target_bytes // (per_px * h * w))))
    return (n_init, *middle, h, w)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
@dataclass
class Plan:
    """Everything needed to build and fill one store."""

    name: str
    src_dir: Path
    store: Path
    files: list  # [{path, name, size, mtime, n_time, start}]
    var: str
    dims: tuple
    shape: tuple
    chunks: tuple
    src_dtype: str
    dtype: str
    var_attrs: dict
    global_attrs: dict
    coords: dict = field(repr=False)  # name -> (dims, values, attrs)

    @property
    def progress(self) -> Path:
        return Path(str(self.store) + ".progress")

    @property
    def signature(self) -> dict:
        return {
            "files": [(f["name"], f["size"], f["mtime"], f["n_time"]) for f in self.files],
            "chunks": list(self.chunks),
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


def discover(src: Path, dst: Path, only: list[str] | None) -> list[tuple[str, Path, Path]]:
    """Find dataset directories (those holding ``*.nc``) -> ``(name, src_dir, store_path)``."""
    out = []
    for d in sorted({p.parent for p in Path(src).rglob("*.nc")}):
        name = d.relative_to(src).as_posix()
        if only and not any(fnmatch.fnmatch(name, o) or name == o or name.endswith("/" + o) for o in only):
            continue
        out.append((name, d, Path(dst) / (name + ".zarr")))
    return out


def scan(name: str, src_dir: Path, store: Path, target_bytes: int, max_init: int) -> Plan:
    """Lazily open every yearly file and build the :class:`Plan`; checks consistency."""
    files, times, first, src_dtypes = [], [], None, set()
    for f in sorted(Path(src_dir).glob("*.nc")):
        with xr.open_dataset(f) as ds:
            if len(ds.data_vars) != 1:
                raise ValueError(f"{f}: expected exactly one data variable, got {list(ds.data_vars)}")
            var = next(iter(ds.data_vars))
            da = ds[var]
            tdim = da.dims[0]
            if not np.issubdtype(ds[tdim].dtype, np.datetime64):
                raise ValueError(f"{f}: first dim {tdim!r} of {var!r} is not a datetime axis")
            src_dtypes.add(str(da.dtype))
            sig = (var, da.dims, np.dtype(da.dtype).kind, {k: ds[k].values for k in ds.coords if k != tdim})
            if first is None:
                first = (sig, dict(da.attrs), dict(ds.attrs), {k: (ds[k].dims, ds[k].attrs) for k in ds.coords})
                other_shape = da.shape[1:]
            else:
                s0 = first[0]
                same = sig[:3] == s0[:3] and da.shape[1:] == other_shape
                same = same and all(np.array_equal(sig[3][k], s0[3][k]) for k in s0[3])
                if not same:
                    raise ValueError(f"{f}: dims/dtype/coordinates differ from {files[0]['name']}")
            t = ds[tdim].values
            times.append(t)
            st = f.stat()
            files.append({"path": str(f), "name": f.name, "size": st.st_size, "mtime": st.st_mtime, "n_time": len(t)})
    if not files:
        raise ValueError(f"no .nc files in {src_dir}")
    time_all = np.concatenate(times)
    if np.any(np.diff(time_all.astype("int64")) <= 0):
        raise ValueError(f"{name}: time axis is not strictly increasing across files")
    start = 0
    for f in files:
        f["start"] = start
        start += f["n_time"]
    sig, vattrs, gattrs, cmeta = first
    var, dims, src_dtype = sig[0], sig[1], sig[2]
    coords = {dims[0]: ((dims[0],), time_all, cmeta[dims[0]][1])}
    for k, v in sig[3].items():
        coords[k] = (cmeta[k][0], v, cmeta[k][1])
    shape = (len(time_all), *other_shape)
    dtype = "float32" if src_dtype == "f" else sorted(src_dtypes)[0]  # float32/float64 sources may be mixed
    src_dtype = ",".join(sorted(src_dtypes))
    chunks = choose_chunks(shape, dims, np.dtype(dtype).itemsize, dims[0], target_bytes, max_init)
    return Plan(name, Path(src_dir), Path(store), files, var, dims, shape, chunks, src_dtype, dtype,
                _jsonable(vattrs), _jsonable(gattrs), coords)


# --------------------------------------------------------------------------
# Store creation and tasks
# --------------------------------------------------------------------------
def _meta(plan: Plan, complete: bool, **extra) -> dict:
    return {
        "complete": complete,
        "signature": _jsonable(plan.signature),
        "source_dir": str(plan.src_dir),
        "source_files": [{k: f[k] for k in ("name", "size", "mtime", "n_time")} for f in plan.files],
        "variable": plan.var,
        "dims": list(plan.dims),
        "chunks": list(plan.chunks),
        "codec": "blosc zstd clevel=5 byte-shuffle",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }


def _read_meta(store: Path) -> dict | None:
    """Attrs record of an existing store, or None if it does not exist / is unreadable."""
    try:
        g = zarr.open_group(str(store), mode="r", use_consolidated=False)
        return dict(g.attrs[ATTR_KEY])
    except Exception:
        return None


def init_store(plan: Plan, overwrite: bool) -> str:
    """Create the store (coords + empty data array) unless usable. Returns 'done'|'resume'|'new'."""
    if plan.store.exists() or plan.progress.exists():
        meta = _read_meta(plan.store)
        if not overwrite:
            if meta is not None and meta["signature"] == _jsonable(plan.signature):
                return "done" if meta["complete"] else "resume"
            raise SystemExit(
                f"{plan.store}: exists with a different plan or unreadable; use --overwrite to rebuild"
            )
        shutil.rmtree(plan.store, ignore_errors=True)
        shutil.rmtree(plan.progress, ignore_errors=True)
    plan.store.parent.mkdir(parents=True, exist_ok=True)
    plan.progress.mkdir(parents=True, exist_ok=True)
    template = xr.Dataset(coords={k: xr.Variable(d, v, a) for k, (d, v, a) in plan.coords.items()},
                          attrs=plan.global_attrs)
    template.to_zarr(plan.store, mode="w", zarr_format=3, consolidated=False)
    g = zarr.open_group(str(plan.store), mode="r+")
    g.create_array(
        plan.var, shape=plan.shape, chunks=plan.chunks, dtype=plan.dtype, fill_value=float("nan"),
        dimension_names=plan.dims, attributes=plan.var_attrs,
        compressors=BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle),
    )
    g.attrs[ATTR_KEY] = _meta(plan, False)
    return "new"


def make_tasks(plan: Plan, block_bytes: int = BLOCK_BYTES) -> list[dict]:
    """Disjoint, chunk-aligned write tasks (see module doc)."""
    nt, itemsize = plan.shape[0], np.dtype(plan.dtype).itemsize
    base = {"store": str(plan.store), "progress": str(plan.progress), "var": plan.var, "dims": plan.dims,
            "dtype": plan.dtype, "name": plan.name}
    tasks = []

    def segments(a, b):
        segs = []
        for f in plan.files:
            lo, hi = max(a, f["start"]), min(b, f["start"] + f["n_time"])
            if lo < hi:
                segs.append((f["path"], lo - f["start"], hi - f["start"], lo - a))
        return segs

    if plan.chunks[0] >= nt:  # full time axis: one task per row of spatial chunks
        ch = plan.chunks[-2]
        for i, a in enumerate(range(0, plan.shape[-2], ch)):
            b = min(a + ch, plan.shape[-2])
            tasks.append({**base, "id": f"band{i:05d}", "t0": 0, "t1": nt, "segs": segments(0, nt),
                          "lat": (a, b)})
    else:
        per_init = int(np.prod(plan.shape[1:])) * itemsize
        k = max(1, block_bytes // (per_init * plan.chunks[0]))
        step = plan.chunks[0] * k
        for i, a in enumerate(range(0, nt, step)):
            b = min(a + step, nt)
            tasks.append({**base, "id": f"blk{i:05d}", "t0": a, "t1": b, "segs": segments(a, b), "lat": None})
    return tasks


def _pool_init():
    zarr.config.set({"threading.max_workers": 2})


def run_task(task: dict) -> dict:
    """Worker: assemble one region from the NetCDF files and write it to the store."""
    marker = Path(task["progress"]) / (task["id"] + ".json")
    if marker.exists():
        return {**json.loads(marker.read_text()), "skipped": True, "name": task["name"]}
    t_start = time.time()
    dtype = np.dtype(task["dtype"])
    z = zarr.open_group(task["store"], mode="r+", use_consolidated=False)[task["var"]]
    lat = task["lat"]
    shp = list(z.shape)
    shp[0] = task["t1"] - task["t0"]
    if lat:
        shp[-2] = lat[1] - lat[0]
    buf = np.empty(shp, dtype)
    for path, s0, s1, d0 in task["segs"]:
        with xr.open_dataset(path) as ds:
            da = ds[task["var"]]
            idx = {da.dims[0]: slice(s0, s1)}
            if lat:
                idx[da.dims[-2]] = slice(*lat)
            src = da.isel(idx).values
        out = src.astype(dtype)
        if src.dtype != dtype and not np.array_equal(out, src, equal_nan=src.dtype.kind == "f"):
            raise ValueError(f"{path}: values are not exactly representable as {dtype}")
        buf[d0 : d0 + (s1 - s0)] = out
    sel = [slice(task["t0"], task["t1"])] + [slice(None)] * (len(shp) - 1)
    if lat:
        sel[-2] = slice(*lat)
    z[tuple(sel)] = buf
    rec = {"id": task["id"], "t0": t_start, "t1": time.time(), "bytes": int(buf.nbytes)}
    marker.write_text(json.dumps(rec))
    return {**rec, "name": task["name"], "skipped": False}


def finalize(plan: Plan) -> dict:
    """Mark the store complete (timings, sizes), then consolidate metadata."""
    recs = [json.loads(p.read_text()) for p in plan.progress.glob("*.json")]
    g = zarr.open_group(str(plan.store), mode="r+", use_consolidated=False)
    meta = dict(g.attrs[ATTR_KEY])
    meta.update(
        complete=True,
        write_cpu_seconds=sum(r["t1"] - r["t0"] for r in recs),
        write_wall_seconds=max(r["t1"] for r in recs) - min(r["t0"] for r in recs),
        finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    g.attrs[ATTR_KEY] = meta
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zarr.consolidate_metadata(str(plan.store))
    return meta


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def verify_tasks(plan: Plan, sample_years: int, seed: int, block_bytes: int = BLOCK_BYTES) -> list[dict]:
    """Exact-comparison tasks: all files for obs-like stores, else first/last + a random sample."""
    idx = list(range(len(plan.files)))
    if plan.chunks[0] < plan.shape[0] and len(idx) > 2:
        rest = idx[1:-1]
        idx = [0, *sorted(random.Random(seed).sample(rest, min(sample_years, len(rest)))), idx[-1]]
    out = []
    for i in idx:
        f = plan.files[i]
        per_t = int(np.prod(plan.shape[1:])) * np.dtype(plan.dtype).itemsize
        step = max(1, block_bytes // per_t)
        for a in range(0, f["n_time"], step):
            out.append({"store": str(plan.store), "var": plan.var, "name": plan.name, "file": f["path"],
                        "s0": a, "s1": min(a + step, f["n_time"]), "d0": f["start"] + a})
    return out


def run_verify(task: dict) -> dict:
    """Worker: compare one file slice with the store, exactly, including the NaN pattern."""
    with xr.open_dataset(task["file"]) as ds:
        da = ds[task["var"]]
        src = da.isel({da.dims[0]: slice(task["s0"], task["s1"])}).values
    z = zarr.open_group(task["store"], mode="r", use_consolidated=False)[task["var"]]
    dst = z[task["d0"] : task["d0"] + src.shape[0]]
    nan_s, nan_d = np.isnan(src), np.isnan(dst)
    bad = int(np.count_nonzero(~((src == dst) | (nan_s & nan_d))))
    return {"name": task["name"], "file": os.path.basename(task["file"]), "n": int(src.size),
            "n_bad": bad, "n_nan": int(nan_s.sum()), "nan_pattern_diff": int(np.count_nonzero(nan_s != nan_d))}


def _norm_attrs(a: dict) -> dict:
    return json.loads(json.dumps(_jsonable(a), sort_keys=True))


def verify_metadata(plan: Plan) -> list[str]:
    """Dims/coords/attrs/dtype/chunks of the store against the concatenated sources."""
    errs = []
    with xr.open_dataset(plan.store, engine="zarr", chunks=None, consolidated=True) as ds:
        da = ds[plan.var]
        if da.dims != plan.dims or da.shape != plan.shape:
            errs.append(f"dims/shape {da.dims}{da.shape} != {plan.dims}{plan.shape}")
        if da.dtype != np.dtype(plan.dtype):
            errs.append(f"dtype {da.dtype}")
        if tuple(da.encoding.get("chunks", ())) != tuple(plan.chunks):
            errs.append(f"chunks {da.encoding.get('chunks')}")
        for k, (d, v, a) in plan.coords.items():
            if ds[k].dims != tuple(d) or not np.array_equal(ds[k].values, v):
                errs.append(f"coord {k} differs")
            if _norm_attrs(ds[k].attrs) != _norm_attrs(a):
                errs.append(f"coord {k} attrs differ")
        extra = set(ds.coords) - set(plan.coords)
        if extra:
            errs.append(f"unexpected coords {extra}")
        if _norm_attrs(da.attrs) != _norm_attrs(plan.var_attrs):
            errs.append("variable attrs differ")
        ga = {k: v for k, v in ds.attrs.items() if k != ATTR_KEY}
        if _norm_attrs(ga) != _norm_attrs(plan.global_attrs):
            errs.append("global attrs differ")
        if ATTR_KEY not in ds.attrs or not ds.attrs[ATTR_KEY]["complete"]:
            errs.append("store not marked complete")
    return errs


# --------------------------------------------------------------------------
# Benchmark and report
# --------------------------------------------------------------------------
def benchmark(plan: Plan) -> dict:
    """Time reading all times/leads/members of the central spatial chunk tile: zarr vs NetCDFs."""
    ch_h, ch_w = plan.chunks[-2:]
    h, w = plan.shape[-2:]
    a, c = (-(-h // ch_h) // 2) * ch_h, (-(-w // ch_w) // 2) * ch_w
    sl = (slice(a, min(a + ch_h, h)), slice(c, min(c + ch_w, w)))
    t0 = time.perf_counter()
    with xr.open_dataset(plan.store, engine="zarr", chunks=None, consolidated=True) as ds:
        da = ds[plan.var]
        x = da.isel({da.dims[-2]: sl[0], da.dims[-1]: sl[1]}).values
    t_zarr = time.perf_counter() - t0
    t0 = time.perf_counter()
    parts = []
    for f in plan.files:
        with xr.open_dataset(f["path"]) as ds:
            da = ds[plan.var]
            parts.append(da.isel({da.dims[-2]: sl[0], da.dims[-1]: sl[1]}).values)
    y = np.concatenate(parts)
    t_nc = time.perf_counter() - t0
    return {"tile": [sl[0].start, sl[0].stop, sl[1].start, sl[1].stop], "tile_bytes": int(x.nbytes),
            "zarr_read_s": t_zarr, "netcdf_read_s": t_nc, "equal": bool(np.array_equal(x, y, equal_nan=True))}


def _load_stats(dst: Path) -> dict:
    p = Path(dst) / STATS_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def _save_stats(dst: Path, stats: dict):
    (Path(dst) / STATS_FILE).write_text(json.dumps(stats, indent=1))


def write_report(plans: list[Plan], stats: dict, path: Path):
    """Markdown table of source/store size, chunks, write time, read times and verification."""
    gb = lambda b: f"{b / 2**30:.2f}"  # noqa: E731
    rows = ["| dataset | files | source GiB | store GiB | shape | chunk | n chunks | write wall s | write cpu-s "
            "| nc tile read s | zarr tile read s | speedup | verify |", "|" + "---|" * 13]
    for p in plans:
        s = stats.get(p.name, {})
        meta = _read_meta(p.store) or {}
        n_ch = int(np.prod([-(-n // c) for n, c in zip(p.shape, p.chunks)]))
        b, v = s.get("benchmark"), s.get("verify")
        rows.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            p.name, len(p.files), gb(sum(f["size"] for f in p.files)), gb(_dir_bytes(p.store)),
            "x".join(map(str, p.shape)), "x".join(map(str, p.chunks)), n_ch,
            f"{meta.get('write_wall_seconds', float('nan')):.0f}", f"{meta.get('write_cpu_seconds', float('nan')):.0f}",
            f"{b['netcdf_read_s']:.2f}" if b else "-", f"{b['zarr_read_s']:.2f}" if b else "-",
            f"{b['netcdf_read_s'] / b['zarr_read_s']:.0f}x" if b else "-",
            ("OK " if v["ok"] else "FAIL ") + f"({v['n_files']} files, {v['n_values']:,} values)" if v else "-"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(rows) + "\n")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _map(fn, items, workers: int):
    """Yield ``fn(item)`` results as completed; inline when ``workers <= 1``."""
    if workers <= 1:
        for it in items:
            yield fn(it)
        return
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(workers, mp_context=ctx, initializer=_pool_init) as ex:
        futs = [ex.submit(fn, it) for it in items]
        for f in cf.as_completed(futs):
            yield f.result()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=Path("/home/hcwf/code/ai-almanac/datasets"), help="source tree (read only)")
    ap.add_argument("--dst", type=Path, default=Path(__file__).resolve().parents[1] / "data", help="output root")
    ap.add_argument("--only", action="append", help="dataset name or glob (e.g. india/obs, '*AIFS_ENS*'); repeatable")
    ap.add_argument("--workers", type=int, default=20, help="worker processes (1 = run inline)")
    ap.add_argument("--overwrite", action="store_true", help="delete and rebuild existing stores")
    ap.add_argument("--verify", action="store_true", help="exact comparison with the sources after writing")
    ap.add_argument("--verify-sample", type=int, default=4, help="random extra years verified for forecasts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true", help="time a full-time-axis tile read, zarr vs NetCDF")
    ap.add_argument("--report", type=Path, help="write the markdown table here")
    ap.add_argument("--target-mb", type=float, default=64.0, help="target uncompressed chunk size (MB)")
    ap.add_argument("--block-mb", type=float, default=256.0, help="in-memory size of one write/verify task (MB)")
    ap.add_argument("--max-init-chunk", type=int, default=8, help="max inits per forecast chunk")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.dst.mkdir(parents=True, exist_ok=True)
    block = int(args.block_mb * 2**20)
    found = discover(args.src, args.dst, args.only)
    if not found:
        print("no matching datasets", file=sys.stderr)
        return 1
    plans = [scan(n, s, d, int(args.target_mb * 2**20), args.max_init_chunk) for n, s, d in found]
    plans.sort(key=lambda p: -np.prod(p.shape))
    todo, tasks = {}, []
    for p in plans:
        state = init_store(p, args.overwrite)
        print(f"{p.name}: {state}; shape {p.shape} chunks {p.chunks}", flush=True)
        if state != "done":
            t = make_tasks(p, block)
            todo[p.name] = [p, len(t)]
            tasks += t
    t0 = time.time()
    for r in _map(run_task, tasks, args.workers):
        p, left = todo[r["name"]]
        todo[r["name"]][1] = left - 1
        if left == 1:
            finalize(p)
            print(f"{p.name}: complete ({time.time() - t0:.0f}s since start)", flush=True)
    stats = _load_stats(args.dst)
    if args.verify:
        vt = [t for p in plans for t in verify_tasks(p, args.verify_sample, args.seed, block)]
        res = {p.name: {"n_files": len({t["file"] for t in vt if t["name"] == p.name}), "n_values": 0,
                        "n_bad": 0, "nan_diff": 0, "n_nan": 0, "errors": verify_metadata(p)} for p in plans}
        for r in _map(run_verify, vt, args.workers):
            e = res[r["name"]]
            e["n_values"] += r["n"]
            e["n_bad"] += r["n_bad"]
            e["nan_diff"] += r["nan_pattern_diff"]
            e["n_nan"] += r["n_nan"]
        for name, e in res.items():
            e["ok"] = not (e["n_bad"] or e["nan_diff"] or e["errors"])
            stats.setdefault(name, {})["verify"] = e
            print(f"verify {name}: {'OK' if e['ok'] else 'FAIL'} {e}", flush=True)
        _save_stats(args.dst, stats)
    if args.benchmark:
        for p in plans:
            stats.setdefault(p.name, {})["benchmark"] = b = benchmark(p)
            print(f"bench {p.name}: {b}", flush=True)
        _save_stats(args.dst, stats)
    if args.report:
        write_report(plans, stats, args.report)
    bad = [n for n, s in stats.items() if n in {p.name for p in plans} and not s.get("verify", {"ok": True})["ok"]]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
