"""The one cache of momp: results kept under a fingerprint of everything they depend on.

A `Store` is a folder of entries for one *kind* of result (observed onsets, thresholds, harmonic seasons, dataset
detection, ...). An entry is addressed by its *parts*: a dict of named inputs (``{"files": ..., "definition": ...}``).
Each part is hashed by value (`part_hash`: arrays by their contents, string and object arrays by their strings, never
by pointer; mappings and sequences recursively; file identities by path, size and mtime), and the fingerprint is the
hash of the hashes. The parts are written next to the payload (``parts.json``), so on a miss the store can say which
parts differ from the nearest cached entry.

On disk: ``<root>/<kind>/<fingerprint>/`` holding ``parts.json`` and the payload file of the store's `Codec`. Writes
are atomic (a temporary file in the same folder, then ``os.replace``), so worker processes and concurrent runs can
share a folder: a reader sees a whole file or none, and two writers of one fingerprint write the same content.
Without a root the store is in memory only. A failure of the disk (read-only folder, damaged entry) is a miss and
never stops a run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

_META = frozenset(("zarr.json", ".zmetadata", ".zgroup", ".zattrs", ".zarray"))
"""Names of the metadata files of a zarr store, whose size and mtime are part of a directory's identity."""


def file_identity(path) -> list:
    """What identifies a file or directory (zarr stores included): [path, size, mtime_ns], [path, None, None] if missing.

    For a directory the size is the number of its entries and the mtime the newest of the directory, its entries
    and the metadata files one level down, so a rewritten zarr array or added file changes it without walking every chunk.
    """
    path = Path(path)
    try:
        stat = os.stat(path)
        if not path.is_dir():
            return [str(path), stat.st_size, stat.st_mtime_ns]
        newest, count = stat.st_mtime_ns, 0
        with os.scandir(path) as entries:
            for entry in entries:
                count += 1
                newest = max(newest, entry.stat().st_mtime_ns)
                if entry.is_dir():
                    with os.scandir(entry.path) as inner:
                        newest = max([newest] + [e.stat().st_mtime_ns for e in inner if e.name in _META])
        return [str(path), count, newest]
    except OSError:
        return [str(path), None, None]


def _feed(h, value) -> None:
    """Add a value to a hash, type-tagged, by value (see part_hash)."""
    if value is None:
        h.update(b"N")
    elif isinstance(value, (bool, np.bool_)):
        h.update(b"b1" if value else b"b0")
    elif isinstance(value, (int, float, np.number)):
        h.update(b"n" + repr(float(value)).encode())
    elif isinstance(value, str):
        h.update(b"s" + value.encode())
    elif isinstance(value, bytes):
        h.update(b"y" + value)
    elif isinstance(value, Path):
        h.update(b"p" + str(value).encode())
    elif isinstance(value, xr.Dataset):
        h.update(b"D")
        for name in sorted(value.data_vars, key=str):
            _feed(h, str(name))
            _feed(h, value[name])
    elif isinstance(value, (xr.DataArray, np.ndarray)):
        da = value if isinstance(value, xr.DataArray) else xr.DataArray(value)
        values = np.asarray(da.values)
        h.update(b"A" + repr((da.dims, da.shape)).encode())
        if values.dtype.kind in "OUS":   # strings (unit ids) and objects by value: tobytes() would be pointers
            h.update(b"u" + json.dumps(values.astype(str).ravel().tolist()).encode())
        else:
            h.update(values.dtype.str.encode() + np.ascontiguousarray(values).tobytes())
    elif isinstance(value, Mapping):
        h.update(b"{")
        for key in sorted(value, key=str):
            _feed(h, str(key))
            _feed(h, value[key])
        h.update(b"}")
    elif isinstance(value, (list, tuple)):
        h.update(b"[")
        for item in value:
            _feed(h, item)
        h.update(b"]")
    elif isinstance(value, (set, frozenset)):
        _feed(h, sorted(value, key=repr))
    else:
        h.update(b"r" + repr(value).encode())
    h.update(b"\0")


def part_hash(*values) -> str:
    """A 12-hex-character hash of the values: arrays by contents (strings by value), containers recursively."""
    h = hashlib.sha256()
    for value in values:
        _feed(h, value)
    return h.hexdigest()[:12]


def atomic_write(path, write: Callable[[Path], None]) -> None:
    """Make `path` by calling write(temporary path) and renaming it over `path`; nothing is left behind on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp{path.suffix}")
    try:
        write(temp)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class Codec:
    """How a store keeps a payload: its file name in the entry folder and the functions that read and write it."""

    filename: str
    read: Callable[[Path], Any]
    write: Callable[[Path, Any], None]


def _read_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _write_pickle(path, value):
    with open(path, "wb") as fh:
        pickle.dump(value, fh)


def _read_netcdf(path):
    with xr.open_dataset(path) as ds:
        loaded = ds.load()
    if "__value__" not in loaded:
        return loaded
    array = loaded["__value__"]
    array.name = array.attrs.pop("__name__", "") or None
    return array


def _write_netcdf(path, value):
    if isinstance(value, xr.DataArray):
        ds = value.to_dataset(name="__value__")
        ds["__value__"].attrs["__name__"] = "" if value.name is None else str(value.name)
        value = ds
    value.to_netcdf(path)


def _read_groups(path):
    with xr.open_dataset(path) as root:
        names = json.loads(root.attrs["names"])
    out = {}
    for name in names:
        with xr.open_dataarray(path, group=name) as stored:
            out[name] = stored.load()
    return out


def _write_groups(path, value):
    xr.Dataset(attrs={"names": json.dumps(list(value))}).to_netcdf(path, mode="w")
    for name, array in value.items():
        array.to_netcdf(path, mode="a", group=name)


PICKLE = Codec("value.pkl", _read_pickle, _write_pickle)
"""Any picklable value."""
NETCDF = Codec("value.nc", _read_netcdf, _write_netcdf)
"""A DataArray (name and attrs kept) or a Dataset."""
NETCDF_GROUPS = Codec("value.nc", _read_groups, _write_groups)
"""A dict of named DataArrays."""


class Store:
    """The entries of one kind of result under ``<root>/<kind>/``; see the module text.

    `codec` says how a payload is kept; `prefix` starts the fingerprint (``obs-1a2b...``); `label` names the store in
    the log lines (hit, miss, and on a miss the parts that differ from the nearest entry); `describe(value)`, if given,
    adds to a hit line; `miss_note` ends the line of a miss with nothing cached. With `memory=False` a root's entries are
    read from disk each time (a fresh copy per get) instead of being kept in this process.
    """

    def __init__(self, kind: str, root=None, *, codec: Codec = PICKLE, prefix: str = "", label: str | None = None,
                 describe: Callable[[Any], str] | None = None, miss_note: str = "", memory: bool = True):
        """A store of `kind` in `root` (None keeps it in memory for this process)."""
        self.kind, self.codec, self.prefix, self.describe, self.miss_note = kind, codec, prefix, describe, miss_note
        self.memory = memory or root is None
        self.label = label or kind
        self.root = None if root is None else Path(root)
        self.announced: set[str] = set()
        self._memory: dict[str, tuple[dict, Any]] = {}

    @property
    def folder(self) -> Path | None:
        """The folder of this kind's entries, or None for an in-memory store."""
        return None if self.root is None else self.root / self.kind

    @staticmethod
    def hash_parts(parts: Mapping) -> dict:
        """The hash of each part."""
        return {name: part_hash(value) for name, value in parts.items()}

    def fingerprint(self, parts: Mapping) -> str:
        """The fingerprint of a dict of parts: the prefix and 12 hex characters of the hash of their hashes."""
        return self._fingerprint(self.hash_parts(parts))

    def entry(self, parts: Mapping) -> Path | None:
        """The entry folder of these parts (it may not exist), or None for an in-memory store."""
        return None if self.folder is None else self.folder / self.fingerprint(parts)

    def get(self, parts: Mapping, *, quiet: bool = False) -> Any | None:
        """The cached value of these parts, or None. The first lookup of a fingerprint logs a hit or a miss.

        A miss names the parts that differ from the nearest cached entry. quiet=True logs nothing.
        """
        hashes = self.hash_parts(parts)
        fp = self._fingerprint(hashes)
        value = self._read(fp)
        if not quiet and fp not in self.announced:
            self.announced.add(fp)
            self._announce(fp, hashes, value)
        return value

    def put(self, parts: Mapping, value: Any) -> None:
        """Keep `value` under these parts (memory, and disk when there is a root); a disk failure is only logged."""
        hashes = self.hash_parts(parts)
        fp = self._fingerprint(hashes)
        if self.memory:
            self._memory[fp] = (hashes, value)
        if self.folder is None:
            return
        try:
            folder = self.folder / fp
            atomic_write(folder / self.codec.filename, lambda temp: self.codec.write(temp, value))
            atomic_write(folder / "parts.json",
                         lambda temp: temp.write_text(json.dumps({"fingerprint": fp, "parts": hashes}, indent=1)))
        except Exception as exc:  # noqa: BLE001 - the cache must never break a run
            log.debug("%s cache write failed (%s)", self.label, exc)

    def get_or_compute(self, parts: Mapping, compute: Callable[[], Any]) -> Any:
        """The cached value of these parts; else compute() it, keep it and return it."""
        value = self.get(parts)
        if value is None:
            value = compute()
            self.put(parts, value)
        return value

    def _fingerprint(self, hashes: dict) -> str:
        digest = part_hash(hashes)
        return f"{self.prefix}-{digest}" if self.prefix else digest

    def _read(self, fp: str) -> Any | None:
        """The value of a fingerprint from memory or disk (a damaged entry is a miss), or None."""
        if fp in self._memory:
            return self._memory[fp][1]
        if self.folder is None or not (self.folder / fp / self.codec.filename).exists():
            return None
        try:
            value = self.codec.read(self.folder / fp / self.codec.filename)
        except Exception as exc:  # noqa: BLE001 - a damaged entry is recomputed
            log.debug("%s cache entry %s unreadable (%s)", self.label, fp, exc)
            return None
        try:
            parts = json.loads((self.folder / fp / "parts.json").read_text())["parts"]
        except (OSError, ValueError, KeyError):
            parts = {}
        if self.memory:
            self._memory[fp] = (parts, value)
        return value

    def nearest(self, hashes: dict, fp: str | None = None) -> tuple[str, list[str]] | None:
        """The other entry sharing most parts with these hashes and the names of the parts that differ, or None."""
        others = {name: stored for name, (stored, _) in self._memory.items()}
        if self.folder is not None and self.folder.is_dir():
            for folder in self.folder.iterdir():
                try:
                    others.setdefault(folder.name, json.loads((folder / "parts.json").read_text())["parts"])
                except (OSError, ValueError, KeyError):
                    continue
        names = list(hashes)
        best = None
        for name, other in others.items():
            if name == fp or not other:
                continue
            differ = [n for n in dict.fromkeys(names + list(other)) if other.get(n) != hashes.get(n)]
            if best is None or len(differ) < len(best[1]):
                best = (name, differ)
        return best

    def _announce(self, fp: str, hashes: dict, value) -> None:
        """Log a hit, or a miss with the parts that differ from the nearest entry."""
        where = "in memory" if self.folder is None else str(self.folder)
        if value is not None:
            detail = f"{self.describe(value)}, " if self.describe else ""
            log.info("%s: cache hit %s (%s%s)", self.label, fp, detail, where)
            return
        near = self.nearest(hashes, fp)
        if near is None:
            log.info("%s: cache miss %s (no earlier entry in %s)%s", self.label, fp, where, self.miss_note)
        else:
            log.info("%s: cache miss %s; nearest cached entry %s differs in: %s", self.label, fp, near[0],
                     ", ".join(near[1]))
