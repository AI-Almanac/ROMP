"""Keep results under a fingerprint of everything they depend on: the one cache of hcwf.

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
never stops a run. `Store.entries` lists a kind's entries (size, age) and `Store.prune` deletes them (``hcwf-run cache``).

The other half of this module is the in-memory side, the state of one run: every table a module keeps for the life of a
run is a `Memo` made by ``memo(name, ...)`` and registered here, so that `reset_run_state` clears them all (`run_scope`
does so at the start and the end of every run entry), the workers' snapshot takes the shared ones (`shared_state`,
`install_shared_state`) and nothing depends on which tables a caller remembered to clear.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import pickle
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

_META = frozenset(("zarr.json", ".zmetadata", ".zgroup", ".zattrs", ".zarray"))
"""Names of the metadata files of a zarr store, whose size and mtime are part of a directory's identity."""


class Memo:
    """A registered in-memory table of one run: a small dict that keeps at most ``maxsize`` items (least recently used out).

    Make one with `memo`, never directly. It holds what a module learns during a run (a loaded threshold set, a built mask,
    the open zarr stores), keyed by what the value depends on. Every operation takes a lock, so the reader threads of a run
    can share one. ``share`` marks a table whose items the workers need (it is part of `shared_state`); ``persistent`` one that
    holds only values fixed by the files of the installation (Natural Earth shapes), which a run reset keeps.

    Parameters
    ----------
    name : str
        The module path under hcwf and what the table holds, e.g. ``"io.thresholds.loaded"``.
    maxsize : int or None
        The most items kept (least recently used out); None for no bound.
    share : bool
        Whether the workers get the items (see `shared_state`).
    persistent : bool
        Whether a run reset keeps the items.
    on_clear : callable or None
        ``on_clear(item)`` is called for each item when the table is cleared.
    """

    def __init__(self, name: str, maxsize: int | None, share: bool, persistent: bool, on_clear):
        self.name, self.maxsize, self.share, self.persistent, self.on_clear = name, maxsize, share, persistent, on_clear
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        """Return the name and the number of items.

        Returns
        -------
        str
            For example ``Memo('io.thresholds.loaded', 3 items)``.
        """
        return f"Memo({self.name!r}, {len(self._items)} items)"

    def __len__(self) -> int:
        """Return the number of items.

        Returns
        -------
        int
            The count.
        """
        return len(self._items)

    def __contains__(self, key) -> bool:
        """Tell whether there is an item for `key` (it does not become the most recently used).

        Parameters
        ----------
        key : hashable
            The key to look for.

        Returns
        -------
        bool
            True when the key is present.
        """
        return key in self._items

    def __iter__(self) -> Iterator:
        """Iterate over a snapshot of the keys.

        Returns
        -------
        collections.abc.Iterator
            The keys, least recently used first.
        """
        with self._lock:
            return iter(list(self._items))

    def get(self, key, default=None):
        """Return the item of ``key`` (it becomes the most recently used), else ``default``.

        Parameters
        ----------
        key : hashable
            The key to look up.
        default : object, optional
            Returned when there is no item for ``key``.

        Returns
        -------
        object
            The item, or ``default``.
        """
        with self._lock:
            if key not in self._items:
                return default
            self._items.move_to_end(key)
            return self._items[key]

    def __getitem__(self, key):
        """Return the item of `key` (it becomes the most recently used).

        Parameters
        ----------
        key : hashable
            The key to look up.

        Returns
        -------
        object
            The item.

        Raises
        ------
        KeyError
            There is no item for `key`.
        """
        with self._lock:
            if key not in self._items:
                raise KeyError(key)
            return self.get(key)

    def __setitem__(self, key, value) -> None:
        """Keep `value` under `key` as the most recently used item, dropping the oldest ones beyond `maxsize`.

        Parameters
        ----------
        key : hashable
            The key.
        value : object
            The item to keep.
        """
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while self.maxsize is not None and len(self._items) > self.maxsize:
                self._items.popitem(last=False)

    def setdefault(self, key, value):
        """Return the item of ``key``, set to ``value`` first when there is none.

        Parameters
        ----------
        key : hashable
            The key.
        value : object
            The item to keep when there is none.

        Returns
        -------
        object
            The item now held under ``key``.
        """
        with self._lock:
            if key not in self._items:
                self[key] = value
            return self.get(key)

    def get_or_make(self, key, make: Callable[[], Any]):
        """Return the item of ``key``; else make it, keep it and return it (one maker at a time, so a value is built once).

        Parameters
        ----------
        key : hashable
            The key.
        make : callable
            Called with no arguments to build the item when there is none.

        Returns
        -------
        object
            The item.
        """
        with self._lock:
            if key in self._items:
                return self.get(key)
            value = make()
            self[key] = value
            return value

    def pop(self, key, *default):
        """Remove and return the item of ``key``, like ``dict.pop``.

        Parameters
        ----------
        key : hashable
            The key.
        *default : object
            Returned when there is no item for ``key``; without it a missing key raises.

        Returns
        -------
        object
            The removed item, or the default.
        """
        with self._lock:
            return self._items.pop(key, *default)

    def items(self) -> list:
        """Return the ``(key, item)`` pairs, least recently used first.

        Returns
        -------
        list of tuple
            A snapshot of the pairs.
        """
        with self._lock:
            return list(self._items.items())

    def values(self) -> list:
        """Return the items, least recently used first.

        Returns
        -------
        list
            A snapshot of the items.
        """
        with self._lock:
            return list(self._items.values())

    def update(self, items: Mapping) -> None:
        """Keep every item of a mapping (as ``__setitem__`` does).

        Parameters
        ----------
        items : collections.abc.Mapping
            The keys and items to add.
        """
        with self._lock:
            for key, value in dict(items).items():
                self[key] = value

    def clear(self) -> None:
        """Drop every item, calling ``on_clear`` with each (e.g. to close what it holds); a failure there is only logged."""
        with self._lock:
            held = list(self._items.values())
            self._items.clear()
        if self.on_clear is not None:
            for value in held:
                try:
                    self.on_clear(value)
                except Exception as exc:  # noqa: BLE001 - a failing close never stops a run
                    log.debug("%s: could not release an item (%s)", self.name, exc)


_MEMOS: dict[str, Memo] = {}
"""Every registered memo, by name."""
_END_HOOKS: list[Callable[[], None]] = []
"""What a run does when it ends, before the memos are cleared (flush a cache, stop the worker pool)."""
_SCOPE = {"depth": 0}
_SCOPE_LOCK = threading.Lock()


def memo(name: str, maxsize: int | None = None, *, share: bool = False, persistent: bool = False,
         on_clear: Callable[[Any], None] | None = None) -> Memo:
    """Register and return the in-memory table ``name``.

    The name is the module path under hcwf and what the table holds, e.g. ``"io.thresholds.loaded"`` in
    hcwf/io/thresholds.py (a shared table's module is imported by name in the workers). Making a name again replaces
    the earlier table.

    Parameters
    ----------
    name : str
        The module path under hcwf and what the table holds.
    maxsize : int, optional
        The most items kept; None for unbounded, for tables that are the size of one run.
    share : bool, default False
        Send the table to the workers (see `shared_state`).
    persistent : bool, default False
        Keep the table across runs (only for values fixed by files of the installation).
    on_clear : callable, optional
        ``on_clear(item)`` runs for each item when the table is cleared.

    Returns
    -------
    Memo
        The registered table.
    """
    table = Memo(name, maxsize, share, persistent, on_clear)
    _MEMOS[name] = table
    return table


def memos() -> dict[str, Memo]:
    """Return the registered memos.

    Returns
    -------
    dict of str to Memo
        A copy of the registry, by name.
    """
    return dict(_MEMOS)


def reset_run_state(*, everything: bool = False) -> None:
    """Clear every registered memo of the run.

    Parameters
    ----------
    everything : bool, default False
        Also clear the persistent memos.
    """
    for table in list(_MEMOS.values()):
        if everything or not table.persistent:
            table.clear()


def on_run_end(hook: Callable[[], None]) -> Callable[[], None]:
    """Register a hook to run when a run ends, before its memos are cleared (use it as a decorator, or call it).

    Parameters
    ----------
    hook : callable
        Called with no arguments at the end of the outermost `run_scope`.

    Returns
    -------
    callable
        The hook, unchanged.
    """
    if hook not in _END_HOOKS:
        _END_HOOKS.append(hook)
    return hook


@contextlib.contextmanager
def run_scope() -> Iterator[None]:
    """Scope one run's state: cleared on entering, the end hooks run and everything cleared again on leaving.

    Every run entry (`hcwf.run`, the command line, export, blend training and forecasting) is made of one. Scopes nest:
    only the outermost resets, so a step called inside a run (the onset export of a benchmark) keeps the run's state.

    Yields
    ------
    None
        Nothing; the block runs inside the scope.
    """
    with _SCOPE_LOCK:
        outermost = _SCOPE["depth"] == 0
        _SCOPE["depth"] += 1
    if outermost:
        reset_run_state()
    try:
        yield
    finally:
        with _SCOPE_LOCK:
            _SCOPE["depth"] -= 1
        if outermost:
            try:
                for hook in list(_END_HOOKS):
                    try:
                        hook()
                    except Exception as exc:  # noqa: BLE001 - the end of a run never raises over its result
                        log.debug("run end hook %s failed (%s)", getattr(hook, "__name__", hook), exc)
            finally:
                reset_run_state()


def shared_state() -> dict:
    """Return the items of every shared memo: what the workers start from (see `hcwf.parallel.snapshot`).

    Returns
    -------
    dict of str to dict
        ``{name: {key: item}}`` for each shared memo that holds items.
    """
    return {name: dict(table.items()) for name, table in _MEMOS.items() if table.share and len(table)}


def install_shared_state(state: Mapping) -> None:
    """Put a `shared_state` into this (worker) process's memos.

    A worker may not have imported the module that makes a memo yet, so a name is a module path under hcwf and the table
    (``"io.thresholds.loaded"`` is made by `hcwf.io.thresholds`): the module is imported first.

    Parameters
    ----------
    state : collections.abc.Mapping
        A `shared_state` made in the run's process.
    """
    import importlib

    for name, items in state.items():
        if name not in _MEMOS and "." in name:
            try:
                importlib.import_module("hcwf." + name.rsplit(".", 1)[0])
            except ImportError:
                pass
        if name in _MEMOS:
            _MEMOS[name].update(items)


def file_identity(path) -> list:
    """Return what identifies a file or directory (zarr stores included).

    For a directory the size is the number of its entries and the mtime the newest of the directory, its entries
    and the metadata files one level down, so a rewritten zarr array or added file changes it without walking every chunk.

    Parameters
    ----------
    path : str or pathlib.Path
        The file or directory.

    Returns
    -------
    list
        ``[path, size, mtime_ns]``, or ``[path, None, None]`` when it is missing.
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
    """Add a value to a hash, type-tagged, by value (see `part_hash`).

    Parameters
    ----------
    h : hashlib._Hash
        The hash to update.
    value : object
        The value; arrays by contents, containers recursively, anything else by ``repr``.
    """
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
    """Return a 12-hex-character hash of the values: arrays by contents (strings by value), containers recursively.

    Parameters
    ----------
    *values : object
        The values to hash together, in order.

    Returns
    -------
    str
        The first 12 hex characters of the SHA-256.
    """
    h = hashlib.sha256()
    for value in values:
        _feed(h, value)
    return h.hexdigest()[:12]


def atomic_write(path, write: Callable[[Path], None]) -> None:
    """Make ``path`` (a file, or a folder ``write`` creates) by writing a temporary path and renaming it over ``path``.

    Nothing is left behind on failure.

    Parameters
    ----------
    path : str or pathlib.Path
        The file or folder to make; its parent folder is created.
    write : callable
        ``write(temporary_path)`` creates the file or folder at the temporary path it is given.

    Raises
    ------
    BaseException
        Whatever ``write`` or the rename raised (including `KeyboardInterrupt`), after the temporary path is removed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp{path.suffix}")
    try:
        write(temp)
        os.replace(temp, path)
    except BaseException:
        if temp.is_dir():
            shutil.rmtree(temp, ignore_errors=True)
        else:
            temp.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class Codec:
    """How a store keeps a payload: its file name in the entry folder and the functions that read and write it."""

    filename: str
    """The payload's file name in the entry folder."""
    read: Callable[[Path], Any]
    """``read(path)`` returns the payload."""
    write: Callable[[Path, Any], None]
    """``write(path, value)`` stores the payload."""


def _read_pickle(path):
    """Read a pickled value.

    Parameters
    ----------
    path : pathlib.Path
        The pickle file.

    Returns
    -------
    object
        The unpickled value.
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _write_pickle(path, value):
    """Write a value as a pickle.

    Parameters
    ----------
    path : pathlib.Path
        The file to write.
    value : object
        Any picklable value.
    """
    with open(path, "wb") as fh:
        pickle.dump(value, fh)


def _read_netcdf(path):
    """Read a NetCDF payload written by `_write_netcdf`, loaded into memory.

    Parameters
    ----------
    path : pathlib.Path
        The NetCDF file.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The stored object; a DataArray when it was stored as one (name and attrs restored).
    """
    with xr.open_dataset(path) as ds:
        loaded = ds.load()
    if "__value__" not in loaded:
        return loaded
    array = loaded["__value__"]
    array.name = array.attrs.pop("__name__", "") or None
    return array


def _write_netcdf(path, value):
    """Write a DataArray (its name and attrs kept) or a Dataset as NetCDF.

    Parameters
    ----------
    path : pathlib.Path
        The file to write.
    value : xarray.Dataset or xarray.DataArray
        The object to store.
    """
    if isinstance(value, xr.DataArray):
        ds = value.to_dataset(name="__value__")
        ds["__value__"].attrs["__name__"] = "" if value.name is None else str(value.name)
        value = ds
    value.to_netcdf(path)


def _read_groups(path):
    """Read a dict of named DataArrays written by `_write_groups`.

    Parameters
    ----------
    path : pathlib.Path
        The NetCDF file.

    Returns
    -------
    dict of str to xarray.DataArray
        The arrays, loaded into memory, in their stored order.
    """
    with xr.open_dataset(path) as root:
        names = json.loads(root.attrs["names"])
    out = {}
    for name in names:
        with xr.open_dataarray(path, group=name) as stored:
            out[name] = stored.load()
    return out


def _write_groups(path, value):
    """Write a dict of named DataArrays as one NetCDF group each.

    Parameters
    ----------
    path : pathlib.Path
        The file to write.
    value : dict of str to xarray.DataArray
        The arrays to store.
    """
    xr.Dataset(attrs={"names": json.dumps(list(value))}).to_netcdf(path, mode="w")
    for name, array in value.items():
        array.to_netcdf(path, mode="a", group=name)


PICKLE = Codec("value.pkl", _read_pickle, _write_pickle)
"""Any picklable value."""
NETCDF = Codec("value.nc", _read_netcdf, _write_netcdf)
"""A DataArray (name and attrs kept) or a Dataset."""
NETCDF_GROUPS = Codec("value.nc", _read_groups, _write_groups)
"""A dict of named DataArrays."""


@dataclass(frozen=True)
class Entry:
    """One cache entry on disk: its kind, fingerprint, folder, size in bytes and the time it was last written."""

    kind: str
    """The kind of result (the folder under the cache root)."""
    name: str
    """The fingerprint, which is also the entry folder's name."""
    path: Path
    """The entry folder."""
    size: int
    """The bytes in all its files."""
    mtime: float
    """When it was last written, in epoch seconds."""


def _tree_size(path: Path) -> tuple[int, float]:
    """Measure a folder.

    Parameters
    ----------
    path : pathlib.Path
        The folder.

    Returns
    -------
    tuple of (int, float)
        The bytes in all the files under it, and the newest modification time of it and its files.
    """
    size, newest = 0, path.stat().st_mtime
    for item in path.rglob("*"):
        try:
            stat = item.stat()
        except OSError:
            continue
        newest = max(newest, stat.st_mtime)
        if item.is_file():
            size += stat.st_size
    return size, newest


def global_root() -> Path:
    """Return the folder of the cache shared by every run: ``$HCWF_CACHE_DIR``, else ``~/.cache/hcwf``.

    Returns
    -------
    pathlib.Path
        The shared cache root.
    """
    return Path(os.environ.get("HCWF_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "hcwf"))


_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}


def parse_duration(text: str) -> float:
    """Convert a duration written as a number and a unit, such as ``90m``, ``12h``, ``30d`` or ``2w``, into seconds.

    Parameters
    ----------
    text : str
        A number and a unit (``s``, ``m``, ``h``, ``d``, ``w``); a bare number is days.

    Returns
    -------
    float
        The duration in seconds.

    Raises
    ------
    ValueError
        The text is not a duration, or is negative.
    """
    text = str(text).strip().lower()
    unit = text[-1:] if text[-1:] in _UNITS else "d"
    number = text[:-1] if text[-1:] in _UNITS else text
    try:
        value = float(number)
    except ValueError:
        raise ValueError(f"{text!r} is not a duration: write a number and a unit, e.g. 12h, 30d, 2w") from None
    if value < 0:
        raise ValueError(f"{text!r} is negative")
    return value * _UNITS[unit]


def store_kinds(root, kinds=None) -> list[Store]:
    """Return a Store for each kind folder of a cache root, to list or prune its entries.

    Parameters
    ----------
    root : str or pathlib.Path
        The cache root.
    kinds : iterable of str, optional
        Only these kinds; None or empty means every kind there is.

    Returns
    -------
    list of Store
        One store per kind folder.
    """
    return [Store(kind, root) for kind in Store.kinds(root) if not kinds or kind in kinds]


SCHEMA = {"observed_onsets": 1, "thresholds": 1, "harmonic_seasons": 1, "model_thresholds": 1, "detect": 1, "alignment": 1, "oracle": 1}
"""The code version of each kind of result. Bump a kind's number in the same commit as any change to how that result is
computed (onset engine and backends, threshold computation, harmonic seasons, detection): `Store` adds it to the parts of
every entry of the kind, so older entries, in any shared cache folder, simply miss once. Kinds not listed carry none."""


class Store:
    """The entries of one kind of result under ``<root>/<kind>/``; see the module text.

    Parameters
    ----------
    kind : str
        The kind of result; the folder name under ``root``.
    root : str or pathlib.Path, optional
        The cache root; None keeps the store in memory for this process.
    codec : Codec, default `PICKLE`
        How a payload is kept.
    prefix : str, default ""
        Starts the fingerprint (``obs-1a2b...``).
    label : str, optional
        Names the store in the log lines (hit, miss, and on a miss the parts that differ from the nearest entry); default
        ``kind``.
    describe : callable, optional
        ``describe(value)`` returns text that is added to the line of a hit.
    miss_note : str, default ""
        Ends the line of a miss that has no earlier entry to compare with.
    memory : bool, default True
        With False a root's entries are read from disk each time (a fresh copy per get) instead of being kept in this
        process.
    """

    def __init__(self, kind: str, root=None, *, codec: Codec = PICKLE, prefix: str = "", label: str | None = None,
                 describe: Callable[[Any], str] | None = None, miss_note: str = "", memory: bool = True):
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
        """Return the hash of each part.

        Parameters
        ----------
        parts : collections.abc.Mapping
            The named inputs of an entry.

        Returns
        -------
        dict of str to str
            `part_hash` of each value, by part name.
        """
        return {name: part_hash(value) for name, value in parts.items()}

    def _hashes(self, parts: Mapping) -> dict:
        """Return the hash of each part, and of the code version of this store's kind (see `SCHEMA`) when it has one.

        Parameters
        ----------
        parts : collections.abc.Mapping
            The named inputs of an entry.

        Returns
        -------
        dict of str to str
            The hashes by part name, plus ``code_version`` for a versioned kind.
        """
        hashes = self.hash_parts(parts)
        if self.kind in SCHEMA:
            hashes["code_version"] = part_hash(SCHEMA[self.kind])
        return hashes

    def fingerprint(self, parts: Mapping) -> str:
        """Return the fingerprint of a dict of parts: the prefix and 12 hex characters of the hash of their hashes.

        Parameters
        ----------
        parts : collections.abc.Mapping
            The named inputs of an entry.

        Returns
        -------
        str
            The fingerprint.
        """
        return self._fingerprint(self._hashes(parts))

    def entry(self, parts: Mapping) -> Path | None:
        """Return the entry folder of these parts (it may not exist).

        Parameters
        ----------
        parts : collections.abc.Mapping
            The named inputs of an entry.

        Returns
        -------
        pathlib.Path or None
            The folder, or None for an in-memory store.
        """
        return None if self.folder is None else self.folder / self.fingerprint(parts)

    def get(self, parts: Mapping, *, quiet: bool = False) -> Any | None:
        """Return the cached value of these parts, or None. The first lookup of a fingerprint logs a hit or a miss.

        A miss names the parts that differ from the nearest cached entry.

        Parameters
        ----------
        parts : collections.abc.Mapping
            The named inputs of an entry.
        quiet : bool, default False
            Log nothing.

        Returns
        -------
        object or None
            The cached value, or None on a miss.
        """
        hashes = self._hashes(parts)
        fp = self._fingerprint(hashes)
        value = self._read(fp)
        if not quiet and fp not in self.announced:
            self.announced.add(fp)
            self._announce(fp, hashes, value)
        return value

    def put(self, parts: Mapping, value: Any) -> None:
        """Keep ``value`` under these parts (memory, and disk when there is a root); a disk failure is only logged.

        Parameters
        ----------
        parts : collections.abc.Mapping
            The named inputs of an entry.
        value : object
            The payload to keep.
        """
        hashes = self._hashes(parts)
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
        """Return the cached value of these parts; else compute it, keep it and return it.

        Parameters
        ----------
        parts : collections.abc.Mapping
            The named inputs of an entry.
        compute : callable
            Called with no arguments on a miss.

        Returns
        -------
        object
            The cached or computed value.
        """
        value = self.get(parts)
        if value is None:
            value = compute()
            self.put(parts, value)
        return value

    def _fingerprint(self, hashes: dict) -> str:
        """Return the fingerprint of the part hashes, with the store's prefix.

        Parameters
        ----------
        hashes : dict of str to str
            The hash of each part, from `_hashes`.

        Returns
        -------
        str
            The fingerprint.
        """
        digest = part_hash(hashes)
        return f"{self.prefix}-{digest}" if self.prefix else digest

    def _read(self, fp: str) -> Any | None:
        """Return the value of a fingerprint from memory or disk (a damaged entry is a miss).

        Parameters
        ----------
        fp : str
            The fingerprint.

        Returns
        -------
        object or None
            The value, or None when there is none or it cannot be read.
        """
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

    @staticmethod
    def kinds(root) -> list[str]:
        """Return the kinds with a folder in a cache root.

        Parameters
        ----------
        root : str or pathlib.Path
            The cache root.

        Returns
        -------
        list of str
            The kind names, sorted; empty when the root is not there.
        """
        root = Path(root)
        return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []

    def entries(self) -> list[Entry]:
        """Return the entries of this kind on disk, oldest first (an in-memory store has none).

        Returns
        -------
        list of Entry
            The entries, by last write time then name.
        """
        folder = self.folder
        if folder is None or not folder.is_dir():
            return []
        found = []
        for path in folder.iterdir():
            if path.is_dir():
                size, newest = _tree_size(path)
                found.append(Entry(self.kind, path.name, path, size, newest))
        return sorted(found, key=lambda e: (e.mtime, e.name))

    def prune(self, *, older_than: float | None = None, dry_run: bool = False, now: float | None = None) -> list[Entry]:
        """Delete this kind's entries last written more than ``older_than`` seconds ago.

        An entry that cannot be deleted is left and not returned.

        Parameters
        ----------
        older_than : float, optional
            The age in seconds; None deletes every entry.
        dry_run : bool, default False
            Touch nothing and return the entries that would be deleted.
        now : float, optional
            The current time in epoch seconds; default ``time.time()``.

        Returns
        -------
        list of Entry
            The entries deleted (or, with ``dry_run``, the ones that would be).
        """
        now = time.time() if now is None else now
        chosen = [e for e in self.entries() if older_than is None or now - e.mtime > older_than]
        if dry_run:
            return chosen
        gone = []
        for entry in chosen:
            try:
                shutil.rmtree(entry.path)
            except OSError as exc:
                log.debug("%s cache entry %s not deleted (%s)", self.label, entry.path, exc)
                continue
            self._memory.pop(entry.name, None)
            self.announced.discard(entry.name)
            gone.append(entry)
        return gone

    def nearest(self, hashes: dict, fp: str | None = None) -> tuple[str, list[str]] | None:
        """Return the other entry sharing most parts with these hashes and the names of the parts that differ.

        Parameters
        ----------
        hashes : dict of str to str
            The hash of each part, from `_hashes`.
        fp : str, optional
            The fingerprint of the entry itself, which is left out of the comparison.

        Returns
        -------
        tuple of (str, list of str) or None
            The nearest entry's fingerprint and the differing part names; None when there is no other entry.
        """
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
        """Log a hit, or a miss with the parts that differ from the nearest entry.

        Parameters
        ----------
        fp : str
            The fingerprint looked up.
        hashes : dict of str to str
            The hash of each part, from `_hashes`.
        value : object or None
            What the lookup found; None is a miss.
        """
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
