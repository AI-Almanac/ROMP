"""Count how often each stored chunk of a NetCDF file is decoded while a block of code runs.

``count_decodes()`` wraps the read of xarray's netCDF4 backend: every read of a chunked variable is turned into the
stored chunks it touches (a chunk the read touches is decompressed, whole), and the touches are counted per chunk.
HDF5's own chunk cache is not counted on, so the count is what a read costs when the cache misses (the chunks of the
ai-almanac files are larger than its default size); a list of rows is read one run of consecutive rows at a time, each
run touching its chunks again. Contiguous variables (coordinates) have no chunks and are not
counted. Reads made in other processes are not seen: run the code with one reader.

::

    with count_decodes() as decodes:
        load_something()
    assert decodes.worst() <= 1          # no chunk decoded twice
"""
from __future__ import annotations

import itertools
from collections import Counter
from contextlib import contextmanager

import numpy as np


class DecodeCount:
    """The chunk decodes seen by `count_decodes`, per ``(file, variable, chunk coordinates)``."""

    def __init__(self):
        """Start with no reads."""
        self.per_chunk: Counter = Counter()
        self.reads = 0

    @property
    def chunks(self) -> int:
        """Return how many distinct chunks were decoded."""
        return len(self.per_chunk)

    @property
    def total(self) -> int:
        """Return how many chunk decodes there were, repeats included."""
        return sum(self.per_chunk.values())

    @property
    def ratio(self) -> float:
        """Return the decodes per distinct chunk (1.0 when every chunk was decoded once, 0.0 for no reads)."""
        return self.total / self.chunks if self.chunks else 0.0

    def worst(self) -> int:
        """Return the most times one chunk was decoded (0 when nothing was read)."""
        return max(self.per_chunk.values(), default=0)

    def only(self, *names: str) -> "DecodeCount":
        """Return the counts of some variables alone (a chunked coordinate is decoded a few times, and is tiny).

        Parameters
        ----------
        *names : str
            The variable names to keep.

        Returns
        -------
        DecodeCount
            The counts of those variables.
        """
        kept = DecodeCount()
        kept.reads = self.reads
        kept.per_chunk = Counter({key: n for key, n in self.per_chunk.items() if key[1] in names})
        return kept

    def add(self, source: str, name: str, key, shape, chunks) -> None:
        """Count the chunks one read of ``variable[key]`` touches.

        Parameters
        ----------
        source : str
            The file the variable is in.
        name : str
            The variable.
        key : tuple
            The outer index of the read: one slice, integer or integer array per dim.
        shape : tuple of int
            The shape of the variable.
        chunks : sequence of int
            The stored chunk shape of the variable.
        """
        self.reads += 1
        axes = []
        for entry, size, step in zip(key, shape, chunks):
            if isinstance(entry, slice):
                axes.append(sorted(set((np.arange(size)[entry] // step).tolist())))
                continue
            rows = np.atleast_1d(entry)
            # HDF5 reads a list of rows one run of consecutive rows at a time: a chunk holding two runs decodes twice
            runs = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
            axes.append([int(c) for run in runs for c in sorted(set((run // step).tolist()))])
        self.per_chunk.update((source, name, coord) for coord in itertools.product(*axes))


@contextmanager
def count_decodes():
    """Count the stored chunks decoded by xarray's netCDF4 backend inside the ``with`` block.

    Yields
    ------
    DecodeCount
        The counts; read them after the block (they fill as the block runs).
    """
    from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

    found = DecodeCount()
    original = NetCDF4ArrayWrapper._getitem

    def counted(self, key):
        with self.datastore.lock:
            variable = self.get_array(needs_lock=False)
            chunks = variable.chunking()
            source = variable.group().filepath()
        if chunks != "contiguous":
            found.add(source, self.variable_name, key, self.shape, chunks)
        return original(self, key)

    NetCDF4ArrayWrapper._getitem = counted
    try:
        yield found
    finally:
        NetCDF4ArrayWrapper._getitem = original
