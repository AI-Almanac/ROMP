"""Blocks and tiles of a read are cut on the file's chunk grid, so every stored chunk is decoded once."""
import itertools
from collections import Counter

import numpy as np
import pytest
import xarray as xr

from hcwf.spatial import select

SHAPE = (23, 3, 20, 22, 4)   # init_time, member, lat, lon, step
DIMS = ("init_time", "member", "lat", "lon", "step")
CHUNKS = (5, 2, 7, 6, 4)


def _dataset():
    rng = np.random.default_rng(1)
    values = rng.random(SHAPE).astype("f4")
    values[values < 0.1] = np.nan
    return xr.Dataset({"tp": (DIMS, values)},
                      coords={"init_time": np.arange(SHAPE[0]).astype("datetime64[D]"), "lat": np.arange(SHAPE[2]) * 1.,
                              "lon": np.arange(SHAPE[3]) * 1.})


@pytest.fixture(params=[CHUNKS, (23, 3, 20, 22, 4), (1, 3, 5, 5, 4), (7, 1, 3, 22, 2)], ids=lambda c: "x".join(map(str, c)))
def chunked(request, tmp_path):
    _dataset().to_netcdf(tmp_path / "a.nc", encoding={"tp": {"chunksizes": request.param, "zlib": True,
                                                              "_FillValue": np.float32(-1)}})
    with xr.open_dataset(tmp_path / "a.nc") as ds:
        yield ds, request.param


def _runs_cover_one_chunk_each(da, axis, stored):
    runs = select._chunk_runs(da, axis, stored)
    assert runs[0][0] == 0 and runs[-1][1] == da.shape[axis]
    assert all(a[1] == b[0] for a, b in zip(runs, runs[1:]))
    return runs


def test_runs_follow_the_file_rows_through_a_box_a_flip_and_a_pick(chunked):
    ds, chunks = chunked
    da = ds["tp"]
    boxed = da.isel(lat=slice(3, 17), lon=slice(5, 21))
    flipped = da.isel(lat=slice(None, None, -1))
    picked = da.isel(init_time=[1, 2, 7, 8, 9, 12, 13, 20, 22])
    for sel, axis, rows in [(boxed, 2, np.arange(20)[3:17]), (boxed, 3, np.arange(22)[5:21]),
                            (flipped, 2, np.arange(20)[::-1]), (picked, 0, np.array([1, 2, 7, 8, 9, 12, 13, 20, 22])),
                            (da, 4, np.arange(4))]:
        runs = _runs_cover_one_chunk_each(sel, axis, chunks[axis])
        chunk_ids = [set(rows[a:b] // chunks[axis]) for a, b in runs]
        assert all(len(ids) == 1 for ids in chunk_ids)   # a run lies in one stored chunk
        assert len({next(iter(ids)) for ids in chunk_ids}) == len(runs)   # and a chunk is in one run


def test_runs_of_an_array_without_chunks_or_a_lazy_key_are_cut_by_position(chunked):
    ds, chunks = chunked
    assert select._chunk_runs(ds["tp"], 2, None) == [(0, SHAPE[2])]
    in_memory = ds["tp"].isel(lat=slice(3, 17)).load()
    assert select._chunk_runs(in_memory, 2, 7) == [(0, 7), (7, 14)]
    assert select._chunk_runs(ds["tp"].isel(lat=[4, 3, 9]), 2, 7) == [(0, 3)]   # not monotonic: by position


def _spy(monkeypatch):
    """Record every read of the file as the set of chunk coordinates (and the number of reads) it touches."""
    from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

    reads, original = [], NetCDF4ArrayWrapper._getitem

    def spy(self, key):
        reads.append(key)
        return original(self, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", spy)
    return reads


def _touches(reads, chunks):
    per_chunk = Counter()
    for key in reads:
        axes = []
        for entry, n, c in zip(key, SHAPE, chunks):
            rows = np.arange(n)[entry] if isinstance(entry, slice) else np.atleast_1d(entry)
            axes.append(sorted(set((rows // c).tolist())))
        per_chunk.update(itertools.product(*axes))
    return per_chunk


@pytest.mark.parametrize("readers", [None, 1])
def test_a_flipped_boxed_pick_reads_each_stored_chunk_once_with_the_plain_values(chunked, monkeypatch, readers):
    ds, chunks = chunked
    inits = [1, 2, 7, 8, 9, 12, 13, 20, 22]

    def selection():
        da = ds["tp"].sel(init_time=ds.init_time.values[inits])
        return da.isel(lat=slice(None, None, -1)).isel(lat=slice(2, 18), lon=slice(3, 20), step=slice(0, 3))

    expected = selection().load().values
    reads = _spy(monkeypatch)
    monkeypatch.setattr(select, "_BLOCK_BYTES", 2000)   # many blocks
    keep = np.ones((16, 17), bool)
    keep[::3, ::2] = False
    for kept in (None, keep):
        reads.clear()
        got = select.load_blocks(selection(), "init_time", keep=kept, readers=readers).values
        want = expected if kept is None else np.where(kept[None, None, :, :, None], expected, np.nan)
        np.testing.assert_array_equal(np.nan_to_num(got, nan=-9), np.nan_to_num(want, nan=-9))
        touches = _touches(reads, chunks)
        assert touches and max(touches.values()) == 1   # no chunk is read by two calls


def test_a_contiguous_file_and_zarr_read_with_the_plain_values(tmp_path):
    ds = _dataset()
    ds.to_netcdf(tmp_path / "plain.nc", encoding={"tp": {"contiguous": True, "_FillValue": np.float32(-1)}})
    ds.chunk({"init_time": 1, "lat": 7}).to_zarr(tmp_path / "s.zarr", mode="w")
    with xr.open_dataset(tmp_path / "plain.nc") as plain, xr.open_zarr(tmp_path / "s.zarr", chunks=None) as zarr:
        for src in (plain, zarr):
            def da():
                return src["tp"].isel(lat=slice(None, None, -1)).isel(init_time=slice(2, 20), lat=slice(1, 17))

            expected = da().load().values
            np.testing.assert_array_equal(select.load_blocks(da(), "init_time").values, expected)
            keep = np.zeros((16, SHAPE[3]), bool)
            keep[3:9, 4:15] = True
            got = select.load_blocks(da(), "init_time", keep=keep).values
            want = np.where(keep[None, None, :, :, None], expected, np.nan)
            np.testing.assert_array_equal(np.nan_to_num(got, nan=-9), np.nan_to_num(want, nan=-9))
