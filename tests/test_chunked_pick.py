"""Inits picked from a chunked NetCDF forecast are read one stored chunk at a time, with the values of the plain read."""
import numpy as np
import pytest
import xarray as xr

from hcwf.spatial import select

INITS = [1, 2, 7, 8, 9, 12, 13, 20, 22]


@pytest.fixture
def source(tmp_path):
    rng = np.random.default_rng(0)
    values = rng.random((23, 3, 7, 8, 4)).astype("f4")
    values[values < 0.1] = np.nan
    ds = xr.Dataset({"tp": (("init_time", "member", "lat", "lon", "step"), values)},
                    coords={"init_time": np.arange(23).astype("datetime64[D]"), "lat": np.arange(7.), "lon": np.arange(8.)})
    ds.to_netcdf(tmp_path / "a.nc", encoding={"tp": {"chunksizes": (5, 2, 7, 4, 4), "zlib": True, "_FillValue": np.float32(-1)}})
    with xr.open_dataset(tmp_path / "a.nc") as opened:
        yield opened


def picked(source, **more):
    chosen = source.sel(init_time=source.init_time.values[INITS], **more)["tp"]
    return chosen.isel(step=slice(1, 3), lat=slice(1, 5))


def test_the_picked_inits_are_read_by_chunk_ranges(source, monkeypatch):
    expected = picked(source).load().values
    from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

    reads, original = [], NetCDF4ArrayWrapper._getitem

    def spy(self, key):
        reads.append(key[0])
        return original(self, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", spy)
    assert select._picked_rows(picked(source), 0) is not None   # the fast path is available (xarray internals unchanged)
    out = select.load_blocks(picked(source), "init_time")
    np.testing.assert_array_equal(out.values, expected)
    assert all(isinstance(r, slice) and r.stop - r.start <= 5 for r in reads)   # contiguous ranges inside one chunk
    assert len(reads) == len({i // 5 for i in INITS})


@pytest.mark.parametrize("block_bytes", [select._BLOCK_BYTES, 3000])
def test_values_equal_the_plain_read_in_any_block_size(source, monkeypatch, block_bytes):
    monkeypatch.setattr(select, "_BLOCK_BYTES", block_bytes)
    np.testing.assert_array_equal(select.load_blocks(picked(source), "init_time").values, picked(source).load().values)
    members = picked(source, member=[0, 2])
    np.testing.assert_array_equal(select.load_blocks(members, "init_time").values, picked(source, member=[0, 2]).load().values)


def test_a_keep_mask_reads_the_same_picked_rows(source):
    keep = np.zeros((4, 8), bool)
    keep[1, 5] = keep[3, 0] = True
    expected = np.where(keep[None, None, :, :, None], picked(source).load().values, np.nan)
    got = select.load_blocks(picked(source), "init_time", keep=keep).values
    np.testing.assert_array_equal(np.nan_to_num(got, nan=-9), np.nan_to_num(expected, nan=-9))


def test_other_selections_use_the_ordinary_read(source):
    contiguous = source["tp"].isel(init_time=slice(2, 9))
    assert select._picked_rows(contiguous, 0) is None
    reversed_rows = source["tp"].isel(init_time=[5, 3, 1])
    assert select._picked_rows(reversed_rows, 0) is None
    np.testing.assert_array_equal(select.load_blocks(reversed_rows, "init_time").values, reversed_rows.load().values)
