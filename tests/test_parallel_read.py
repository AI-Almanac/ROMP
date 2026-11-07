"""A big NetCDF forecast is read by reader processes, one stored chunk's rows of one tile at a time, skipping the tiles of
masked cells; the array is the one the plain read gives."""
import numpy as np
import pytest
import xarray as xr

from hcwf import parallel
from hcwf.config.model import AreaSpec, Selection
from hcwf.spatial import select

INITS = [1, 2, 7, 8, 9, 12, 13, 20, 22]


@pytest.fixture
def source(tmp_path):
    rng = np.random.default_rng(1)
    values = rng.random((23, 4, 12, 14, 3)).astype("f4")
    values[values < 0.1] = np.nan
    lat, lon = 3.0 + np.arange(12.0), 33.0 + np.arange(14.0)
    ds = xr.Dataset({"tp": (("init_time", "member", "lat", "lon", "step"), values)},
                    coords={"init_time": np.arange(23).astype("datetime64[D]"), "lat": lat, "lon": lon})
    ds.to_netcdf(tmp_path / "a.nc", encoding={"tp": {"chunksizes": (5, 2, 6, 7, 3), "zlib": True,
                                                      "_FillValue": np.float32(-1)}})
    with xr.open_dataset(tmp_path / "a.nc") as opened:
        yield opened


def picked(source):
    return source.sel(init_time=source.init_time.values[INITS])["tp"]


@pytest.fixture
def readers(monkeypatch):
    """Reader processes for even a small read (two of them), stopped afterwards."""
    monkeypatch.setattr(select, "_PARALLEL_READ_BYTES", 0)
    monkeypatch.setattr(select, "_READ_BYTES_EACH", 1)
    yield 2
    parallel.shutdown_readers()


def test_a_small_read_does_not_start_processes(source):
    assert select._read_workers(picked(source), 8) == 1
    assert select._read_workers(picked(source), None) == 1


def test_the_rows_are_picked_along_their_own_axis_when_another_axis_is_blocked(source, monkeypatch):
    """Blocks of a read cut along lat or lon (a block of inits would not fit): the inits are still read by chunk ranges."""
    from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

    expected = picked(source).load().values
    reads, original = [], NetCDF4ArrayWrapper._getitem

    def spy(self, key):
        if len(key) == 5:   # the forecast variable, not a coordinate
            reads.append(key[0])
        return original(self, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", spy)
    monkeypatch.setattr(select, "_BLOCK_BYTES", 2000)
    da = picked(source)
    out = select.load_blocks(da, "init_time")
    np.testing.assert_array_equal(out.values, expected)
    assert all(isinstance(r, slice) and r.stop - r.start <= 5 for r in reads)
    assert len(reads) <= len({i // 5 for i in INITS}) * 2   # one range per stored chunk and block, not one per init


@pytest.mark.parametrize("keep_cells", [False, True])
def test_reader_processes_give_the_plain_reads_array(source, readers, keep_cells):
    keep = None
    if keep_cells:
        keep = np.zeros((12, 14), bool)
        keep[1:5, 2:9] = keep[9, 12] = True
    plain = select.load_blocks(picked(source), "init_time", keep=keep).values
    got = select.load_blocks(picked(source), "init_time", keep=keep, readers=readers).values
    np.testing.assert_array_equal(np.nan_to_num(got, nan=-9), np.nan_to_num(plain, nan=-9))


def test_compact_cells_by_reader_processes_are_the_kept_cells_of_the_plain_read(source, readers):
    keep = np.zeros((12, 14), bool)
    keep[1:5, 2:9] = keep[9, 12] = True
    plain = select.load_blocks(picked(source), "init_time", keep=keep).values        # init, member, lat, lon, step
    cells = select.load_blocks(picked(source), "init_time", keep=keep, compact=True, readers=readers)
    rows, cols = np.nonzero(keep)
    assert cells.dims == ("init_time", "member", "step", "cell")
    np.testing.assert_array_equal(np.nan_to_num(cells.values, nan=-9),
                                  np.nan_to_num(np.moveaxis(plain[:, :, rows, cols, :], -1, 2), nan=-9))


def test_masked_cells_are_not_read_and_the_array_is_the_masked_one(source):
    """region_select with skip_masked equals reading everything and masking in place; compact is its kept cells."""
    area, chosen = AreaSpec("Ethiopia", None, None, None, False), Selection(shp_only=False)
    ds = source[["tp"]].isel(lat=slice(None, None, -1))
    ds = ds.assign_coords(lat=ds["lat"].values)   # ascending, as the loader's
    full = select.region_select(ds, area=area, select=chosen, read_by="init_time")["tp"]
    skipped = select.region_select(ds, area=area, select=chosen, read_by="init_time", skip_masked=True)["tp"]
    keep = select.implicit_keep(full, area, chosen)
    assert keep is not None and not keep.all()
    np.testing.assert_array_equal(np.nan_to_num(skipped.values, nan=-9), np.nan_to_num(full.values, nan=-9))
    compact = select.region_select(ds, area=area, select=chosen, read_by="init_time", skip_masked=True,
                                   compact=True)["tp"]
    assert compact.dims[-1] == "cell" and compact.sizes["cell"] == keep.sum()
    on_grid = np.full(full.shape, np.nan, dtype=full.dtype)
    lat_at = {v: i for i, v in enumerate(full["lat"].values)}
    lon_at = {v: i for i, v in enumerate(full["lon"].values)}
    for k, (la, lo) in enumerate(zip(compact["lat"].values, compact["lon"].values)):
        on_grid[:, :, lat_at[la], lon_at[lo], :] = compact.values[..., k]   # (init, member, step)
    np.testing.assert_array_equal(np.nan_to_num(on_grid, nan=-9), np.nan_to_num(full.values, nan=-9))


def test_a_small_box_of_big_chunks_is_read_by_processes_for_what_is_decoded(source, monkeypatch):
    """The box is a few values, but each stored chunk it touches is decoded whole: that size decides; the values are the plain read's."""
    box = source.isel(lat=slice(1, 3), lon=slice(2, 4)).sel(init_time=source.init_time.values[INITS])["tp"]
    assert select._decoded_bytes(box) > 10 * box.size * box.dtype.itemsize
    expected = select.load_blocks(box, "init_time").values   # the plain read: the box is below every threshold
    assert select._read_workers(box, 2) == 1
    monkeypatch.setattr(select, "_PARALLEL_READ_BYTES", select._decoded_bytes(box))
    monkeypatch.setattr(select, "_READ_BYTES_EACH", 1)
    assert box.size * box.dtype.itemsize < select._PARALLEL_READ_BYTES
    assert select._read_workers(box, 2) == 2
    try:
        got = select.load_blocks(box, "init_time", readers=2).values
    finally:
        parallel.shutdown_readers()
    np.testing.assert_array_equal(got, expected)
