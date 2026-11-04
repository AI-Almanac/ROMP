"""The detection sample is a set of whole stored chunks, each read once, on the file's own chunk grid."""
import itertools

import numpy as np
import pytest
import xarray as xr

from hcwf.cache import reset_run_state
from hcwf.config import detect
from hcwf.spatial import select

ROLES = {"init_time": "time", "step": "step", "member": "number", "lat": "lat", "lon": "lon"}
DIMS = ("time", "number", "step", "lat", "lon")
CHUNKS = (11, 3, 4, 33, 34)
SHAPE = (24, 4, 6, 129, 135)
BIG = (11, 3, 4, SHAPE[3], SHAPE[4])
SMALL = (5, 2, 3, 10, 9)


def _array():
    """Build a forecast with a member dim, the India files in miniature (several chunks on every dim)."""
    rng = np.random.default_rng(1)
    data = rng.gamma(0.5, 0.004, size=SHAPE).astype("float32")
    return xr.DataArray(
        data, dims=DIMS, name="tp", attrs={"units": "m"},
        coords={"time": np.datetime64("2020-05-01") + np.arange(SHAPE[0]).astype("timedelta64[D]"),
                "number": np.arange(SHAPE[1]), "step": np.arange(1, SHAPE[2] + 1) * np.timedelta64(1, "D"),
                "lat": np.linspace(30, 5, SHAPE[3]), "lon": np.linspace(60, 100, SHAPE[4])})


def _netcdf(tmp_path, chunks=CHUNKS, flip=False):
    """Write the array as NetCDF (chunked and compressed, or contiguous with ``chunks=None``) and open it lazily."""
    da = _array()
    encoding = {"tp": {"chunksizes": chunks, "zlib": True} if chunks else {"contiguous": True}}
    da.to_dataset().to_netcdf(tmp_path / "f.nc", encoding=encoding)
    ds = xr.open_dataset(tmp_path / "f.nc")
    return ds, ds["tp"], _array()


def _record(monkeypatch):
    """Record the boxes `select.read_box` is asked to read."""
    boxes = []
    real = select.read_box

    def spy(da, box):
        """Read the box and note it."""
        boxes.append(tuple(box))
        return real(da, box)

    monkeypatch.setattr(select, "read_box", spy)
    return boxes


def _expected(source, sample):
    """Return ``source`` cut to the positions that ``sample`` holds on every dim (by coordinate)."""
    return source.isel({d: source.get_index(d).get_indexer(sample[d].values) for d in sample.dims})


def _chunk_ids(boxes, chunks, flip_extent=None):
    """Return every stored chunk that the boxes read (a chunk read by two boxes is listed twice)."""
    seen = []
    for box in boxes:
        spans = []
        for axis, ((a, b), size) in enumerate(zip(box, chunks)):
            if flip_extent is not None and axis == 3:
                a, b = flip_extent - b, flip_extent - a
            spans.append(range(a // size, (b - 1) // size + 1))
        seen += list(itertools.product(*spans))
    return seen


def test_chunked_sample_reads_whole_chunks_once(tmp_path, monkeypatch):
    """Start, middle and end time chunks, the member chunk of member 0 and one lat/lon chunk: each chunk read once."""
    ds, da, source = _netcdf(tmp_path)
    boxes = _record(monkeypatch)
    out = detect.load_sample(da, ROLES)
    ids = _chunk_ids(boxes, CHUNKS)
    assert len(set(ids)) == len(ids)                                    # no chunk twice
    assert {i[1] for i in ids} == {0}                                   # the member chunk of member 0 only
    assert {i[0] for i in ids} == {0, (SHAPE[0] - 2) // 2 // 11, (SHAPE[0] - 1) // 11}          # start, middle and end time chunks
    assert {i[2] for i in ids} == {0, 1}                                # the lead in full
    assert len({i[3] for i in ids}) == 1 and len({i[4] for i in ids}) == 1   # one lat/lon block
    assert out.sizes["number"] == 1 and out.sizes["step"] == SHAPE[2]
    assert out.sizes["lat"] == CHUNKS[3] and out.sizes["lon"] == CHUNKS[4]
    # the cells are those of the chunk that holds the centre, the steps those of the start, middle and end blocks
    assert out.lat.values[0] == source.lat.values[(SHAPE[3] // 2 // CHUNKS[3]) * CHUNKS[3]]
    n = SHAPE[0]
    assert list(out.time.values) == list(source.time.values[[0, 1, (n - 2) // 2, (n - 2) // 2 + 1, n - 2, n - 1]])
    np.testing.assert_array_equal(out.values, _expected(source, out).values)
    assert out.attrs["units"] == "m"
    ds.close()


def test_flipped_latitude_sample_reads_each_chunk_once(tmp_path, monkeypatch):
    """A latitude stored north to south and flipped on opening is read on the file's chunk edges."""
    ds, da, source = _netcdf(tmp_path)
    flipped = da.isel(lat=slice(None, None, -1))
    boxes = _record(monkeypatch)
    out = detect.load_sample(flipped, ROLES)
    ids = _chunk_ids(boxes, CHUNKS, flip_extent=SHAPE[3])
    assert len(set(ids)) == len(ids)
    np.testing.assert_array_equal(out.values, _expected(source.isel(lat=slice(None, None, -1)), out).values)
    ds.close()


def test_small_chunks_are_joined_to_a_block(tmp_path, monkeypatch):
    """Chunks shorter than the least block are joined with their neighbours, still one read of each."""
    ds, da, source = _netcdf(tmp_path, chunks=SMALL)
    boxes = _record(monkeypatch)
    out = detect.load_sample(da, ROLES)
    ids = _chunk_ids(boxes, SMALL)
    assert len(set(ids)) == len(ids)
    assert out.sizes["lat"] >= detect._SAMPLE_MIN_CELLS and out.sizes["lon"] >= detect._SAMPLE_MIN_CELLS
    np.testing.assert_array_equal(out.values, _expected(source, out).values)
    ds.close()


def test_big_chunk_is_cut_to_the_cells_in_memory(tmp_path, monkeypatch):
    """A chunk longer than the cell limit is read once and cut to the limit around the centre."""
    monkeypatch.setattr(detect, "_SAMPLE_CELLS", 24)
    ds, da, source = _netcdf(tmp_path, chunks=BIG)
    boxes = _record(monkeypatch)
    out = detect.load_sample(da, ROLES)
    ids = _chunk_ids(boxes, BIG)
    assert len(boxes) == 3 and len(set(ids)) == len(ids) == 6    # a read per time block, the lead's two chunks each
    assert out.sizes["lat"] == 24 and out.sizes["lon"] == 24
    assert abs(float(out.lat.values.mean()) - float(source.lat.values.mean())) < 1.0
    np.testing.assert_array_equal(out.values, _expected(source, out).values)
    ds.close()


def test_contiguous_file_reads_ranges_of_the_wanted_cells(tmp_path, monkeypatch):
    """A contiguous variable is read as today: a centred block of cells, a range of steps per block."""
    ds, da, source = _netcdf(tmp_path, chunks=None)
    boxes = _record(monkeypatch)
    out = detect.load_sample(da, ROLES)
    start = (SHAPE[3] - detect._SAMPLE_CELLS) // 2
    assert out.sizes["lat"] == min(SHAPE[3], detect._SAMPLE_CELLS)
    assert len(boxes) == 3                                   # one range per block of time steps
    assert all(box[1] == (0, 1) for box in boxes)
    assert boxes[0][3] == (start, start + detect._SAMPLE_CELLS)
    np.testing.assert_array_equal(out.values, _expected(source, out).values)
    ds.close()


def test_zarr_sample_follows_its_chunks(tmp_path, monkeypatch):
    """A zarr store is sampled on its chunk grid like a NetCDF file."""
    pytest.importorskip("zarr")
    _array().chunk(dict(zip(DIMS, CHUNKS))).to_dataset().to_zarr(tmp_path / "f.zarr")
    ds = detect.open_data(str(tmp_path / "f.zarr"))
    da = ds["tp"]
    assert da.encoding["chunks"] == CHUNKS
    boxes = _record(monkeypatch)
    out = detect.load_sample(da, ROLES)
    ids = _chunk_ids(boxes, CHUNKS)
    assert len(set(ids)) == len(ids) and {i[1] for i in ids} == {0}
    assert out.sizes["lat"] == CHUNKS[3]
    np.testing.assert_array_equal(out.values, _expected(_array(), out).values)
    ds.close()


def test_time_steps_do_not_depend_on_the_chunk_grid(tmp_path):
    """The start, middle and end steps are those of an unchunked read, whatever the chunks are."""
    reset_run_state()
    picked = []
    for chunks in (CHUNKS, SMALL, None):
        ds, da, _ = _netcdf(tmp_path, chunks=chunks)
        picked.append(list(detect.load_sample(da, ROLES).time.values))
        ds.close()
    assert picked[0] == picked[1] == picked[2]
