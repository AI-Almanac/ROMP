"""The one-pass numba tenths kernel gives bit-for-bit what the numpy reference gives (values, flags, counts, lowest)."""
import numpy as np
import pytest
import xarray as xr

from hcwf.onset import ingest

pytestmark = pytest.mark.skipif(ingest.numba is None, reason="needs numba")

SCALES = (10.0, 10000.0, 1.0, 0.1, 25.4 * 10.0)
"""Tenths per input unit: millimetres, metres, and a few odd factors."""


def awkward(rng, dtype, scale, n=200_000):
    """Random values of the dtype plus the awkward ones: NaN, inf, negatives, the -0.05 boundary, halves, huge."""
    pieces = [rng.normal(0, 20, n // 4), rng.uniform(-0.2, 0.2, n // 4), rng.uniform(-3, 3, n // 8) * 1e-3,
              np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, -0.05, -0.05000001, -0.04999999, -0.5, 1e30, -1e30,
                        3e38, -3e38, 1e300, 214748364.7, 214748364.75, 214748364.8, -214748364.8, 2147483647.0,
                        2147483647.5, 2147483648.0, -2147483648.0, 0.05, 0.15, 0.25, 0.35, 0.45]),
              (rng.integers(-2000, 2000, n // 4) + 0.5) / scale,        # scaled values on exact or near halves
              rng.integers(-2000, 2000, n // 8) / scale]
    values = np.concatenate(pieces)
    with np.errstate(all="ignore"):
        values = values.astype(dtype)
    rng.shuffle(values)
    return values


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("scale", SCALES)
def test_kernel_equals_numpy_reference(dtype, clip, scale):
    """Tenths, missing flags, number of bad values and the lowest scaled one are identical on awkward random data."""
    rng = np.random.default_rng(7)
    values = awkward(rng, dtype, scale)
    expected = ingest._to_tenths_numpy(values, scale, clip)
    got = ingest._to_tenths(values, scale, clip)
    assert got[0].dtype == np.int32 and got[1].dtype == bool
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])
    assert got[2] == expected[2]
    assert got[3] == expected[3]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_every_half_and_boundary_value(dtype):
    """All the multiples of a quarter tenth around zero and the int32 limit round half to even, as numpy does."""
    values = np.concatenate([np.arange(-4000, 4000) / 4.0, 2147483647.0 + np.arange(-4, 5) / 4.0]).astype(dtype)
    for clip in (False, True):
        expected = ingest._to_tenths_numpy(values, 1.0, clip)
        got = ingest._to_tenths(values, 1.0, clip)
        for a, b in zip(got[:2], expected[:2]):
            np.testing.assert_array_equal(a, b)
        assert got[2:] == expected[2:]


def test_shapes_strides_and_other_dtypes():
    """Multi-dimensional, non-contiguous and integer/float16 input work, with and without a preallocated output."""
    rng = np.random.default_rng(3)
    big = rng.normal(0, 5, (6, 7, 9)).astype(np.float32)
    for values in (big, big[:, ::2, :], big.transpose(2, 0, 1), big.astype(np.float16), (big * 10).astype(np.int16),
                   np.zeros((0, 3), np.float32), np.float32(2.5) * np.ones(())):
        expected = ingest._to_tenths_numpy(values, 10.0, False)
        got = ingest._to_tenths(values, 10.0, False)
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])
        assert got[2:] == expected[2:]
    out = (np.empty(big.shape, np.int32), np.empty(big.shape, bool))
    got = ingest._to_tenths(big, 10.0, True, out=out)
    assert got[0] is out[0] and got[1] is out[1]
    np.testing.assert_array_equal(out[0], ingest._to_tenths_numpy(big, 10.0, True)[0])


@pytest.mark.parametrize("units,clip", [("mm", False), ("m", False), ("mm", True)])
def test_convert_equals_numpy_path(monkeypatch, units, clip):
    """_convert (blocks, gaps, warnings) gives the same arrays and the same warning text with and without numba."""
    rng = np.random.default_rng(11)
    values = rng.normal(2, 15, (5, 4, 30)).astype(np.float32)
    values[0, 0, :3] = (np.nan, -0.07, -3.0)
    da = xr.DataArray(values, dims=("lat", "lon", "time"))
    scale = 1000.0 if units == "m" else 1.0
    positions = np.arange(30) * 2
    monkeypatch.setattr(ingest, "_BLOCK_VALUES", 50)   # several blocks
    results = []
    for use_numba in (True, False):
        with monkeypatch.context() as patch:
            if not use_numba:
                patch.setattr(ingest, "numba", None)
            for pos, shape in ((None, da.shape), (positions, da.shape[:-1] + (60,))):
                with pytest.warns(UserWarning) as caught:
                    t, m = ingest._convert(da, scale, shape, pos, clip_negative=clip)
                results.append((t, m, [str(w.message) for w in caught]))
    for (t1, m1, w1), (t2, m2, w2) in zip(results[:2], results[2:]):
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(m1, m2)
        assert w1 == w2
