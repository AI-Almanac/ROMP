"""Phase 7 speed-ups that must not change a result: year-set texts, shared reference rain, pre-warming, one worker pool."""
import shutil

import numpy as np

from momp.stats import bins


def _joined(sorted_years, code):
    """The text of a year set the slow, obvious way."""
    return ",".join(str(sorted_years[n]) for n in range(len(sorted_years)) if (code >> n) & 1)


def test_year_set_texts_equal_the_plain_join():
    """_year_set_texts gives the comma-separated years of the set bits, for several set sizes, repeated calls included."""
    rng = np.random.default_rng(1)
    for n_years in (1, 7, 10, 11, 25, 30):
        years = [np.int64(1990 + 3 * k) for k in range(n_years)]
        codes = np.unique(np.concatenate([rng.integers(0, 1 << n_years, 500), [0, (1 << n_years) - 1, 1]]))
        want = [_joined(years, int(c)) for c in codes]
        for _ in range(2):   # the second call is served from what the first wrote
            assert list(bins._year_set_texts(years, codes)) == want
    assert bins._year_set_texts(years, np.zeros(0, np.int64)).shape == (0,)


def _obs_files(folder, years):
    """One NetCDF file of daily rain per year (all of the year), wet from a year-dependent day on."""
    import pandas as pd
    import xarray as xr

    folder.mkdir()
    for year in years:
        rng = np.random.default_rng(year)
        time = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        wet = np.where(time.dayofyear.to_numpy()[:, None, None] > 150 + rng.integers(0, 40), 6.0, 0.4)
        rain = (rng.gamma(0.8, 5.0, size=(len(time), 3, 4)) * wet).astype("float32")
        rain[::17, 0, 0] = np.nan
        xr.Dataset({"rain": (("time", "lat", "lon"), rain)},
                   coords={"time": time, "lat": [10.0, 10.5, 11.0], "lon": [70.0, 70.5, 71.0, 71.5]}
                   ).to_netcdf(folder / f"obs_{year}.nc")


def _reference_config(tmp_path):
    """Definition-mode settings with thresholds computed from the observations of 2000-2005."""
    return dict(onset_legacy=False, onset_definition="moron_robertson_mok", obs_day_boundary="00 UTC, labelled by start",
                obs_dir=str(tmp_path / "obs"), obs_file_pattern=["obs_{}.nc"], obs_var="rain", obs_unit_cvt=None,
                thresh_file=None, thresh_var=None, wet_threshold=20.0, wet_init=1.0, wet_spell=2, dry_spell=1,
                dry_threshold=1.0, dry_extent=1, start_date=(2000, 5, 1), end_date=(2000, 8, 31), fallback_date=None,
                earliest_onset_date=None, threshold_years=(2000, 2005), years_clim=tuple(range(2000, 2007)),
                years=(2006,), region="x", dir_out=str(tmp_path / "out"), cache_dir=str(tmp_path / "cache"),
                max_forecast_day=10, model="M", grid_point=False, parallel=False)


def test_climatology_uses_the_reference_rain_the_thresholds_read(tmp_path, monkeypatch):
    """Climatological onsets from the thresholds' ingested reference years equal those from reading every year again."""
    from momp.io import input as io_input
    from momp.stats import climatology, observed_cache

    _obs_files(tmp_path / "obs", range(2000, 2007))
    monkeypatch.setattr(io_input, "region_select", lambda ds, **kwargs: ds)
    config = _reference_config(tmp_path)
    results, reads = {}, {}
    for shared in (True, False):
        shutil.rmtree(tmp_path / "cache", ignore_errors=True)   # the thresholds are computed afresh, not read from disk
        io_input.clear_obs_cache()
        io_input._loaded_thresholds.clear()
        climatology.reset_memo()
        observed_cache.activate(None)
        thresholds = io_input.load_thresholds(**config)
        assert io_input.reference_held() and thresholds["wet_spell_amount"].notnull().any()
        if not shared:
            io_input.release_reference()
        seen = []
        original = io_input._read_obs_year
        monkeypatch.setattr(io_input, "_read_obs_year", lambda year, *a, **k: seen.append(year) or original(year, *a, **k))
        results[shared] = climatology.compute_climatological_onset_dataset(**config)
        monkeypatch.setattr(io_input, "_read_obs_year", original)
        reads[shared] = sorted(seen)
        assert not io_input.reference_held()   # freed once the climatology has used it
        observed_cache.reset()
    assert reads[True] == [2006]   # only the year outside the thresholds' years is read
    assert set(reads[False]) >= {2000, 2001, 2002, 2003, 2004, 2006}
    assert results[True].equals(results[False]) and results[True].shape[0] == 7


def test_reference_year_is_the_year_ingested_alone(tmp_path, monkeypatch):
    """A year of the shared reference rain equals ingest of that year read on its own, array for array."""
    from momp.io import input as io_input
    from momp.lib.onset_config import onset_setup
    from momp.onset.ingest import ingest

    _obs_files(tmp_path / "obs", range(2000, 2007))
    monkeypatch.setattr(io_input, "region_select", lambda ds, **kwargs: ds)
    config = _reference_config(tmp_path)
    io_input.clear_obs_cache()
    io_input._loaded_thresholds.clear()
    io_input.load_thresholds(**config)
    boundary = onset_setup(config).ingest_boundary
    for year in (2000, 2003, 2005):
        shared = io_input.reference_year(year, config, boundary)
        alone = ingest(io_input.load_imd_rainfall(year, **config), day_boundary=boundary, units="mm")
        np.testing.assert_array_equal(shared.tenths, alone.tenths)
        np.testing.assert_array_equal(shared.missing, alone.missing)
        for name in ("year", "month", "day"):
            np.testing.assert_array_equal(getattr(shared, name), getattr(alone, name))
        assert shared.dims == alone.dims and shared.calendar == alone.calendar
        np.testing.assert_array_equal(shared.coords["time"], alone.coords["time"])
    assert io_input.reference_year(2006, config, boundary) is None                       # outside the reference years
    assert io_input.reference_year(2003, {**config, "obs_var": "other"}, boundary) is None  # other observations
    io_input.clear_obs_cache()
    assert not io_input.reference_held()


def test_shared_reference_rain_keeps_exact_rain_in_millimetres(tmp_path, monkeypatch):
    """With rain_resolution exact, the multi-year reference rain is float mm, equal to each year ingested alone.

    It was preallocated as int32, which truncated exact rain to whole mm and made computed thresholds ~10x too low."""
    from momp.io import input as io_input
    from momp.lib.onset_config import onset_setup
    from momp.onset.ingest import ingest

    _obs_files(tmp_path / "obs", range(2000, 2007))
    monkeypatch.setattr(io_input, "region_select", lambda ds, **kwargs: ds)
    config = {**_reference_config(tmp_path), "rain_resolution": "exact"}
    io_input.clear_obs_cache()
    io_input._loaded_thresholds.clear()
    thresholds = io_input.load_thresholds(**config)
    boundary = onset_setup(config).ingest_boundary
    for year in (2000, 2005):
        shared = io_input.reference_year(year, config, boundary)
        alone = ingest(io_input.load_imd_rainfall(year, **config), day_boundary=boundary, units="mm",
                       rain_resolution="exact")
        assert shared.tenths.dtype.kind == "f"
        np.testing.assert_array_equal(shared.tenths, alone.tenths)
    tenths = {**config, "rain_resolution": None}
    io_input.clear_obs_cache()
    io_input._loaded_thresholds.clear()
    coarse = io_input.load_thresholds(**tenths)
    for name in thresholds:   # exact and tenths thresholds agree to the rounding of the daily rain, not by 10x
        np.testing.assert_allclose(thresholds[name].values, coarse[name].values, rtol=0.05, equal_nan=True)
    io_input.clear_obs_cache()
