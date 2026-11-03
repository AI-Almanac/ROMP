"""The reference years held as only the cells the masks keep give the thresholds, seasons and onsets of the whole box."""
import shutil

import numpy as np
import pandas as pd
import pytest
import typed_io
import xarray as xr
from hcwf.cache import reset_run_state
from hcwf.io import observations as observations_io
from hcwf.io import reference as reference_io
from hcwf.io import thresholds as thresholds_io
from hcwf.onset.climatology import harmonic_seasons

YEARS = range(2000, 2007)


def _obs_files(folder, order):
    """One NetCDF file of daily rain per year over a box of Ethiopia and the sea beside it (the land mask drops cells)."""
    folder.mkdir()
    lat, lon = np.arange(3.0, 15.5, 1.5), np.arange(33.0, 48.5, 1.5)
    for year in YEARS:
        rng = np.random.default_rng(year)
        time = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        wet = np.where(time.dayofyear.to_numpy()[:, None, None] > 150 + rng.integers(0, 40), 6.0, 0.4)
        rain = (rng.gamma(0.8, 5.0, size=(len(time), len(lat), len(lon))) * wet).astype("float32")
        rain[::17, 2, 3] = np.nan
        ds = xr.Dataset({"rain": (("time", "lat", "lon"), rain)}, coords={"time": time, "lat": lat, "lon": lon})
        ds["rain"].transpose(*order).to_dataset().to_netcdf(folder / f"obs_{year}.nc")


def _config(tmp_path, **extra):
    return dict(onset_definition="moron_robertson_mok", obs_day_boundary="00 UTC, labelled by start",
                obs_dir=str(tmp_path / "obs"), obs_file_pattern=["obs_{}.nc"], obs_var="rain", obs_unit_cvt=None,
                start_date=(2000, 5, 1), end_date=(2000, 8, 31), threshold_years=(2000, 2005),
                years_clim=tuple(YEARS), years=(2006,), region="Ethiopia", shp_only=False,
                dir_out=str(tmp_path / "out"), cache_dir=str(tmp_path / "cache"), max_forecast_day=10, model="M",
                grid_point=False, parallel=False, **extra)


def _fresh():
    reset_run_state()
    thresholds_io._loaded_thresholds.clear()
    reset_run_state()


@pytest.mark.parametrize("order", [("time", "lat", "lon"), ("time", "lon", "lat")])
@pytest.mark.parametrize("extra", [{}, {"missing": "zero"}, {"rain_resolution": "exact"}])
def test_kept_cells_give_the_results_of_the_whole_box(tmp_path, monkeypatch, order, extra):
    """Thresholds, harmonic seasons, a year of the reference rain and the climatological onsets: held as the kept
    cells or as the whole box, the same, array for array (cells the masks drop included)."""
    from hcwf.references import climatology

    _obs_files(tmp_path / "obs", order)
    config = _config(tmp_path, **extra)
    got = {}
    for compact in (True, False):
        monkeypatch.setattr(reference_io, "reference_cells",
                            observations_io.reference_cells if compact else (lambda *a, **k: None))
        _fresh()
        shutil.rmtree(tmp_path / "cache", ignore_errors=True)
        thresholds = typed_io.thresholds(config)
        held = reference_io._reference["held"][1]
        assert (held.cells is not None) == compact
        rain = held.on_grid()
        assert not rain.missing.all()
        got[compact] = dict(
            thresholds=thresholds, rain=rain, seasons=harmonic_seasons(held, (2000, 2005)),
            year=reference_io.reference_rain_year(2003, *_inputs(config)).on_grid(),
            onsets=climatology.climatological_onset_dataset(*typed_io.from_case_kwargs(config)))
        if compact:
            assert held.tenths.shape[0] < len(rain.coords["lat"]) * len(rain.coords["lon"])
        reset_run_state()
    a, b = got[True], got[False]
    assert set(a["thresholds"]) == set(b["thresholds"])
    for name in a["thresholds"]:
        xr.testing.assert_identical(a["thresholds"][name], b["thresholds"][name])
    xr.testing.assert_identical(a["seasons"], b["seasons"])
    xr.testing.assert_identical(a["onsets"], b["onsets"])
    for one in ("rain", "year"):
        assert a[one].dims == b[one].dims
        np.testing.assert_array_equal(a[one].tenths, b[one].tenths)
        np.testing.assert_array_equal(a[one].missing, b[one].missing)
        for name in ("lat", "lon", "time"):
            np.testing.assert_array_equal(a[one].coords[name], b[one].coords[name])


def _inputs(config):
    from hcwf.config.onset import onset_setup

    got = typed_io.inputs(config)
    return got.obs, got.area, got.select, onset_setup(config).obs_day_boundary
