"""A forecast of only the cells the masks keep (the loader's `cells`) gives the tables of the same forecast on the whole grid,
whether the cube or a function that reads it is given, and the observed dates of the tables are datetime64."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import typed_io
from hcwf.onset.detect import detect_deterministic, detect_ensemble
from old_flat_rule import old_rule

LAT, LON = np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0, 12.0, 13.0])
INITS = pd.to_datetime(["2020-05-18", "2020-05-25", "2020-06-01"])
KEPT = np.array([[1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 1]], dtype=bool)
RUN = dict(max_forecast_day=20, members=None, onset_percentage_threshold=0.5)


def _grid_cube(members=True):
    """A forecast over the grid whose masked cells are NaN, wet spells at random leads."""
    rng = np.random.default_rng(3)
    shape = (len(INITS), 3, len(LAT), len(LON), 30)
    rain = np.where(rng.random(shape) < 0.5, rng.gamma(0.8, 6.0, shape), 0.0).astype("float32")
    rain[..., ~KEPT, :] = np.nan
    dims = ("init_time", "member", "lat", "lon", "step")
    cube = xr.DataArray(rain, dims=dims, coords={"init_time": INITS, "member": np.arange(3), "lat": LAT, "lon": LON,
                                                  "step": np.arange(1, 31)})
    return cube if members else cube.isel(member=0, drop=True)


def _as_cells(cube):
    """The cube of the kept cells only, as the loader gives it: (init_time, [member], step, cell)."""
    rows, cols = np.nonzero(KEPT)
    kept = cube.values[..., rows, cols, :]                     # (init, [member], cell, step)
    dims = tuple(d for d in cube.dims if d not in ("lat", "lon", "step")) + ("cell", "step")
    coords = {d: cube[d] for d in cube.dims if d not in ("lat", "lon")}
    coords.update(lat=("cell", LAT[rows]), lon=("cell", LON[cols]), cell=rows * len(LON) + cols)
    return xr.DataArray(kept, dims=dims, coords=coords).transpose(*dims[:-2], "step", "cell")


@pytest.fixture
def observed():
    """Observed onsets: dates on the kept cells (one with none), NaT on the masked ones."""
    rng = np.random.default_rng(4)
    days = np.datetime64("2020-06-10", "ns") + rng.integers(0, 30, (len(LAT), len(LON))).astype("timedelta64[D]")
    days = np.where(KEPT, days, np.datetime64("NaT", "ns"))
    days[0, 0] = np.datetime64("NaT", "ns")
    return xr.DataArray(days, coords=[("lat", LAT), ("lon", LON)])


def _typed():
    return typed_io.from_case_kwargs({**old_rule(wet_init=1.0, wet_spell=2, dry_spell=1, dry_extent=1), **RUN})


def _same(a, b):
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True)


@pytest.mark.parametrize("as_function", [False, True])
def test_the_ensemble_tables_of_the_cells_are_the_grids(observed, as_function):
    cube = _grid_cube()
    cells = _as_cells(cube)
    given = (lambda: cells) if as_function else cells
    grid_members, grid_ensemble = detect_ensemble(cube, {}, observed, *_typed())
    cell_members, cell_ensemble = detect_ensemble(given, {}, observed, *_typed())
    assert len(grid_members) and len(grid_ensemble)
    _same(grid_members, cell_members)
    _same(grid_ensemble, cell_ensemble)


def test_the_deterministic_table_of_the_cells_is_the_grids(observed):
    cube = _grid_cube(members=False)
    grid = detect_deterministic(cube, {}, observed, *_typed())
    assert len(grid)
    _same(grid, detect_deterministic(_as_cells(cube), {}, observed, *_typed()))


def test_the_tables_hold_their_dates_as_datetime64(observed):
    members, ensemble = detect_ensemble(_grid_cube(), {}, observed, *_typed())
    deterministic = detect_deterministic(_grid_cube(members=False), {}, observed, *_typed())
    for table, columns in ((members, ["obs_onset_date"]), (ensemble, ["onset_date", "obs_onset_date"]),
                           (deterministic, ["onset_date", "obs_onset_date"])):
        for column in columns:
            assert table[column].dtype == np.dtype("datetime64[ns]"), column
    assert ensemble["onset_date"].isna().any() and ensemble["onset_date"].notna().any()
    found = ensemble["onset_day"].notna()
    expected = (ensemble["init_time"] + pd.to_timedelta(ensemble["onset_day"], "D")).astype("datetime64[ns]")
    pd.testing.assert_series_equal(ensemble.loc[found, "onset_date"], expected[found], check_names=False)


def test_the_alignment_check_of_the_cells_is_the_grids():
    """The forecast is checked against the observations on the grid: the cells are put back on it (NaN elsewhere)."""
    from hcwf.onset.detect import alignment_report

    rng = np.random.default_rng(5)
    time = pd.date_range("2020-04-01", "2020-09-30")
    obs = xr.DataArray(np.where(KEPT[None], rng.gamma(0.8, 6.0, (len(time), len(LAT), len(LON))), np.nan),
                       coords=[("time", time), ("lat", LAT), ("lon", LON)])
    cube = _grid_cube()
    rc, case, _ = _typed()
    grid_report, note = alignment_report(obs, cube, rc, case)
    assert grid_report is not None, note
    cell_report, note = alignment_report(obs, _as_cells(cube), rc, case)
    assert cell_report is not None, note
    assert cell_report.summary() == grid_report.summary()
