"""A blend trains and forecasts from what the benchmark reads: a zarr store of all years, a file pattern or a list of
files give the same blend (the NetCDF files listed one by one are the reference)."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from _recipes import converted
from test_blend_forecast import (FORECAST_YEARS, LEAD_DAYS, OBS_YEARS, blending_config, write_forecasts,
                                 write_observations)

from hcwf.blend import TrainedBlend
from hcwf.blend.training import train
from hcwf.errors import BlendFileError, IncompatibleForecast


@pytest.fixture(scope="module", autouse=True)
def _detection_cache(tmp_path_factory):
    """The dataset-detection cache of these tests is theirs."""
    patch = pytest.MonkeyPatch()
    patch.setenv("HCWF_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    yield
    patch.undo()


def _store(folder, target, years):
    """Write the yearly files of a folder as one zarr store along time."""
    parts = []
    for year in years:
        with xr.open_dataset(folder / f"{year}.nc") as ds:
            parts.append(ds.load())
    xr.concat(parts, dim="time").to_zarr(target)


def _rc(root, **flat):
    """The run config of the synthetic blend, its flat keys replaced by ``flat``."""
    return converted(blending_config(root)).rc(parallel_workers=1, **flat)


@pytest.fixture(scope="module")
def region(tmp_path_factory):
    """Yearly files, the same as stores, and the blend trained from the files."""
    root = tmp_path_factory.mktemp("stores")
    write_observations(root / "obs")
    write_forecasts(root / "ens")
    _store(root / "obs", root / "obs.zarr", OBS_YEARS)
    _store(root / "ens", root / "ens.zarr", FORECAST_YEARS)
    return root, train(_rc(root))


def _same_features(result, other):
    """The weekly features and the pooled scores of two training runs agree."""
    pd.testing.assert_frame_equal(other.inputs.weekly.frame(), result.inputs.weekly.frame())
    pd.testing.assert_frame_equal(other.evaluation.pooled, result.evaluation.pooled)


def test_training_reads_zarr_stores_as_it_reads_files(region):
    root, files = region

    stores = train(_rc(root, obs_dir=str(root / "obs.zarr"), obs_paths=None,
                       model_dir_list=(str(root / "ens.zarr"),), model_paths_list=(None,)))

    _same_features(files, stores)


def test_a_store_of_more_years_than_the_evaluation_trains_the_blend_of_those_years(region):
    """The issues read are those of the evaluation years (as the benchmark reads them): the store's other years are left."""
    root, files = region
    write_forecasts(root / "ens_more", years=range(2005, 2014))
    _store(root / "ens_more", root / "ens_more.zarr", range(2005, 2014))

    stores = train(_rc(root, obs_dir=str(root / "obs.zarr"), obs_paths=None,
                       model_dir_list=(str(root / "ens_more.zarr"),), model_paths_list=(None,)))

    _same_features(files, stores)


def test_training_reads_a_file_pattern_as_it_reads_files(region):
    root, files = region

    patterned = train(_rc(root, obs_dir=str(root / "obs"), obs_paths=None, obs_file_pattern=("{}.nc",),
                          model_dir_list=(str(root / "ens"),), model_paths_list=(None,), file_pattern_list=("{}.nc",)))

    _same_features(files, patterned)


def test_the_blend_of_a_store_has_the_signature_of_the_blend_of_its_files(region):
    root, files = region
    stores = train(_rc(root, obs_dir=str(root / "obs.zarr"), obs_paths=None,
                       model_dir_list=(str(root / "ens.zarr"),), model_paths_list=(None,)))

    from_files, from_store = TrainedBlend.from_training(files), TrainedBlend.from_training(stores)

    assert from_store.signatures == from_files.signatures
    assert len(from_store.provenance["training_files"]["models"]["ens"]) == 1


def test_a_store_forecasts_an_issue_date_as_its_file_does(region):
    root, files = region
    blend = TrainedBlend.from_training(files)

    from_file = blend.forecast({"ens": root / "ens" / "2013.nc"}, issue_date="2013-05-15")
    from_store = blend.forecast({"ens": root / "ens.zarr"}, issue_date="2013-05-15")

    pd.testing.assert_frame_equal(from_store.weekly_table, from_file.weekly_table)
    assert from_store.issue_dates == [dt.date(2013, 5, 15)]


def test_a_store_of_several_seasons_needs_an_issue_date(region):
    root, files = region

    with pytest.raises(IncompatibleForecast, match="issue_date"):
        TrainedBlend.from_training(files).forecast({"ens": root / "ens.zarr"})


def test_the_years_of_a_store_must_share_one_structure(tmp_path):
    """A store whose last year is issued at another hour is refused like a file of another hour."""
    write_observations(tmp_path / "obs")
    write_forecasts(tmp_path / "ens", years=FORECAST_YEARS[:-1])
    write_forecasts(tmp_path / "late", years=FORECAST_YEARS[-1:],
                    issues_of=lambda y: [dt.datetime(y, 5, 1, 12) + dt.timedelta(days=7 * k) for k in range(13)])
    late = tmp_path / "late" / f"{FORECAST_YEARS[-1]}.nc"
    late.replace(tmp_path / "ens" / late.name)
    _store(tmp_path / "obs", tmp_path / "obs.zarr", OBS_YEARS)
    _store(tmp_path / "ens", tmp_path / "ens.zarr", FORECAST_YEARS)
    result = train(_rc(tmp_path, obs_dir=str(tmp_path / "obs.zarr"), obs_paths=None,
                       model_dir_list=(str(tmp_path / "ens.zarr"),), model_paths_list=(None,)))

    with pytest.raises(BlendFileError, match=r"ens: its training files differ in issue_hours.*ens\.zarr\[2013\]"):
        TrainedBlend.from_training(result)
