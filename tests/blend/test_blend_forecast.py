"""A trained blend on a small synthetic region: train, save, load, forecast (onset_blending's tests/test_pipeline.py
forecast tests, run on hcwf), what the file holds, and a damaged or stale file."""
import datetime as dt
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from _recipes import converted

from hcwf.blend import TrainedBlend
from hcwf.blend import forecast as forecast_module
from hcwf.blend.trained import FEATURES_VERSION, FORMAT
from hcwf.blend.training import train
from hcwf.errors import BlendFileError, IncompatibleForecast

@pytest.fixture(scope="module", autouse=True)
def _detection_cache(tmp_path_factory):
    """The dataset-detection cache of these tests is theirs (not the user's ~/.cache/hcwf)."""
    patch = pytest.MonkeyPatch()
    patch.setenv("HCWF_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    yield
    patch.undo()


LATS, LONS = [10.0, 10.25], [70.0, 70.25]
OBS_YEARS = range(2000, 2014)
FORECAST_YEARS = range(2010, 2014)
LEAD_DAYS = 40


def onset_date(year: int, cell: int, shift: int = 0) -> dt.date:
    """Onsets between late May and early July, varying by year and cell."""
    return dt.date(year, 5, 25) + dt.timedelta(days=(year * 7 + cell * 5) % 40 + shift)


def rain(dates, onset: dt.date) -> np.ndarray:
    """Dry before the onset, 10 mm a day for its first 3 days, then 6 mm a day."""
    days = np.array([(d - onset).days for d in dates])
    return np.where(days < 0, 0.0, np.where(days < 3, 10.0, 6.0)).astype("float32")


def write_observations(folder):
    folder.mkdir()
    for year in OBS_YEARS:
        dates = [dt.date(year, 4, 1) + dt.timedelta(days=i) for i in range(200)]
        data = np.stack([np.stack([rain(dates, onset_date(year, i * 2 + j)) for j in range(len(LONS))], axis=-1)
                         for i in range(len(LATS))], axis=1)
        xr.Dataset({"precip": (("time", "lat", "lon"), data)},
                   coords={"time": pd.to_datetime(dates), "lat": LATS, "lon": LONS}).to_netcdf(folder / f"{year}.nc")


def write_forecasts(folder, years=FORECAST_YEARS, members=3, issues_of=None, name=None, lead_days=LEAD_DAYS,
                    lats=LATS, lons=LONS):
    """{year}.nc season files; issues_of(year) and name(year) give other issues and file names."""
    folder.mkdir(exist_ok=True)
    for year in years:
        issues = issues_of(year) if issues_of else [dt.date(year, 5, 1) + dt.timedelta(days=7 * k) for k in range(13)]
        data = np.zeros((len(issues), members, lead_days, len(lats), len(lons)), dtype="float32")
        for t, issue in enumerate(issues):
            day = dt.date(issue.year, issue.month, issue.day)   # issues may carry a time of day
            leads = [day + dt.timedelta(days=d) for d in range(1, lead_days + 1)]
            for m in range(members):
                for i in range(len(lats)):
                    for j in range(len(lons)):
                        data[t, m, :, i, j] = rain(leads, onset_date(year, i * 2 + j, shift=3 * (m - 1)))
        xr.Dataset({"tp": (("time", "number", "day", "lat", "lon"), data, {"units": "mm"})},
                   coords={"time": pd.to_datetime(issues), "number": np.arange(members),
                           "day": np.arange(1, lead_days + 1), "lat": lats, "lon": lons},
                   ).to_netcdf(folder / (name(year) if name else f"{year}.nc"))


def blending_config(root, forest=False) -> dict:
    evaluation = {"years": {"training": "2010:2012", "cv_holdout": "2010:2012", "true_holdout": 2013},
                  "formulas": {"blended_model": "outcome ~ prob_clim_qx + diff_ens_qx"}}
    if forest:
        evaluation["random_forest"] = {"trees": 5, "min_samples_leaf": 5}
    return {"name": "synthetic", "data_root": str(root), "domain": {"grid_ids": {"decimal_digits": 2, "format": "fixed"}},
            "observations": {"files": {"folder": "obs"}, "variable": "precip"},
            "models": [{"name": "ens", "files": {"folder": "ens"}, "lead_days": {"max": LEAD_DAYS},
                        "rain_predictors": [{"agg": "diff"}]}],
            "onset": {"threshold": {"mm": 20}, "definition": {"window": 3, "dry_spell": {"min_dry_days": 5}}},
            "climatology": {"forecast_window": LEAD_DAYS, "min_onset_years": 5, "cv_by_year": False},
            "evaluation": evaluation}


def trained(root, forest=False):
    loaded = converted(blending_config(root, forest))
    return train(loaded.rc(parallel_workers=1))


@pytest.fixture(scope="module")
def region(tmp_path_factory):
    root = tmp_path_factory.mktemp("region")
    write_observations(root / "obs")
    write_forecasts(root / "ens")
    result = trained(root)
    saved = result.save(root / "trained")
    return root, result, TrainedBlend.load(saved / "trained.blend"), saved / "trained.blend"


@pytest.fixture(scope="module")
def forest_region(region):
    root, _, _, _ = region
    result = trained(root, forest=True)
    return root, result, TrainedBlend.load(result.save(root / "trained_forest") / "trained.blend")


def test_the_random_forest_blend_forecasts_by_day(forest_region, tmp_path):
    root, _, blend = forest_region

    forecast = blend.forecast({"ens": root / "ens" / "2013.nc"})

    daily = forecast.daily_table
    assert list(daily.columns[:4]) == ["id", "time", "lat", "lon"]
    days = [f"p_day_{d}" for d in range(1, 29)] + ["p_later"]
    assert np.allclose(daily[days].sum(axis=1), 1.0)
    assert np.allclose(daily["cv_week1"], daily[[f"p_day_{d}" for d in range(1, 8)]].sum(axis=1))
    assert set(forecast.daily["p_day"].dims) == {"time", "id", "day"}
    saved = forecast.save(tmp_path / "forecast")
    assert (saved / "daily_onset_probabilities.csv").exists() and not list(saved.glob("*.nc"))
    twins = forecast.save(tmp_path / "twins", netcdf=True)
    assert (twins / "daily_onset_probabilities.nc").exists() and (twins / "blended_forecast_probabilities.nc").exists()


def test_a_saved_blend_forecasts_like_its_model(region):
    root, result, blend, _ = region

    forecast = blend.forecast({"ens": root / "ens" / "2013.nc"})

    probabilities = forecast.weekly_table.filter(like="cv_")
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    direct = result.model.predict(forecast.features)
    assert np.array_equal(direct[probabilities.columns].to_numpy(), probabilities.to_numpy())
    one_date = blend.forecast({"ens": root / "ens" / "2013.nc"}, issue_date="2013-05-15")
    assert set(pd.to_datetime(one_date.weekly_table["time"]).dt.date) == {dt.date(2013, 5, 15)}
    assert one_date.issue_dates == [dt.date(2013, 5, 15)]
    assert float(forecast.weekly["cv_week1"].sel(id="10.00_70.00", time="2013-05-15")) == float(
        one_date.weekly["cv_week1"].sel(id="10.00_70.00", time="2013-05-15"))


def test_a_single_issue_reads_only_that_issue(region, monkeypatch):
    """With issue_date the live file is read for that issue alone; the forecast is the one of the full read."""
    from hcwf.io import forecasts as forecasts_io

    root, _, blend, _ = region
    files = {"ens": root / "ens" / "2013.nc"}
    seen = []
    original = forecasts_io.UnitForecast.runs

    def spy(self, bounds=None):
        seen.append([pd.Timestamp(i).date() for i in self.issues])
        return original(self, bounds)

    monkeypatch.setattr(forecasts_io.UnitForecast, "runs", spy)
    full = blend.forecast(files, workers=1)
    assert len(seen[-1]) > 1
    seen.clear()
    one = blend.forecast(files, issue_date="2013-05-15", workers=1)
    assert seen == [[dt.date(2013, 5, 15)]]
    for column in one.weekly_table.filter(like="cv_"):
        whole = full.weekly_table[pd.to_datetime(full.weekly_table["time"]).dt.date == dt.date(2013, 5, 15)]
        assert np.array_equal(whole[column].to_numpy(), one.weekly_table[column].to_numpy())


def test_the_forecast_files_are_blendings_contract(region, tmp_path):
    root, _, blend, _ = region

    saved = blend.forecast({"ens": root / "ens" / "2013.nc"}).save(tmp_path / "forecast")

    table = pd.read_csv(saved / "blended_forecast_probabilities.csv")
    assert list(table.columns[:5]) == ["id", "time", "onset_threshold", "lat", "lon"]
    assert [c for c in table.columns if c.startswith("cv_")] == [f"cv_week{k}" for k in range(1, 5)] + ["cv_later"]
    assert not (saved / "daily_onset_probabilities.csv").exists()
    with pytest.raises(FileExistsError):
        blend.forecast({"ens": root / "ens" / "2013.nc"}).save(saved)


def test_one_initialisation_per_model_is_forecast_for_its_issue_date(region, tmp_path):
    _, _, blend, _ = region
    init = dt.date(2014, 6, 9)
    write_forecasts(tmp_path, years=[2014], issues_of=lambda _: [init], name=lambda _: "latest.nc")

    forecast = blend.forecast({"ens": tmp_path / "latest.nc"})

    assert set(pd.to_datetime(forecast.weekly_table["time"]).dt.date) == {init}
    assert len(forecast.weekly_table) == len(LATS) * len(LONS)


def test_a_live_file_holding_more_forecasts_what_was_trained(region, tmp_path):
    # More members, longer leads and a larger domain than training: trimmed.
    _, _, blend, _ = region
    init = dict(years=[2014], issues_of=lambda _: [dt.date(2014, 6, 9)])
    write_forecasts(tmp_path / "trained", **init)
    write_forecasts(tmp_path / "bigger", members=5, lead_days=LEAD_DAYS + 6, lats=LATS + [10.5], lons=LONS + [70.5],
                    **init)

    expected = blend.forecast({"ens": tmp_path / "trained" / "2014.nc"})
    bigger = blend.forecast({"ens": tmp_path / "bigger" / "2014.nc"})

    pd.testing.assert_frame_equal(bigger.weekly_table, expected.weekly_table, check_exact=True)


@pytest.mark.parametrize("reverse", [{"day": slice(None, None, -1), "lat": slice(None, None, -1),
                                      "time": slice(None, None, -1)}, None])
def test_the_order_of_a_files_dimensions_does_not_matter(region, tmp_path, reverse):
    # Lead days, latitudes and issues reversed, or the dimensions transposed: the same forecast.
    _, _, blend, _ = region
    write_forecasts(tmp_path / "usual", years=[2014])
    with xr.open_dataset(tmp_path / "usual" / "2014.nc") as ds:
        other = (ds.isel(**reverse) if reverse else ds.transpose("lat", "day", "number", "lon", "time")).load()
    (tmp_path / "other").mkdir()
    other.to_netcdf(tmp_path / "other" / "2014.nc")

    expected = blend.forecast({"ens": tmp_path / "usual" / "2014.nc"})
    got = blend.forecast({"ens": tmp_path / "other" / "2014.nc"})

    pd.testing.assert_frame_equal(got.weekly_table, expected.weekly_table, check_exact=True)


def test_files_may_be_named_by_glob_or_list(region, tmp_path):
    root, _, blend, _ = region
    expected = blend.forecast({"ens": root / "ens" / "2013.nc"})

    by_glob = blend.forecast({"ens": str(root / "ens" / "*13.nc")})
    by_list = blend.forecast({"ens": [str(root / "ens" / "2013.nc")]})

    for got in (by_glob, by_list):
        pd.testing.assert_frame_equal(got.weekly_table, expected.weekly_table, check_exact=True)


def test_two_issues_on_one_date_are_refused(region, tmp_path):
    _, _, blend, _ = region
    write_forecasts(tmp_path, years=[2014], issues_of=lambda _: [dt.datetime(2014, 6, 9), dt.datetime(2014, 6, 9, 12)])

    with pytest.raises(ValueError, match="several issues on the same date \\(2014-06-09\\)|issue hour"):
        blend.forecast({"ens": tmp_path / "2014.nc"})


def test_models_initialised_on_different_dates_are_refused(region):
    _, _, blend, _ = region
    held = {"ens": [dt.date(2014, 6, 9)], "det": [dt.date(2014, 6, 2)]}

    issue_date, _, problems = forecast_module._issue_problems(blend, None, held)

    assert issue_date is None
    assert problems == [
        "the live files are initialised on different dates (ens 2014-06-09, det 2014-06-02); "
        "every model needs the same one"]


def test_an_initialisation_outside_the_season_is_refused(region):
    _, _, blend, _ = region

    _, _, problems = forecast_module._issue_problems(blend, None, {"ens": [dt.date(2014, 11, 3)]})

    assert len(problems) == 1 and problems[0].startswith("issue date 2014-11-03 is outside the season")


def test_only_the_trained_models_may_be_given(region, tmp_path):
    root, _, blend, _ = region

    with pytest.raises(IncompatibleForecast) as refused:
        blend.forecast({"other": root / "ens" / "2013.nc"})

    assert any("exactly the trained models (ens); missing ens; not trained on other" in p
               for p in refused.value.problems)


def test_a_forecast_with_other_members_is_refused(region, tmp_path):
    _, _, blend, _ = region
    write_forecasts(tmp_path / "live", years=[2014], members=2)

    with pytest.raises(IncompatibleForecast) as refused:
        blend.forecast({"ens": tmp_path / "live" / "2014.nc"})

    assert any("members [0, 1], trained on [0, 1, 2]" in p for p in refused.value.problems)


def test_a_forecast_with_fewer_members_may_be_allowed(region, tmp_path):
    _, _, blend, _ = region
    write_forecasts(tmp_path / "live", years=[2014], members=2)

    forecast = blend.forecast({"ens": tmp_path / "live" / "2014.nc"}, fewer_members=True)

    probabilities = forecast.weekly_table.filter(like="cv_")
    assert len(probabilities) and np.allclose(probabilities.sum(axis=1), 1.0)


def test_a_forecast_needs_no_data_but_the_file_and_the_live_files(region, tmp_path):
    # Observations, the training forecasts and everything else the run read are gone: the file is the whole blend.
    root, _, _, path = region
    kept = tmp_path / "trained.blend"
    kept.write_bytes(path.read_bytes())
    live = tmp_path / "live"
    live.mkdir()
    (live / "2013.nc").write_bytes((root / "ens" / "2013.nc").read_bytes())
    expected = TrainedBlend.load(path).forecast({"ens": live / "2013.nc"})
    stash = root.rename(root.with_name(root.name + "_gone"))
    try:
        got = TrainedBlend.load(kept).forecast({"ens": live / "2013.nc"})
    finally:
        stash.rename(root)

    pd.testing.assert_frame_equal(got.weekly_table, expected.weekly_table, check_exact=True)


def test_a_training_run_saves_the_trained_blend_beside_its_artifacts(region):
    root, result, _, path = region

    assert path.parent == root / "trained"
    assert not (path.parent / "weekly_features.nc").exists()   # intermediates are opt-in
    assert not [p for p in path.parent.iterdir() if ".tmp" in p.name]


# ---------------------------------------------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------------------------------------------


def members(path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def rewrite(path, target, **changes):
    """A copy of the file with some members changed (None drops one), its digests refreshed."""
    held = members(path)
    manifest = json.loads(held.pop("manifest.json"))
    for name, data in changes.items():
        if data is None:
            held.pop(name)
        else:
            held[name] = data if isinstance(data, bytes) else json.dumps(data).encode()
    manifest["members"] = {name: hashlib.sha256(data).hexdigest() for name, data in held.items()}
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in held.items():
            archive.writestr(name, data)
    return target


def test_the_file_holds_json_yaml_and_arrays_and_no_pickles(region):
    _, _, _, path = region
    held = members(path)
    manifest = json.loads(held["manifest.json"])

    assert manifest["format"] == FORMAT and manifest["format_version"] == 1
    assert manifest["features_version"] == FEATURES_VERSION
    assert {"run.yaml", "definition.yaml", "recipe.json", "model.json", "model.npz", "units.json", "weights.json",
            "reference.json", "reference.npz", "observed.npz", "signatures.json"} <= set(held)
    assert set(manifest["members"]) == set(held) - {"manifest.json"}
    assert all(manifest["members"][n] == hashlib.sha256(held[n]).hexdigest() for n in manifest["members"])
    assert manifest["definition_hash"] and "hcwf_version" in manifest and "git_commit" in manifest
    assert manifest["provenance"]["training_files"]["models"]["ens"]
    for name in held:
        if name.endswith(".npz"):
            np.load(__import__("io").BytesIO(held[name]), allow_pickle=False).files   # loads without pickles
    run = yaml.safe_load(held["run.yaml"])
    assert run["blend"] and run["models"][0]["name"] == "ens"


def test_a_loaded_blend_is_the_trained_one(region):
    _, result, blend, _ = region

    assert blend.units == result.inputs.units
    assert blend.recipe.model_names == ("ens",)
    assert blend.model.features == result.model.features
    assert np.array_equal(blend.model.coef, result.model.coef)
    assert blend.signatures["ens"]["members"] == [0, 1, 2] and blend.signatures["ens"]["lead_days"][-1] == LEAD_DAYS
    assert blend.recipe.climatology.train_years == tuple(result.inputs.climatology.train_years)
    assert blend.format == FORMAT


def test_a_file_saved_under_the_pre_rename_format_id_still_loads(region, tmp_path):
    _, result, _, path = region
    manifest = json.loads(members(path)["manifest.json"])
    manifest["format"] = "momp/trained-blend"
    held = members(path)
    held.pop("manifest.json")
    with zipfile.ZipFile(tmp_path / "old.blend", "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in held.items():
            archive.writestr(name, data)

    blend = TrainedBlend.load(tmp_path / "old.blend")

    assert blend.format == FORMAT and np.array_equal(blend.model.coef, result.model.coef)


def test_a_damaged_blend_is_refused(region, tmp_path):
    _, _, _, path = region
    damaged = tmp_path / "trained.blend"
    held = members(path)
    held["model.npz"] = held["model.npz"][:-8] + b"\x00" * 8
    manifest = held.pop("manifest.json")
    with zipfile.ZipFile(damaged, "w") as archive:
        archive.writestr("manifest.json", manifest)
        for name, data in held.items():
            archive.writestr(name, data)

    with pytest.raises(BlendFileError, match="damaged: model.npz"):
        TrainedBlend.load(damaged)


@pytest.mark.parametrize("member", ["model.npz", "run.yaml", "observed.npz", "manifest.json"])
def test_a_byte_flipped_inside_any_member_is_refused(region, tmp_path, member):
    _, _, _, path = region
    data = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    start = info.header_offset + 30 + len(info.filename) + 4          # inside the member's own (compressed) bytes
    data[start + info.compress_size // 2] ^= 0xFF
    broken = tmp_path / "trained.blend"
    broken.write_bytes(bytes(data))

    with pytest.raises(BlendFileError):
        TrainedBlend.load(broken)


def test_a_file_that_is_not_a_blend_is_refused(tmp_path):
    path = tmp_path / "x.blend"
    path.write_text("not a zip")

    with pytest.raises(BlendFileError, match="not a readable trained blend"):
        TrainedBlend.load(path)


def test_another_features_version_is_refused(region, tmp_path):
    _, _, _, path = region
    held = members(path)
    manifest = json.loads(held["manifest.json"])
    manifest["features_version"] = FEATURES_VERSION + 1
    with zipfile.ZipFile(tmp_path / "stale.blend", "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in held.items():
            if name != "manifest.json":
                archive.writestr(name, data)

    with pytest.raises(BlendFileError, match="feature version"):
        TrainedBlend.load(tmp_path / "stale.blend")


def test_another_format_is_refused(region, tmp_path):
    _, _, _, path = region
    held = members(path)
    manifest = json.loads(held["manifest.json"])
    manifest["format"] = "someone/else"
    with zipfile.ZipFile(tmp_path / "other.blend", "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in held.items():
            if name != "manifest.json":
                archive.writestr(name, data)

    with pytest.raises(BlendFileError, match="not a trained blend this version can read"):
        TrainedBlend.load(tmp_path / "other.blend")


def test_a_definition_that_no_longer_reads_alike_is_refused(region, tmp_path):
    _, _, _, path = region
    definition = yaml.safe_load(members(path)["definition.yaml"])
    definition["onset"]["window"] = "4 days"
    changed = rewrite(path, tmp_path / "changed.blend", **{"definition.yaml": yaml.safe_dump(definition).encode()})

    with pytest.raises(BlendFileError, match="onset definition .* hash differs"):
        TrainedBlend.load(changed)


def test_blend_settings_that_no_longer_rebuild_alike_are_refused(region, tmp_path):
    _, _, _, path = region
    run = yaml.safe_load(members(path)["run.yaml"])
    run["blend"]["fixed_cutoff"] = "06-20"
    changed = rewrite(path, tmp_path / "changed.blend", **{"run.yaml": yaml.safe_dump(run).encode()})

    with pytest.raises(BlendFileError, match="blend settings stored in the file no longer rebuild"):
        TrainedBlend.load(changed)


def test_the_file_is_written_atomically(region, tmp_path):
    _, _, blend, path = region
    target = tmp_path / "deep" / "folder" / "trained.blend"

    assert blend.save(target) == target
    assert sorted(p.name for p in target.parent.iterdir()) == ["trained.blend"]
    assert TrainedBlend.load(target).units == blend.units


def test_a_blend_that_cannot_be_written_leaves_nothing(region, tmp_path, monkeypatch):
    _, _, blend, _ = region
    monkeypatch.setattr(blend, "signatures", {"ens": {"bad": object()}})

    with pytest.raises(TypeError):
        blend.save(tmp_path / "trained.blend")

    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------------------------------------------
# onset_blending's own files
# ---------------------------------------------------------------------------------------------------------------

ORACLE = Path("/home/hcwf/code/onset_blending")


@pytest.mark.skipif(not ORACLE.exists(), reason="the onset_blending checkout is not available")
@pytest.mark.parametrize("forest", [False, True])
def test_an_onset_blending_file_loads_and_forecasts_like_onset_blending(region, tmp_path, forest):
    from _recipes import oracle

    oracle()
    from onset_blending import BlendConfig
    from onset_blending.trained import TrainedBlend as OldBlend
    from onset_blending.training import Resources
    from onset_blending.training import train as train_blending

    root = region[0]
    config = BlendConfig.from_dict(blending_config(root, forest))
    old = train_blending(config, resources=Resources(workers=1, budget_bytes=None))
    path = old.blend.save(tmp_path / "old.blend")
    live = {"ens": root / "ens" / "2013.nc"}

    expected = OldBlend.load(path).forecast(live)
    blend = TrainedBlend.load(path)
    got = blend.forecast(live)

    assert blend.format == "onset-blending/trained-blend" and blend.signatures["ens"]["members"] == [0, 1, 2]
    pd.testing.assert_frame_equal(got.weekly_table, expected.weekly, check_exact=True)
    if forest:
        pd.testing.assert_frame_equal(got.daily_table, expected.daily, check_exact=True)
    with pytest.raises(BlendFileError, match="not written again"):
        blend.save(tmp_path / "again.blend")


@pytest.mark.parametrize("name", ["regrid", "transform", "table", "computed"])
def test_a_blend_on_weights_and_threshold_files_needs_none_of_them_to_forecast(tmp_path, name):
    # Regrid and cell-transform weights, a threshold table, thresholds computed from observations, a reference onset:
    # all are in the file; the files they were read from are gone.
    import differential_training as synthetic

    root = tmp_path / "data"
    root.mkdir()
    synthetic.write_data(root)
    loaded = converted(synthetic.configs(root)[name])
    result = train(loaded.rc(parallel_workers=1))
    path = result.save(tmp_path / "trained") / "trained.blend"
    expected = TrainedBlend.load(path).forecast({"ens": root / "ens" / "2013.nc"})
    kept = tmp_path / "live.nc"
    kept.write_bytes((root / "ens" / "2013.nc").read_bytes())
    root.rename(tmp_path / "gone")

    got = TrainedBlend.load(path).forecast({"ens": kept})

    pd.testing.assert_frame_equal(got.weekly_table, expected.weekly_table, check_exact=True)
    assert len(got.weekly_table)


def test_a_registry_definition_is_frozen_as_text(region):
    # The file holds the definition's text: it is read from there, never looked up in the registry again.
    import copy

    from hcwf.blend.trained import frozen_config
    from hcwf.onset import definition as definition_mod

    _, result, blend, _ = region
    mapping = copy.deepcopy(blend.mapping)
    mapping["onset"].pop("rule")
    mapping["onset"]["definition"] = "icpac_blend"
    mapping["onset"].pop("thresholds")
    registry = definition_mod.load("icpac_blend")

    frozen = frozen_config(mapping, registry.source_yaml)
    cfg, setup = frozen.values, frozen.onset

    assert cfg["onset_definition"] is None and cfg["onset_rule"] is not None
    assert definition_mod.definition_hash(setup.definition) == definition_mod.definition_hash(registry)
    assert setup.sources["definition"].startswith("config onset.rule")


# ---------------------------------------------------------------------------------------------------------------
# hcwf-run forecast, hcwf.api.forecast
# ---------------------------------------------------------------------------------------------------------------


def run_cli(argv):
    from hcwf.cli import main

    try:
        main(["forecast", *map(str, argv)])
    except SystemExit as exit_:
        return exit_.code
    return 0


def test_the_command_writes_beside_the_blend_by_default(region, tmp_path, capsys):
    root, _, _, path = region
    blend = tmp_path / "trained.blend"
    blend.write_bytes(path.read_bytes())

    code = run_cli([blend, "--forecast", f"ens={root / 'ens' / '2013.nc'}", "--issue-date", "2013-06-05"])

    folder = tmp_path / "forecasts" / "2013-06-05"
    assert code == 0 and f"Wrote {folder}" in capsys.readouterr().out
    assert (folder / "blended_forecast_probabilities.csv").exists() and not list(folder.glob("*.nc"))
    assert set(pd.read_csv(folder / "blended_forecast_probabilities.csv")["time"]) == {"2013-06-05"}


def test_the_netcdf_twin_is_written_only_when_asked(region, tmp_path):
    root, _, _, path = region
    blend = tmp_path / "trained.blend"
    blend.write_bytes(path.read_bytes())

    code = run_cli([blend, "--forecast", f"ens={root / 'ens' / '2013.nc'}", "--issue-date", "2013-06-05", "--netcdf",
                    "-o", tmp_path / "out"])

    assert code == 0 and (tmp_path / "out" / "blended_forecast_probabilities.nc").exists()


def test_several_blends_are_applied_one_after_another_into_one_folder_each(region, tmp_path, capsys):
    root, _, _, path = region
    blends = []
    for name in ("india", "kenya"):
        (tmp_path / name).mkdir()
        blends.append(tmp_path / name / "trained.blend")
        blends[-1].write_bytes(path.read_bytes())

    code = run_cli([*blends, "--forecast", f"ens={root / 'ens' / '2013.nc'}", "-o", tmp_path / "out"])

    assert code == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["india", "kenya"]
    first, second = (pd.read_csv(tmp_path / "out" / n / "blended_forecast_probabilities.csv") for n in ("india", "kenya"))
    pd.testing.assert_frame_equal(first, second)


def test_a_refused_forecast_is_reported_and_the_others_still_run(region, tmp_path, capsys):
    root, _, _, path = region
    blend = tmp_path / "trained.blend"
    blend.write_bytes(path.read_bytes())
    write_forecasts(tmp_path / "live", years=[2014], members=2)

    code = run_cli([blend, tmp_path / "missing.blend", "--forecast", f"ens={tmp_path / 'live' / '2014.nc'}",
                    "-o", tmp_path / "out"])

    error = capsys.readouterr().err
    assert code == 1
    assert "members [0, 1], trained on [0, 1, 2] (missing [2])" in error and "missing.blend" in error
    assert not (tmp_path / "out").exists()


def test_fewer_members_and_the_issue_date_are_options_of_the_command(region, tmp_path):
    _, _, _, path = region
    blend = tmp_path / "trained.blend"
    blend.write_bytes(path.read_bytes())
    write_forecasts(tmp_path / "live", years=[2014], members=2)

    code = run_cli([blend, "--forecast", f"ens={tmp_path / 'live' / '2014.nc'}", "--fewer-members", "--issue-date",
                    "2014-06-05", "-o", tmp_path / "out"])

    assert code == 0 and set(pd.read_csv(tmp_path / "out" / "blended_forecast_probabilities.csv")["time"]) == {
        "2014-06-05"}


def test_a_forecast_argument_without_a_model_is_an_error(region, capsys):
    assert run_cli([region[3], "--forecast", "no-equals-sign"]) == 2
    assert "--forecast expects MODEL=PATH" in capsys.readouterr().err


def test_the_api_returns_the_forecast_and_can_write_it(region, tmp_path):
    from hcwf import api

    root, _, blend, path = region
    live = {"ens": root / "ens" / "2013.nc"}

    forecast = api.forecast(blend, live, issue_date="2013-05-08")
    written = api.forecast(path, live, issue_date="2013-05-08", output=tmp_path / "out", write=True)

    pd.testing.assert_frame_equal(written.weekly_table, forecast.weekly_table)
    assert written.folder == tmp_path / "out" and forecast.folder is None
    with pytest.raises(ValueError, match="give output"):
        api.forecast(blend, live, write=True)


def test_a_models_files_are_read_in_parallel_with_the_same_result(region, tmp_path):
    # One season held in two files (as a model's issues may be): read by two workers, the numbers of one.
    root, _, blend, _ = region
    with xr.open_dataset(root / "ens" / "2013.nc") as ds:
        ds = ds.load()
    for index, part in enumerate((slice(0, 6), slice(6, 13))):
        ds.isel(time=part).to_netcdf(tmp_path / f"2013_{index}.nc")
    expected = blend.forecast({"ens": root / "ens" / "2013.nc"})

    got = blend.forecast({"ens": [tmp_path / "2013_0.nc", tmp_path / "2013_1.nc"]}, workers=2)

    pd.testing.assert_frame_equal(got.weekly_table, expected.weekly_table, check_exact=True)


def test_frozen_weights_may_name_their_units_adm3_name_as_onset_blendings_files_do():
    from hcwf.blend.trained import FrozenUnits

    grid = [(a, b) for a in LATS for b in LONS]
    table = pd.DataFrame({"lat": [a for a, _ in grid], "lon": [b for _, b in grid],
                          "adm3_name": ["north", "north", "south", "south"], "weight": [1.0, 2.0, 1.0, 1.0]})

    units, positions = FrozenUnits("regrid", table).select(lat=LATS + [10.5], lon=LONS, keep_ids={"south"})

    assert list(units.ids[positions]) == ["south"]
    transform = units.transform(np.array(LATS + [10.5]), np.array(LONS))
    assert transform.shape == (2, 6) and transform.weighted_matrix[1].sum() == 2.0
