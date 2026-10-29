"""Run experience: logging and progress, run folders, --check and its hints, momp.run, the observed-onset cache,
accumulation and daily_boundary in the ROMP ingest, climatology_cv reaching the scoring, and region names."""

import logging
import os
import re
import subprocess
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

import momp
from momp import export
from momp.lib import check as check_mod
from momp.lib import dataset_detect as detect
from momp.lib import run_folder
from momp.lib.config_yaml import build_config
from momp.lib.convention import Setting
from momp.lib.dataset_schema import DatasetError, parse_dataset
from momp.lib.loader import load_run_config
from momp.params.region_def import domain
from momp.stats import detect as stats_detect
from momp.stats import observed_cache
from momp.utils import log as logs

ROOT = Path(__file__).resolve().parents[1]
LAT, LON = [6.0, 8.0, 10.0], [38.0, 40.0, 42.0]
STEPS, MAX_LEAD = 14, 10
BOUNDARY = "00 UTC, labelled by start"
CLIM_YEARS = range(2006, 2021)


# --- synthetic data ------------------------------------------------------------------------------------------

def _obs_rain(year):
    """Daily rain (mm) of May-August, (time, lat, lon): dry until a per-cell onset day, then wet."""
    rng = np.random.default_rng(year)
    time = pd.date_range(f"{year}-05-01", f"{year}-08-31")
    onset = rng.integers(8, 40, size=(1, len(LAT), len(LON)))
    day = np.arange(len(time))[:, None, None]
    rain = rng.gamma(0.6, 6.0, size=(len(time), len(LAT), len(LON))) * np.where(day < onset, 0.1, 2.0)
    return time, rain


def write_obs(folder, lat=None):
    """One observation file per climatology year; lat can be given in another order."""
    folder.mkdir(parents=True)
    lat = LAT if lat is None else lat
    for year in CLIM_YEARS:
        time, rain = _obs_rain(year)
        if lat != LAT:
            rain = rain[:, ::-1]
        xr.Dataset({"precip": (("time", "lat", "lon"), rain, {"units": "mm"})},
                   coords={"time": time, "lat": lat, "lon": LON}).to_netcdf(folder / f"{year}.nc")


def write_forecast(folder, cumulative=False, lat=None, years=(2019, 2020)):
    """Daily-init forecasts for May (lead k holds the observed day init + k - 1, with noise), optionally accumulated."""
    folder.mkdir(parents=True)
    lat = LAT if lat is None else lat
    for year in years:
        time, rain = _obs_rain(year)
        inits = pd.date_range(f"{year}-05-01", f"{year}-05-31")
        noise = np.random.default_rng(year + 7)
        data = np.stack([rain[i:i + STEPS] for i in range(len(inits))])
        data = data * noise.uniform(0.5, 1.6, size=data.shape)
        if cumulative:
            data = np.cumsum(data, axis=1)
        xr.Dataset({"tp": (("time", "step", "lat", "lon"), data, {"units": "mm"})},
                   coords={"time": inits, "step": np.arange(1, STEPS + 1), "lat": lat, "lon": LON}
                   ).to_netcdf(folder / f"{year}.nc")


def raw_config(base, models=("DET",), **extra):
    """A nested run config over the synthetic data, written without an output folder (so a run folder is used)."""
    raw = {"region": "ethiopia",
           "observations": {"path": str(base / "obs"), "units": "mm", "day_boundary": BOUNDARY},
           "models": [{"name": m, "path": str(base / m), "units": "mm", "lead_label": "start"} for m in models],
           "onset": {"dry_spell_check": False, "check_alignment": False},
           "evaluation": {"start_date": [2019, 5, 1], "end_date": [2020, 5, 31], "init_days": [0, 1, 2, 3, 4, 5, 6],
                          "start_year_clim": 2006, "end_year_clim": 2020, "verification_windows": [[1, MAX_LEAD]],
                          "tolerances": [3], "max_forecast_day": MAX_LEAD, "day_bins": [[1, 5], [6, 10]]},
           "output": {"save_fig": False, "plot_spatial_far_mr_mae": False, "plot_climatology_onset": False,
                      "plot_panel_heatmap_error": False}}
    for section, values in extra.items():
        raw.setdefault(section, {}).update(values)
    return raw


def write_config(base, raw, name="run.yaml"):
    """Write a nested config as YAML next to the data and return its path."""
    path = base / name
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Observations and two deterministic forecasts (DET, and DETC holding the same rain accumulated since init)."""
    base = tmp_path_factory.mktemp("exp")
    write_obs(base / "obs")
    write_forecast(base / "DET")
    write_forecast(base / "DETC", cumulative=True)
    return base


def _fresh(world, tmp_path, models=("DET",), **extra):
    """A config file in its own folder pointing at the shared data; returns the path."""
    raw = raw_config(world, models, **extra)
    return write_config(tmp_path, raw)


# --- logging and progress ------------------------------------------------------------------------------------

def test_session_levels_and_run_log(tmp_path, capsys):
    """Normal shows info and warnings, quiet only warnings, verbose everything; the log file always has debug."""
    log = logging.getLogger("momp.test_levels")
    for verbosity, expected in ((logs.NORMAL, ["i", "warning: w"]), (logs.QUIET, ["warning: w"]),
                                (logs.VERBOSE, ["d", "i", "warning: w"])):
        logfile = tmp_path / f"run{verbosity}.log"
        with logs.session(verbosity, logfile):
            log.debug("d")
            log.info("i")
            log.warning("w")
        assert capsys.readouterr().out.splitlines() == expected
        text = logfile.read_text()
        assert "[DEBUG] momp.test_levels: d" in text and "[WARNING] momp.test_levels: w" in text
    assert logging.getLogger("momp").handlers == []


def test_python_warnings_become_log_records(capsys):
    """UserWarnings print as warnings; ingest notes as plain info."""
    from momp.onset.ingest import IngestInfo

    with logs.session(logs.NORMAL):
        warnings.warn("careful", UserWarning)
        warnings.warn("note this", IngestInfo)
    assert capsys.readouterr().out.splitlines() == ["warning: careful", "note this"]


def test_progress_logs_lines_without_tqdm_and_is_silent_when_quiet(capsys):
    """Without a terminal the progress is `desc: i/n` log lines; quiet shows none."""
    with logs.session(logs.NORMAL):
        assert list(logs.progress(range(3), "years")) == [0, 1, 2]
    lines = capsys.readouterr().out.splitlines()
    assert [l.split(" (")[0] for l in lines] == ["years: 1/3", "years: 2/3", "years: 3/3"]
    with logs.session(logs.QUIET):
        assert list(logs.progress(range(2), "years")) == [0, 1]
    assert capsys.readouterr().out == ""


def test_progress_uses_tqdm_when_installed_and_on_a_terminal(monkeypatch, capsys):
    """tqdm is used if importable and stderr is a terminal; it is not a dependency."""
    seen = {}

    def fake_tqdm(items, **kwargs):
        """Record the call and pass the items through."""
        seen.update(kwargs)
        return iter(items)

    monkeypatch.setitem(sys.modules, "tqdm", types.SimpleNamespace(tqdm=fake_tqdm))
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    with logs.session(logs.NORMAL):
        assert list(logs.progress([1, 2], "bar")) == [1, 2]
    assert seen["desc"] == "bar" and seen["total"] == 2 and capsys.readouterr().out == ""


def test_no_print_is_left_on_the_run_path():
    """The run path logs instead of printing (the CLI front end, onset explain and init print on purpose)."""
    allowed = {"momp/cli.py", "momp/onset/__main__.py", "momp/lib/init_config.py", "momp/utils/printing.py",
               "momp/lib/assertion.py", "momp/driver.py"}
    pattern = re.compile(r"(?<![\w.])print\(")
    offenders = []
    for path in (ROOT / "momp").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in allowed or "/viz/" in rel or rel in ("momp/onset/explain.py", "momp/onset/describe.py"):
            continue
        if pattern.search(path.read_text()):
            offenders.append(rel)
    assert offenders == []


# --- region names ---------------------------------------------------------------------------------------------

def test_region_box_comes_from_the_registry_for_any_case():
    """Lowercase and alias names work; the four old names keep their boxes; unknown names are errors."""
    assert domain("ethiopia") == domain("Ethiopia") == (3, 15, 33, 48)
    assert domain("India") == (6.46, 35.51, 68.11, 91.4)
    assert domain("india") == (6.5, 38.5, 66.5, 100)
    from momp.onset import regions

    with pytest.raises(regions.RegionError):
        domain("atlantis")


def test_day_convention_sources_are_in_the_region_notes():
    """The IMD and CHIRPS day-convention findings are recorded with their sources."""
    from momp.onset import regions

    india, ethiopia = regions.load_region("india").notes, regions.load_region("ethiopia").notes
    assert "0830 hrs" in india and "mausam.imd.gov.in" in india and "WEAVR" in india
    assert "CHIRPS/v3.0/daily/readme.txt" in ethiopia and "00:00 UTC on D" in ethiopia


# --- accumulation and daily_boundary ---------------------------------------------------------------------------

def _cumulative(tmp_path, name="fc", growth=True):
    """A forecast folder of per-day rain (growth False: monotone but flat) or its running total."""
    folder = tmp_path / name
    folder.mkdir()
    rng = np.random.default_rng(3)
    daily = rng.gamma(2.0, 3.0, size=(6, 6, 3, 3)) + 1.0
    data = np.cumsum(daily, axis=1) if growth else np.sort(rng.uniform(40, 41, size=daily.shape), axis=1)
    xr.Dataset({"tp": (("time", "step", "lat", "lon"), data, {"units": "mm"})},
               coords={"time": pd.date_range("2020-05-01", periods=6), "step": np.arange(1, 7), "lat": LAT,
                       "lon": LON}).to_netcdf(folder / "2020.nc")
    return str(folder), daily


def test_detection_reports_accumulation_with_evidence(tmp_path):
    """Running totals are detected as since_init, per-day values as daily; both with their evidence."""
    path, _ = _cumulative(tmp_path)
    res = detect.resolve_entry({"path": path, "lead_label": "start"}, "forecast")
    assert res.values["accumulation"] == "since_init"
    assert any(l.startswith("accumulation: since_init  [detected: non-decreasing along lead in 100.0 %")
               for l in res.report())
    folder = tmp_path / "daily"
    folder.mkdir()
    rng = np.random.default_rng(0)
    xr.Dataset({"tp": (("time", "step", "lat", "lon"), rng.gamma(0.5, 8, size=(6, 6, 3, 3)), {"units": "mm"})},
               coords={"time": pd.date_range("2020-05-01", periods=6), "step": np.arange(1, 7), "lat": LAT,
                       "lon": LON}).to_netcdf(folder / "2020.nc")
    res = detect.resolve_entry({"path": str(folder), "lead_label": "start"}, "forecast")
    assert res.values["accumulation"] == "daily" and not res.problems


def test_given_accumulation_wins_and_an_ambiguous_one_is_a_problem_naming_the_key(tmp_path):
    """A written accumulation is kept as specified; flat monotone values must be told by the user."""
    path, _ = _cumulative(tmp_path)
    res = detect.resolve_entry({"path": path, "lead_label": "start", "accumulation": "daily"}, "forecast")
    assert res.values["accumulation"] == "daily" and "accumulation: daily  [specified]" in res.report()
    flat, _ = _cumulative(tmp_path, "flat", growth=False)
    res = detect.resolve_entry({"path": flat, "lead_label": "start"}, "forecast")
    assert "Fix: set accumulation" in res.problems["accumulation"]


def test_unreadable_units_name_the_likely_ones_from_detect_units(tmp_path):
    """Without a units attribute the problem says what the daily totals look like."""
    folder = tmp_path / "m"
    folder.mkdir()
    rng = np.random.default_rng(0)
    xr.Dataset({"tp": (("time", "step", "lat", "lon"), rng.gamma(0.5, 8, size=(6, 6, 3, 3)) / 1000)},
               coords={"time": pd.date_range("2020-05-01", periods=6), "step": np.arange(1, 7), "lat": LAT,
                       "lon": LON}).to_netcdf(folder / "2020.nc")
    res = detect.resolve_entry({"path": str(folder), "lead_label": "start"}, "forecast")
    assert "Daily totals look like m" in res.problems["units"]


def test_daily_boundary_is_a_forecast_dataset_key():
    """A forecast entry takes a UTC clock time; observations and bad times are refused; the flat config carries it."""
    base = {"kind": "forecast", "path": "/x", "variable": "tp", "lead_label": "start"}
    assert parse_dataset({**base, "daily_boundary": "03 UTC"}, "x", require_id=False).daily_boundary == "03 UTC"
    with pytest.raises(DatasetError, match="daily_boundary"):
        parse_dataset({**base, "daily_boundary": "teatime"}, "x", require_id=False)
    with pytest.raises(DatasetError, match="only for forecast"):
        parse_dataset({"kind": "observations", "path": "/x", "variable": "p", "day_boundary": BOUNDARY,
                       "daily_boundary": "03 UTC"}, "x", require_id=False)
    flat = build_config({"region": "ethiopia", "observations": {"path": "/nowhere", "variable": "p", "units": "mm",
                                                                "day_boundary": BOUNDARY, "file_pattern": "{}.nc"},
                         "models": [{"name": "A", "path": "/nowhere", "variable": "tp", "units": "mm",
                                     "lead_label": "start", "file_pattern": "{}.nc", "accumulation": "since_init",
                                     "daily_boundary": "03 UTC"}],
                         "evaluation": {"start_date": [2019, 5, 1], "end_date": [2019, 6, 1]}})
    assert flat["daily_boundary_list"] == ("03 UTC",) and flat["accumulation_list"] == ("since_init",)
    from momp.lib import loader
    from momp.lib.resolved import resolved_config_text

    loader._set_onset_mode(flat)

    text = resolved_config_text(flat)
    assert "daily_boundary" in text and "since_init" in text   # given by the user: written back
    case = export.case_kwargs({**flat, "lead_window": (1, 10)}, "A")
    assert case["accumulation"] == "since_init" and case["daily_boundary"] == "03 UTC"


def test_ingest_model_forecast_uses_the_models_accumulation_and_boundary():
    """Accumulated totals are differenced to the same days as per-day data; absent kwargs mean daily as before."""
    from momp.io.input import ingest_model_forecast

    rng = np.random.default_rng(1)
    daily = xr.DataArray(rng.gamma(2.0, 3.0, size=(4, 3, 3, 5)), dims=("init_time", "lat", "lon", "step"),
                         coords={"init_time": pd.date_range("2020-05-01", periods=4), "lat": LAT, "lon": LON,
                                 "step": np.arange(1, 6)})
    total = daily.cumsum("step")
    setup = types.SimpleNamespace(forecast_boundary="00 UTC, labelled by start", lead_label="start")
    plain = ingest_model_forecast(daily, setup, {"model": "m"})
    differenced = ingest_model_forecast(total, setup, {"model": "m", "accumulation": "since_init"})
    assert np.array_equal(plain.tenths, differenced.tenths)
    assert differenced.ingest_info.accumulation == "since_init"
    wrong = ingest_model_forecast(total, setup, {"model": "m", "accumulation": "daily"})
    assert not np.array_equal(plain.tenths, wrong.tenths)
    six_hourly = xr.concat([daily.isel(step=0).assign_coords(step=pd.Timedelta(hours=6) * k) for k in range(1, 9)],
                           dim="step")
    shifted = ingest_model_forecast(six_hourly.assign_coords(step=pd.to_timedelta([6 * k for k in range(1, 9)], "h")),
                                    setup, {"model": "m", "accumulation": "daily", "daily_boundary": "06 UTC"})
    assert shifted.ingest_info.daily_boundary == "06 UTC" and shifted.ingest_info.resolution == "6-hourly"


def test_accumulated_forecasts_score_like_per_day_forecasts(world, tmp_path):
    """The since_init copy of the same rain, detected automatically, gives the same scores as the per-day files."""
    result = momp.run(_fresh(world, tmp_path, ("DET", "DETC")), quiet=True)
    summary = result.scores["summary"].set_index("model")
    assert list(summary.loc["DET"]) == list(summary.loc["DETC"]) and len(summary) == 3
    assert any("accumulation: since_init  [detected" in line for line in result.config["dataset_report"])


# --- climatology_cv --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("configured, expected", [(None, "leave_one_year_out"), ("none", "none"),
                                                  ("leave_one_year_out", "leave_one_year_out")])
def test_climatology_cv_reaches_the_conditional_scoring(world, tmp_path, monkeypatch, configured, expected):
    """Setting and the scoring kwargs carry climatology_cv, in both modes, to the conditional climatology."""
    seen = []
    from momp.stats import benchmark

    real = benchmark.compute_conditional_climatology_as_forecast

    def spy(*args, **kwargs):
        """Record the mode the scoring passes."""
        seen.append(kwargs["climatology_cv"])
        return real(*args, **kwargs)

    monkeypatch.setattr(benchmark, "compute_conditional_climatology_as_forecast", spy)
    reference = {"model": "conditional_climatology"}
    if configured:
        reference["climatology_cv"] = configured
    raw = raw_config(world, reference=reference)
    result = momp.run(raw, quiet=True)
    assert set(seen) == {expected} and seen
    assert "conditional_climatology" in set(result.scores["summary"]["model"])
    assert "climatology_cv" in Setting.__annotations__
    kw = export.case_kwargs(result.config, "DET", reference=True)
    assert kw["climatology_cv"] == expected


def test_conditional_climatology_differs_between_the_two_modes(world):
    """Refitting without the evaluated year changes the conditional climatology that is scored."""
    outs = {}
    for mode in ("none", "leave_one_year_out"):
        raw = raw_config(world, reference={"model": "conditional_climatology", "climatology_cv": mode})
        scores = momp.run(raw, quiet=True).scores["spatial"][("conditional_climatology", "1-10")]
        outs[mode] = scores["mean_mae"].values
    assert not np.array_equal(outs["none"], outs["leave_one_year_out"], equal_nan=True)


# --- the observed-onset cache ------------------------------------------------------------------------------------

def test_observed_onsets_are_detected_once_per_run_for_all_models_and_references(world, tmp_path, monkeypatch):
    """Two models, a window, the climatology reference: each year is detected once, and the result is returned."""
    calls = []
    real = stats_detect.detect_observed_result

    def counting(rain, thresh, year, **kwargs):
        """Count detections per year."""
        calls.append(year)
        return real(rain, thresh, year, **kwargs)

    monkeypatch.setattr(stats_detect, "detect_observed_result", counting)
    result = momp.run(_fresh(world, tmp_path, ("DET", "DETC")), quiet=True)
    assert sorted(calls) == list(CLIM_YEARS)
    assert isinstance(result.observed, xr.Dataset) and list(result.observed["year"].values) == list(CLIM_YEARS)
    assert result.observed["onset_date"].dtype == np.dtype("datetime64[ns]")
    assert result.observed.attrs["fingerprint"].startswith("obs-")
    assert not (tmp_path / "runs").exists()


def test_cache_hit_miss_and_nearest_entry_are_reported(world, tmp_path, capsys):
    """A first run misses, the same observations hit across runs, and a changed definition names what differs."""
    path = _fresh(world, tmp_path)
    momp.run(path, write=True)
    first = capsys.readouterr().out
    assert re.search(r"Observed onsets: cache miss obs-[0-9a-f]{12} \(no earlier entry", first)
    raw = raw_config(world, evaluation={"verification_windows": [[1, 8]], "tolerances": [3]})
    momp.run(write_config(tmp_path, raw), write=True)
    second = capsys.readouterr().out
    assert re.search(r"Observed onsets: cache hit obs-[0-9a-f]{12} \(15 years cached", second)
    raw = raw_config(world, onset={"override": {"wet_day": "at least 2 mm"}})
    momp.run(write_config(tmp_path, raw), write=True)
    third = capsys.readouterr().out
    assert re.search(r"cache miss obs-[0-9a-f]{12}; nearest cached entry obs-[0-9a-f]{12} differs in: definition", third)
    assert len(list((tmp_path / "runs" / ".cache" / "observed_onsets").glob("obs-*"))) == 2


def test_cached_onsets_equal_detected_onsets(world, tmp_path):
    """What the cache returns is the detected array, bit for bit, including no-onset cells."""
    path = _fresh(world, tmp_path)
    fresh = momp.run(path, quiet=True).observed
    again = momp.run(path, write=True, quiet=True).observed
    third = momp.run(path, write=True, overwrite=True, quiet=True).observed   # now read from disk
    for other in (again, third):
        assert np.array_equal(fresh["onset_date"].values, other["onset_date"].values)
        assert np.array_equal(fresh["lat"].values, other["lat"].values)


def test_the_cache_is_inactive_outside_a_run():
    """Without a run, detect_observed_onset behaves as before: nothing is looked up or stored."""
    observed_cache.reset()
    assert observed_cache.lookup(None, None, 2020, {}) is None
    observed_cache.store(None, None, 2020, {}, None)
    assert observed_cache.current() == {}


# --- run folders ---------------------------------------------------------------------------------------------------

def test_run_folder_name_has_region_definition_years_and_fingerprint(world, tmp_path):
    """The name is <region>_<definition>[+mod]_<years>_<4 hex>, at most 40 characters, next to the config file."""
    path = _fresh(world, tmp_path)
    dic = load_run_config(["-p", str(path)])
    paths = dic["run_paths"]
    assert re.fullmatch(r"ethiopia_icpac\+mod_2019-2020_[0-9a-f]{4}", paths.name) and len(paths.name) <= 40   # dry check off
    plain = {"region": "Ethiopia", "start_date": (2019, 5, 1), "end_date": (2020, 5, 31), "onset_legacy": False,
             "onset_definition": "icpac"}
    assert run_folder.run_name(plain, "ab12cd") == "ethiopia_icpac_2019-2020_ab12"
    assert paths.root == tmp_path / "runs" / paths.name and paths.cache_dir == tmp_path / "runs" / ".cache"
    assert not (tmp_path / "runs").exists()   # planning creates nothing
    modified = load_run_config(["-p", str(_fresh(world, tmp_path, onset={"override": {"wet_day": "at least 2 mm"}}))])
    assert re.fullmatch(r"ethiopia_icpac\+mod_2019-2020_[0-9a-f]{4}", modified["run_paths"].name)
    assert modified["run_paths"].name != paths.name
    assert modified["run_paths"].fingerprint != paths.fingerprint


def test_fingerprint_follows_results_not_output_switches(world, tmp_path):
    """Plotting and the backend leave the fingerprint alone; windows and data change it."""
    base = load_run_config(["-p", str(_fresh(world, tmp_path))])
    plots = load_run_config(["-p", str(_fresh(world, tmp_path, output={"plot_reliability": True}, onset={"backend": "reference"}))])
    other = load_run_config(["-p", str(_fresh(world, tmp_path, evaluation={"verification_windows": [[1, 8]], "tolerances": [3]}))])
    assert base["run_paths"].fingerprint == plots["run_paths"].fingerprint != other["run_paths"].fingerprint


def test_long_names_are_cut_to_forty_characters():
    """A long region and definition id still give a short name that ends with years and fingerprint."""
    dic = {"region": "south_asia", "start_date": (2019, 5, 1), "end_date": (2024, 7, 31), "onset_legacy": False,
           "onset_definition": "moron_robertson_mok", "onset_override": {"a": 1}}
    name = run_folder.run_name(dic, "abcd1234")
    assert len(name) <= 40 and name.endswith("_2019-2024_abcd")


def test_subfolders_only_when_they_would_hold_more_than_one_file(world, tmp_path):
    """Scores and onsets get folders here; figures (nothing planned) stay in the run folder."""
    paths = load_run_config(["-p", str(_fresh(world, tmp_path))])["run_paths"]
    assert paths.dir_out == paths.root / "scores" and paths.onsets == paths.root / "onsets"
    assert paths.dir_fig == paths.root
    quiet = load_run_config(["-p", str(_fresh(world, tmp_path, output={
        "save_nc_spatial_far_mr_mae": False, "save_nc_climatology": False}))])["run_paths"]
    assert quiet.dir_out == quiet.root
    figs = load_run_config(["-p", str(_fresh(world, tmp_path, output={
        "plot_spatial_far_mr_mae": True, "plot_climatology_onset": True}))])["run_paths"]
    assert figs.dir_fig == figs.root / "figures"


def test_run_writes_the_folder_refuses_a_rerun_and_overwrite_clears_it(world, tmp_path):
    """The folder holds run.yaml, definition.yaml, run.log and short-named scores; re-running needs overwrite."""
    path = _fresh(world, tmp_path)
    result = momp.run(path, write=True, quiet=True)
    folder = result.folder
    assert folder.parent == tmp_path / "runs"
    assert {p.name for p in folder.iterdir()} == {"run.yaml", "definition.yaml", "run.log", "scores"}
    assert sorted(p.name for p in (folder / "scores").iterdir()) == [
        "climatology_onset.nc", "spatial_DET_1-10.nc", "spatial_climatology_1-10.nc"]
    text = (folder / "run.log").read_text()
    assert "[INFO]" in text and "[DEBUG]" in text and "Observed onsets: cache" in text
    assert "dir_out" not in (folder / "run.yaml").read_text() and "data_dir" not in (folder / "run.yaml").read_text()
    with pytest.raises(run_folder.RunExistsError, match="--overwrite"):
        momp.run(path, write=True, quiet=True)
    (folder / "stale.txt").write_text("old")
    momp.run(path, write=True, overwrite=True, quiet=True)
    assert not (folder / "stale.txt").exists() and (tmp_path / "runs" / ".cache").is_dir()


def test_explicit_data_dir_wins_and_keeps_the_old_file_names(world, tmp_path):
    """With output.data_dir the files and names are the historical ones; re-running overwrites without a flag."""
    out = tmp_path / "out"
    path = _fresh(world, tmp_path, output={"data_dir": str(out), "figure_dir": str(tmp_path / "fig"),
                                           "save_onsets": True})
    for _ in range(2):
        result = momp.run(path, write=True, quiet=True)
    assert result.folder == out and not (tmp_path / "runs").exists()
    assert sorted(p.name for p in out.glob("*.nc")) == [
        "climatology_onset_doy_2006-2020.nc", "spatial_metrics_DET_1-10.nc", "spatial_metrics_climatology_1-10.nc"]
    assert (out / "resolved_config.yaml").exists() and (out / "resolved_definition.yaml").exists()
    assert (out / "run.log").exists() and (out / ".onset_cache").is_dir()
    assert sorted(p.name for p in (out / "onsets").glob("*.nc")) == [
        "onsets_DET.nc", "onsets_climatology.nc", "onsets_observed.nc"]


def test_export_goes_to_the_runs_onsets_folder_with_short_names(world, tmp_path):
    """momp-run export in a run folder writes onsets/<source>.nc."""
    path = _fresh(world, tmp_path)
    paths = export.export_onsets(export.load_cfg(path), sources=["observed", "DET"])
    folder = tmp_path / "runs" / load_run_config(["-p", str(path)])["run_paths"].name / "onsets"
    assert {k: v for k, v in paths.items()} == {"observed": folder / "observed.nc", "DET": folder / "DET.nc"}
    assert all(p.exists() for p in paths.values())


# --- momp.run ------------------------------------------------------------------------------------------------------

def test_run_returns_xarray_and_pandas_and_writes_nothing_by_default(world, tmp_path, monkeypatch):
    """write=False: results in memory, no run folder or output folder is created in the working directory."""
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tmp_path))
    result = momp.run(raw_config(world), quiet=True)
    assert set(os.listdir(tmp_path)) == before and result.folder is None
    assert isinstance(result.observed, xr.Dataset) and isinstance(result.scores["summary"], pd.DataFrame)
    assert set(result.scores["summary"]["model"]) == {"DET", "climatology"}
    assert {"mean_mae", "false_alarm_rate", "miss_rate"} <= set(result.scores["summary"].columns)
    spatial = result.scores["spatial"][("DET", "1-10")]
    assert isinstance(spatial, xr.Dataset) and spatial["mean_mae"].dims == ("lat", "lon")


def test_run_forecasts_are_the_scored_onsets_and_computed_on_use(world):
    """result.forecasts holds a Dataset per model and climatology reference."""
    result = momp.run(raw_config(world), quiet=True)
    forecasts = result.forecasts
    assert set(forecasts) == {"DET", "climatology"}
    assert "onset_date" in forecasts["DET"] and forecasts["DET"]["onset_date"].dims[0] == "init_time"
    assert result.forecasts is forecasts


def test_run_overrides_config_values_by_flat_name_and_rejects_unknown_ones(world):
    """Keyword arguments override flat keys (checked); a misspelt key is an error with a suggestion."""
    result = momp.run(raw_config(world), max_forecast_day=8, verification_window_list=[[1, 8]],
                      tolerance_days_list=[3], day_bins=[[1, 4], [5, 8]], quiet=True)
    assert result.config["max_forecast_day"] == 8 and ("DET", "1-8") in result.scores["spatial"]
    with pytest.raises(TypeError, match="max_forecast_day"):
        momp.run(raw_config(world), max_forcast_day=8)


def test_run_accepts_a_config_object(world):
    """A momp.config.Config runs like its YAML."""
    from momp.config import Config

    result = momp.run(Config.from_mapping(raw_config(world)), quiet=True)
    assert "DET" in set(result.scores["summary"]["model"])


# --- --check and its hints --------------------------------------------------------------------------------------------

def _cli(cwd, *args):
    """Run momp-run in a subprocess."""
    return subprocess.run([sys.executable, "-m", "momp.cli", *args], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(ROOT)})


def test_check_prints_the_plan_and_runs_nothing(world, tmp_path):
    """Every section is there, the exit code is 0, and no run folder or cache is created."""
    path = _fresh(world, tmp_path)
    done = _cli(tmp_path, "-p", str(path), "--check")
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    for heading in ("Settings", "Datasets", "Files", "Grids", "Lead windows", "Alignment", "Run folder", "Cache",
                    "Summary"):
        assert f"== {heading} ==" in done.stdout
    assert "observations: 15 of 15 yearly files found" in done.stdout and "DET: 2 of 2 yearly files found" in done.stdout
    assert "accumulation: daily  [detected" in done.stdout and "[run config]" in done.stdout
    assert "new: a run would create it" in done.stdout and "0 error(s)" in done.stdout
    assert not (tmp_path / "runs").exists()


def test_check_fails_on_missing_files_a_refused_rerun_and_unresolvable_configs(world, tmp_path):
    """Non-zero exit with the problem named; --overwrite makes an existing run folder acceptable."""
    raw = raw_config(world, evaluation={"end_date": [2021, 5, 31]})
    done = _cli(tmp_path, "-p", str(write_config(tmp_path, raw)), "--check")
    assert done.returncode == 1 and "ERROR: DET forecast files missing for 2021" in done.stdout
    assert "ERROR: observation files missing for 2021" in done.stdout
    ok = _fresh(world, tmp_path)
    momp.run(ok, write=True, quiet=True)
    refused = _cli(tmp_path, "-p", str(ok), "--check")
    assert refused.returncode == 1 and "already ran" in refused.stdout
    assert _cli(tmp_path, "-p", str(ok), "--check", "--overwrite").returncode == 0
    broken = write_config(tmp_path, {"region": "ethiopia", "models": "nope"}, "broken.yaml")
    done = _cli(tmp_path, "-p", str(broken), "--check")
    assert done.returncode == 1 and "nothing was run" in done.stdout


def test_check_through_python(world, tmp_path):
    """momp.check returns the report object."""
    report = momp.check(_fresh(world, tmp_path))
    assert report.ok and "== Summary ==" in report.text() and report.errors == []


def test_cli_run_quiet_verbose_and_overwrite(world, tmp_path):
    """--quiet hides progress, --verbose adds debug lines, a second run is refused until --overwrite."""
    path = _fresh(world, tmp_path)
    quiet = _cli(tmp_path, "-p", str(path), "--quiet")
    assert quiet.returncode == 0 and "Starting ROMP Workflow" not in quiet.stdout
    runs = list((tmp_path / "runs").glob("ethiopia_*"))
    assert len(runs) == 1 and "Starting ROMP Workflow" in (runs[0] / "run.log").read_text()
    again = _cli(tmp_path, "-p", str(path))
    assert again.returncode == 1 and "--overwrite" in again.stdout
    loud = _cli(tmp_path, "-p", str(path), "--overwrite", "--verbose")
    assert loud.returncode == 0 and "Starting ROMP Workflow" in loud.stdout and "Processing" in loud.stdout
    assert "ROMP Workflow completed successfully!" in loud.stdout


def test_help_lists_the_run_control_flags(world, tmp_path):
    """-h with a config lists --check, --overwrite, --quiet and --verbose."""
    done = _cli(tmp_path, "-p", str(_fresh(world, tmp_path)), "-h")
    assert all(flag in done.stdout for flag in ("--check", "--overwrite", "--quiet", "--verbose"))


def _dic(world, tmp_path):
    """The resolved flat config of the synthetic run."""
    return load_run_config(["-p", str(_fresh(world, tmp_path))])


def test_hint_lat_order_and_grid_mismatch(tmp_path):
    """Descending latitudes are noted; forecast cells without an observation cell nearby are a warning with the fix."""
    base = tmp_path
    write_obs(base / "obs", lat=[10.0, 8.0, 6.0])
    write_forecast(base / "DET", lat=[6.5, 8.5, 10.5])
    dic = {"region": "ethiopia", "model_list": ("DET",), "model_dims_list": None}
    report = check_mod.CheckReport()
    samples = {"observations": base / "obs" / "2019.nc", "DET": base / "DET" / "2019.nc"}
    check_mod._grids(dic, report, samples)
    assert any(h.startswith("[observations] latitudes run north to south") for h in report.warnings)
    assert any(h.startswith("[DET] only 0% of the model's cells have an observation cell") and "cdo remapcon" in h
               for h in report.warnings)


def test_hint_leads_not_whole_days_and_units_and_accumulation_lines(tmp_path):
    """Sub-daily steps are flagged; dataset-report warnings and detected running totals become keyed hints."""
    folder = tmp_path / "hourly"
    folder.mkdir()
    xr.Dataset({"tp": (("time", "step", "lat", "lon"), np.ones((2, 8, 3, 3)))},
               coords={"time": pd.date_range("2020-05-01", periods=2),
                       "step": pd.to_timedelta(np.arange(1, 9) * 6, "h"), "lat": LAT, "lon": LON}
               ).to_netcdf(folder / "2020.nc")
    dic = {"model_list": ("H",), "model_dims_list": None, "onset_lead_lines": [], "verification_window_list": ((1, 5),),
           "tolerance_days_list": (3,), "max_forecast_day": 5, "day_bins": ((1, 5),)}
    report = check_mod.CheckReport()
    check_mod._leads(dic, report, {"H": folder / "2020.nc"})
    assert any(h.startswith("[H] the leads are not whole days (6-hourly steps") for h in report.warnings)
    dic["dataset_report"] = ("models[0] H:", "  warning: units are mm but the daily totals look like m (x); check units",
                             "  accumulation: since_init  [detected: non-decreasing along lead in 100.0 % of series]")
    report = check_mod.CheckReport()
    check_mod._datasets(dic, report)
    keys = [w.split("]")[0] for w in report.warnings]
    assert "[units" in keys[0] and "[accumulation" in keys[1] if len(keys) > 1 else False


def test_hint_short_climatology(world, tmp_path):
    """A climatology of a few years is warned about with the keys to set."""
    dic = _dic(world, tmp_path)
    dic.update(start_year_clim=2016, end_year_clim=2020)
    report = check_mod.CheckReport()
    check_mod._short_climatology(dic, report)
    assert report.warnings and report.warnings[0].startswith("[evaluation.start_year_clim] the climatology has only 5 years")


# --- ensembles and the export command ---------------------------------------------------------------------------

def _write_ensemble(folder, members=3):
    """Ensemble forecasts like write_forecast, with a `number` dimension."""
    folder.mkdir(parents=True)
    for year in (2019, 2020):
        time, rain = _obs_rain(year)
        inits = pd.date_range(f"{year}-05-01", f"{year}-05-31")
        rng = np.random.default_rng(year + 7)
        data = np.stack([rain[i:i + STEPS] for i in range(len(inits))])
        data = np.stack([data * rng.uniform(0.5, 1.6, size=data.shape) for _ in range(members)], axis=1)
        xr.Dataset({"tp": (("time", "number", "step", "lat", "lon"), data, {"units": "mm"})},
                   coords={"time": inits, "number": np.arange(members), "step": np.arange(1, STEPS + 1), "lat": LAT,
                           "lon": LON}).to_netcdf(folder / f"{year}.nc")


def test_ensemble_run_returns_score_tables_and_writes_short_csv_names(world, tmp_path):
    """Ensemble scores come back as DataFrames with and without files; run folders get overall_/binned_ CSVs."""
    _write_ensemble(tmp_path / "ENS")
    raw = raw_config(tmp_path, ("ENS",), evaluation={"probabilistic": True}, output={
        "plot_heatmap_bss_auc": False, "plot_reliability": False, "plot_panel_heatmap_skill": False,
        "plot_bar_bss_rpss_auc": False})
    raw["observations"]["path"] = str(world / "obs")
    result = momp.run(raw, quiet=True)
    assert set(result.scores) == {"overall", "binned"}
    assert list(result.scores["overall"][["model", "window"]].iloc[0]) == ["ENS", "1-10"]
    assert "Fair_Brier_Skill_Score" in result.scores["overall"].columns
    written = momp.run(write_config(tmp_path, raw), write=True, quiet=True)
    assert sorted(p.name for p in (written.folder / "scores").iterdir()) == ["binned_ENS_1-10.csv", "overall_ENS_1-10.csv"]
    saved = pd.read_csv(written.folder / "scores" / "overall_ENS_1-10.csv")
    assert saved["Fair_Brier_Skill_Score"].iloc[0] == pytest.approx(result.scores["overall"]["Fair_Brier_Skill_Score"].iloc[0])


def test_export_command_writes_into_the_run_folder(world, tmp_path):
    """`momp-run export` in run-folder mode writes onsets/<source>.nc."""
    path = _fresh(world, tmp_path)
    done = _cli(tmp_path, "export", "-p", str(path), "--sources", "observed,DET")
    assert done.returncode == 0, done.stdout[-800:] + done.stderr[-800:]
    files = sorted(p.name for p in (tmp_path / "runs").glob("*/onsets/*.nc"))
    assert files == ["DET.nc", "observed.nc"]
