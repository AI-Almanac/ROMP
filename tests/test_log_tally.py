"""The notes about negative rain and empty means are shown by the run's verbosity and added up per dataset, once per run
whatever the number of years and workers."""
import warnings

import numpy as np
import pytest
import xarray as xr

geopandas = pytest.importorskip("geopandas")

import hcwf  # noqa: E402
from hcwf import log as logs  # noqa: E402
from hcwf.log import Counted  # noqa: E402
from hcwf.onset.detect import ensemble_onset  # noqa: E402
from test_units_benchmark import _raw, _write_data  # noqa: E402


def _event(dataset, count, lowest):
    warnings.warn(Counted(f"{count} negative values", "negative rain set to 0", dataset, count, lowest,
                          "{dataset}: {count} negative values (lowest {extreme:.2f} mm) were set to 0 ({reads} read{s})",
                          info=True))


def test_counted_warnings_of_a_session_are_added_up_per_dataset(capsys, tmp_path):
    log_file = tmp_path / "run.log"
    with logs.session(logs.NORMAL, log_file):
        for count, lowest in ((2, -0.1), (3, -0.4), (1, -0.2)):
            _event("AIFS", count, lowest)
        _event("GraphCast", 5, -1.0)
    shown = capsys.readouterr().out.splitlines()
    assert shown == ["AIFS: 6 negative values (lowest -0.40 mm) were set to 0 (3 reads)",
                     "GraphCast: 5 negative values (lowest -1.00 mm) were set to 0 (1 read)"]
    assert "AIFS: 6 negative values" in log_file.read_text() and log_file.read_text().count("AIFS:") == 1


def test_quiet_shows_none_of_the_counted_notes_but_the_log_file_keeps_them(capsys, tmp_path):
    log_file = tmp_path / "run.log"
    with logs.session(logs.QUIET, log_file):
        _event("AIFS", 2, -0.1)
        _event("AIFS", 2, -0.1)
    assert capsys.readouterr().out == ""
    assert "AIFS: 4 negative values" in log_file.read_text()


def test_the_ensemble_mean_of_members_without_an_onset_does_not_warn():
    day = np.full((3, 4, 2), np.nan)
    day[0, :2, 0] = [3.0, 4.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        count, fraction, ensemble_day, _ = ensemble_onset(day, np.array(["2020-05-01"] * 3, "datetime64[ns]"), 0.5)
    assert count[:, 0].tolist() == [2, 0, 0] and np.isnan(ensemble_day[1, 0]) and ensemble_day[0, 0] == 4.0


def test_the_fair_brier_term_of_one_member_does_not_warn():
    from hcwf.metrics.probabilistic import _fair_terms

    P = np.array([[0.0, 0.5], [1.0, 0.25]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fair = _fair_terms(np.zeros((2, 2)), P, np.array([1.0, 26.0]))
    assert np.isnan(fair[0, 0]) and np.isinf(fair[0, 1]) and np.isfinite(fair[1]).all()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Units data whose forecasts hold some negative rain."""
    base = tmp_path_factory.mktemp("tally")
    _write_data(base)
    for path in sorted((base / "DET").glob("*.nc")):
        with xr.open_dataset(path) as ds:
            data = ds.load()
        values = data["tp"].values
        values[::5, 0] = -0.3
        data["tp"].values = values
        data.to_netcdf(path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
    return base


def _config(base, workers):
    cfg = _raw(base, "DET", False)
    cfg["output"] = {"save_fig": False}
    cfg["parallel"] = {"workers": workers} if workers > 1 else False
    return cfg


@pytest.mark.parametrize("workers", [1, 2])
def test_a_run_reports_negative_rain_once_per_dataset_and_not_when_quiet(world, capsys, workers):
    hcwf.run(_config(world, workers), quiet=True)
    assert "negative values" not in capsys.readouterr().out

    hcwf.run(_config(world, workers))
    lines = [line for line in capsys.readouterr().out.splitlines() if "negative values" in line]
    assert len(lines) == 1 and lines[0].startswith("DET: ") and "were set to 0" in lines[0]
