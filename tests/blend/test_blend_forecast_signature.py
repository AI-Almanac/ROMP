"""A live forecast must cover what the blend was trained on: the same units, accumulation, daily aggregation, grid
spacing and issue hour, and at least the trained lead days, members and domain (more is trimmed).

onset_blending's tests/test_forecast_signature.py, on `hcwf.config.detect`'s one signature (built on the resolved
dataset entry); the reasons are the ones blending gave."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hcwf.config.detect import forecast_signature, signature_differences

@pytest.fixture(scope="module", autouse=True)
def _detection_cache(tmp_path_factory):
    """The dataset-detection cache of these tests is theirs (not the user's ~/.cache/hcwf)."""
    patch = pytest.MonkeyPatch()
    patch.setenv("HCWF_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    yield
    patch.undo()


WINDOW = "End of the UTC calendar-day aggregation window (00:00 UTC of the following date)"


def entry(**changes) -> dict:
    """The model's dataset entry as the run YAML holds it (dims: role -> the file's dimension)."""
    return {"variable": "tp", "unit_cvt": 1.0, "dims": {"init_time": "time", "step": "step", "member": "number",
                                                       "lat": "lat", "lon": "lon"}, **changes}


def write(path, *, members=4, leads=4, hour=0, units="mm", lat=(10.0, 10.25), lon=(70.0, 70.25), year=2020,
          lead_dim="step", method="sum", window=WINDOW):
    times = pd.date_range(f"{year}-05-01 {hour:02d}:00", periods=3, freq="D")
    dims = ("time",) + (("number",) if members else ()) + (lead_dim, "lat", "lon")
    shape = (len(times),) + ((members,) if members else ()) + (leads, len(lat), len(lon))
    attrs = {"units": units} if units else {}
    if method:
        attrs["cell_methods"] = f"{lead_dim}: {method}"
    coords = {"time": times, lead_dim: (lead_dim, np.arange(1, leads + 1), {"description": window} if window else {}),
              "lat": list(lat), "lon": list(lon)}
    if members:
        coords["number"] = np.arange(members)
    rain = np.random.default_rng(1).gamma(0.6, 4.0, shape).astype("float32") / (1000 if units == "m" else 1)
    xr.Dataset({"tp": (dims, rain, attrs)}, coords=coords).to_netcdf(path)
    return path


def differences(tmp_path, trained_kwargs, live_kwargs, fewer_members=False, lead_days=(1, 3), max_member=None):
    """As the blend compares them: the trained signature after the model's selection, the live one as the file holds."""
    trained = forecast_signature(write(tmp_path / "trained_2020.nc", **trained_kwargs), entry(),
                                 lead_days=lead_days, max_member=max_member)
    live = forecast_signature(write(tmp_path / "live_2026.nc", year=2026, **live_kwargs), entry(), selected=False)
    return signature_differences(trained, live, fewer_members=fewer_members)


def test_a_file_like_the_training_files_matches(tmp_path):
    assert differences(tmp_path, {}, {}) == []


@pytest.mark.parametrize(
    ("live", "expected"),
    [
        ({"members": 3}, "members [0, 1, 2], trained on [0, 1, 2, 3]"),
        ({"leads": 2}, "lead days [1, 2], trained on [1, 2, 3]"),
        ({"hour": 12}, "issue hour (UTC) [12], trained on [0]"),
        ({"units": None}, "rainfall units 'undeclared (x1)', trained on 'mm'"),
        ({"lat": (10.0, 10.5)}, "latitudes every 0.5°, trained on every 0.25° (another resolution)"),
        ({"lat": (10.0,)}, "latitudes 10–10, trained on 10–10.25: the trained domain is not covered"),
        ({"members": 0}, "members None, trained on [0, 1, 2, 3]"),
        ({"method": "mean"}, "daily aggregation"),
        ({"window": "Accumulation over the 24 h after the lead"}, "daily aggregation"),
        ({"window": None, "method": None}, "daily aggregation"),
    ],
)
def test_each_structural_difference_refuses_the_forecast(tmp_path, live, expected):
    [found] = differences(tmp_path, {}, live)
    assert found.startswith(expected)


def test_members_and_lead_days_the_trained_selection_drops_do_not_count(tmp_path):
    # Trained on members 0-1 (max_member 1) and lead days 1-3 of longer files.
    assert differences(tmp_path, {"members": 4, "leads": 5}, {"members": 6, "leads": 4}, max_member=1) == []
    assert differences(tmp_path, {"members": 4}, {"members": 1}, max_member=1) == [
        "members [0], trained on [0, 1] (missing [1])"
    ]


def test_fewer_members_may_be_allowed(tmp_path):
    # Some of the trained members will do; a file with no members still refuses.
    assert differences(tmp_path, {}, {"members": 2}, fewer_members=True) == []
    [found] = differences(tmp_path, {}, {"members": 0}, fewer_members=True)
    assert found.startswith("members None, trained on [0, 1, 2, 3]")


def test_a_live_file_holding_more_is_covered(tmp_path):
    # More members, longer leads and a larger domain at the same resolution.
    bigger = {"members": 8, "leads": 10, "lat": (9.75, 10.0, 10.25, 10.5), "lon": (69.75, 70.0, 70.25)}
    assert differences(tmp_path, {}, bigger) == []


def test_a_smaller_domain_is_refused(tmp_path):
    trained = {"lat": (10.0, 10.25, 10.5)}
    assert differences(tmp_path, trained, {"lat": (10.0, 10.25)}) == [
        "latitudes 10–10.25, trained on 10–10.5: the trained domain is not covered"
    ]


def test_files_in_other_formats_read_alike(tmp_path):
    # An archive in metres with its own lead-day name; a live file in mm on `day`.
    archive = write(tmp_path / "archive_2020.nc", units="m", lead_dim="prediction_timedelta_daily")
    live = write(tmp_path / "live_2026.nc", year=2026, units="mm", lead_dim="day")
    names = {"init_time": "time", "member": "number"}
    a = forecast_signature(archive, entry(dims={**names, "step": "prediction_timedelta_daily"}), lead_days=(1, 3))
    b = forecast_signature(live, entry(dims={**names, "step": "day"}), selected=False)
    assert signature_differences(a, b) == []
    assert a["units"] == "mm"


def test_a_signature_lists_what_a_file_holds(tmp_path):
    signature = forecast_signature(write(tmp_path / "f_2020.nc"), entry(), selected=False)

    assert signature == {
        "variable": "tp", "units": "mm", "accumulation": signature["accumulation"], "dims": {},
        "aggregation": {"method": "sum", "window": WINDOW.lower()}, "issue_hours": [0], "lead_days": [1, 2, 3, 4],
        "members": [0, 1, 2, 3], "lat": [10.0, 10.25], "lon": [70.0, 70.25]}
    assert signature["accumulation"] in ("daily", "auto")


def test_a_stored_signature_with_fewer_fields_compares_only_those(tmp_path):
    # onset_blending's signatures say nothing of the accumulation: only the fields they hold are compared.
    live = forecast_signature(write(tmp_path / "live_2026.nc"), entry(), selected=False)
    stored = {k: v for k, v in live.items() if k != "accumulation"}

    assert signature_differences(stored, live) == []
    assert signature_differences({**stored, "accumulation": "since_init"}, {**live, "accumulation": "daily"}) == [
        "accumulation 'daily', trained on 'since_init'"]


def test_a_file_that_cannot_be_described_says_why(tmp_path):
    with pytest.raises(ValueError, match="no 'rain' variable|rain"):
        forecast_signature(write(tmp_path / "f_2020.nc"), entry(variable="rain"))
