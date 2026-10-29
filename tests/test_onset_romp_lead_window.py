"""The lead window is shortened to the leads whose whole rule fits in the forecast."""

import numpy as np
import pytest
import xarray as xr

from momp.lib.lead_window import apply_lead_window, count_leads, usable_last_lead
from momp.lib.onset_config import OnsetConfigError


def _cfg(definition="icpac", **values):
    """A definition-mode config dict with two verification windows and day bins."""
    return {"onset_legacy": False, "onset_definition": definition, "obs_day_boundary": "00 UTC, labelled by start",
            "threshold_years": (2000, 2010), "model_list": ("AIFS",), "max_forecast_day": 30,
            "verification_window_list": ((1, 15), (16, 30)), "tolerance_days_list": (3, 5),
            "day_bins": ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)), **values}


def test_usable_last_lead_arithmetic():
    """A candidate on lead k needs leads up to k + R - 1."""
    assert usable_last_lead(50, 22) == 29
    assert usable_last_lead(50, 35) == 16


def test_icpac_with_50_leads_is_shortened_to_29():
    """ICPAC reads 22 days from the onset day, so 50 leads allow candidates up to lead 29."""
    cfg = apply_lead_window(_cfg(), lambda dic, i: 50)
    assert cfg["max_forecast_day"] == 29 and cfg["lead_window"] == (1, 29)
    assert cfg["verification_window_list"] == ((1, 15), (16, 29))
    assert cfg["day_bins"][-2:] == ((21, 25), (26, 29))
    assert cfg["onset_lead_lines"] == [
        "Lead window      : 1-29 for AIFS (icpac reads 22 days from the onset day; forecasts have 50 leads)  "
        "[shortened from max_forecast_day 30]"]


def test_no_dry_spell_check_means_no_shortening():
    """Without the veto the rule reads 3 days, so 50 leads cover max_forecast_day 30."""
    cfg = apply_lead_window(_cfg(dry_spell_check=False), lambda dic, i: 50)
    assert cfg["max_forecast_day"] == 30 and cfg["verification_window_list"] == ((1, 15), (16, 30))
    assert "[max_forecast_day]" in cfg["onset_lead_lines"][0]


def test_moron_robertson_reads_35_days():
    """A 5-day window plus 30 days after it: 50 leads leave candidates up to lead 16; later windows are dropped."""
    cfg = apply_lead_window(_cfg("moron_robertson"), lambda dic, i: 50)
    assert cfg["lead_window"] == (1, 16)
    assert cfg["verification_window_list"] == ((1, 15), (16, 16)) and cfg["tolerance_days_list"] == (3, 5)
    cfg = apply_lead_window(_cfg("moron_robertson", verification_window_list=((1, 10), (20, 30))), lambda d, i: 50)
    assert cfg["verification_window_list"] == ((1, 10),) and cfg["tolerance_days_list"] == (3,)
    assert cfg["day_bins"] == ((1, 5), (6, 10), (11, 15), (16, 16))


def test_too_few_leads_is_an_error():
    """When no lead fits, the error names the definition, R and the lead count, and suggests dry_spell_check."""
    with pytest.raises(OnsetConfigError, match=r"icpac.*22 days.*20 leads.*dry_spell_check = False"):
        apply_lead_window(_cfg(), lambda dic, i: 20)


def test_models_with_different_lengths_use_the_shortest():
    """One line per model, and the run uses the shortest window."""
    cfg = apply_lead_window(_cfg(model_list=("A", "B")), lambda dic, i: (50, 40)[i])
    assert cfg["lead_window"] == (1, 19) and len(cfg["onset_lead_lines"]) == 3


def test_legacy_mode_is_untouched():
    """Flat-key configs keep their windows."""
    cfg = {"wet_init": 1, "max_forecast_day": 30}
    assert apply_lead_window(cfg, lambda dic, i: 1) == {"wet_init": 1, "max_forecast_day": 30}


def test_count_leads_reads_the_first_year_file(tmp_path):
    """Leads are the steps from 1 on, found through model_dims."""
    ds = xr.Dataset({"tp": (("time", "day", "lat", "lon"), np.zeros((1, 6, 1, 1)))},
                    coords={"time": np.array(["2015-05-01"], dtype="datetime64[ns]"), "day": np.arange(0, 6),
                            "lat": [0.0], "lon": [0.0]})
    ds.to_netcdf(tmp_path / "2015.nc")
    dic = {"start_date": (2015, 5, 1), "model_dir_list": (str(tmp_path),), "file_pattern_list": ("{}.nc",)}
    assert count_leads(dic, 0) == 5
