import numpy as np
import xarray as xr

from momp.io.output import save_metrics_to_netcdf
from momp.stats.onset_rule import resolve_rule


def test_metrics_netcdf_records_the_onset_definition(tmp_path):
    cfg = {
        "dir_out": str(tmp_path),
        "case_name": "c",
        "onset_rule": "two_stage",
        "onset_rule_params": {"stage1_mm": 15.0},
    }
    save_metrics_to_netcdf({"mae": (("lat",), np.zeros(2))}, cfg)

    attrs = xr.open_dataset(tmp_path / "spatial_metrics_c.nc").attrs
    assert attrs["onset_rule"] == "two_stage"
    assert attrs["onset_rule_fingerprint"] == resolve_rule("two_stage", {"stage1_mm": 15.0}).fingerprint()
    assert '"stage1_mm": 15.0' in attrs["onset_rule_params"]
    assert attrs["momp_version"]


def test_legacy_flat_keys_feed_the_recorded_definition(tmp_path):
    cfg = {"dir_out": str(tmp_path), "case_name": "c", "wet_spell": 5}
    save_metrics_to_netcdf({"mae": (("lat",), np.zeros(2))}, cfg)

    attrs = xr.open_dataset(tmp_path / "spatial_metrics_c.nc").attrs
    assert attrs["onset_rule"] == "legacy"
    assert attrs["onset_rule_fingerprint"] == resolve_rule(wet_spell=5).fingerprint()
