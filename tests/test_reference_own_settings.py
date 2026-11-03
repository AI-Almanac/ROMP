"""A reference model is read with its own accumulation, daily boundary and dims (RunConfig.reference_cases), in both flows.

The pre-R2 flows gave it models[0]'s accumulation and daily boundary (and, in the deterministic flow, no dims), and the
ensemble flow the current model's. Here the only forecast model is daily (DET) and the reference (DETC) holds the same rain
accumulated since init, so the old rule would have read it as daily.
"""
import pytest

from config_world import build_world
from test_run_experience import raw_config, write_config
from typed_io import case_kwargs, flat_config
from hcwf.config.model import RunConfig
from hcwf.config.convert import flat_to_mapping


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("own_ref"))


def _raw(world, **reference):
    return raw_config(world, ("DET",), reference={"model": "DETC", "path": str(world / "DETC"), "variable": "tp",
                                                  "units": "mm", **reference})


def _config(world, name, **reference):
    return flat_config(write_config(world, _raw(world, **reference), name))


def test_detected_accumulation_is_the_references_own(world):
    dic = _config(world, "detected.yaml")
    rc = RunConfig.from_flat(dic)
    model, ref = rc.models[0], rc.reference
    assert model.accumulation == "daily" and ref.accumulation == "since_init" and ref.accumulation_detected
    assert dic["ref_model_accumulation"] == "since_init" and dic["ref_model_accumulation_detected"] is True
    # the same source in both flows, the cases of the ensemble flow named after the forecast model
    for case in rc.reference_cases():
        assert case.model.accumulation == "since_init" and case.model.accumulation_detected
        assert case_kwargs(rc, case, "deterministic")["accumulation"] == "since_init"
    for case in rc.reference_cases(for_model=model):
        assert case.model.accumulation == "since_init" and case.for_model == "DET" and case.name.startswith("DET_")
        assert case_kwargs(rc, case, "ensemble")["accumulation"] == "since_init"


def test_given_settings_win_over_detection_and_every_flow_reads_them(world):
    dic = _config(world, "given.yaml", accumulation="daily", daily_boundary="03 UTC", dims={"member": "realization"})
    rc = RunConfig.from_flat(dic)
    assert not rc.reference.accumulation_detected
    want = {"accumulation": "daily", "daily_boundary": "03 UTC", "model_dims": {"member": "realization"}}
    assert {key: dic[f"ref_model_{key}"] for key in ("accumulation", "daily_boundary", "dims")} == \
        {"accumulation": "daily", "daily_boundary": "03 UTC", "dims": {"member": "realization"}}
    for for_model in (None, rc.models[0]):
        for case, flow in ((c, "ensemble" if for_model else "deterministic") for c in rc.reference_cases(for_model=for_model)):
            kw = case_kwargs(rc, case, flow)
            assert {k: kw[k] for k in want} == want
    # the models keep their own
    assert (rc.models[0].accumulation, rc.models[0].daily_boundary) == ("daily", None)


def test_climatology_reference_has_no_settings(world):
    rc = RunConfig.from_flat(flat_config(write_config(world, raw_config(world, ("DET",)), "clim.yaml")))
    assert rc.reference.kind == "climatology"
    assert (rc.reference.accumulation, rc.reference.daily_boundary) == (None, None)


def test_detected_accumulation_is_not_written_back_but_a_given_one_is(world):
    detected, _ = flat_to_mapping(_config(world, "d2.yaml"))
    assert "accumulation" not in detected["reference"]
    given, _ = flat_to_mapping(_config(world, "g2.yaml", accumulation="since_init", daily_boundary="03 UTC"))
    assert given["reference"]["accumulation"] == "since_init" and given["reference"]["daily_boundary"] == "03 UTC"
