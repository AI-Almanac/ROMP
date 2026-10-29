"""Onset settings resolve as explicit value, then region default, then an error naming the key."""

import warnings

import pytest

from momp.lib.assertion import ROMPValidator
from momp.lib.onset_config import OnsetConfigError, check_conflicts, detect_legacy, is_legacy, onset_setup

BOUNDARY = "00 UTC, labelled by start"


def _cfg(**values):
    """A definition-mode config with the loader's flags set."""
    return {"onset_legacy": False, "onset_definition": None, **values}


def test_region_defaults_fill_every_setting():
    """With only the region set, definition, boundary and lead label come from it."""
    setup = onset_setup(_cfg(region="ethiopia"))
    assert setup.definition.id == "icpac"
    assert setup.obs_day_boundary == BOUNDARY
    assert setup.lead_label == "start"
    assert setup.season == "main"
    assert setup.sources["definition"] == "region default (ethiopia)"
    assert setup.sources["lead_label"] == "derived from obs_day_boundary"


def test_region_names_are_case_insensitive_and_aliased():
    """Region lookup ignores case and accepts aliases such as India."""
    assert onset_setup(_cfg(region="INDIA")).definition.id == "moron_robertson_mok"
    assert onset_setup(_cfg(region="India")).lead_label == "end"


def test_explicit_values_beat_the_region():
    """Explicit definition, boundary and lead label override the region's."""
    setup = onset_setup(_cfg(region="ethiopia", onset_definition="sivakumar", obs_day_boundary="03 UTC, labelled by end",
                             model_lead_label="start", onset_backend="reference", threshold_years=[2000, 2010]))
    assert setup.definition.id == "sivakumar"
    assert setup.obs_day_boundary == "03 UTC, labelled by end"
    assert setup.lead_label == "start"
    assert setup.backend == "reference"
    assert setup.threshold_years == (2000, 2010)
    assert setup.sources["definition"] == "config onset_definition"


def test_a_country_inherits_its_region_default():
    """Kenya has no definitions and uses East Africa's default."""
    assert onset_setup(_cfg(region="Kenya", obs_day_boundary=BOUNDARY)).definition.id == "icpac"


def test_no_definition_anywhere_is_an_error():
    """An unknown region with no onset_definition names the key and the fix."""
    with pytest.raises(OnsetConfigError, match=r"onset_definition is not set.*Fix: set onset_definition"):
        onset_setup(_cfg(region="atlantis"))


def test_missing_day_boundary_is_an_error():
    """A region that declares no day boundary needs obs_day_boundary."""
    with pytest.raises(OnsetConfigError, match=r"obs_day_boundary is not set.*Fix: set obs_day_boundary"):
        onset_setup(_cfg(region="east_africa"))


def test_missing_threshold_years_is_an_error():
    """A definition with thresholds needs threshold_years or precomputed thresholds."""
    with pytest.raises(OnsetConfigError, match=r"threshold_years is not set.*thresholds = "):
        onset_setup(_cfg(onset_definition="moron_robertson", obs_day_boundary=BOUNDARY, region="atlantis"))


def test_precomputed_thresholds_replace_threshold_years():
    """Giving every threshold as a file needs no years; names and syntax are checked."""
    cfg = _cfg(onset_definition="moron_robertson", obs_day_boundary=BOUNDARY, region="atlantis")
    setup = onset_setup({**cfg, "thresholds": {"wet_spell_amount": "t.nc:mwset"}})
    assert setup.threshold_files == (("wet_spell_amount", "t.nc", "mwset"),)
    with pytest.raises(OnsetConfigError, match="does not use"):
        onset_setup({**cfg, "thresholds": {"nope": "t.nc:v"}})
    with pytest.raises(OnsetConfigError, match="path.nc:variable"):
        onset_setup({**cfg, "thresholds": {"wet_spell_amount": "t.nc"}})


def test_several_seasons_need_onset_season(tmp_path):
    """A definition with several seasons needs onset_season, and the name must exist."""
    file = tmp_path / "two.yaml"
    file.write_text("id: two\nversion: 1\ntitle: t\nstatus: draft\nextends: icpac\nseasons:\n"
                    "  long_rains: {from: 02-01, to: 05-31}\n  short_rains: {from: 09-01, to: 12-31}\n")
    cfg = _cfg(onset_definition=str(file), obs_day_boundary=BOUNDARY)
    with pytest.raises(OnsetConfigError, match=r"several seasons \(long_rains, short_rains\).*onset_season"):
        onset_setup(cfg)
    assert onset_setup({**cfg, "onset_season": "short_rains"}).season == "short_rains"
    with pytest.raises(OnsetConfigError, match="not a season"):
        onset_setup({**cfg, "onset_season": "monsoon"})


@pytest.mark.parametrize(("key", "value", "message"), [
    ("model_lead_label", "middle", r"model_lead_label.*start.*end"),
    ("onset_backend", "cuda", "onset_backend.*unknown"),
    ("threshold_years", (2010, 2000), "first year after last"),
    ("obs_day_boundary", "noon", "obs_day_boundary"),
    ("onset_definition", "no_such_definition", "onset_definition = 'no_such_definition'"),
])
def test_bad_values_are_errors(key, value, message):
    """Each unusable explicit setting is an error naming the key."""
    with pytest.raises(OnsetConfigError, match=message):
        onset_setup(_cfg(region="ethiopia", **{key: value}))


def test_unavailable_backend_falls_back_with_a_warning(monkeypatch):
    """A backend that cannot load is replaced by the reference backend, with a warning."""
    from momp.onset import engine

    def refuse(name):
        """Pretend no backend module can be imported."""
        raise ImportError("no module")

    monkeypatch.setattr(engine, "_load_backend", refuse)
    with pytest.warns(UserWarning, match="using 'reference'"):
        setup = onset_setup(_cfg(region="ethiopia", onset_backend="gpu"))
    assert setup.backend == "reference"


def test_definition_outside_its_region_warns():
    """A definition listing other regions warns when used elsewhere."""
    with pytest.warns(UserWarning, match="check it applies there"):
        onset_setup(_cfg(region="india", onset_definition="icpac"))


def test_legacy_mode_needs_flat_keys_and_no_onset_definition():
    """Legacy mode is no onset_definition key plus old flat keys; the loader flag wins when present."""
    assert detect_legacy({"wet_init": 1})
    assert not detect_legacy({"wet_init": 1, "onset_definition": None})
    assert not detect_legacy({"region": "india"})
    assert not detect_legacy({"earliest_onset_date": None})
    assert is_legacy({"wet_init": 1}) and not is_legacy({"onset_definition": None})
    assert not is_legacy({"onset_legacy": False, "wet_init": 1})


def test_legacy_setup_uses_the_old_conventions():
    """Legacy: lead k is init + k, no thresholds, and the day boundary comes from the region if known."""
    cfg = dict(wet_init=1, wet_threshold=20, wet_spell=3, dry_spell=7, dry_extent=0, end_date=(2019, 7, 31),
               region="India")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        setup = onset_setup(cfg)
        unknown = onset_setup({**cfg, "region": "atlantis"})
    assert setup.legacy and setup.lead_label == "end" and setup.threshold_years is None
    assert setup.obs_day_boundary == "03 UTC, labelled by end"
    assert unknown.obs_day_boundary is None and unknown.ingest_boundary == BOUNDARY


@pytest.mark.parametrize("flat", [{"wet_init": 1}, {"wet_spell": 3}, {"thresh_file": "t.nc"}, {"earliest_onset_date": (6, 2)}])
def test_flat_keys_beside_a_definition_are_an_error(flat):
    """Any set flat key together with onset_definition is an error pointing to extends."""
    with pytest.raises(OnsetConfigError, match=r"cannot be combined with onset_definition.*extends"):
        check_conflicts({"onset_definition": "icpac", **flat})


def test_unset_flat_keys_do_not_conflict():
    """Flat keys left at None are not a conflict."""
    check_conflicts({"onset_definition": "icpac", "thresh_file": None, "earliest_onset_date": None})


def test_the_validator_reports_onset_errors():
    """ROMPValidator collects the onset errors with the other config errors."""
    base = dict(start_date=(2019, 5, 1), end_date=(2019, 7, 31), start_year_clim=1998, end_year_clim=2024,
                model_list=("AIFS",), verification_window_list=((1, 15),), tolerance_days_list=(3,),
                day_bins=((1, 5),), max_forecast_day=30, probabilistic=False)
    validator = ROMPValidator({**base, "onset_definition": "icpac", "wet_init": 1, "region": "ethiopia"})
    with pytest.raises(Exception, match="cannot be combined with onset_definition"):
        validator.validate()
    assert ROMPValidator({**base, "onset_definition": None, "region": "ethiopia"}).validate()
