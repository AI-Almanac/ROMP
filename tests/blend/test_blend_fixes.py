"""Fixes found while writing the docs: a blend run without a `units:` section, `hcwf-run init --blend`, several blends in
one `hcwf-run forecast`, the intermediates a blend can write, and the dead branches around them."""
import inspect
import sys
from pathlib import Path

import pytest
import yaml

import differential_onsets as synthetic
from hcwf.blend import recipe as R
from hcwf.cli import main as cli_main
from hcwf.config.convert_specs import convert_blend
from hcwf.errors import RecipeError


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    synthetic.write_data(root)
    return root


def run_cli(args, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["hcwf-run", *args])
    from hcwf.cli.driver import run_hcwf

    code = 0
    try:
        run_hcwf()
    except SystemExit as exit_:
        code = exit_.code or 0
    return code, capsys.readouterr()


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def run_without_units(tmp_path, data, name="plain"):
    raw = yaml.safe_load(convert_blend(synthetic.configs(data)[name]).to_yaml())
    raw.pop("units")
    return _write(tmp_path, "run.yaml", yaml.safe_dump(raw))


# -- 1: a blend without units: -------------------------------------------------------------------------------------

def test_a_blend_without_units_uses_the_grid_cells_like_convert(tmp_path, data):
    from hcwf.config.build import resolve
    from hcwf.spatial.units import UnitsSpec

    converted = resolve(_write(tmp_path, "with.yaml", convert_blend(synthetic.configs(data)["plain"]).to_yaml()),
                        write=False, tmp=tmp_path)
    bare = resolve(run_without_units(tmp_path, data), write=False, tmp=tmp_path)
    for source in ("observations", "ens"):
        spec = UnitsSpec.from_flat(bare.values, source)
        assert spec is not None and spec.how == "cells"
        assert spec.id_format == "trimmed" and spec.dissemination_scope == "inputs"
        assert UnitsSpec.from_flat(converted.values, source).how == spec.how
    assert bare.values["config_sources"]["units_mode"] == "default (blend: grid cells)"


def test_a_units_key_of_a_blend_is_left_alone(tmp_path, data):
    from hcwf.config.build import resolve

    raw = yaml.safe_load(convert_blend(synthetic.configs(data)["plain"]).to_yaml())
    raw["units"] = {"id_format": "fixed", "id_digits": 2}
    rc = resolve(_write(tmp_path, "run.yaml", yaml.safe_dump(raw)), write=False, tmp=tmp_path)
    assert rc.values.get("units_mode") is None and rc.values["units_id_format"] == "fixed"


def test_a_benchmark_without_units_still_has_none():
    from hcwf.config.build import _default_blend_units

    values = {"model_list": ("m",)}
    _default_blend_units(values)
    assert "units_mode" not in values


def test_a_blend_without_units_trains(tmp_path, data):
    import hcwf

    assert hcwf.run(run_without_units(tmp_path, data), parallel=False).blend is not None


def test_check_shows_the_default_units(tmp_path, data, monkeypatch, capsys):
    code, out = run_cli(["-p", str(run_without_units(tmp_path, data)), "--check"], monkeypatch, capsys)
    assert code == 0, out.out
    assert "== Units ==\nobservations: cells" in out.out and "ens: cells" in out.out


def test_the_init_blend_skeleton_checks_and_trains(tmp_path, data, monkeypatch, capsys):
    import hcwf
    from hcwf.config.init import write_init

    path = write_init(tmp_path / "skeleton.yaml", obs=str(data / "obs"), models=[str(data / "ens")],
                      years=(2010, 2013), blend=True)
    raw = yaml.safe_load(path.read_text())
    assert "units" not in raw and "blend" in raw
    # what the skeleton asks for (TODOs) and the synthetic files do not declare; a region of its own
    raw["observations"].update(units="mm", day_boundary="00 UTC, labelled by end")
    raw["region"] = None
    raw["onset"] = yaml.safe_load(convert_blend(synthetic.configs(data)["plain"]).to_yaml())["onset"]
    path.write_text(yaml.safe_dump(raw))
    code, out = run_cli(["-p", str(path), "--check"], monkeypatch, capsys)
    assert code == 0, out.out
    assert "== Units ==\nobservations: cells" in out.out
    assert hcwf.run(path, parallel=False).blend is not None


# -- 2: several blends in one forecast ------------------------------------------------------------------------------

def test_blends_of_run_folders_are_named_after_the_run(tmp_path):
    from hcwf.cli import _forecast_targets

    blends = [str(tmp_path / "runs" / name / "blend" / "trained.blend") for name in ("north", "south")]
    assert _forecast_targets(blends, str(tmp_path / "out")) == [str(tmp_path / "out" / "north"),
                                                                 str(tmp_path / "out" / "south")]
    other = [str(tmp_path / "a.blend"), str(tmp_path / "x" / "south" / "trained.blend")]
    assert _forecast_targets(other, "out") == [str(Path("out") / tmp_path.name), str(Path("out") / "south")]
    assert _forecast_targets(blends[:1], "out") == ["out"] and _forecast_targets(blends, None) == [None, None]


def test_two_blends_of_one_name_are_reported_before_anything_runs(tmp_path, monkeypatch, capsys):
    called = []
    monkeypatch.setattr("hcwf.api.forecast", lambda *a, **k: called.append(a))
    blends = [str(tmp_path / "a" / "r1" / "blend" / "trained.blend"), str(tmp_path / "b" / "r1" / "blend" / "trained.blend"),
              str(tmp_path / "c" / "other" / "blend" / "trained.blend")]
    with pytest.raises(SystemExit) as caught:
        cli_main(["forecast", *blends, "--forecast", "m=live.nc", "-o", str(tmp_path / "out")])
    assert caught.value.code == 2 and called == [] and not (tmp_path / "out").exists()
    err = capsys.readouterr().err
    assert "'r1'" in err and blends[0] in err and blends[1] in err and blends[2] not in err


def test_forecast_gives_each_blend_its_own_folder(tmp_path, monkeypatch):
    got = []

    def fake(path, files, **kwargs):
        got.append((path, kwargs["output"]))
        return type("F", (), {"folder": kwargs["output"]})()

    monkeypatch.setattr("hcwf.api.forecast", fake)
    blends = [str(tmp_path / "runs" / n / "blend" / "trained.blend") for n in ("a", "b")]
    assert cli_main(["forecast", *blends, "--forecast", "m=live.nc", "-o", str(tmp_path / "out")]) in (0, None)
    assert [Path(out).name for _, out in got] == ["a", "b"]


# -- 3: intermediates -----------------------------------------------------------------------------------------------

BASE = {"model_list": ("m",), "blend_years": {"training": [2000], "cv_holdout": [2000], "true_holdout": [2001]},
        "blend_formulas": {"f": "outcome ~ prob_clim_qx"}}


@pytest.mark.parametrize("name", ["forecasts", "climatology"])
def test_unwritten_intermediates_are_refused(name):
    with pytest.raises(RecipeError) as caught:
        R.recipe_from_config({**BASE, "blend_outputs": {"intermediates": [name]}})
    text = "\n".join(caught.value.problems)
    assert f"blend.outputs.intermediates: '{name}' is not one of 'observations', 'features'" in text


def test_written_intermediates_are_accepted():
    recipe = R.recipe_from_config({**BASE, "blend_outputs": {"intermediates": ["observations", "features"]}})
    assert recipe.outputs.intermediates == ("observations", "features")


def test_convert_leaves_out_intermediates_hcwf_does_not_write(data):
    config = synthetic.configs(data)["plain"]
    config["outputs"] = {"intermediates": ["observations", "forecasts", "climatology", "features"]}
    conversion = convert_blend(config)
    assert yaml.safe_load(conversion.to_yaml())["blend"]["outputs"]["intermediates"] == ["observations", "features"]
    assert any("forecasts" in note and "not written" in note for note in conversion.notes)
    config["outputs"] = {"intermediates": ["climatology"]}
    assert "outputs" not in yaml.safe_load(convert_blend(config).to_yaml())["blend"]


# -- 4, 5, 6: dead branch, unused parameter, wording ----------------------------------------------------------------

def test_no_blend_extras_branch_and_no_setting_parameter():
    import hcwf.api
    from hcwf.cli.driver import run_hcwf

    assert not hasattr(hcwf.api, "_trainer") and "blend extras" not in inspect.getsource(hcwf.api)
    assert list(inspect.signature(run_hcwf).parameters) == ["cfg"]


def test_check_describes_the_implicit_masks():
    from hcwf.runs import check

    source = inspect.getsource(check)
    assert "old behaviour" not in source and "implicit land and region masks" in source
