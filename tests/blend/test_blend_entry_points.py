"""A run YAML with a `blend:` section: `hcwf-run -p run.yaml` trains it (and checks it with --check), `hcwf.run` returns the
result, run files can be batched, and `hcwf-run convert` writes such a file. Training is a stub here
(hcwf.blend.training.train is tested with the training itself): the run's own part is checked, which is the config it
loads, what it hands to train and where it puts the result."""
import sys
import types
from pathlib import Path

import pytest
import yaml

import differential_onsets as synthetic
from hcwf.cli import main as cli_main
from hcwf.config.convert_specs import convert_blend


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    synthetic.write_data(root)
    return root


class Result:
    def __init__(self, calls):
        self.calls = calls

    def save(self, directory, intermediates=()):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "trained.blend").write_text("stub")
        (directory / "summary_models_pooled.csv").write_text("model\n")


@pytest.fixture
def trained(monkeypatch):
    """hcwf.blend.training with a stub `train` that records what a run hands it."""
    calls = []

    def train(rc):
        calls.append(rc)
        return Result(calls)

    module = types.ModuleType("hcwf.blend.training")
    module.train = train
    monkeypatch.setitem(sys.modules, "hcwf.blend.training", module)
    return calls


def write_run(tmp_path, data, name="plain", file="run.yaml", **changes):
    config = synthetic.configs(data)[name]
    config.update(changes)
    path = tmp_path / file
    path.write_text(convert_blend(config).to_yaml())
    return path


def run_cli(args, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["hcwf-run", *args])
    from hcwf.cli.driver import run_hcwf

    code = 0
    try:
        run_hcwf()
    except SystemExit as exit_:
        code = exit_.code or 0
    return code, capsys.readouterr()


def test_hcwf_run_trains_the_blend_and_writes_into_the_run_folder(tmp_path, data, trained, monkeypatch, capsys):
    run = write_run(tmp_path, data)
    code, out = run_cli(["-p", str(run), "--quiet"], monkeypatch, capsys)
    assert code == 0, out
    [rc] = trained
    cfg, recipe, setup = dict(rc.values), rc.blend, rc.onset
    assert recipe.model_names == ("ens",) and cfg["rain_resolution"] == "exact" and setup.definition.id.endswith("_rule")
    [folder] = (tmp_path / "runs").glob("*_*")
    assert sorted(p.name for p in folder.iterdir()) == ["blend", "definition.yaml", "run.log", "run.yaml"]
    assert sorted(p.name for p in (folder / "blend").iterdir()) == ["summary_models_pooled.csv", "trained.blend"]
    # the written run.yaml reproduces the run: it loads again, with the same blend section and sources
    from hcwf.config.mapping import build_config

    again = build_config(yaml.safe_load((folder / "run.yaml").read_text()), source="run.yaml")
    assert again["blend_years"] == cfg["blend_years"] and tuple(again["obs_paths"]) == tuple(cfg["obs_paths"])
    assert again["model_paths_list"] == cfg["model_paths_list"] and again["units_mode"] == cfg["units_mode"]
    # the same run again is refused unless --overwrite
    code, out = run_cli(["-p", str(run), "--quiet"], monkeypatch, capsys)
    assert code == 1 and "already ran" in out.out
    assert run_cli(["-p", str(run), "--quiet", "--overwrite"], monkeypatch, capsys)[0] == 0


def test_hcwf_run_returns_the_blend_result(tmp_path, data, trained):
    import hcwf

    run = write_run(tmp_path, data)
    result = hcwf.run(run)
    assert isinstance(result.blend, Result) and result.folder is None and result.scores == {}
    written = hcwf.run(run, write=True)
    assert (written.folder / "blend" / "trained.blend").exists()


def test_check_describes_the_blend_and_the_units(tmp_path, data, monkeypatch, capsys):
    run = write_run(tmp_path, data, "regrid")
    code, out = run_cli(["-p", str(run), "--check"], monkeypatch, capsys)
    text = out.out
    assert code == 0, text
    for expected in ("== Units ==", "observations: regrid; weights", "== Blend ==", "outcomes week1, week2, week3, week4, later",
                     "formula b (final): outcome ~ prob_clim_qx + diff_ens_qx", "years           : training 2010, 2011, 2012",
                     "leads 1-40", "output tag '_fixed_cutoff_2010_2013'", "area            : the whole domain of the files (box: all)"):
        assert expected in text, expected
    assert "listed file(s) found" in text


def test_a_blend_section_that_contradicts_itself_cannot_be_loaded(tmp_path, data, monkeypatch, capsys):
    run = write_run(tmp_path, data)
    raw = yaml.safe_load(run.read_text())
    raw["blend"]["cutoff_mode"] = "ref_onset"
    raw["blend"].pop("reference_onset")
    raw["blend"]["bins"] = {"count": 5}
    run.write_text(yaml.safe_dump(raw))
    code, out = run_cli(["-p", str(run), "--check"], monkeypatch, capsys)
    assert code == 1
    assert "'ref_onset' needs onset.reference_onset" in out.out and "need 35 probability days" in out.out
    raw["blend"]["bins"]["wide"] = 1
    raw["blend"]["modles"] = {}
    run.write_text(yaml.safe_dump(raw))
    code, out = run_cli(["-p", str(run), "--check"], monkeypatch, capsys)
    assert code == 1 and 'unknown key \'blend.modles\'. Did you mean "models"?' in out.out
    del raw["blend"]["modles"]
    run.write_text(yaml.safe_dump(raw))
    code, out = run_cli(["-p", str(run), "--check"], monkeypatch, capsys)
    assert code == 1 and 'blend.bins.wide: unknown key. Keys: count, days' in out.out


def test_check_finds_a_formula_naming_a_feature_the_run_does_not_build(tmp_path, data, monkeypatch, capsys):
    run = write_run(tmp_path, data)
    raw = yaml.safe_load(run.read_text())
    raw["blend"]["formulas"] = {"b": "outcome ~ prob_clim_qx + diff_en_qx"}
    run.write_text(yaml.safe_dump(raw))
    code, out = run_cli(["-p", str(run), "--check"], monkeypatch, capsys)
    assert code == 1 and "diff_en_qx" in out.out and "did you mean diff_ens_qx" in out.out


def test_a_missing_units_file_is_a_check_error(tmp_path, data, monkeypatch, capsys):
    run = write_run(tmp_path, data, "regrid")
    (data / "regrid.csv").unlink()
    code, out = run_cli(["-p", str(run), "--check"], monkeypatch, capsys)
    assert code == 1 and "the units' weights file is missing" in out.out


def test_run_files_are_run_one_after_another(tmp_path, data, trained, monkeypatch, capsys):
    a = write_run(tmp_path, data, "plain", "a.yaml")
    b = write_run(tmp_path, data, "table", "b.yaml")
    code, out = run_cli(["-p", str(a), "-p", str(b), "--quiet"], monkeypatch, capsys)
    assert code == 0 and "2 of 2 run files finished" in out.out and len(trained) == 2
    assert [Path(rc.config_file).name for rc in trained] == ["a.yaml", "b.yaml"]
    # a glob names every file it matches, and --check goes through each
    code, out = run_cli(["-p", str(tmp_path / "[ab].yaml"), "--check", "--overwrite"], monkeypatch, capsys)
    assert code == 0 and "2 of 2 run files checked" in out.out and out.out.count("== Blend ==") == 2


def test_one_failing_file_does_not_stop_the_others(tmp_path, data, trained, monkeypatch, capsys):
    good = write_run(tmp_path, data, "plain", "good.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("region: null\nbogus: 1\n")
    code, out = run_cli(["-p", str(bad), "-p", str(good), "--quiet"], monkeypatch, capsys)
    assert code == 1 and len(trained) == 1 and "failed: " in out.out and "bad.yaml" in out.out


def test_a_glob_that_matches_nothing_is_an_error(tmp_path, monkeypatch, capsys):
    code, out = run_cli(["-p", str(tmp_path / "*.yaml")], monkeypatch, capsys)
    assert code == 1 and "no run file matches" in out.out


def test_forecast_needs_the_live_files(capsys):
    # `hcwf-run forecast` applies a trained blend (tests/blend/test_blend_forecast.py); it asks for the live files.
    with pytest.raises(SystemExit) as caught:
        cli_main(["forecast", "trained.blend"])
    assert caught.value.code == 2 and "--forecast" in capsys.readouterr().err


def test_convert_writes_a_run_yaml_from_a_blending_config(tmp_path, data, capsys):
    source = tmp_path / "blend.yml"
    source.write_text(yaml.safe_dump(synthetic.configs(data)["plain"]))
    target = tmp_path / "out.yaml"
    cli_main(["convert", str(source), "-o", str(target), "--region", "India"])
    raw = yaml.safe_load(target.read_text())
    assert raw["region"] == "India" and "blend" in raw and raw["onset"]["rain_resolution"] == "exact"
    assert target.read_text().startswith("# Converted from onset_blending")
    with pytest.raises(SystemExit):   # an existing file is not overwritten
        cli_main(["convert", str(source), "-o", str(target)])
    assert "exists" in capsys.readouterr().err


def test_convert_refuses_a_run_config_and_bad_blending_configs(tmp_path, capsys):
    run = tmp_path / "run.yaml"
    run.write_text("region: ethiopia\n")
    with pytest.raises(SystemExit):
        cli_main(["convert", str(run), "-o", str(tmp_path / "o.yaml")])
    assert "nothing to convert" in capsys.readouterr().err
    bad = tmp_path / "bad.yml"
    bad.write_text(yaml.safe_dump({"name": "x", "observations": {"files": {"folder": "o"}, "variable": "r"},
                                   "models": [], "onset": {"threshold": {"mm": 1}, "defintion": {}},
                                   "evaluation": {}}))
    with pytest.raises(SystemExit):
        cli_main(["convert", str(bad), "-o", str(tmp_path / "o.yaml")])
    assert 'Did you mean "definition"?' in capsys.readouterr().err


def test_init_can_write_a_blend_section(tmp_path):
    from hcwf.config.init import render

    text = render(blend=True, models=[], years=(2000, 2010))
    raw = yaml.safe_load(text)
    assert raw["blend"]["years"]["training"] == "2000:2009" and "MODEL" in raw["blend"]["models"]
    assert "blend" not in yaml.safe_load(render())
