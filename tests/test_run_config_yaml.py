"""YAML run configs reproduce old config.in runs byte for byte; init, datasets, convert and resolved_config work."""

import filecmp
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from momp.lib import convert
from momp.lib.config_yaml import build_config
from momp.lib.resolved import write_resolved
from tests.synthetic_data import write_forecast, write_obs

ROOT = Path(__file__).resolve().parents[1]
OLD_CONFIG = f'''
project_name = "tiny"
work_dir = "."
pkg_dir = "{ROOT}"
layout = ("model", "verification_window")
model_list = ("AIFS",)
obs = "IMD"
obs_dir = "{ROOT}/demo/data/obs"
obs_file_pattern = ("{{}}.nc",)
obs_var = "RAINFALL"
obs_unit_cvt = None
ref_model = "climatology"
ref_model_dir = "{ROOT}/demo/data/obs"
ref_model_file_pattern = "{{}}.nc"
ref_model_var = "RAINFALL"
ref_model_unit_cvt = None
model_dir_list = ("{ROOT}/demo/data/aifs",)
model_var_list = ("tp",)
unit_cvt_list = (None,)
file_pattern_list = ("{{}}.nc",)
region = "India"
nc_mask = None
shpfile_dir = None
polygon = False
onset_definition = "moron_robertson_mok"
onset_season = None
threshold_years = (2013, 2020)
thresholds = None
obs_day_boundary = None
model_lead_label = None
check_alignment = False
dry_spell_check = False
forecast_history = "none"
threshold_source = "observations"
onset_backend = "reference"
onset_percentage_threshold = 0.5
explain_cells = None
start_date = (2015, 5, 1)
end_date = (2015, 6, 30)
start_year_clim = 2013
end_year_clim = 2020
init_days = (2, 5)
date_filter_year = 2015
verification_window_list = ((1, 15),)
tolerance_days_list = (3,)
max_forecast_day = 15
day_bins = ((1, 5), (6, 10), (11, 15))
FAR = True
MAE = True
MR = True
probabilistic = False
members = "All"
BS = True
RPS = True
AUC = True
Reliability = True
skill_score = True
dir_out = "out"
dir_fig = "fig"
save_fig = False
save_nc_spatial_far_mr_mae = True
save_csv_score = True
save_nc_climatology = True
plot_spatial_far_mr_mae = False
plot_heatmap_bss_auc = False
plot_reliability = False
plot_climatology_onset = False
plot_panel_heatmap_error = False
plot_panel_heatmap_skill = False
plot_bar_bss_rpss_auc = False
show_plot = False
show_panel = False
parallel = False
debug = False
'''
OLD_RUN = ("import sys\nimport momp.lib.loader as L\nfrom momp.lib.convert import read_legacy\n"
           "L._read_config = read_legacy\nsys.argv = ['momp-run', '-p', 'c.in']\n"
           "from momp.driver import run_momp\nrun_momp()\n")


def _run(cwd, *args, code=None):
    """Run momp in a subprocess in cwd, returning the completed process."""
    cmd = [sys.executable, "-c", code] if code else [sys.executable, "-m", "momp.cli", *args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})


def _same_outputs(a: Path, b: Path):
    """Every NetCDF output of a exists in b with identical bytes."""
    files = sorted(p.name for p in (a / "out").glob("*.nc"))
    assert files
    for name in files:
        assert filecmp.cmp(a / "out" / name, b / "out" / name, shallow=False), name


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Run the demo once with the old loader on config.in, once from the converted YAML, once from its resolved config."""
    base = tmp_path_factory.mktemp("runs")
    old, new, again = base / "old", base / "new", base / "again"
    for d in (old, new, again):
        d.mkdir()
    (old / "c.in").write_text(OLD_CONFIG)
    done = _run(old, code=OLD_RUN)
    assert done.returncode == 0, done.stderr[-2000:]
    (new / "c.in").write_text(OLD_CONFIG)
    done = _run(new, "convert", "c.in", "-o", "run.yaml")
    assert done.returncode == 0, done.stderr
    done = _run(new, "-p", "run.yaml")
    assert done.returncode == 0, done.stderr[-2000:]
    resolved = yaml.safe_load((new / "out" / "resolved_config.yaml").read_text())
    resolved["output"].update(data_dir="out", figure_dir="fig")
    resolved["work_dir"] = "."
    (again / "r.yaml").write_text(yaml.safe_dump(resolved))
    done = _run(again, "-p", "r.yaml")
    assert done.returncode == 0, done.stderr[-2000:]
    return old, new, again


def test_converted_yaml_reproduces_the_old_config_byte_for_byte(runs):
    """momp-run on the converted YAML writes the same NetCDF files as the old config.in with the old loader."""
    old, new, _ = runs
    _same_outputs(old, new)


def test_resolved_config_reproduces_the_run(runs):
    """Re-running resolved_config.yaml gives the same outputs, and the definition and its hash are recorded."""
    old, new, again = runs
    _same_outputs(old, again)
    text = (new / "out" / "resolved_definition.yaml").read_text()
    assert "hash:" in text and "id: moron_robertson_mok" in text
    config = (new / "out" / "resolved_config.yaml").read_text()
    assert "# region default (india)" in config and "definition: moron_robertson_mok" in config


def test_converted_flat_dict_matches_the_old_one(tmp_path):
    """The flat dict from the converted YAML equals the old config's, apart from the keys YAML adds."""
    (tmp_path / "c.in").write_text(OLD_CONFIG)
    old = convert.read_legacy(tmp_path / "c.in")
    mapping, _ = convert.flat_to_mapping(old)
    new = build_config(mapping)
    extra = {"dataset_report", "config_sources", "model_dims_list", "accumulation_list", "catalog_dirs",
             "dataset_files", "climatology_cv", "years", "years_clim", "fallback_date"}
    for key, value in old.items():
        assert new[key] == value, key
    assert set(new) - set(old) <= extra | {"plot_portrait", "save_onsets"}


def test_convert_maps_units_and_rejects_free_multipliers(tmp_path):
    """unit_cvt 1000 becomes units m; an odd multiplier is an error asking the user."""
    flat = convert.read_legacy(_write(tmp_path, OLD_CONFIG.replace("unit_cvt_list = (None,)", "unit_cvt_list = (1000,)")))
    mapping, _ = convert.flat_to_mapping(flat)
    assert mapping["models"][0]["units"] == "m" and mapping["observations"]["units"] == "mm"
    flat["unit_cvt_list"] = (25.4,)
    with pytest.raises(Exception, match="only mm or m"):
        convert.flat_to_mapping(flat)


def _write(tmp_path, text, name="c.in"):
    """Write text to tmp_path/name and return the path."""
    path = tmp_path / name
    path.write_text(text)
    return path


def test_convert_refuses_to_overwrite(tmp_path):
    """convert and init refuse an existing target unless --force."""
    (tmp_path / "c.in").write_text(OLD_CONFIG)
    (tmp_path / "run.yaml").write_text("x")
    assert _run(tmp_path, "convert", "c.in", "-o", "run.yaml").returncode == 1
    assert _run(tmp_path, "convert", "c.in", "-o", "run.yaml", "--force").returncode == 0


def test_top_level_help_works_without_a_config(tmp_path):
    """momp-run -h lists init, datasets, convert and run without needing a config."""
    done = _run(tmp_path, "-h")
    assert done.returncode == 0
    for word in ("init", "datasets", "convert", "run"):
        assert word in done.stdout


def test_old_config_in_is_rejected_by_run(tmp_path):
    """momp-run -p old.in stops and points at convert."""
    (tmp_path / "c.in").write_text(OLD_CONFIG)
    done = _run(tmp_path, "-p", "c.in")
    assert done.returncode == 1 and "momp-run convert" in done.stdout


@pytest.fixture
def synthetic(tmp_path):
    """Synthetic observation and ensemble forecast folders."""
    write_obs(tmp_path / "obs")
    write_forecast(tmp_path / "fc", members=3)
    return tmp_path


def test_init_from_paths_writes_a_valid_config_with_evidence(synthetic):
    """init --obs --model detects dataset values, writes evidence comments, and the file loads and validates."""
    done = _run(synthetic, "init", "--obs", "obs", "--model", "fc", "--region", "Ethiopia", "--years", "2020", "2020")
    assert done.returncode == 0, done.stderr
    text = (synthetic / "run.yaml").read_text()
    assert "[detected: only gridded variable in 2020.nc]" in text
    assert "members: 3" in text and "probabilistic: true" in text
    assert "definition: icpac" not in text and "start_year_clim" not in text and "day_boundary: " not in text.replace("# day_boundary", "")
    raw = yaml.safe_load(text)
    assert raw["region"] == "Ethiopia" and raw["models"][0]["lead_label"] in ("start", "end")
    checked = _run(synthetic, "datasets", "-p", "run.yaml")
    assert checked.returncode == 0, checked.stdout


def test_init_without_paths_writes_placeholders_and_refuses_to_overwrite(tmp_path):
    """Plain init writes TODO placeholders; a second init needs --force; the template prints a blank entry."""
    assert _run(tmp_path, "init").returncode == 0
    assert "/path/to/observations" in (tmp_path / "run.yaml").read_text()
    again = _run(tmp_path, "init")
    assert again.returncode == 1 and "--force" in again.stderr
    assert _run(tmp_path, "init", "--force").returncode == 0
    template = _run(tmp_path, "init", "--dataset-template", "forecast")
    assert "lead_label" in template.stdout


def test_datasets_command_reports_detections_and_problems(synthetic):
    """momp-run datasets -p lists what the config's datasets resolve to; a problem gives exit code 1."""
    run = {"region": "ethiopia", "observations": {"path": str(synthetic / "obs"), "day_boundary": "00 UTC, labelled by start"},
           "models": [{"name": "M", "path": str(synthetic / "fc")}],
           "evaluation": {"start_date": [2020, 5, 1], "end_date": [2020, 7, 31]}}
    (synthetic / "run.yaml").write_text(yaml.safe_dump(run))
    done = _run(synthetic, "datasets", "-p", "run.yaml")
    assert done.returncode == 0, done.stdout
    assert "[detected:" in done.stdout and 'lead_label: start' in done.stdout
    run["observations"]["variable"] = "nope"
    (synthetic / "run.yaml").write_text(yaml.safe_dump(run))
    bad = _run(synthetic, "datasets", "-p", "run.yaml")
    assert bad.returncode == 1 and "'nope' is not in" in bad.stdout


def test_write_resolved_records_overrides(tmp_path):
    """An onset override and dry_spell_check = false show up in the printed and recorded override list."""
    from momp.lib.resolved import definition_lines

    cfg = {"region": "ethiopia", "onset_legacy": False, "onset_definition": "icpac", "obs_day_boundary": None,
           "onset_override": {"wet_day": "at least 2 mm"}, "dry_spell_check": False, "dir_out": str(tmp_path),
           "config_sources": {}}
    lines = "\n".join(definition_lines(cfg))
    assert "overridden: wet_day, reject if (dry_spell_check = false)" in lines and "hash" in lines
    from momp.lib.onset_config import onset_setup

    setup = onset_setup(cfg)
    assert setup.definition.wet_day.__repr__().count("20") >= 1
