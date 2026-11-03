"""A YAML run config and its resolved_config.yaml give the same run; init, datasets, convert and resolved_config work."""

import filecmp
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from hcwf.lib import convert
from hcwf.lib.config_yaml import build_config
from hcwf.lib.resolved import write_resolved
from tests.synthetic_data import write_forecast, write_obs

ROOT = Path(__file__).resolve().parents[1]
# The flat config of the demo run, as the old config.in held it (now a dict: config.in files are no longer read).
FLAT = {
    'project_name': 'tiny',
    'model_list': ('AIFS',),
    'obs': 'IMD',
    'obs_dir': f'{ROOT}/demo/data/obs',
    'obs_file_pattern': ('{}.nc',),
    'obs_var': 'RAINFALL',
    'obs_unit_cvt': None,
    'ref_model': 'climatology',
    'ref_model_dir': f'{ROOT}/demo/data/obs',
    'ref_model_file_pattern': '{}.nc',
    'ref_model_var': 'RAINFALL',
    'ref_model_unit_cvt': None,
    'model_dir_list': (f'{ROOT}/demo/data/aifs',),
    'model_var_list': ('tp',),
    'unit_cvt_list': (None,),
    'file_pattern_list': ('{}.nc',),
    'region': 'india',
    'nc_mask': None,
    'shpfile_dir': None,
    'polygon': False,
    'onset_definition': 'moron_robertson_mok',
    'onset_season': None,
    'threshold_years': (2013, 2020),
    'thresholds': None,
    'obs_day_boundary': None,
    'model_lead_label': None,
    'check_alignment': False,
    'dry_spell_check': False,
    'forecast_history': 'none',
    'threshold_source': 'observations',
    'onset_backend': 'reference',
    'onset_percentage_threshold': 0.5,
    'explain_cells': None,
    'start_date': (2015, 5, 1),
    'end_date': (2015, 6, 30),
    'start_year_clim': 2013,
    'end_year_clim': 2020,
    'init_days': (2, 5),
    'date_filter_year': 2015,
    'verification_window_list': ((1, 15),),
    'tolerance_days_list': (3,),
    'max_forecast_day': 15,
    'day_bins': ((1, 5), (6, 10), (11, 15)),
    'FAR': True,
    'MAE': True,
    'MR': True,
    'probabilistic': False,
    'members': 'All',
    'BS': True,
    'RPS': True,
    'AUC': True,
    'Reliability': True,
    'skill_score': True,
    'dir_out': 'out',
    'dir_fig': 'fig',
    'save_fig': False,
    'save_nc_spatial_far_mr_mae': True,
    'save_csv_score': True,
    'save_nc_climatology': True,
    'plot_spatial_far_mr_mae': False,
    'plot_heatmap_bss_auc': False,
    'plot_reliability': False,
    'plot_climatology_onset': False,
    'plot_panel_heatmap_error': False,
    'plot_panel_heatmap_skill': False,
    'plot_bar_bss_rpss_auc': False,
    'show_plot': False,
    'show_panel': False,
    'parallel': False,
}
def _run(cwd, *args, code=None):
    """Run hcwf in a subprocess in cwd, returning the completed process."""
    cmd = [sys.executable, "-c", code] if code else [sys.executable, "-m", "hcwf.cli", *args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})


def _same_outputs(a: Path, b: Path):
    """Every NetCDF output of a exists in b with identical bytes."""
    files = sorted(p.name for p in (a / "out").glob("*.nc"))
    assert files
    for name in files:
        assert filecmp.cmp(a / "out" / name, b / "out" / name, shallow=False), name


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Run the demo once from a YAML config, once from the resolved config that run wrote."""
    base = tmp_path_factory.mktemp("runs")
    new, again = base / "new", base / "again"
    for d in (new, again):
        d.mkdir()
    mapping, _ = convert.flat_to_mapping(FLAT)
    (new / "run.yaml").write_text(convert.render_yaml(mapping))
    done = _run(new, "-p", "run.yaml")
    assert done.returncode == 0, done.stderr[-2000:]
    resolved = yaml.safe_load((new / "out" / "resolved_config.yaml").read_text())
    resolved["output"].update(data_dir="out", figure_dir="fig")
    (again / "r.yaml").write_text(yaml.safe_dump(resolved))
    done = _run(again, "-p", "r.yaml")
    assert done.returncode == 0, done.stderr[-2000:]
    return new, again


def test_resolved_config_reproduces_the_run(runs):
    """Re-running resolved_config.yaml gives the same outputs, and the definition and its hash are recorded."""
    new, again = runs
    _same_outputs(new, again)
    text = (new / "out" / "resolved_definition.yaml").read_text()
    assert "hash:" in text and "id: moron_robertson_mok" in text
    config = (new / "out" / "resolved_config.yaml").read_text()
    assert "# region default (india)" in config and "definition: moron_robertson_mok" in config


def test_the_flat_dict_survives_the_yaml_round_trip():
    """The flat dict written as YAML sections and built again equals the original, apart from the keys YAML adds."""
    old = FLAT
    mapping, _ = convert.flat_to_mapping(old)
    new = build_config(mapping)
    extra = {"dataset_report", "config_sources", "model_dims_list", "accumulation_list", "catalog_dirs",
             "dataset_files", "climatology_cv", "daily_boundary_list", "accumulation_detected"}
    for key, value in old.items():
        assert new[key] == value, key
    assert set(new) - set(old) <= extra | {"plot_portrait", "save_onsets"}


def test_convert_maps_units_and_rejects_free_multipliers():
    """unit_cvt 1000 becomes units m; an odd multiplier is an error asking the user."""
    flat = {**FLAT, "unit_cvt_list": (1000,)}
    mapping, _ = convert.flat_to_mapping(flat)
    assert mapping["models"][0]["units"] == "m" and mapping["observations"]["units"] == "mm"
    flat["unit_cvt_list"] = (25.4,)
    with pytest.raises(Exception, match="only mm or m"):
        convert.flat_to_mapping(flat)


def test_convert_refuses_to_overwrite(tmp_path):
    """convert refuses an existing target unless --force."""
    flat = {"wet_init": 1, "wet_threshold": 20, "wet_spell": 3, "dry_spell": 7, "dry_extent": 0,
            "region": "ethiopia", "evaluation": {"start_date": [2020, 5, 1], "end_date": [2020, 7, 31]}}
    (tmp_path / "old.yaml").write_text(yaml.safe_dump(flat))
    (tmp_path / "run.yaml").write_text("x")
    assert _run(tmp_path, "convert", "old.yaml", "-o", "run.yaml").returncode == 1
    assert _run(tmp_path, "convert", "old.yaml", "-o", "run.yaml", "--force").returncode == 0
    assert "rule" in yaml.safe_load((tmp_path / "run.yaml").read_text())["onset"]


def test_top_level_help_works_without_a_config(tmp_path):
    """hcwf-run -h lists init, datasets, convert and run without needing a config."""
    done = _run(tmp_path, "-h")
    assert done.returncode == 0
    for word in ("init", "datasets", "convert", "run"):
        assert word in done.stdout


def test_old_config_in_is_rejected_by_run(tmp_path):
    """hcwf-run -p old.in stops: only YAML configs are read."""
    (tmp_path / "c.in").write_text("model_list = ('A',)\n")
    done = _run(tmp_path, "-p", "c.in")
    assert done.returncode == 1 and "reads YAML only" in done.stdout


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
    """hcwf-run datasets -p lists what the config's datasets resolve to; a problem gives exit code 1."""
    run = {"region": "ethiopia", "observations": {"path": str(synthetic / "obs"), "day_boundary": "00 UTC, labelled by start"},
           "models": [{"name": "M", "path": str(synthetic / "fc")}],
           "evaluation": {"start_date": [2020, 5, 1], "end_date": [2020, 7, 31]},
           "onset": {"check_alignment": True}}
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
    from hcwf.lib.onset_config import onset_setup
    from hcwf.lib.resolved import definition_lines

    cfg = {"region": "ethiopia", "onset_definition": "icpac", "obs_day_boundary": None,
           "onset_override": {"wet_day": "at least 2 mm"}, "dry_spell_check": False, "dir_out": str(tmp_path),
           "config_sources": {}}
    lines = "\n".join(definition_lines(onset_setup(cfg)))
    assert "overridden: wet_day, reject if (dry_spell_check = false)" in lines and "hash" in lines
    setup = onset_setup(cfg)
    assert setup.definition.wet_day.__repr__().count("20") >= 1
