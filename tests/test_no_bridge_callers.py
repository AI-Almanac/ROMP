"""After step S10 of docs/review/runconfig_design.md, only the `# R2-bridge` wrappers (and tests) use the flat bridge:
every other caller reads inputs, detects and scores through the typed API."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "hcwf"

WRAPPER_MODULES = {"io/input.py", "stats/detect.py", "stats/climatology.py", "stats/conditional_climatology.py",
                   "stats/observed_cache.py", "stats/bins.py", "stats/benchmark.py", "metrics/skill.py",
                   "config/flat_bridge.py"}
"""The modules that hold R2-bridge wrappers (step S11 deletes them with the bridge)."""
WRAPPERS = {"load_imd_rainfall", "load_history", "get_forecast_deterministic_twice_weekly",
            "get_forecast_probabilistic_twice_weekly", "get_initialization_dates", "onset_steps_needed",
            "ingest_options", "ingest_model_forecast", "detect_observed_result",
            "detect_observed_onset", "_detect_forecast", "member_onset_counts", "compute_onset_for_deterministic_model",
            "compute_onset_for_all_members", "check_alignment_once", "compute_climatological_onset",
            "compute_climatological_onset_dataset", "climatology_cv_mode", "create_forecast_observation_pairs_with_bins",
            "multi_year_forecast_obs_pairs", "multi_year_climatological_forecast_obs_pairs",
            "multi_year_conditional_climatological_forecast_obs_pairs", "compute_metrics_multiple_years",
            "compute_onset_metrics_with_windows", "prepare_score_cache", "obs_files", "restore_args", "case_kwargs",
            "model_kwargs", "reader_inputs", "from_case_kwargs", "resolve_flat",
            "resolve_cli"}
"""The names of the wrappers (and bridge functions) of the flat forms."""


def _names(tree):
    """Every name the module imports, reads or calls, and every attribute it reads."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.ImportFrom):
            yield from (alias.name for alias in node.names)
            if node.module:
                yield node.module.rsplit(".", 1)[-1]
        elif isinstance(node, ast.Import):
            yield from (alias.name.rsplit(".", 1)[-1] for alias in node.names)


def _modules():
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel not in WRAPPER_MODULES and rel not in ("config/build.py", "config/model.py"):
            yield rel, path


@pytest.mark.parametrize("rel,path", list(_modules()), ids=lambda value: value if isinstance(value, str) else "")
def test_only_wrapper_modules_use_the_bridge_or_a_wrapper(rel, path):
    used = set(_names(ast.parse(path.read_text())))
    assert "flat_bridge" not in used, f"{rel} imports the flat bridge"
    # a wrapper's name may be a method or a local of an unrelated class (resolve_flat belongs to config.build, whose
    # wrappers are for the tests); the callers of the typed API use none of them
    assert not (used & WRAPPERS), f"{rel} uses {sorted(used & WRAPPERS)}"
