"""The fixture world of the config tests: synthetic observations and forecasts, and the run configs that point at them.

`world` and `configs` are module-scoped fixtures (import them into a test module); `build_world` and `fixture_configs`
are the plain builders. When the four regression configs are present they are added to `configs`.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_run_experience import MAX_LEAD, _write_ensemble, raw_config, write_config, write_forecast, write_obs  # noqa: E402

REGRESSION_CFG = Path("/tmp/claude-1000/-home-hcwf-code-ROMP/f3fbeea6-493c-485b-bfb6-ad02edb09411/scratchpad/out_base/cfg")


def build_world(base: Path) -> Path:
    """Observations, DET, DETC (accumulated) and ENS forecasts under base; returns base."""
    write_obs(base / "obs")
    write_forecast(base / "DET")
    write_forecast(base / "DETC", cumulative=True)
    _write_ensemble(base / "ENS")
    return base


def fixture_configs(base: Path, folder: Path) -> dict[str, Path]:
    """{name: YAML path} of the fixture run configs (written into folder, pointing at the data in base)."""
    folder.mkdir(parents=True, exist_ok=True)
    two = {"verification_windows": [[1, MAX_LEAD], [1, 5]], "tolerances": [3, 2]}
    raws = {
        "det_climatology": raw_config(base, ("DET",)),
        "two_models_windows": raw_config(base, ("DET", "DETC"), evaluation=two),
        "model_reference": raw_config(base, ("DET", "DETC"), evaluation=two,
                                      reference={"model": "DETC", "path": str(base / "DETC"), "variable": "tp",
                                                 "units": "mm"}),
        "ensemble": raw_config(base, ("ENS",), evaluation={"probabilistic": True}),
    }
    return {name: write_config(folder, raw, f"{name}.yaml") for name, raw in raws.items()}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("oracle_world"))


@pytest.fixture(scope="module")
def configs(world, tmp_path_factory):
    out = fixture_configs(world, tmp_path_factory.mktemp("oracle_cfgs"))
    if REGRESSION_CFG.is_dir():
        out.update({p.stem: p for p in sorted(REGRESSION_CFG.glob("*.yaml"))})
    return out


def _norm(value):
    """value without the " (cached)" the dataset detection adds to its notes the second time it reads the same files."""
    if isinstance(value, str):
        return value.replace(" (cached)", "")
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_norm(v) for v in value)
    return value


def _same(a, b):
    """Resolved flat dicts equal (run_paths compared by their fields; they carry the fingerprint)."""
    assert a.keys() == b.keys()
    for key in a:
        assert _norm(a[key]) == _norm(b[key]), key
