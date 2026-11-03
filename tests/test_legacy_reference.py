"""End to end: a config with the old flat onset keys, converted by today's `hcwf-run convert` and run, gives the outputs
the old flat-key mode (commit 1ffc0cb^) wrote for it. The references are in tests/data/legacy_reference/; the data and
configs are tests/legacy_reference.py. Floats agree to rtol 1e-9, everything else exactly."""
import json
from pathlib import Path

import pytest
import yaml

from hcwf.cli import main as cli_main
from hcwf.lib.convert import convert_flat_onset
from tests import legacy_reference as ref

REFERENCE = Path(__file__).parent / "data" / "legacy_reference"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """The synthetic observations, forecasts and threshold grid, written once."""
    root = tmp_path_factory.mktemp("legacy_data")
    return ref.write_dataset(root / "data")


@pytest.mark.parametrize("name", list(ref.CASES))
def test_a_converted_flat_config_reproduces_the_old_outputs(name, dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("HCWF_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    out = tmp_path / "out"
    old = tmp_path / "old.yaml"
    old.write_text(yaml.safe_dump(ref.flat_config(name, dataset, out), sort_keys=False))
    new = convert_flat_onset(old, tmp_path / "new.yaml")
    cli_main(["-p", str(new), "--quiet", "--overwrite"])
    expected = json.loads((REFERENCE / f"{name}.json").read_text())
    assert ref.same(ref.condense(out), expected, name) == []


def test_the_comparison_notices_a_difference():
    """The comparison is not vacuous: a value changed in the last digits it checks, a missing file, a column."""
    run = {"a.nc": {"v": {"dims": ["x"], "dtype": "float64", "values": [1.0, None]}},
           "b.csv": {"columns": ["c"], "rows": [[1.5], [2]]}}
    assert ref.same(run, json.loads(json.dumps(run)), "x") == []
    close = json.loads(json.dumps(run))
    close["a.nc"]["v"]["values"][0] = 1.0 + 1e-12
    assert ref.same(close, run, "x") == []
    for change in (lambda r: r["a.nc"]["v"]["values"].__setitem__(0, 1.0 + 1e-6),
                   lambda r: r["a.nc"]["v"]["values"].__setitem__(1, 0.0),
                   lambda r: r["b.csv"]["rows"][1].__setitem__(0, 3),
                   lambda r: r.pop("b.csv")):
        other = json.loads(json.dumps(run))
        change(other)
        assert ref.same(other, run, "x")
