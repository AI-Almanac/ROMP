"""Write the reference outputs of the old flat-key mode (tests/legacy_reference.py) into this folder.

Run it with the code of commit 1ffc0cb^ first on the path, from a scratch folder:

    git worktree add --detach <scratch>/wt 1ffc0cb^
    PYTHONPATH=<scratch>/wt python <repo>/tests/data/legacy_reference/make_reference.py <scratch>/work

Every case runs in legacy mode; its NetCDF and CSV outputs are condensed into <case>.json next to this script (values
as they were written). Nothing else is kept.
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def main(work: Path) -> None:
    from hcwf.cli import main as run   # the old code, first on PYTHONPATH

    spec = importlib.util.spec_from_file_location("legacy_reference", HERE.parents[1] / "legacy_reference.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    os.environ["HCWF_CACHE_DIR"] = str(work / "cache")
    os.environ["MPLBACKEND"] = "Agg"
    paths = ref.write_dataset(work / "data")
    for name in ref.CASES:
        out = work / "out" / name
        config = work / f"{name}.yaml"
        config.write_text(yaml.safe_dump(ref.flat_config(name, paths, out), sort_keys=False))
        run(["-p", str(config), "--quiet", "--overwrite"])
        (HERE / f"{name}.json").write_text(json.dumps(ref.condense(out), separators=(",", ":")))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()))
