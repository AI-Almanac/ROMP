"""Importing the config and the command line must not load the plotting, machine-learning or JIT stacks."""
import subprocess
import sys

import pytest

HEAVY = ("matplotlib", "cartopy", "sklearn", "numba", "scipy.stats")


@pytest.mark.parametrize("module", ["hcwf.config", "hcwf.cli", "hcwf.parallel", "hcwf.scoring.probabilistic"])
def test_light_modules_do_not_import_heavy_stacks(module):
    heavy = HEAVY if module != "hcwf.scoring.probabilistic" else ("matplotlib", "cartopy", "sklearn")
    code = (f"import sys, {module}\n"
            f"print(sorted(m for m in {heavy!r} if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", f"import {module} loaded {out.strip()}"
