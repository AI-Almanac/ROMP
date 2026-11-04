"""Importing a command-line module must not parse argv or run a command (Sphinx imports every module)."""
import subprocess
import sys

import pytest

MODULES = ["hcwf.cli", "hcwf.cli.onset", "hcwf.cli.driver", "hcwf.cli.parser", "hcwf.cli.__main__", "hcwf.onset.__main__"]


@pytest.mark.parametrize("module", MODULES)
def test_import_with_foreign_argv_does_nothing(module):
    code = f"import sys; sys.argv = ['x', 'html', 'docs']; import {module}; print('imported')"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "imported"


def test_python_m_hcwf_onset_still_runs_the_onset_commands():
    done = subprocess.run([sys.executable, "-m", "hcwf.onset", "regions"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip()
