"""One command line (`hcwf-run onset ...` included) and one config error hierarchy."""
import subprocess
import sys

import pytest

from hcwf.lib.assertion import HCWFConfigError


def test_every_config_error_is_an_hcwf_config_error():
    from hcwf.blend.recipe import RecipeError
    from hcwf.blend.trained import BlendFileError
    from hcwf.lib.config_schema import ConfigSchemaError
    from hcwf.onset.definition import DefinitionError
    from hcwf.onset.regions import RegionError
    from hcwf.onset.units import UnitsError
    from hcwf.spatial.units import AreaError, UnitsSpecError
    from hcwf.utils.mask import MaskError

    assert issubclass(HCWFConfigError, ValueError)
    for cls in (RecipeError, BlendFileError, ConfigSchemaError, DefinitionError, RegionError, UnitsError,
                AreaError, UnitsSpecError, MaskError):
        assert issubclass(cls, HCWFConfigError), cls


def _run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True)


@pytest.mark.parametrize("args", [("-m", "hcwf.cli", "onset", "list"), ("-m", "hcwf.onset", "list")])
def test_onset_list_from_both_entry_points(args):
    out = _run(*args)
    assert out.returncode == 0 and "icpac" in out.stdout


def test_onset_error_is_a_clean_message():
    out = _run("-m", "hcwf.cli", "onset", "describe", "no_such_definition")
    assert out.returncode == 1 and "Traceback" not in out.stderr and out.stderr.startswith("error:")


def test_cli_does_not_import_driver_until_it_runs():
    code = "import sys, hcwf.cli; sys.exit('hcwf.driver' in sys.modules)"
    assert _run("-c", code).returncode == 0
    assert "hcwf.cli" not in open(__import__("hcwf.driver", fromlist=["x"]).__file__).read().split("if __name__")[0]
