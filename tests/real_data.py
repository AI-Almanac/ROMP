"""Where the optional real-data checks find their files (ethiopia/ and india/ obs and models), defined once.

``HCWF_TEST_DATA`` if set, else the ai-almanac datasets folder when it exists (only ever read), else the checks skip.
"""
import os
from pathlib import Path

import pytest

_DEFAULT = Path("/home/hcwf/code/ai-almanac/datasets")
DATA = Path(os.environ.get("HCWF_TEST_DATA") or _DEFAULT)

needs_data = pytest.mark.skipif(
    not DATA.exists(), reason=f"no real data: set HCWF_TEST_DATA (default {_DEFAULT}) to run real-data checks")
