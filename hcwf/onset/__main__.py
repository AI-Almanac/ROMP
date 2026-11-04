"""Run the onset command line: ``python -m hcwf.onset ...`` is ``hcwf-run onset ...``."""
import sys

from hcwf.cli import main

if __name__ == "__main__":   # not on import, so that Sphinx autosummary can import every module
    main(["onset", *sys.argv[1:]])
