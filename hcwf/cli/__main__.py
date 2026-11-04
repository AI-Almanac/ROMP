"""`python -m hcwf.cli ...` is `hcwf-run ...`."""
from hcwf.cli import main

if __name__ == "__main__":   # not on import, so that Sphinx autosummary can import every module
    main()
