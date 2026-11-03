"""`python -m hcwf.onset ...` is `hcwf-run onset ...`."""
import sys

from hcwf.cli import main

main(["onset", *sys.argv[1:]])
