"""Logging for ROMP: one logger per module under ``momp``, a console handler, an optional run.log and progress.

Modules log with ``log = logging.getLogger(__name__)``; nothing is configured on import. A run calls
``session(verbosity, logfile)`` (a context manager) which

* shows INFO (default), only warnings (``verbosity=-1``, ``--quiet``) or DEBUG (``verbosity=1``, ``--verbose``)
  on the console, without time stamps, writing to whatever ``sys.stdout`` is at that moment;
* writes everything (DEBUG up, with time stamps and module names) to ``logfile`` when one is given;
* turns Python warnings into log records (``IngestInfo`` notes as INFO, the rest as warnings);
* restores the previous logging setup on exit.

``progress(iterable, desc)`` shows a tqdm bar when tqdm is installed and stderr is a terminal, and otherwise logs
``desc: i/n`` lines (nothing at all when quiet).
"""
from __future__ import annotations

import contextlib
import logging
import sys
import time
import warnings
from pathlib import Path

ROOT = "momp"
QUIET, NORMAL, VERBOSE = -1, 0, 1
_LEVELS = {QUIET: logging.WARNING, NORMAL: logging.INFO, VERBOSE: logging.DEBUG}
_state = {"console_level": logging.INFO}


class _Console(logging.Handler):
    """Writes plain messages to the current sys.stdout; warnings and errors are prefixed with their level."""

    def emit(self, record):
        """Write one record to stdout."""
        try:
            text = record.getMessage()
            if record.levelno >= logging.ERROR:
                text = f"error: {text}"
            elif record.levelno >= logging.WARNING:
                text = f"warning: {text}"
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 - a logging failure must never stop a run
            self.handleError(record)


@contextlib.contextmanager
def session(verbosity: int = NORMAL, logfile=None):
    """Configure ROMP's logging for the duration of a run (see the module text); yields the ``momp`` logger."""
    root = logging.getLogger(ROOT)
    previous = (root.level, list(root.handlers), root.propagate, warnings.showwarning, _state["console_level"])
    level = _LEVELS.get(verbosity, logging.INFO)
    root.setLevel(logging.DEBUG)
    root.propagate = False
    console = _Console()
    console.setLevel(level)
    root.handlers = [console]
    _state["console_level"] = level
    file_handler = None
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                                    "%Y-%m-%d %H:%M:%S"))
        root.addHandler(file_handler)

    def show(message, category, filename, lineno, file=None, line=None):
        """Log a Python warning (ingest notes as INFO)."""
        from momp.onset.ingest import IngestInfo

        logging.getLogger(f"{ROOT}.warnings").log(
            logging.INFO if issubclass(category, IngestInfo) else logging.WARNING, "%s", message)

    warnings.showwarning = show
    try:
        yield root
    finally:
        warnings.showwarning = previous[3]
        if file_handler is not None:
            file_handler.close()
        root.handlers = previous[1]
        root.setLevel(previous[0])
        root.propagate = previous[2]
        _state["console_level"] = previous[4]


def verbosity_from(quiet: bool = False, verbose: bool = False) -> int:
    """The verbosity number for the --quiet / --verbose flags (quiet wins)."""
    return QUIET if quiet else (VERBOSE if verbose else NORMAL)


def progress(iterable, desc: str = "", total: int | None = None):
    """Iterate over iterable, showing progress: a tqdm bar, else ``desc: i/n`` log lines (none when quiet).

    tqdm is used only when it is installed, stderr is a terminal and the console shows INFO; it is never required.
    """
    items = iterable
    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)
    log = logging.getLogger(f"{ROOT}.progress")
    if _state["console_level"] <= logging.INFO and sys.stderr.isatty():
        try:
            from tqdm import tqdm

            yield from tqdm(items, desc=desc, total=total, leave=False)
            return
        except ImportError:
            pass
    start = time.perf_counter()
    for i, item in enumerate(items, 1):
        yield item
        log.info("%s: %d/%s (%.0f s)", desc, i, total if total is not None else "?", time.perf_counter() - start)
