"""Configure logging for HCWF: one logger per module under ``hcwf``, a console handler, an optional run.log and progress.

Modules log with ``log = logging.getLogger(__name__)``; nothing is configured on import. A run calls
`session` (a context manager) which

* shows INFO (default), only warnings (``verbosity=-1``, ``--quiet``) or DEBUG (``verbosity=1``, ``--verbose``)
  on the console, without time stamps, writing to whatever ``sys.stdout`` is at that moment;
* writes everything (DEBUG up, with time stamps and module names) to ``logfile`` when one is given;
* turns Python warnings into log records (``IngestInfo`` notes as INFO, the rest as warnings);
* restores the previous logging setup on exit.

Worker processes (see ``hcwf.parallel``) forward their log records and warnings to the process that owns the
session through a queue (`forwarding` / `worker_logging`), so they reach run.log and the console like any other
record, with the worker's year in the message.

`progress` shows a tqdm bar when tqdm is installed and stderr is a terminal, and otherwise logs
``desc: i/n`` lines (nothing at all when quiet).
"""
from __future__ import annotations

import contextlib
import logging
import logging.handlers
import multiprocessing
import sys
import time
import warnings
from collections.abc import Iterator
from importlib.metadata import version as pkg_version
from pathlib import Path

ROOT = "hcwf"
QUIET, NORMAL, VERBOSE = -1, 0, 1
_LEVELS = {QUIET: logging.WARNING, NORMAL: logging.INFO, VERBOSE: logging.DEBUG}
_state = {"console_level": logging.INFO}


class _Console(logging.Handler):
    """A handler that writes plain messages to the current ``sys.stdout``; warnings and errors get their level as prefix."""

    def emit(self, record):
        """Write one record to stdout.

        Parameters
        ----------
        record : logging.LogRecord
            The record to show; the prefix ``error:`` or ``warning:`` is put in front of ERROR and WARNING records.
        """
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


def log_warning(message, category, filename=None, lineno=None, file=None, line=None):
    """Log a Python warning as a ``hcwf.warnings`` record (ingest notes as INFO, the rest as warnings).

    It has the signature of ``warnings.showwarning`` and replaces it while a session or a worker is active.

    Parameters
    ----------
    message : Warning or str
        The warning; its text becomes the log message.
    category : type
        The warning class; subclasses of ``hcwf.onset.ingest.IngestInfo`` are logged at INFO, others at WARNING.
    filename : str, optional
        Unused; present for the ``warnings.showwarning`` signature.
    lineno : int, optional
        Unused; present for the ``warnings.showwarning`` signature.
    file : file object, optional
        Unused; present for the ``warnings.showwarning`` signature.
    line : str, optional
        Unused; present for the ``warnings.showwarning`` signature.
    """
    from hcwf.onset.ingest import IngestInfo

    logging.getLogger(f"{ROOT}.warnings").log(
        logging.INFO if issubclass(category, IngestInfo) else logging.WARNING, "%s", message)


_worker = {"year": None}


class _YearFilter(logging.Filter):
    """A filter that puts the year the worker is processing in front of each record's message (and in ``record.year``)."""

    def filter(self, record):
        """Prefix the message with the worker's year, if it has one.

        Parameters
        ----------
        record : logging.LogRecord
            The record to change in place.

        Returns
        -------
        bool
            Always True: no record is dropped.
        """
        year = _worker["year"]
        if year is not None:
            record.year = year
            record.msg = f"[year {year}] {record.getMessage()}"
            record.args = None
        return True


def worker_logging(queue) -> None:
    """Initialize a worker process: send every ``hcwf`` record and Python warning to the parent through `queue`.

    Parameters
    ----------
    queue : multiprocessing.Queue
        The queue made by `forwarding` in the parent process.
    """
    root = logging.getLogger(ROOT)
    handler = logging.handlers.QueueHandler(queue)
    handler.addFilter(_YearFilter())
    root.handlers = [handler]
    root.setLevel(logging.DEBUG)
    root.propagate = False
    warnings.showwarning = log_warning


def set_worker_year(year) -> None:
    """Name the year this worker is processing, for the messages it forwards.

    Parameters
    ----------
    year : int or None
        The year; None clears it.
    """
    _worker["year"] = year


@contextlib.contextmanager
def forwarding() -> Iterator:
    """Yield a queue whose records go to the handlers of the current session.

    A listener thread passes what workers put on the queue to the ``hcwf`` logger's handlers (run.log and console,
    each at its own level). It is stopped on exit, error or not; leave the block after the workers have finished.

    Yields
    ------
    multiprocessing.Queue or None
        The queue for `worker_logging`, or None when there are no handlers to feed.
    """
    handlers = list(logging.getLogger(ROOT).handlers)
    if not handlers:
        yield None
        return
    queue = multiprocessing.get_context("spawn").Queue()
    listener = logging.handlers.QueueListener(queue, *handlers, respect_handler_level=True)
    listener.start()
    try:
        yield queue
    finally:
        listener.stop()
        queue.close()
        queue.join_thread()


@contextlib.contextmanager
def session(verbosity: int = NORMAL, logfile=None) -> Iterator[logging.Logger]:
    """Configure HCWF's logging for the duration of a run (see the module text).

    The previous logging setup (handlers, level, ``warnings.showwarning``) is restored on exit.

    Parameters
    ----------
    verbosity : int, default NORMAL
        ``QUIET`` (warnings only), ``NORMAL`` (INFO) or ``VERBOSE`` (DEBUG) on the console.
    logfile : str or pathlib.Path, optional
        A file that receives everything (DEBUG up, with time stamps); its folder is created. None writes no file.

    Yields
    ------
    logging.Logger
        The ``hcwf`` logger.
    """
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

    warnings.showwarning = log_warning
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
    """Return the verbosity number for the ``--quiet`` / ``--verbose`` flags (quiet wins).

    Parameters
    ----------
    quiet : bool, default False
        The ``--quiet`` flag.
    verbose : bool, default False
        The ``--verbose`` flag.

    Returns
    -------
    int
        ``QUIET``, ``VERBOSE`` or ``NORMAL``.
    """
    return QUIET if quiet else (VERBOSE if verbose else NORMAL)


def progress(iterable, desc: str = "", total: int | None = None) -> Iterator:
    """Iterate over ``iterable``, showing progress: a tqdm bar, else ``desc: i/n`` log lines (none when quiet).

    tqdm is used only when it is installed, stderr is a terminal and the console shows INFO; it is never required.

    Parameters
    ----------
    iterable : iterable
        The items to pass through.
    desc : str, default ""
        The label of the bar or of each log line.
    total : int, optional
        The number of items; taken from ``len(iterable)`` when None and it has one.

    Yields
    ------
    object
        The items of ``iterable``, one at a time.
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


def print_hcwf_banner(project_name):
    """Log the HCWF start-up banner with package version, project name and start time.

    Parameters
    ----------
    project_name : str
        The project name of the run config.
    """
    version = pkg_version("hcwf")

    banner = fr"""
================================================================================
  _   _   ____  __        __  _____
 | | | | / ___| \ \      / / |  ___|
 | |_| || |      \ \ /\ / /  | |_
 |  _  || |___    \ V  V /   |  _|
 |_| |_| \____|    \_/\_/    |_|

 Human-Centered Weather Forecasts (HCWF)
 Version : {version}

--------------------------------------------------------------------------------
 Project    : {project_name}
 Start Time : {__import__('datetime').datetime.now().isoformat(timespec='seconds')}
--------------------------------------------------------------------------------

 Initializing analysis pipeline...
================================================================================
"""
    logging.getLogger(__name__).info(banner)
