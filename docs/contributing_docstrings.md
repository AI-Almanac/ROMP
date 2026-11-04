# Docstring conventions

Every module, class, function and method of `hcwf` has a NumPy-style docstring that Sphinx renders (napoleon). This is
true of private (`_name`) objects too. The conventions are checked by ruff, pydoclint and a guard test, so a
docstring that follows this page passes.

The top-level modules (`hcwf/errors.py`, `hcwf/log.py`, `hcwf/parallel.py`, `hcwf/cache.py`, `hcwf/api.py`, ...) are
the reference: copy their style.

## Rules

* **Style**: NumPy (`Parameters`, `Returns`, `Yields`, `Raises`, and `Attributes`, `Notes`, `Examples` where useful).
  Sections use the order `Parameters`, `Returns` / `Yields`, `Raises`, `See Also`, `Notes`, `Examples`.
* **Summary line**: one line, imperative for functions and methods ("Return the thresholds ...", "Compute ...",
  "Write ..."), ending with a period, then a blank line. Classes, modules and properties take a noun phrase ("The
  folder of ...", "A cache of ..."). A summary line that does not fit in 120 characters is too long: shorten it and
  move the rest to the extended description.
* **Extended description**: the existing explanatory prose is domain knowledge. Keep it, put it after the summary
  line and convert Markdown to reST: single-backtick literals become double backticks, `*` lists are fine, no
  Markdown headings or links, no tables in Markdown syntax.
* **Re-raises**: pydoclint reads a bare ``raise`` inside ``except X:`` as raising `X`; list `X` under `Raises` (for
  ``except BaseException:`` cleanup blocks, `BaseException`: "whatever the wrapped call raised").
* **All objects**: public and private module-level functions, classes and methods get the full sections. Nested
  functions (defined inside a function) need a summary line; add sections only when they are not trivial. `__init__`
  has no docstring of its own: the class docstring carries its `Parameters` (pydoclint rejects an `__init__`
  docstring). Property setters and deleters need none. Other dunder methods (`__repr__`, `__len__`, ...) do.
* **Modules**: a summary line, then what the module holds and how it relates to the other modules (keep the existing
  text, converted to reST). Module-level constants and dataclass fields are documented with an attribute docstring
  (a string right under the assignment).
* **Dataclass and NamedTuple fields**: document each field with an attribute docstring under it, not in an
  `Attributes` section. Instance attributes set in a plain `__init__` go in an `Attributes` section of the class
  docstring (not enforced).
* **Annotations**: optional, except that a generator or `@contextmanager` function declares `-> Iterator[...]` (pydoclint
  needs it for `Yields`). Docstring types are what Sphinx renders (`autodoc_typehints = "none"`), because they carry
  shape, dtype and meaning that annotations cannot. Keep the annotations that are already there; do not add or
  remove any for the docstring's sake. pydoclint does not compare the two.

## Types in the type line

Every parameter, return and yield has a type line. Write what the object is, in numpydoc style:

| Case | Write |
|---|---|
| Optional argument, default `None` | `x : int, optional` (say in the description what `None` means) |
| Argument with a default | `x : int, default 3` (strings: `default "mm"`) |
| Choice of values | `how : {"mean", "max"}, default "mean"` |
| Several types | `str or pathlib.Path` |
| Containers | `list of str`, `dict of str to float`, `tuple of (int, int)` |
| Arrays | state dtype and shape: ``numpy.ndarray of float64, shape (case, bin)`` |
| Datasets | `xarray.Dataset` with the variables and dims in the description |
| Callables | `callable` (say what it is called with) |
| `*args`, `**kwargs` | `*args : tuple`, `**kwargs : dict` and what they are passed on to |

Shape letters are the dimension names of the data (`year`, `lat`, `lon`, `member`, `lead`), as in the code.

## Referring to things

* An object of HCWF or of a library: single backticks, the role is `py:obj` (`default_role` of the build):
  `hcwf.api.run`, `~hcwf.api.run` (the `~` shows only `run`), `numpy.ndarray`, `pandas.DataFrame`. In a module, a
  name defined in the same module needs no prefix: `part_hash`; a method is `Store.get`.
* A literal (code, a value, a file name, a command, an environment variable): double backticks, ``parallel: false``,
  ``"mm"``, ``hcwf-run -p run.yaml``, ``$HCWF_CACHE_DIR``.
* A config key: double backticks with its section, for example ``evaluation.day_bins`` or ``units.mode``. These are not
  Python objects, so they are never single-backticked.
* Literal blocks (examples, trees): end the paragraph before with `::` and indent the block. A `>>>` doctest block is
  fine but is not run.
* An object that Sphinx cannot resolve is a plain-text mistake: check the spelling, or use double backticks if it is
  not an object.

## Examples

### A function

```python
def pair_years(obs, fcst, years, *, window=(1, 15), skip_missing=True):
    """Pair each observed onset with the forecast onset of the same year.

    A year without a forecast file is skipped when `skip_missing` is true. The pairing keeps the member axis, so
    ensemble scores can be computed from the result.

    Parameters
    ----------
    obs : xarray.DataArray
        Observed onset day of year, ``float64`` with dims ``(year, lat, lon)``.
    fcst : xarray.DataArray
        Forecast onset day of year, ``float64`` with dims ``(year, member, lat, lon)``.
    years : list of int
        The years to pair, in the order of the result.
    window : tuple of (int, int), default (1, 15)
        The verification window, in days after the issue date.
    skip_missing : bool, default True
        Leave out a year that has no forecast instead of raising.

    Returns
    -------
    numpy.ndarray of float64, shape (year, member, lat, lon)
        The pairs, ``NaN`` where either side has no onset.

    Raises
    ------
    KeyError
        A year is missing in `fcst` and `skip_missing` is false.
    """
    if not skip_missing and any(y not in fcst.year for y in years):
        raise KeyError("a year is missing")
    return fcst.sel(year=years).values
```

### A class or dataclass

Parameters of an explicit `__init__` are documented in the class docstring; the fields of a dataclass carry an
attribute docstring each.

```python
@dataclass(frozen=True)
class Window:
    """A verification window: the lead days over which a forecast onset is compared with the observed one."""

    start: int
    """First lead day, counted from 1."""
    end: int
    """Last lead day, inclusive."""


class Ledger:
    """The pairs a run has made, kept in memory.

    Parameters
    ----------
    capacity : int, default 100
        The most entries kept; the oldest go first.

    Attributes
    ----------
    entries : list of tuple
        The ``(year, pairs)`` entries, oldest first.
    """

    def __init__(self, capacity=100):
        self.capacity = capacity
        self.entries = []
```

### A property

A property takes a noun phrase and needs no `Returns` section.

```python
class Folder:
    """A run folder."""

    root = "runs"

    @property
    def log_file(self):
        """The path of ``run.log`` inside the run folder."""
        return f"{self.root}/run.log"
```

### A generator

A generator documents `Yields`, a `@contextmanager` function too. pydoclint only accepts a `Yields` section when the
function declares an `Iterator`, `Generator` or `Iterable` return annotation (`Iterator[None]` for a context
manager), so generators are the one place where a return annotation is required.

```python
def chunks(items, size) -> Iterator[list]:
    """Yield consecutive chunks of `items`.

    Parameters
    ----------
    items : sequence
        What to split.
    size : int
        The length of each chunk; the last one may be shorter.

    Yields
    ------
    list
        The next chunk of `items`.
    """
    for i in range(0, len(items), size):
        yield list(items[i:i + size])
```

### A numba kernel

Kernels are documented like any function; say that they are compiled and what the arrays must be.

```python
@njit(cache=True)
def _first_wet_day(rain, threshold):
    """Find the first day on which the rain reaches the threshold, for every cell.

    Compiled with numba (``njit``): the arrays must be C-contiguous and `threshold` a float.

    Parameters
    ----------
    rain : numpy.ndarray of float64, shape (day, cell)
        Daily rain in mm.
    threshold : float
        The wet-day threshold in mm.

    Returns
    -------
    numpy.ndarray of int64, shape (cell,)
        The index of the first wet day, ``-1`` where there is none.
    """
    out = np.full(rain.shape[1], -1, dtype=np.int64)
    for c in range(rain.shape[1]):
        for d in range(rain.shape[0]):
            if rain[d, c] >= threshold:
                out[c] = d
                break
    return out
```

### A CLI handler

The handler of a subcommand documents the namespace it gets and the exit code it returns. Text that argparse shows
(`description=__doc__`) is the first paragraphs of the module docstring, so write those for a reader of ``--help``.

```python
def _cmd_info(args):
    """Print what the cache holds.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line: ``dir`` (the cache folder, or None for the default) and ``kind`` (a list of kinds).

    Returns
    -------
    int
        The exit code: 0, or 1 when the cache folder does not exist.
    """
    return 0 if args.dir is None else 1
```

## Docstrings that appear at run time

A few docstrings are read by the program, not only by Sphinx: argparse help uses a module docstring
(`description=__doc__` in `hcwf/cli/onset.py`) and the config classes build their `__doc__` when they are created
(`hcwf/config/__init__.py`, `hcwf/config/model.py`). When you convert such a module, keep the text a user sees in
``--help`` or ``help(...)`` sensible: reST markup (double backticks) is acceptable there, headings and sections
are not.

## Checking

Run these from the repository root. The first three are what the guard test runs.

```bash
uv run --extra dev ruff check hcwf                       # pydocstyle D rules (numpy convention, D401, D417)
uv run --extra dev pydoclint hcwf | grep -v DOC108       # signature and sections (DOC108 is deliberately off)
uv run --extra dev python tests/test_docstrings.py       # both of the above and the missing-docstring check, filtered
uv run --extra dev pytest tests/test_docstrings.py       # the guard test
uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html   # the rendered documentation
```

`python tests/test_docstrings.py` is the fastest loop: it prints every violation and the count per package
(`--all` includes the packages not converted yet; a path argument limits it to one file or folder).

What the tools check, and what they cannot:

* ruff (`D`): the summary line (a period, imperative mood, a blank line after it), section names and underlines,
  every public object has a docstring, every parameter is documented (D417).
* pydoclint: the `Parameters` names and order equal the signature; `Returns`, `Yields` and `Raises` exist when the code
  returns, yields or raises (and the raised names match); each parameter has a type line. It also checks short
  docstrings and private functions. Nested functions are exempt unless they have sections. It does not check types
  against annotations, nor that the descriptions are true.
* Sphinx: cross-references (`py:obj`), reST syntax and that the numpy sections render.

## Converting a package

`PENDING` in `tests/test_docstrings.py` lists the packages whose docstrings are not converted yet; the guard test, ruff
and the Sphinx warnings skip them. To convert a package:

1. Convert every module of it. `uv run --extra dev python tests/test_docstrings.py --all hcwf/<package>` lists what is
   left (the package is still pending, hence `--all`).
2. Remove its line from `PENDING` in `tests/test_docstrings.py`, and its line (`"hcwf/<package>/**" = ["D"]`) from
   `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml`. A test fails when the two lists differ. The Sphinx
   build reads `PENDING`, so its warnings for the package become errors at once.
3. Run the checks above until they pass (`python tests/test_docstrings.py hcwf/<package>`, now without `--all`, must
   print `total 0`), then the Sphinx build and the full test suite.
