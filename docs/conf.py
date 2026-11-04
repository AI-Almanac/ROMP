"""Sphinx configuration of the HCWF documentation.

Build with ``uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html``.
See ``docs/contributing_docstrings.md`` for the docstring conventions this configuration renders.
"""

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT))

# -- Project ------------------------------------------------------------------------------------------------------------

project = "HCWF"
author = "Bo Dong"
copyright = "HCWF contributors"
release = "0.0.1"

# -- Extensions ---------------------------------------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # NumPy-style sections only (see below)
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_nb",  # MyST markdown guides and the notebooks (rendered, never executed)
]

exclude_patterns = [
    "_build",
    "_templates",
    "review",  # historical records, not part of the documentation
    "blend_merge",
    "blend_merge.md",
    "**/.ipynb_checkpoints",
]
templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}

# Single backticks are object references (`hcwf.api.run`, `~hcwf.api.run`); literals use double backticks.
default_role = "py:obj"

# -- API reference ------------------------------------------------------------------------------------------------------

autosummary_generate = True  # docs/api.rst -> docs/_api/*.rst, generated at build time and not committed
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": True,  # the private (_name) helpers are documented too
    "special-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
# Parameter and return types come from the NumPy docstring type lines (they carry shape, dtype and meaning that
# annotations cannot), so signature annotations are not rendered a second time: "none" drops them from the signature
# and does not add them to the parameter list. Annotations stay in the code for editors and type checkers.
autodoc_typehints = "none"
autodoc_inherit_docstrings = True

# Heavy optional imports that fail in the docs environment are listed here; empty because every import is guarded.
autodoc_mock_imports: list[str] = []

# -- Napoleon (NumPy style only) ----------------------------------------------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = False
napoleon_preprocess_types = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_attr_annotations = True


def _attribute_docstring_without_inline_type(self):
    """Return an attribute docstring's lines whole, never splitting a ``type: description`` first line.

    napoleon reads ``text: more`` on the first line of an attribute docstring as a type and a description, so a
    sentence such as "The yes/no keys: ..." rendered as "Type: The yes/no keys". HCWF never writes types in
    attribute docstrings (the annotation is the type), so the whole docstring is the description.

    Returns
    -------
    list of str
        The docstring's lines, formatted as a field description.
    """
    lines = [*self._dedent(self._consume_to_end())]
    return self._format_field("", "", self.__class__(lines, self._config).lines())


from sphinx.ext.napoleon.docstring import GoogleDocstring  # noqa: E402

GoogleDocstring._parse_attribute_docstring = _attribute_docstring_without_inline_type

# -- Type lines under nitpicky ------------------------------------------------------------------------------------------
# napoleon turns every word of a type line into a cross-reference. The HCWF type lines also carry shape groups
# (``shape (case, bin)``, ``same shape as `year```) and parenthesised tuples (``tuple of (int, int)``) that napoleon splits
# at the commas into fragments such as "shape (case" and "bin)", which are not objects. Two small patches keep them
# out of the cross-reference machinery without touching a docstring:
#   * a shape group is rendered as a literal;
#   * a "(" or ")" around an object is kept as text, so ``(int, int)`` still links both ``int``.
# Everything else (dtype words, array_like, ...) is resolved with ``napoleon_type_aliases`` below.
import sphinx.ext.napoleon.docstring as _napoleon_docstring  # noqa: E402

_SHAPE_GROUP = re.compile(
    r"\b(?:same |any )?shape\b"
    r"(?: as `[^`]*`| ``[^`]*``| \((?:[^()]|\([^()]*\))*\)(?:,? or \((?:[^()]|\([^()]*\))*\))*| [A-Za-z_]\w*)?"
)
_SHAPE_MARK = re.compile(r":py:class:`\x00(\d+)\x00`")
_original_convert_type_spec = _napoleon_docstring._convert_type_spec
_original_convert_type_spec_obj = _napoleon_docstring._convert_type_spec_obj


def _convert_type_spec_with_shapes(_type, translations=None, debug_location=None):
    """Convert a type line, rendering its shape groups as literals instead of cross-references.

    Parameters
    ----------
    _type : str
        The type line.
    translations : dict of str to str, optional
        The ``napoleon_type_aliases``.
    debug_location : str, optional
        Where the line is, for napoleon's own warnings.

    Returns
    -------
    str
        The reST of the type line.
    """
    shapes = []

    def mark(match):
        """Replace a shape group by a placeholder object name and remember the group."""
        shapes.append(match.group(0))
        return f"\x00{len(shapes) - 1}\x00"

    converted = _original_convert_type_spec(_SHAPE_GROUP.sub(mark, _type), translations, debug_location)
    return _SHAPE_MARK.sub(lambda match: f"``{shapes[int(match.group(1))]}``", converted)


def _convert_type_spec_obj_with_parentheses(obj, translations):
    """Convert one object of a type line, keeping a surrounding ``(`` and ``)`` as plain text.

    Parameters
    ----------
    obj : str
        The object name, possibly with an opening or closing parenthesis from a tuple type.
    translations : dict of str to str
        The ``napoleon_type_aliases``.

    Returns
    -------
    str
        The reST of the object.
    """
    core = obj.strip("()")
    head = obj[: len(obj) - len(obj.lstrip("("))]
    tail = obj[len(obj.rstrip(")")) :]
    if not re.fullmatch(r"[A-Za-z_][\w.\-]*(?:\[\w+\])?", core):
        # Not an object name: punctuation such as "()" stays text, a number or an expression such as "1 << 22" is a literal.
        return obj if not re.search(r"\w", core) else f"{head}``{core}``{tail}"
    return head + _original_convert_type_spec_obj(core, translations) + tail


_napoleon_docstring._convert_type_spec = _convert_type_spec_with_shapes
_napoleon_docstring._convert_type_spec_obj = _convert_type_spec_obj_with_parentheses

# Words of the type lines that are not importable objects, mapped to what they stand for.
_NUMPY_DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"]
napoleon_type_aliases = {
    **{dtype: f":py:obj:`numpy.{dtype}`" for dtype in _NUMPY_DTYPES},
    "datetime64": ":py:obj:`numpy.datetime64`",
    "array_like": ":external+numpy:term:`array_like`",
    "array-like": ":external+numpy:term:`array_like`",
    "sequence": ":external+python:term:`sequence`",
    "iterable": ":external+python:term:`iterable`",
    "mapping": ":external+python:term:`mapping`",
    "callable": ":external+python:term:`callable`",
    "hashable": ":external+python:term:`hashable`",
    "collection": ":class:`collections.abc.Collection`",
    "module": ":external+python:term:`module`",
    "file object": ":external+python:term:`file object`",
    "datetime-like": ":literal:`datetime-like`",  # any of datetime.datetime, numpy.datetime64, pandas.Timestamp, cftime
    "datetime64[ns]": ":py:obj:`datetime64[ns] <numpy.datetime64>`",
    "datetime64[D]": ":py:obj:`datetime64[D] <numpy.datetime64>`",
    "hashlib._Hash": ":literal:`hashlib._Hash`",  # the private type of hashlib's hash objects
    # HCWF names written bare in type lines, where "default NO_SELECTION" and "Selection" cannot carry a module path.
    "Selection": ":py:obj:`~hcwf.config.model.Selection`",
    "NO_SELECTION": ":py:obj:`~hcwf.config.model.NO_SELECTION`",
    "UnitsSpec": ":py:obj:`~hcwf.spatial.units.UnitsSpec`",
    "ForecastRead": ":py:obj:`~hcwf.io.forecasts.ForecastRead`",
    "Status": ":py:obj:`~hcwf.onset.model.Status`",
    "NORMAL": ":literal:`NORMAL`",  # hcwf.log.NORMAL, a module constant that is not documented
    "WASH_ALPHA": ":literal:`WASH_ALPHA`",  # hcwf.onset.viz.style.WASH_ALPHA, a module constant that is not documented
    "_Context": ":literal:`_Context`",  # private dataclass of hcwf.blend.evaluation, not rendered
    "_Cells": ":py:obj:`~hcwf.spatial.units._Cells`",
}

# Objects of libraries whose documentation publishes no inventory, and names that are not objects.
nitpick_ignore_regex = [
    # cartopy publishes no intersphinx inventory.
    (r"py:class", r"cartopy\..*"),
    # PyYAML publishes no intersphinx inventory.
    (r"py:class", r"yaml\..*"),
    (r"py:exc", r"yaml\..*"),
    # shapely documents its geometry base class on a manual page only, so the class has no inventory entry.
    (r"py:class", r"shapely\.(Geometry|geometry\.base\.BaseGeometry)"),
    # The Raises section of hcwf.io.reference.read_years names the variable that is re-raised (`raise error`).
    (r"py:exc", r"error"),
]

# -- Intersphinx --------------------------------------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
intersphinx_timeout = 20
nitpicky = True

# -- Notebooks and markdown ---------------------------------------------------------------------------------------------

nb_execution_mode = "off"  # the notebooks are rendered as stored, never run
suppress_warnings = ["myst.header"]  # the notebooks start at H2 and skip levels; not worth restructuring them
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3

# -- HTML ---------------------------------------------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "HCWF"
html_theme_options = {
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "github_url": "https://github.com/AI-Almanac/HCWF",
}

# -- Config classes -----------------------------------------------------------------------------------------------------
# The config classes (`hcwf.config.Config`, its sections) are dataclasses whose every argument defaults to the
# ``UNSET`` marker (a key that was not set). Their generated signature lists every key "=UNSET", which says nothing
# the attribute list below it does not; it is shown as ``(**keys)`` instead.


def _unset_signature(app, what, name, obj, options, signature, return_annotation):
    """Show the signature of a class whose every argument defaults to ``UNSET`` as ``(**keys)``.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The application.
    what : str
        The kind of object.
    name : str
        Its full name.
    obj : object
        The object.
    options : dict
        The autodoc options.
    signature : str or None
        The signature autodoc made.
    return_annotation : str or None
        The return annotation autodoc made.

    Returns
    -------
    tuple of (str, str) or None
        The new signature and return annotation, or None to keep autodoc's.
    """
    import dataclasses

    from hcwf.config import UNSET

    if what != "class" or not dataclasses.is_dataclass(obj):
        return None
    fields = dataclasses.fields(obj)
    if fields and all(f.default is UNSET for f in fields if f.init):
        return "(**keys)", return_annotation
    return None


def setup(app):
    """Connect the signature hook.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The application.
    """
    app.connect("autodoc-process-signature", _unset_signature)
