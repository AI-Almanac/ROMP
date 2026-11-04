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
    "**/.ipynb_checkpoints",
]
templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}

# Single backticks are object references (`hcwf.api.run`, `~hcwf.api.run`); literals use double backticks.
default_role = "py:obj"

# -- API reference ------------------------------------------------------------------------------------------------------

autosummary_generate = True  # docs/api.rst -> docs/_api/*.rst, generated at build time and not committed
autosummary_imported_members = False
# Package modules with an ``__all__`` must still be recursed into (``autosummary_ignore_module_all = False`` would list only the
# submodules named in ``hcwf.__all__``). The public interface is enforced where names are rendered: the module template
# uses ``automodule :members:``, which honours each module's ``__all__``.
autosummary_ignore_module_all = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,  # the rendered docs show the public interface (each module's __all__) only
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
#   * a "(" or ")" around an object is kept as text, so ``(int, int)`` still links both ``int``;
#   * a type line that is one literal (``like this``) is kept whole: the config classes of hcwf.config list each attribute
#     with the schema's description of the value expected ("a whole number of 1 or more"), which is text, not an object.
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
    if _type.startswith("``") and _type.endswith("``") and _type.count("``") == 2:
        return _type  # a free-text type line, written as a literal (the generated attributes of hcwf.config)
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
# The config classes (`hcwf.config.Config`, `hcwf.config.Dataset`, `hcwf.config.Mask`, the section classes) are dataclasses
# whose every argument defaults to the ``UNSET`` marker (a key that was not set). Three things keep them readable:
#   * the class docstring ends with a ``Parameters`` section that hcwf/config/__init__.py generates from the schema
#     (Python type, default, help), so ``help(Onset)`` shows it too and the schema stays the one source;
#   * the generated signature, which would list every key "=UNSET", is shown as the keyword-only keys with the schema
#     default or ``...`` (`_keyword_signature`, from the same table as the Parameters section);
#   * autodoc would also list each dataclass field as an empty attribute "key = UNSET", a second entry next to the one the
#     Parameters section makes (and a duplicate-object warning); `_skip_unset_fields` drops those. The hook tests the
#     value (the ``UNSET`` sentinel, on a class), not the class names, so a new config class needs nothing here.


def _skip_unset_fields(app, what, name, obj, skip, options):
    """Skip a class attribute whose value is the ``UNSET`` sentinel: the Parameters section documents it.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The application.
    what : str
        The kind of object whose member this is.
    name : str
        The member's name.
    obj : object
        The member.
    skip : bool
        Whether autodoc would skip the member.
    options : dict
        The autodoc options.

    Returns
    -------
    bool or None
        True for a dataclass field defaulting to ``UNSET``; None to leave autodoc's choice.
    """
    from hcwf.config import UNSET

    return True if what == "class" and obj is UNSET else None


def _keyword_signature(app, what, name, obj, options, signature, return_annotation):
    """Show a config class's signature as its keyword-only keys, from the schema (never ``=UNSET``).

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
    from hcwf.config import _PARAMETERS, keyword_signature

    if what == "class" and obj in _PARAMETERS:
        return keyword_signature(obj), return_annotation
    return None


# -- Generated reference pages ------------------------------------------------------------------------------------------
# Two directives generate their page at build time from the code, so nothing is copied by hand:
#   * ``hcwf-cli``: the text of ``hcwf-run -h``, of the run overrides and of every subcommand's ``-h``, taken by calling the
#     handlers of `hcwf.cli.COMMANDS` (argparse prints the help and exits);
#   * ``hcwf-config-keys``: one table per config section (key, type, default, help) from `hcwf.config.schema.FIELDS`.

import contextlib  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402

from docutils import nodes  # noqa: E402
from docutils.parsers.rst import Directive  # noqa: E402


def _section(directive, title, children):
    """Return a section node with a title, registered as a link target of the document.

    Parameters
    ----------
    directive : docutils.parsers.rst.Directive
        The directive that makes the section.
    title : str
        The section title.
    children : list of docutils.nodes.Node
        The section's content.

    Returns
    -------
    docutils.nodes.section
        The section, with the title first.
    """
    section = nodes.section(ids=[nodes.make_id(title)], names=[nodes.fully_normalize_name(title)])
    section += nodes.title(title, title)
    section.extend(children)
    directive.state.document.note_implicit_target(section, section)
    return section


def _help_of(handler, args, prog="hcwf-run", width="90"):
    """Capture what a command prints for ``-h``.

    Parameters
    ----------
    handler : callable
        Called with the argument list; argparse prints the help and raises ``SystemExit``.
    args : list of str
        The arguments (ending in ``-h``).
    prog : str
        The program name argparse puts in the usage line.
    width : str
        The terminal width argparse wraps at.

    Returns
    -------
    str
        The help text.
    """
    out = io.StringIO()
    saved = (sys.argv, os.environ.get("COLUMNS"))
    sys.argv, os.environ["COLUMNS"] = [prog], width
    try:
        with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
            handler(args)
    finally:
        sys.argv = saved[0]
        if saved[1] is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = saved[1]
    return out.getvalue().rstrip() + "\n"


class HcwfCli(Directive):
    """Render the help of ``hcwf-run``: the overview, the run overrides and every subcommand."""

    has_content = False

    def run(self):
        """Build the sections from the argparse parsers and handlers of `hcwf.cli`.

        Returns
        -------
        list of docutils.nodes.Node
            One section per command, each holding the command's help as a literal block.
        """
        from hcwf import cli
        from hcwf.cli.parser import create_parser

        def block(text):
            return nodes.literal_block(text, text, language="text")

        parser = create_parser()
        parser.prog = "hcwf-run"
        result = [
            _section(self, "hcwf-run", [block(cli.HELP)]),
            _section(self, "Run overrides", [
                nodes.paragraph(text="With a config, ``hcwf-run -p run.yaml -h`` lists the command-line overrides of config values:"
                                .replace("``", "")),
                block(parser.format_help()),
            ]),
        ]
        for name, handler in cli.COMMANDS.items():
            text = _help_of(handler, ["-h"], prog=f"hcwf-run {name}")
            children = [block(text)]
            if name == "onset":
                import re

                for sub in re.findall(r"\{([\w,\-]+)\}", text)[:1] and re.findall(r"\{([\w,\-]+)\}", text)[0].split(","):
                    children.append(_section(self, f"hcwf-run onset {sub}",
                                             [block(_help_of(handler, [sub, "-h"], prog=f"hcwf-run onset {sub}"))]))
            result.append(_section(self, f"hcwf-run {name}", children))
        return result


class HcwfConfigKeys(Directive):
    """Render one table of keys per section of the run file, from `hcwf.config.schema`."""

    has_content = False

    def run(self):
        """Build the tables from `hcwf.config.schema.FIELDS`.

        Returns
        -------
        list of docutils.nodes.Node
            One section per config section (the top-level keys first), each holding a table of key, type, default and help.
        """
        from hcwf.config import schema

        def prose(text):
            return re.sub(r"(?<!`)`([^`]+)`(?!`)", r"\1", text)

        def default_of(field):
            if field.required:
                return "required"
            if field.internal:
                return "set by HCWF"
            if not field.fill or callable(field.default):
                return "chosen by HCWF"
            return repr(field.default)

        groups = {None: [f for f in schema.FIELDS if f.section is None]}
        for name in schema.SECTIONS:
            groups[name] = [f for f in schema.FIELDS if f.section == name]
        result = []
        for name, fields in groups.items():
            if not fields:
                continue
            table = nodes.table()
            tgroup = nodes.tgroup(cols=4)
            table += tgroup
            for width in (22, 28, 16, 60):
                tgroup += nodes.colspec(colwidth=width)
            head = nodes.thead()
            tgroup += head
            body = nodes.tbody()
            tgroup += body

            def row(cells, rows=body, literal_first=False):
                r = nodes.row()
                for i, text in enumerate(cells):
                    para = nodes.paragraph()
                    para += nodes.literal(text, text) if (literal_first and i in (0, 2)) else nodes.Text(text)
                    r += nodes.entry("", para)
                rows += r

            row(["Key", "Type", "Default", "Help"], rows=head)
            for field in fields:
                row([field.path, field.expected, default_of(field), prose(field.help)], literal_first=True)
            if name is None:
                title, intro = "Top-level keys", "Keys written at the top of the run file."
            else:
                title, intro = f"{name}:", f"The ``{name}:`` section: {schema.SECTIONS[name]}.".replace("``", "")
            result.append(_section(self, title, [nodes.paragraph(text=intro), table]))
        return result


def _literal_for_internal_name(app, env, node, contnode):
    """Show a reference to an existing but unrendered (internal) name as a literal instead of a broken link.

    The rendered documentation shows the public interface only (each module's ``__all__``), but the docstrings of public
    names may still mention the private helpers and module constants they use. Such a mention is kept as plain code text.
    A target that cannot be imported at all is left alone, so it is still reported as a broken reference.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The application.
    env : sphinx.environment.BuildEnvironment
        The build environment.
    node : sphinx.addnodes.pending_xref
        The reference that was not found.
    contnode : docutils.nodes.Node
        The text the reference would have shown.

    Returns
    -------
    docutils.nodes.Node or None
        The literal text of an existing hcwf object, or None for a target that does not exist.
    """
    import importlib

    if node.get("refdomain") != "py":
        return None
    target = node["reftarget"].lstrip("~.")
    module = node.get("py:module")
    candidates = [target] + ([f"{module}.{target}"] if module else [])
    for name in candidates:
        if not name.startswith("hcwf"):
            continue
        parts = name.split(".")
        for i in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:i]))
            except ImportError:
                continue
            try:
                for part in parts[i:]:
                    obj = getattr(obj, part)
            except AttributeError:
                break
            return nodes.literal(target, node.astext() or target)
    return None


def setup(app):
    """Connect the config-class hooks and add the generated-reference directives.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The application.
    """
    app.connect("autodoc-process-signature", _keyword_signature)
    app.connect("autodoc-skip-member", _skip_unset_fields)
    app.connect("missing-reference", _literal_for_internal_name)
    app.add_directive("hcwf-cli", HcwfCli)
    app.add_directive("hcwf-config-keys", HcwfConfigKeys)
