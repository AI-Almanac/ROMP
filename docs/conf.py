"""Sphinx configuration of the HCWF documentation.

Build with ``uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html``.
See ``docs/contributing_docstrings.md`` for the docstring conventions this configuration renders.
"""

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

# -- Intersphinx --------------------------------------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
intersphinx_timeout = 20

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
