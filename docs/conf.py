"""Sphinx configuration for well-log-toolkit documentation."""

import importlib.metadata

project = "well-log-toolkit"
copyright = "2024, Kristian dF Kollsgård"
author = "Kristian dF Kollsgård"
release = importlib.metadata.version("well-log-toolkit")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "well-log-toolkit"
html_theme_options = {
    "source_repository": "https://github.com/kkollsga/well_log_toolkit",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- Extension configuration -------------------------------------------------

# Napoleon (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# MyST
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Source suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
