"""Sphinx configuration."""
import sys
from pathlib import Path
import py21cmemu

sys.path.insert(0, str(Path(__file__).absolute().parent.parent / "src"))

master_doc = 'index'
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

project = "21cmEMU"
author = "Daniela Breitman"
copyright = "2023, Daniela Breitman"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting"
]
version = release = py21cmemu.__version__
autodoc_typehints = "description"
html_theme = "furo"
html_logo = "images/21cmEMU_logo_vertical.png"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": True,
}
autosectionlabel_prefix_document = True

autosummary_generate = True

nbsphinx_execute = "never"

exclude_patterns = [
    "_build",
    "build",
    "**.ipynb_checkpoints"
]
