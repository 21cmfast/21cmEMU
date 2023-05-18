"""Sphinx configuration."""
import sys
from pathlib import Path
import subprocess
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).absolute().parent.parent / "src"))


class Mock(MagicMock):
    """Make a Mock so that a package doesn't have to actually exist."""

    @classmethod
    def __getattr__(cls, name):
        """Get stuff."""
        return MagicMock()

MOCK_MODULES = ["py21cmemu"]

try:
    from py21cmemu import Emulator
except ImportError:
    raise

master_doc = 'index'
source_suffix = ".rst"

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
out = subprocess.run(["python", "setup.py", "--version"], capture_output=True)
version = release = py21cmemu.__version__
autodoc_typehints = "description"
html_theme = "furo"
autosectionlabel_prefix_document = True

autosummary_generate = True

exclude_patterns = [
    "_build",
    "build",
    "**.ipynb_checkpoints"
]
