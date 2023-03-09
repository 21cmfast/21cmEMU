"""Sphinx configuration."""
from unittest.mock import MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent / "src"))

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


MOCK_MODULES = ['py21cmfast']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

project = "21cmEMU"
author = "Daniela Breitman"
copyright = "2023, Daniela Breitman"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_mock_imports = ['py21cmfast']
autodoc_typehints = "description"
html_theme = "furo"
