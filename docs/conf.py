"""Sphinx configuration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent / "src"))


project = "21cmEMU"
author = "Daniela Breitman"
copyright = "2023, Daniela Breitman"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
