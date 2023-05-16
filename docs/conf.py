"""Sphinx configuration."""
import sys
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).absolute().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).absolute().parent.parent / "src" / "py21cmemu"))

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
    "nbsphinx"
]
out = subprocess.run(["python", "setup.py", "--version"], capture_output=True)
version = (
    release
) = out.stdout.decode().rstrip()
autodoc_typehints = "description"
html_theme = "furo"
autosectionlabel_prefix_document = True

autosummary_generate = True

exclude_patterns = [
    "_build",
    "build",
    "**.ipynb_checkpoints"
]
