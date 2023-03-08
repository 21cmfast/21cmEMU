"""Sphinx configuration."""
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
