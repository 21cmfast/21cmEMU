#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

from setuptools import find_packages, setup

import io
import re
from glob import glob
from os.path import basename, dirname, join, splitext


def _read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ).read()


def _find_version(*file_paths):
    version_file = _read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# ======================================================================================================================

setup(
    name="EMU21cmFAST",
    version=_find_version("21cmemu", "__init__.py"),
    license="MIT license",
    description="Emulator of 21cmFAST summaries.",
    long_description="%s\n%s"
    % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", _read("README.md")
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", _read("CHANGELOG.md")),
    ),
    author="Daniela Breitman",
    author_email="daniela.breitman@sns.it",
    url="https://github.com/21cmFAST/21cmEMU",
    packages=find_packages("src"),
    package_dir={"": "21cmemu"},
    py_modules=[],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords=["Epoch of Reionization", "Cosmology"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib>=2.1",
        "cached_property",
        "21cmFAST",
        "tensorflow>=2.6",
    ],
)

