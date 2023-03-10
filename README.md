# 21cmEMU

[![PyPI](https://img.shields.io/pypi/v/py21cmemu.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/py21cmemu.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/py21cmemu)][python version]
[![License](https://img.shields.io/pypi/l/py21cmemu)][license]

[![Read the documentation at https://21cmEMU.readthedocs.io/](https://img.shields.io/readthedocs/py21cmEMU/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/21cmFAST/21cmEMU/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/21cmFAST/21cmEMU/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/py21cmemu/
[status]: https://pypi.org/project/py21cmemu/
[python version]: https://pypi.org/project/py21cmemu
[read the docs]: https://21cmemu.readthedocs.io/
[tests]: https://github.com/21cmFAST/21cmEMU/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/21cmFAST/21cmEMU
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

- Uses Tensorflow to emulate the following summary statistics: 21-cm power spectrum, 21-cm global brightness temperature, IGM spin temperature, and neutral fraction.
- Uses 21cmFAST to analytically calculate the UV luminosity functions and the Thomson optical depth to the CMB.

## Requirements

- Tensorflow >= 2.6
- 21cmFAST

## Installation

You can install _py21cmEMU_ via [pip] from [PyPI]:
```console
$ pip install py21cmemu
```

Note that you need `gcc` and the `fftw3` and `gsl` packages for the `21cmFAST` installation.

## Usage

Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_21cmEMU_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/21cmFAST/21cmEMU/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/21cmFAST/21cmEMU/blob/main/LICENSE
[contributor guide]: https://github.com/21cmFAST/21cmEMU/blob/main/CONTRIBUTING.md
[command-line reference]: https://21cmEMU.readthedocs.io/en/latest/usage.html
