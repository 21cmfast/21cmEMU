# EMU21cmFAST

[![PyPI](https://img.shields.io/pypi/v/EMU21cmFAST.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/EMU21cmFAST.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/EMU21cmFAST)][python version]
[![License](https://img.shields.io/pypi/l/EMU21cmFAST)][license]

[![Read the documentation at https://EMU21cmFAST.readthedocs.io/](https://img.shields.io/readthedocs/EMU21cmFAST/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/DanielaBreitman/EMU21cmFAST/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/DanielaBreitman/EMU21cmFAST/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/EMU21cmFAST/
[status]: https://pypi.org/project/EMU21cmFAST/
[python version]: https://pypi.org/project/EMU21cmFAST
[read the docs]: https://EMU21cmFAST.readthedocs.io/
[tests]: https://github.com/DanielaBreitman/EMU21cmFAST/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/DanielaBreitman/EMU21cmFAST
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

* Uses Tensorflow to emulate the following summary statistics: 21-cm power spectrum, 21-cm global brightness temperature, IGM spin temperature, and neutral fraction.
* Uses 21cmFAST to analytically calculate the UV luminosity functions and the Thomson optical depth to the CMB.


## Requirements

- Tensorflow >= 2.6
- 21cmFAST

## Installation

You can install _EMU21cmFAST_ via [pip] from [PyPI]:

```console
$ pip install EMU21cmFAST
```

## Usage

Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_EMU21cmFAST_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/DanielaBreitman/EMU21cmFAST/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/DanielaBreitman/EMU21cmFAST/blob/main/LICENSE
[contributor guide]: https://github.com/DanielaBreitman/EMU21cmFAST/blob/main/CONTRIBUTING.md
[command-line reference]: https://EMU21cmFAST.readthedocs.io/en/latest/usage.html
