![21cmEMU](docs/21cmEMU_logo_horizontal.png)

# A 21cmFAST emulator of the first billion years

[![PyPI](https://img.shields.io/pypi/v/py21cmemu.svg)](https://pypi.org/project/py21cmemu/)
[![Status](https://img.shields.io/pypi/status/py21cmemu.svg)](https://pypi.org/project/py21cmemu/)
[![Python](https://img.shields.io/pypi/pyversions/py21cmemu.svg)](https://pypi.org/project/py21cmemu/)
[![License](https://img.shields.io/pypi/l/py21cmemu.svg)](https://github.com/21cmfast/21cmEMU/blob/main/LICENSE)
[![Tests](https://github.com/21cmfast/21cmEMU/actions/workflows/tests.yml/badge.svg)](https://github.com/21cmfast/21cmEMU/actions/workflows/tests.yml)
[![Codecov](https://codecov.io/gh/21cmfast/21cmEMU/branch/main/graph/badge.svg?token=yUOqyTlZ3z)](https://codecov.io/gh/21cmfast/21cmEMU)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![RTD](https://readthedocs.org/projects/21cmemu/badge/?version=latest)](https://21cmemu.readthedocs.io/en/latest/)

An emulator of 21cmFAST summaries, supporting three galaxy-formation models:

- **acg** (v1) — Atomic Cooling Galaxies (Pop II only). A 9-parameter emulator
  for the standard reionization scenario without mini-halos.
- **radio** (v2) — Radio Background. A 5-parameter emulator that adds a radio
  background sourced by molecular cooling (Pop III) stars on top of atomic
  cooling galaxies.
- **mcg** (v3) — Molecular Cooling Galaxies. The most complete 11-parameter
  emulator, jointly modelling atomic and molecular cooling galaxies (Pop II +
  Pop III). Uniquely emulates the **2D** cylindrical power spectrum
  P(k<sub>⊥</sub>, k<sub>∥</sub>) via a score-based diffusion model.

## Emulated summary statistics per model

| Output | acg (v1) | radio (v2) | mcg (v3) |
|--------|:--------:|:----------:|:--------:|
| Global brightness temperature T<sub>b</sub> | ✓ | ✓ | ✓ |
| Neutral fraction x<sub>HI</sub> | ✓ | ✓ | ✓ |
| Thomson optical depth τ | ✓ | ✓ | ✓ |
| IGM spin temperature T<sub>s</sub> | ✓ | ✗ | ✓ |
| Radio temperature T<sub>r</sub> | ✗ | ✓ | ✗ |
| 1D power spectrum P(k) | ✓ | ✓ | ✓ |
| 2D power spectrum P(k<sub>⊥</sub>, k<sub>∥</sub>) | ✗ | ✗ | ✓ |
| UV luminosity functions | ✓ | ✗ | ✓ |

## Documentation

See [the documentation](https://21cmemu.readthedocs.io/en/latest/) for tutorials and API.

## Issues

If you encounter any problems, please [file an issue](https://github.com/21cmFAST/21cmEMU/issues) along with a detailed description.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide](https://github.com/21cmFAST/21cmEMU/blob/main/CONTRIBUTING.md).

## Citation

If you use `21cmEMU` in your research, please cite [Breitman+23](https://arxiv.org/abs/2309.05697). In particular, if you use `21cmEMUv2`, also cite [Cang+24](https://arxiv.org/abs/2411.08134).
If you use the newest **mcg/v3** emulator, please also cite [Breitman+26](https://arxiv.org/abs/2606.00219).

## License

Distributed under the terms of the [MIT license](https://github.com/21cmFAST/21cmEMU/blob/main/LICENSE), `21cmEMU` is free and open source software.
