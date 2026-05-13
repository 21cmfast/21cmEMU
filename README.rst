.. image:: 21cmEMU_logo_horizontal.png
   :alt: 21cmEMU

=======
A 21cmFAST emulator of the first billion years
=======


|PyPI| |Status| |Python| |License| |RTD| |Tests| |Codecov| |pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/py21cmemu.svg
   :target: https://pypi.org/project/py21cmemu/
.. |Status| image:: https://img.shields.io/pypi/status/py21cmemu.svg
   :target: https://pypi.org/project/py21cmemu/
.. |Python| image:: https://img.shields.io/pypi/pyversions/py21cmemu.svg

.. |License| image:: https://img.shields.io/pypi/l/py21cmemu.svg
    :target: https://github.com/21cmfast/21cmEMU/blob/main/LICENSE
.. |Tests| image:: https://github.com/21cmfast/21cmEMU/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/21cmfast/21cmEMU/actions/workflows/tests.yml
.. |Codecov| image:: https://codecov.io/gh/21cmfast/21cmEMU/branch/main/graph/badge.svg?token=yUOqyTlZ3z
    :target: https://codecov.io/gh/21cmfast/21cmEMU
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
.. |RTD| image:: https://readthedocs.org/projects/21cmemu/badge/?version=latest
    :target: https://21cmemu.readthedocs.io/en/latest/
    :alt: Documentation Status

An emulator of 21cmFAST summaries, supporting three galaxy-formation models:

- **acg** (v1) — Atomic Cooling Galaxies (Pop II only). A 9-parameter emulator
  for the standard reionization scenario without mini-halos.
- **radio** (v2) — Radio Background. A 5-parameter emulator that adds a radio
  background sourced by molecular cooling (Pop III) stars on top of atomic
  cooling galaxies.
- **mcg** (v3) — Molecular Cooling Galaxies. The most complete 11-parameter
  emulator, jointly modelling atomic and molecular cooling galaxies (Pop II + Pop
  III). Uniquely emulates the **2D** cylindrical power spectrum
  P(k\ :sub:`⊥`\ , k\ :sub:`∥`\ ) via a score-based diffusion model.

Emulated summary statistics per model:

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Output
     - acg (v1)
     - radio (v2)
     - mcg (v3)
   * - Global brightness temperature T\ :sub:`b`
     - ✓
     - ✓
     - ✓
   * - Neutral fraction x\ :sub:`HI`
     - ✓
     - ✓
     - ✓
   * - Thomson optical depth τ
     - ✓
     - ✓
     - ✓
   * - IGM spin temperature T\ :sub:`s`
     - ✓
     - ✗
     - ✓
   * - Radio temperature T\ :sub:`r`
     - ✗
     - ✓
     - ✗
   * - 1D power spectrum P(k)
     - ✓
     - ✓
     - ✓
   * - 2D power spectrum P(k\ :sub:`⊥`\ , k\ :sub:`∥`\ )
     - ✗
     - ✗
     - ✓
   * - UV luminosity functions
     - ✓
     - ✗
     - ✓



Documentation
=============

See `the documentation <https://21cmemu.readthedocs.io/en/latest/>`_ for tutorials and API.

Issues
======

If you encounter any problems, please `file an issue <https://github.com/21cmFAST/21cmEMU/issues>`_ along with a detailed description.

Contributing
============

Contributions are very welcome.
To learn more, see the `Contributor Guide <https://github.com/21cmFAST/21cmEMU/blob/main/CONTRIBUTING.md>`_.

Citation
========

If you use ``21cmEMU`` in your research, please cite `Breitman+23 <https://arxiv.org/abs/2309.05697>`_.

License
=======
Distributed under the terms of the `MIT license <https://github.com/21cmFAST/21cmEMU/blob/main/LICENSE>`_, ``21cmEMU`` is free and open source software.


Credits
-------
This project was generated from `@cjolowicz <https://github.com/cjolowicz>`_'s `Hypermodern Python Cookiecutter <https://github.com/cjolowicz/cookiecutter-hypermodern-python>`_ template.
