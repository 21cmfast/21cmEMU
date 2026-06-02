Quickstart
==========

Installation
------------

Make sure you have ``git-lfs`` installed on your system. Otherwise, ``21cmEMU`` will not be installed properly.

For users
^^^^^^^^^

To use ``21cmEMU``, install it with pip:

.. code-block:: console

    $ pip install py21cmemu

For developers
^^^^^^^^^^^^^^

Clone the repository and install the development dependencies using `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: console

    $ git clone https://github.com/21cmfast/21cmEMU
    $ cd 21cmEMU
    $ uv sync --group dev

Or with pip:

.. code-block:: console

    $ git clone https://github.com/21cmfast/21cmEMU
    $ cd 21cmEMU
    $ pip install -e .

.. note::

   If you need GPU (CUDA) support, install the appropriate version of PyTorch
   before installing ``21cmEMU`` by following the
   `official PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.
   Conda/mamba can be convenient for installing CUDA libraries, but only the
   minimal CUDA toolkit is needed — all Python dependencies can then be managed
   with ``uv`` or ``pip``.

To test the installation, run:

.. code-block:: python

    from py21cmemu import Emulator
    emu_instance = Emulator()

Emulating 21cmFAST Summaries
----------------------------

``py21cmemu`` currently supports three emulators:

- ``default`` (v1, PyTorch runtime; converted from the original TensorFlow model)
- ``radio_background`` (v2, PyTorch)
- ``mh`` (v3, PyTorch minihalo emulator)

The emulator ``predict`` method accepts two kinds of input parameters:

- Fully normalized parameters i.e. nine numbers :math:`\\in [0,1]`. These can either be in a ``numpy`` array
  in the following order: ``['F_STAR10', 'ALPHA_STAR', 'F_ESC10', 'ALPHA_ESC', 'M_TURN', 't_STAR', 'L_X', 'NU_X_THRESH', 'X_RAY_SPEC_INDEX']`` or in a dictionary with these labels as keys.

- Parameters with the following units (same as ``21cmFAST``):

If you have ``21cmFAST`` installed, you can supply the parameters directly with the ``defining_dict`` attribute of ``p21.AstroParams`` objects.

You can batch evaluate by putting a set of parameters into a ``list`` or an ``np.ndarray``.

Let's look at a basic example. After importing the emulator in the previous code snippet:

.. code-block:: python

    import numpy as np
    theta = np.random.rand(9*5).reshape((5,9))
    theta, output, output_errors = emu_instance.predict(theta)

The output here will contain all the summaries i.e. power spectrum, global 21-cm brightness temperature,
IGM spin temperature, neutral fraction, as well as emulated $\tau_e$ and UV LFs.

Using the v3 (Minihalo) Emulator
--------------------------------

The v3 minihalo emulator uses an LSTM architecture for global summaries and supports
2D power spectrum emulation via a score-based diffusion model. It uses 11 astrophysical
parameters instead of 9.

.. code-block:: python

    from py21cmemu import Emulator

    # Create v3 emulator (without 2D PS for faster predictions)
    emu = Emulator(emulator="mcg", emulate_2d_ps=False)

    # Use 11 parameters for the minihalo model
    import numpy as np
    theta = np.random.rand(11*3).reshape((3, 11))
    theta, output, errors = emu.predict(theta)

    # Access outputs with units (astropy Quantities)
    print(output.Tb.shape)      # Brightness temperature [mK]
    print(output.xHI.shape)     # Neutral fraction [dimensionless]
    print(output.Ts.shape)      # Spin temperature [K]
    print(output.UVLFs.shape)   # UV luminosity functions [dex(Mpc^-3 mag^-1)]
    print(output.tau)           # Optical depth [dimensionless]
    print(output.PS.shape)      # 1D Power spectrum [dex(mK^2)]

The v3 parameter keys are:

.. code-block:: python

    ['F_STAR10', 'ALPHA_STAR', 't_STAR', 'F_ESC10', 'ALPHA_ESC',
     'F_STAR7_MINI', 'F_ESC7_MINI', 'L_X', 'L_X_MINI', 'A_LW', 'NU_X_THRESH']

Output Units
------------

All output quantities are returned as astropy ``Quantity`` objects with units attached:

**Linear quantities** (physical units):

- ``Tb``: Brightness temperature [mK]
- ``xHI``: Neutral hydrogen fraction [dimensionless, 0-1]
- ``Ts``: Spin temperature [K]
- ``tau``: Optical depth [dimensionless]

**Logarithmic quantities** (dex units, i.e., log10 of physical values):

- ``PS``: 1D power spectrum [dex(mK²)] = log10(Δ²)
- ``PS_2D``: 2D power spectrum [dex(mK²)] = log10(Δ²)
- ``UVLFs``: UV luminosity functions [dex(Mpc⁻³ mag⁻¹)] = log10(φ)

To convert log quantities to linear, use ``.physical``:

.. code-block:: python

    # Get power spectrum in linear mK^2
    ps_linear = output.PS.physical

Error Statistics
----------------

Error statistics are accessible via properties like ``output.PS_err``. All errors are
**Fractional Errors (FE%)** computed as:

.. math::

    \\text{FE\\%} = \\frac{|\\text{true} - \\text{predicted}|}{|\\text{true}|} \\times 100

**IMPORTANT**: Power spectrum errors are computed on **log10(PS)**, not linear PS.

A 5% FE on log10(PS) corresponds to approximately 12% error on linear PS, because
a 5% uncertainty in the exponent multiplies the result by 10^0.05 ≈ 1.12.

Available error statistics:

- ``PS_err``: Median FE% on 1D PS log10 values, shape (32 z, 32 k)
- ``PS_2D_err``: Median FE% on 2D PS log10 values, shape (32 kperp, 64 kpar)
- ``PS_2D_var``: Variance of FE% across test set
- ``PS_2D_cov``: Full covariance matrix of errors between pixels

To compute absolute error in log10 units at each pixel:

.. code-block:: python

    abs_err_dex = output.PS_err / 100.0 * output.PS.value

See ``MHEmulatorProperties`` for comprehensive documentation of all error statistics,
aggregation methods (median vs mean), and interpretation guidance.

Continue on to the tutorials to see how to make plots of the output (first tutorial) and how to use ``21cmFAST`` in conjunction with ``21cmEMU`` for analytic calculations of :math:`\\tau_e` and UV luminosity functions.
