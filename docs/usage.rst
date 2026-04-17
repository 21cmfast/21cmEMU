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

Start by cloning the git repository and creating a new ``conda`` environment from the ``yaml`` provided file:

.. code-block:: console

    $ conda env create -f ENV.yml

Then you can install your local installation of the code with:

.. code-block:: console

    $ pip install -e .

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
    emu = Emulator(emulator="mh", emulate_ps=False)

    # Use 11 parameters for the minihalo model
    import numpy as np
    theta = np.random.rand(11*3).reshape((3, 11))
    theta, output, errors = emu.predict(theta)

    # Access outputs
    print(output.Tb.shape)      # Brightness temperature
    print(output.xHI.shape)     # Neutral fraction
    print(output.Ts.shape)      # Spin temperature
    print(output.UVLFs.shape)   # UV luminosity functions
    print(output.tau)           # Optical depth

The v3 parameter keys are:

.. code-block:: python

    ['F_STAR10', 'ALPHA_STAR', 't_STAR', 'F_ESC10', 'ALPHA_ESC',
     'F_STAR7_MINI', 'F_ESC7_MINI', 'L_X', 'L_X_MINI', 'A_LW', 'NU_X_THRESH']

Continue on to the tutorials to see how to make plots of the output (first tutorial) and how to use ``21cmFAST`` in conjunction with ``21cmEMU`` for analytic calculations of :math:`\\tau_e` and UV luminosity functions.
