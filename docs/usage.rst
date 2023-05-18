Quickstart
==========

Installation
------------

For users
^^^^^^^^^

To use ``21cmEMU``, install it with pip:

.. code-block:: console

    $ pip install py21cmemu

For developers
^^^^^^^^^^^^^^

Start by cloning the git repository and creating a new ``conda`` environment from the ``yaml`` file:

.. code-block:: console

    $ conda env create -f env.yml

Then you can install your local installation of the code with:

.. code-block:: console

    $ pip install -e .

To test the installation, run:

.. code-block:: python

    from py21cmemu import Emulator
    emu_instance = Emulator()

Emulating 21cmFAST Summaries
----------------------------

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

Continue on to the tutorials to see how to make plots of the output (first tutorial) and how to use ``21cmFAST`` in conjunction with ``21cmEMU`` for analytic calculations of :math:`\\tau_e` and UV luminosity functions.
