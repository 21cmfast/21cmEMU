# Usage

```{eval-rst}
.. click:: emu21cmfast.__main__:main
    :prog: EMU21cmFAST
    :nested: full
```

## Downloading the emulator

Once you installed the repository with `python setup.py install`, try running the following:

```python
from emulator import EMU21cmFAST
emu_instance = EMU21cmFAST()
```

The second line will create an instance of the emulator class. During this step,
the emulator will be downloaded onto your machine if you did not download it manually beforehand,
so make sure that your machine has internet connection.

Now that we have downloaded the emulator, let's see how we can use it to predict 21cmFAST summaries.

## Emulating 21cmFAST Summaries

The emulator `predict` method accepts two kinds of input parameters:

- Fully normalized parameters i.e. nine numbers $\in [0,1]$. These can either be in a `numpy` array
  in the following order: `['F_STAR10', 'ALPHA_STAR', 'F_ESC10', 'ALPHA_ESC', 'M_TURN', 't_STAR', 'L_X', 'NU_X_THRESH', 'X_RAY_SPEC_INDEX']` or in a dictionary / array of dictionaries with these labels as keys.

- `p21.AstroParams` objects, in an array (i.e. a batch of parameter sets) or a `numpy` array
  with the same units as `p21.AstroParams.defining_dict`.

Do not provide an array with multiple parameter sets where some parameters are in one format and others in the other format.

Let's look at one example, continuing from the previous code snippet:

```python
theta = np.random.rand(9*5).reshape((5,9))

emulator_output = emu_instance.predict(theta)
```

The output here will contain all the summaries i.e. power spectrum, global 21-cm brightness temperature,
IGM spin temperature, neutral fraction, as well as analytically calculated $\tau_e$ and UV LFs.
Setting `emu_only = True` upon creating an instance of the class will skip the analytic calculation of the UV LFs and $\tau_e$.
This will yield a speed-up by about a factor of 10.

Let's make a plot with the results:
