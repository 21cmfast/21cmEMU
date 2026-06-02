"""Module containing functionality for handling emulator inputs.

Input Parameters by Emulator
============================

All parameters can be supplied as dicts (recommended) or as numpy arrays/lists
in the order specified below. The emulator accepts values in 21cmFAST units
(no astropy.units needed).

MCG Emulator (mcg/v3)
---------------------
11 parameters (6 must be provided in log10 space — see ``MHEmulatorInput.LOG_PARAMETERS``):

+---------------+---------------------------+---------+---------------------+
| Key           | Description               | Unit    | Valid Range         |
+===============+===========================+=========+=====================+
| F_STAR10      | Star formation efficiency | log10   | [-2.0, 1.0]         |
|               | at 10^10 M_sun            |         |                     |
+---------------+---------------------------+---------+---------------------+
| ALPHA_STAR    | Power-law index of star   | linear  | [0.0, 1.17]         |
|               | formation efficiency      |         |                     |
+---------------+---------------------------+---------+---------------------+
| t_STAR        | Star formation timescale  | linear  | [0.01, 1.0]         |
|               | in units of Hubble time   |         |                     |
+---------------+---------------------------+---------+---------------------+
| F_ESC10       | Escape fraction at        | log10   | [-3.0, 0.0]         |
|               | 10^10 M_sun               |         |                     |
+---------------+---------------------------+---------+---------------------+
| ALPHA_ESC     | Power-law index of escape | linear  | [-1.0, 1.0]         |
|               | fraction                  |         |                     |
+---------------+---------------------------+---------+---------------------+
| F_STAR7_MINI  | Star formation efficiency | log10   | [-4.0, -1.0]        |
|               | for mini-halos at 10^7 M  |         |                     |
+---------------+---------------------------+---------+---------------------+
| F_ESC7_MINI   | Escape fraction for       | log10   | [-3.0, -1.0]        |
|               | mini-halos at 10^7 M_sun  |         |                     |
+---------------+---------------------------+---------+---------------------+
| L_X           | X-ray luminosity          | log10   | [38, 43] log10(erg  |
|               | per SFR for ACG           | (erg/s  | s^-1 M_sun^-1 yr)   |
|               |                           | /M_sun  |                     |
|               |                           | /yr)    |                     |
+---------------+---------------------------+---------+---------------------+
| L_X_MINI      | X-ray luminosity          | log10   | [39, 44] log10(erg  |
|               | per SFR for MCG           |         | s^-1 M_sun^-1 yr)   |
+---------------+---------------------------+---------+---------------------+
| NU_X_THRESH   | X-ray energy threshold    | eV      | [100, 1500]         |
+---------------+---------------------------+---------+---------------------+
| SIGMA_8       | Amplitude of matter       | linear  | [0.76, 0.85]        |
|               | fluctuations              |         |                     |
+---------------+---------------------------+---------+---------------------+

ACG Emulator (acg/v1)
---------------------
9 parameters for atomic cooling galaxies:

+---------------+---------------------------+---------+---------------------+
| Key           | Description               | Unit    | Valid Range         |
+===============+===========================+=========+=====================+
| F_STAR10      | log10 star formation eff. | log10   | [-3.0, 0.0]         |
+---------------+---------------------------+---------+---------------------+
| ALPHA_STAR    | Power-law index           | linear  | [-0.5, 1.0]         |
+---------------+---------------------------+---------+---------------------+
| F_ESC10       | log10 escape fraction     | log10   | [-3.0, 0.0]         |
+---------------+---------------------------+---------+---------------------+
| ALPHA_ESC     | Escape fraction index     | linear  | [-1.0, 0.5]         |
+---------------+---------------------------+---------+---------------------+
| M_TURN        | log10 turnover halo mass  | log10   | [8.0, 10.0]         |
|               |                           | M_sun   |                     |
+---------------+---------------------------+---------+---------------------+
| t_STAR        | Star formation timescale  | linear  | [0.01, 1.0]         |
+---------------+---------------------------+---------+---------------------+
| L_X           | log10 X-ray luminosity    | log10   | [38, 42]            |
+---------------+---------------------------+---------+---------------------+
| NU_X_THRESH   | X-ray energy threshold    | eV      | [100, 1500]         |
+---------------+---------------------------+---------+---------------------+
| X_RAY_SPEC    | X-ray spectral index      | linear  | [-1.0, 3.0]         |
| _INDEX        |                           |         |                     |
+---------------+---------------------------+---------+---------------------+

Radio Emulator (radio/v2)
-------------------------
5 parameters for radio background:

+---------------+---------------------------+---------+---------------------+
| Key           | Description               | Unit    | Valid Range         |
+===============+===========================+=========+=====================+
| fR_mini       | log10 radio efficiency    | log10   | [-2, 6]             |
+---------------+---------------------------+---------+---------------------+
| L_X_MINI      | log10 X-ray luminosity    | log10   | [33, 45]            |
+---------------+---------------------------+---------+---------------------+
| F_STAR7_MINI  | log10 star formation eff. | log10   | [-5, 0]             |
+---------------+---------------------------+---------+---------------------+
| F_ESC7_MINI   | log10 escape fraction     | log10   | [-6, -1]            |
+---------------+---------------------------+---------+---------------------+
| A_LW          | Lyman-Werner feedback     | linear  | [0, 10]             |
+---------------+---------------------------+---------+---------------------+

Example
-------
>>> from py21cmemu import Emulator
>>> emu = Emulator(emulator="mcg")
>>> params = {
...     'F_STAR10': -1.3,       # log10(f_*,10) — valid range [-2.0, 1.0]
...     'ALPHA_STAR': 0.5,
...     't_STAR': 0.5,
...     'F_ESC10': -1.0,        # log10(f_esc,10) — valid range [-3.0, 0.0]
...     'ALPHA_ESC': 0.0,
...     'F_STAR7_MINI': -3.0,   # log10(f_*,7) — valid range [-4.0, -1.0]
...     'F_ESC7_MINI': -2.0,    # log10(f_esc,7) — valid range [-3.0, -1.0]
...     'L_X': 40.0,            # log10(erg/s/(Msun/yr))
...     'L_X_MINI': 40.0,       # log10(erg/s/(Msun/yr))
...     'NU_X_THRESH': 500,     # eV
...     'SIGMA_8': 0.82,
... }
>>> thetas, output, errors = emu.predict(params)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np

from .properties import emulator_properties

SingleParamVecType = Union[dict[str, float], np.ndarray, Sequence[float]]
ParamVecType = Union[Sequence[SingleParamVecType], SingleParamVecType]


class EmulatorInput:
    """Class for handling emulator inputs."""

    def __init__(self, emulator: str = "default"):
        self.properties = emulator_properties(emulator=emulator)

    def _format_single_theta_vector(self, theta: SingleParamVecType) -> np.ndarray:
        if len(theta) != len(self.astro_param_keys):
            raise ValueError(
                "One of the parameter vectors given is not the correct length. Got "
                f"{len(theta)} but require {len(self.astro_param_keys)}."
            )

        if isinstance(theta, dict):
            return np.array([float(theta[key]) for key in self.astro_param_keys])
        elif isinstance(theta, np.ndarray):
            return theta.astype(float)
        elif isinstance(theta, list):
            return np.array(theta, dtype=float)
        else:
            raise TypeError(
                "astro_params is in the wrong format. Should be a dict of astro params,"
                " list of astro params (in same order as astro_param_keys), or an array"
                " of astro params (in same order as astro_param_keys), OR a sequence of"
                " such."
            )

    def make_param_array(
        self,
        astro_params: ParamVecType,
        normed: bool = True,
        kind: str = "PS",
    ) -> np.ndarray:
        """Format the astro_params input to be a numpy array.

        Parameters
        ----------
        astro_params : dict or np.ndarray
            Input parameters to the emulator, to be reformatted as a numpy array
            compatible with the emulator. Can be a dict of astro params or a list/array
            of floats, in the same order as astro_param_keys. It could also be a list
            or array of such objects, for batch evaluation (the type can be mixed).
        normed : bool, optional
            Whether to return the parameters normalized or not (i.e. between 0 and 1).
        """
        if not hasattr(astro_params, "__len__"):
            raise TypeError(
                "astro_params is in the wrong format. Should be a dict of astro params,"
                " list of astro params (in same order as astro_param_keys), or an array"
                " of astro params (in same order as astro_param_keys), OR a sequence of"
                " such."
            )
        if isinstance(astro_params, dict):
            astro_params = [astro_params]
        else:
            if not hasattr(astro_params[0], "__len__"):
                astro_params = [astro_params]

        theta = np.array(
            [self._format_single_theta_vector(theta) for theta in astro_params]
        )

        if normed:
            return self.normalize(theta, kind=kind)
        return theta

    def make_list_of_dicts(
        self, theta: ParamVecType, normed: bool = True
    ) -> list[dict[str, float]]:
        """Make a list of dicts from a theta array.

        Parameters
        ----------
        theta : np.ndarray
            Input parameters, in any format accepted by :func:`~make_param_array`.
        normed : bool, optional
            Whether to return the parameters normalized or not (i.e. between 0 and 1).

        Returns
        -------
        list of dicts
            List of dicts of astro params, one for each parameter set.
        """
        theta = self.make_param_array(theta, normed=normed)
        return [
            dict(zip(self.astro_param_keys, theta[i], strict=False))
            for i in range(len(theta))
        ]


class DefaultEmulatorInput(EmulatorInput):
    """Class for handling ACG (v1) emulator inputs.

    Parameters in ``LOG_PARAMETERS`` must be supplied as log10 values.
    ``NU_X_THRESH`` is supplied in eV (converted internally to match stored limits).
    """

    #: Ordered mapping of all parameter names to their physical units.
    PARAMETERS: dict[str, str] = {
        "F_STAR10": "log10",
        "ALPHA_STAR": "dimensionless",
        "F_ESC10": "log10",
        "ALPHA_ESC": "dimensionless",
        "M_TURN": "log10(Msun)",
        "t_STAR": "dimensionless",
        "L_X": "log10(erg/s/(Msun/yr))",
        "NU_X_THRESH": "eV",
        "X_RAY_SPEC_INDEX": "dimensionless",
    }

    #: Parameters that must be supplied as log10 values, mapped to their log-space units.
    LOG_PARAMETERS: dict[str, str] = {
        "F_STAR10": "log10",
        "F_ESC10": "log10",
        "M_TURN": "log10(Msun)",
        "L_X": "log10(erg/s/(Msun/yr))",
    }

    def __init__(self):
        self.astro_param_keys = (
            "F_STAR10",
            "ALPHA_STAR",
            "F_ESC10",
            "ALPHA_ESC",
            "M_TURN",
            "t_STAR",
            "L_X",
            "NU_X_THRESH",
            "X_RAY_SPEC_INDEX",
        )
        super().__init__(emulator="default")

    def normalize(self, theta: np.ndarray, kind: str = "PS") -> np.ndarray:
        """Normalize the parameters.

        Parameters
        ----------
        theta : np.ndarray
            Input parameters, strictly in 2D array format, with shape
            (n_batch, n_params).

        Returns
        -------
        np.ndarray
            Normalized parameters, with shape (n_batch, n_params).
        """
        theta_woutdims = theta.copy()
        theta_woutdims[:, 7] /= 1000
        theta_woutdims -= self.properties.limits[:, 0]
        theta_woutdims /= self.properties.limits[:, 1] - self.properties.limits[:, 0]
        return theta_woutdims

    def undo_normalization(self, theta: np.ndarray, kind: str = "PS") -> np.ndarray:
        """Undo the normalization of the parameters.

        Parameters
        ----------
        theta : np.ndarray
            Input parameters, strictly in 2D array format, with shape
            (n_batch, n_params).

        Returns
        -------
        np.ndarray
            Un-normalized parameters, with shape (n_batch, n_params).
        """
        theta_wdims = theta.copy()
        theta_wdims *= self.properties.limits[:, 1] - self.properties.limits[:, 0]
        theta_wdims += self.properties.limits[:, 0]
        theta_wdims[:, 7] *= 1000
        return theta_wdims


class RadioEmulatorInput(EmulatorInput):
    """Class for handling radio background (v2) emulator inputs.

    Parameters in ``LOG_PARAMETERS`` must be supplied as log10 values.
    """

    #: Ordered mapping of all parameter names to their physical units.
    PARAMETERS: dict[str, str] = {
        "fR_mini": "log10",
        "L_X_MINI": "log10(erg/s/(Msun/yr))",
        "F_STAR7_MINI": "log10",
        "F_ESC7_MINI": "log10",
        "A_LW": "dimensionless",
    }

    #: Parameters that must be supplied as log10 values, mapped to their log-space units.
    LOG_PARAMETERS: dict[str, str] = {
        "fR_mini": "log10",
        "L_X_MINI": "log10(erg/s/(Msun/yr))",
        "F_STAR7_MINI": "log10",
        "F_ESC7_MINI": "log10",
    }

    def __init__(self):
        self.astro_param_keys = (
            "fR_mini",
            "L_X_MINI",
            "F_STAR7_MINI",
            "F_ESC7_MINI",
            "A_LW",
        )
        super().__init__(emulator="radio_background")

    def normalize(self, theta: np.ndarray, kind: str = "PS") -> np.ndarray:
        """Normalize the parameters.

        Parameters
        ----------
        theta : np.ndarray
            Input parameters, strictly in 2D array format, with shape
            (n_batch, n_params).

        Returns
        -------
        np.ndarray
            Normalized parameters, with shape (n_batch, n_params).
        """
        theta_woutdims = theta.copy()
        theta_woutdims -= self.properties.limits[:, 0]
        theta_woutdims /= self.properties.limits[:, 1] - self.properties.limits[:, 0]
        return theta_woutdims

    def undo_normalization(self, theta: np.ndarray, kind: str = "PS") -> np.ndarray:
        """Undo the normalization of the parameters.

        Parameters
        ----------
        theta : np.ndarray
            Input parameters, strictly in 2D array format, with shape
            (n_batch, n_params).

        Returns
        -------
        np.ndarray
            Un-normalized parameters, with shape (n_batch, n_params).
        """
        theta_wdims = theta.copy()
        theta_wdims *= self.properties.limits[:, 1] - self.properties.limits[:, 0]
        theta_wdims += self.properties.limits[:, 0]
        return theta_wdims


class MHEmulatorInput(EmulatorInput):
    """Class for handling MCG/minihalo (v3) emulator inputs.

    Parameters in ``LOG_PARAMETERS`` must be supplied as log10 values.
    ``NU_X_THRESH`` is supplied in eV. All other parameters are linear.
    """

    #: Ordered mapping of all parameter names to their physical units.
    PARAMETERS: dict[str, str] = {
        "F_STAR10": "log10",
        "ALPHA_STAR": "dimensionless",
        "t_STAR": "dimensionless",
        "F_ESC10": "log10",
        "ALPHA_ESC": "dimensionless",
        "F_STAR7_MINI": "log10",
        "F_ESC7_MINI": "log10",
        "L_X": "log10(erg/s/(Msun/yr))",
        "L_X_MINI": "log10(erg/s/(Msun/yr))",
        "NU_X_THRESH": "eV",
        "SIGMA_8": "dimensionless",
    }

    #: Parameters that must be supplied as log10 values, mapped to their log-space units.
    LOG_PARAMETERS: dict[str, str] = {
        "F_STAR10": "log10",
        "F_ESC10": "log10",
        "F_STAR7_MINI": "log10",
        "F_ESC7_MINI": "log10",
        "L_X": "log10(erg/s/(Msun/yr))",
        "L_X_MINI": "log10(erg/s/(Msun/yr))",
    }

    def __init__(self):
        self.astro_param_keys = tuple(emulator_properties("mh").astro_param_keys)
        super().__init__(emulator="mh")

    def normalize(self, theta: np.ndarray, kind: str = "summaries") -> np.ndarray:
        theta_out = theta.copy().astype(float)
        if kind.upper() in ("LSTM", "SUMMARIES"):
            limits = self.properties.lstm_limits[:-1]
        elif kind.upper() in ("PS", "PS_2D"):
            limits = self.properties.ps_limits[:-1]
        else:
            raise ValueError(
                f"Unknown kind '{kind}'. Use 'summaries', 'LSTM', 'PS', or 'PS_2D'."
            )

        theta_out = (theta_out - limits[:, 0]) / (limits[:, 1] - limits[:, 0])
        return np.clip(theta_out, 0.0, 1.0)

    def undo_normalization(
        self, theta: np.ndarray, kind: str = "summaries"
    ) -> np.ndarray:
        theta_out = theta.copy().astype(float)
        if kind.upper() in ("LSTM", "SUMMARIES"):
            limits = self.properties.lstm_limits[:-1]
        elif kind.upper() in ("PS", "PS_2D"):
            limits = self.properties.ps_limits[:-1]
        else:
            raise ValueError(
                f"Unknown kind '{kind}'. Use 'summaries', 'LSTM', 'PS', or 'PS_2D'."
            )

        theta_out = theta_out * (limits[:, 1] - limits[:, 0]) + limits[:, 0]
        return theta_out

    def format_theta_for_ps(
        self, theta: np.ndarray, ps_redshifts: np.ndarray
    ) -> np.ndarray:
        z_min, z_max = self.properties.ps_limits[-1]
        normed_redshifts = (ps_redshifts - z_min) / (z_max - z_min)

        n_samples = theta.shape[0]
        n_z = len(ps_redshifts)
        theta_rep = np.repeat(theta, n_z, axis=0)
        z_rep = np.tile(normed_redshifts, n_samples)
        return np.column_stack([theta_rep, z_rep])

    def format_theta_for_summaries(
        self, theta: np.ndarray, redshifts: np.ndarray | None = None
    ) -> np.ndarray:
        """Format normalized parameters for the summaries LSTM model.

        Creates 3D array with redshifts appended, as expected by the LSTM model.
        Shape: (n_samples, n_redshifts, n_params + 1)

        Parameters
        ----------
        theta : np.ndarray
            The normalized parameters, with shape (n_samples, n_params).
            Should already be normalized using normalize(kind="summaries").
        redshifts : np.ndarray, optional
            The redshifts to use. If None, uses self.properties.redshifts.

        Returns
        -------
        np.ndarray
            The formatted theta array, with shape (n_samples, n_redshifts, n_params + 1).
        """
        if redshifts is None:
            redshifts = self.properties.redshifts

        # Normalize redshifts using LSTM model limits
        z_min, z_max = self.properties.lstm_limits[-1]
        normed_redshifts = (redshifts - z_min) / (z_max - z_min)

        n_samples = theta.shape[0]
        n_z = len(redshifts)

        # Replicate parameters across redshift dimension
        theta_3d = np.repeat(
            theta[:, np.newaxis, :], n_z, axis=1
        )  # (n_samples, n_z, n_params)

        # Create redshift array and tile for all samples
        z_array = np.tile(
            normed_redshifts[np.newaxis, :, np.newaxis], (n_samples, 1, 1)
        )  # (n_samples, n_z, 1)

        # Append redshift as last column
        theta_with_z = np.concatenate(
            [theta_3d, z_array], axis=-1
        )  # (n_samples, n_z, n_params + 1)

        return theta_with_z

    def format_theta(self, theta: np.ndarray, ps_redshifts: np.ndarray) -> np.ndarray:
        return self.format_theta_for_ps(theta, ps_redshifts)
