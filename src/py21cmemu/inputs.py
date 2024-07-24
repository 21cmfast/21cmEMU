"""Module containing functionality for handling emulator inputs."""

from __future__ import annotations

from typing import Dict
from typing import Sequence
from typing import Union

import numpy as np

from .properties import emulator_properties


SingleParamVecType = Union[Dict[str, float], np.ndarray, Sequence[float]]
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

        params_normed = theta.min() >= 0 and theta.max() <= 1
        if (params_normed and normed) or (not params_normed and not normed):
            return theta
        elif params_normed:
            return self.undo_normalization(theta)
        else:
            return self.normalize(theta)

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
        return [dict(zip(self.astro_param_keys, theta[i])) for i in range(len(theta))]


class DefaultEmulatorInput(EmulatorInput):
    """Class for handling emulator inputs."""

    def __init__(self):
        """Class for handling emulator inputs."""
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

    def normalize(self, theta: np.ndarray) -> np.ndarray:
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

    def undo_normalization(self, theta: np.ndarray) -> np.ndarray:
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
    """Class for handling radio background emulator inputs."""

    def __init__(self):
        self.astro_param_keys = (
            "fR_mini",
            "L_X_MINI",
            "F_STAR7_MINI",
            "F_ESC7_MINI",
            "A_LW",
        )
        super().__init__(emulator="radio_background")

    def normalize(self, theta: np.ndarray) -> np.ndarray:
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

    def undo_normalization(self, theta: np.ndarray) -> np.ndarray:
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
