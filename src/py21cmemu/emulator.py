"""Module that interacts with the Emulator Tensorflow model."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import tensorflow as tf

from .config import CONFIG
from .get_emulator import get_emu_data
from .inputs import EmulatorInput
from .inputs import ParamVecType
from .outputs import EmulatorOutput
from .outputs import RawEmulatorOutput
from .properties import emulator_properties


log = logging.getLogger(__name__)


class Emulator:
    r"""A class that loads an emulator and uses it to obtain 21cmFAST summaries.

    Parameters
    ----------
    version : str, optional
        Emulator version to use/download, default is 'latest'.
    """

    def __init__(self, version: str = "latest"):
        get_emu_data(version=version)

        emu = tf.keras.models.load_model(CONFIG.emu_path, compile=False)

        self.model = emu
        self.inputs = EmulatorInput()
        self.properties = emulator_properties

    def __getattr__(self, name: str) -> Any:
        """Allow access to emulator properties directly from the emulator object."""
        return getattr(self.properties, name)

    def predict(
        self, astro_params: ParamVecType, verbose: bool = False
    ) -> tuple[np.ndarray, EmulatorOutput, dict[str, np.ndarray]]:
        r"""Call the emulator, evaluate it at the given parameters, restore dimensions.

        Parameters
        ----------
        astro_params : np.ndarray or dict
            An array with the nine astro_params input all $\in [0,1]$ OR in the
            21cmFAST AstroParams input units. Dicts (e.g. p21.AstroParams.defining_dict)
            are also accepted formats. Arrays of only dicts are accepted as well
            (for batch evaluation).
        verbose : bool, optional
            If True, prints the emulator prediction.

        Returns
        -------
        theta : np.ndarray
            The normalized parameters used to evaluate the emulator.
        emu : EmulatorOutput
            The emulator output, with dimensions restored.
        errors : dict
            The mean error on the test set (i.e. independent of theta).
        """
        theta = self.inputs.make_param_array(astro_params, normed=True)
        emu = RawEmulatorOutput(self.model.predict(theta, verbose=verbose))
        emu = emu.get_renormalized()

        errors = self.get_errors(emu, theta)

        return theta, emu, errors

    def get_errors(
        self, emu: EmulatorOutput, theta: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Calculate the emulator error on its outputs.

        Parameters
        ----------
        emu : dict
            Dict containing the emulator predictions, defined in Emulator.predict
        theta : dict
            Dict containing the normalized parameters, also defined in Emulator.predict

        Returns
        -------
        The mean error on the test set (i.e. independent of theta) with all units
        restored and logs removed.
        """
        # For now, we return the mean emulator error (obtained from the test set) for
        # each summary. All errors are the median absolute difference between test set
        # and prediction AFTER units have been restored AND log has been removed.
        return {
            "PS_err": self.PS_err,
            "Tb_err": self.Tb_err,
            "xHI_err": self.xHI_err,
            "Ts_err": self.Ts_err,
            "UVLFs_err": self.UVLFs_err,
            "UVLFs_logerr": self.UVLFs_logerr,
            "tau_err": self.tau_err,
        }
