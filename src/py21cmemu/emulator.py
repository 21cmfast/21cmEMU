"""Module that interacts with the Emulator Tensorflow model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import CONFIG
from .get_emulator import get_emu_data
from .inputs import DefaultEmulatorInput
from .inputs import ParamVecType
from .inputs import RadioEmulatorInput
from .outputs import DefaultRawEmulatorOutput
from .outputs import EmulatorOutput
from .outputs import RadioRawEmulatorOutput
from .properties import emulator_properties


log = logging.getLogger(__name__)


class Emulator:
    r"""A class that loads an emulator and uses it to obtain 21cmFAST summaries.

    Parameters
    ----------
    version : str, optional
        Emulator version to use/download, default is 'latest'.
    emulator : str, optional
        Emulator to use. Options are: 'radio_background' and 'default'.
        The radio background emulator is the emulator used in Cang+24
        It is a model that predicts the radio background
        temperature :math:`T_{\rm r} \rm{[K]}`,
        the global IGM neutral fraction :math:`\overline{x}_{\rm HI}`,
        the global 21-cm brightness temperature :math:`T{\rm b} \rm{[mK]}`,
        the 21-cm spherically-averaged power spectrum :math:`P(k) \rm{[mK^2]}`, and
        the Thomson scattering optical depth :math:`\tau`.
        It has five input parameters:
        ["fR_mini", "L_X_MINI",  "F_STAR7_MINI", "F_ESC7_MINI", "A_LW"]
        See 21cmFAST documentation for more information about the input parameters.

        The default emulator is the emulator described in Breitman+23.
        It emulates six summary statistics with 9 input astrophysical parameters.
    """

    def __init__(self, emulator: str = "default", version: str = "latest"):

        self.which_emulator = emulator
        if self.which_emulator == "default":
            import tensorflow as tf

            get_emu_data(version=version)
            model = tf.keras.models.load_model(CONFIG.emu_path, compile=False)
            self.inputs = DefaultEmulatorInput()

        elif self.which_emulator == "radio_background":
            import torch

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.set_default_device(device)
            from .models.radio_background.model import Radio_Emulator

            here = Path(__file__).parent
            model = Radio_Emulator()
            model.load_state_dict(
                torch.load(
                    here / "models/radio_background/Radio_Background_Emu_Weights",
                    map_location=device,
                ),
            )
            model.eval()
            self.inputs = RadioEmulatorInput()

        else:
            raise ValueError(
                "Please supply one of the following emulator names:"
                + "'default' or 'radio_background'. "
                + f"{emulator} is not a valid emulator name."
            )

        self.model = model
        self.properties = emulator_properties(emulator=emulator)

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
        if self.which_emulator == "default":
            emu = DefaultRawEmulatorOutput(self.model.predict(theta, verbose=verbose))
        if self.which_emulator == "radio_background":
            import torch

            emu = RadioRawEmulatorOutput(
                self.model(torch.Tensor(theta)).detach().cpu().numpy()
            )

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
        if self.which_emulator == "default":
            return {
                "PS_err": self.PS_err,
                "Tb_err": self.Tb_err,
                "xHI_err": self.xHI_err,
                "Ts_err": self.Ts_err,
                "UVLFs_err": self.UVLFs_err,
                "UVLFs_logerr": self.UVLFs_logerr,
                "tau_err": self.tau_err,
            }
        else:
            return {
                "PS_err": self.PS_err,
                "Tb_err": self.Tb_err,
                "xHI_err": self.xHI_err,
                "Tr_err": self.Tr_err,
                "tau_err": self.tau_err,
            }
