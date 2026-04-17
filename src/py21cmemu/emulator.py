"""Module that interacts with the emulator models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import CONFIG
from .get_emulator import get_emu_data
from .inputs import DefaultEmulatorInput
from .inputs import MHEmulatorInput
from .inputs import ParamVecType
from .inputs import RadioEmulatorInput
from .outputs import DefaultRawEmulatorOutput
from .outputs import EmulatorOutput
from .outputs import MHRawEmulatorOutput
from .outputs import RadioRawEmulatorOutput
from .properties import get_emulator_properties
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

    def __init__(
        self,
        emulator: str = "default",
        version: str = "latest",
        emulate_ps: bool = True,
        model_path: str | None = None,
        PS_scale: float | None = None,
        PS_bias: float | None = None,
    ):

        self.which_emulator = emulator
        self.emulate_ps = emulate_ps
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.which_emulator == "default":
            get_emu_data(version=version)
            
            # Load bundled PyTorch model
            from .models.default.v1_pytorch import load_converted_model
            here = Path(__file__).parent
            model_path = here / "models/default/default_model.pt"
            model = load_converted_model(str(model_path), self.device)
            self.inputs = DefaultEmulatorInput()

        elif self.which_emulator == "radio_background":
            from .models.radio_background.model import Radio_Emulator

            here = Path(__file__).parent
            model = Radio_Emulator()
            model.load_state_dict(
                torch.load(
                    here / "models/radio_background/Radio_Background_Emu_Weights",
                    map_location=self.device,
                ),
            )
            model.to(self.device)
            model.eval()
            self.inputs = RadioEmulatorInput()

        elif self.which_emulator == "mh":
            from .models.MHs.lstm_model import MH_Emulator
            from .models.MHs.score_model import UNet
            from .sample_pytorch import GetEMSampler, GetODESampler
            from .sde import VPSDE

            self.properties = get_emulator_properties(emulator="mh")
            here = Path(__file__).parent

            # Model config matching production_ema training
            lstm_model = MH_Emulator({
                "n_params": 11,
                "N_z": 93,
                "N_LF_z": 7,
                "N_mag": 45,
                "N_PS_Z": 32,
                "N_PS_K": 32,
            })
            lstm_model.load_state_dict(
                torch.load(here / "models/MHs/lstm_model_weights.pt", map_location=self.device)
            )
            lstm_model.to(self.device)
            lstm_model.eval()
            self.lstm_model = lstm_model

            self.score_model = None
            self.sample = None
            if self.emulate_ps:
                ps_model_path = (
                    Path(model_path)
                    if model_path is not None
                    else here / "models/MHs/score_model_weights.pt"
                )
                score_model = UNet(
                    dim=(32, 64),
                    init_dim=48,
                    channels=1,
                    dim_mults=(1, 2, 4, 8, 16),
                    cdn_len=12,
                )
                score_model.load_state_dict(
                    torch.load(ps_model_path, map_location=self.device)
                )
                score_model.to(self.device)
                score_model.eval()
                self.score_model = score_model

                self.ps_bias = self.properties.PS_bias if PS_bias is None else PS_bias
                self.ps_scale = self.properties.PS_scale if PS_scale is None else PS_scale
                self._vpsde_cls = VPSDE
                self._em_sampler_cls = GetEMSampler
                self._ode_sampler_cls = GetODESampler

            self.inputs = MHEmulatorInput()
            model = self.lstm_model

        else:
            raise ValueError(
                "Please supply one of the following emulator names:"
                + "'default', 'radio_background' or 'mh'. "
                + f"{emulator} is not a valid emulator name."
            )

        self.model = model
        if self.which_emulator != "mh":
            self.properties = emulator_properties(emulator=emulator)

    def __getattr__(self, name: str) -> Any:
        """Allow access to emulator properties directly from the emulator object."""
        return getattr(self.properties, name)

    def predict(
        self,
        astro_params: ParamVecType,
        verbose: bool = False,
        ps_redshifts: np.ndarray | None = None,
        n_ps_batch: int | None = None,
        num_ps_samples: int = 100,
        sde: Any | None = None,
        denoise: bool = True,
        ps_sampling_method: str = "em",
    ) -> tuple[np.ndarray | tuple[np.ndarray | None, np.ndarray], EmulatorOutput, dict[str, np.ndarray]]:
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
        ps_redshifts : np.ndarray, optional
            Redshifts at which to evaluate the 2D PS (for 'mh' emulator only).
        n_ps_batch : int, optional
            Batch size for PS sampling.
        num_ps_samples : int, optional
            Number of samples per conditioning (default: 100).
        sde : VPSDE, optional
            SDE object for diffusion sampling.
        denoise : bool, optional
            Whether to apply final denoising step.
        ps_sampling_method : str, optional
            Sampling method for 2D PS: 'em' (Euler-Maruyama, default) or 'ode' 
            (probability flow ODE). 'em' is stochastic and recommended for
            uncertainty quantification. 'ode' is deterministic.

        Returns
        -------
        theta : np.ndarray
            The normalized parameters used to evaluate the emulator.
        emu : EmulatorOutput
            The emulator output, with dimensions restored.
        errors : dict
            The mean error on the test set (i.e. independent of theta).
        """
        if self.which_emulator == "default":
            theta = self.inputs.make_param_array(astro_params, normed=True)
            # PyTorch inference
            theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                raw = self.model.forward_stacked(theta_t).detach().cpu().numpy()
            emu = DefaultRawEmulatorOutput(raw)
            emu = emu.get_renormalized()
            errors = self.get_errors(emu, theta)
            return theta, emu, errors

        if self.which_emulator == "radio_background":
            theta = self.inputs.make_param_array(astro_params, normed=True)
            emu = RadioRawEmulatorOutput(
                self.model(torch.tensor(theta, dtype=torch.float32, device=self.device))
                .detach()
                .cpu()
                .numpy()
            )
            emu = emu.get_renormalized()
            errors = self.get_errors(emu, theta)
            return theta, emu, errors

        theta_PS = (
            self.inputs.make_param_array(astro_params, normed=True, kind="PS")
            if self.emulate_ps
            else None
        )
        theta_LSTM = self.inputs.make_param_array(astro_params, normed=True, kind="LSTM")

        # Pass raw params to model - model handles formatting internally
        theta_lstm_t = torch.tensor(theta_LSTM, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predicted = list(self.lstm_model(theta_lstm_t))

        if self.emulate_ps:
            if ps_redshifts is None:
                ps_redshifts = self.properties.default_ps_redshifts
            n_zs = len(ps_redshifts)
            n_params = theta_PS.shape[0]
            theta_sbm = self.inputs.format_theta(theta_PS, ps_redshifts)
            if n_ps_batch is None:
                n_ps_batch = theta_sbm.shape[0]
            if sde is None:
                sde = self._vpsde_cls(beta_min=0.1, beta_max=20.0)

            # Select sampling method
            ps_sampling_method = ps_sampling_method.lower()
            if ps_sampling_method not in ("em", "ode"):
                raise ValueError(
                    f"ps_sampling_method must be 'em' or 'ode', got '{ps_sampling_method}'"
                )
            
            if ps_sampling_method == "em":
                self.sample = self._em_sampler_cls(
                    sde,
                    (n_ps_batch, num_ps_samples, 32, 64),
                    device=self.device,
                    denoise=denoise,
                ).get_em_sampler()
            else:  # ode
                self.sample = self._ode_sampler_cls(
                    sde,
                    (n_ps_batch, num_ps_samples, 32, 64),
                    device=self.device,
                    denoise=denoise,
                    rtol=1e-5,
                    atol=1e-5,
                ).get_ode_sampler()
            
            self._current_ps_method = ps_sampling_method

            theta_sbm = theta_sbm.reshape(
                (theta_sbm.shape[0] // n_ps_batch, n_ps_batch, theta_sbm.shape[1])
            )
            samples_pred = self.get_pred(theta_sbm)
            samples_pred = samples_pred.reshape(
                (theta_sbm.shape[0] * n_ps_batch, num_ps_samples, 32, 64)
            ).reshape((n_params, n_zs, num_ps_samples, 32, 64))
            predicted.extend([samples_pred, ps_redshifts])
        else:
            self._current_ps_method = None
            predicted.extend([None, None])

        emu = MHRawEmulatorOutput(predicted)
        emu = emu.get_renormalized()
        errors = self.get_errors(emu, theta_LSTM, theta_PS, ps_sampling_method)
        return (theta_PS, theta_LSTM), emu, errors

    @torch.no_grad()
    def get_pred_single(self, cdn: torch.Tensor) -> np.ndarray:
        from .utils import reverse_transform

        cdn = cdn.to(self.device)
        samples = self.sample(self.score_model, cdn=cdn, progress=False).cpu().detach()
        samples_w_units = reverse_transform(samples, self.ps_scale, self.ps_bias)
        return samples_w_units.cpu().detach().numpy()

    @torch.no_grad()
    def get_pred(self, cdns: np.ndarray) -> np.ndarray:
        all_preds = []
        for i in range(cdns.shape[0]):
            samples = self.get_pred_single(torch.tensor(cdns[i], dtype=torch.float32))
            all_preds.append(samples)
        return np.array(all_preds)

    def get_errors(
        self,
        emu: EmulatorOutput,
        theta_lstm: np.ndarray | None = None,
        theta_ps: np.ndarray | None = None,
        ps_sampling_method: str | None = None,
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
        elif self.which_emulator == "radio_background":
            return {
                "PS_err": self.PS_err,
                "Tb_err": self.Tb_err,
                "xHI_err": self.xHI_err,
                "Tr_err": self.Tr_err,
                "tau_err": self.tau_err,
            }

        m = np.logical_and(self.properties.UVLFs_MUVs <= -10, self.properties.UVLFs_MUVs >= -20)
        
        # Select method-specific PS error if available and method specified
        if ps_sampling_method == "ode":
            ps_med_err = self.properties.PS_med_err_ode
        else:  # default to EM
            ps_med_err = self.properties.PS_med_err_em
        
        # Handle PS error - may have shape mismatch between LSTM PS and score model error
        if emu.PS is not None:
            try:
                ps_err = ps_med_err / 100.0 * emu.PS
            except ValueError:
                # Shape mismatch - use scalar median error instead
                ps_err = np.nanmedian(ps_med_err) / 100.0 * emu.PS
        else:
            ps_err = np.nan
        
        return {
            "PS_err": ps_err,
            "Tb_err": self.Tb_med_err / 100.0 * np.abs(emu.Tb),
            "xHI_err": self.xHI_med_err / 100.0 * emu.xHI,
            "Ts_err": self.Ts_med_err / 100.0 * emu.Ts,
            "UVLFs_err": np.swapaxes(self.UVLFs_med_err[m] / 100.0, 1, 0) * 10 ** emu.UVLFs,
            "UVLFs_logerr": np.swapaxes(self.UVLFs_med_logerr[m] / 100.0, 1, 0) * np.abs(emu.UVLFs),
            "tau_err": self.tau_med_err / 100.0 * emu.tau,
        }
