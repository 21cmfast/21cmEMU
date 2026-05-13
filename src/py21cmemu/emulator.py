"""Module that interacts with the emulator models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import torch

from .inputs import DefaultEmulatorInput
from .inputs import MHEmulatorInput
from .inputs import ParamVecType
from .inputs import RadioEmulatorInput
from .outputs import ACGEmulatorErrors
from .outputs import DefaultRawEmulatorOutput
from .outputs import EmulatorOutput
from .outputs import MHEmulatorErrors
from .outputs import MHRawEmulatorOutput
from .outputs import RadioEmulatorErrors
from .outputs import RadioRawEmulatorOutput
from .properties import DEFAULT_EMULATOR
from .properties import EMULATOR_ACG
from .properties import EMULATOR_MCG
from .properties import EMULATOR_RADIO
from .properties import emulator_properties
from .properties import get_emulator_properties
from .properties import resolve_emulator_name


if TYPE_CHECKING:
    pass


log = logging.getLogger(__name__)


class Emulator:
    r"""A class that loads an emulator and uses it to obtain 21cmFAST summaries.

    Parameters
    ----------
    emulator : str, optional
        Which emulator to use. Default is 'mcg' (v3).

        Available emulators:

        +-----------+---------+-------------+--------+-----------------------------+
        | Name      | Aliases | Paper       | Params | Outputs                     |
        +===========+=========+=============+========+=============================+
        | ``mcg``   | v3, mh  | [upcoming]  | 11     | Tb, xHI, Ts, tau, 2D-PS,    |
        |           |         |             |        | UVLFs                       |
        +-----------+---------+-------------+--------+-----------------------------+
        | ``acg``   | v1      | Breitman+24 | 9      | Tb, xHI, Ts, tau, 1D-PS,    |
        |           |         |             |        | UVLFs                       |
        +-----------+---------+-------------+--------+-----------------------------+
        | ``radio`` | v2      | Cang+24     | 5      | Tb, xHI, Tr, tau, 1D-PS     |
        +-----------+---------+-------------+--------+-----------------------------+

        - **mcg** (Molecular Cooling Galaxies): Full 11-parameter emulator
          including mini-halos/molecular cooling galaxies. Predicts 2D power
          spectrum P(k_perp, k_par) using a score-based diffusion model.

        - **acg** (Atomic Cooling Galaxies): Original 9-parameter emulator
          for atomic cooling galaxies only. Predicts 1D spherically-averaged
          power spectrum P(k).

        - **radio**: Radio background emulator with 5 parameters. Predicts
          radio temperature Tr instead of spin temperature Ts.

    emulate_2d_ps : bool, optional
        Whether to emulate the 2D power spectrum (for 'mcg' only). Default is False
        since the 2D PS score model is slower than the LSTM. When False, the 1D PS
        from the LSTM model is used.
    model_path : str, optional
        Custom path to model weights (for 'mcg' score model only).
    PS_log_std : float, optional
        Custom PS normalization std (for 'mcg' only; log10 space).
    PS_log_mean : float, optional
        Custom PS normalization mean (for 'mcg' only; log10 space).
    PS_log_mean : float, optional
        Custom PS normalization mean (for 'mcg' only; log10 space).
    """

    def __init__(
        self,
        emulator: str = DEFAULT_EMULATOR,
        emulate_2d_ps: bool = False,
    ):

        self.which_emulator = resolve_emulator_name(emulator)
        self.emulate_2d_ps = emulate_2d_ps
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.which_emulator == EMULATOR_ACG:
            # Load bundled PyTorch model (no download needed)
            from .models.default.v1_pytorch import load_converted_model

            here = Path(__file__).parent
            model_path = here / "models/default/default_model.pt"
            model = load_converted_model(str(model_path), self.device)
            self.inputs = DefaultEmulatorInput()

        elif self.which_emulator == EMULATOR_RADIO:
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

        elif self.which_emulator == EMULATOR_MCG:
            from .models.MHs.lstm_model import MH_Emulator
            from .models.MHs.score_model import UNet
            from .sample_pytorch import GetEMSampler
            from .sample_pytorch import GetODESampler
            from .sde import VPSDE

            self.properties = get_emulator_properties(emulator=EMULATOR_MCG)
            here = Path(__file__).parent

            # Model config matching production_ema training
            lstm_model = MH_Emulator(
                {
                    "n_params": 11,
                    "N_z": 93,
                    "N_LF_z": 7,
                    "N_mag": 45,
                    "N_PS_Z": 32,
                    "N_PS_K": 32,
                }
            )
            lstm_model.load_state_dict(
                torch.load(
                    here / "models/MHs/lstm_model_weights.pt", map_location=self.device
                )
            )
            lstm_model.to(self.device)
            lstm_model.eval()
            self.lstm_model = lstm_model

            self.score_model = None
            self.sample = None
            if self.emulate_2d_ps:
                ps_model_path = here / "models/MHs/score_model_weights.pt"

                score_model = UNet(
                    dim=(32, 64),
                    init_dim=48,
                    channels=1,
                    dim_mults=(1, 2, 4, 8, 16),
                    cdn_len=12,
                )
                # Weights are stored in float16 to stay under GitHub's 100MB limit.
                score_model.load_state_dict(
                    torch.load(
                        ps_model_path, map_location=self.device, weights_only=True
                    )
                )
                score_model.to(self.device)
                score_model.eval()
                self.score_model = score_model

                self.ps_log_mean = self.properties.PS_log_mean
                self.ps_log_std = self.properties.PS_log_std
                # Backward compatibility aliases
                self.ps_bias = self.ps_log_mean
                self.ps_scale = self.ps_log_std
                self._vpsde_cls = VPSDE
                self._em_sampler_cls = GetEMSampler
                self._ode_sampler_cls = GetODESampler

            self.inputs = MHEmulatorInput()
            model = self.lstm_model

        self.model = model
        if self.which_emulator != EMULATOR_MCG:
            self.properties = emulator_properties(emulator=self.which_emulator)

    def __getattr__(self, name: str) -> Any:
        """Allow access to emulator properties directly from the emulator object."""
        return getattr(self.properties, name)

    def predict(
        self,
        astro_params: ParamVecType,
        verbose: bool = False,
        ps_2d_redshifts: np.ndarray | None = None,
        n_ps_batch: int | None = None,
        n_lstm_batch: int | None = None,
        n_realisations: int = 100,
        sde: Any | None = None,
        denoise: bool = True,
        ps_sampling_method: str = "ode",
    ) -> tuple[
        np.ndarray | tuple[np.ndarray | None, np.ndarray],
        EmulatorOutput,
        dict[str, np.ndarray],
    ]:
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
        ps_2d_redshifts : np.ndarray, optional
            Redshifts at which to evaluate the 2D PS (for 'mcg' emulator only).
        n_ps_batch : int, optional
            Batch size for PS sampling.
        n_lstm_batch : int, optional
            Batch size for LSTM inference (for 'mcg' emulator only). If None,
            all parameter sets are evaluated in a single forward pass. Use this
            to avoid OOM errors when evaluating many parameter sets at once.
        n_realisations : int, optional
            Number of diffusion model realisations per redshift (default: 100).
            More realisations give better uncertainty estimates but take longer.
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
        if self.which_emulator == EMULATOR_ACG:
            theta = self.inputs.make_param_array(astro_params, normed=True)
            # PyTorch inference
            theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                raw = self.model.forward_stacked(theta_t).detach().cpu().numpy()
            emu = DefaultRawEmulatorOutput(raw)
            emu = emu.get_renormalized()
            errors = self.get_errors(emu, theta)
            return theta, emu, errors

        if self.which_emulator == EMULATOR_RADIO:
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
            if self.emulate_2d_ps
            else None
        )
        theta_LSTM = self.inputs.make_param_array(
            astro_params, normed=True, kind="LSTM"
        )

        # Pass raw params to model - model handles formatting internally
        theta_lstm_t = torch.tensor(theta_LSTM, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if n_lstm_batch is None or n_lstm_batch >= theta_lstm_t.shape[0]:
                predicted = list(self.lstm_model(theta_lstm_t))
            else:
                chunks = torch.split(theta_lstm_t, n_lstm_batch, dim=0)
                batch_outputs = [self.lstm_model(chunk) for chunk in chunks]
                predicted = [
                    torch.cat([b[i] for b in batch_outputs], dim=0)
                    for i in range(len(batch_outputs[0]))
                ]

        if self.emulate_2d_ps:
            if ps_2d_redshifts is None:
                ps_2d_redshifts = self.properties.default_ps_redshifts
            n_zs = len(ps_2d_redshifts)
            n_params = theta_PS.shape[0]
            theta_sbm = self.inputs.format_theta(theta_PS, ps_2d_redshifts)
            if n_ps_batch is None:
                n_ps_batch = theta_sbm.shape[0]
            if sde is None:
                sde = self._vpsde_cls(beta_min=0.1, beta_max=20.0)

            # Select sampling method
            ps_sampling_method = ps_sampling_method.lower()
            if ps_sampling_method not in ("em", "ode"):
                raise ValueError(
                    f"ps_sampling_method must be 'em' or 'ode',"
                    f" got {ps_sampling_method!r}"
                )

            if ps_sampling_method == "em":
                self.sample = self._em_sampler_cls(
                    sde,
                    (n_ps_batch, n_realisations, 32, 64),
                    device=self.device,
                    denoise=denoise,
                ).get_em_sampler()
            else:  # ode
                self.sample = self._ode_sampler_cls(
                    sde,
                    (n_ps_batch, n_realisations, 32, 64),
                    device=self.device,
                    denoise=denoise,
                    rtol=1e-5,
                    atol=1e-5,
                ).get_ode_sampler()

            self._current_ps_method = ps_sampling_method
            self._n_realisations = n_realisations
            self._denoise = denoise

            theta_sbm = theta_sbm.reshape(
                (theta_sbm.shape[0] // n_ps_batch, n_ps_batch, theta_sbm.shape[1])
            )
            samples_pred = self.get_pred(theta_sbm, verbose=verbose)
            samples_pred = samples_pred.reshape(
                (theta_sbm.shape[0] * n_ps_batch, n_realisations, 32, 64)
            ).reshape((n_params, n_zs, n_realisations, 32, 64))
            predicted.extend([samples_pred, ps_2d_redshifts])
        else:
            self._current_ps_method = None
            predicted.extend([None, None])

        emu = MHRawEmulatorOutput(predicted)
        emu = emu.get_renormalized()
        errors = self.get_errors(emu, theta_LSTM, theta_PS, ps_sampling_method)
        return (theta_PS, theta_LSTM), emu, errors

    @torch.no_grad()
    def get_pred_single(self, cdn: torch.Tensor) -> np.ndarray:
        """Run one diffusion-model forward pass on a single conditioning batch."""
        from .utils import reverse_transform

        cdn = cdn.to(self.device)
        samples = self.sample(self.score_model, cdn=cdn).cpu().detach()
        samples_w_units = reverse_transform(samples, self.ps_log_std, self.ps_log_mean)
        return samples_w_units.cpu().detach().numpy()

    @torch.no_grad()
    def get_pred(self, cdns: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Run diffusion-model sampling, using multiple GPUs when available."""
        from .utils import reverse_transform

        n_gpus = torch.cuda.device_count()

        # Single-GPU (or CPU) path – original sequential behaviour.
        if n_gpus <= 1:
            all_preds = []
            iterator = range(cdns.shape[0])
            if verbose:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Computing 2D PS", unit="batch")
            for i in iterator:
                samples = self.get_pred_single(
                    torch.tensor(cdns[i], dtype=torch.float32)
                )
                all_preds.append(samples)
            return np.array(all_preds)

        # Multi-GPU path: distribute outer-batch iterations across all GPUs.
        # Threads give true parallelism because PyTorch releases the GIL
        # during CUDA kernel execution.
        from concurrent.futures import ThreadPoolExecutor

        n_ps_batch = cdns.shape[1]
        n_realisations = getattr(self, "_n_realisations", 100)
        denoise = getattr(self, "_denoise", True)
        config_key = (self._current_ps_method, n_ps_batch, n_realisations, denoise)

        if getattr(self, "_ps_gpu_config_key", None) != config_key:
            sde = self._vpsde_cls(beta_min=0.1, beta_max=20.0)
            replicas = []
            for gpu_id in range(n_gpus):
                device = torch.device(f"cuda:{gpu_id}")
                if gpu_id == 0:
                    model = self.score_model
                else:
                    from .models.MHs.score_model import UNet

                    here = Path(__file__).parent
                    model = UNet(
                        dim=(32, 64),
                        init_dim=48,
                        channels=1,
                        dim_mults=(1, 2, 4, 8, 16),
                        cdn_len=12,
                    )
                    model.load_state_dict(
                        torch.load(
                            here / "models/MHs/score_model_weights.pt",
                            map_location=device,
                            weights_only=True,
                        )
                    )
                    model.to(device)
                    model.eval()

                if self._current_ps_method == "em":
                    sampler = self._em_sampler_cls(
                        sde,
                        (n_ps_batch, n_realisations, 32, 64),
                        device=device,
                        denoise=denoise,
                    ).get_em_sampler()
                else:
                    sampler = self._ode_sampler_cls(
                        sde,
                        (n_ps_batch, n_realisations, 32, 64),
                        device=device,
                        denoise=denoise,
                        rtol=1e-5,
                        atol=1e-5,
                    ).get_ode_sampler()

                replicas.append((device, model, sampler))

            self._ps_gpu_replicas = replicas
            self._ps_gpu_config_key = config_key

        replicas = self._ps_gpu_replicas
        n_batches = cdns.shape[0]
        results: list = [None] * n_batches

        def _worker(gpu_id: int) -> None:
            device, model, sampler = replicas[gpu_id]
            for idx in range(gpu_id, n_batches, n_gpus):
                cdn_t = torch.tensor(cdns[idx], dtype=torch.float32, device=device)
                samples = sampler(model, cdn=cdn_t).cpu().detach()
                results[idx] = reverse_transform(
                    samples, self.ps_log_std, self.ps_log_mean
                ).numpy()

        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = [executor.submit(_worker, g) for g in range(n_gpus)]
            for f in futures:
                f.result()  # re-raises any exception from a worker

        return np.array(results)

    def get_errors(
        self,
        emu: EmulatorOutput,
        theta_lstm: np.ndarray | None = None,
        theta_ps: np.ndarray | None = None,
        ps_sampling_method: str | None = None,
    ) -> ACGEmulatorErrors | RadioEmulatorErrors | MHEmulatorErrors:
        """Calculate the emulator error on its outputs.

        Parameters
        ----------
        emu : EmulatorOutput
            The emulator output to compute errors for.
        theta_lstm : np.ndarray, optional
            Normalized LSTM parameters (for MH emulator).
        theta_ps : np.ndarray, optional
            Normalized PS parameters (for MH emulator with 2D PS).
        ps_sampling_method : str, optional
            Sampling method for 2D PS: 'em' or 'ode'.

        Returns
        -------
        EmulatorErrors
            ACGEmulatorErrors, RadioEmulatorErrors, or MHEmulatorErrors
            depending on the emulator type. All provide dict-like access.

        See Also
        --------
        ACGEmulatorErrors : Errors for ACG/Default (v1) emulator.
        RadioEmulatorErrors : Errors for Radio (v2) emulator.
        MHEmulatorErrors : Errors for MH/MCG (v3) emulator.
        """
        if self.which_emulator == EMULATOR_ACG:
            return ACGEmulatorErrors.from_output(emu, self.properties)
        elif self.which_emulator == EMULATOR_RADIO:
            return RadioEmulatorErrors.from_properties(self.properties)

        # For MH emulator, use output-dependent absolute errors
        return MHEmulatorErrors.from_output(
            output=emu,
            properties=self.properties,
            ps_sampling_method=ps_sampling_method or "em",
        )
