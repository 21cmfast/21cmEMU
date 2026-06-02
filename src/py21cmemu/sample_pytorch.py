"""Sampling routines for conditional score-based diffusion models.

Provides two sampler families (matching Yang Song's score_sde_pytorch):
  1. **Probability-flow ODE** – deterministic, via scipy.integrate.solve_ivp
  2. **Euler-Maruyama (EM)** – stochastic, step-by-step reverse-SDE

Based on Yang Song's score_sde_pytorch/sampling.py:
https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py
"""

from __future__ import annotations

import abc
import gc
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy import integrate

from .model_utils import get_score_fn
from .utils import from_flattened_numpy, to_flattened_numpy

if TYPE_CHECKING:
    from typing import Callable


# ===================================================================
# Predictors
# ===================================================================

class Predictor(abc.ABC):
    """Abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow: bool = False):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x_cdn: torch.Tensor | None = None,
        cdn: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One update step of the predictor.

        Args:
            x: Current state tensor.
            t: Current time step tensor.
            x_cdn: Optional conditional input image.
            cdn: Optional conditional parameter vector.

        Returns:
            x: Next state (with noise for stochastic samplers).
            x_mean: Next state without random noise (for denoising).
        """
        pass


class EulerMaruyamaPredictor(Predictor):
    """Euler-Maruyama discretisation of the reverse SDE.

    This is a stochastic sampler that generates diverse samples by following
    the reverse SDE with noise injection at each step.
    """

    def __init__(self, sde, score_fn, probability_flow: bool = False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x_cdn: torch.Tensor | None = None,
        cdn: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = -1.0 / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t, x_cdn=x_cdn, cdn=cdn)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


class ReverseDiffusionPredictor(Predictor):
    """DDPM-style ancestral sampling predictor (discrete reverse diffusion)."""

    def __init__(self, sde, score_fn, probability_flow: bool = False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x_cdn: torch.Tensor | None = None,
        cdn: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f, G = self.rsde.discretize(x, t, x_cdn=x_cdn, cdn=cdn)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


class NoneCorrector:
    """Dummy corrector that does nothing (used with EM or ODE samplers)."""

    def __init__(self, sde, score_fn, snr: float = 0.0, n_steps: int = 0):
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    def update_fn(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x_cdn: torch.Tensor | None = None,
        cdn: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x


# ===================================================================
# Euler-Maruyama Sampler
# ===================================================================

class GetEMSampler:
    """Euler-Maruyama sampler for stochastic reverse-SDE sampling.

    This sampler follows the reverse SDE with noise injection at each step,
    producing more diverse samples than the ODE sampler. This is the
    recommended sampler for uncertainty quantification since it naturally
    captures the stochastic variance of the generative model.

    Args:
        sde: Forward SDE object (e.g., VPSDE).
        shape: Shape tuple (N_params, N_samples, H, W).
        inverse_scaler: Optional function to un-normalise samples.
        denoise: If True, apply one final denoising step for sharper samples.
        eps: Smallest time value for numerical stability.
        device: PyTorch device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        sde,
        shape: tuple[int, ...],
        inverse_scaler: Callable | None = None,
        denoise: bool = True,
        eps: float = 1e-3,
        device: str | torch.device = "cuda",
    ):
        self.sde = sde
        self.shape = shape
        self.n_samples = shape[1]  # Number of samples per parameter set
        self.single_batch_shape = (shape[0], 1, shape[2], shape[3])
        self.inverse_scaler = inverse_scaler
        self.denoise = denoise
        self.eps = eps
        self.device = device

    def _em_sample_once(
        self,
        model: torch.nn.Module,
        score_fn,
        predictor,
        corrector,
        timesteps,
        cdn: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate a single sample for each parameter in the batch."""
        x = self.sde.prior_sampling(self.single_batch_shape).to(self.device)

        for t_val in timesteps:
            vec_t = torch.ones(self.single_batch_shape[0], device=self.device) * t_val
            x, x_mean = corrector.update_fn(x, vec_t, x_cdn=None, cdn=cdn)
            x, x_mean = predictor.update_fn(x, vec_t, x_cdn=None, cdn=cdn)

        result = x_mean if self.denoise else x
        if self.inverse_scaler is not None:
            result = self.inverse_scaler(result)
        return result

    def em_sampler(
        self,
        model: torch.nn.Module,
        z: torch.Tensor | None = None,
        x_cdn: torch.Tensor | None = None,
        cdn: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run Euler-Maruyama sampling.

        Args:
            model: Trained score model.
            z: Optional latent code (not used, kept for API compatibility).
            x_cdn: Optional conditional input image (not used).
            cdn: Conditional parameter vector (batch, n_params).

        Returns:
            samples: Tensor of shape (N_params, N_samples, H, W).
        """
        with torch.no_grad():
            score_fn = get_score_fn(self.sde, model, train=False, continuous=True)
            predictor = EulerMaruyamaPredictor(
                self.sde, score_fn, probability_flow=False
            )
            corrector = NoneCorrector(self.sde, score_fn)

            if cdn is not None:
                cdn = cdn.to(self.device)

            timesteps = torch.linspace(
                self.sde.T, self.eps, self.sde.N, device=self.device
            )

            # Generate n_samples independent samples
            all_samples = []
            for _ in range(self.n_samples):
                sample = self._em_sample_once(
                    model, score_fn, predictor, corrector, timesteps, cdn=cdn
                )
                all_samples.append(sample)

            # Stack: (n_samples, N_params, 1, H, W) -> (N_params, n_samples, H, W)
            result = torch.cat(all_samples, dim=1)

            gc.collect()
            torch.cuda.empty_cache()

            return result

    def get_em_sampler(self) -> Callable:
        """Return the sampling function."""
        return self.em_sampler


def get_em_sampler(
    sde,
    shape: tuple[int, ...],
    *,
    inverse_scaler: Callable | None = None,
    denoise: bool = True,
    eps: float = 1e-3,
    device: str | torch.device = "cuda",
) -> Callable:
    """Create an Euler-Maruyama sampler function.

    This is a convenience wrapper around GetEMSampler.

    Args:
        sde: Forward SDE object.
        shape: Shape tuple (N_params, 1, H, W) for a single sample.
        inverse_scaler: Optional function to un-normalise samples.
        denoise: If True, apply one final denoising step.
        eps: Smallest time value for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function: samples = sampler(model, cdn=...)
    """
    sampler = GetEMSampler(
        sde,
        shape,
        inverse_scaler=inverse_scaler,
        denoise=denoise,
        eps=eps,
        device=device,
    )
    return sampler.get_em_sampler()


# ===================================================================
# Probability-Flow ODE Sampler
# ===================================================================

class GetODESampler:
    """Probability-flow ODE sampler using scipy's black-box ODE solver.

    Uses the probability-flow ODE (deterministic) to generate samples.
    This sampler produces consistent (deterministic) outputs for the same
    initial noise, which is useful for debugging but may have slightly
    higher fractional error compared to EM sampling.

    Args:
        sde: Forward SDE object (e.g., VPSDE).
        shape: Shape tuple (N_params, N_samples, H, W).
        inverse_scaler: Optional function to un-normalise samples.
        denoise: If True, apply one-step denoising to final samples.
        rtol: Relative tolerance for the ODE solver.
        atol: Absolute tolerance for the ODE solver.
        method: ODE solver method ('RK45', 'RK23', 'DOP853', etc.).
        eps: Smallest time value for numerical stability.
        device: PyTorch device.
    """

    def __init__(
        self,
        sde,
        shape: tuple[int, ...],
        inverse_scaler: Callable | None = None,
        denoise: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        method: str = "RK45",
        eps: float = 1e-3,
        device: str | torch.device = "cuda",
    ):
        self.sde = sde
        self.shape = shape
        self.single_batch_shape = (shape[0] * shape[1], 1, shape[2], shape[3])
        self.inverse_scaler = inverse_scaler
        self.denoise = denoise
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.eps = eps
        self.device = device

    def denoise_update_fn(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        x_cdn: torch.Tensor | None = None,
        cdn: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One-step denoising via reverse diffusion predictor."""
        score_fn = get_score_fn(self.sde, model, train=False, continuous=True)
        predictor = ReverseDiffusionPredictor(
            self.sde, score_fn, probability_flow=False
        )
        vec_eps = torch.ones(x.shape[0], device=x.device) * self.eps
        _, x = predictor.update_fn(x, vec_eps, x_cdn=x_cdn, cdn=cdn)
        return x

    def drift_fn(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        x_cdn: torch.Tensor | None = None,
        cdn: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get the drift function of the reverse-time probability flow ODE."""
        score_fn = get_score_fn(self.sde, model, train=False, continuous=True)
        rsde = self.sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, x_cdn=x_cdn, cdn=cdn)[0]

    def ode_sampler(
        self,
        model: torch.nn.Module,
        z: torch.Tensor | None = None,
        x_cdn: torch.Tensor | None = None,
        cdn: torch.Tensor | None = None,
        return_nfe: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, int]:
        """Run the probability-flow ODE sampler.

        Args:
            model: Trained score model (on GPU).
            z: Optional latent code to generate samples from.
            x_cdn: Optional conditional input image.
            cdn: Conditional parameter vector (batch, n_params).
            return_nfe: If True, also return number of function evaluations.

        Returns:
            samples: Tensor of shape (N_params, N_samples, H, W).
            nfe (optional): Number of function evaluations if return_nfe=True.
        """
        with torch.no_grad():
            if z is None:
                x = self.sde.prior_sampling(self.single_batch_shape).to(self.device)
            else:
                x = z.to(self.device)

            if x_cdn is not None:
                x_cdn = (
                    x_cdn.tile((self.shape[1],))
                    .reshape(self.single_batch_shape)
                    .to(self.device)
                )
            if cdn is not None:
                cdn = (
                    cdn.tile((self.shape[1],))
                    .reshape((self.single_batch_shape[0], cdn.shape[1]))
                    .to(self.device)
                )

            def ode_func(t: float, x_np: np.ndarray) -> np.ndarray:
                x_t = from_flattened_numpy(x_np, self.single_batch_shape)
                x_t = x_t.to(self.device).float()
                vec_t = torch.ones(self.single_batch_shape[0], device=x_t.device) * t
                drift = self.drift_fn(model, x_t, vec_t, x_cdn=x_cdn, cdn=cdn)
                return to_flattened_numpy(drift)

            # Black-box ODE solver
            solution = integrate.solve_ivp(
                ode_func,
                (self.sde.T, self.eps),
                to_flattened_numpy(x),
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
            )
            nfe = solution.nfev

            sln = (
                torch.tensor(solution.y[:, -1])
                .reshape(self.single_batch_shape)
                .to(self.device)
                .float()
            )

            if self.denoise:
                sln = self.denoise_update_fn(model, sln, x_cdn=x_cdn, cdn=cdn)
            if self.inverse_scaler is not None:
                sln = self.inverse_scaler(sln)

            sln = sln.reshape(self.shape)

            gc.collect()
            torch.cuda.empty_cache()

            if return_nfe:
                return sln, nfe
            return sln

    def get_ode_sampler(self) -> Callable:
        """Return the sampling function."""
        return self.ode_sampler
