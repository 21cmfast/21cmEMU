"""21cmEMUv3 Model Components - MiniHalo Emulator (v3).

This subpackage provides the neural network models for the v3 emulator:

- ``lstm_model``: LSTM-based model for global signals, UVLFs
- ``score_model``: UNet diffusion model for 2D power spectrum

The SDE and sampling routines are in the parent package:
- ``py21cmemu.sde``: Stochastic differential equations for diffusion
- ``py21cmemu.sample_pytorch``: Sampling routines for diffusion model

Usage
-----
>>> from py21cmemu.models.MCG.lstm_model import MH_Emulator
>>> from py21cmemu.models.MCG.score_model_final import UNet
>>> from py21cmemu.sde import VPSDE
>>> from py21cmemu.sample_pytorch import GetODESampler
"""

from .lstm_model import MH_Emulator
from .score_model import UNet

__all__ = [
    "MH_Emulator",
    "UNet",
]
