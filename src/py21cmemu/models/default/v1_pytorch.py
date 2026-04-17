"""PyTorch implementation of the v1 21cmEMU default emulator.

This module provides the PyTorch model class for the v1 emulator, originally
implemented in TensorFlow/Keras. The model was converted using the
convert_v1_to_pytorch.py script.

Usage
-----
    from py21cmemu.models.default.v1_pytorch import load_converted_model
    
    model = load_converted_model("/path/to/default_model.pt")
    output = model(input_tensor)
"""

from __future__ import annotations

from collections import OrderedDict as OD
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _crop_like_tf_same(x: torch.Tensor, kernel_h: int, kernel_w: int) -> torch.Tensor:
    """Crop output to match TensorFlow 'same' padding for Conv2DTranspose.
    
    TF crops (kernel-1) total from each dimension, splitting floor/ceil for odd/even.
    For odd kernel k: crop (k-1)//2 from start and end
    For even kernel k: crop (k-1)//2 from start, k//2 from end
    """
    crop_h_start = (kernel_h - 1) // 2
    crop_h_end = kernel_h // 2
    crop_w_start = (kernel_w - 1) // 2  
    crop_w_end = kernel_w // 2
    
    _, _, h, w = x.shape
    return x[:, :, crop_h_start:h-crop_h_end if crop_h_end else h,
                    crop_w_start:w-crop_w_end if crop_w_end else w]


class DefaultEmulatorV1(nn.Module):
    """PyTorch replica of the v1 TensorFlow 21cmEMU model.
    
    This model emulates six summary statistics from 9 input astrophysical parameters:
    
    Outputs (1098 total):
        - Tb: 21cm brightness temperature (84 redshift bins)
        - xHI: neutral fraction (84 redshift bins)
        - Ts: spin temperature (84 redshift bins)
        - discont: discontinuity (1 value)
        - PS: power spectrum (720 = 60 k-bins × 12 z-bins)
        - tau: Thomson optical depth (1 value)
        - UVLF: UV luminosity function (124 values)
    
    Parameters
    ----------
    negative_slope : float
        Negative slope for LeakyReLU. TF model uses 0.1.
    """
    
    OUTPUT_SIZES = OD([
        ('Tb', 84), ('xHI', 84), ('Ts', 84), ('discont', 1),
        ('PS', 720), ('tau', 1), ('UVLF', 124),
    ])
    
    def __init__(self, negative_slope: float = 0.1):
        super().__init__()
        self.negative_slope = negative_slope
        
        # Shared layers: 8 × (Dense(1000) + LeakyReLU)
        self.shared_0 = nn.Linear(9, 1000)
        self.shared_1 = nn.Linear(1000, 1000)
        self.shared_2 = nn.Linear(1000, 1000)
        self.shared_3 = nn.Linear(1000, 1000)
        self.shared_4 = nn.Linear(1000, 1000)
        self.shared_5 = nn.Linear(1000, 1000)
        self.shared_6 = nn.Linear(1000, 1000)
        self.shared_7 = nn.Linear(1000, 1000)
        
        # Tb head: 1000→600→700→700→1000→1000→800→800→84
        self.tb_0 = nn.Linear(1000, 600)
        self.tb_1 = nn.Linear(600, 700)
        self.tb_2 = nn.Linear(700, 700)
        self.tb_3 = nn.Linear(700, 1000)
        self.tb_4 = nn.Linear(1000, 1000)
        self.tb_5 = nn.Linear(1000, 800)
        self.tb_6 = nn.Linear(800, 800)
        self.tb_final = nn.Linear(800, 84)
        
        # xHI head: 1000→500×5→84 (ReLU, sigmoid output)
        self.xhi_0 = nn.Linear(1000, 500)
        self.xhi_1 = nn.Linear(500, 500)
        self.xhi_2 = nn.Linear(500, 500)
        self.xhi_3 = nn.Linear(500, 500)
        self.xhi_4 = nn.Linear(500, 500)
        self.xhi_final = nn.Linear(500, 84)
        
        # Ts head: 1000→400×5→84
        self.ts_0 = nn.Linear(1000, 400)
        self.ts_1 = nn.Linear(400, 400)
        self.ts_2 = nn.Linear(400, 400)
        self.ts_3 = nn.Linear(400, 400)
        self.ts_4 = nn.Linear(400, 400)
        self.ts_final = nn.Linear(400, 84)
        
        # discont head: 1000→400×5→1
        self.discont_0 = nn.Linear(1000, 400)
        self.discont_1 = nn.Linear(400, 400)
        self.discont_2 = nn.Linear(400, 400)
        self.discont_3 = nn.Linear(400, 400)
        self.discont_4 = nn.Linear(400, 400)
        self.discont_final = nn.Linear(400, 1)
        
        # tau head: 1000→30×3→1
        self.tau_0 = nn.Linear(1000, 30)
        self.tau_1 = nn.Linear(30, 30)
        self.tau_2 = nn.Linear(30, 30)
        self.tau_final = nn.Linear(30, 1)
        
        # UVLF head: 1000→400×5→124
        self.uvlf_0 = nn.Linear(1000, 400)
        self.uvlf_1 = nn.Linear(400, 400)
        self.uvlf_2 = nn.Linear(400, 400)
        self.uvlf_3 = nn.Linear(400, 400)
        self.uvlf_4 = nn.Linear(400, 400)
        self.uvlf_final = nn.Linear(400, 124)
        
        # PS decoder: Conv2DTranspose layers (no input padding, crop output for 'same')
        self.ps_conv_0 = nn.ConvTranspose2d(1000, 256, kernel_size=(4, 2), stride=1, padding=0)
        self.ps_conv_1 = nn.ConvTranspose2d(256, 256, kernel_size=(7, 3), stride=1, padding=0)
        self.ps_conv_2 = nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=1, padding=0)
        self.ps_conv_3 = nn.ConvTranspose2d(256, 128, kernel_size=(7, 3), stride=1, padding=0)
        self.ps_conv_4 = nn.ConvTranspose2d(128, 128, kernel_size=(7, 3), stride=1, padding=0)
        self.ps_conv_5 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 1), stride=1, padding=0)
        self.ps_conv_6 = nn.ConvTranspose2d(64, 64, kernel_size=(5, 3), stride=1, padding=0)
        self.ps_conv_7 = nn.ConvTranspose2d(64, 32, kernel_size=(7, 3), stride=1, padding=0)
        self.ps_conv_8 = nn.ConvTranspose2d(32, 32, kernel_size=(7, 3), stride=1, padding=0)
        self.ps_conv_10 = nn.ConvTranspose2d(32, 8, kernel_size=(3, 1), stride=1, padding=0)
        self.ps_conv_11 = nn.ConvTranspose2d(8, 8, kernel_size=(9, 3), stride=1, padding=0)
        self.ps_conv_13 = nn.ConvTranspose2d(8, 8, kernel_size=(9, 3), stride=1, padding=0)
        self.ps_conv_14 = nn.ConvTranspose2d(8, 1, kernel_size=(11, 3), stride=1, padding=0)
    
    def _lrelu(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, self.negative_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 9) with normalized astrophysical parameters.
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1098) with concatenated predictions.
        """
        # Shared layers
        h = self._lrelu(self.shared_0(x))
        h = self._lrelu(self.shared_1(h))
        h = self._lrelu(self.shared_2(h))
        h = self._lrelu(self.shared_3(h))
        h = self._lrelu(self.shared_4(h))
        h = self._lrelu(self.shared_5(h))
        h = self._lrelu(self.shared_6(h))
        h = self._lrelu(self.shared_7(h))
        
        # Tb head (LeakyReLU)
        tb = self._lrelu(self.tb_0(h))
        tb = self._lrelu(self.tb_1(tb))
        tb = self._lrelu(self.tb_2(tb))
        tb = self._lrelu(self.tb_3(tb))
        tb = self._lrelu(self.tb_4(tb))
        tb = self._lrelu(self.tb_5(tb))
        tb = self._lrelu(self.tb_6(tb))
        tb = self.tb_final(tb)
        
        # xHI head (ReLU, sigmoid output)
        xhi = F.relu(self.xhi_0(h))
        xhi = F.relu(self.xhi_1(xhi))
        xhi = F.relu(self.xhi_2(xhi))
        xhi = F.relu(self.xhi_3(xhi))
        xhi = F.relu(self.xhi_4(xhi))
        xhi = torch.sigmoid(self.xhi_final(xhi))
        
        # Ts head (LeakyReLU)
        ts = self._lrelu(self.ts_0(h))
        ts = self._lrelu(self.ts_1(ts))
        ts = self._lrelu(self.ts_2(ts))
        ts = self._lrelu(self.ts_3(ts))
        ts = self._lrelu(self.ts_4(ts))
        ts = self.ts_final(ts)
        
        # discont head (LeakyReLU)
        dc = self._lrelu(self.discont_0(h))
        dc = self._lrelu(self.discont_1(dc))
        dc = self._lrelu(self.discont_2(dc))
        dc = self._lrelu(self.discont_3(dc))
        dc = self._lrelu(self.discont_4(dc))
        dc = self.discont_final(dc)
        
        # tau head (LeakyReLU)
        tau = self._lrelu(self.tau_0(h))
        tau = self._lrelu(self.tau_1(tau))
        tau = self._lrelu(self.tau_2(tau))
        tau = self.tau_final(tau)
        
        # UVLF head (LeakyReLU)
        uvlf = self._lrelu(self.uvlf_0(h))
        uvlf = self._lrelu(self.uvlf_1(uvlf))
        uvlf = self._lrelu(self.uvlf_2(uvlf))
        uvlf = self._lrelu(self.uvlf_3(uvlf))
        uvlf = self._lrelu(self.uvlf_4(uvlf))
        uvlf = self.uvlf_final(uvlf)
        
        # PS decoder with TF 'same' padding emulation
        ps = h.view(-1, 1000, 1, 1)
        ps = self._lrelu(self.ps_conv_0(ps))                              # valid: (1,1) → (4,2)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_1(ps), 7, 3))    # same: (4,2)
        ps = self._lrelu(self.ps_conv_2(ps))                              # valid: (4,2) → (6,4)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_3(ps), 7, 3))    # same: (6,4)
        ps = self._lrelu(self.ps_conv_4(ps))                              # valid: (6,4) → (12,6)
        ps = self._lrelu(self.ps_conv_5(ps))                              # valid: (12,6) → (14,6)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_6(ps), 5, 3))    # same: (14,6)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_7(ps), 7, 3))    # same: (14,6)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_8(ps), 7, 3))    # same: (14,6)
        ps = self._lrelu(F.interpolate(ps, scale_factor=2, mode='nearest'))  # upsample: (28,12)
        ps = self._lrelu(self.ps_conv_10(ps))                             # valid: (28,12) → (30,12)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_11(ps), 9, 3))   # same: (30,12)
        ps = self._lrelu(F.interpolate(ps, scale_factor=(2,1), mode='nearest'))  # upsample: (60,12)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_13(ps), 9, 3))   # same: (60,12)
        ps = _crop_like_tf_same(self.ps_conv_14(ps), 11, 3)               # same: (60,12,1)
        ps = ps.contiguous().view(-1, 720)
        
        return torch.cat([tb, xhi, ts, dc, ps, tau, uvlf], dim=-1)
    
    def forward_stacked(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward() for compatibility."""
        return self.forward(x)
    
    def forward_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning dictionary of outputs."""
        out = self.forward(x)
        result = {}
        idx = 0
        for name, size in self.OUTPUT_SIZES.items():
            result[name] = out[:, idx:idx+size]
            idx += size
        return result


def load_converted_model(
    model_path: str,
    device: torch.device | str = 'cpu',
) -> DefaultEmulatorV1:
    """Load converted PyTorch model.
    
    Parameters
    ----------
    model_path : str
        Path to PyTorch model (.pt file).
    device : torch.device or str
        Device to load model on.
    
    Returns
    -------
    DefaultEmulatorV1
        Loaded model in eval mode.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = DefaultEmulatorV1(negative_slope=checkpoint.get('negative_slope', 0.1))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model
