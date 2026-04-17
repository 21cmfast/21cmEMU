"""PyTorch implementation of the v1 (default) 21cmEMU model.

This module provides the exact PyTorch equivalent of the original TensorFlow
model from Breitman+23. The architecture consists of:

- 8 shared Dense layers (1000 units each) with LeakyReLU
- Separate MLP heads for Tb, xHI, Ts, discont, tau, UVLF
- A Conv2DTranspose-based decoder for the power spectrum (PS)

Output order (concatenated): Tb(84), xHI(84), Ts(84), discont(1), PS(720=60×12), tau(1), UVLF(124)
Total: 1098 values
"""

from __future__ import annotations

from typing import Dict, List, OrderedDict
from collections import OrderedDict as OD

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefaultEmulatorV1(nn.Module):
    """Exact PyTorch replica of the v1 TensorFlow 21cmEMU model.
    
    The architecture follows the original Keras model exactly:
    - 8 shared layers: Dense(1000) + LeakyReLU each
    - Tb head: 600→700→700→1000→1000→800→800→84 (7 layers, LeakyReLU, linear output)
    - xHI head: 500→500→500→500→500→84 (5 layers, ReLU, sigmoid output)
    - Ts head: 400→400→400→400→400→84 (5 layers, LeakyReLU, linear output)
    - discont head: 400→400→400→400→400→1 (5 layers, LeakyReLU, linear output)
    - tau head: 30→30→30→1 (3 layers, LeakyReLU, linear output)
    - UVLF head: 400→400→400→400→400→124 (5 layers, LeakyReLU, linear output)
    - PS head: ConvTranspose2d-based decoder (reshapes 1000→1×1×1000 then upsamples to 60×12×1)
    """
    
    # Output sizes in concatenation order
    OUTPUT_SIZES = OD([
        ('Tb', 84),
        ('xHI', 84),
        ('Ts', 84),
        ('discont', 1),
        ('PS', 720),  # 60 × 12 flattened
        ('tau', 1),
        ('UVLF', 124),
    ])
    
    def __init__(self, negative_slope: float = 0.3):
        super().__init__()
        self.negative_slope = negative_slope
        
        # ================================================================
        # SHARED LAYERS: 8 × (Dense(1000) + LeakyReLU)
        # ================================================================
        self.shared = nn.Sequential(
            nn.Linear(9, 1000),
            nn.LeakyReLU(negative_slope),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(negative_slope),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(negative_slope),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(negative_slope),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(negative_slope),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(negative_slope),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(negative_slope),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(negative_slope),
        )
        
        # ================================================================
        # Tb HEAD: 1000 → 600 → 700 → 700 → 1000 → 1000 → 800 → 800 → 84
        # ================================================================
        self.tb_head = nn.Sequential(
            nn.Linear(1000, 600),
            nn.LeakyReLU(negative_slope),
            nn.Linear(600, 700),
            nn.LeakyReLU(negative_slope),
            nn.Linear(700, 700),
            nn.LeakyReLU(negative_slope),
            nn.Linear(700, 1000),
            nn.LeakyReLU(negative_slope),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(negative_slope),
            nn.Linear(1000, 800),
            nn.LeakyReLU(negative_slope),
            nn.Linear(800, 800),
            nn.LeakyReLU(negative_slope),
            nn.Linear(800, 84),  # final, linear activation
        )
        
        # ================================================================
        # xHI HEAD: 1000 → 500 → 500 → 500 → 500 → 500 → 84 (sigmoid)
        # Uses ReLU instead of LeakyReLU, sigmoid at output
        # ================================================================
        self.xhi_head = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 84),
            nn.Sigmoid(),  # final activation
        )
        
        # ================================================================
        # Ts HEAD: 1000 → 400 → 400 → 400 → 400 → 400 → 84
        # ================================================================
        self.ts_head = nn.Sequential(
            nn.Linear(1000, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 84),  # final, linear
        )
        
        # ================================================================
        # discont HEAD: 1000 → 400 → 400 → 400 → 400 → 400 → 1
        # ================================================================
        self.discont_head = nn.Sequential(
            nn.Linear(1000, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 1),  # final, linear
        )
        
        # ================================================================
        # tau HEAD: 1000 → 30 → 30 → 30 → 1
        # ================================================================
        self.tau_head = nn.Sequential(
            nn.Linear(1000, 30),
            nn.LeakyReLU(negative_slope),
            nn.Linear(30, 30),
            nn.LeakyReLU(negative_slope),
            nn.Linear(30, 30),
            nn.LeakyReLU(negative_slope),
            nn.Linear(30, 1),  # final, linear
        )
        
        # ================================================================
        # UVLF HEAD: 1000 → 400 → 400 → 400 → 400 → 400 → 124
        # ================================================================
        self.uvlf_head = nn.Sequential(
            nn.Linear(1000, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 400),
            nn.LeakyReLU(negative_slope),
            nn.Linear(400, 124),  # final, linear
        )
        
        # ================================================================
        # PS HEAD: Conv2DTranspose decoder
        # 1000 → reshape to (1, 1, 1000) → series of ConvTranspose2d → (60, 12, 1) → flatten to 720
        # ================================================================
        # Note: TensorFlow Conv2DTranspose has kernel shape (H, W, out_ch, in_ch)
        # PyTorch ConvTranspose2d has kernel shape (in_ch, out_ch, H, W)
        # TensorFlow default is "channels_last" (NHWC), PyTorch default is "channels_first" (NCHW)
        
        self.ps_decoder = nn.Sequential(
            # PS_Conv2DTr_0: (1,1,1000) → (4,2,256), kernel=(4,2), stride=(4,2)
            nn.ConvTranspose2d(1000, 256, kernel_size=(4, 2), stride=(4, 2)),
            nn.LeakyReLU(negative_slope),
            # PS_Conv2DTr_1: (4,2,256) → (4,2,256), kernel=(1,1), stride=(1,1) - same size
            nn.ConvTranspose2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope),
            # PS_Conv2DTr_2: (4,2,256) → (6,4,256), kernel=(3,3), stride=(1,2), padding depends
            nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 2), output_padding=(0, 1)),
            nn.LeakyReLU(negative_slope),
            # PS_Conv2DTr_3: (6,4,256) → (6,4,128), kernel=(1,1)
            nn.ConvTranspose2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope),
            # PS_Conv2DTr_4: (6,4,128) → (12,6,128), kernel=(3,3), stride=(2,1), padding
            nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 1), output_padding=(1, 0)),
            nn.LeakyReLU(negative_slope),
            # PS_Conv2DTr_5: (12,6,128) → (14,6,64), kernel=(3,1)
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope),
            # PS_Conv2DTr_6: (14,6,64) → (14,6,64), kernel=(1,1)
            nn.ConvTranspose2d(64, 64, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope),
            # PS_Conv2DTr_7: (14,6,64) → (14,6,32), kernel=(1,1)
            nn.ConvTranspose2d(64, 32, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope),
            # PS_Conv2DTr_8: (14,6,32) → (14,6,32), kernel=(1,1)
            nn.ConvTranspose2d(32, 32, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope),
            # PS_Upsamp2D_9: (14,6,32) → (28,12,32)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.LeakyReLU(negative_slope),
            # PS_Conv2DTr_10: (28,12,32) → (30,12,8), kernel=(3,1)
            nn.ConvTranspose2d(32, 8, kernel_size=(3, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope),
            # PS_Conv2DTr_11: (30,12,8) → (30,12,8), kernel=(1,1)
            nn.ConvTranspose2d(8, 8, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope),
            # PS_Upsamp2D_12: (30,12,8) → (60,12,8)
            nn.Upsample(scale_factor=(2, 1), mode='nearest'),
            nn.LeakyReLU(negative_slope),
            # PS_Conv2DTr_13: (60,12,8) → (60,12,8), kernel=(1,1)
            nn.ConvTranspose2d(8, 8, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope),
            # PS_Conv2DTr_14: (60,12,8) → (60,12,1), kernel=(1,1)
            nn.ConvTranspose2d(8, 1, kernel_size=(1, 1), stride=(1, 1)),
            # No activation at the end
            nn.Flatten(),  # (60, 12, 1) → 720
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning concatenated outputs.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 9)
            Normalized input parameters.
            
        Returns
        -------
        torch.Tensor, shape (batch, 1098)
            Concatenated outputs in order: Tb(84), xHI(84), Ts(84), discont(1), PS(720), tau(1), UVLF(124)
        """
        # Shared layers
        h = self.shared(x)
        
        # Compute each head
        tb_out = self.tb_head(h)
        xhi_out = self.xhi_head(h)
        ts_out = self.ts_head(h)
        discont_out = self.discont_head(h)
        tau_out = self.tau_head(h)
        uvlf_out = self.uvlf_head(h)
        
        # PS requires reshape: (batch, 1000) → (batch, 1000, 1, 1)
        ps_input = h.view(-1, 1000, 1, 1)
        ps_out = self.ps_decoder(ps_input)
        
        # Concatenate in the same order as TensorFlow model
        return torch.cat([tb_out, xhi_out, ts_out, discont_out, ps_out, tau_out, uvlf_out], dim=-1)
    
    def forward_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning dictionary of outputs."""
        out = self.forward(x)
        result = {}
        idx = 0
        for name, size in self.OUTPUT_SIZES.items():
            result[name] = out[:, idx:idx+size]
            idx += size
        return result
