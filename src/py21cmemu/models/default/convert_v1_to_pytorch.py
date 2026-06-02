#!/usr/bin/env python3
"""Convert v1 TensorFlow 21cmEMU model to PyTorch.

This script loads the original Keras SavedModel and creates an equivalent PyTorch
model with transferred weights. The conversion handles:

1. Dense layers: Weight transpose (TF: in×out → PyTorch: out×in)
2. Conv2DTranspose: Kernel permutation (TF: H,W,out,in → PyTorch: in,out,H,W)
3. Data format: TF NHWC → PyTorch NCHW

Usage
-----
    python convert_v1_to_pytorch.py [--input PATH] [--output PATH] [--validate]

Or from Python:

    >>> from convert_v1_to_pytorch import convert_and_save
    >>> convert_and_save("/path/to/tf_model", "output_model.pt")
"""

from __future__ import annotations

import argparse
import logging
from collections import OrderedDict as OD
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# =============================================================================
# PyTorch Model Definition (matching TF architecture exactly)
# =============================================================================


def _crop_like_tf_same(x: torch.Tensor, kernel_h: int, kernel_w: int) -> torch.Tensor:
    """Crop output to match TensorFlow 'same' padding for Conv2DTranspose.

    TF crops (kernel-1) total from each dimension, splitting floor/ceil for odd/even.
    For odd kernel k: crop (k-1)//2 from start and end
    For even kernel k: crop (k-1)//2 from start, k//2 from end
    """
    # Calculate crops following TF's asymmetric pattern
    crop_h_start = (kernel_h - 1) // 2
    crop_h_end = kernel_h // 2
    crop_w_start = (kernel_w - 1) // 2
    crop_w_end = kernel_w // 2

    # Apply cropping (x is NCHW)
    _, _, h, w = x.shape
    return x[
        :,
        :,
        crop_h_start : h - crop_h_end if crop_h_end else h,
        crop_w_start : w - crop_w_end if crop_w_end else w,
    ]


class DefaultEmulatorV1(nn.Module):
    """Exact PyTorch replica of the v1 TensorFlow 21cmEMU model."""

    OUTPUT_SIZES = OD(
        [
            ("Tb", 84),
            ("xHI", 84),
            ("Ts", 84),
            ("discont", 1),
            ("PS", 720),
            ("tau", 1),
            ("UVLF", 124),
        ]
    )

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

        # PS decoder: Conv2DTranspose layers with NO padding (we'll crop output manually)
        # TF 'same' padding crops the output, doesn't pad the input
        # All layers use padding=0, we crop after for 'same' ones

        # valid: output = input + kernel - 1
        # same: output = input (after cropping kernel-1)

        self.ps_conv_0 = nn.ConvTranspose2d(
            1000, 256, kernel_size=(4, 2), stride=1, padding=0
        )  # valid
        self.ps_conv_1 = nn.ConvTranspose2d(
            256, 256, kernel_size=(7, 3), stride=1, padding=0
        )  # same (crop)
        self.ps_conv_2 = nn.ConvTranspose2d(
            256, 256, kernel_size=(3, 3), stride=1, padding=0
        )  # valid
        self.ps_conv_3 = nn.ConvTranspose2d(
            256, 128, kernel_size=(7, 3), stride=1, padding=0
        )  # same (crop)
        self.ps_conv_4 = nn.ConvTranspose2d(
            128, 128, kernel_size=(7, 3), stride=1, padding=0
        )  # valid
        self.ps_conv_5 = nn.ConvTranspose2d(
            128, 64, kernel_size=(3, 1), stride=1, padding=0
        )  # valid
        self.ps_conv_6 = nn.ConvTranspose2d(
            64, 64, kernel_size=(5, 3), stride=1, padding=0
        )  # same (crop)
        self.ps_conv_7 = nn.ConvTranspose2d(
            64, 32, kernel_size=(7, 3), stride=1, padding=0
        )  # same (crop)
        self.ps_conv_8 = nn.ConvTranspose2d(
            32, 32, kernel_size=(7, 3), stride=1, padding=0
        )  # same (crop)
        self.ps_conv_10 = nn.ConvTranspose2d(
            32, 8, kernel_size=(3, 1), stride=1, padding=0
        )  # valid
        self.ps_conv_11 = nn.ConvTranspose2d(
            8, 8, kernel_size=(9, 3), stride=1, padding=0
        )  # same (crop)
        self.ps_conv_13 = nn.ConvTranspose2d(
            8, 8, kernel_size=(9, 3), stride=1, padding=0
        )  # same (crop)
        self.ps_conv_14 = nn.ConvTranspose2d(
            8, 1, kernel_size=(11, 3), stride=1, padding=0
        )  # same (crop)

    def _lrelu(self, x):
        return F.leaky_relu(x, self.negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # PS decoder with TF 'same' padding emulation via output cropping
        # Reshape: (batch, 1000) → (batch, 1000, 1, 1) for PyTorch NCHW
        ps = h.view(-1, 1000, 1, 1)

        # conv0: valid, (1,1) → (4,2)
        ps = self._lrelu(self.ps_conv_0(ps))
        # conv1: same (7,3), crop to keep (4,2)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_1(ps), 7, 3))
        # conv2: valid, (4,2) → (6,4)
        ps = self._lrelu(self.ps_conv_2(ps))
        # conv3: same (7,3), crop to keep (6,4)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_3(ps), 7, 3))
        # conv4: valid, (6,4) → (12,6)
        ps = self._lrelu(self.ps_conv_4(ps))
        # conv5: valid, (12,6) → (14,6)
        ps = self._lrelu(self.ps_conv_5(ps))
        # conv6: same (5,3), crop to keep (14,6)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_6(ps), 5, 3))
        # conv7: same (7,3), crop to keep (14,6)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_7(ps), 7, 3))
        # conv8: same (7,3), crop to keep (14,6)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_8(ps), 7, 3))
        # upsample 9: (14,6) → (28,12)
        ps = self._lrelu(F.interpolate(ps, scale_factor=2, mode="nearest"))
        # conv10: valid, (28,12) → (30,12)
        ps = self._lrelu(self.ps_conv_10(ps))
        # conv11: same (9,3), crop to keep (30,12)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_11(ps), 9, 3))
        # upsample 12: (30,12) → (60,12)
        ps = self._lrelu(F.interpolate(ps, scale_factor=(2, 1), mode="nearest"))
        # conv13: same (9,3), crop to keep (60,12)
        ps = self._lrelu(_crop_like_tf_same(self.ps_conv_13(ps), 9, 3))
        # conv14: same (11,3), final output (60,12,1)
        ps = _crop_like_tf_same(self.ps_conv_14(ps), 11, 3)
        ps = ps.contiguous().view(
            -1, 720
        )  # Make contiguous after cropping, then flatten

        return torch.cat([tb, xhi, ts, dc, ps, tau, uvlf], dim=-1)

    def forward_stacked(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward() for compatibility."""
        return self.forward(x)

    def forward_dict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.forward(x)
        result = {}
        idx = 0
        for name, size in self.OUTPUT_SIZES.items():
            result[name] = out[:, idx : idx + size]
            idx += size
        return result


# =============================================================================
# Weight Conversion Functions
# =============================================================================


def convert_dense_weights(
    tf_weights: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert TF Dense weights to PyTorch Linear weights.

    TF Dense: weight (in, out), bias (out,)
    PyTorch Linear: weight (out, in), bias (out,)
    """
    w, b = tf_weights
    return torch.from_numpy(w.T.copy()), torch.from_numpy(b.copy())


def convert_conv_transpose_weights(
    tf_weights: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert TF Conv2DTranspose weights to PyTorch ConvTranspose2d weights.

    TF Conv2DTranspose kernel: (H, W, out_ch, in_ch)
    PyTorch ConvTranspose2d kernel: (in_ch, out_ch, H, W)
    """
    w, b = tf_weights
    # Permute: (H, W, out, in) → (in, out, H, W)
    w_pt = np.transpose(w, (3, 2, 0, 1))
    return torch.from_numpy(w_pt.copy()), torch.from_numpy(b.copy())


def load_tf_model(model_path: str):
    """Load TensorFlow model."""
    import tensorflow as tf

    return tf.keras.models.load_model(model_path, compile=False)


def extract_tf_weights(tf_model) -> dict[str, list[np.ndarray]]:
    """Extract all weights from TensorFlow model."""
    weights = {}
    for layer in tf_model.layers:
        if layer.get_weights():
            weights[layer.name] = layer.get_weights()
    return weights


def transfer_weights(
    pt_model: DefaultEmulatorV1, tf_weights: dict[str, list[np.ndarray]]
):
    """Transfer weights from TensorFlow to PyTorch model."""

    # Shared layers
    for i in range(8):
        w, b = convert_dense_weights(tf_weights[f"Dense_shared_{i}"])
        getattr(pt_model, f"shared_{i}").weight.data = w
        getattr(pt_model, f"shared_{i}").bias.data = b

    # Tb head
    tb_tf_names = [
        "Dense_Tb_0",
        "Dense_Tb_1",
        "Dense_Tb_2",
        "Dense_Tb_3",
        "Dense_Tb_4",
        "Dense_Tb_5",
        "Dense_Tb_6",
        "Dense_Tb_final",
    ]
    tb_pt_names = ["tb_0", "tb_1", "tb_2", "tb_3", "tb_4", "tb_5", "tb_6", "tb_final"]
    for tf_name, pt_name in zip(tb_tf_names, tb_pt_names, strict=False):
        w, b = convert_dense_weights(tf_weights[tf_name])
        getattr(pt_model, pt_name).weight.data = w
        getattr(pt_model, pt_name).bias.data = b

    # xHI head
    xhi_tf_names = [
        "Dense_xHI_0",
        "Dense_xHI_1",
        "Dense_xHI_2",
        "Dense_xHI_3",
        "Dense_xHI_4",
        "Dense_xHI_final",
    ]
    xhi_pt_names = ["xhi_0", "xhi_1", "xhi_2", "xhi_3", "xhi_4", "xhi_final"]
    for tf_name, pt_name in zip(xhi_tf_names, xhi_pt_names, strict=False):
        w, b = convert_dense_weights(tf_weights[tf_name])
        getattr(pt_model, pt_name).weight.data = w
        getattr(pt_model, pt_name).bias.data = b

    # Ts head
    ts_tf_names = [
        "Dense_Ts_0",
        "Dense_Ts_1",
        "Dense_Ts_2",
        "Dense_Ts_3",
        "Dense_Ts_4",
        "Dense_Ts_final",
    ]
    ts_pt_names = ["ts_0", "ts_1", "ts_2", "ts_3", "ts_4", "ts_final"]
    for tf_name, pt_name in zip(ts_tf_names, ts_pt_names, strict=False):
        w, b = convert_dense_weights(tf_weights[tf_name])
        getattr(pt_model, pt_name).weight.data = w
        getattr(pt_model, pt_name).bias.data = b

    # discont head
    dc_tf_names = [
        "Dense_discont_0",
        "Dense_discont_1",
        "Dense_discont_2",
        "Dense_discont_3",
        "Dense_discont_4",
        "Dense_discont_final",
    ]
    dc_pt_names = [
        "discont_0",
        "discont_1",
        "discont_2",
        "discont_3",
        "discont_4",
        "discont_final",
    ]
    for tf_name, pt_name in zip(dc_tf_names, dc_pt_names, strict=False):
        w, b = convert_dense_weights(tf_weights[tf_name])
        getattr(pt_model, pt_name).weight.data = w
        getattr(pt_model, pt_name).bias.data = b

    # tau head
    tau_tf_names = ["Dense_tau_0", "Dense_tau_1", "Dense_tau_2", "Dense_tau_final"]
    tau_pt_names = ["tau_0", "tau_1", "tau_2", "tau_final"]
    for tf_name, pt_name in zip(tau_tf_names, tau_pt_names, strict=False):
        w, b = convert_dense_weights(tf_weights[tf_name])
        getattr(pt_model, pt_name).weight.data = w
        getattr(pt_model, pt_name).bias.data = b

    # UVLF head
    uvlf_tf_names = [
        "Dense_UVLF_0",
        "Dense_UVLF_1",
        "Dense_UVLF_2",
        "Dense_UVLF_3",
        "Dense_UVLF_4",
        "Dense_UVLF_final",
    ]
    uvlf_pt_names = ["uvlf_0", "uvlf_1", "uvlf_2", "uvlf_3", "uvlf_4", "uvlf_final"]
    for tf_name, pt_name in zip(uvlf_tf_names, uvlf_pt_names, strict=False):
        w, b = convert_dense_weights(tf_weights[tf_name])
        getattr(pt_model, pt_name).weight.data = w
        getattr(pt_model, pt_name).bias.data = b

    # PS Conv2DTranspose layers
    ps_conv_mapping = [
        ("PS_Conv2DTr_0", "ps_conv_0"),
        ("PS_Conv2DTr_1", "ps_conv_1"),
        ("PS_Conv2DTr_2", "ps_conv_2"),
        ("PS_Conv2DTr_3", "ps_conv_3"),
        ("PS_Conv2DTr_4", "ps_conv_4"),
        ("PS_Conv2DTr_5", "ps_conv_5"),
        ("PS_Conv2DTr_6", "ps_conv_6"),
        ("PS_Conv2DTr_7", "ps_conv_7"),
        ("PS_Conv2DTr_8", "ps_conv_8"),
        ("PS_Conv2DTr_10", "ps_conv_10"),
        ("PS_Conv2DTr_11", "ps_conv_11"),
        ("PS_Conv2DTr_13", "ps_conv_13"),
        ("PS_Conv2DTr_14", "ps_conv_14"),
    ]
    for tf_name, pt_name in ps_conv_mapping:
        w, b = convert_conv_transpose_weights(tf_weights[tf_name])
        getattr(pt_model, pt_name).weight.data = w
        getattr(pt_model, pt_name).bias.data = b


def convert_and_save(
    tf_model_path: str,
    output_path: str,
    validate: bool = True,
    verbose: bool = True,
):
    """Convert TensorFlow v1 model to PyTorch and save.

    Parameters
    ----------
    tf_model_path : str
        Path to TensorFlow SavedModel directory.
    output_path : str
        Path to save PyTorch model (.pt file).
    validate : bool
        Whether to validate conversion against TF outputs.
    verbose : bool
        Whether to print progress.
    """
    if verbose:
        print(f"Loading TensorFlow model from {tf_model_path}...")

    tf_model = load_tf_model(tf_model_path)
    tf_weights = extract_tf_weights(tf_model)

    if verbose:
        print(f"Extracted weights from {len(tf_weights)} layers")

    # Create PyTorch model
    pt_model = DefaultEmulatorV1(negative_slope=0.1)

    if verbose:
        print("Transferring weights...")

    transfer_weights(pt_model, tf_weights)

    if validate:
        if verbose:
            print("Validating conversion...")

        # Generate test input
        np.random.seed(42)
        test_input = np.random.rand(5, 9).astype(np.float32)

        # TensorFlow prediction
        tf_output = tf_model.predict(test_input, verbose=0)

        # PyTorch prediction
        pt_model.eval()
        with torch.no_grad():
            pt_output = pt_model(torch.from_numpy(test_input)).numpy()

        # Compare outputs
        max_diff = np.abs(tf_output - pt_output).max()
        mean_diff = np.abs(tf_output - pt_output).mean()

        if verbose:
            print(f"Max absolute difference: {max_diff:.6e}")
            print(f"Mean absolute difference: {mean_diff:.6e}")

        # Per-output validation
        idx = 0
        for name, size in DefaultEmulatorV1.OUTPUT_SIZES.items():
            tf_slice = tf_output[:, idx : idx + size]
            pt_slice = pt_output[:, idx : idx + size]
            diff = np.abs(tf_slice - pt_slice).max()
            if verbose:
                print(f"  {name:10s}: max_diff = {diff:.6e}")
            idx += size

        if max_diff > 1e-4:
            log.warning(f"Large difference detected: {max_diff}")

    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": pt_model.state_dict(),
        "model_class": "DefaultEmulatorV1",
        "negative_slope": 0.1,
        "output_sizes": dict(pt_model.OUTPUT_SIZES),
    }
    torch.save(checkpoint, output_path)

    if verbose:
        print(f"Saved PyTorch model to {output_path}")

    return pt_model


def load_pytorch_model(model_path: str, device: str = "cpu") -> DefaultEmulatorV1:
    """Load converted PyTorch model.

    Parameters
    ----------
    model_path : str
        Path to PyTorch model (.pt file).
    device : str
        Device to load model on.

    Returns
    -------
    DefaultEmulatorV1
        Loaded model in eval mode.
    """
    checkpoint = torch.load(model_path, map_location=device)

    model = DefaultEmulatorV1(negative_slope=checkpoint.get("negative_slope", 0.1))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    parser = argparse.ArgumentParser(description="Convert v1 TF model to PyTorch")
    parser.add_argument(
        "--input",
        default=os.path.expanduser("~/.local/share/py21cmEMU/21cmEMU/21cmEMU"),
        help="Path to TensorFlow SavedModel",
    )
    parser.add_argument(
        "--output", default="default_model.pt", help="Output PyTorch model path"
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")

    args = parser.parse_args()

    convert_and_save(
        args.input,
        args.output,
        validate=not args.no_validate,
        verbose=True,
    )
