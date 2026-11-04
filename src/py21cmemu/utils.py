from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

#: Environment variable that sets the default torch device, e.g. ``cpu`` or ``mps``.
DEVICE_ENV_VAR = "PY21CMEMU_DEVICE"


def mps_is_available() -> bool:
    """Return whether torch can use an Apple-silicon GPU through the MPS backend."""
    mps = getattr(torch.backends, "mps", None)  # torch < 1.12 has no MPS backend
    return mps is not None and mps.is_available()


def get_device(device: str | torch.device | None = None) -> torch.device:
    """Return the torch device to run the emulators on.

    Parameters
    ----------
    device : str or torch.device, optional
        Explicit device, e.g. ``"cpu"``, ``"cuda:1"`` or ``"mps"``. If None, the
        ``PY21CMEMU_DEVICE`` environment variable is used when it is set; otherwise
        the first available of CUDA, Apple-silicon GPU (MPS) and CPU.

    Returns
    -------
    torch.device
        The selected device.
    """
    if device is None:
        device = os.environ.get(DEVICE_ENV_VAR) or None
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()


def get_gpu_count() -> int:
    """Return the number of GPUs available for sampling.

    CUDA reports every card; Apple silicon exposes a single GPU through MPS.
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if mps_is_available():
        return 1
    return 0


def empty_cache(device: str | torch.device) -> None:
    """Release cached GPU memory on ``device``; does nothing on the CPU."""
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.cpu().gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def transform(x, scale, bias):
    d = torch.log10(x)
    unit = (d - bias) / scale
    y = unit * 2 - 1  # scale to [-1, 1]
    return y


def reverse_transform(y, scale, bias):
    unit = (y + 1) / 2
    # Avoid numpy/torch interop (triggers __array_wrap__ DeprecationWarning in
    # NumPy 2.0 when a torch Tensor is multiplied by a numpy array).
    if isinstance(unit, torch.Tensor):
        scale = torch.as_tensor(scale, dtype=unit.dtype, device=unit.device)
        bias = torch.as_tensor(bias, dtype=unit.dtype, device=unit.device)
    d = unit * scale + bias
    return 10**d


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def restore_checkpoint(ckpt_dir, state, device):
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        ckpt_path.mkdir(exist_ok=True)
        logging.warning(
            f"No checkpoint found at {ckpt_dir}. Returned the same state as input"
        )
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state["optimizer"].load_state_dict(loaded_state["optimizer"])
        state["model"].load_state_dict(loaded_state["model"], strict=False)
        state["ema"].load_state_dict(loaded_state["ema"])
        state["step"] = loaded_state["step"]
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }
    torch.save(saved_state, ckpt_dir)
