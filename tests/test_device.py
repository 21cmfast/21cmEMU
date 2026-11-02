"""Tests for torch device selection (CUDA, Apple-silicon MPS, CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from py21cmemu import Emulator, utils
from py21cmemu.utils import DEVICE_ENV_VAR, get_device, get_gpu_count


@pytest.fixture
def no_device_env(monkeypatch):
    """Make sure the device environment variable does not leak into a test."""
    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)


@pytest.fixture
def fake_hardware(monkeypatch, no_device_env):
    """Pretend that the given accelerators are (un)available."""

    def _set(cuda: bool, mps: bool, n_cuda: int = 1) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n_cuda if cuda else 0)
        monkeypatch.setattr(utils, "mps_is_available", lambda: mps)

    return _set


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_get_device_order(fake_hardware, cuda, mps, expected):
    """CUDA is preferred over MPS, and MPS over the CPU."""
    fake_hardware(cuda=cuda, mps=mps)
    assert get_device().type == expected


def test_get_device_explicit(fake_hardware):
    """An explicit device wins over the automatic choice."""
    fake_hardware(cuda=False, mps=True)
    assert get_device("cpu") == torch.device("cpu")
    assert get_device(torch.device("cpu")) == torch.device("cpu")


def test_get_device_env(fake_hardware, monkeypatch):
    """The environment variable sets the default, an explicit device still wins."""
    fake_hardware(cuda=False, mps=True)
    monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")
    assert get_device() == torch.device("cpu")
    monkeypatch.setenv(DEVICE_ENV_VAR, "")  # empty means unset
    assert get_device().type == "mps"


@pytest.mark.parametrize(
    ("cuda", "mps", "n_cuda", "expected"),
    [(True, True, 4, 4), (False, True, 0, 1), (False, False, 0, 0)],
)
def test_get_gpu_count(fake_hardware, cuda, mps, n_cuda, expected):
    """Every CUDA card counts; Apple silicon has a single GPU."""
    fake_hardware(cuda=cuda, mps=mps, n_cuda=n_cuda)
    assert get_gpu_count() == expected


def test_emulator_device_argument(no_device_env):
    """The emulator and its model live on the requested device."""
    emu = Emulator(emulator="acg", device="cpu")
    assert emu.device == torch.device("cpu")
    assert next(emu.model.parameters()).device == torch.device("cpu")


def test_emulator_device_env(monkeypatch):
    """Without a device argument the emulator follows the environment variable."""
    monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")
    assert Emulator(emulator="acg").device == torch.device("cpu")


@pytest.mark.skipif(not utils.mps_is_available(), reason="needs an Apple-silicon GPU")
def test_acg_mps_matches_cpu(no_device_env):
    """The acg emulator gives the same predictions on the Apple GPU as on the CPU.

    Both run in float32 with different kernels, so they agree to rounding: within
    1e-4 of each output's largest value. (A per-element relative tolerance fails
    where an output crosses zero, e.g. log10 UV LFs.)
    """
    cpu_emu = Emulator(emulator="acg", device="cpu")
    rng = np.random.default_rng(0)
    theta = cpu_emu.inputs.undo_normalization(rng.random((16, 9)))  # training range
    _, cpu, _ = cpu_emu.predict(theta)
    _, mps, _ = Emulator(emulator="acg", device="mps").predict(theta)
    for name in ("Tb", "xHI", "Ts", "PS", "tau", "UVLFs"):
        expected = np.asarray(getattr(cpu, name), dtype=float)
        np.testing.assert_allclose(
            np.asarray(getattr(mps, name), dtype=float),
            expected,
            rtol=0,
            atol=1e-4 * np.nanmax(np.abs(expected)),
            err_msg=name,
        )
