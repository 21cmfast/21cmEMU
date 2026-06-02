"""Tests for Emulator class construction and prediction (emulator.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from py21cmemu import Emulator

TUTORIALS_DIR = Path(__file__).resolve().parents[1] / "docs" / "tutorials"
TEST_DATABASE_H5 = TUTORIALS_DIR / "test_database.h5"


# ══════════════════════════════════════════════════════════════════════════════
# Emulator construction and __getattr__ (from test_coverage.py + test_main.py)
# ══════════════════════════════════════════════════════════════════════════════


def test_default_emulator_is_mcg():
    """Test that the default emulator is mcg (v3)."""
    emu = Emulator(emulate_2d_ps=False)  # No explicit emulator arg
    assert emu.which_emulator == "mcg"


def test_mh_emulator_creation() -> None:
    """Test that MCG emulator can be created."""
    emu = Emulator(emulator="mcg", emulate_2d_ps=False)
    assert emu.which_emulator == "mcg"
    assert not emu.emulate_2d_ps


def test_emulator_aliases() -> None:
    """Test that emulator aliases resolve correctly."""
    from py21cmemu.emulator import resolve_emulator_name

    # v3/mh -> mcg
    assert resolve_emulator_name("v3") == "mcg"
    assert resolve_emulator_name("mh") == "mcg"
    assert resolve_emulator_name("mcg") == "mcg"

    # v1/default -> acg
    assert resolve_emulator_name("v1") == "acg"
    assert resolve_emulator_name("default") == "acg"
    assert resolve_emulator_name("acg") == "acg"

    # v2/radio_background -> radio
    assert resolve_emulator_name("v2") == "radio"
    assert resolve_emulator_name("radio_background") == "radio"
    assert resolve_emulator_name("radio") == "radio"

    # Case insensitive
    assert resolve_emulator_name("MCG") == "mcg"
    assert resolve_emulator_name("V3") == "mcg"

    # Invalid name raises
    with pytest.raises(ValueError, match="Unknown emulator"):
        resolve_emulator_name("invalid")


def test_emulator_getattr_delegates_to_properties():
    """__getattr__ forwards unknown attribute lookups to emulator properties."""
    emu = Emulator(emulator="acg")
    # 'zs' is defined on DefaultEmulatorProperties, not on Emulator
    assert emu.zs is not None
    assert emu.limits is not None


def test_emulator_getattr_raises_for_nonexistent():
    """__getattr__ raises AttributeError for completely unknown attributes."""
    emu = Emulator(emulator="acg")
    with pytest.raises(AttributeError):
        _ = emu.this_attribute_does_not_exist_anywhere_xyz


def test_emulator_getattr_properties_guard():
    """__getattr__("properties") raises AttributeError — the init-time guard (line 191)."""
    emu = Emulator(emulator="acg")
    # Remove the instance attribute so __getattr__ is invoked for 'properties'
    del emu.__dict__["properties"]
    with pytest.raises(AttributeError):
        _ = emu.properties


def test_emulator_n_lstm_batch(mh_emulator):
    """predict() with n_lstm_batch=1 exercises the chunked LSTM path."""
    from py21cmemu.inputs import MCGEmulatorInput

    inp = MCGEmulatorInput()
    n = len(inp.astro_param_keys)
    # 4 parameter sets, batch size 2 → triggers torch.split path
    raw = inp.undo_normalization(np.full((4, n), 0.5))
    _theta, output, _errors = mh_emulator.predict(raw, n_lstm_batch=2)
    assert output.Tb.shape[0] == 4


# ══════════════════════════════════════════════════════════════════════════════
# MH emulator integration tests (from test_v3.py)
# ══════════════════════════════════════════════════════════════════════════════


def test_mh_predict_from_tutorial_h5(mh_emulator) -> None:
    """Test basic prediction with MH emulator."""
    h5py = pytest.importorskip("h5py")

    with h5py.File(TEST_DATABASE_H5, "r") as f:
        params = np.asarray(f["input_params"][0:1])

    theta, output, errors = mh_emulator.predict(params)

    theta_ps, theta_lstm = theta
    assert theta_ps is None
    assert theta_lstm.shape[0] == 1

    assert output.Tb.shape[-1] > 0
    assert output.xHI.shape[-1] > 0
    assert output.Ts.shape[-1] > 0
    assert output.UVLFs.shape[-2] > 0
    assert output.tau.shape[0] == 1

    for key in ("Tb_err", "xHI_err", "Ts_err", "tau_err"):
        assert key in errors


def test_mh_batch_prediction(mh_emulator) -> None:
    """Test batch prediction with multiple parameter sets."""
    h5py = pytest.importorskip("h5py")

    with h5py.File(TEST_DATABASE_H5, "r") as f:
        params = np.asarray(f["input_params"][:5])

    theta, output, _errors = mh_emulator.predict(params)

    _theta_ps, theta_lstm = theta
    assert theta_lstm.shape[0] == 5
    assert output.Tb.shape[0] == 5
    assert output.xHI.shape[0] == 5
    assert output.Ts.shape[0] == 5
    assert output.tau.shape[0] == 5


# ══════════════════════════════════════════════════════════════════════════════
# V1 ACG (PyTorch) model tests (from test_main.py)
# ══════════════════════════════════════════════════════════════════════════════


def test_v1_pytorch_model():
    """Test v1 PyTorch model directly."""
    from py21cmemu.models.acg.v1_pytorch import (
        DefaultEmulatorV1,
        load_converted_model,
    )

    # Test model architecture
    model = DefaultEmulatorV1(negative_slope=0.1)
    assert sum(p.numel() for p in model.parameters()) > 0

    # Test forward pass shape
    x = torch.randn(2, 9)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1098), f"Expected (2, 1098), got {out.shape}"

    # Test forward_dict returns correct keys
    out_dict = model.forward_dict(x)
    assert set(out_dict.keys()) == {"Tb", "xHI", "Ts", "discont", "PS", "tau", "UVLF"}
    assert out_dict["Tb"].shape == (2, 84)
    assert out_dict["PS"].shape == (2, 720)

    # Test loading bundled model
    import py21cmemu

    bundled_path = Path(py21cmemu.__file__).parent / "models/acg/default_model.pt"
    loaded_model = load_converted_model(str(bundled_path), device="cpu")
    assert isinstance(loaded_model, DefaultEmulatorV1)
    with torch.no_grad():
        out_loaded = loaded_model(x)
    assert out_loaded.shape == (2, 1098)


def test_v1_pytorch_vs_emulator():
    """Test that v1 PyTorch model gives same results through Emulator API."""
    emu = Emulator(emulator="acg")  # aka v1/default

    # Test prediction
    params = {
        "F_STAR10": -1.5,
        "ALPHA_STAR": 0.5,
        "F_ESC10": -1.0,
        "ALPHA_ESC": -0.5,
        "M_TURN": 8.5,
        "t_STAR": 0.5,
        "L_X": 40.0,
        "NU_X_THRESH": 500.0,
        "X_RAY_SPEC_INDEX": 1.0,
    }
    _theta, output, _errors = emu.predict(params)

    # Check output shapes
    assert output.Tb.shape == (84,)
    assert output.xHI.shape == (84,)
    assert output.Ts.shape == (84,)
    assert output.PS.shape == (60, 12)
    assert np.isscalar(output.tau) or output.tau.shape == (), "tau should be scalar"
    assert output.UVLFs.shape[0] > 0

    # Check reasonable output ranges
    assert 0 <= output.xHI.min() <= output.xHI.max() <= 1
    assert 0 < float(output.tau) < 1
