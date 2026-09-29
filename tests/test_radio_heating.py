"""Tests for the radio-heating emulators ``rh`` and ``norh``.

The trained weights are not part of the repository, so the prediction
pipeline is tested with randomly initialised networks saved to a temporary
file; tests that need the real weights are skipped when they are absent.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from py21cmemu import (
    Emulator,
    NoRHEmulatorErrors,
    NoRHEmulatorInput,
    NoRHEmulatorOutput,
    RHEmulatorErrors,
    RHEmulatorInput,
    RHEmulatorOutput,
)
from py21cmemu.models.radio_heating.model import (
    SummaryEmulator,
    find_weights,
    weights_search_paths,
)
from py21cmemu.properties import (
    NoRHEmulatorProperties,
    RHEmulatorProperties,
    emulator_properties,
    resolve_emulator_name,
)

NAMES = ("rh", "norh")
INPUTS = {"rh": RHEmulatorInput, "norh": NoRHEmulatorInput}
OUTPUTS = {"rh": RHEmulatorOutput, "norh": NoRHEmulatorOutput}
ERRORS = {"rh": RHEmulatorErrors, "norh": NoRHEmulatorErrors}
PARAM_KEYS = (
    "fR_mini",
    "F_STAR7_MINI",
    "L_X_MINI",
    "F_ESC7_MINI",
    "m_chi",
    "sigma_SDM",
)


def _real_weights_available(name: str) -> bool:
    return any(p.exists() for p in weights_search_paths(name))


@pytest.fixture(scope="module", params=NAMES)
def name(request) -> str:
    """Emulator name."""
    return request.param


@pytest.fixture(scope="module")
def random_weights(name, tmp_path_factory):
    """Randomly initialised weights with the architecture of ``name``."""
    props = emulator_properties(name)
    torch.manual_seed(0)
    model = SummaryEmulator.from_config(props.model_config)
    model.set_redshifts(props.redshifts)
    path = tmp_path_factory.mktemp(name) / f"{name}_random_weights.pt"
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture(scope="module")
def random_emulator(name, random_weights) -> Emulator:
    """Emulator with random weights (tests the pipeline, not the accuracy)."""
    return Emulator(emulator=name, weights_path=random_weights)


def _sample_params(name: str, n: int, seed: int = 1) -> np.ndarray:
    lims = emulator_properties(name).limits
    u = np.random.default_rng(seed).random((n, len(lims)))
    return lims[:, 0] + u * (lims[:, 1] - lims[:, 0])


# ══════════════════════════════════════════════════════════════════════════════
# Names and properties
# ══════════════════════════════════════════════════════════════════════════════


def test_names_and_aliases():
    """Canonical names, aliases and case-insensitivity."""
    assert resolve_emulator_name("rh") == "rh"
    assert resolve_emulator_name("RH") == "rh"
    assert resolve_emulator_name("radio_heating") == "rh"
    assert resolve_emulator_name("soft_photon_heating") == "rh"
    assert resolve_emulator_name("sph") == "rh"
    assert resolve_emulator_name("norh") == "norh"
    assert resolve_emulator_name("noRH") == "norh"
    assert resolve_emulator_name("no_radio_heating") == "norh"
    assert resolve_emulator_name("no_rh") == "norh"


def test_properties_classes():
    """The factory returns the right properties class."""
    assert isinstance(emulator_properties("rh"), RHEmulatorProperties)
    assert isinstance(emulator_properties("norh"), NoRHEmulatorProperties)


def test_properties_content(name):
    """Constants file content: parameters, redshifts, normalisation, errors."""
    p = emulator_properties(name)
    assert p.name == name
    assert p.astro_param_keys == PARAM_KEYS
    assert p.limits.shape == (6, 2)
    assert np.all(p.limits[:, 1] > p.limits[:, 0])
    assert p.redshifts.shape == (110,)
    assert np.all(np.diff(p.redshifts) > 0)
    np.testing.assert_array_equal(p.zs, p.redshifts)
    assert set(p.targets) == {"xHI", "Tb", "Tr", "tau"}
    assert p.model_config["n_params"] == 6
    assert p.model_config["n_z"] == 110
    assert p.normalized_quantities == ["Tb", "Tr", "tau"]
    for t in ("xHI", "Tb", "Tr"):
        for s in ("med_err", "mean_err", "std_err", "p68_err", "p95_err"):
            arr = getattr(p, f"{t}_{s}")
            assert arr.shape == (110,)
            assert np.all(np.isfinite(arr))
            assert np.all(arr >= 0)
    assert isinstance(p.tau_med_err, float)
    # medians below 5% everywhere in the global sense
    for t in p.targets:
        assert 0 <= getattr(p, f"{t}_global_med_err") < 5
    assert set(p.fe_floors) >= {"xHI", "Tb", "Tr", "tau"}


def test_radio_heating_flag():
    """rh and norh differ by the radio-heating flag of the training set."""
    assert emulator_properties("rh").flag_options["USE_RADIO_HEATING"] is True
    assert emulator_properties("norh").flag_options["USE_RADIO_HEATING"] is False


def test_normalisation_round_trip(name):
    """normalise and denormalise are inverse of each other."""
    p = emulator_properties(name)
    z = p.redshifts
    samples = {
        "xHI": np.clip(np.linspace(0, 1, z.size), 0, 1)[None],
        "Tb": np.linspace(-1e4, 30, z.size)[None],
        "Tr": np.logspace(-3, 5, z.size)[None],
        "tau": np.array([0.02, 0.05, 0.3]),
    }
    for t, x in samples.items():
        y = p.normalise(t, x)
        back = p.denormalise(t, y)
        np.testing.assert_allclose(back, x, rtol=1e-8, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# Inputs
# ══════════════════════════════════════════════════════════════════════════════


def test_input_normalize_round_trip(name):
    """Prior edges map to 0 and 1, and normalisation is invertible."""
    inp = INPUTS[name]()
    assert inp.astro_param_keys == PARAM_KEYS
    assert set(inp.LOG_PARAMETERS) == set(PARAM_KEYS)
    lims = inp.properties.limits
    np.testing.assert_allclose(inp.normalize(lims.T), [[0] * 6, [1] * 6])
    theta = _sample_params(name, 4)
    np.testing.assert_allclose(inp.undo_normalization(inp.normalize(theta)), theta)


def test_input_dict_and_array_agree(name):
    """Dict and array inputs give the same normalised parameters."""
    inp = INPUTS[name]()
    theta = _sample_params(name, 1)[0]
    as_dict = dict(zip(PARAM_KEYS, theta, strict=True))
    np.testing.assert_allclose(
        inp.make_param_array(as_dict), inp.make_param_array(theta)
    )
    with pytest.raises(ValueError, match="not the correct length"):
        inp.make_param_array(theta[:5])


# ══════════════════════════════════════════════════════════════════════════════
# Model and prediction pipeline (random weights)
# ══════════════════════════════════════════════════════════════════════════════


def test_model_forward_shapes(name):
    """The network returns one normalised array per summary."""
    p = emulator_properties(name)
    model = SummaryEmulator.from_config(p.model_config)
    with torch.no_grad():
        out = model(torch.rand(3, 6))
    assert out["xHI"].shape == (3, 110)
    assert out["Tb"].shape == (3, 110)
    assert out["Tr"].shape == (3, 110)
    assert out["tau"].shape == (3,)
    assert torch.all((out["xHI"] >= 0) & (out["xHI"] <= 1))


def test_missing_weights_raise(name, tmp_path):
    """A missing weights file raises a helpful FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        Emulator(emulator=name, weights_path=tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError):
        find_weights(name, tmp_path / "nope.pt")


def test_predict_pipeline(name, random_emulator):
    """predict() returns outputs with units and matching error estimates."""
    theta = _sample_params(name, 3)
    normed, out, errors = random_emulator.predict(theta)
    assert isinstance(out, OUTPUTS[name])
    assert isinstance(errors, ERRORS[name])
    np.testing.assert_allclose(normed, INPUTS[name]().normalize(theta))
    for t in ("xHI", "Tb", "Tr"):
        assert getattr(out, t).shape == (3, 110)
        assert np.all(np.isfinite(getattr(out, t).value))
    assert out.tau.shape == (3,)
    assert out.Tb.unit == "mK"
    assert out.Tr.unit == "K"
    assert np.all(out.Tr.value >= 0)
    assert np.all((out.xHI.value >= 0) & (out.xHI.value <= 1))
    assert np.all(out.tau.value > 0)
    np.testing.assert_array_equal(out.redshifts.value, random_emulator.redshifts)
    for key in ("Tb_err", "xHI_err", "Tr_err", "tau_err", "Tb_fe", "tau_fe"):
        assert key in errors
    assert errors["Tb_err"].shape == (3, 110)
    assert errors["Tb_err"].unit == "mK"
    assert errors["Tb_fe"].unit == "%"
    assert np.all(errors["Tb_err"].value >= 0)
    assert "Tb_err" in errors.summary()


def test_predict_single_and_batched(name, random_emulator):
    """Single inputs are squeezed; chunked evaluation gives the same result."""
    theta = _sample_params(name, 5)
    _, out, _ = random_emulator.predict(theta)
    _, out_b, _ = random_emulator.predict(theta, n_lstm_batch=2)
    for t in ("xHI", "Tb", "Tr", "tau"):
        np.testing.assert_allclose(
            getattr(out_b, t).value, getattr(out, t).value, rtol=1e-5, atol=1e-6
        )
    as_dict = dict(zip(PARAM_KEYS, theta[0], strict=True))
    _, out1, _ = random_emulator.predict(as_dict)
    assert out1.Tb.shape == (110,)
    assert out1.tau.shape == ()
    np.testing.assert_allclose(out1.Tb.value, out.Tb.value[0], rtol=1e-5, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Real weights (skipped unless the weights file is installed)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("emu_name", NAMES)
def test_real_weights_physical_output(emu_name):
    """With the trained weights, outputs are physically sensible."""
    if not _real_weights_available(emu_name):
        pytest.skip(f"{emu_name} weights not installed")
    emu = Emulator(emulator=emu_name)
    lims = emu.properties.limits
    theta = lims.mean(axis=1)[None]  # centre of the prior
    _, out, _ = emu.predict(theta)
    z = out.redshifts.value
    # fully neutral and no radio background at the highest redshifts
    assert np.all(out.xHI.value[z > 40] > 0.99)
    assert np.all(out.Tr.value[z > 40] < 1e-2)
    # dark-ages absorption at z ~ 45-49
    assert np.all(out.Tb.value[z > 45] < 0)
    assert 0.01 < float(out.tau.value) < 0.4
