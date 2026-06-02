"""Test cases for the __main__ module."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from typeguard import suppress_type_checks

from py21cmemu import DefaultEmulatorInput, Emulator, RadioEmulatorInput
from py21cmemu.config import CONFIG
from py21cmemu.outputs import DefaultRawEmulatorOutput
from py21cmemu.properties import emulator_properties


def test_default_emulator_is_mcg():
    """Test that the default emulator is mcg (v3)."""
    emu = Emulator(emulate_2d_ps=False)  # No explicit emulator arg
    assert emu.which_emulator == "mcg"


@pytest.mark.parametrize("emu_type", ["default", "radio_background"])
def test_output(tmp_path, emu_type):
    """Test outputs.py and emulator.py."""
    if emu_type == "radio_background":
        npars = 5
    else:
        npars = 9
    emu = Emulator(emulator=emu_type)
    # Generate physical params via undo_normalization so input is valid for the emulator
    if emu_type == "radio_background":
        theta = RadioEmulatorInput().undo_normalization(
            np.random.rand(npars * 5).reshape((5, npars))
        )
    else:
        theta = DefaultEmulatorInput().undo_normalization(
            np.random.rand(npars * 5).reshape((5, npars))
        )

    theta, output, errors = emu.predict(theta)

    # Test writing
    write_dir = tmp_path / "sub"
    write_dir.mkdir()
    output.write(write_dir / "test_writing", theta=theta, store=None)
    check = np.load(write_dir / "test_writing.npz", allow_pickle=True)["arr_0"].item()

    assert (check["inputs"] == theta).all()
    output_keys = []
    for i in output.keys():
        output_keys.append(i)
    assert len(check.keys()) == len(output_keys) + 1
    # Compare raw values (output.PS is a Quantity with units)
    ps_values = output.PS.value if hasattr(output.PS, "value") else output.PS
    assert (check["PS"] == ps_values).all()

    with pytest.raises(ValueError):
        output.write(write_dir / "test_writing.npz", clobber=False)

    # Test that setting store restricts what is written
    output.write(write_dir / "test_writing_small", store=["PS"])
    check = np.load(write_dir / "test_writing_small.npz", allow_pickle=True)[
        "arr_0"
    ].item()
    assert "xHI" not in check
    assert "theta" not in check
    if emu_type == "default":
        out2 = DefaultRawEmulatorOutput(np.random.rand(1098))
        with pytest.raises(ValueError):
            out2.renormalize("foo")

        assert np.all(output["xHI"] == output.xHI)

        output.k
        output.Muv
        output.UVLF_redshifts
        output.PS_redshifts
        output.redshifts

        # --- Error shape consistency for single-sample prediction ---
        # Error arrays must match output shapes for a 1-sample input.
        # Critical for MCMC samplers that pass 1 or many samples.
        # Do NOT call squeeze() before comparing.
        theta_single = DefaultEmulatorInput().undo_normalization(
            np.random.rand(9).reshape((1, 9))
        )
        _, out_single, err_single = emu.predict(theta_single)
        assert out_single.xHI.shape == err_single.xHI_err.shape, (
            f"ACG xHI shape mismatch: output {out_single.xHI.shape} vs error {err_single.xHI_err.shape}"
        )
        assert out_single.Tb.shape == err_single.Tb_err.shape, (
            f"ACG Tb shape mismatch: output {out_single.Tb.shape} vs error {err_single.Tb_err.shape}"
        )
        assert out_single.Ts.shape == err_single.Ts_err.shape, (
            f"ACG Ts shape mismatch: output {out_single.Ts.shape} vs error {err_single.Ts_err.shape}"
        )
        assert out_single.tau.shape == err_single.tau_err.shape, (
            f"ACG tau shape mismatch: output {out_single.tau.shape} vs error {err_single.tau_err.shape}"
        )
        assert out_single.PS.shape == err_single.PS_err.shape, (
            f"ACG PS shape mismatch: output {out_single.PS.shape} vs error {err_single.PS_err.shape}"
        )
        assert out_single.UVLFs.shape == err_single.UVLFs_logerr.shape, (
            f"ACG UVLFs shape mismatch: output {out_single.UVLFs.shape} vs error {err_single.UVLFs_logerr.shape}"
        )

        # --- Error shape consistency for multi-sample prediction (n_params > 1) ---
        # Errors must be broadcast to (n_params, ...) so shapes always match the output.
        N = 5
        theta_multi = DefaultEmulatorInput().undo_normalization(
            np.random.rand(N * 9).reshape((N, 9))
        )
        _, out_multi, err_multi = emu.predict(theta_multi)
        assert out_multi.xHI.shape == err_multi.xHI_err.shape, (
            f"ACG xHI shape mismatch (n={N}): output {out_multi.xHI.shape} vs error {err_multi.xHI_err.shape}"
        )
        assert out_multi.Tb.shape == err_multi.Tb_err.shape, (
            f"ACG Tb shape mismatch (n={N}): output {out_multi.Tb.shape} vs error {err_multi.Tb_err.shape}"
        )
        assert out_multi.Ts.shape == err_multi.Ts_err.shape, (
            f"ACG Ts shape mismatch (n={N}): output {out_multi.Ts.shape} vs error {err_multi.Ts_err.shape}"
        )
        assert out_multi.tau.shape == err_multi.tau_err.shape, (
            f"ACG tau shape mismatch (n={N}): output {out_multi.tau.shape} vs error {err_multi.tau_err.shape}"
        )
        assert out_multi.PS.shape == err_multi.PS_err.shape, (
            f"ACG PS shape mismatch (n={N}): output {out_multi.PS.shape} vs error {err_multi.PS_err.shape}"
        )
        assert out_multi.UVLFs.shape == err_multi.UVLFs_logerr.shape, (
            f"ACG UVLFs shape mismatch (n={N}): output {out_multi.UVLFs.shape} vs error {err_multi.UVLFs_logerr.shape}"
        )
    else:
        errors["Tr_err"]
        output.Tr
        output.PS_ks

        with pytest.raises(ValueError):
            emu = Emulator(emulator="foo")


def test_properties():
    """Test that the properties are loaded correctly."""
    from py21cmemu.properties import emulator_properties

    # Default is now mcg (v3)
    properties = emulator_properties()
    assert hasattr(properties, "lstm_limits")  # MCG-specific

    # Test canonical names
    properties = emulator_properties(emulator="acg")
    properties.limits

    properties = emulator_properties(emulator="radio")
    properties.logTr_mean

    properties = emulator_properties(emulator="mcg")
    properties.lstm_limits

    # Test legacy aliases still work
    properties = emulator_properties(emulator="default")  # -> acg
    properties.limits

    properties = emulator_properties(emulator="radio_background")  # -> radio
    properties.logTr_mean

    properties = emulator_properties(emulator="mh")  # -> mcg
    properties.lstm_limits

    # Invalid name raises
    with pytest.raises(ValueError):
        properties = emulator_properties(emulator="foo")


@pytest.mark.parametrize("emu_type", ["default", "radio_background"])
def test_inputs(emu_type):
    """Test that we perform parameter normalization properly."""
    properties = emulator_properties(emulator=emu_type)

    if emu_type == "radio_background":
        emu_in = RadioEmulatorInput()
        limits = properties.limits.copy()
        npars = len(limits)
    else:
        emu_in = DefaultEmulatorInput()
        limits = properties.limits.copy()
        limits[7, :] *= 1000.0  # keV to eV for NU_X_THRESH
        npars = len(limits)

    rng = np.random.default_rng(42)

    # Generate physical params via undo_normalization of random [0,1] values
    single_phys = emu_in.undo_normalization(rng.random(npars).reshape(1, npars)).ravel()

    # normed=True: physical params → normalized to [0,1]
    inp = emu_in.make_param_array(single_phys, normed=True)
    assert inp.min() >= 0 and inp.max() <= 1, (
        "Single param 1D: normed=True should give [0,1]."
    )

    # normed=False: physical params returned unchanged and within limits
    inp = emu_in.make_param_array(single_phys, normed=False)
    assert np.allclose(inp.ravel(), single_phys), (
        "Single param 1D: normed=False should return unchanged."
    )
    assert (inp >= limits[:, 0]).all() and (inp <= limits[:, 1]).all(), (
        "Single param 1D: physical params not within limits."
    )

    # 2D array (1, npars)
    single_phys_2d = single_phys.reshape(1, npars)
    inp = emu_in.make_param_array(single_phys_2d, normed=True)
    assert inp.min() >= 0 and inp.max() <= 1, (
        "Single 2D param: normed=True should give [0,1]."
    )

    inp = emu_in.make_param_array(single_phys_2d, normed=False)
    assert (inp >= limits[:, 0]).all() and (inp <= limits[:, 1]).all(), (
        "Single 2D param: physical params not within limits."
    )

    # Batch (5, npars)
    many_phys = emu_in.undo_normalization(rng.random((5, npars)))
    inp = emu_in.make_param_array(many_phys, normed=True)
    assert inp.shape == (5, npars)
    assert inp.min() >= 0 and inp.max() <= 1, "Batch: normed=True should give [0,1]."

    inp = emu_in.make_param_array(many_phys, normed=False)
    assert np.array(
        [(i >= limits[:, 0]).all() and (i <= limits[:, 1]).all() for i in inp]
    ).all(), "Batch: physical params not within limits."

    # Single dict of physical params
    single_dict = {k: single_phys[i] for i, k in enumerate(emu_in.astro_param_keys)}
    inp = emu_in.make_param_array(single_dict, normed=True)
    assert inp.min() >= 0 and inp.max() <= 1, "Dict: normed=True should give [0,1]."

    inp = emu_in.make_param_array(single_dict, normed=False)
    assert (inp >= limits[:, 0]).all() and (inp <= limits[:, 1]).all(), (
        "Dict: normed=False physical params not within limits."
    )

    # List of dicts
    many_params_list = [single_dict, single_dict, single_dict]
    inp = emu_in.make_param_array(many_params_list, normed=False)
    assert np.array(
        [(i >= limits[:, 0]).all() and (i <= limits[:, 1]).all() for i in inp]
    ).all(), "List of dicts: physical params not within limits."

    many_params_arr = np.array([single_dict, single_dict, single_dict])
    inp = emu_in.make_param_array(many_params_arr, normed=False)
    assert np.array(
        [(i >= limits[:, 0]).all() and (i <= limits[:, 1]).all() for i in inp]
    ).all(), "Array of dicts: physical params not within limits."

    # List / list of lists
    arr_list = list(single_phys)
    inp = emu_in.make_param_array(arr_list, normed=True)
    assert inp.min() >= 0 and inp.max() <= 1, "List: normed=True should give [0,1]."

    many_params_list = [arr_list, arr_list, arr_list]
    inp = emu_in.make_param_array(many_params_list, normed=False)
    assert np.array(
        [(i >= limits[:, 0]).all() and (i <= limits[:, 1]).all() for i in inp]
    ).all(), "List of lists: physical params not within limits."

    # Roundtrip: normalize → undo_normalization recovers original physical params
    normed_out = emu_in.make_param_array(single_phys, normed=True)
    recovered = emu_in.undo_normalization(normed_out)
    assert np.allclose(recovered.ravel(), single_phys, rtol=1e-5), (
        "Roundtrip normalization failed."
    )

    # make_list_of_dicts
    phys_batch = emu_in.undo_normalization(rng.random((10, npars)))
    emu_in.make_list_of_dicts(phys_batch, normed=True)

    # Error: wrong number of params
    arr = rng.random((5, 10))
    with pytest.raises(ValueError):
        emu_in.make_param_array(arr, normed=True)

    with pytest.raises(TypeError):
        with suppress_type_checks():
            emu_in.make_param_array(7, normed=True)

    # Error: list of tuples is not a supported type
    arr_tup = [tuple(i) for i in emu_in.undo_normalization(rng.random((5, npars)))]
    with pytest.raises(TypeError):
        emu_in.make_param_array(arr_tup, normed=True)

    properties = emulator_properties("radio_background")


def test_config(tmp_path):
    """Test config.py."""
    from pathlib import Path

    from appdirs import AppDirs

    from py21cmemu.config import Config
    from py21cmemu.get_emulator import get_emu_data

    APPDIR = AppDirs("py21cmEMU")
    config_file = Path(APPDIR.user_config_dir) / "config.toml"
    Config(config_file=config_file)

    conf = Config(config_file=tmp_path / "foo.toml")
    assert conf.__str__() == str(conf.config)
    assert conf.__repr__() == repr(conf.config)
    get_emu_data()

    conf_keys = list(conf.keys())
    assert len(list(conf.items())) == 2
    assert len(list(conf.values())) == 2

    conf.__delitem__(key=conf_keys[0])
    assert len(list(conf.items())) == 1
    assert len(list(conf.values())) == 1

    # Change data-path to something that dne
    # for L40
    conf.__setitem__("data-path", tmp_path / "new")
    conf = Config(config_file=tmp_path / "foo.toml")


def test_get_emulator():
    """Test get_emulator.py."""
    import shutil

    # import git
    from py21cmemu.config import CONFIG
    from py21cmemu.get_emulator import get_emu_data

    version = "foo"
    with pytest.raises(
        ValueError,
    ):
        get_emu_data(version=version)

    get_emu_data(version="v1.0.0")
    # Modify the saved_model.pb file for the test to fail
    np.savetxt(
        CONFIG.data_path / "21cmEMU" / "21cmEMU" / "saved_model.pb", np.zeros(10)
    )
    with pytest.raises(RuntimeError):
        get_emu_data()
    shutil.rmtree(CONFIG.data_path / "21cmEMU")
    get_emu_data()


def test_get_emulator_no_internet():
    """Test get_emulator.py but when there is no internet."""
    from py21cmemu.get_emulator import get_emu_data

    # Temporarily move the huggingface repo
    if (CONFIG.data_path / "21cmEMU").exists():
        shutil.move(CONFIG.data_path / "21cmEMU", CONFIG.data_path / "21cmEMU_temp")

    # The data is there, but it cannot do pulls
    with pytest.raises(
        RuntimeError, match="The emulator huggingface repo was not cloned properly"
    ):
        with CONFIG.use(**{"disable-network": True}):
            get_emu_data()

    # Move the repo back
    if (CONFIG.data_path / "21cmEMU_temp").exists():
        shutil.move(CONFIG.data_path / "21cmEMU_temp", CONFIG.data_path / "21cmEMU")

    # Now, make sure the data is there
    get_emu_data()

    # Access again, but without pulling.
    with CONFIG.use(**{"disable-network": True}):
        with pytest.warns(
            UserWarning, match="Skipping the pulling step. Error received:"
        ):
            get_emu_data()


def test_v1_pytorch_model():
    """Test v1 PyTorch model directly."""
    from py21cmemu.models.ACG.v1_pytorch import (
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
    from pathlib import Path

    import py21cmemu

    bundled_path = Path(py21cmemu.__file__).parent / "models/ACG/default_model.pt"
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
    assert 0 <= output.xHI.min() <= output.xHI.max() <= 1, "xHI should be in [0,1]"
    assert 0 < float(output.tau) < 1, "tau should be small positive"


@pytest.mark.skipif(
    os.environ.get("CI_MERGE_TEST") != "1",
    reason="TF comparison test only runs on merge to main (set CI_MERGE_TEST=1)",
)
def test_v1_tensorflow_vs_pytorch_equivalence():
    """Test that PyTorch model produces identical outputs to original TensorFlow model.

    This test requires TensorFlow and only runs during merge to main in CI.
    """
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("TensorFlow not installed")

    from pathlib import Path

    import py21cmemu
    from py21cmemu.models.ACG.v1_pytorch import load_converted_model

    # Load TensorFlow model from HuggingFace cache
    tf_model_path = CONFIG.emu_path
    if not (tf_model_path / "saved_model.pb").exists():
        pytest.skip("TensorFlow model not available")

    tf_model = tf.keras.models.load_model(str(tf_model_path), compile=False)

    # Load PyTorch model
    bundled_path = Path(py21cmemu.__file__).parent / "models/ACG/default_model.pt"
    pt_model = load_converted_model(str(bundled_path), device="cpu")
    pt_model.eval()

    # Generate test inputs
    np.random.seed(42)
    test_input = np.random.rand(10, 9).astype(np.float32)

    # TensorFlow prediction
    tf_output = tf_model.predict(test_input, verbose=0)

    # PyTorch prediction
    with torch.no_grad():
        pt_output = pt_model(torch.from_numpy(test_input)).numpy()

    # Compare outputs - should be nearly identical (within floating point precision)
    max_diff = np.abs(tf_output - pt_output).max()
    mean_diff = np.abs(tf_output - pt_output).mean()

    assert max_diff < 1e-4, f"Max difference {max_diff} exceeds tolerance 1e-4"
    assert mean_diff < 1e-5, f"Mean difference {mean_diff} exceeds tolerance 1e-5"


# ═══════════════════════════════════════════════════════════════════════════════
# ACCURACY COMPARISON TESTS
# ═══════════════════════════════════════════════════════════════════════════════

TUTORIALS_DIR = Path(__file__).resolve().parents[1] / "docs" / "tutorials"
V1_TEST_DATA = TUTORIALS_DIR / "Test_data_sample.npz"


@pytest.mark.skipif(
    not V1_TEST_DATA.exists(), reason="Test_data_sample.npz not available"
)
def test_v1_emulator_vs_database():
    """Compare v1 emulator predictions against 21cmFAST database samples.

    This test verifies that emulated outputs are within expected tolerances
    compared to actual 21cmFAST simulation outputs.
    """
    # Load test data
    test_data = np.load(V1_TEST_DATA, allow_pickle=True)
    X_test = test_data["X_test"]  # (100, 9) normalized params

    # Ground truth
    xHI_true = test_data["xHI"]  # (100, 84)
    Tb_true = test_data["Tb"]  # (100, 84) in mK
    tau_true = test_data["tau"]  # (100,) log10(tau)
    PS_true = test_data["PS"]  # (100, 60, 12)

    # Run emulator
    emu = Emulator(emulator="acg")  # Test ACG (v1) emulator specifically
    # X_test is stored in [0,1] normalized space; convert to physical units first
    from py21cmemu.inputs import DefaultEmulatorInput

    X_test_phys = DefaultEmulatorInput().undo_normalization(X_test)
    _, output, _ = emu.predict(X_test_phys)

    # Calculate median fractional errors (%)
    def median_frac_err(true, pred, floor=1e-3):
        denom = np.abs(true)
        denom = np.where(denom < floor, floor, denom)
        fe = np.abs((true - pred) / denom) * 100
        return np.nanmedian(fe)

    # xHI: Should be very accurate where xHI > 0.01
    mask = xHI_true > 0.01
    xHI_vals = output.xHI.value if hasattr(output.xHI, "value") else output.xHI
    xHI_fe = median_frac_err(xHI_true[mask], xHI_vals[mask])
    assert xHI_fe < 5, f"xHI median FE {xHI_fe:.2f}% exceeds 5%"

    # Tb: Median FE should be < 10% for most cases
    Tb_vals = output.Tb.value if hasattr(output.Tb, "value") else output.Tb
    Tb_fe = median_frac_err(Tb_true, Tb_vals, floor=1.0)
    assert Tb_fe < 15, f"Tb median FE {Tb_fe:.2f}% exceeds 15%"

    # tau: Log-space comparison
    tau_vals = output.tau.value if hasattr(output.tau, "value") else output.tau
    tau_fe = median_frac_err(tau_true, np.log10(tau_vals), floor=0.01)
    assert tau_fe < 5, f"tau median FE {tau_fe:.2f}% exceeds 5%"

    # PS: Log power spectrum (test data is log10, emulator returns linear)
    PS_vals = output.PS.value if hasattr(output.PS, "value") else output.PS
    PS_emu_log = np.log10(PS_vals)
    PS_fe = median_frac_err(PS_true, PS_emu_log)
    assert PS_fe < 20, f"PS median FE {PS_fe:.2f}% exceeds 20%"

    print(
        f"V1 accuracy: xHI={xHI_fe:.2f}%, Tb={Tb_fe:.2f}%, tau={tau_fe:.2f}%, PS={PS_fe:.2f}%"
    )
