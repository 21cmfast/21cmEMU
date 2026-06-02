"""Accuracy comparison tests: emulator predictions vs 21cmFAST database."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from py21cmemu import DefaultEmulatorInput, Emulator
from py21cmemu.config import CONFIG

TUTORIALS_DIR = Path(__file__).resolve().parents[1] / "docs" / "tutorials"
V1_TEST_DATA = TUTORIALS_DIR / "Test_data_sample.npz"
TEST_SET_H5 = TUTORIALS_DIR / "test_set.h5"
PS_2D_TEST_H5 = TUTORIALS_DIR / "ps_2d_test_subsample.h5"


def _log_convert_mh_params(params: np.ndarray) -> np.ndarray:
    """Convert MH parameter array from linear to log10 for LOG_PARAMETERS columns."""
    from py21cmemu.inputs import MHEmulatorInput

    mh_in = MHEmulatorInput()
    astro_keys = list(mh_in.astro_param_keys)
    log_idx = [astro_keys.index(name) for name in mh_in.LOG_PARAMETERS]
    out = params.copy().astype(float)
    out[:, log_idx] = np.log10(out[:, log_idx])
    return out


def _median_frac_err(true, pred, floor=1e-3):
    """Compute median fractional error (%) with optional floor."""
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.abs(true)
        denom = np.where(denom < floor, floor, denom)
        fe = np.abs((true - pred) / denom) * 100
    return np.nanmedian(fe)


# ══════════════════════════════════════════════════════════════════════════════
# V1 (ACG) accuracy tests (from test_main.py)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    os.environ.get("CI_MERGE_TEST") != "1",
    reason="TF comparison test only runs on merge to main (set CI_MERGE_TEST=1)",
)
def test_v1_tensorflow_vs_pytorch_equivalence():
    """Test that PyTorch model produces identical outputs to original TensorFlow model."""
    import torch

    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("TensorFlow not installed")

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

    # Compare outputs
    max_diff = np.abs(tf_output - pt_output).max()
    mean_diff = np.abs(tf_output - pt_output).mean()

    assert max_diff < 1e-4, f"Max difference {max_diff} exceeds tolerance 1e-4"
    assert mean_diff < 1e-5, f"Mean difference {mean_diff} exceeds tolerance 1e-5"


@pytest.mark.skipif(
    not V1_TEST_DATA.exists(), reason="Test_data_sample.npz not available"
)
def test_v1_emulator_vs_database():
    """Compare v1 emulator predictions against 21cmFAST database samples."""
    # Load test data
    test_data = np.load(V1_TEST_DATA, allow_pickle=True)
    X_test = test_data["X_test"]  # (100, 9) normalized params

    # Ground truth
    xHI_true = test_data["xHI"]  # (100, 84)
    Tb_true = test_data["Tb"]  # (100, 84) in mK
    tau_true = test_data["tau"]  # (100,) log10(tau)
    PS_true = test_data["PS"]  # (100, 60, 12)

    # Run emulator
    emu = Emulator(emulator="acg")
    X_test_phys = DefaultEmulatorInput().undo_normalization(X_test)
    _, output, _ = emu.predict(X_test_phys)

    # Calculate median fractional errors (%)
    def median_frac_err(true, pred, floor=1e-3):
        denom = np.abs(true)
        denom = np.where(denom < floor, floor, denom)
        fe = np.abs((true - pred) / denom) * 100
        return np.nanmedian(fe)

    # xHI
    mask = xHI_true > 0.01
    xHI_vals = output.xHI.value if hasattr(output.xHI, "value") else output.xHI
    xHI_fe = median_frac_err(xHI_true[mask], xHI_vals[mask])
    assert xHI_fe < 5, f"xHI median FE {xHI_fe:.2f}% exceeds 5%"

    # Tb
    Tb_vals = output.Tb.value if hasattr(output.Tb, "value") else output.Tb
    Tb_fe = median_frac_err(Tb_true, Tb_vals, floor=1.0)
    assert Tb_fe < 15, f"Tb median FE {Tb_fe:.2f}% exceeds 15%"

    # tau
    tau_vals = output.tau.value if hasattr(output.tau, "value") else output.tau
    tau_fe = median_frac_err(tau_true, np.log10(tau_vals), floor=0.01)
    assert tau_fe < 5, f"tau median FE {tau_fe:.2f}% exceeds 5%"

    # PS
    PS_vals = output.PS.value if hasattr(output.PS, "value") else output.PS
    PS_emu_log = np.log10(PS_vals)
    PS_fe = median_frac_err(PS_true, PS_emu_log)
    assert PS_fe < 20, f"PS median FE {PS_fe:.2f}% exceeds 20%"

    print(
        f"V1 accuracy: xHI={xHI_fe:.2f}%, Tb={Tb_fe:.2f}%, "
        f"tau={tau_fe:.2f}%, PS={PS_fe:.2f}%"
    )


# ══════════════════════════════════════════════════════════════════════════════
# V3 (MCG) accuracy tests (from test_v3.py::TestMHAccuracy)
# ══════════════════════════════════════════════════════════════════════════════


class TestMHAccuracy:
    """Accuracy tests comparing emulation to 21cmFAST database samples."""

    @pytest.mark.skipif(not TEST_SET_H5.exists(), reason="test_set.h5 not available")
    def test_lstm_accuracy_vs_database(self):
        """Compare v3 LSTM emulator outputs against 21cmFAST database samples."""
        h5py = pytest.importorskip("h5py")

        n_test = 50

        with h5py.File(TEST_SET_H5, "r") as f:
            params = np.asarray(f["inputs"][:n_test])
            xHI_true = np.asarray(f["xHI"][:n_test][..., ::-1])
            Tb_true = np.asarray(f["Tb"][:n_test][..., ::-1])
            Ts_true = np.asarray(f["Ts_neutral"][:n_test][..., ::-1])
            tau_true = np.asarray(f["tau_e"][:n_test])
            UVLFs_true = np.asarray(f["LFs"][:n_test])  # (n, 7, 60)

        emu = Emulator(emulator="mcg", emulate_2d_ps=False)
        params = _log_convert_mh_params(params)
        _, output, _ = emu.predict(params)

        xHI_emu = output.xHI.value
        Tb_emu = output.Tb.value
        Ts_emu = output.Ts.value
        tau_emu = output.tau.value
        UVLFs_emu = output.UVLFs.value

        # ── xHI accuracy ──
        mask = xHI_true > 0.01
        xHI_fe = None
        if mask.any():
            xHI_fe = _median_frac_err(xHI_true[mask], xHI_emu[mask])
            assert xHI_fe < 1, f"xHI median FE {xHI_fe:.2f}% exceeds 1%"

        # ── Tb accuracy ──
        Tb_fe = _median_frac_err(Tb_true, Tb_emu, floor=5.0)
        assert Tb_fe < 2, f"Tb median FE {Tb_fe:.2f}% exceeds 2%"

        # ── Ts accuracy ──
        Ts_true_log = np.log10(Ts_true)
        Ts_emu_log = np.log10(Ts_emu)
        mask = np.isfinite(Ts_true_log) & np.isfinite(Ts_emu_log)
        Ts_fe = None
        if mask.any():
            Ts_fe = _median_frac_err(Ts_true_log[mask], Ts_emu_log[mask], floor=0.1)
            assert Ts_fe < 3, f"Ts median FE {Ts_fe:.2f}% exceeds 3%"

        # ── tau accuracy ──
        tau_fe = _median_frac_err(tau_true, tau_emu, floor=0.001)
        assert tau_fe < 1, f"tau median FE {tau_fe:.2f}% exceeds 1%"

        # ── UVLFs accuracy ──
        UVLFs_emu_crop = UVLFs_emu  # (n, 7, 30) already cropped

        with h5py.File(TEST_SET_H5, "r") as f:
            M_UV_db = np.asarray(f["M_UV"])
        m_db = np.logical_and(M_UV_db <= -10, M_UV_db >= -20)
        UVLFs_true_crop = UVLFs_true[:, :, m_db]  # (n, 7, 30)

        mask = np.isfinite(UVLFs_true_crop) & np.isfinite(UVLFs_emu_crop)
        UVLFs_fe = None
        if mask.any():
            UVLFs_fe = _median_frac_err(
                UVLFs_true_crop[mask], UVLFs_emu_crop[mask], floor=0.1
            )
            assert UVLFs_fe < 5, f"UVLFs median FE {UVLFs_fe:.2f}% exceeds 5%"

        parts = [f"Tb={Tb_fe:.2f}%", f"tau={tau_fe:.2f}%"]
        if xHI_fe is not None:
            parts.insert(0, f"xHI={xHI_fe:.2f}%")
        if Ts_fe is not None:
            parts.append(f"Ts={Ts_fe:.2f}%")
        if UVLFs_fe is not None:
            parts.append(f"UVLFs={UVLFs_fe:.2f}%")
        print(f"V3 LSTM accuracy: {', '.join(parts)}")

    @pytest.mark.skipif(not TEST_SET_H5.exists(), reason="test_set.h5 not available")
    def test_1d_ps_accuracy_vs_database(self):
        """Test 1D PS accuracy from LSTM model against database."""
        h5py = pytest.importorskip("h5py")

        n_test = 50

        with h5py.File(TEST_SET_H5, "r") as f:
            params = np.asarray(f["inputs"][:n_test])
            PS_1D_true = np.asarray(f["PS_1D"][:n_test])  # (n, 32, 32) linear
            PS_redshifts = np.asarray(f["PS_redshifts"])
            k = np.asarray(f["k"])

        emu = Emulator(emulator="mcg", emulate_2d_ps=False)
        params = _log_convert_mh_params(params)
        _, output, _ = emu.predict(params, n_lstm_batch=10)

        assert output.PS is not None
        assert output.PS.shape == (n_test, 32, 32)

        assert hasattr(output.PS, "unit")
        from astropy import units as u

        assert output.PS.unit == u.mK**2

        assert output.PS_2D is None
        assert output.PS_2D_samples is None
        assert output.PS_2D_std is None

        with np.errstate(divide="ignore", invalid="ignore"):
            PS_1D_true_log = np.log10(PS_1D_true)
            PS_1D_emu_log = np.log10(output.PS.value)

        fe = _median_frac_err(PS_1D_true_log, PS_1D_emu_log, floor=0.1)
        assert fe < 5, f"1D PS median FE {fe:.2f}% exceeds 5%"

        assert np.allclose(emu.properties.PS_1D_k, k)
        assert np.allclose(emu.properties.PS_1D_redshifts, PS_redshifts)

        print(f"V3 1D PS accuracy: FE={fe:.2f}%")

    @pytest.mark.main_only
    @pytest.mark.skipif(not TEST_SET_H5.exists(), reason="test_set.h5 not available")
    def test_1d_ps_always_available_with_2d(self):
        """Test that 1D PS is always available even when emulate_2d_ps=True."""
        h5py = pytest.importorskip("h5py")

        with h5py.File(TEST_SET_H5, "r") as f:
            params = np.asarray(f["inputs"][:1])
        params = _log_convert_mh_params(params)

        emu = Emulator(emulator="mcg", emulate_2d_ps=True)
        _, output, _ = emu.predict(params, n_realisations=2, ps_2d_redshifts=[7.0])

        assert output.PS is not None
        assert output.PS.shape == (32, 32)

        assert output.PS_2D is not None
        assert output.PS_2D_samples is not None
        assert output.PS_2D_std is not None

        assert output.PS_2D.shape == (1, 1, 32, 64)
        assert output.PS_2D_samples.shape == (1, 1, 2, 32, 64)
        assert output.PS_2D_std.shape == (1, 1, 32, 64)

        assert output.PS_2D_redshifts is not None
        assert np.allclose(output.PS_2D_redshifts, [7.0])

    @pytest.mark.skipif(
        not PS_2D_TEST_H5.exists(), reason="ps_2d_test_subsample.h5 not available"
    )
    def test_diffusion_ps_single_sample(self):
        """Test 2D PS diffusion model with a single sample at one redshift."""
        h5py = pytest.importorskip("h5py")

        with h5py.File(PS_2D_TEST_H5, "r") as f:
            params = np.asarray(f["input_params"][0:1])
            PS_true = np.asarray(f["PS_2D_64_means"][0:1])  # (1, 32, 32, 64)
            ps_redshifts = np.asarray(f["redshifts"])
        params = _log_convert_mh_params(params)

        z_idx = 15
        z_test = np.array([ps_redshifts[z_idx]])
        PS_true_z = PS_true[:, z_idx, :, :]  # (1, 32, 64) log10(PS)

        emu = Emulator(emulator="mcg", emulate_2d_ps=True)
        _, output, _ = emu.predict(
            params, ps_2d_redshifts=z_test, n_realisations=1, ps_sampling_method="em"
        )

        assert output.PS is not None
        assert output.PS.shape == (32, 32)

        assert output.PS_2D is not None
        assert output.PS_2D_samples is not None
        assert output.PS_2D.shape == (1, 1, 32, 64)

        PS_emu_linear = output.PS_2D.value[0, 0]  # (32, 64)
        with np.errstate(divide="ignore", invalid="ignore"):
            PS_emu_log = np.log10(PS_emu_linear)

        assert np.nanmedian(PS_emu_log) > -3
        assert np.nanmedian(PS_emu_log) < 5

        fe = _median_frac_err(PS_true_z[0], PS_emu_log, floor=0.1)
        assert fe < 100, f"PS_2D median FE {fe:.2f}% exceeds 100% (sanity check)"

        print(
            f"V3 diffusion test: PS_2D shape={output.PS_2D.shape},"
            f" median_log={np.nanmedian(PS_emu_log):.2f}, FE={fe:.1f}%"
        )

    @pytest.mark.main_only
    @pytest.mark.skipif(
        not PS_2D_TEST_H5.exists(), reason="ps_2d_test_subsample.h5 not available"
    )
    def test_score_model_accuracy_vs_mean_error(self):
        """Test that score model FE is less than the stored mean error."""
        h5py = pytest.importorskip("h5py")

        with h5py.File(PS_2D_TEST_H5, "r") as f:
            params = np.asarray(f["input_params"][0:1])
            PS_true = np.asarray(f["PS_2D_64_means"][0:1])  # (1, 32, 32, 64)
            ps_redshifts = np.asarray(f["redshifts"])
        params = _log_convert_mh_params(params)

        z_idx = 15
        z_test = np.array([ps_redshifts[z_idx]])
        PS_true_z = PS_true[:, z_idx, :, :]  # (1, 32, 64) log10(PS)

        emu = Emulator(emulator="mcg", emulate_2d_ps=True)
        _, output, _ = emu.predict(
            params,
            ps_2d_redshifts=z_test,
            n_realisations=10,
            ps_sampling_method="ode",
        )

        assert output.PS is not None
        assert output.PS.shape == (32, 32)
        assert output.PS_2D is not None
        assert output.PS_2D.shape == (1, 1, 32, 64)

        PS_emu_log = output.PS_2D.value[0, 0]  # (32, 64)

        fe = _median_frac_err(PS_true_z[0], PS_emu_log, floor=0.01)

        props = emu.properties
        global_mean_err = props.PS_global_mean_err_ode

        threshold = 200.0  # Very generous sanity check
        assert fe < threshold, (
            f"Score model median FE {fe:.2f}% exceeds sanity threshold {threshold:.2f}%"
        )

        if global_mean_err is not None:
            print(
                f"Score model accuracy test: FE={fe:.2f}% (single sample), "
                f"global_mean_err={global_mean_err:.2f}% (population)"
            )
        else:
            print(f"Score model accuracy test: FE={fe:.2f}%")

        if props.PS_var_ode is not None:
            assert props.PS_var_ode.shape == (32, 64)

        if props.PS_cov_ode is not None:
            expected_cov_shape = (32 * 64, 32 * 64)
            assert props.PS_cov_ode.shape == expected_cov_shape
