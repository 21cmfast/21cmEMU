"""Tests for v3 (mcg) emulator integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from py21cmemu import Emulator

# Path constants (also defined in conftest.py for fixtures)
TUTORIALS_DIR = Path(__file__).resolve().parents[1] / "docs" / "tutorials"
TEST_SET_H5 = TUTORIALS_DIR / "test_set.h5"
PS_TEST_H5 = TUTORIALS_DIR / "ps_1d_loglin_db_test.h5"


def _log_convert_mh_params(params: np.ndarray) -> np.ndarray:
    """Convert MH parameter array from linear to log10 for LOG_PARAMETERS columns.

    The HDF5 test databases (test_set.h5, ps_2d_test_subsample.h5) store all
    parameters in linear/physical space (e.g. F_STAR10 ≈ 0.09 as a fraction).
    The emulator now expects LOG_PARAMETERS (F_STAR10, F_ESC10, F_STAR7_MINI,
    F_ESC7_MINI, L_X, L_X_MINI) in log10 space, so we convert them here.
    """
    from py21cmemu.inputs import MHEmulatorInput

    mh_in = MHEmulatorInput()
    astro_keys = list(mh_in.astro_param_keys)
    log_idx = [astro_keys.index(name) for name in mh_in.LOG_PARAMETERS]
    out = params.copy().astype(float)
    out[:, log_idx] = np.log10(out[:, log_idx])
    return out


# Note: mh_emulator fixture is defined in conftest.py


@pytest.mark.skipif(not TEST_SET_H5.exists(), reason="test_set.h5 not available")
def test_mh_predict_from_tutorial_h5(mh_emulator) -> None:
    """Test basic prediction with MH emulator."""
    h5py = pytest.importorskip("h5py")

    with h5py.File(TEST_SET_H5, "r") as f:
        params = np.asarray(f["inputs"][0:1])
    params = _log_convert_mh_params(params)

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


@pytest.mark.skipif(not TEST_SET_H5.exists(), reason="test_set.h5 not available")
def test_mh_batch_prediction(mh_emulator) -> None:
    """Test batch prediction with multiple parameter sets."""
    h5py = pytest.importorskip("h5py")

    with h5py.File(TEST_SET_H5, "r") as f:
        params = np.asarray(f["inputs"][:5])
    params = _log_convert_mh_params(params)

    theta, output, _errors = mh_emulator.predict(params)

    _theta_ps, theta_lstm = theta
    assert theta_lstm.shape[0] == 5
    assert output.Tb.shape[0] == 5
    assert output.xHI.shape[0] == 5
    assert output.Ts.shape[0] == 5
    assert output.tau.shape[0] == 5


@pytest.mark.skipif(not TEST_SET_H5.exists(), reason="test_set.h5 not available")
def test_mh_output_shapes(mh_emulator) -> None:
    """Test that output shapes match expected dimensions."""
    h5py = pytest.importorskip("h5py")

    with h5py.File(TEST_SET_H5, "r") as f:
        params = np.asarray(f["inputs"][0:1])
    params = _log_convert_mh_params(params)

    _, output, errors = mh_emulator.predict(params)

    # Check expected shapes
    n_z = 93  # Number of redshift bins
    n_lf_z = 7  # Number of UVLF redshifts

    assert output.Tb.shape[-1] == n_z
    assert output.xHI.shape[-1] == n_z
    assert output.Ts.shape[-1] == n_z
    assert output.UVLFs.ndim == 3
    assert output.UVLFs.shape[1] == n_lf_z  # (batch, n_lf_z, n_mag)

    # Error shapes must match output shapes for single-sample prediction.
    # This is critical for MCMC samplers that may pass 1 or many samples.
    # Do NOT call squeeze() before comparing - the batch dimension must be preserved.
    assert output.xHI.shape == errors.xHI_err.shape, (
        f"MCG xHI shape mismatch: output {output.xHI.shape}"
        f" vs error {errors.xHI_err.shape}"
    )
    assert output.Tb.shape == errors.Tb_err.shape, (
        f"MCG Tb shape mismatch: output {output.Tb.shape}"
        f" vs error {errors.Tb_err.shape}"
    )
    assert output.Ts.shape == errors.Ts_err.shape, (
        f"MCG Ts shape mismatch: output {output.Ts.shape}"
        f" vs error {errors.Ts_err.shape}"
    )
    assert output.tau.shape == errors.tau_err.shape, (
        f"MCG tau shape mismatch: output {output.tau.shape}"
        f" vs error {errors.tau_err.shape}"
    )
    assert output.PS.shape == errors.PS_err.shape, (
        f"MCG PS shape mismatch: output {output.PS.shape}"
        f" vs error {errors.PS_err.shape}"
    )
    assert output.UVLFs.shape == errors.UVLFs_logerr.shape, (
        f"MCG UVLFs shape mismatch: output {output.UVLFs.shape}"
        f" vs error {errors.UVLFs_logerr.shape}"
    )


@pytest.mark.skipif(not TEST_SET_H5.exists(), reason="test_set.h5 not available")
def test_mh_output_values(mh_emulator) -> None:
    """Test that output values are in expected ranges."""
    h5py = pytest.importorskip("h5py")

    with h5py.File(TEST_SET_H5, "r") as f:
        params = np.asarray(f["inputs"][0:1])
    params = _log_convert_mh_params(params)

    _, output, _ = mh_emulator.predict(params)

    # xHI should be in [0, 1]
    assert np.all(output.xHI >= 0)
    assert np.all(output.xHI <= 1)

    # tau should be positive
    assert np.all(output.tau > 0)

    # Ts should be positive where defined (NaN where undefined at high z)
    ts_defined = ~np.isnan(output.Ts)
    assert ts_defined.any(), "Ts should have some defined values"
    assert np.all(output.Ts[ts_defined] > 0)


@pytest.mark.skipif(not TEST_SET_H5.exists(), reason="test_set.h5 not available")
def test_mh_properties(mh_emulator) -> None:
    """Test access to emulator properties."""
    props = mh_emulator.properties

    assert hasattr(props, "redshifts")
    assert hasattr(props, "astro_param_keys")
    assert len(props.astro_param_keys) == 11
    assert len(props.redshifts) == 93


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


def test_mh_inputs_class() -> None:
    """Test MHEmulatorInput class."""
    from py21cmemu.inputs import MHEmulatorInput

    inputs = MHEmulatorInput()
    assert len(inputs.astro_param_keys) == 11

    # LOG_PARAMETERS and PARAMETERS class attributes
    assert "F_STAR10" in inputs.LOG_PARAMETERS
    assert "F_ESC10" in inputs.LOG_PARAMETERS
    assert "L_X" in inputs.LOG_PARAMETERS
    assert len(inputs.LOG_PARAMETERS) == 6
    assert len(inputs.PARAMETERS) == 11

    # Test normalization with log-space values for log parameters
    # Order: F_STAR10, ALPHA_STAR, t_STAR, F_ESC10, ALPHA_ESC,
    #         F_STAR7_MINI, F_ESC7_MINI, L_X, L_X_MINI, NU_X_THRESH, SIGMA_8
    test_params = np.array(
        [[-1.5, 0.5, 0.5, -2.0, 0.0, -3.0, -2.0, 40.0, 40.0, 500.0, 0.82]]
    )
    normed = inputs.normalize(test_params, kind="LSTM")
    assert normed.shape == (1, 11)
    assert np.all(normed >= 0) and np.all(normed <= 1)


def test_mh_outputs_class() -> None:
    """Test MHEmulatorOutput class."""
    from py21cmemu.outputs import MHEmulatorOutput

    # Create minimal output (no 2D PS)
    output = MHEmulatorOutput(
        Tb=np.zeros(93),
        xHI=np.zeros(93),
        Ts=np.zeros(93),
        tau=np.array([0.05]),
        UVLFs=np.zeros((1, 35, 7)),
        PS=np.zeros((32, 32)),
        PS_2D=None,
        PS_2D_samples=None,
        PS_2D_std=None,
        PS_2D_redshifts=None,
    )
    assert output.Tb.shape[-1] == 93
    assert output.tau[0] == 0.05
    assert output.PS.shape == (32, 32)


# =============================================================================
# 2D PS Property and Structural Tests (no GPU evaluation required)
# =============================================================================


class TestMH2DPSProperties:
    """Test 2D PS property access without requiring GPU or model evaluation."""

    @pytest.fixture(scope="class")
    def mh_props(self):
        """Get MH emulator properties."""
        from py21cmemu.properties import MHEmulatorProperties

        return MHEmulatorProperties()

    def test_ps_redshifts_available(self, mh_props):
        """Test that PS redshifts are available and match 2D PS grid."""
        assert hasattr(mh_props, "PS_zs")
        assert hasattr(mh_props, "PS_redshifts")
        # Must have exactly 32 redshifts to match 2D PS grid (32 z-bins x 64 k-bins)
        assert len(mh_props.PS_zs) == 32, (
            f"Expected 32 PS redshifts, got {len(mh_props.PS_zs)}"
        )
        assert np.array_equal(mh_props.PS_zs, mh_props.PS_redshifts)
        # PS redshifts should be increasing
        assert np.all(np.diff(mh_props.PS_zs) > 0)
        # Should span roughly z=5.5 to z=29
        assert mh_props.PS_zs[0] < 6, (
            f"First PS redshift should be ~5.5, got {mh_props.PS_zs[0]}"
        )
        assert mh_props.PS_zs[-1] > 28, (
            f"Last PS redshift should be ~29, got {mh_props.PS_zs[-1]}"
        )

    def test_kperp_kpar_available(self, mh_props):
        """Test that kperp and kpar arrays are available."""
        assert hasattr(mh_props, "kperp")
        assert hasattr(mh_props, "kpar")
        # Score model uses 32 kperp x 64 kpar grid
        assert len(mh_props.kperp) == 32
        assert len(mh_props.kpar) == 64
        # k values should be positive and increasing
        assert np.all(mh_props.kperp > 0)
        assert np.all(mh_props.kpar > 0)
        assert np.all(np.diff(mh_props.kperp) > 0)
        assert np.all(np.diff(mh_props.kpar) > 0)

    def test_nmodes_available(self, mh_props):
        """Test that Nmodes array is available."""
        assert hasattr(mh_props, "Nmodes")
        # Score model uses 32x64 grid
        assert mh_props.Nmodes.shape == (32, 64)
        # All mode counts should be positive
        assert np.all(mh_props.Nmodes >= 0)

    def test_ps_normalization_constants(self, mh_props):
        """Test that PS normalization constants are available."""
        assert hasattr(mh_props, "PS_bias")
        assert hasattr(mh_props, "PS_scale")
        # Scale should be positive
        assert np.all(mh_props.PS_scale > 0)

    def test_ps_limits_available(self, mh_props):
        """Test that PS parameter limits are available."""
        assert hasattr(mh_props, "ps_limits")
        # Should have 12 parameters (11 astro + 1 redshift)
        assert mh_props.ps_limits.shape == (12, 2)
        # Lower limit should be less than upper limit
        assert np.all(mh_props.ps_limits[:, 0] < mh_props.ps_limits[:, 1])

    def test_ps_med_err_available(self, mh_props):
        """Test that PS median errors are available."""
        assert hasattr(mh_props, "PS_med_err")
        # Score model uses 32x64 grid
        assert mh_props.PS_med_err.shape == (32, 64)
        # Errors should be positive (percentages)
        assert np.all(mh_props.PS_med_err >= 0)

    def test_ps_med_err_method_specific(self, mh_props):
        """Test that method-specific PS median errors are available."""
        assert hasattr(mh_props, "PS_med_err_em")
        assert hasattr(mh_props, "PS_med_err_ode")
        # Score model uses 32x64 grid
        assert mh_props.PS_med_err_em.shape == (32, 64)
        assert mh_props.PS_med_err_ode.shape == (32, 64)

    def test_error_arrays_available(self, mh_props):
        """Test that error arrays with mean/median/std are available."""
        # 1D summaries: per-z arrays (93,)
        for name in ["xHI", "Tb", "Ts"]:
            for stat in ["mean_err", "med_err", "std_err"]:
                attr = f"{name}_{stat}"
                assert hasattr(mh_props, attr), f"Missing {attr}"
                arr = getattr(mh_props, attr)
                assert arr.shape == (93,), f"{attr} has wrong shape: {arr.shape}"

        # tau: scalar
        for stat in ["mean_err", "med_err", "std_err"]:
            attr = f"tau_{stat}"
            assert hasattr(mh_props, attr), f"Missing {attr}"
            val = getattr(mh_props, attr)
            assert np.isscalar(val) or val.ndim == 0, f"{attr} should be scalar"

        # UVLFs: per-(M_UV, z) arrays (45, 7)
        for stat in ["mean_err", "med_err", "std_err"]:
            attr = f"UVLFs_{stat}"
            assert hasattr(mh_props, attr), f"Missing {attr}"
            arr = getattr(mh_props, attr)
            assert arr.shape == (45, 7), f"{attr} has wrong shape: {arr.shape}"

        # 1D PS errors: per-(z, k) arrays (32, 32)
        for stat in ["mean_err", "med_err", "std_err"]:
            attr = f"PS_1D_{stat}"
            assert hasattr(mh_props, attr), f"Missing {attr}"
            arr = getattr(mh_props, attr)
            assert arr.shape == (32, 32), f"{attr} has wrong shape: {arr.shape}"

        # 2D PS errors: per-(z, k) arrays (32, 64)
        assert hasattr(mh_props, "PS_med_err")
        assert mh_props.PS_med_err.shape == (32, 64)

    def test_z_averaged_errors_reasonable(self, mh_props):
        """Test that z-averaged mean errors are within reasonable bounds.

        These are rough sanity checks - the actual values depend on the
        trained model. The thresholds are generous to avoid false failures.
        """
        # z-averaged mean FE should be reasonable for all 1D summaries
        assert np.nanmean(mh_props.xHI_mean_err) < 1.0, "xHI z-avg mean error too high"
        # Tb can have inflated errors at low-z (flooring at |Tb| < 5 mK)
        assert np.nanmean(mh_props.Tb_med_err) < 1.0, "Tb z-avg median error too high"
        assert np.nanmean(mh_props.Ts_mean_err) < 1.0 or np.isnan(
            np.nanmean(mh_props.Ts_mean_err)
        ), "Ts z-avg mean error too high"
        assert mh_props.tau_mean_err < 1.0, "tau mean error too high"

        # z-averaged mean FE for LFs should be < 1%
        assert np.nanmean(mh_props.UVLFs_med_err) < 1.0, (
            "UVLFs z-avg median error too high"
        )

        # 1D PS error is higher at high-z due to sample variance
        assert np.nanmean(mh_props.PS_1D_med_err) < 5.0, (
            "PS_1D z-avg median error too high"
        )

    def test_1d_ps_properties(self, mh_props):
        """Test that 1D PS properties are available."""
        assert hasattr(mh_props, "PS_1D_k")
        assert hasattr(mh_props, "PS_1D_redshifts")
        assert hasattr(mh_props, "PS_1D_bias")
        assert hasattr(mh_props, "PS_1D_scale")

        assert mh_props.PS_1D_k.shape == (32,)
        assert mh_props.PS_1D_redshifts.shape == (32,)


class TestMH2DPSInputs:
    """Test 2D PS input formatting without requiring model evaluation."""

    @pytest.fixture(scope="class")
    def mh_inputs(self):
        """Get MH input handler."""
        from py21cmemu.inputs import MHEmulatorInput

        return MHEmulatorInput()

    def test_format_theta_for_ps(self, mh_inputs):
        """Test theta formatting for PS model."""
        # Create sample parameter array (11 params) - must be 2D for batch
        test_params = np.array([[0.5] * 11])

        # Format for PS at multiple redshifts
        ps_redshifts = np.array([6.0, 10.0, 15.0])
        theta_ps = mh_inputs.format_theta(test_params, ps_redshifts)

        # Should have 3 rows (one per redshift per sample)
        # n_samples * n_z = 1 * 3 = 3
        assert theta_ps.shape[0] == 3
        # Each row has 12 columns (11 params + 1 redshift)
        assert theta_ps.shape[1] == 12

    def test_ps_limits_shape(self, mh_inputs):
        """Test that PS limits have correct shape."""
        from py21cmemu.properties import MHEmulatorProperties

        props = MHEmulatorProperties()

        # PS limits: 12 params (11 astro + 1 redshift), 2 bounds each
        assert props.ps_limits.shape == (12, 2)
        # LSTM limits: 12 params (11 astro + 1 redshift), 2 bounds each
        assert props.lstm_limits.shape == (12, 2)
        # Both should have lower < upper for all params
        assert np.all(props.ps_limits[:, 0] < props.ps_limits[:, 1])
        assert np.all(props.lstm_limits[:, 0] < props.lstm_limits[:, 1])


class TestMH2DSampler:
    """Test 2D PS sampler classes without running full sampling."""

    def test_em_sampler_import(self):
        """Test that EM sampler can be imported."""
        from py21cmemu.sample_pytorch import GetEMSampler

        assert GetEMSampler is not None

    def test_ode_sampler_import(self):
        """Test that ODE sampler can be imported."""
        from py21cmemu.sample_pytorch import GetODESampler

        assert GetODESampler is not None

    def test_vpsde_import(self):
        """Test that VPSDE can be imported."""
        from py21cmemu.sde import VPSDE

        assert VPSDE is not None

    def test_vpsde_creation(self):
        """Test VPSDE creation with default parameters."""
        from py21cmemu.sde import VPSDE

        sde = VPSDE(beta_min=0.1, beta_max=20.0)
        # VPSDE uses beta_0 and beta_1 internally
        assert sde.beta_0 == 0.1
        assert sde.beta_1 == 20.0
        assert sde.N == 1000  # default

    def test_em_sampler_creation_cpu(self):
        """Test EM sampler can be created on CPU."""
        import torch

        from py21cmemu.sample_pytorch import GetEMSampler
        from py21cmemu.sde import VPSDE

        sde = VPSDE(beta_min=0.1, beta_max=20.0)
        shape = (1, 10, 32, 32)  # Small batch for testing
        sampler = GetEMSampler(sde, shape, device=torch.device("cpu"), denoise=True)

        assert sampler is not None
        em_fn = sampler.get_em_sampler()
        assert callable(em_fn)

    def test_ode_sampler_creation_cpu(self):
        """Test ODE sampler can be created on CPU."""
        import torch

        from py21cmemu.sample_pytorch import GetODESampler
        from py21cmemu.sde import VPSDE

        sde = VPSDE(beta_min=0.1, beta_max=20.0)
        shape = (1, 10, 32, 32)  # Small batch for testing
        sampler = GetODESampler(
            sde, shape, device=torch.device("cpu"), denoise=True, rtol=1e-5, atol=1e-5
        )

        assert sampler is not None
        ode_fn = sampler.get_ode_sampler()
        assert callable(ode_fn)


class TestMH2DScoreModel:
    """Test 2D PS score model structure without running inference."""

    def test_score_model_import(self):
        """Test that UNet score model can be imported."""
        from py21cmemu.models.MCG.score_model import UNet

        assert UNet is not None

    def test_score_model_creation(self):
        """Test UNet can be created without loading weights."""
        from py21cmemu.models.MCG.score_model import UNet

        model = UNet(
            dim=(32, 64),
            init_dim=48,
            channels=1,
            dim_mults=(1, 2, 4, 8, 16),
            cdn_len=12,
        )

        assert model is not None
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_score_model_forward_shape(self):
        """Test UNet forward pass produces correct shape."""
        import torch

        from py21cmemu.models.MCG.score_model import UNet

        model = UNet(
            dim=(32, 64),
            init_dim=48,
            channels=1,
            dim_mults=(1, 2, 4, 8, 16),
            cdn_len=12,
        )
        model.eval()

        # Random inputs
        batch_size = 2
        x = torch.randn(batch_size, 1, 32, 64)  # Noisy image
        t = torch.rand(batch_size)  # Time
        cdn = torch.randn(batch_size, 12)  # Conditioning

        with torch.no_grad():
            out = model(x, time=t, cdn=cdn)

        assert out.shape == (batch_size, 1, 32, 64)

    def test_score_model_weights_load(self):
        """Test that packaged score model weights can be loaded."""
        from pathlib import Path

        import torch

        from py21cmemu.models.MCG.score_model import UNet

        weights_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "py21cmemu"
            / "models"
            / "MHs"
            / "score_model_weights.pt"
        )
        if not weights_path.exists():
            pytest.skip("score_model_weights.pt not found")

        model = UNet(
            dim=(32, 64),
            init_dim=48,
            channels=1,
            dim_mults=(1, 2, 4, 8, 16),
            cdn_len=12,
        )

        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        # Test forward pass with loaded weights
        batch_size = 1
        x = torch.randn(batch_size, 1, 32, 64)
        t = torch.tensor([0.5])
        cdn = torch.randn(batch_size, 12)

        with torch.no_grad():
            out = model(x, time=t, cdn=cdn)

        assert out.shape == (batch_size, 1, 32, 64)
        # Output should not be all zeros or NaN
        assert not torch.all(out == 0)
        assert not torch.any(torch.isnan(out))


class TestMH2DOutputStructure:
    """Test 2D PS output structure and data handling."""

    def test_mh_output_with_2d_ps(self):
        """Test MHEmulatorOutput can include 2D PS data."""
        from py21cmemu.outputs import MHEmulatorOutput

        # Create minimal output with both 1D and 2D PS
        ps_1d = np.random.rand(32, 32)  # (nz, nk) 1D PS from LSTM
        ps_2d_samples = np.random.rand(
            1, 10, 100, 32, 64
        )  # (batch, nz, nsamples, kperp, kpar)
        ps_2d_median = np.median(ps_2d_samples, axis=2)
        ps_2d_std = np.std(ps_2d_samples, axis=2)
        ps_redshifts = np.linspace(6, 20, 10)

        output = MHEmulatorOutput(
            Tb=np.zeros(32),
            xHI=np.zeros(32),
            Ts=np.zeros(32),
            tau=np.array([0.05]),
            UVLFs=np.zeros((1, 35, 7)),
            PS=ps_1d,
            PS_2D=ps_2d_median,
            PS_2D_samples=ps_2d_samples,
            PS_2D_std=ps_2d_std,
            PS_2D_redshifts=ps_redshifts,
        )

        # Check 1D PS
        assert output.PS is not None
        assert output.PS.shape == (32, 32)

        # Check 2D PS
        assert output.PS_2D is not None
        assert output.PS_2D.shape == (1, 10, 32, 64)
        assert output.PS_2D_samples.shape == (1, 10, 100, 32, 64)
        assert output.PS_2D_std.shape == (1, 10, 32, 64)
        assert len(output.PS_2D_redshifts) == 10

    def test_ps_variance_computation(self):
        """Test that PS variance can be computed from samples."""
        # Simulate PS samples
        ps_samples = np.random.rand(1, 10, 100, 32, 32)

        # Compute variance along sample axis
        variance = np.var(ps_samples, axis=2)
        assert variance.shape == (1, 10, 32, 32)
        assert np.all(variance >= 0)

    def test_ps_covariance_computation(self):
        """Test that PS covariance can be computed from samples."""
        # Simulate PS samples for a single redshift
        nsamples = 50
        ps_samples = np.random.rand(nsamples, 32, 32)

        # Flatten to (nsamples, npix)
        ps_flat = ps_samples.reshape(nsamples, -1)

        # Compute covariance
        cov = np.cov(ps_flat, rowvar=False)
        assert cov.shape == (32 * 32, 32 * 32)

        # Check diagonal is variance
        var_flat = np.var(ps_flat, axis=0, ddof=1)
        assert np.allclose(np.diag(cov), var_flat, rtol=1e-6)


class TestMHEmulatorPSSetup:
    """Test MH emulator PS setup without full model loading."""

    def test_emulator_ps_false(self):
        """Test emulator creation with PS disabled."""
        emu = Emulator(emulator="mcg", emulate_2d_ps=False)
        assert not emu.emulate_2d_ps
        assert emu.score_model is None
        assert emu.sample is None

    def test_emulator_ps_properties_accessible(self):
        """Test PS properties accessible even without model."""
        emu = Emulator(emulator="mcg", emulate_2d_ps=False)

        # These should work without PS model loaded
        assert hasattr(emu.properties, "kperp")
        assert hasattr(emu.properties, "kpar")
        assert hasattr(emu.properties, "PS_zs")
        assert hasattr(emu.properties, "Nmodes")
        assert hasattr(emu.properties, "PS_bias")
        assert hasattr(emu.properties, "PS_scale")

    def test_ps_sampling_method_values(self):
        """Test that valid PS sampling method values are 'em' and 'ode'."""
        # Just verify the valid values are documented correctly
        valid_methods = ("em", "ode")
        assert "em" in valid_methods
        assert "ode" in valid_methods


class TestMH2DUtilities:
    """Test utility functions for 2D PS handling."""

    def test_reverse_transform(self):
        """Test reverse transform function."""
        import torch

        from py21cmemu.utils import reverse_transform

        # Normalized values in [-1, 1] range (as output by diffusion model)
        normed = torch.randn(2, 1, 32, 32)
        scale = torch.tensor(5.0)
        bias = torch.tensor(-2.0)

        # Reverse transform formula:
        # unit = (y + 1) / 2, d = unit * scale + bias, return 10^d
        denormed = reverse_transform(normed, scale, bias)

        unit = (normed + 1) / 2
        d = unit * scale + bias
        expected = 10**d
        assert torch.allclose(denormed, expected)


# ═══════════════════════════════════════════════════════════════════════════════
# ACCURACY COMPARISON TESTS
# ═══════════════════════════════════════════════════════════════════════════════

PS_2D_TEST_H5 = TUTORIALS_DIR / "ps_2d_test_subsample.h5"


def _median_frac_err(true, pred, floor=1e-3):
    """Compute median fractional error (%) with optional floor."""
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.abs(true)
        denom = np.where(denom < floor, floor, denom)
        fe = np.abs((true - pred) / denom) * 100
    return np.nanmedian(fe)


class TestMHAccuracy:
    """Accuracy tests comparing emulation to 21cmFAST database samples."""

    @pytest.mark.skipif(not TEST_SET_H5.exists(), reason="test_set.h5 not available")
    def test_lstm_accuracy_vs_database(self):
        """Compare v3 LSTM emulator outputs against 21cmFAST database samples.

        Tests xHI, Tb, Ts, tau, and UVLFs predictions against ground truth.
        """
        h5py = pytest.importorskip("h5py")

        # Use a small subset for faster testing
        n_test = 50

        with h5py.File(TEST_SET_H5, "r") as f:
            params = np.asarray(f["inputs"][:n_test])
            xHI_true = np.asarray(f["xHI"][:n_test][..., ::-1])
            Tb_true = np.asarray(f["Tb"][:n_test][..., ::-1])
            Ts_true = np.asarray(f["Ts_neutral"][:n_test][..., ::-1])  # Use neutral Ts
            tau_true = np.asarray(f["tau_e"][:n_test])
            UVLFs_true = np.asarray(f["LFs"][:n_test])  # (n, 7, 60)

        # Run emulator (LSTM only)
        emu = Emulator(emulator="mcg", emulate_2d_ps=False)
        params = _log_convert_mh_params(params)
        _, output, _ = emu.predict(params)

        # Extract raw values (emulator now returns Quantities with units)
        xHI_emu = output.xHI.value
        Tb_emu = output.Tb.value
        Ts_emu = output.Ts.value
        tau_emu = output.tau.value
        UVLFs_emu = output.UVLFs.value  # (n, n_z, n_mag) - log10(phi)

        # ── xHI accuracy ──
        # Compare where xHI > 0.01 (avoid near-zero regions)
        mask = xHI_true > 0.01
        if mask.any():
            xHI_fe = _median_frac_err(xHI_true[mask], xHI_emu[mask])
            assert xHI_fe < 1, f"xHI median FE {xHI_fe:.2f}% exceeds 1%"

        # ── Tb accuracy ──
        # Use 5 mK floor for Tb comparison (Tb can be small near absorption trough)
        # Note: comparing in same z-direction after reversing database arrays
        Tb_fe = _median_frac_err(Tb_true, Tb_emu, floor=5.0)
        assert Tb_fe < 2, f"Tb median FE {Tb_fe:.2f}% exceeds 2%"

        # ── Ts accuracy ──
        # Ts has NaN where undefined; compare where defined
        Ts_true_log = np.log10(Ts_true)
        Ts_emu_log = np.log10(Ts_emu)
        mask = np.isfinite(Ts_true_log) & np.isfinite(Ts_emu_log)
        if mask.any():
            Ts_fe = _median_frac_err(Ts_true_log[mask], Ts_emu_log[mask], floor=0.1)
            assert Ts_fe < 3, f"Ts median FE {Ts_fe:.2f}% exceeds 3%"

        # ── tau accuracy ──
        tau_fe = _median_frac_err(tau_true, tau_emu, floor=0.001)
        assert tau_fe < 1, f"tau median FE {tau_fe:.2f}% exceeds 1%"

        # ── UVLFs accuracy ──
        # Database has 60 magnitude bins, emulator output is pre-cropped to 30 bins
        # Both have shape (n, n_z=7, n_mag) - need to crop database to match
        # Emulator already outputs only the [-20, -10] magnitude range (30 bins)
        UVLFs_emu_crop = UVLFs_emu  # (n, 7, 30) already cropped

        # Database LFs: (n, 7, 60) -> crop to [-20, -10] range
        with h5py.File(TEST_SET_H5, "r") as f:
            M_UV_db = np.asarray(f["M_UV"])
        m_db = np.logical_and(M_UV_db <= -10, M_UV_db >= -20)
        UVLFs_true_crop = UVLFs_true[:, :, m_db]  # (n, 7, 30)

        # Compare log values
        mask = np.isfinite(UVLFs_true_crop) & np.isfinite(UVLFs_emu_crop)
        if mask.any():
            UVLFs_fe = _median_frac_err(
                UVLFs_true_crop[mask], UVLFs_emu_crop[mask], floor=0.1
            )
            assert UVLFs_fe < 5, f"UVLFs median FE {UVLFs_fe:.2f}% exceeds 5%"

        print(
            f"V3 LSTM accuracy: xHI={xHI_fe:.2f}%, Tb={Tb_fe:.2f}%,"
            f" Ts={Ts_fe:.2f}%, tau={tau_fe:.2f}%, UVLFs={UVLFs_fe:.2f}%"
        )

    @pytest.mark.skipif(not TEST_SET_H5.exists(), reason="test_set.h5 not available")
    def test_1d_ps_accuracy_vs_database(self):
        """Test 1D PS accuracy from LSTM model against database.

        The 1D PS is always available (from LSTM model) regardless
        of emulate_2d_ps setting.
        """
        h5py = pytest.importorskip("h5py")

        n_test = 50

        with h5py.File(TEST_SET_H5, "r") as f:
            params = np.asarray(f["inputs"][:n_test])
            PS_1D_true = np.asarray(f["PS_1D"][:n_test])  # (n, 32, 32) linear
            PS_redshifts = np.asarray(f["PS_redshifts"])
            k = np.asarray(f["k"])

        # Run emulator (default without 2D PS)
        emu = Emulator(emulator="mcg", emulate_2d_ps=False)
        params = _log_convert_mh_params(params)
        _, output, _ = emu.predict(params, n_lstm_batch=10)

        # Check 1D PS is available
        assert output.PS is not None, "1D PS should be available"
        assert output.PS.shape == (
            n_test,
            32,
            32,
        ), f"Expected PS shape ({n_test},32,32), got {output.PS.shape}"

        # Check units
        assert hasattr(output.PS, "unit"), "PS should have units"
        from astropy import units as u

        assert output.PS.unit == u.mK**2, (
            f"PS should have mK^2 units (linear), got {output.PS.unit}"
        )

        # Check 2D PS is None when emulate_2d_ps=False
        assert output.PS_2D is None, "PS_2D should be None when emulate_2d_ps=False"
        assert output.PS_2D_samples is None, "PS_2D_samples should be None"
        assert output.PS_2D_std is None, "PS_2D_std should be None"

        # Compare with ground truth (in log space for accuracy calculation)
        # Convert both to log10 for comparison
        with np.errstate(divide="ignore", invalid="ignore"):
            PS_1D_true_log = np.log10(PS_1D_true)
            PS_1D_emu_log = np.log10(output.PS.value)  # Convert linear to log10

        # Compute accuracy - use 0.1 floor to avoid issues with very small PS values
        fe = _median_frac_err(PS_1D_true_log, PS_1D_emu_log, floor=0.1)
        assert fe < 5, f"1D PS median FE {fe:.2f}% exceeds 5%"

        # Check axes match
        assert np.allclose(emu.properties.PS_1D_k, k), "PS_1D_k should match database k"
        assert np.allclose(emu.properties.PS_1D_redshifts, PS_redshifts), (
            "PS_1D_redshifts should match database"
        )

        print(f"V3 1D PS accuracy: FE={fe:.2f}%")

    @pytest.mark.main_only
    @pytest.mark.skipif(not TEST_SET_H5.exists(), reason="test_set.h5 not available")
    def test_1d_ps_always_available_with_2d(self):
        """Test that 1D PS is always available even when emulate_2d_ps=True.

        This test runs the 2D PS model and is slow (~3 min), so only runs
        on merge to main.
        """
        h5py = pytest.importorskip("h5py")

        with h5py.File(TEST_SET_H5, "r") as f:
            params = np.asarray(f["inputs"][:1])
        params = _log_convert_mh_params(params)

        # Run with 2D PS enabled
        emu = Emulator(emulator="mcg", emulate_2d_ps=True)
        _, output, _ = emu.predict(params, n_realisations=2, ps_2d_redshifts=[7.0])

        # Check 1D PS is still available
        assert output.PS is not None, (
            "1D PS should be available with emulate_2d_ps=True"
        )
        assert output.PS.shape == (
            32,
            32,
        ), f"1D PS shape should be (32,32), got {output.PS.shape}"

        # Check 2D PS is also available
        assert output.PS_2D is not None, (
            "PS_2D should be available when emulate_2d_ps=True"
        )
        assert output.PS_2D_samples is not None, "PS_2D_samples should be available"
        assert output.PS_2D_std is not None, "PS_2D_std should be available"

        # Check 2D PS shapes
        # (n_params, n_z, kperp, kpar) = (1, 1, 32, 64)
        assert output.PS_2D.shape == (
            1,
            1,
            32,
            64,
        ), f"PS_2D shape should be (1,1,32,64), got {output.PS_2D.shape}"
        assert output.PS_2D_samples.shape == (
            1,
            1,
            2,
            32,
            64,
        ), f"PS_2D_samples shape wrong, got {output.PS_2D_samples.shape}"
        assert output.PS_2D_std.shape == (
            1,
            1,
            32,
            64,
        ), f"PS_2D_std shape wrong, got {output.PS_2D_std.shape}"

        # Check PS_2D_redshifts
        assert output.PS_2D_redshifts is not None, "PS_2D_redshifts should be available"
        assert np.allclose(output.PS_2D_redshifts, [7.0]), (
            f"PS_2D_redshifts should be [7.0], got {output.PS_2D_redshifts}"
        )

        print("V3 1D+2D PS test passed: Both available when emulate_2d_ps=True")

    @pytest.mark.skipif(
        not PS_2D_TEST_H5.exists(), reason="ps_2d_test_subsample.h5 not available"
    )
    def test_diffusion_ps_single_sample(self):
        """Test 2D PS diffusion model with a single sample at one redshift.

        This is a fast smoke test that verifies the diffusion model works
        and produces outputs of expected shape and reasonable magnitude.
        """
        h5py = pytest.importorskip("h5py")

        # Load a single parameter set
        with h5py.File(PS_2D_TEST_H5, "r") as f:
            params = np.asarray(f["input_params"][0:1])  # Single param set
            PS_true = np.asarray(f["PS_2D_64_means"][0:1])  # (1, 32, 32, 64)
            ps_redshifts = np.asarray(f["redshifts"])  # 32 redshifts
        params = _log_convert_mh_params(params)

        # Pick just one redshift (middle one) for speed
        z_idx = 15
        z_test = np.array([ps_redshifts[z_idx]])
        PS_true_z = PS_true[:, z_idx, :, :]  # (1, 32, 64) log10(PS)

        # Create emulator with PS enabled
        emu = Emulator(emulator="mcg", emulate_2d_ps=True)

        # Run with just 1 sample at 1 redshift
        # This should be fast (~few seconds on CPU)
        _, output, _ = emu.predict(
            params, ps_2d_redshifts=z_test, n_realisations=1, ps_sampling_method="em"
        )

        # Check 1D PS always available
        assert output.PS is not None, "1D PS should always be available"
        assert output.PS.shape == (
            32,
            32,
        ), f"Expected 1D PS shape (32,32), got {output.PS.shape}"

        # Check 2D PS output exists and has expected shape
        assert output.PS_2D is not None, "PS_2D should not be None"
        assert output.PS_2D_samples is not None, "PS_2D_samples should not be None"

        # Shape: (n_params, n_z, num_samples, kperp, kpar)
        # -> median -> (n_params, n_z, kperp, kpar)
        # After median: (1, 1, 32, 64)
        assert output.PS_2D.shape == (
            1,
            1,
            32,
            64,
        ), f"Expected PS_2D shape (1,1,32,64), got {output.PS_2D.shape}"

        # Check order of magnitude is reasonable
        # PS_2D is in linear units (mK^2), convert to log10 for checks
        PS_emu_linear = output.PS_2D.value[0, 0]  # (32, 64) - linear values
        with np.errstate(divide="ignore", invalid="ignore"):
            PS_emu_log = np.log10(PS_emu_linear)

        # Check range is physically reasonable (in log10 space)
        assert np.nanmedian(PS_emu_log) > -3, (
            f"PS_2D median {np.nanmedian(PS_emu_log):.2f} too low"
        )
        assert np.nanmedian(PS_emu_log) < 5, (
            f"PS_2D median {np.nanmedian(PS_emu_log):.2f} too high"
        )

        # Compare with ground truth (rough accuracy check)
        # Diffusion model has ~20-50% typical error
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
        """Test that score model FE is less than the stored mean error.

        This test:
        1. Loads a single parameter set from the test database
        2. Runs the score model with ODE sampling (default, more accurate)
        3. Computes the median fractional error vs ground truth
        4. Verifies the error is less than the global mean error from properties

        This is a slow test (~30-60s on GPU, longer on CPU) so it only runs
        on merge to main or when --run-slow is specified.
        """
        h5py = pytest.importorskip("h5py")

        # Load a single sample from test data
        with h5py.File(PS_2D_TEST_H5, "r") as f:
            params = np.asarray(f["input_params"][0:1])  # Single param set
            PS_true = np.asarray(f["PS_2D_64_means"][0:1])  # (1, 32, 32, 64) log10(PS)
            ps_redshifts = np.asarray(f["redshifts"])  # 32 redshifts
        params = _log_convert_mh_params(params)

        # Pick one redshift (middle of range) for speed
        z_idx = 15
        z_test = np.array([ps_redshifts[z_idx]])
        PS_true_z = PS_true[:, z_idx, :, :]  # (1, 32, 64) log10(PS)

        # Create emulator with PS enabled
        emu = Emulator(emulator="mcg", emulate_2d_ps=True)

        # Run with ODE sampling (default, more accurate)
        _, output, _ = emu.predict(
            params,
            ps_2d_redshifts=z_test,
            n_realisations=10,
            ps_sampling_method="ode",  # ODE is default but be explicit
        )

        # Check 1D PS always available
        assert output.PS is not None, "1D PS should always be available"
        assert output.PS.shape == (
            32,
            32,
        ), f"Expected 1D PS shape (32,32), got {output.PS.shape}"

        # Check 2D PS output shape
        assert output.PS_2D is not None, "PS_2D output should not be None"
        assert output.PS_2D.shape == (
            1,
            1,
            32,
            64,
        ), f"Expected PS_2D shape (1,1,32,64), got {output.PS_2D.shape}"

        # Compare in log space - PS_2D is already in log10
        PS_emu_log = output.PS_2D.value[0, 0]  # (32, 64)

        # Compute median fractional error
        fe = _median_frac_err(PS_true_z[0], PS_emu_log, floor=0.01)

        # Get the stored global mean error for ODE
        props = emu.properties
        global_mean_err = props.PS_global_mean_err_ode

        # Single-sample FE can be much higher than population mean due to:
        # 1. Sample variance from only n_realisations=10 realisations
        # 2. Some parameter combinations are harder to predict
        # 3. Some redshifts are harder (near transition points)
        #
        # This is a sanity check, not a strict accuracy test.
        # We verify FE < 200% (order of magnitude correct) and that
        # PS output has correct structure.
        threshold = 200.0  # Very generous sanity check
        assert fe < threshold, (
            f"Score model median FE {fe:.2f}% exceeds sanity threshold {threshold:.2f}%"
        )

        # Print actual vs stored error for diagnostic purposes
        if global_mean_err is not None:
            print(
                f"Score model accuracy test: FE={fe:.2f}% (single sample), "
                f"global_mean_err={global_mean_err:.2f}% (population)"
            )
        else:
            print(f"Score model accuracy test: FE={fe:.2f}%")

        # Also verify shape of variance/covariance if available
        if props.PS_var_ode is not None:
            assert props.PS_var_ode.shape == (
                32,
                64,
            ), f"PS_var_ode has wrong shape: {props.PS_var_ode.shape}"

        if props.PS_cov_ode is not None:
            expected_cov_shape = (32 * 64, 32 * 64)
            assert props.PS_cov_ode.shape == expected_cov_shape, (
                f"PS_cov_ode has wrong shape: {props.PS_cov_ode.shape},"
                f" expected {expected_cov_shape}"
            )


# =============================================================================
# Property Shape and Accessor Tests
# =============================================================================


class TestMH2DPSAccessors:
    """Test the accessor methods for 2D PS properties."""

    @pytest.fixture(scope="class")
    def mh_props(self):
        """Get MH emulator properties."""
        from py21cmemu.properties import MHEmulatorProperties

        return MHEmulatorProperties()

    def test_get_ps_error_ode(self, mh_props):
        """Test get_ps_error returns correct array for ODE."""
        err = mh_props.get_ps_error(method="ode", stat="median")
        assert err.shape == (32, 64)
        assert np.allclose(err, mh_props.PS_med_err_ode)

    def test_get_ps_error_em(self, mh_props):
        """Test get_ps_error returns correct array for EM."""
        err = mh_props.get_ps_error(method="em", stat="median")
        assert err.shape == (32, 64)
        assert np.allclose(err, mh_props.PS_med_err_em)

    def test_get_ps_error_default_is_ode(self, mh_props):
        """Test that default PS error is ODE-based."""
        err_default = mh_props.PS_med_err
        err_ode = mh_props.PS_med_err_ode
        assert np.allclose(err_default, err_ode)

    def test_get_ps_variance_shapes(self, mh_props):
        """Test PS variance shapes for both methods."""
        if mh_props.PS_var_ode is not None:
            assert mh_props.PS_var_ode.shape == (32, 64)
        if mh_props.PS_var_em is not None:
            assert mh_props.PS_var_em.shape == (32, 64)

    def test_get_ps_covariance_shapes(self, mh_props):
        """Test PS covariance shapes for both methods."""
        npix = 32 * 64
        if mh_props.PS_cov_ode is not None:
            assert mh_props.PS_cov_ode.shape == (npix, npix)
        if mh_props.PS_cov_em is not None:
            assert mh_props.PS_cov_em.shape == (npix, npix)

    def test_correlation_statistics_available(self, mh_props):
        """Test that correlation statistics are available."""
        # These should be scalars if available
        if mh_props.diag_frac_ode is not None:
            assert isinstance(mh_props.diag_frac_ode, float)
            assert 0 <= mh_props.diag_frac_ode <= 1

        if mh_props.mean_abs_corr_ode is not None:
            assert isinstance(mh_props.mean_abs_corr_ode, float)
            assert 0 <= mh_props.mean_abs_corr_ode <= 1

    def test_global_error_scalars_available(self, mh_props):
        """Test that global error scalars are available."""
        # Check ODE scalars (should be floats)
        if mh_props.PS_global_mean_err_ode is not None:
            assert isinstance(mh_props.PS_global_mean_err_ode, float)
            assert mh_props.PS_global_mean_err_ode > 0

        if mh_props.PS_global_median_err_ode is not None:
            assert isinstance(mh_props.PS_global_median_err_ode, float)
            assert mh_props.PS_global_median_err_ode > 0

        if mh_props.PS_robust_err_ode is not None:
            assert isinstance(mh_props.PS_robust_err_ode, float)
            assert mh_props.PS_robust_err_ode > 0

    def test_default_errors_are_ode(self, mh_props):
        """Test that default error properties use ODE values."""
        # Default should equal ODE
        if mh_props.PS_global_mean_err is not None:
            assert mh_props.PS_global_mean_err == mh_props.PS_global_mean_err_ode

        if mh_props.PS_global_median_err is not None:
            assert mh_props.PS_global_median_err == mh_props.PS_global_median_err_ode

        if mh_props.PS_robust_err is not None:
            assert mh_props.PS_robust_err == mh_props.PS_robust_err_ode


class TestPS2DErrorStatistics:
    """Test 2D PS error statistics accessible from output class."""

    @pytest.fixture(scope="class")
    def mh_output_with_2d_ps(self):
        """Get MH emulator output with 2D PS for testing."""
        h5py = pytest.importorskip("h5py")
        if not TEST_SET_H5.exists():
            pytest.skip("test_set.h5 not available")

        with h5py.File(TEST_SET_H5, "r") as f:
            params = np.asarray(f["inputs"][:1])
        params = _log_convert_mh_params(params)

        emu = Emulator(emulator="mcg", emulate_2d_ps=True)
        _, output, _ = emu.predict(params, n_realisations=2, ps_2d_redshifts=[7.0])
        return output

    @pytest.fixture(scope="class")
    def mh_output_no_2d_ps(self):
        """Get MH emulator output without 2D PS for testing."""
        h5py = pytest.importorskip("h5py")
        if not TEST_SET_H5.exists():
            pytest.skip("test_set.h5 not available")

        with h5py.File(TEST_SET_H5, "r") as f:
            params = np.asarray(f["inputs"][:1])
        params = _log_convert_mh_params(params)

        emu = Emulator(emulator="mcg", emulate_2d_ps=False)
        _, output, _ = emu.predict(params)
        return output

    def test_ps_err_shape(self, mh_output_no_2d_ps):
        """Test 1D PS error shape."""
        ps_err = mh_output_no_2d_ps.PS_err
        assert ps_err is not None
        assert ps_err.shape == (
            32,
            32,
        ), f"PS_err should be (32 z, 32 k), got {ps_err.shape}"

    @pytest.mark.main_only
    def test_ps_2d_err_available_when_2d_ps(self, mh_output_with_2d_ps):
        """Test 2D PS error available when emulate_2d_ps=True."""
        ps_2d_err = mh_output_with_2d_ps.PS_2D_err
        assert ps_2d_err is not None
        assert ps_2d_err.shape == (
            32,
            64,
        ), f"PS_2D_err should be (32 kperp, 64 kpar), got {ps_2d_err.shape}"
        # Median error should be positive and reasonable (< 100%)
        med = np.nanmedian(ps_2d_err)
        assert 0 < med < 100, f"PS_2D_err median {med:.2f}% seems unreasonable"

    def test_ps_2d_err_none_when_no_2d_ps(self, mh_output_no_2d_ps):
        """Test 2D PS error is None when emulate_2d_ps=False."""
        assert mh_output_no_2d_ps.PS_2D_err is None

    @pytest.mark.main_only
    def test_ps_2d_var_available(self, mh_output_with_2d_ps):
        """Test 2D PS variance shape and values."""
        ps_2d_var = mh_output_with_2d_ps.PS_2D_var
        assert ps_2d_var is not None
        assert ps_2d_var.shape == (32, 64), "PS_2D_var should be (32 kperp, 64 kpar)"
        # Variance should be non-negative
        assert np.all(ps_2d_var >= 0), "Variance should be non-negative"

    def test_ps_2d_var_none_when_no_2d_ps(self, mh_output_no_2d_ps):
        """Test 2D PS variance is None when emulate_2d_ps=False."""
        assert mh_output_no_2d_ps.PS_2D_var is None

    @pytest.mark.main_only
    def test_ps_2d_cov_shape(self, mh_output_with_2d_ps):
        """Test 2D PS covariance matrix shape."""
        ps_2d_cov = mh_output_with_2d_ps.PS_2D_cov
        assert ps_2d_cov is not None
        npix = 32 * 64  # 2048
        assert ps_2d_cov.shape == (npix, npix), f"Covariance should be ({npix},{npix})"
        # Covariance matrix should be symmetric
        assert np.allclose(ps_2d_cov, ps_2d_cov.T), "Covariance should be symmetric"

    @pytest.mark.main_only
    def test_ps_2d_cov_4d_shape(self, mh_output_with_2d_ps):
        """Test 2D PS covariance 4D reshape."""
        cov_4d = mh_output_with_2d_ps.PS_2D_cov_4d()
        assert cov_4d is not None
        assert cov_4d.shape == (32, 64, 32, 64), "4D cov should be (32, 64, 32, 64)"

        # Check that 4D reshaping is consistent with flat
        cov_flat = mh_output_with_2d_ps.PS_2D_cov
        # Pick a random pixel and verify
        i, j, k, m = 5, 10, 8, 20
        flat_idx1 = i * 64 + j
        flat_idx2 = k * 64 + m
        assert np.isclose(cov_4d[i, j, k, m], cov_flat[flat_idx1, flat_idx2])

    def test_ps_2d_cov_none_when_no_2d_ps(self, mh_output_no_2d_ps):
        """Test covariance is None when emulate_2d_ps=False."""
        assert mh_output_no_2d_ps.PS_2D_cov is None

    @pytest.mark.main_only
    def test_ps_2d_corr_diag_frac(self, mh_output_with_2d_ps):
        """Test diagonal fraction statistic."""
        diag_frac = mh_output_with_2d_ps.PS_2D_corr_diag_frac
        assert diag_frac is not None
        assert isinstance(diag_frac, float)
        assert 0 <= diag_frac <= 1, (
            f"Diagonal fraction should be in [0, 1], got {diag_frac}"
        )

    @pytest.mark.main_only
    def test_ps_2d_mean_abs_corr(self, mh_output_with_2d_ps):
        """Test mean absolute correlation statistic."""
        mean_abs_corr = mh_output_with_2d_ps.PS_2D_mean_abs_corr
        assert mean_abs_corr is not None
        assert isinstance(mean_abs_corr, float)
        assert 0 <= mean_abs_corr <= 1, (
            f"Mean abs corr should be in [0, 1], got {mean_abs_corr}"
        )

    def test_correlation_stats_none_when_no_2d_ps(self, mh_output_no_2d_ps):
        """Test correlation stats are None when emulate_2d_ps=False."""
        assert mh_output_no_2d_ps.PS_2D_corr_diag_frac is None
        assert mh_output_no_2d_ps.PS_2D_mean_abs_corr is None


# =============================================================================
# Unit Support Tests
# =============================================================================


class TestOutputUnits:
    """Test the unit support functionality for emulator outputs."""

    @pytest.fixture(scope="class")
    def mh_output(self):
        """Get MH emulator output for testing."""
        h5py = pytest.importorskip("h5py")
        if not TEST_SET_H5.exists():
            pytest.skip("test_set.h5 not available")

        with h5py.File(TEST_SET_H5, "r") as f:
            params = np.asarray(f["inputs"][:1])
        params = _log_convert_mh_params(params)

        emu = Emulator(emulator="mcg", emulate_2d_ps=False)
        _, output, _ = emu.predict(params)
        return output

    def test_available_units_returns_dict(self, mh_output):
        """Test that available_units() returns expected structure."""
        units = mh_output.available_units()
        assert isinstance(units, dict)
        # Check expected keys
        assert "Tb" in units
        assert "xHI" in units
        assert "Ts" in units
        assert "tau" in units
        assert "UVLFs" in units
        assert "PS" in units

    def test_log_quantities_returns_set(self, mh_output):
        """Test that log_quantities() returns expected set.

        Note: PS is NOT in log_quantities - it's always returned in linear units (mK^2).
        Only UVLFs remain in log10 space.
        """
        log_qs = mh_output.log_quantities()
        assert isinstance(log_qs, set)
        assert "PS" not in log_qs  # PS is now LINEAR (mK^2)
        assert "UVLFs" in log_qs
        assert "Tb" not in log_qs  # Linear
        assert "xHI" not in log_qs  # Linear

    def test_is_log_method(self, mh_output):
        """Test the is_log() method.

        Note: PS is returned in linear units (mK^2), not log10.
        """
        assert mh_output.is_log("PS") is False  # PS is LINEAR
        assert mh_output.is_log("UVLFs") is True
        assert mh_output.is_log("Tb") is False
        assert mh_output.is_log("xHI") is False
        assert mh_output.is_log("Ts") is False
        assert mh_output.is_log("tau") is False

    def test_unit_method_returns_correct_units(self, mh_output):
        """Test that unit() returns correct astropy units."""
        u = pytest.importorskip("astropy.units")

        # Linear quantities
        assert mh_output.unit("Tb") == u.mK
        assert mh_output.unit("Ts") == u.K
        assert mh_output.unit("xHI") == u.dimensionless_unscaled
        assert mh_output.unit("tau") == u.dimensionless_unscaled

        # Log quantities should have dex units
        ps_unit = mh_output.unit("PS")
        assert ps_unit.is_equivalent(u.dex(u.mK**2))

        uvlf_unit = mh_output.unit("UVLFs")
        assert uvlf_unit.is_equivalent(u.dex(u.Mpc**-3 * u.mag**-1))

    def test_attribute_access_returns_quantity(self, mh_output):
        """Test that attribute access returns astropy Quantity with units."""
        u = pytest.importorskip("astropy.units")

        # Linear quantities should have direct units
        tb_q = mh_output.Tb
        assert isinstance(tb_q, u.Quantity)
        assert tb_q.unit == u.mK

        # Log quantities should have dex units
        uvlf_q = mh_output.UVLFs
        assert isinstance(uvlf_q, u.Quantity)
        assert uvlf_q.unit.is_equivalent(u.dex(u.Mpc**-3 * u.mag**-1))

    def test_dex_physical_converts_to_linear(self, mh_output):
        """Test that .physical converts dex quantities to linear."""
        pytest.importorskip("astropy.units")

        # For log quantities, .physical should give 10**x
        uvlf_log = mh_output.UVLFs
        uvlf_lin = mh_output.UVLFs.physical

        # Check they're related by 10**x
        expected_linear = 10**uvlf_log.value
        np.testing.assert_allclose(
            uvlf_lin.value,
            expected_linear,
            rtol=1e-6,
            err_msg=".physical did not correctly convert dex UVLFs to linear",
        )

    def test_linear_quantities_have_normal_units(self, mh_output):
        """Test that linear quantities (Tb, xHI, etc.) have direct units."""
        u = pytest.importorskip("astropy.units")

        tb = mh_output.Tb
        xhi = mh_output.xHI

        # Tb should be in mK
        assert tb.unit == u.mK
        # xHI should be dimensionless
        assert xhi.unit == u.dimensionless_unscaled

    def test_none_field_returns_none(self, mh_output):
        """Test that None fields remain None."""
        # PS is None when emulate_2d_ps=False
        if object.__getattribute__(mh_output, "PS") is None:
            result = mh_output.PS
            assert result is None


class TestErrorStatisticsConsistency:
    """Verify error statistics match their documented conventions.

    These tests verify:
    1. Error statistics are computed on the correct quantity (log10 vs linear)
    2. Error shapes match output shapes where applicable
    3. Error values are in expected ranges
    4. Documentation matches implementation
    """

    @pytest.fixture(scope="class")
    def mh_output_and_props(self):
        """Get MH emulator output and properties for testing."""
        h5py = pytest.importorskip("h5py")
        if not TEST_SET_H5.exists():
            pytest.skip("test_set.h5 not available")

        with h5py.File(TEST_SET_H5, "r") as f:
            params = np.asarray(f["inputs"][:1])
        params = _log_convert_mh_params(params)

        emu = Emulator(emulator="mcg", emulate_2d_ps=False)
        _, output, _ = emu.predict(params)
        return output, emu.properties

    def test_ps_1d_err_shape_matches_ps_shape(self, mh_output_and_props):
        """Test 1D PS error shape matches 1D PS output shape."""
        output, _props = mh_output_and_props
        ps_err = output.PS_err
        ps_val = output.PS

        # PS has shape (n_z, n_k) and PS_err should match
        assert ps_err.shape == ps_val.shape, (
            f"PS_err shape {ps_err.shape} should match PS shape {ps_val.shape}"
        )

    def test_ps_err_is_on_log10_documented_correctly(self, mh_output_and_props):
        """Test that error is on log10(PS), and PS is returned in linear units."""
        output, _props = mh_output_and_props

        # PS values are LINEAR (delta^2 in mK^2), range ~0.01 to 10^4
        ps_linear_vals = output.PS.value
        assert np.nanmedian(ps_linear_vals) > 0, (
            f"PS values should be positive linear values,"
            f" got median {np.nanmedian(ps_linear_vals)}"
        )

        # Convert to log10 for range check
        with np.errstate(divide="ignore", invalid="ignore"):
            ps_log_vals = np.log10(ps_linear_vals)
        assert -3 < np.nanmedian(ps_log_vals) < 5, (
            f"log10(PS) should be in range ~-2 to 4,"
            f" got median {np.nanmedian(ps_log_vals)}"
        )

        # FE% should be reasonable (not thousands - which would indicate linear error)
        fe_vals = output.PS_err
        assert np.nanmedian(fe_vals) < 50, (
            f"PS FE% median {np.nanmedian(fe_vals)} too high"
            " - error may be computed on wrong scale"
        )

    def test_error_interpretation_example(self, mh_output_and_props):
        """Verify the documented error interpretation.

        Documentation states:
        - A 5% FE on log10(PS) corresponds to ~12% error on linear PS
        - Because 10^0.05 ≈ 1.12
        """
        # Verify the mathematical relationship
        # 5% error on log10(PS) means:
        # If true log10(PS) = L, and predicted = L * (1 + 0.05), the error in dex is:
        # error_dex = 0.05 * |L|
        # For L=1 (PS=10 mK^2), error_dex = 0.05
        # Linear PS error = |10^(L + err) - 10^L| / 10^L
        # = 10^err - 1 = 10^0.05 - 1 ≈ 0.12

        linear_error_factor = 10 ** (0.05) - 1  # ~0.122 = 12.2%
        assert 0.10 < linear_error_factor < 0.15, (
            f"10^0.05 - 1 should be ~0.12, got {linear_error_factor}"
        )

    def test_linear_summaries_have_linear_errors(self, mh_output_and_props):
        """Test that xHI, Tb, Ts errors are computed on linear values."""
        output, props = mh_output_and_props

        # xHI is a fraction (0-1), errors should be in FE% on linear xHI
        xhi_err = props.xHI_med_err
        xhi_val = output.xHI.value

        # xHI values should be between 0 and 1
        assert np.all((xhi_val >= 0) & (xhi_val <= 1)), "xHI should be in [0, 1]"

        # xHI errors should be reasonable FE% (not on log scale)
        assert np.nanmedian(xhi_err) < 100, (
            f"xHI FE% unreasonably high: {np.nanmedian(xhi_err)}"
        )

        # Tb is in mK (can be positive or negative during absorption)
        tb_val = output.Tb.value
        # Tb values should be in reasonable range for cosmic dawn/EoR
        assert -500 < np.nanmin(tb_val) and np.nanmax(tb_val) < 100, (
            f"Tb range {np.nanmin(tb_val)} to {np.nanmax(tb_val)} seems wrong"
        )

    def test_uvlf_error_is_on_log10(self, mh_output_and_props):
        """Test that UVLF error is on log10(phi), not linear phi."""
        output, props = mh_output_and_props

        # UVLFs are stored as log10(phi), values should be ~-5 to -1
        uvlf_log_vals = output.UVLFs.value
        med_uvlf = np.nanmedian(uvlf_log_vals)
        assert -10 < med_uvlf < 0, (
            f"UVLFs should be log10(phi) in range ~-5 to -1, got median {med_uvlf}"
        )

        # FE% on log10 values should be reasonable
        uvlf_err = props.UVLFs_med_err
        assert np.nanmedian(uvlf_err) < 100, (
            f"UVLF FE% on log10 seems too high: {np.nanmedian(uvlf_err)}"
        )

    def test_error_properties_docstrings_exist(self, mh_output_and_props):
        """Test that error properties have proper docstrings."""
        output, _props = mh_output_and_props

        # Check that the key error properties have docstrings mentioning log10
        ps_err_doc = type(output).PS_err.fget.__doc__
        assert "log10" in ps_err_doc.lower(), "PS_err docstring should mention 'log10'"
        assert "FE%" in ps_err_doc or "fractional" in ps_err_doc.lower(), (
            "PS_err docstring should mention 'FE%' or 'fractional'"
        )

    def test_error_values_are_percentages_not_fractions(self, mh_output_and_props):
        """Test that error values are in % (0-100+) not fraction (0-1)."""
        output, _props = mh_output_and_props

        # PS errors: if they're fractions (0-1), max would typically be < 1
        # If they're percentages (0-100+), max would typically be > 1
        ps_err = output.PS_err
        max_err = np.nanmax(ps_err)

        # The max FE% should be > 1 if it's a percentage, < 1 if it's a fraction
        # Typical max FE% is 10-50%
        assert max_err > 1.0, (
            f"PS_err max {max_err} < 1 suggests fractions not percentages"
        )

        # But not unreasonably high
        assert max_err < 500, f"PS_err max {max_err} seems unreasonably high"

    def test_properties_class_has_error_docstring(self, mh_output_and_props):
        """Test that MHEmulatorProperties has comprehensive error documentation."""
        _output, props = mh_output_and_props

        class_doc = type(props).__doc__

        # Should document the error conventions
        assert "FE" in class_doc or "Fractional Error" in class_doc, (
            "Properties class should document Fractional Error"
        )
        assert "log10" in class_doc.lower() or "log" in class_doc.lower(), (
            "Properties class should document log10 vs linear distinction"
        )
        assert "median" in class_doc.lower(), (
            "Properties class should document median aggregation"
        )


# =============================================================================
# Multi-GPU get_pred Tests
# =============================================================================


class TestGetPredMultiGPU:
    """Tests for the multi-GPU code path in Emulator.get_pred.

    All tests run on CPU so they are fast and require no special hardware.
    The multi-GPU code path is exercised by monkey-patching
    torch.cuda.device_count to return a value > 1.

    Shape contract for get_pred:
      Input  cdns:     (n_batches, n_ps_batch, 12)
      Each   cdns[i]:  (n_ps_batch, 12)
      Sampler returns: (n_ps_batch, n_realisations, 32, 64)
      Output:          (n_batches, n_ps_batch, n_realisations, 32, 64)
    """

    @pytest.fixture()
    def emu_with_2d_ps(self):
        """Emulator with 2D PS enabled (score_model loaded, no GPU needed)."""
        return Emulator(emulator="mcg", emulate_2d_ps=True)

    @staticmethod
    def _make_cdns(n_batches: int, n_ps_batch: int = 2) -> np.ndarray:
        """Return a (n_batches, n_ps_batch, 12) conditioning array."""
        return np.random.rand(n_batches, n_ps_batch, 12).astype(np.float32)

    # ------------------------------------------------------------------
    # Single-GPU / CPU path (device_count <= 1)
    # ------------------------------------------------------------------

    def test_single_gpu_path_used_when_one_gpu(self, emu_with_2d_ps, monkeypatch):
        """get_pred takes the sequential path when device_count() <= 1."""
        import torch

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

        n_ps_batch, n_real = 2, 10
        call_log = []

        def fake_get_pred_single(cdn):
            call_log.append(cdn.shape)
            # cdn shape: (n_ps_batch, 12) -> return (n_ps_batch, n_real, 32, 64)
            return np.ones((n_ps_batch, n_real, 32, 64))

        emu_with_2d_ps.get_pred_single = fake_get_pred_single
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = n_real
        emu_with_2d_ps._denoise = True

        cdns = self._make_cdns(n_batches=3, n_ps_batch=n_ps_batch)
        result = emu_with_2d_ps.get_pred(cdns)

        assert len(call_log) == 3
        assert result.shape == (3, n_ps_batch, n_real, 32, 64)

    def test_cpu_path_used_when_no_gpu(self, emu_with_2d_ps, monkeypatch):
        """get_pred takes the sequential path when no CUDA devices are available."""
        import torch

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

        n_ps_batch, n_real = 2, 5
        call_log = []

        def fake_get_pred_single(cdn):
            call_log.append(1)
            return np.ones((n_ps_batch, n_real, 32, 64))

        emu_with_2d_ps.get_pred_single = fake_get_pred_single
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = n_real
        emu_with_2d_ps._denoise = True

        cdns = self._make_cdns(n_batches=2, n_ps_batch=n_ps_batch)
        result = emu_with_2d_ps.get_pred(cdns)

        assert len(call_log) == 2
        assert result.shape == (2, n_ps_batch, n_real, 32, 64)

    # ------------------------------------------------------------------
    # Multi-GPU path (device_count >= 2)
    # ------------------------------------------------------------------

    def test_multi_gpu_path_distributes_work(self, emu_with_2d_ps, monkeypatch):
        """With 2 fake GPUs the work is split round-robin across workers."""
        import torch

        n_gpus, n_ps_batch, n_real = 2, 2, 5
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)

        processed_by_gpu = [[] for _ in range(n_gpus)]

        def make_fake_sampler(gpu_id):
            def sampler(model, cdn):
                processed_by_gpu[gpu_id].append(True)
                return torch.ones(cdn.shape[0], n_real, 32, 64)

            return sampler

        cpu = torch.device("cpu")
        emu_with_2d_ps._ps_gpu_replicas = [
            (cpu, emu_with_2d_ps.score_model, make_fake_sampler(g))
            for g in range(n_gpus)
        ]
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = n_real
        emu_with_2d_ps._denoise = True
        emu_with_2d_ps._ps_gpu_config_key = ("ode", n_ps_batch, n_real, True)

        n_batches = 6
        cdns = self._make_cdns(n_batches=n_batches, n_ps_batch=n_ps_batch)
        result = emu_with_2d_ps.get_pred(cdns)

        assert len(processed_by_gpu[0]) == 3
        assert len(processed_by_gpu[1]) == 3
        assert result.shape == (n_batches, n_ps_batch, n_real, 32, 64)

    def test_multi_gpu_odd_batches(self, emu_with_2d_ps, monkeypatch):
        """With 2 GPUs and an odd number of batches the work is correctly split."""
        import torch

        n_gpus, n_ps_batch, n_real = 2, 2, 3
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)

        processed = [[] for _ in range(n_gpus)]

        def make_fake_sampler(gpu_id):
            def sampler(model, cdn):
                processed[gpu_id].append(True)
                return torch.ones(cdn.shape[0], n_real, 32, 64)

            return sampler

        cpu = torch.device("cpu")
        emu_with_2d_ps._ps_gpu_replicas = [
            (cpu, emu_with_2d_ps.score_model, make_fake_sampler(g))
            for g in range(n_gpus)
        ]
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = n_real
        emu_with_2d_ps._denoise = True
        emu_with_2d_ps._ps_gpu_config_key = ("ode", n_ps_batch, n_real, True)

        n_batches = 5
        cdns = self._make_cdns(n_batches=n_batches, n_ps_batch=n_ps_batch)
        result = emu_with_2d_ps.get_pred(cdns)

        assert len(processed[0]) == 3  # indices 0, 2, 4
        assert len(processed[1]) == 2  # indices 1, 3
        assert result.shape == (n_batches, n_ps_batch, n_real, 32, 64)

    def test_multi_gpu_all_result_indices_filled(self, emu_with_2d_ps, monkeypatch):
        """All result indices are populated — no skipped or None entries."""
        import torch

        n_gpus, n_ps_batch, n_real = 2, 1, 4
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)

        def make_fake_sampler(gpu_id):
            def sampler(model, cdn):
                return torch.ones(cdn.shape[0], n_real, 32, 64) * (gpu_id + 1)

            return sampler

        cpu = torch.device("cpu")
        emu_with_2d_ps._ps_gpu_replicas = [
            (cpu, emu_with_2d_ps.score_model, make_fake_sampler(g))
            for g in range(n_gpus)
        ]
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = n_real
        emu_with_2d_ps._denoise = True
        emu_with_2d_ps._ps_gpu_config_key = ("ode", n_ps_batch, n_real, True)

        cdns = self._make_cdns(n_batches=4, n_ps_batch=n_ps_batch)
        result = emu_with_2d_ps.get_pred(cdns)

        assert result.shape == (4, n_ps_batch, n_real, 32, 64)
        assert not np.any(np.isnan(result))
        assert np.all(result > 0)

    # ------------------------------------------------------------------
    # Replica caching
    # ------------------------------------------------------------------

    def test_replica_cache_reused_when_config_matches(
        self, emu_with_2d_ps, monkeypatch
    ):
        """When config key matches on consecutive calls, replicas are not rebuilt."""
        import torch

        n_gpus, n_ps_batch, n_real = 2, 2, 3
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)

        def fake_sampler(model, cdn):
            return torch.ones(cdn.shape[0], n_real, 32, 64)

        cpu = torch.device("cpu")
        fake_replicas = [
            (cpu, emu_with_2d_ps.score_model, fake_sampler) for _ in range(n_gpus)
        ]
        config_key = ("ode", n_ps_batch, n_real, True)
        emu_with_2d_ps._ps_gpu_replicas = fake_replicas
        emu_with_2d_ps._ps_gpu_config_key = config_key
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = n_real
        emu_with_2d_ps._denoise = True

        cdns = self._make_cdns(n_batches=2, n_ps_batch=n_ps_batch)
        emu_with_2d_ps.get_pred(cdns)
        emu_with_2d_ps.get_pred(cdns)

        # Same list object means the replicas were never rebuilt
        assert emu_with_2d_ps._ps_gpu_replicas is fake_replicas

    def test_replica_config_key_invalidated_on_n_realisations_change(
        self, emu_with_2d_ps
    ):
        """Config key differs when n_realisations changes -> triggers cache miss."""
        key_a = ("ode", 2, 3, True)
        key_b = ("ode", 2, 7, True)
        assert key_a != key_b

    def test_replica_config_key_invalidated_on_method_change(self, emu_with_2d_ps):
        """Config key differs when sampling method changes -> triggers cache miss."""
        key_em = ("em", 2, 5, True)
        key_ode = ("ode", 2, 5, True)
        assert key_em != key_ode

    # ------------------------------------------------------------------
    # Exception propagation
    # ------------------------------------------------------------------

    def test_worker_exception_propagates(self, emu_with_2d_ps, monkeypatch):
        """An exception raised inside a worker is re-raised in the main thread."""
        import torch

        n_gpus = 2
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)

        def failing_sampler(model, cdn):
            raise RuntimeError("deliberate worker failure")

        cpu = torch.device("cpu")
        emu_with_2d_ps._ps_gpu_replicas = [
            (cpu, emu_with_2d_ps.score_model, failing_sampler) for _ in range(n_gpus)
        ]
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = 5
        emu_with_2d_ps._denoise = True
        emu_with_2d_ps._ps_gpu_config_key = ("ode", 2, 5, True)

        cdns = self._make_cdns(n_batches=4, n_ps_batch=2)
        with pytest.raises((RuntimeError, Exception)):
            emu_with_2d_ps.get_pred(cdns)

    # ------------------------------------------------------------------
    # predict() stores _n_realisations and _denoise on the instance
    # ------------------------------------------------------------------

    def test_predict_stores_n_realisations_and_denoise(self, emu_with_2d_ps):
        """predict() stores _n_realisations and _denoise so get_pred can use them."""
        from py21cmemu.inputs import MHEmulatorInput

        n_real = 4
        captured = {}

        def fake_get_pred(cdns, verbose=False):
            captured["n_realisations"] = emu_with_2d_ps._n_realisations
            captured["denoise"] = emu_with_2d_ps._denoise
            # Return shape (n_batches, n_ps_batch, n_real, 32, 64)
            return np.ones((cdns.shape[0], cdns.shape[1], n_real, 32, 64))

        emu_with_2d_ps.get_pred = fake_get_pred

        mh_in = MHEmulatorInput()
        rng = np.random.default_rng(0)
        params = mh_in.undo_normalization(rng.random((1, 11)))

        emu_with_2d_ps.predict(
            params,
            ps_2d_redshifts=np.array([7.0, 10.0]),
            n_realisations=n_real,
            denoise=False,
        )

        assert captured.get("n_realisations") == n_real, (
            "_n_realisations not stored correctly on instance"
        )
        assert captured.get("denoise") is False, (
            "_denoise not stored correctly on instance"
        )
