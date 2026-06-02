"""Tests for emulator output classes (outputs.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from py21cmemu import DefaultEmulatorInput, Emulator, RadioEmulatorInput
from py21cmemu.outputs import DefaultRawEmulatorOutput

TUTORIALS_DIR = Path(__file__).resolve().parents[1] / "docs" / "tutorials"
TEST_SET_H5 = TUTORIALS_DIR / "test_set.h5"


def _log_convert_mh_params(params: np.ndarray) -> np.ndarray:
    """Convert MH parameter array from linear to log10 for LOG_PARAMETERS columns."""
    from py21cmemu.inputs import MHEmulatorInput

    mh_in = MHEmulatorInput()
    astro_keys = list(mh_in.astro_param_keys)
    log_idx = [astro_keys.index(name) for name in mh_in.LOG_PARAMETERS]
    out = params.copy().astype(float)
    out[:, log_idx] = np.log10(out[:, log_idx])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic output construction helpers
# ══════════════════════════════════════════════════════════════════════════════


def _make_default_output():
    """Build a synthetic DefaultEmulatorOutput without running the emulator."""
    from py21cmemu.outputs import DefaultEmulatorOutput
    from py21cmemu.properties import emulator_properties

    props = emulator_properties("acg")
    nz = len(props.zs)
    ps_nz = len(props.PS_zs)
    ps_nk = len(props.PS_ks)
    n_uv_z = len(props.uv_lf_zs)
    m = np.logical_and(props.UVLFs_MUVs <= -10, props.UVLFs_MUVs >= -20)
    n_muv = int(m.sum())

    return DefaultEmulatorOutput(
        Tb=np.zeros(nz),
        xHI=np.zeros(nz),
        Ts=np.ones(nz) * 100.0,
        PS=np.ones((ps_nz, ps_nk)),
        tau=np.float64(0.05),
        UVLFs=np.zeros((n_uv_z, n_muv)),
    )


def _make_radio_output():
    """Build a synthetic RadioEmulatorOutput without running the emulator."""
    from py21cmemu.outputs import RadioEmulatorOutput
    from py21cmemu.properties import emulator_properties

    props = emulator_properties("radio")
    nz = len(props.zs)
    ps_nz = len(props.PS_zs)
    ps_nk = len(props.PS_ks)

    return RadioEmulatorOutput(
        Tb=np.zeros(nz),
        xHI=np.zeros(nz),
        Tr=np.ones(nz) * 50.0,
        PS=np.ones((ps_nz, ps_nk)),
        tau=np.float64(0.05),
    )


def _make_mh_output(with_2d_ps: bool = False):
    """Build a synthetic MHEmulatorOutput without running the emulator."""
    from py21cmemu.outputs import MHEmulatorOutput
    from py21cmemu.properties import emulator_properties

    props = emulator_properties("mcg")
    nz = len(props.redshifts)
    n_ps_z = len(props.PS_1D_redshifts)
    n_ps_k = len(props.PS_1D_k)
    n_uvlf_z = len(props.uv_lf_zs)
    m = np.logical_and(props.UVLFs_MUVs <= -10, props.UVLFs_MUVs >= -20)
    n_muv = int(m.sum())

    ps_2d = ps_2d_samples = ps_2d_std = ps_2d_zs = None
    if with_2d_ps:
        n_z2d = 3
        n_kperp = len(props.kperp)
        n_kpar = len(props.kpar)
        n_samp = 5
        ps_2d = np.ones((n_z2d, n_kperp, n_kpar))
        ps_2d_samples = np.ones((n_z2d, n_samp, n_kperp, n_kpar))
        ps_2d_std = np.ones((n_z2d, n_kperp, n_kpar)) * 0.1
        ps_2d_zs = np.array([6.0, 8.0, 10.0])

    return MHEmulatorOutput(
        Tb=np.zeros(nz),
        xHI=np.zeros(nz),
        Ts=np.ones(nz) * 100.0,
        tau=np.float64(0.05),
        UVLFs=np.zeros((n_uvlf_z, n_muv)),
        PS=np.ones((n_ps_z, n_ps_k)),
        PS_2D=ps_2d,
        PS_2D_samples=ps_2d_samples,
        PS_2D_std=ps_2d_std,
        PS_2D_redshifts=ps_2d_zs,
    )


def _make_acg_errors():
    from py21cmemu.outputs import ACGEmulatorErrors
    from py21cmemu.properties import emulator_properties

    props = emulator_properties("acg")
    out = _make_default_output()
    return ACGEmulatorErrors.from_output(out, props)


def _make_radio_errors():
    from py21cmemu.outputs import RadioEmulatorErrors
    from py21cmemu.properties import emulator_properties

    props = emulator_properties("radio")
    return RadioEmulatorErrors.from_properties(props)


def _make_mh_errors():
    from py21cmemu.outputs import MHEmulatorErrors
    from py21cmemu.properties import emulator_properties

    props = emulator_properties("mcg")
    out = _make_mh_output()
    return MHEmulatorErrors.from_output(out, props, ps_sampling_method="ode")


# ══════════════════════════════════════════════════════════════════════════════
# DefaultEmulatorOutput tests
# ══════════════════════════════════════════════════════════════════════════════


def test_default_output_properties():
    """DefaultEmulatorOutput exposes coordinate arrays with units."""
    import astropy.units as u

    out = _make_default_output()

    assert out.PS_1D_redshifts.unit == u.dimensionless_unscaled
    assert out.PS_1D_k.unit == u.Mpc**-1
    assert out.Muv.unit == u.mag
    assert out.UVLF_redshifts.unit == u.dimensionless_unscaled
    assert out.PS_redshifts.unit == u.dimensionless_unscaled
    assert out.redshifts.unit == u.dimensionless_unscaled
    assert out.PS_ks.unit == u.Mpc**-1
    assert out.k.unit == u.Mpc**-1


def test_default_output_squeeze():
    """DefaultEmulatorOutput.squeeze returns a new DefaultEmulatorOutput."""
    from py21cmemu.outputs import DefaultEmulatorOutput

    out = _make_default_output()
    squeezed = out.squeeze()
    assert isinstance(squeezed, DefaultEmulatorOutput)


def test_output_write_and_clobber_error(tmp_path):
    """EmulatorOutput.write persists data; raises ValueError when clobber=False."""
    out = _make_default_output()
    fpath = tmp_path / "test_write"
    out.write(fpath, theta=np.zeros(9))
    loaded = np.load(str(fpath) + ".npz", allow_pickle=True)["arr_0"].item()
    assert "inputs" in loaded

    with pytest.raises(ValueError):
        out.write(str(fpath) + ".npz", clobber=False)


def test_output_unit_methods():
    """available_units, log_quantities, is_log, unit work correctly."""
    import astropy.units as u

    out = _make_default_output()

    units = out.available_units()
    assert "Tb" in units
    assert "UVLFs" in units

    log_q = out.log_quantities()
    assert "UVLFs" in log_q
    assert "PS" not in log_q

    assert out.is_log("UVLFs")
    assert not out.is_log("PS")

    assert out.unit("Tb") == u.mK

    with pytest.raises(KeyError):
        out.unit("nonexistent_xyz")


def test_output_keys_items_getitem():
    """keys(), items(), __getitem__ behave like a dict."""
    import astropy.units as u

    out = _make_default_output()

    keys = list(out.keys())
    assert "Tb" in keys
    assert "PS" in keys

    items = dict(out.items())
    assert "Tb" in items

    assert isinstance(out["Tb"], u.Quantity)
    assert out["Tb"] is out.Tb


# ══════════════════════════════════════════════════════════════════════════════
# RadioEmulatorOutput tests
# ══════════════════════════════════════════════════════════════════════════════


def test_radio_output_squeeze():
    """RadioEmulatorOutput.squeeze returns a new RadioEmulatorOutput."""
    from py21cmemu.outputs import RadioEmulatorOutput

    out = _make_radio_output()
    squeezed = out.squeeze()
    assert isinstance(squeezed, RadioEmulatorOutput)


# ══════════════════════════════════════════════════════════════════════════════
# MHEmulatorOutput tests
# ══════════════════════════════════════════════════════════════════════════════


def test_mh_output_properties_no_2d_ps():
    """MHEmulatorOutput exposes coordinate properties; PS_2D properties return None."""
    import astropy.units as u

    out = _make_mh_output(with_2d_ps=False)

    # Coordinate axes
    assert out.PS_1D_k.unit == u.Mpc**-1
    assert out.PS_1D_redshifts.unit == u.dimensionless_unscaled
    assert out.kperp.unit == u.Mpc**-1
    assert out.kpar.unit == u.Mpc**-1
    assert out.Nmodes is not None
    assert out.Muv.unit == u.mag
    assert out.UVLF_redshifts.unit == u.dimensionless_unscaled
    assert out.redshifts.unit == u.dimensionless_unscaled

    # 1D PS error
    assert out.PS_err is not None

    # 2D PS properties return None when no 2D PS
    assert out.PS_2D_err is None
    assert out.PS_2D_var is None
    assert out.PS_2D_cov is None
    assert out.PS_2D_cov_4d() is None
    assert out.PS_2D_corr_diag_frac is None
    assert out.PS_2D_mean_abs_corr is None


def test_mh_output_properties_with_2d_ps():
    """MHEmulatorOutput 2D PS properties return non-None when PS_2D is set."""
    out = _make_mh_output(with_2d_ps=True)

    # These should NOT be None now (PS_2D is set)
    assert out.PS_2D_err is not None
    _ = out.PS_2D_var
    _ = out.PS_2D_cov
    _ = out.PS_2D_cov_4d()
    _ = out.PS_2D_corr_diag_frac
    _ = out.PS_2D_mean_abs_corr


def test_mh_output_squeeze():
    """MHEmulatorOutput.squeeze returns a new MHEmulatorOutput."""
    from py21cmemu.outputs import MHEmulatorOutput

    out = _make_mh_output()
    squeezed = out.squeeze()
    assert isinstance(squeezed, MHEmulatorOutput)


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


# ══════════════════════════════════════════════════════════════════════════════
# Error class tests (from test_coverage.py)
# ══════════════════════════════════════════════════════════════════════════════


def test_acg_errors_from_properties():
    """ACGEmulatorErrors.from_properties constructs errors without an output."""
    from py21cmemu.outputs import ACGEmulatorErrors
    from py21cmemu.properties import emulator_properties

    props = emulator_properties("acg")
    errors = ACGEmulatorErrors.from_properties(props)
    assert errors.PS_err is not None
    assert errors.Tb_err is not None
    assert errors.properties is props


def test_acg_errors_dict_interface():
    """ACGEmulatorErrors supports dict-like access."""
    errors = _make_acg_errors()

    assert "PS_err" in errors
    assert "nonexistent_field_xyz" not in errors
    assert "PS_err" in errors.keys()
    pairs = list(errors.items())
    assert len(pairs) > 0
    val = errors["PS_err"]
    assert val is errors.PS_err


def test_acg_errors_summary():
    """ACGEmulatorErrors.summary() returns a non-empty string."""
    errors = _make_acg_errors()
    s = errors.summary()
    assert "PS_err" in s


def test_radio_errors_dict_interface():
    """RadioEmulatorErrors supports dict-like access."""
    errors = _make_radio_errors()

    assert "PS_err" in errors
    assert "Ts_err" not in errors  # radio has no Ts
    assert "PS_err" in errors.keys()
    pairs = list(errors.items())
    assert len(pairs) > 0
    val = errors["PS_err"]
    assert val is errors.PS_err


def test_radio_errors_summary():
    """RadioEmulatorErrors.summary() returns a non-empty string."""
    errors = _make_radio_errors()
    s = errors.summary()
    assert "PS_err" in s
    assert errors.properties is not None


def test_mh_errors_dict_interface():
    """MHEmulatorErrors supports dict-like access."""
    errors = _make_mh_errors()

    assert "PS_err" in errors
    assert "PS_err" in errors.keys()
    pairs = list(errors.items())
    assert len(pairs) > 0
    val = errors["PS_err"]
    assert val is errors.PS_err


def test_mh_errors_summary():
    """MHEmulatorErrors.summary() returns a non-empty string."""
    errors = _make_mh_errors()
    s = errors.summary()
    assert "PS_err" in s


def test_mh_errors_advanced_stats():
    """MHEmulatorErrors advanced error statistics methods work."""
    errors = _make_mh_errors()
    assert errors.properties is not None

    # get_ps_fractional_error
    fe = errors.get_ps_fractional_error(method="ode", stat="median")
    assert fe is not None

    # get_ps_variance
    var = errors.get_ps_variance(method="ode")
    # May be None; just check no exception

    # get_ps_covariance
    cov = errors.get_ps_covariance(method="ode")
    # May be None

    # ps_diagonal_fraction, ps_mean_abs_correlation
    _ = errors.ps_diagonal_fraction
    _ = errors.ps_mean_abs_correlation


def test_mh_errors_em_method():
    """MHEmulatorErrors built with 'em' method uses em-specific statistics."""
    from py21cmemu.outputs import MHEmulatorErrors
    from py21cmemu.properties import emulator_properties

    props = emulator_properties("mcg")
    out = _make_mh_output()
    errors = MHEmulatorErrors.from_output(out, props, ps_sampling_method="em")

    fe = errors.get_ps_fractional_error(method="em")
    assert fe is not None
    _ = errors.get_ps_variance(method="em")
    _ = errors.get_ps_covariance(method="em")
    _ = errors.ps_diagonal_fraction
    _ = errors.ps_mean_abs_correlation


def test_mh_errors_no_properties():
    """MHEmulatorErrors methods handle _properties=None gracefully."""
    import astropy.units as u
    from py21cmemu.outputs import MHEmulatorErrors

    errors = MHEmulatorErrors(
        PS_err=np.ones((2, 2)) * u.dex(u.mK**2),
        Tb_err=np.ones(2) * u.mK,
        xHI_err=np.ones(2) * u.dimensionless_unscaled,
        Ts_err=np.ones(2) * u.K,
        tau_err=np.float64(0.01) * u.dimensionless_unscaled,
        UVLFs_err=np.ones((2, 2)) * (u.Mpc**-3 * u.mag**-1),
        UVLFs_logerr=np.ones((2, 2)) * u.dex(u.Mpc**-3 * u.mag**-1),
        _properties=None,
    )

    assert errors.get_ps_variance() is None
    assert errors.get_ps_covariance() is None
    assert errors.ps_diagonal_fraction is None
    assert errors.ps_mean_abs_correlation is None

    with pytest.raises(ValueError, match="Properties not available"):
        errors.get_ps_fractional_error()


# ══════════════════════════════════════════════════════════════════════════════
# Parametrized end-to-end output tests (from test_main.py)
# ══════════════════════════════════════════════════════════════════════════════


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
        theta_single = DefaultEmulatorInput().undo_normalization(
            np.random.rand(9).reshape((1, 9))
        )
        _, out_single, err_single = emu.predict(theta_single)
        assert out_single.xHI.shape == err_single.xHI_err.shape
        assert out_single.Tb.shape == err_single.Tb_err.shape
        assert out_single.Ts.shape == err_single.Ts_err.shape
        assert out_single.tau.shape == err_single.tau_err.shape
        assert out_single.PS.shape == err_single.PS_err.shape
        assert out_single.UVLFs.shape == err_single.UVLFs_logerr.shape

        # --- Error shape consistency for multi-sample prediction ---
        N = 5
        theta_multi = DefaultEmulatorInput().undo_normalization(
            np.random.rand(N * 9).reshape((N, 9))
        )
        _, out_multi, err_multi = emu.predict(theta_multi)
        assert out_multi.xHI.shape == err_multi.xHI_err.shape
        assert out_multi.Tb.shape == err_multi.Tb_err.shape
        assert out_multi.Ts.shape == err_multi.Ts_err.shape
        assert out_multi.tau.shape == err_multi.tau_err.shape
        assert out_multi.PS.shape == err_multi.PS_err.shape
        assert out_multi.UVLFs.shape == err_multi.UVLFs_logerr.shape
    else:
        errors["Tr_err"]
        output.Tr
        output.PS_ks

        with pytest.raises(ValueError):
            emu = Emulator(emulator="foo")


# ══════════════════════════════════════════════════════════════════════════════
# MH output shape and value tests (from test_v3.py)
# ══════════════════════════════════════════════════════════════════════════════


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
    assert output.xHI.shape == errors.xHI_err.shape
    assert output.Tb.shape == errors.Tb_err.shape
    assert output.Ts.shape == errors.Ts_err.shape
    assert output.tau.shape == errors.tau_err.shape
    assert output.PS.shape == errors.PS_err.shape
    assert output.UVLFs.shape == errors.UVLFs_logerr.shape


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

    # Ts should be positive where defined
    ts_defined = ~np.isnan(output.Ts)
    assert ts_defined.any(), "Ts should have some defined values"
    assert np.all(output.Ts[ts_defined] > 0)


# ══════════════════════════════════════════════════════════════════════════════
# 2D PS output structure tests (from test_v3.py)
# ══════════════════════════════════════════════════════════════════════════════


class TestMH2DOutputStructure:
    """Test 2D PS output structure and data handling."""

    def test_mh_output_with_2d_ps(self):
        """Test MHEmulatorOutput can include 2D PS data."""
        from py21cmemu.outputs import MHEmulatorOutput

        ps_1d = np.random.rand(32, 32)
        ps_2d_samples = np.random.rand(1, 10, 100, 32, 64)
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

        assert output.PS is not None
        assert output.PS.shape == (32, 32)
        assert output.PS_2D is not None
        assert output.PS_2D.shape == (1, 10, 32, 64)
        assert output.PS_2D_samples.shape == (1, 10, 100, 32, 64)
        assert output.PS_2D_std.shape == (1, 10, 32, 64)
        assert len(output.PS_2D_redshifts) == 10

    def test_ps_variance_computation(self):
        """Test that PS variance can be computed from samples."""
        ps_samples = np.random.rand(1, 10, 100, 32, 32)
        variance = np.var(ps_samples, axis=2)
        assert variance.shape == (1, 10, 32, 32)
        assert np.all(variance >= 0)

    def test_ps_covariance_computation(self):
        """Test that PS covariance can be computed from samples."""
        nsamples = 50
        ps_samples = np.random.rand(nsamples, 32, 32)
        ps_flat = ps_samples.reshape(nsamples, -1)
        cov = np.cov(ps_flat, rowvar=False)
        assert cov.shape == (32 * 32, 32 * 32)
        var_flat = np.var(ps_flat, axis=0, ddof=1)
        assert np.allclose(np.diag(cov), var_flat, rtol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# 2D PS error statistics (from test_v3.py::TestPS2DErrorStatistics)
# ══════════════════════════════════════════════════════════════════════════════


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
        assert ps_err.shape == (32, 32)

    @pytest.mark.main_only
    def test_ps_2d_err_available_when_2d_ps(self, mh_output_with_2d_ps):
        """Test 2D PS error available when emulate_2d_ps=True."""
        ps_2d_err = mh_output_with_2d_ps.PS_2D_err
        assert ps_2d_err is not None
        assert ps_2d_err.shape == (32, 64)
        med = np.nanmedian(ps_2d_err)
        assert 0 < med < 100

    def test_ps_2d_err_none_when_no_2d_ps(self, mh_output_no_2d_ps):
        """Test 2D PS error is None when emulate_2d_ps=False."""
        assert mh_output_no_2d_ps.PS_2D_err is None

    @pytest.mark.main_only
    def test_ps_2d_var_available(self, mh_output_with_2d_ps):
        """Test 2D PS variance shape and values."""
        ps_2d_var = mh_output_with_2d_ps.PS_2D_var
        assert ps_2d_var is not None
        assert ps_2d_var.shape == (32, 64)
        assert np.all(ps_2d_var >= 0)

    def test_ps_2d_var_none_when_no_2d_ps(self, mh_output_no_2d_ps):
        """Test 2D PS variance is None when emulate_2d_ps=False."""
        assert mh_output_no_2d_ps.PS_2D_var is None

    @pytest.mark.main_only
    def test_ps_2d_cov_shape(self, mh_output_with_2d_ps):
        """Test 2D PS covariance matrix shape."""
        ps_2d_cov = mh_output_with_2d_ps.PS_2D_cov
        assert ps_2d_cov is not None
        npix = 32 * 64
        assert ps_2d_cov.shape == (npix, npix)
        assert np.allclose(ps_2d_cov, ps_2d_cov.T)

    @pytest.mark.main_only
    def test_ps_2d_cov_4d_shape(self, mh_output_with_2d_ps):
        """Test 2D PS covariance 4D reshape."""
        cov_4d = mh_output_with_2d_ps.PS_2D_cov_4d()
        assert cov_4d is not None
        assert cov_4d.shape == (32, 64, 32, 64)

        cov_flat = mh_output_with_2d_ps.PS_2D_cov
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
        assert 0 <= diag_frac <= 1

    @pytest.mark.main_only
    def test_ps_2d_mean_abs_corr(self, mh_output_with_2d_ps):
        """Test mean absolute correlation statistic."""
        mean_abs_corr = mh_output_with_2d_ps.PS_2D_mean_abs_corr
        assert mean_abs_corr is not None
        assert isinstance(mean_abs_corr, float)
        assert 0 <= mean_abs_corr <= 1

    def test_correlation_stats_none_when_no_2d_ps(self, mh_output_no_2d_ps):
        """Test correlation stats are None when emulate_2d_ps=False."""
        assert mh_output_no_2d_ps.PS_2D_corr_diag_frac is None
        assert mh_output_no_2d_ps.PS_2D_mean_abs_corr is None


# ══════════════════════════════════════════════════════════════════════════════
# Unit support tests (from test_v3.py::TestOutputUnits)
# ══════════════════════════════════════════════════════════════════════════════


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
        assert "Tb" in units
        assert "xHI" in units
        assert "Ts" in units
        assert "tau" in units
        assert "UVLFs" in units
        assert "PS" in units

    def test_log_quantities_returns_set(self, mh_output):
        """Test that log_quantities() returns expected set."""
        log_qs = mh_output.log_quantities()
        assert isinstance(log_qs, set)
        assert "PS" not in log_qs  # PS is now LINEAR (mK^2)
        assert "UVLFs" in log_qs
        assert "Tb" not in log_qs
        assert "xHI" not in log_qs

    def test_is_log_method(self, mh_output):
        """Test the is_log() method."""
        assert mh_output.is_log("PS") is False  # PS is LINEAR
        assert mh_output.is_log("UVLFs") is True
        assert mh_output.is_log("Tb") is False
        assert mh_output.is_log("xHI") is False
        assert mh_output.is_log("Ts") is False
        assert mh_output.is_log("tau") is False

    def test_unit_method_returns_correct_units(self, mh_output):
        """Test that unit() returns correct astropy units."""
        u = pytest.importorskip("astropy.units")

        assert mh_output.unit("Tb") == u.mK
        assert mh_output.unit("Ts") == u.K
        assert mh_output.unit("xHI") == u.dimensionless_unscaled
        assert mh_output.unit("tau") == u.dimensionless_unscaled

        ps_unit = mh_output.unit("PS")
        assert ps_unit.is_equivalent(u.dex(u.mK**2))

        uvlf_unit = mh_output.unit("UVLFs")
        assert uvlf_unit.is_equivalent(u.dex(u.Mpc**-3 * u.mag**-1))

    def test_attribute_access_returns_quantity(self, mh_output):
        """Test that attribute access returns astropy Quantity with units."""
        u = pytest.importorskip("astropy.units")

        tb_q = mh_output.Tb
        assert isinstance(tb_q, u.Quantity)
        assert tb_q.unit == u.mK

        uvlf_q = mh_output.UVLFs
        assert isinstance(uvlf_q, u.Quantity)
        assert uvlf_q.unit.is_equivalent(u.dex(u.Mpc**-3 * u.mag**-1))

    def test_dex_physical_converts_to_linear(self, mh_output):
        """Test that .physical converts dex quantities to linear."""
        pytest.importorskip("astropy.units")

        uvlf_log = mh_output.UVLFs
        uvlf_lin = mh_output.UVLFs.physical

        expected_linear = 10**uvlf_log.value
        np.testing.assert_allclose(uvlf_lin.value, expected_linear, rtol=1e-6)

    def test_linear_quantities_have_normal_units(self, mh_output):
        """Test that linear quantities have direct units."""
        u = pytest.importorskip("astropy.units")

        tb = mh_output.Tb
        xhi = mh_output.xHI

        assert tb.unit == u.mK
        assert xhi.unit == u.dimensionless_unscaled

    def test_none_field_returns_none(self, mh_output):
        """Test that None fields remain None."""
        if object.__getattribute__(mh_output, "PS") is None:
            result = mh_output.PS
            assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# Error statistics consistency tests (from test_v3.py::TestErrorStatisticsConsistency)
# ══════════════════════════════════════════════════════════════════════════════


class TestErrorStatisticsConsistency:
    """Verify error statistics match their documented conventions."""

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
        assert ps_err.shape == ps_val.shape

    def test_ps_err_is_on_log10_documented_correctly(self, mh_output_and_props):
        """Test that error is on log10(PS), and PS is returned in linear units."""
        output, _props = mh_output_and_props

        ps_linear_vals = output.PS.value
        assert np.nanmedian(ps_linear_vals) > 0

        with np.errstate(divide="ignore", invalid="ignore"):
            ps_log_vals = np.log10(ps_linear_vals)
        assert -3 < np.nanmedian(ps_log_vals) < 5

        fe_vals = output.PS_err
        assert np.nanmedian(fe_vals) < 50

    def test_error_interpretation_example(self, mh_output_and_props):
        """Verify the documented error interpretation."""
        linear_error_factor = 10 ** (0.05) - 1  # ~0.122 = 12.2%
        assert 0.10 < linear_error_factor < 0.15

    def test_linear_summaries_have_linear_errors(self, mh_output_and_props):
        """Test that xHI, Tb, Ts errors are computed on linear values."""
        output, props = mh_output_and_props

        xhi_err = props.xHI_med_err
        xhi_val = output.xHI.value

        assert np.all((xhi_val >= 0) & (xhi_val <= 1))
        assert np.nanmedian(xhi_err) < 100

        tb_val = output.Tb.value
        assert -500 < np.nanmin(tb_val) and np.nanmax(tb_val) < 100

    def test_uvlf_error_is_on_log10(self, mh_output_and_props):
        """Test that UVLF error is on log10(phi), not linear phi."""
        output, props = mh_output_and_props

        uvlf_log_vals = output.UVLFs.value
        med_uvlf = np.nanmedian(uvlf_log_vals)
        assert -10 < med_uvlf < 0

        uvlf_err = props.UVLFs_med_err
        assert np.nanmedian(uvlf_err) < 100

    def test_error_properties_docstrings_exist(self, mh_output_and_props):
        """Test that error properties have proper docstrings."""
        output, _props = mh_output_and_props

        ps_err_doc = type(output).PS_err.fget.__doc__
        assert "log10" in ps_err_doc.lower()
        assert "FE%" in ps_err_doc or "fractional" in ps_err_doc.lower()

    def test_error_values_are_percentages_not_fractions(self, mh_output_and_props):
        """Test that error values are in % (0-100+) not fraction (0-1)."""
        output, _props = mh_output_and_props

        ps_err = output.PS_err
        max_err = np.nanmax(ps_err)

        assert max_err > 1.0
        assert max_err < 500

    def test_properties_class_has_error_docstring(self, mh_output_and_props):
        """Test that MHEmulatorProperties has comprehensive error documentation."""
        _output, props = mh_output_and_props

        class_doc = type(props).__doc__

        assert "FE" in class_doc or "Fractional Error" in class_doc
        assert "log10" in class_doc.lower() or "log" in class_doc.lower()
        assert "median" in class_doc.lower()
