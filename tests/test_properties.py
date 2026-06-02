"""Tests for emulator properties classes (properties.py)."""

from __future__ import annotations

import numpy as np
import pytest

from py21cmemu.properties import emulator_properties

# ══════════════════════════════════════════════════════════════════════════════
# Unit tests (from test_coverage.py)
# ══════════════════════════════════════════════════════════════════════════════


def test_emulator_properties_factory_all_types():
    """emulator_properties returns the right class for each emulator."""
    from py21cmemu.properties import (
        ACGEmulatorProperties,
        MCGEmulatorProperties,
        RadioEmulatorProperties,
    )

    assert isinstance(emulator_properties("acg"), ACGEmulatorProperties)
    assert isinstance(emulator_properties("radio"), RadioEmulatorProperties)
    assert isinstance(emulator_properties("mcg"), MCGEmulatorProperties)


def test_default_properties_normalized_quantities():
    """ACGEmulatorProperties.normalized_quantities uses base-class implementation."""
    props = emulator_properties("acg")
    nq = props.normalized_quantities
    assert isinstance(nq, list)
    assert "PS" in nq
    assert "Tb" in nq


def test_mcg_properties_zs_alias():
    """MCGEmulatorProperties.zs is an alias for redshifts."""
    props = emulator_properties("mcg")
    np.testing.assert_array_equal(props.zs, props.redshifts)


def test_mcg_properties_get_ps_error():
    """get_ps_error returns arrays for both methods and stats."""
    props = emulator_properties("mcg")

    for method in ("ode", "em"):
        for stat in ("median", "mean", "std"):
            result = props.get_ps_error(method=method, stat=stat)
            assert result is not None
            assert result.ndim == 2


def test_mcg_properties_get_ps_variance():
    """get_ps_variance returns an array or None."""
    props = emulator_properties("mcg")
    for method in ("ode", "em"):
        result = props.get_ps_variance(method=method)
        # May be None if not in the npz file
        if result is not None:
            assert result.ndim == 2


def test_mcg_properties_get_ps_covariance():
    """get_ps_covariance returns a 2-D array or None."""
    props = emulator_properties("mcg")
    for method in ("ode", "em"):
        result = props.get_ps_covariance(method=method)
        if result is not None:
            assert result.ndim == 2


def test_resolve_emulator_name_invalid():
    """resolve_emulator_name raises ValueError for unknown names."""
    from py21cmemu.properties import resolve_emulator_name

    with pytest.raises(ValueError, match="Unknown emulator"):
        resolve_emulator_name("nonexistent_emu_xyz")


def test_emulator_aliases_fn():
    """get_emulator_aliases returns canonical + all alias names."""
    from py21cmemu.properties import get_emulator_aliases

    aliases = get_emulator_aliases("mcg")
    assert "mcg" in aliases
    assert "v3" in aliases
    assert "mh" in aliases


# ══════════════════════════════════════════════════════════════════════════════
# Integration tests from test_main.py
# ══════════════════════════════════════════════════════════════════════════════


def test_properties():
    """Test that the properties are loaded correctly."""
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


# ══════════════════════════════════════════════════════════════════════════════
# MH emulator properties tests (from test_v3.py)
# ══════════════════════════════════════════════════════════════════════════════


def test_mh_properties(mh_emulator) -> None:
    """Test access to emulator properties."""
    props = mh_emulator.properties

    assert hasattr(props, "redshifts")
    assert hasattr(props, "astro_param_keys")
    assert len(props.astro_param_keys) == 11
    assert len(props.redshifts) == 93


class TestMH2DPSProperties:
    """Test 2D PS property access without requiring GPU or model evaluation."""

    @pytest.fixture(scope="class")
    def mh_props(self):
        """Get MH emulator properties."""
        from py21cmemu.properties import MCGEmulatorProperties

        return MCGEmulatorProperties()

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
        """Test that z-averaged mean errors are within reasonable bounds."""
        assert np.nanmean(mh_props.xHI_mean_err) < 1.0, "xHI z-avg mean error too high"
        assert np.nanmean(mh_props.Tb_med_err) < 1.0, "Tb z-avg median error too high"
        assert np.nanmean(mh_props.Ts_mean_err) < 1.0 or np.isnan(
            np.nanmean(mh_props.Ts_mean_err)
        ), "Ts z-avg mean error too high"
        assert mh_props.tau_mean_err < 1.0, "tau mean error too high"
        assert np.nanmean(mh_props.UVLFs_med_err) < 1.0, (
            "UVLFs z-avg median error too high"
        )
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


class TestMH2DPSAccessors:
    """Test the accessor methods for 2D PS properties."""

    @pytest.fixture(scope="class")
    def mh_props(self):
        """Get MH emulator properties."""
        from py21cmemu.properties import MCGEmulatorProperties

        return MCGEmulatorProperties()

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
        if mh_props.diag_frac_ode is not None:
            assert isinstance(mh_props.diag_frac_ode, float)
            assert 0 <= mh_props.diag_frac_ode <= 1

        if mh_props.mean_abs_corr_ode is not None:
            assert isinstance(mh_props.mean_abs_corr_ode, float)
            assert 0 <= mh_props.mean_abs_corr_ode <= 1

    def test_global_error_scalars_available(self, mh_props):
        """Test that global error scalars are available."""
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
        if mh_props.PS_global_mean_err is not None:
            assert mh_props.PS_global_mean_err == mh_props.PS_global_mean_err_ode

        if mh_props.PS_global_median_err is not None:
            assert mh_props.PS_global_median_err == mh_props.PS_global_median_err_ode

        if mh_props.PS_robust_err is not None:
            assert mh_props.PS_robust_err == mh_props.PS_robust_err_ode
