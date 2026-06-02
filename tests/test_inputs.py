"""Tests for emulator input classes (inputs.py)."""

from __future__ import annotations

import numpy as np
import pytest
from typeguard import suppress_type_checks

from py21cmemu import ACGEmulatorInput, MCGEmulatorInput, RadioEmulatorInput
from py21cmemu.properties import emulator_properties

# ══════════════════════════════════════════════════════════════════════════════
# EmulatorInput unit tests
# ══════════════════════════════════════════════════════════════════════════════


def test_mh_input_normalize_all_kinds():
    """MCGEmulatorInput.normalize accepts all valid kind strings."""

    inp = MCGEmulatorInput()
    n = len(inp.astro_param_keys)
    theta = np.zeros((2, n))

    for kind in ("PS", "PS_2D", "LSTM", "SUMMARIES"):
        result = inp.normalize(theta, kind=kind)
        assert result.shape == (2, n)


def test_mh_input_normalize_invalid_kind():
    """MCGEmulatorInput.normalize raises ValueError for unknown kind."""

    inp = MCGEmulatorInput()
    theta = np.zeros((1, len(inp.astro_param_keys)))
    with pytest.raises(ValueError, match="Unknown kind"):
        inp.normalize(theta, kind="invalid_kind")


def test_mh_input_undo_normalization_all_kinds():
    """MCGEmulatorInput.undo_normalization accepts all valid kind strings."""

    inp = MCGEmulatorInput()
    n = len(inp.astro_param_keys)
    theta_normed = np.full((2, n), 0.5)

    for kind in ("PS", "PS_2D", "LSTM", "SUMMARIES"):
        result = inp.undo_normalization(theta_normed, kind=kind)
        assert result.shape == (2, n)


def test_mh_input_undo_normalization_invalid_kind():
    """MCGEmulatorInput.undo_normalization raises ValueError for unknown kind."""

    inp = MCGEmulatorInput()
    theta = np.zeros((1, len(inp.astro_param_keys)))
    with pytest.raises(ValueError):
        inp.undo_normalization(theta, kind="bad_kind")


def test_mh_input_format_theta_for_summaries_default_redshifts():
    """format_theta_for_summaries with redshifts=None uses properties.redshifts."""

    inp = MCGEmulatorInput()
    n = len(inp.astro_param_keys)
    theta_normed = np.full((2, n), 0.5)

    result = inp.format_theta_for_summaries(theta_normed, redshifts=None)
    n_z = len(inp.properties.redshifts)
    assert result.shape == (2, n_z, n + 1)


def test_mh_input_format_theta_for_summaries_explicit_redshifts():
    """format_theta_for_summaries with explicit redshifts."""

    inp = MCGEmulatorInput()
    n = len(inp.astro_param_keys)
    theta_normed = np.full((3, n), 0.5)
    zs = np.array([6.0, 8.0, 10.0])

    result = inp.format_theta_for_summaries(theta_normed, redshifts=zs)
    assert result.shape == (3, 3, n + 1)


def test_mh_input_format_theta_for_ps():
    """format_theta_for_ps concatenates params with normalised redshifts."""

    inp = MCGEmulatorInput()
    n = len(inp.astro_param_keys)
    theta_normed = np.full((2, n), 0.5)
    zs = np.array([6.0, 8.0, 10.0])

    result = inp.format_theta_for_ps(theta_normed, zs)
    # 2 params × 3 redshifts = 6 rows
    assert result.shape == (6, n + 1)


def test_mh_input_format_theta_wrapper():
    """format_theta delegates to format_theta_for_ps."""

    inp = MCGEmulatorInput()
    n = len(inp.astro_param_keys)
    theta_normed = np.full((1, n), 0.5)
    zs = np.array([6.0, 8.0])

    result = inp.format_theta(theta_normed, zs)
    assert result.shape == (2, n + 1)


def test_make_param_array_no_len_raises():
    """make_param_array raises TypeError when input has no __len__."""

    inp = MCGEmulatorInput()
    # suppress_type_checks lets the int reach the function body so TypeError
    # is raised there, rather than typeguard raising TypeCheckError at the call site.
    with suppress_type_checks(), pytest.raises(TypeError):
        inp.make_param_array(42)  # int has no __len__


def test_format_single_theta_vector_wrong_type():
    """_format_single_theta_vector raises TypeError for unsupported types."""
    inp = ACGEmulatorInput()
    n = len(inp.astro_param_keys)
    # tuple has __len__ but is not dict/ndarray/list → TypeError
    bad_input = tuple([0.5] * n)
    with pytest.raises(TypeError):
        inp._format_single_theta_vector(bad_input)


def test_default_input_normalize_and_undo():
    """DefaultEmulatorInput normalize/undo_normalization round-trip."""
    inp = ACGEmulatorInput()
    n = len(inp.astro_param_keys)
    raw = inp.undo_normalization(np.full((2, n), 0.5))
    normed = inp.normalize(raw)
    np.testing.assert_allclose(normed, 0.5, atol=1e-6)


def test_radio_input_normalize_and_undo():
    """RadioEmulatorInput normalize/undo_normalization round-trip."""
    inp = RadioEmulatorInput()
    n = len(inp.astro_param_keys)
    raw = inp.undo_normalization(np.full((2, n), 0.5))
    normed = inp.normalize(raw)
    np.testing.assert_allclose(normed, 0.5, atol=1e-6)


def test_make_list_of_dicts():
    """make_list_of_dicts returns one dict per parameter set."""
    inp = ACGEmulatorInput()
    n = len(inp.astro_param_keys)
    raw = inp.undo_normalization(np.full((3, n), 0.5))
    dicts = inp.make_list_of_dicts(raw, normed=True)
    assert len(dicts) == 3
    assert set(dicts[0].keys()) == set(inp.astro_param_keys)


# ══════════════════════════════════════════════════════════════════════════════
# MHEmulatorInput structural tests (from test_v3.py)
# ══════════════════════════════════════════════════════════════════════════════


def test_mh_inputs_class() -> None:
    """Test MCGEmulatorInput class."""

    inputs = MCGEmulatorInput()
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


class TestMH2DPSInputs:
    """Test 2D PS input formatting without requiring model evaluation."""

    @pytest.fixture(scope="class")
    def mh_inputs(self):
        """Get MH input handler."""

        return MCGEmulatorInput()

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
        from py21cmemu.properties import MCGEmulatorProperties

        props = MCGEmulatorProperties()

        # PS limits: 12 params (11 astro + 1 redshift), 2 bounds each
        assert props.ps_limits.shape == (12, 2)
        # LSTM limits: 12 params (11 astro + 1 redshift), 2 bounds each
        assert props.lstm_limits.shape == (12, 2)
        # Both should have lower < upper for all params
        assert np.all(props.ps_limits[:, 0] < props.ps_limits[:, 1])
        assert np.all(props.lstm_limits[:, 0] < props.lstm_limits[:, 1])


# ══════════════════════════════════════════════════════════════════════════════
# Parametrized make_param_array tests (from test_main.py)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("emu_type", ["default", "radio_background"])
def test_inputs(emu_type):
    """Test that we perform parameter normalization properly."""
    properties = emulator_properties(emulator=emu_type)

    if emu_type == "radio_background":
        emu_in = RadioEmulatorInput()
        limits = properties.limits.copy()
        npars = len(limits)
    else:
        emu_in = ACGEmulatorInput()
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
