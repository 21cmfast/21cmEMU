"""Test cases for the __main__ module."""
from pathlib import Path

import numpy as np

from py21cmemu import Emulator


def test_prediction():
    """Simply test that we can make a prediction without erroring."""
    emu = Emulator(version="latest")
    theta = np.random.rand(9 * 5).reshape((5, 9))

    theta, output, errors = emu.predict(theta)

    # Test writing
    tmp_path = Path("PYTEST_TMPDIR/test_create_file")
    dir = tmp_path / "test_tmp"
    dir.mkdir()
    output.write(dir + "/test_writing", theta=theta)
    check = np.load(dir + "/test_writing")
    assert (check["inputs"] == theta).all()
    assert len(check.keys()) == len(output.keys()) + 1
    assert (check["delta"] == output["delta"]).all()


def test_properties():
    """Test that the properties are loaded correctly."""
    from py21cmemu.properties import emulator_properties as properties

    properties.limits


def test_inputs():
    """Test that we perform parameter normalization properly."""
    from py21cmemu import EmulatorInput
    from py21cmemu.properties import emulator_properties as properties

    emu_in = EmulatorInput()
    limits = properties.limits.copy()
    limits[7, :] *= 1000.0  # keV to eV

    single_param = np.random.rand(9)
    inp = emu_in.make_param_array(single_param, normed=True)

    assert (inp == single_param).all(), "Single param 1D array not normalized properly."

    inp = emu_in.make_param_array(single_param, normed=False)

    assert (inp >= limits[:, 0]).all() and (
        inp <= limits[:, 1]
    ).all(), "Single param 1D array dimensions not restored properly."

    single_param = np.random.rand(9).reshape((1, 9))
    inp = emu_in.make_param_array(single_param, normed=True)

    assert (inp == single_param).all(), "Single param array not normalized properly."

    inp = emu_in.make_param_array(single_param, normed=False)

    assert (inp >= limits[:, 0]).all() and (
        inp <= limits[:, 1]
    ).all(), "Single param array dimensions not restored properly."

    # Test for many params at once, array
    many_params = np.random.rand(9 * 5).reshape((5, 9))

    inp = emu_in.make_param_array(many_params, normed=True)

    assert (
        (inp == many_params).ravel().all()
    ), "Many params array not normalized properly."

    inp = emu_in.make_param_array(many_params, normed=False)

    assert np.array(
        [(i >= limits[:, 0]).all() and (i <= limits[:, 1]).all() for i in inp]
    ).all(), "Many params array dimensions not restored properly."

    # Test for single dict
    single_param = {}
    arr = np.zeros(len(EmulatorInput.astro_param_keys))
    for k, i in enumerate(EmulatorInput.astro_param_keys):
        single_param[i] = np.random.rand()
        arr[k] = single_param[i]

    inp = emu_in.make_param_array(single_param, normed=True)

    assert (inp == arr).all(), "Single param dict not normalized properly."

    inp = emu_in.make_param_array(single_param, normed=False)
    assert (inp >= limits[:, 0]).all() and (
        inp <= limits[:, 1]
    ).all(), "Single param dict dimensions not restored properly."

    # Test for array / list of dicts

    many_params_list = [single_param, single_param, single_param]
    inp = emu_in.make_param_array(many_params_list, normed=False)

    assert np.array(
        [(i >= limits[:, 0]).all() and (i <= limits[:, 1]).all() for i in inp]
    ).all(), "Many params list of dicts dimensions not restored properly."

    many_params_list = np.array([single_param, single_param, single_param])
    inp = emu_in.make_param_array(many_params_list, normed=False)

    assert np.array(
        [(i >= limits[:, 0]).all() and (i <= limits[:, 1]).all() for i in inp]
    ).all(), "Many params array of dicts dimensions not restored properly."

    # Test for list / list of lists
    arr_list = list(arr)
    inp = emu_in.make_param_array(arr_list, normed=True)

    assert (inp == arr).all(), "Single param list not normalized properly."

    many_params_list = [arr_list, arr_list, arr_list]
    inp = emu_in.make_param_array(many_params_list, normed=False)

    assert np.array(
        [(i >= limits[:, 0]).all() and (i <= limits[:, 1]).all() for i in inp]
    ).all(), "Many params list of lists dimensions not restored properly."

    # Test undo_normalisation

    arr = (
        np.random.rand(len(EmulatorInput.astro_param_keys))
        * (limits[:, 1] - limits[:, 0])
        + limits[:, 0]
    )
    arr[[0, 2, 4, 6]] = 10 ** (arr[[0, 2, 4, 6]])
    arr[7] *= 1000  # keV to eV

    inp = emu_in.make_param_array(arr, normed=False)

    assert (arr == inp).all(), "Single param array w norm failed."

    arr[7] /= 1000  # eV to keV
    inp = emu_in.make_param_array(arr, normed=True)

    assert inp.min() >= 0 and inp.max() <= 1, "Single param array w norm undo failed."

    # Test make_list_of_dicts

    pars = np.random.rand(10 * 9).reshape((10, 9))

    inp = emu_in.make_list_of_dicts(pars, normed=True)
