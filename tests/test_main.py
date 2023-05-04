"""Test cases for the __main__ module."""
import numpy as np

from py21cmemu import Emulator


def test_basic_prediction():
    """Simply test that we can make a prediction without erroring."""
    emu = Emulator(version="latest")
    theta = np.random.rand(9 * 5).reshape((5, 9))

    theta, output, errors = emu.predict(theta)


def test_properties():
    """Test that the properties are loaded correctly."""
    from .properties import emulator_properties as properties

    properties.limits


def test_inputs():
    """Test that we perform parameter normalization properly."""
    from py21cmemu import EmulatorInput

    from .properties import emulator_properties as properties

    limits = properties.limits

    single_param = np.random.rand(9).reshape((1, 9))
    inp = EmulatorInput().make_param_array(single_param, normed=True)

    assert (
        inp.min() >= 0 and inp.max() <= 1
    ), "Single param set not normalized properly."

    inp = EmulatorInput().make_param_array(single_param, normed=False)

    assert (inp >= limits[:, 0]).all() and (
        inp <= limits[:, 1]
    ).all(), "Single param set dimensions not restored properly."

    # Test for many params at once, array
    many_params = np.random.rand(9 * 5).reshape((5, 9))

    inp = EmulatorInput().make_param_array(many_params, normed=True)

    assert inp.min() >= 0 and inp.max() <= 1, "Many params not normalized properly."

    inp = EmulatorInput().make_param_array(many_params, normed=False)

    assert (
        [(i >= limits[:, 0]).all() and (i <= limits[:, 1]).all() for i in inp]
    ).all(), "Many params dimensions not restored properly."

    # Test for single dict
    single_param = {}
    for i in EmulatorInput.astro_param_keys:
        single_param[i] = np.random.rand()

    inp = EmulatorInput().make_param_array(single_param, normed=True)

    assert (
        inp.min() >= 0 and inp.max() <= 1
    ), "Single param dict not normalized properly."

    inp = EmulatorInput().make_param_array(single_param, normed=False)
    assert (inp >= limits[:, 0]).all() and (
        inp <= limits[:, 1]
    ).all(), "Single param dict dimensions not restored properly."

    # Test for array / list of dicts

    many_params_list = [single_param, single_param, single_param]
    inp = EmulatorInput().make_param_array(many_params_list, normed=False)

    assert (
        [(i >= limits[:, 0]).all() and (i <= limits[:, 1]).all() for i in inp]
    ).all(), "Many params list dimensions not restored properly."

    many_params_list = np.array([single_param, single_param, single_param])
    inp = EmulatorInput().make_param_array(many_params_list, normed=False)

    assert (
        [(i >= limits[:, 0]).all() and (i <= limits[:, 1]).all() for i in inp]
    ).all(), "Many params array dimensions not restored properly."
