"""Test cases for the __main__ module."""
import numpy as np
import pytest

from py21cmemu import Emulator
from py21cmemu.outputs import RawEmulatorOutput


def test_output(tmp_path):
    """Test outputs.py and emulator.py."""
    emu = Emulator(version="latest")
    theta = np.random.rand(9 * 5).reshape((5, 9))

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
    assert (check["PS"] == output.PS).all()

    with pytest.raises(ValueError):
        output.write(write_dir / "test_writing.npz", clobber=False)

    # Test that setting store restricts what is written
    output.write(write_dir / "test_writing_small", store=["PS"])
    check = np.load(write_dir / "test_writing_small.npz", allow_pickle=True)[
        "arr_0"
    ].item()
    assert "xHI" not in check
    assert "theta" not in check

    out2 = RawEmulatorOutput(np.random.rand(1098))
    with pytest.raises(ValueError):
        out2.renormalize("foo")

    assert np.all(output["xHI"] == output.xHI)

    output.k
    output.Muv
    output.UVLF_redshifts
    output.PS_redshifts
    output.redshifts


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

    with pytest.raises(ValueError):
        arr = np.random.rand(10 * 5).reshape((5, 10))
        emu_in.make_param_array(arr, normed=True)

    with pytest.raises(TypeError):
        emu_in.make_param_array(7, normed=True)

    with pytest.raises(TypeError):
        arr = np.random.rand(9 * 5).reshape((5, 9))
        arr_tup = [tuple(i) for i in arr]
        emu_in.make_param_array(arr_tup, normed=True)


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
    assert len(list(conf.items())) == 1
    assert len(list(conf.values())) == 1

    conf.__delitem__(key=conf_keys[0])
    assert len(list(conf.items())) == 0
    assert len(list(conf.values())) == 0

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


def test_get_emulator_no_internet(disable_socket):
    """Test get_emulator.py but when there is no internet."""
    from py21cmemu.get_emulator import get_emu_data

    # The data is there, but it cannot do pulls
    with pytest.raises(UserWarning):
        get_emu_data()
