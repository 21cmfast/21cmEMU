"""Tests for emulator download / data initialisation (get_emulator.py).

These tests require network access and modify the on-disk emulator cache.
They are excluded from the standard fast test run with:
  -k "not test_config and not test_get_emulator"
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from py21cmemu.config import CONFIG


def test_config(tmp_path):
    """Test config.py."""
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
