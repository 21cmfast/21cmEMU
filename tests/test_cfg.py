"""Tests for Config class (config.py)."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_cfg_with_custom_path(tmp_path):
    """Config initialised with an explicit path that does not yet exist."""
    from py21cmemu.config import Config

    cfg_file = tmp_path / "test_config.toml"
    # Path does not exist → triggers the file-creation branch
    cfg = Config(config_file=cfg_file)

    assert cfg_file.exists()
    assert "data-path" in cfg
    assert "disable-network" in cfg


def test_cfg_existing_file(tmp_path):
    """Config initialised twice: second time the file already exists."""
    from py21cmemu.config import Config

    cfg_file = tmp_path / "existing_config.toml"
    # First creation makes the file
    Config(config_file=cfg_file)
    # Second creation reads an existing file (different branch for existence check)
    cfg2 = Config(config_file=cfg_file)
    assert "data-path" in cfg2


def test_cfg_delitem(tmp_path):
    """__delitem__ removes a key and persists the change."""
    from py21cmemu.config import Config

    cfg = Config(config_file=tmp_path / "cfg.toml")
    cfg["my-key"] = "hello"
    assert "my-key" in cfg

    del cfg["my-key"]
    assert "my-key" not in cfg


def test_cfg_repr_and_str(tmp_path):
    """__repr__ and __str__ return non-empty strings containing config data."""
    from py21cmemu.config import Config

    cfg = Config(config_file=tmp_path / "cfg.toml")
    r = repr(cfg)
    s = str(cfg)
    assert "data-path" in r
    assert "data-path" in s


def test_cfg_keys_values_items(tmp_path):
    """keys(), values(), items() yield the config contents."""
    from py21cmemu.config import Config

    cfg = Config(config_file=tmp_path / "cfg.toml")
    keys = list(cfg.keys())
    vals = list(cfg.values())
    items = list(cfg.items())

    assert "data-path" in keys
    assert len(vals) == len(keys)
    assert len(items) == len(keys)
    assert all(isinstance(k, str) for k, _ in items)

    # Path objects should be converted to strings (line 72 coverage)
    cfg["my-path-key"] = Path("/some/test/path")
    assert cfg["my-path-key"] == "/some/test/path"


def test_cfg_update(tmp_path):
    """update() sets multiple keys atomically."""
    from py21cmemu.config import Config

    cfg = Config(config_file=tmp_path / "cfg.toml")
    cfg.update(**{"foo": "bar", "baz": 42})
    assert cfg["foo"] == "bar"
    assert cfg["baz"] == 42


def test_cfg_emu_path(tmp_path):
    """emu_path property returns a Path under data-path."""
    from py21cmemu.config import Config

    cfg = Config(config_file=tmp_path / "cfg.toml")
    assert "21cmEMU" in str(cfg.emu_path)


def test_cfg_pytorch_models_path(tmp_path):
    """pytorch_models_path creates the directory if absent."""
    from py21cmemu.config import Config

    cfg = Config(config_file=tmp_path / "cfg.toml")
    pt_path = cfg.pytorch_models_path
    assert pt_path.exists()


def test_cfg_use_context_manager(tmp_path):
    """use() restores original values after the context exits."""
    from py21cmemu.config import Config

    cfg = Config(config_file=tmp_path / "cfg.toml")
    orig = cfg["data-path"]

    with cfg.use(**{"data-path": "/tmp/overridden_test_value"}):
        assert cfg["data-path"] == "/tmp/overridden_test_value"

    assert cfg["data-path"] == orig
