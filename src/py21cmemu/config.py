"""User-facing configuration for py21cmEMU."""
from __future__ import annotations

import logging
from pathlib import Path

import toml
from appdirs import AppDirs


log = logging.getLogger(__name__)

APPDIR = AppDirs("py21cmEMU")
LATEST = "v1"


class Config:
    """Class that handles the configuration file."""

    def __init__(self, config_file=None):
        if config_file is None:
            config_file = Path(APPDIR.user_config_dir) / "config.toml"
        else:
            config_file = Path(config_file)

        self.config_file = config_file

        if not self.config_file.exists():
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.config_file.touch()
            log.info(f"Created new config file at {self.config_file}")

        self.config = toml.loads(self.config_file.read_text())

        # Ensure we have the version listed in the config file
        if "emu-versions" not in self:
            self["emu-versions"] = ()
        if "data-path" not in self:
            self["data-path"] = APPDIR.user_data_dir

        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

    def add_emulator(self, emu: str):
        """Add a new emulator version to the config."""
        self["emu-versions"] += (emu,)
        self.config_file.write_text(toml.dumps(self.config))

    def get_emulator(self, emu: str):
        """Get the path to the emulator data."""
        if emu not in self["emu-versions"]:
            raise ValueError(f"Emulator {emu} not found in config file.")
        return Path(self["data-path"]) / emu

    @property
    def data_path(self):
        """Get the path to the data directory."""
        return Path(self["data-path"])

    def __getitem__(self, key):
        """Get a value from the config file."""
        return self.config[key]

    def __setitem__(self, key, value):
        """Set a value in the config file."""
        self.config[key] = value
        self.config_file.write_text(toml.dumps(self.config))

    def __delitem__(self, key):
        """Delete a value from the config file."""
        del self.config[key]
        self.config_file.write_text(toml.dumps(self.config))

    def __contains__(self, key):
        """Check if a key is in the config file."""
        return key in self.config

    def __repr__(self):
        """Get the string representation of the config file."""
        return repr(self.config)

    def __str__(self):
        """Get the string representation of the config file."""
        return str(self.config)

    def keys(self):
        """Get the keys in the config file."""
        return self.config.keys()

    def values(self):
        """Get the values in the config file."""
        return self.config.values()

    def items(self):
        """Get the items in the config file."""
        return self.config.items()


CONFIG = Config()
