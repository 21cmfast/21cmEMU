
import logging
import toml
from appdirs import AppDirs
from pathlib import Path

log = logging.getLogger(__name__)

APPDIR = AppDirs("EMU21cmFAST", "21cmFAST")
LATEST = 'v1'

class Config:
    """
    Class that handles the configuration file.
    """
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = Path(APPDIR.user_config_dir) / 'config.toml'
        else:
            config_file = Path(config_file)

        self.config_file = config_file

        if not self.config_file.exists():
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.config_file.touch()
            log.info(f"Created new config file at {self.config_file}")

        self.config = toml.loads(self.config_file.read_text())

        # Ensure we have the version listed in the config file
        if 'emu-versions' not in self:
            self['emu-versions'] = ()
        if 'data-path' not in self:
            self['data-path'] = APPDIR.user_data_dir

        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

    def add_emulator(self, emu: str):
        self['emu-versions'] += (emu,)
        self.config_file.write_text(toml.dumps(self.config))

    def get_emulator(self, emu: str):
        if emu not in self['emu-versions']:
            raise ValueError(f"Emulator {emu} not found in config file.")
        return Path(self['data-path']) / emu
    
    @property
    def data_path(self):
        return Path(self['data-path'])
    
    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value
        self.config_file.write_text(toml.dumps(self.config))

    def __delitem__(self, key):
        del self.config[key]
        self.config_file.write_text(toml.dumps(self.config))

    def __contains__(self, key):
        return key in self.config

    def __iter__(self):
        return iter(self.config)

    def __len__(self):
        return len(self.config)

    def __repr__(self):
        return repr(self.config)

    def __str__(self):
        return str(self.config)

    def delete(self, key):
        del self.config[key]
        self.config_file.write_text(toml.dumps(self.config))

    def keys(self):
        return self.config.keys()

    def values(self):
        return self.config.values()

    def items(self):
        return self.config.items()

    def update(self, other):
        self.config.update(other)
        self.config_file.write_text(toml.dumps(self.config))

CONFIG = Config()