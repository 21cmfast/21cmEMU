"""21cmEMU: An emulator of 21cmFAST summaries."""

from importlib.metadata import version

__version__ = version("py21cmEMU")
from .emulator import Emulator as Emulator
from .get_emulator import get_emu_data as get_emu_data
from .inputs import ACGEmulatorInput as ACGEmulatorInput
from .inputs import MCGEmulatorInput as MCGEmulatorInput
from .inputs import RadioEmulatorInput as RadioEmulatorInput
from .outputs import ACGEmulatorErrors as ACGEmulatorErrors
from .outputs import EmulatorOutput as EmulatorOutput
from .outputs import MCGEmulatorErrors as MCGEmulatorErrors
from .outputs import MCGRawEmulatorOutput as MCGRawEmulatorOutput
from .outputs import RadioEmulatorErrors as RadioEmulatorErrors
from .outputs import RawEmulatorOutput as RawEmulatorOutput
from .properties import (
    DEFAULT_EMULATOR as DEFAULT_EMULATOR,
)
from .properties import (
    EMULATOR_ACG as EMULATOR_ACG,
)
from .properties import (
    EMULATOR_CONFIG as EMULATOR_CONFIG,
)
from .properties import (
    EMULATOR_MCG as EMULATOR_MCG,
)
from .properties import (
    EMULATOR_RADIO as EMULATOR_RADIO,
)
from .properties import (
    EmulatorProperties as EmulatorProperties,
)
from .properties import (
    emulator_properties as emulator_properties,
)
from .properties import (
    resolve_emulator_name as resolve_emulator_name,
)
