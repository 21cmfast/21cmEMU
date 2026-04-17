"""21cmEMU: An emulator of 21cmFAST summaries."""

from importlib.metadata import version

__version__ = version("py21cmEMU")
from .emulator import Emulator as Emulator
from .get_emulator import get_emu_data as get_emu_data
from .inputs import DefaultEmulatorInput as DefaultEmulatorInput
from .inputs import MHEmulatorInput as MHEmulatorInput
from .inputs import RadioEmulatorInput as RadioEmulatorInput
from .outputs import ACGEmulatorErrors as ACGEmulatorErrors
from .outputs import EmulatorOutput as EmulatorOutput
from .outputs import MHEmulatorErrors as MHEmulatorErrors
from .outputs import MHRawEmulatorOutput as MHRawEmulatorOutput
from .outputs import RadioEmulatorErrors as RadioEmulatorErrors
from .outputs import RawEmulatorOutput as RawEmulatorOutput
from .properties import (
    DEFAULT_EMULATOR as DEFAULT_EMULATOR,
    EMULATOR_ACG as EMULATOR_ACG,
    EMULATOR_CONFIG as EMULATOR_CONFIG,
    EMULATOR_MCG as EMULATOR_MCG,
    EMULATOR_RADIO as EMULATOR_RADIO,
    EmulatorProperties as EmulatorProperties,
    emulator_properties as emulator_properties,
    resolve_emulator_name as resolve_emulator_name,
)
