"""A module definining the static properties of the Emulator."""
from __future__ import annotations

from pathlib import Path

import numpy as np


USER_PARAMS = {
    "BOX_LEN": 250,
    "DIM": 512,
    "HII_DIM": 128,
    "USE_FFTW_WISDOM": True,
    "HMF": 1,
    "USE_RELATIVE_VELOCITIES": False,
    "POWER_SPECTRUM": 0,
    "N_THREADS": 1,
    "PERTURB_ON_HIGH_RES": False,
    "NO_RNG": False,
    "USE_INTERPOLATION_TABLES": True,
    "FAST_FCOLL_TABLES": False,
    "USE_2LPT": True,
    "MINIMIZE_MEMORY": False,
}

COSMO_PARAMS = {
    "SIGMA_8": 0.82,
    "hlittle": 0.6774,
    "OMm": 0.3075,
    "OMb": 0.0486,
    "POWER_INDEX": 0.97,
}

FLAG_OPTIONS = {
    "USE_HALO_FIELD": False,
    "USE_MINI_HALOS": False,
    "USE_MASS_DEPENDENT_ZETA": True,
    "SUBCELL_RSD": True,
    "INHOMO_RECO": True,
    "USE_TS_FLUCT": True,
    "M_MIN_in_Mass": False,
    "PHOTON_CONS": True,
    "FIX_VCB_AVG": False,
}


class EmulatorProperties:
    """A class that contains the properties of the emulator."""

    def __init__(self):
        here = Path(__file__).parent
        all_emulator_numbers = np.load(here / "emulator_constants.npz")
        self._data = all_emulator_numbers

        self.zs = all_emulator_numbers["zs"]
        self.limits = all_emulator_numbers["limits"]
        self.zs_cut = self.zs[:60]
        self.ks_cut = all_emulator_numbers["ks"][1:-3]
        self.PS_mean = all_emulator_numbers["PS_mean"]
        self.PS_std = all_emulator_numbers["PS_std"]
        self.Tb_mean = all_emulator_numbers["Tb_mean"]
        self.Tb_std = all_emulator_numbers["Tb_std"]
        self.Ts_mean = all_emulator_numbers["Ts_mean"]
        self.Ts_std = all_emulator_numbers["Ts_std"]

        self.tau_mean = all_emulator_numbers["tau_mean"]
        self.tau_std = all_emulator_numbers["tau_std"]

        self.UVLFs_mean = all_emulator_numbers["UVLFs_mean"]
        self.UVLFs_std = all_emulator_numbers["UVLFs_std"]
        self.UVLFs_MUVs = np.append(np.arange(-25, -15, 1.0), np.arange(-15, -4.5, 0.5))
        self.uv_lf_zs = np.array([6, 7, 8, 10])

        self.PS_err = all_emulator_numbers["PS_err"]
        self.Tb_err = all_emulator_numbers["Tb_err"]
        self.Ts_err = all_emulator_numbers["Ts_err"]
        self.xHI_err = all_emulator_numbers["xHI_err"]
        self.tau_err = all_emulator_numbers["tau_err"]
        self.UVLFs_err = all_emulator_numbers["UVLFs_err"]
        self.UVLFs_logerr = all_emulator_numbers["UVLFs_logerr"]

        self.flag_options = FLAG_OPTIONS
        self.user_params = USER_PARAMS
        self.cosmo_params = COSMO_PARAMS

    @property
    def normalized_quantities(self) -> list[str]:
        """Return a list of the normalized quantities predicted by the emulator."""
        return [k.split("_")[0] for k in self._data if k.endswith("_mean")]


emulator_properties = EmulatorProperties()
