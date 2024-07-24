"""A module definining the static properties of the Emulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class EmulatorProperties:
    """A class that contains the properties of the emulator."""

    @property
    def normalized_quantities(self) -> list[str]:
        """Return a list of the normalized quantities predicted by the emulator."""
        return [k.split("_")[0] for k in self._data if k.endswith("_mean")]


class DefaultEmulatorProperties(EmulatorProperties):
    """A class that contains the properties of the default emulator."""

    def __init__(self):
        here = Path(__file__).parent
        all_emulator_numbers = np.load(here / "emulator_constants.npz")
        self._data = all_emulator_numbers

        self.zs = all_emulator_numbers["zs"]
        self.limits = all_emulator_numbers["limits"]
        self.PS_zs = self.zs[:60]
        self.PS_ks = all_emulator_numbers["ks"][1:-3]
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
        self.parameter_labels = np.array(
            [
                r"$\log_{10} f_{*,10}$",
                r"$\alpha_\ast$",
                r"$\log_{10} f_{\rm esc, 10}$",
                r"$\alpha_{\rm esc}$",
                r"$\log_{10}M_{\rm turn}$",
                r"$t_{\ast}$",
                r"$\log_{10}L_{\rm X<2keV}/{\rm SFR}$",
                r"$E_0$",
                r"$\alpha_{\rm X}$",
            ]
        )
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

        self.flag_options = FLAG_OPTIONS
        self.user_params = USER_PARAMS
        self.cosmo_params = COSMO_PARAMS


class RadioBackgroundEmulatorProperties(EmulatorProperties):
    """A class that contains the properties of the radio background emulator."""

    def __init__(self):
        here = Path(__file__).parent
        all_emulator_numbers = np.load(
            here / "models/radio_background/radio_background_emu_csts.npz"
        )
        self._data = all_emulator_numbers

        self.logPS_mean = all_emulator_numbers["logPS_mean"]
        self.logPS_std = all_emulator_numbers["logPS_std"]
        self.PS_ks = all_emulator_numbers["PS_k"]
        self.PS_zs = all_emulator_numbers["PS_z"]
        self.zs = all_emulator_numbers["redshifts"]
        self.limits = np.array([[-2, 6], [33, 45], [-5, 0], [-6, -1], [0, 10]])
        self.logTb_std = all_emulator_numbers["Tb_std"]
        self.logTb_mean = all_emulator_numbers["Tb_mean"]
        self.Tb_scale = all_emulator_numbers["Tb_scale"]
        self.logTr_mean = all_emulator_numbers["logTr_mean"]
        self.logTr_std = all_emulator_numbers["logTr_std"]

        with np.load(here / "models/radio_background/median_test_errors.npz") as f:
            self.PS_err = f["PS_err"]
            self.Tr_err = f["Tr_err"]
            self.Tb_err = f["Tb_err"]
            self.xHI_err = f["xHI_err"]
            self.tau_err = f["tau_err"]
        self.parameter_labels = [
            r"log$_{10}$ f$_{\rm R,mini}$",
            r"log$_{10}$L$_{\rm X, mini}$",
            r"log$_{10}$F$_{\ast,mini}$",
            r"log$_{10}$F$_{\rm esc, mini}$",
            r"A$_{\rm LW}$",
        ]

        USER_PARAMS = {
            "HII_DIM": 50,
            "N_THREADS": 1,
            "USE_RELATIVE_VELOCITIES": True,
            "USE_INTERPOLATION_TABLES": True,
            "FAST_FCOLL_TABLES": True,
            "MINIMIZE_MEMORY": False,
            "BOX_LEN": 500,
        }

        COSMO_PARAMS = {
            "SIGMA_8": 0.8102,
            "hlittle": 0.6766,
            "OMm": 0.30964144154550644,
            "OMb": 0.04897468161869667,
            "POWER_INDEX": 0.9665,
        }

        FLAG_OPTIONS = {
            "USE_MINI_HALOS": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
            "USE_RADIO_ACG": False,
            "USE_RADIO_MCG": True,
            "Calibrate_EoR_feedback": True,
        }
        self.flag_options = FLAG_OPTIONS
        self.user_params = USER_PARAMS
        self.cosmo_params = COSMO_PARAMS


def emulator_properties(emulator: str = "default") -> EmulatorProperties:
    """Return the properties of the corresponding emulator."""
    if emulator == "default":
        return DefaultEmulatorProperties()
    elif emulator == "radio_background":
        return RadioBackgroundEmulatorProperties()
    else:
        raise ValueError(
            "Please supply one of the following emulator names: 'default'"
            + "or 'radio_background'. "
            + f"{emulator} is not a valid emulator name."
        )
