"""The actual emulator code."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import CONFIG
from .config import LATEST
from .get_emulator import download_emu_data


log = logging.getLogger(__name__)

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


class Emulator:
    r"""A class that loads an emulator and uses it to obtain 21cmFAST summaries.

    Parameters
    ----------
    io_options : dict, optional
        Dict containing 'store' and 'cache_dir' keys with the keys of summaries to
        store and folder path where to store them, respectively. This must be
        provided only if you want to save the emulator output at each evaluation.
    emu_only : bool, optional
        If set to True, skips the 21cmFAST calls to calculate tau_e and UV LFs.
        Set to True only if you don't need tau_e and UV LFs in your output.
    version : str, optional
        Emulator version to use/download, default is 'latest'.

    """

    def __init__(
        self,
        io_options: dict | None = None,
        version: str = "latest",
    ):
        if version == "latest":
            version = LATEST

        if version not in CONFIG["emu-versions"]:
            log.info(
                f"Emulator version {version} not found in config file. Downloading..."
            )
            download_emu_data(version=version)

        emu = tf.keras.models.load_model(CONFIG.get_emulator(version), compile=False)

        self.model = emu
        self.io_options = io_options

        here = Path(__file__).parent
        all_emulator_numbers = np.load(here / "emulator_constants.npz")

        self.flag_options = FLAG_OPTIONS
        self.user_params = USER_PARAMS
        self.cosmo_params = COSMO_PARAMS

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
        self.UVLFs_MUVs = np.append(
            np.arange(-25, -15, 1.0), np.arange(-15, -4.5, 0.5)
        )  # all_emulator_numbers['UVLFs_MUVs']
        self.uv_lf_zs = np.array([6, 7, 8, 10])

        self.PS_err = all_emulator_numbers["PS_err"]
        self.Tb_err = all_emulator_numbers["Tb_err"]
        self.Ts_err = all_emulator_numbers["Ts_err"]
        self.xHI_err = all_emulator_numbers["xHI_err"]
        self.tau_err = all_emulator_numbers["tau_err"]

    def predict(self, astro_params: np.ndarray | dict | list, verbose: bool = False):
        r"""Call the emulator, evaluate it at the given parameters, restore dimensions.

        Parameters
        ----------
        astro_params : np.ndarray or dict
            An array with the nine astro_params input all $\in [0,1]$ OR in the
            21cmFAST AstroParams input units. Dicts (e.g. p21.AstroParams.defining_dict)
            are also accepted formats. Arrays of only dicts are accepted as well
            (for batch evaluation).
        verbose : bool, optional
            If True, prints the emulator prediction.
        emulate_LFs : bool, optional
            Default is True, the UV LFs are emulated.
            If False, uses 21cmFAST to calculate analytically.
        emulate_tau : bool, optional
            Default is True, tau is emulated.
            If False, use 21cmFAST to calculate analytically.
        """
        astro_params, theta = self.format_theta(astro_params)
        emu_pred = self.model.predict(theta, verbose=verbose)

        Tb_pred_normed = emu_pred[:, :84]  # First 84 numbers of emu prediction are Tb
        xHI_pred = emu_pred[:, 84 : 84 * 2]  # Next 84 numbers are xHI
        Ts_pred_normed = emu_pred[:, 2 * 84 : 84 * 3]  # Next 84 numbers are Ts
        Ts_undefined_pred = emu_pred[
            :, 84 * 3
        ]  # Right after Ts is the redshift at which Ts becomes undefined
        PS_pred_normed = emu_pred[:, 84 * 3 + 1 : 84 * 3 + 1 + 60 * 12].reshape(
            (theta.shape[0], 60, 12)
        )  # The 60 z x 12 k PS
        tau_pred_normed = emu_pred[:, 84 * 3 + 1 + 60 * 12]  # tau_e is one number
        UVLFs_pred_normed = emu_pred[
            :, 84 * 3 + 1 + 60 * 12 + 1 :
        ]  # Last 124 numbers are UV LFs at z = 6, 7, 8, 10.

        # Restore dimensions
        PS_pred = self.PS_mean + self.PS_std * PS_pred_normed  # log10(PS[mK^2])
        Ts_pred = self.Ts_mean + self.Ts_std * Ts_pred_normed  # log10(Ts[mK])
        Tb_pred = self.Tb_mean + self.Tb_std * Tb_pred_normed  # Tb[mK]
        tau_pred = self.tau_mean + self.tau_std * tau_pred_normed  # log10(tau_e)
        UVLFs_pred = (
            self.UVLFs_mean + self.UVLFs_std * UVLFs_pred_normed
        )  # log10(\phi/Mpc^{-3})

        UVLFs = np.zeros(
            (UVLFs_pred.shape[0], len(self.uv_lf_zs), len(self.UVLFs_MUVs))
        )
        current_idx = 0
        for i in range(len(self.uv_lf_zs)):
            UVLFs[:, i, :] = UVLFs_pred[
                :, current_idx : current_idx + len(self.UVLFs_MUVs)
            ]
            current_idx += len(self.UVLFs_MUVs)
        # Set the xHI < z(Ts undefined) to 0
        # For Ts, set it to NaN
        xHI_pred_fix = np.zeros(xHI_pred.shape)
        Ts_pred_fix = np.zeros(Ts_pred.shape)
        for i in range(theta.shape[0]):
            zbin = np.argmin(abs(self.zs - Ts_undefined_pred[i]))
            if xHI_pred[i, zbin] < 1e-1:
                xHI_pred_fix[i, zbin:] = xHI_pred[i, zbin:]
            else:
                xHI_pred_fix[i, :] = xHI_pred[i, :]
            Ts_pred_fix[i, zbin:] = Ts_pred[i, zbin:]
            Ts_pred_fix[i, :zbin] = np.nan
        if theta.shape[0] == 1:
            summaries = {
                "delta": 10 ** PS_pred[0, ...],
                "k": self.ks_cut,
                "brightness_temp": Tb_pred[0, ...],
                "spin_temp": 10 ** Ts_pred_fix[0, ...],
                "tau_e": 10 ** tau_pred[0],
                "Muv": self.UVLFs_MUVs[0, ...]
                if len(self.UVLFs_MUVs.shape) == 3
                else self.UVLFs_MUVs,
                "lfunc": UVLFs,
                "uv_lfs_redshifts": self.uv_lf_zs,
                "ps_redshifts": self.zs_cut,
                "redshifts": self.zs,
                "xHI": xHI_pred_fix[0, ...],
            }
        else:
            summaries = {
                "delta": 10**PS_pred,
                "k": self.ks_cut,
                "brightness_temp": Tb_pred,
                "spin_temp": 10**Ts_pred_fix,
                "tau_e": 10**tau_pred,
                "Muv": self.UVLFs_MUVs,
                "lfunc": UVLFs,
                "uv_lfs_redshifts": self.uv_lf_zs,
                "ps_redshifts": self.zs_cut,
                "redshifts": self.zs,
                "xHI": xHI_pred_fix,
            }
        errors = self.get_errors(summaries, theta)
        # Put the summaries and errors in one single dict
        output = summaries.copy()
        for k in errors.keys():
            output[k] = errors[k]

        if (
            self.io_options is not None
            and self.io_options["cache_dir"] is not None
            and len(self.io_options["store"]) > 0
        ):
            fname = "_".join(
                [str(np.round(astro_params[i], 5)) for i in range(len(theta))]
            )
            to_save = {i: output[i] for i in self.io_options["store"]}
            np.savez(fname, to_save)

        return output

    def get_errors(self, summaries: dict, theta: np.ndarray | None = None) -> dict:
        r"""Calculate the emulator error on its outputs.

        Parameters
        ----------
        summaries : dict
            Dict containing the emulator predictions, defined in Emulator.predict
        theta : dict
            Dict containing the normalized parameters, also defined in Emulator.predict

        Returns
        -------
        The mean error on the test set (i.e. independent of theta).
        """
        # For now, we return the mean emulator error (obtained from the test set) for
        # each summary. Some errors are fractional => actual error = fractional
        # error * value
        return {
            "delta_err": self.PS_err / 100.0 * summaries["delta"],
            "brightness_temp_err": self.Tb_err,
            "xHI_err": self.xHI_err,
            "spin_temp_err": self.Ts_err,
            "tau_e_err": self.tau_err / 100.0 * summaries["tau_e"],
        }

    def format_theta(self, astro_params):
        """Format the astro_params input to be a numpy array."""
        astro_param_keys = [
            "F_STAR10",
            "ALPHA_STAR",
            "F_ESC10",
            "ALPHA_ESC",
            "M_TURN",
            "t_STAR",
            "L_X",
            "NU_X_THRESH",
            "X_RAY_SPEC_INDEX",
        ]
        is_astroparams = False
        if isinstance(astro_params, dict):
            theta = np.array([astro_params[key] for key in astro_param_keys])
        elif type(astro_params) == np.ndarray:
            if len(astro_params.shape) > 1 and astro_params.shape[0] > 1:
                theta = np.zeros(astro_params.shape)
                if isinstance(astro_params, dict):
                    theta = np.array([astro_params[key] for key in astro_param_keys])
                elif type(astro_params[0]) == np.ndarray:
                    theta = astro_params.copy()
                else:
                    raise TypeError(
                        "theta is in the wrong format. Should be AstroParams object, "
                        "dict of astro params or nine astrophysical parameters in same "
                        "order as astro_param_keys. It can also be an array of either "
                        "AstroParams objects, dicts, or arrays (not mixed together)."
                    )
            else:
                theta = astro_params.copy()
        else:
            raise TypeError(
                "theta is in the wrong format. Should be AstroParams object, dict or "
                "nine astrophysical parameters in same order as astro_param_keys. It "
                "can also be an array of either AstroParams objects, dicts, or arrays "
                "(not mixed together)."
            )
        if len(theta.shape) == 1:
            theta = theta.reshape([1, -1])
        normed = True
        if not is_astroparams and max(theta.ravel()) <= 1 and min(theta.ravel()) >= 0:
            return (
                (self.undo_normalization(theta), theta)
                if normed
                else (np.array([astro_params]), theta)
            )
        normed = False  # to indicate that input params was not normalised
        theta[:, [0, 2, 4, 6]] = np.log10(theta[:, [0, 2, 4, 6]])
        theta[:, 7] /= 1000
        theta -= self.limits[:, 0]
        theta /= self.limits[:, 1] - self.limits[:, 0]
        # Restore dimensions i.e. undo the limits
        all_astro_params = self.undo_normalization(theta)

        return all_astro_params, theta

    def undo_normalization(self, theta):
        """Undo the normalization of the parameters."""
        theta_wdims = theta.copy()
        theta_wdims *= self.limits[:, 1] - self.limits[:, 0]
        theta_wdims += self.limits[:, 0]
        theta_wdims[:, 7] *= 1000
        all_astro_params = []
        for i in range(theta.shape[0]):
            all_astro_params.append(
                {
                    "F_STAR10": theta_wdims[i, 0],
                    "ALPHA_STAR": theta_wdims[i, 1],
                    "F_ESC10": theta_wdims[i, 2],
                    "ALPHA_ESC": theta_wdims[i, 3],
                    "M_TURN": theta_wdims[i, 4],
                    "t_STAR": theta_wdims[i, 5],
                    "L_X": theta_wdims[i, 6],
                    "NU_X_THRESH": theta_wdims[i, 7],
                    "X_RAY_SPEC_INDEX": theta_wdims[i, 8],
                }
            )
        return all_astro_params
