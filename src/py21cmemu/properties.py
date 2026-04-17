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


class MHEmulatorProperties(EmulatorProperties):
    """A class that contains the properties of the minihalo (v3) emulator.
    
    This loads constants from two separate files:
    - lstm_emulator_constants.npz: LSTM model (summaries + 1D PS)
    - score_model_constants.npz: 2D PS score model
    """

    @property
    def normalized_quantities(self) -> list[str]:
        """Return a list of the normalized quantities predicted by the emulator.
        
        Note: LSTM uses _bias/_scale naming convention; the base class looks for
        _mean in _data keys which doesn't work here. xHI is not normalized
        (it's already in [0, 1] range).
        """
        return ["Tb", "Ts", "tau", "UVLFs"]

    def __init__(self):
        here = Path(__file__).parent
        
        # Load LSTM model constants (summaries + 1D PS)
        lstm_data = np.load(
            here / "models/MHs/lstm_emulator_constants.npz", allow_pickle=True
        )
        self._lstm_data = lstm_data
        # Set _data for base class compatibility (normalized_quantities property)
        self._data = lstm_data
        
        # Load 2D PS score model constants
        score_data = np.load(
            here / "models/MHs/score_model_constants.npz", allow_pickle=True
        )
        self._score_data = score_data

        # === Parameter info (from LSTM) ===
        self.astro_param_keys = tuple(lstm_data["param_names"])
        self.parameter_labels = lstm_data["param_labels"]

        # === LSTM model properties ===
        self.redshifts = lstm_data["lstm_redshifts"]
        self.zs = self.redshifts
        self.lstm_limits = lstm_data["LSTM_limits"]

        # Normalization constants
        self.Tb_mean = float(lstm_data["Tb_bias"])
        self.Tb_std = float(lstm_data["Tb_scale"])
        self.Ts_mean = float(lstm_data["Ts_allgas_bias"])
        self.Ts_std = float(lstm_data["Ts_allgas_scale"])
        self.tau_mean = float(lstm_data["tau_bias"])
        self.tau_std = float(lstm_data["tau_scale"])

        self.UVLFs_mean = np.array(lstm_data["UVLFs_bias"])
        self.UVLFs_std = np.array(lstm_data["UVLFs_scale"])
        self.UVLFs_MUVs = np.array(lstm_data["M_UV"])
        self.uv_lf_zs = np.array(lstm_data["UVLF_zs"])

        # 1D PS properties (from LSTM model)
        self.PS_1D_k = np.array(lstm_data["PS_k"])
        self.PS_1D_redshifts = np.array(lstm_data["PS_redshifts"])
        self.PS_1D_bias = float(lstm_data["PS_bias"])
        self.PS_1D_scale = float(lstm_data["PS_scale"])

        # === 2D PS score model properties ===
        self.ps_limits = score_data["PS_2D_limits"]
        self.PS_zs = np.array(score_data["ps_redshifts"])
        self.PS_redshifts = self.PS_zs
        # Default redshifts for PS emulation (user can override)
        self.default_ps_redshifts = np.array([
            5.5, 6.97446005, 7.54906604, 7.9582024, 9.82883407,
            10.36152691, 10.63860385, 16.66170964, 19.52022545, 24.10859229
        ])
        self.kperp = np.array(score_data["kperp"])
        self.kpar = np.array(score_data["kpar"])
        self.Nmodes = np.array(score_data["Nmodes"])
        self.PS_bias = np.array(score_data["PS_2D_bias"])
        self.PS_scale = np.array(score_data["PS_2D_scale"])

        # === LSTM model errors (per-z arrays) ===
        self.xHI_med_err = np.array(lstm_data["xHI_med_err"])
        self.xHI_mean_err = np.array(lstm_data["xHI_mean_err"])
        self.xHI_std_err = np.array(lstm_data["xHI_std_err"])
        
        self.Tb_med_err = np.array(lstm_data["Tb_med_err"])
        self.Tb_mean_err = np.array(lstm_data["Tb_mean_err"])
        self.Tb_std_err = np.array(lstm_data["Tb_std_err"])
        
        self.Ts_med_err = np.array(lstm_data["Ts_med_err"])
        self.Ts_mean_err = np.array(lstm_data["Ts_mean_err"])
        self.Ts_std_err = np.array(lstm_data["Ts_std_err"])
        
        self.tau_med_err = float(lstm_data["tau_med_err"])
        self.tau_mean_err = float(lstm_data["tau_mean_err"])
        self.tau_std_err = float(lstm_data["tau_std_err"])
        
        self.UVLFs_med_err = np.array(lstm_data["UVLFs_med_err"])
        self.UVLFs_mean_err = np.array(lstm_data["UVLFs_mean_err"])
        self.UVLFs_std_err = np.array(lstm_data["UVLFs_std_err"])
        self.UVLFs_med_logerr = np.array(lstm_data["UVLFs_med_logerr"])
        
        # 1D PS errors (shape: 32 z, 32 k)
        self.PS_1D_med_err = np.array(lstm_data["PS_med_err"])
        self.PS_1D_mean_err = np.array(lstm_data["PS_mean_err"])
        self.PS_1D_std_err = np.array(lstm_data["PS_std_err"])

        # === 2D PS score model errors (shape: 32 kperp, 64 kpar) ===
        # Default to ODE (more accurate sampler)
        if "PS_med_err" in score_data:
            self.PS_med_err = np.array(score_data["PS_med_err"])
        elif "PS_med_err_ode" in score_data:
            self.PS_med_err = np.array(score_data["PS_med_err_ode"])
        else:
            raise KeyError("PS_med_err or PS_med_err_ode required in score_model_constants.npz")
        
        # Method-specific errors
        if "PS_med_err_em" in score_data:
            self.PS_med_err_em = np.array(score_data["PS_med_err_em"])
            self.PS_med_err_ode = np.array(score_data["PS_med_err_ode"])
        else:
            self.PS_med_err_em = self.PS_med_err
            self.PS_med_err_ode = self.PS_med_err
        
        # 2D PS mean/std errors (default to ODE - more accurate sampler)
        if "PS_mean_err" in score_data:
            self.PS_mean_err = np.array(score_data["PS_mean_err"])
        elif "PS_mean_err_ode" in score_data:
            self.PS_mean_err = np.array(score_data["PS_mean_err_ode"])
        else:
            self.PS_mean_err = self.PS_med_err.copy()
        
        if "PS_mean_err_em" in score_data:
            self.PS_mean_err_em = np.array(score_data["PS_mean_err_em"])
            self.PS_mean_err_ode = np.array(score_data["PS_mean_err_ode"])
        else:
            self.PS_mean_err_em = self.PS_mean_err
            self.PS_mean_err_ode = self.PS_mean_err
        
        if "PS_std_err" in score_data:
            self.PS_std_err = np.array(score_data["PS_std_err"])
        elif "PS_std_err_ode" in score_data:
            self.PS_std_err = np.array(score_data["PS_std_err_ode"])
        else:
            self.PS_std_err = np.zeros_like(self.PS_med_err)
        
        if "PS_std_err_em" in score_data:
            self.PS_std_err_em = np.array(score_data["PS_std_err_em"])
            self.PS_std_err_ode = np.array(score_data["PS_std_err_ode"])
        else:
            self.PS_std_err_em = self.PS_std_err
            self.PS_std_err_ode = self.PS_std_err
        
        # === 2D PS variance and covariance (from score model) ===
        # Variance: mean variance across test set (shape: H, W)
        # Default to ODE
        if "PS_var_ode" in score_data:
            self.PS_var_ode = np.array(score_data["PS_var_ode"])
            self.PS_var = self.PS_var_ode
        else:
            self.PS_var_ode = None
            self.PS_var = None
        
        if "PS_var_em" in score_data:
            self.PS_var_em = np.array(score_data["PS_var_em"])
        else:
            self.PS_var_em = None
        
        # Covariance: mean covariance matrix across test set (shape: H*W, H*W)
        # Default to ODE
        if "PS_cov_ode" in score_data:
            self.PS_cov_ode = np.array(score_data["PS_cov_ode"])
            self.PS_cov = self.PS_cov_ode
        else:
            self.PS_cov_ode = None
            self.PS_cov = None
        
        if "PS_cov_em" in score_data:
            self.PS_cov_em = np.array(score_data["PS_cov_em"])
        else:
            self.PS_cov_em = None
        
        # === 2D PS correlation statistics (from score model) ===
        # Diagonal fraction: fraction of total variance on covariance diagonal
        if "diag_frac_ode" in score_data:
            self.diag_frac_ode = float(score_data["diag_frac_ode"])
            self.diag_frac = self.diag_frac_ode
        else:
            self.diag_frac_ode = None
            self.diag_frac = None
        
        if "diag_frac_em" in score_data:
            self.diag_frac_em = float(score_data["diag_frac_em"])
        else:
            self.diag_frac_em = None
        
        # Mean absolute off-diagonal correlation
        if "mean_abs_corr_ode" in score_data:
            self.mean_abs_corr_ode = float(score_data["mean_abs_corr_ode"])
            self.mean_abs_corr = self.mean_abs_corr_ode
        else:
            self.mean_abs_corr_ode = None
            self.mean_abs_corr = None
        
        if "mean_abs_corr_em" in score_data:
            self.mean_abs_corr_em = float(score_data["mean_abs_corr_em"])
        else:
            self.mean_abs_corr_em = None
        
        # === 2D PS global error scalars (from score model) ===
        # These are headline numbers summarizing emulator accuracy
        # Default to ODE (more accurate sampler)
        if "global_median_ode_means" in score_data:
            self.PS_global_median_err_ode = float(score_data["global_median_ode_means"])
            self.PS_global_median_err = self.PS_global_median_err_ode
        else:
            self.PS_global_median_err_ode = None
            self.PS_global_median_err = None
        
        if "global_median_em_means" in score_data:
            self.PS_global_median_err_em = float(score_data["global_median_em_means"])
        else:
            self.PS_global_median_err_em = None
        
        if "global_mean_ode_means" in score_data:
            self.PS_global_mean_err_ode = float(score_data["global_mean_ode_means"])
            self.PS_global_mean_err = self.PS_global_mean_err_ode
        else:
            self.PS_global_mean_err_ode = None
            self.PS_global_mean_err = None
        
        if "global_mean_em_means" in score_data:
            self.PS_global_mean_err_em = float(score_data["global_mean_em_means"])
        else:
            self.PS_global_mean_err_em = None
        
        # Two-stage robust error (median of sample medians - most robust to outliers)
        if "twostage_median_of_sample_median_ode_means" in score_data:
            self.PS_robust_err_ode = float(score_data["twostage_median_of_sample_median_ode_means"])
            self.PS_robust_err = self.PS_robust_err_ode
        else:
            self.PS_robust_err_ode = None
            self.PS_robust_err = None
        
        if "twostage_median_of_sample_median_em_means" in score_data:
            self.PS_robust_err_em = float(score_data["twostage_median_of_sample_median_em_means"])
        else:
            self.PS_robust_err_em = None
    
    def get_ps_error(self, method: str = "ode", stat: str = "median") -> np.ndarray:
        """Get 2D PS error array for the specified sampling method and statistic.
        
        Parameters
        ----------
        method : str
            Sampling method, either "ode" (default, more accurate) or "em".
        stat : str
            Error statistic: "median" (or "med"), "mean", or "std".
        
        Returns
        -------
        np.ndarray
            Error array of shape (kperp, kpar).
        """
        # Map "median" to "med" to match attribute names
        stat_key = "med" if stat == "median" else stat
        attr_name = f"PS_{stat_key}_err_{method}"
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        # Fall back to default (ODE)
        return getattr(self, f"PS_{stat_key}_err")
    
    def get_ps_variance(self, method: str = "ode") -> np.ndarray | None:
        """Get 2D PS variance array for the specified sampling method.
        
        Parameters
        ----------
        method : str
            Sampling method, either "ode" (default) or "em".
        
        Returns
        -------
        np.ndarray | None
            Variance array of shape (kperp, kpar), or None if not available.
        """
        attr_name = f"PS_var_{method}"
        return getattr(self, attr_name, None)
    
    def get_ps_covariance(self, method: str = "ode") -> np.ndarray | None:
        """Get 2D PS covariance matrix for the specified sampling method.
        
        Parameters
        ----------
        method : str
            Sampling method, either "ode" (default) or "em".
        
        Returns
        -------
        np.ndarray | None
            Covariance matrix of shape (kperp*kpar, kperp*kpar), or None if not available.
        """
        attr_name = f"PS_cov_{method}"
        return getattr(self, attr_name, None)


def emulator_properties(emulator: str = "default") -> EmulatorProperties:
    """Return the properties of the corresponding emulator."""
    if emulator == "default":
        return DefaultEmulatorProperties()
    elif emulator == "radio_background":
        return RadioBackgroundEmulatorProperties()
    elif emulator == "mh":
        return MHEmulatorProperties()
    else:
        raise ValueError(
            "Please supply one of the following emulator names: 'default'"
            + "or 'radio_background' or 'mh'. "
            + f"{emulator} is not a valid emulator name."
        )


def get_emulator_properties(emulator: str = "default") -> EmulatorProperties:
    """Alias for compatibility with v3 helper modules."""
    return emulator_properties(emulator=emulator)
