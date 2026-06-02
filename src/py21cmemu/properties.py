"""A module defining the static properties of the Emulator.

This module provides:
- Emulator naming configuration (canonical names, aliases)
- EmulatorProperties classes for each emulator variant
- Error statistics and normalization constants
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Emulator Configuration
# ══════════════════════════════════════════════════════════════════════════════
# This is the single source of truth for emulator naming.

# Canonical emulator names (physics-based)
EMULATOR_ACG: str = "acg"  # Atomic Cooling Galaxies only (Breitman+24)
EMULATOR_RADIO: str = "radio"  # Radio background (Cang+24)
EMULATOR_MCG: str = "mcg"  # Molecular Cooling Galaxies / Mini-halos

# Default emulator
DEFAULT_EMULATOR: str = EMULATOR_MCG

# Emulator configuration: maps canonical name to metadata
EMULATOR_CONFIG: dict[str, dict[str, Any]] = {
    EMULATOR_ACG: {
        "aliases": ["v1", "default"],
        "n_params": 9,
        "paper": "Breitman+24",
        "outputs": ["Tb", "xHI", "Ts", "tau", "PS", "UVLFs"],
    },
    EMULATOR_RADIO: {
        "aliases": ["v2", "radio_background"],
        "n_params": 5,
        "paper": "Cang+24",
        "outputs": ["Tb", "xHI", "Tr", "tau", "PS"],
    },
    EMULATOR_MCG: {
        "aliases": ["v3", "mh"],
        "n_params": 11,
        "paper": "[upcoming]",
        "outputs": ["Tb", "xHI", "Ts", "tau", "PS", "PS_2D", "UVLFs"],
    },
}

# Build aliases map from config (both directions)
_EMULATOR_ALIASES: dict[str, str] = {}
for canonical, config in EMULATOR_CONFIG.items():
    _EMULATOR_ALIASES[canonical] = canonical  # Self-map
    for alias in config["aliases"]:
        _EMULATOR_ALIASES[alias] = canonical


def resolve_emulator_name(name: str) -> str:
    """Resolve an emulator name or alias to its canonical name.

    Parameters
    ----------
    name : str
        Emulator name or alias (case-insensitive).

    Returns
    -------
    str
        Canonical emulator name ('acg', 'radio', or 'mcg').

    Raises
    ------
    ValueError
        If the name is not a valid emulator or alias.

    Examples
    --------
    >>> resolve_emulator_name("v3")
    'mcg'
    >>> resolve_emulator_name("mh")
    'mcg'
    >>> resolve_emulator_name("default")
    'acg'
    """
    canonical = _EMULATOR_ALIASES.get(name.lower())
    if canonical is None:
        valid = sorted(set(_EMULATOR_ALIASES.keys()))
        raise ValueError(f"Unknown emulator '{name}'. Valid options: {valid}")
    return canonical


def get_emulator_aliases(emulator: str) -> list[str]:
    """Get all aliases for an emulator, including the canonical name.

    Parameters
    ----------
    emulator : str
        Emulator name (canonical or alias).

    Returns
    -------
    list[str]
        List of all valid names for this emulator.
    """
    canonical = resolve_emulator_name(emulator)
    return [canonical] + EMULATOR_CONFIG[canonical]["aliases"]


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
        self.astro_param_keys = [
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


class RadioEmulatorProperties(EmulatorProperties):
    """A class that contains the properties of the radio emulator.

    Note: Radio emulator does not use the renormalize() method from the base class,
    so it does not define normalized_quantities. Denormalization is handled
    directly in RadioEmulatorOutput.get_renormalized().
    """

    def __init__(self):
        here = Path(__file__).parent
        all_emulator_numbers = np.load(
            here / "models/radio/radio_background_emu_csts.npz"
        )
        self._data = all_emulator_numbers

        # Normalization constants (all in log10 space)
        self.PS_log_mean = all_emulator_numbers["logPS_mean"]
        self.PS_log_std = all_emulator_numbers["logPS_std"]
        # Backward compatibility aliases
        self.logPS_mean = self.PS_log_mean
        self.logPS_std = self.PS_log_std

        self.PS_ks = all_emulator_numbers["PS_k"]
        self.PS_zs = all_emulator_numbers["PS_z"]
        self.zs = all_emulator_numbers["redshifts"]
        self.limits = np.array([[-2, 6], [33, 45], [-5, 0], [-6, -1], [0, 10]])

        # Tb uses special normalization: -(10^(norm*std + mean)) + scale
        self.Tb_log_std = all_emulator_numbers["Tb_std"]
        self.Tb_log_mean = all_emulator_numbers["Tb_mean"]
        self.Tb_scale = all_emulator_numbers["Tb_scale"]
        # Backward compatibility aliases
        self.logTb_std = self.Tb_log_std
        self.logTb_mean = self.Tb_log_mean

        # Tr normalization (log10 space)
        self.Tr_log_mean = all_emulator_numbers["logTr_mean"]
        self.Tr_log_std = all_emulator_numbers["logTr_std"]
        # Backward compatibility aliases
        self.logTr_mean = self.Tr_log_mean
        self.logTr_std = self.Tr_log_std

        with np.load(here / "models/radio/median_test_errors.npz") as f:
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
        self.astro_param_keys = ["f_R_MINI", "L_X_MINI", "F_STAR7", "F_ESC7", "A_LW"]

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
    """Properties and error statistics for the minihalo (v3) emulator.

    This class loads constants from two separate files:

    - ``lstm_emulator_constants.npz``: LSTM model (summaries + 1D PS)
    - ``score_model_constants.npz``: 2D PS score model

    Error Statistics Conventions
    ----------------------------
    All error statistics are **Fractional Errors (FE%)** computed as::

        FE% = |true - predicted| / |true| × 100

    A floor is applied to the denominator to avoid division by small values:

    - xHI: floor at 0.01 (reionization fraction)
    - Tb: floor at 5 mK (brightness temperature)
    - Ts: no floor (spin temperature)
    - tau: no floor (optical depth)
    - UVLFs: floor at 0.01 (log10 values)
    - PS: floor at 0.01 (log10 values)

    Aggregation Methods
    -------------------
    There are three types of aggregation statistics:

    **med_err** (Median FE%):
        ``median(FE across all test samples at each (z,k) pixel)``

        Most robust to outliers. This is the primary error metric.

    **mean_err** (Mean FE%):
        ``mean(FE across all test samples at each (z,k) pixel)``

        Sensitive to outliers but preserves total error budget.

    **std_err** (Standard Deviation of FE%):
        ``std(FE across all test samples at each (z,k) pixel)``

        Characterizes the spread of errors at each pixel.

    Log vs Linear Quantities
    ------------------------
    The emulator works in log10 space for certain quantities:

    **Log10 quantities** (errors computed on log10(value)):
        - PS (power spectrum): values are log10(Δ² / mK²)
        - UVLFs: values are log10(φ / Mpc⁻³ mag⁻¹)
        - tau: values are log10(τ), converted to linear on output

    **Linear quantities** (errors computed on physical values):
        - xHI: neutral hydrogen fraction (0-1)
        - Tb: brightness temperature (mK)
        - Ts: spin temperature (K)

    Important: ``PS_med_err`` gives FE% on **log10(PS)**, not on linear PS.
    To interpret: a 5% FE on log10(PS) ≈ 5% uncertainty in the exponent,
    corresponding to ~12% uncertainty in linear PS (10^0.05 ≈ 1.12).

    Per-Pixel vs Global Statistics
    ------------------------------
    **Per-pixel arrays** (e.g., ``PS_med_err`` shape (32, 64)):
        Error at each (kperp, kpar) pixel, computed as median across test set.

    **Global scalars** (e.g., ``PS_global_median_err``):
        Single-number summary. Computed as::

            median(median(FE at each pixel across test set) across all pixels)

        This is median-of-medians: first median over test samples at each pixel,
        then median over all pixels.

    **Two-stage robust error** (e.g., ``PS_robust_err``):
        ``median_over_samples(median_over_pixels(FE within each sample))``

        First summarize each test sample to one number (median across pixels),
        then take median across samples. Most robust to outlier parameter sets.

    Covariance Statistics
    ---------------------
    The covariance matrix ``PS_cov`` characterizes error correlations:

    - Shape: (n_pixels, n_pixels) = (32×64, 32×64) = (2048, 2048)
    - Units: FE%² (covariance of fractional error)
    - ``diag_frac``: fraction of variance on diagonal (near 1 = uncorrelated)
    - ``mean_abs_corr``: mean |correlation| off-diagonal

    Attributes
    ----------
    PS_med_err : ndarray, shape (32, 64)
        Median FE% at each (kperp, kpar) pixel for 2D PS, on **log10(PS)**.
    PS_mean_err : ndarray, shape (32, 64)
        Mean FE% at each pixel for 2D PS, on log10(PS).
    PS_std_err : ndarray, shape (32, 64)
        Standard deviation of FE% at each pixel for 2D PS.
    PS_1D_med_err : ndarray, shape (32, 32)
        Median FE% at each (z, k) pixel for 1D PS, on **log10(PS)**.
    PS_var : ndarray, shape (32, 64)
        Variance of FE% at each pixel (units: FE%²).
    PS_cov : ndarray, shape (2048, 2048)
        Covariance matrix of FE% between all pixel pairs.
    diag_frac : float
        Fraction of total variance on the diagonal of covariance matrix.
    mean_abs_corr : float
        Mean absolute off-diagonal correlation coefficient.
    xHI_med_err, Tb_med_err, Ts_med_err : ndarray, shape (n_z,)
        Per-redshift median FE% for linear summaries.
    tau_med_err : float
        Median FE% for optical depth (scalar).
    UVLFs_med_err : ndarray, shape (n_Muv, n_z)
        Per-(Muv, z) median FE% for UV luminosity functions, on **log10(φ)**.
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
            here / "models/MCG/lstm_emulator_constants.npz", allow_pickle=True
        )
        self._lstm_data = lstm_data
        # Set _data for base class compatibility (normalized_quantities property)
        self._data = lstm_data

        # Load 2D PS score model constants
        score_data = np.load(
            here / "models/MCG/score_model_constants.npz", allow_pickle=True
        )
        self._score_data = score_data

        # === Parameter info (from LSTM) ===
        self.astro_param_keys = tuple(lstm_data["param_names"])
        self.parameter_labels = lstm_data["param_labels"]

        # === LSTM model properties ===
        self.redshifts = lstm_data["lstm_redshifts"][::-1]
        self.lstm_limits = lstm_data["LSTM_limits"]

        # Normalization constants
        # Linear quantities
        self.Tb_mean = float(lstm_data["Tb_bias"])
        self.Tb_std = float(lstm_data["Tb_scale"])
        self.Ts_mean = float(lstm_data["Ts_allgas_bias"])
        self.Ts_std = float(lstm_data["Ts_allgas_scale"])
        # Log10 quantities
        self.tau_log_mean = float(lstm_data["tau_bias"])
        self.tau_log_std = float(lstm_data["tau_scale"])
        self.UVLFs_log_mean = np.array(lstm_data["UVLFs_bias"])
        self.UVLFs_log_std = np.array(lstm_data["UVLFs_scale"])

        # Backward compatibility aliases
        self.tau_mean = self.tau_log_mean
        self.tau_std = self.tau_log_std
        self.UVLFs_mean = self.UVLFs_log_mean
        self.UVLFs_std = self.UVLFs_log_std
        self.UVLFs_MUVs = np.array(lstm_data["M_UV"])
        self.uv_lf_zs = np.array(lstm_data["UVLF_zs"])

        # 1D PS properties (from LSTM model)
        self.PS_1D_k = np.array(lstm_data["PS_k"])
        self.PS_1D_redshifts = np.array(lstm_data["PS_redshifts"])
        self.PS_1D_log_mean = float(lstm_data["PS_bias"])
        self.PS_1D_log_std = float(lstm_data["PS_scale"])

        # Backward compatibility aliases
        self.PS_ks = self.PS_1D_k  # Duplicate coordinate array
        self.PS_1D_bias = self.PS_1D_log_mean
        self.PS_1D_scale = self.PS_1D_log_std

        # === 2D PS score model properties ===
        self.PS_2D_limits = score_data["PS_2D_limits"]
        self.PS_redshifts = np.array(score_data["ps_redshifts"])

        # Backward compatibility aliases
        self.ps_limits = self.PS_2D_limits  # Old name
        self.PS_zs = self.PS_redshifts  # Duplicate coordinate array
        # Default redshifts for PS emulation (user can override)
        self.default_ps_redshifts = np.array(
            [
                5.5,
                6.97446005,
                7.54906604,
                7.9582024,
                9.82883407,
                10.36152691,
                10.63860385,
                16.66170964,
                19.52022545,
                24.10859229,
            ]
        )
        self.kperp = np.array(score_data["kperp"])
        self.kpar = np.array(score_data["kpar"])
        self.PS_2D_Nmodes = np.array(score_data["Nmodes"])
        self.PS_log_mean = np.array(score_data["PS_2D_bias"])
        self.PS_log_std = np.array(score_data["PS_2D_scale"])

        # Backward compatibility aliases
        self.Nmodes = self.PS_2D_Nmodes  # Old name without prefix
        self.PS_bias = self.PS_log_mean
        self.PS_scale = self.PS_log_std

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
        # Linear LF errors (error on phi = 10^log10(phi))
        if "UVLFs_lin_med_err" in lstm_data:
            self.UVLFs_lin_med_err = np.array(lstm_data["UVLFs_lin_med_err"])
            self.UVLFs_lin_mean_err = np.array(lstm_data["UVLFs_lin_mean_err"])
            self.UVLFs_lin_std_err = np.array(lstm_data["UVLFs_lin_std_err"])
        else:
            # For older constants files without linear errors, use None
            self.UVLFs_lin_med_err = None
            self.UVLFs_lin_mean_err = None
            self.UVLFs_lin_std_err = None

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
            raise KeyError(
                "PS_med_err or PS_med_err_ode required in score_model_constants.npz"
            )

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

        # === 2D PS variance and covariance of emulator errors ===
        # These statistics characterize the error distribution of the score model
        # at each (kperp, kpar) pixel. All are computed on the fractional error
        # (FE%) of log10(PS), averaged over the test set.
        #
        # Variance: shape (32 kperp, 64 kpar), units FE%^2
        #   Mean variance of FE at each pixel across the test set.
        #   sqrt(PS_var) gives typical spread of error at each pixel.
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

        # Covariance: shape (2048, 2048) = (32*64, 32*64), units FE%^2
        #   Mean covariance of FE between all pixel pairs.
        #   Pixels are flattened by raveling (32, 64) -> (2048,).
        #   To reshape: cov.reshape(32, 64, 32, 64) gives cov[i, j, k, l] =
        #   covariance between pixels (kperp_i, kpar_j) and (kperp_k, kpar_l).
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

        # === 2D PS correlation statistics ===
        # Summary statistics characterizing pixel-to-pixel error correlations.
        #
        # Diagonal fraction: fraction of total covariance variance on diagonal.
        #   diag_frac = sum(diag(cov)) / sum(cov)
        #   Values near 1 mean errors are nearly pixel-independent;
        #   low values indicate significant spatial correlations.
        if "diag_frac_ode" in score_data:
            self.PS_cov_diag_frac_ode = float(score_data["diag_frac_ode"])
            self.PS_cov_diag_frac = self.PS_cov_diag_frac_ode
        else:
            self.PS_cov_diag_frac_ode = None
            self.PS_cov_diag_frac = None

        if "diag_frac_em" in score_data:
            self.PS_cov_diag_frac_em = float(score_data["diag_frac_em"])
        else:
            self.PS_cov_diag_frac_em = None

        # Backward compatibility aliases (old names without prefix)
        self.diag_frac = self.PS_cov_diag_frac
        self.diag_frac_ode = self.PS_cov_diag_frac_ode
        self.diag_frac_em = self.PS_cov_diag_frac_em

        # Mean absolute off-diagonal correlation: mean |r_ij| for i != j
        #   where r_ij is the Pearson correlation between pixels i and j.
        #   Measures typical strength of error correlations.
        if "mean_abs_corr_ode" in score_data:
            self.PS_cov_mean_abs_corr_ode = float(score_data["mean_abs_corr_ode"])
            self.PS_cov_mean_abs_corr = self.PS_cov_mean_abs_corr_ode
        else:
            self.PS_cov_mean_abs_corr_ode = None
            self.PS_cov_mean_abs_corr = None

        if "mean_abs_corr_em" in score_data:
            self.PS_cov_mean_abs_corr_em = float(score_data["mean_abs_corr_em"])
        else:
            self.PS_cov_mean_abs_corr_em = None

        # Backward compatibility aliases (old names without prefix)
        self.mean_abs_corr = self.PS_cov_mean_abs_corr
        self.mean_abs_corr_ode = self.PS_cov_mean_abs_corr_ode
        self.mean_abs_corr_em = self.PS_cov_mean_abs_corr_em

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
            self.PS_robust_err_ode = float(
                score_data["twostage_median_of_sample_median_ode_means"]
            )
            self.PS_robust_err = self.PS_robust_err_ode
        else:
            self.PS_robust_err_ode = None
            self.PS_robust_err = None

        if "twostage_median_of_sample_median_em_means" in score_data:
            self.PS_robust_err_em = float(
                score_data["twostage_median_of_sample_median_em_means"]
            )
        else:
            self.PS_robust_err_em = None

        USER_PARAMS = {
            "HII_DIM": 200,
            "BOX_LEN": 400.0,
            "USE_INTERPOLATION_TABLES": True,
            "USE_FFTW_WISDOM": True,
            "PERTURB_ON_HIGH_RES": True,
            "OUTPUT_ALL_VEL": False,
            "USE_RELATIVE_VELOCITIES": True,
            "POWER_SPECTRUM": 5,
        }

        COSMO_PARAMS = {
            "hlittle": 0.6688,
            "OMm": 0.321,
            "OMb": 0.04952,
            "POWER_INDEX": 0.9626,
        }

        FLAG_OPTIONS = {
            "USE_MASS_DEPENDENT_ZETA": True,
            "INHOMO_RECO": True,
            "PHOTON_CONS": False,
            "EVOLVING_R_BUBBLE_MAX": False,
            "USE_TS_FLUCT": True,
            "USE_MINI_HALOS": True,
        }

        self.flag_options = FLAG_OPTIONS
        self.user_params = USER_PARAMS
        self.cosmo_params = COSMO_PARAMS

    # Coordinate array backward compatibility aliases
    @property
    def zs(self) -> np.ndarray:
        """Alias for redshifts (backward compatibility)."""
        return self.redshifts

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


def emulator_properties(emulator: str = EMULATOR_MCG) -> EmulatorProperties:
    """Return the properties of the corresponding emulator.

    Parameters
    ----------
    emulator
        Emulator name or alias. See :class:`~py21cmemu.Emulator` for the
        full list of available emulators and aliases.

    Returns
    -------
    EmulatorProperties
        Properties object for the specified emulator.
    """
    canonical = resolve_emulator_name(emulator)
    if canonical == EMULATOR_ACG:
        return DefaultEmulatorProperties()
    elif canonical == EMULATOR_RADIO:
        return RadioEmulatorProperties()
    elif canonical == EMULATOR_MCG:
        return MHEmulatorProperties()
    # Should never reach here due to resolve_emulator_name validation
    raise ValueError(f"Unknown emulator: {emulator}")


def get_emulator_properties(emulator: str = EMULATOR_MCG) -> EmulatorProperties:
    """Alias for compatibility with v3 helper modules."""
    return emulator_properties(emulator=emulator)
