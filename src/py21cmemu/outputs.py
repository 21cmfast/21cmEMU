"""Module whose functionality is to organise the emulator output.

Output Quantities and Units
---------------------------
All output quantities are returned as astropy Quantities with units attached.

Linear quantities (returned in physical units):
- Tb : Brightness temperature [mK]
- xHI : Neutral hydrogen fraction [dimensionless, 0-1]
- Ts : Spin temperature [K]
- Tr : Radio temperature [K] (radio emulator only)
- tau : Optical depth to reionization [dimensionless]

Logarithmic quantities (returned in dex units):
- UVLFs : UV luminosity functions [dex(Mpc^-3 mag^-1)] = log10(phi)
- PS : Dimensionless power spectrum delta^2 [dex(mK^2)] = log10(delta^2)
- PS_samples : Power spectrum samples [dex(mK^2)]

To convert log quantities to linear, use `.physical`:
    >>> output.PS.physical  # returns mK^2

Coordinate axes (returned with units):
- redshifts : Redshift values [dimensionless]
- kperp, kpar, k : Wavenumbers [Mpc^-1]
- Muv : UV magnitude [mag]

Error Statistics
----------------
Error statistics are available via properties like ``output.PS_err``.

**IMPORTANT**: All PS error statistics are computed on **log10(PS)**, not linear PS.

Error types:
- ``PS_err``: Median fractional error (FE%) on log10(PS) values
- ``PS_2D_err``: Median FE% on 2D PS log10 values
- ``PS_2D_var``: Variance of FE% across test set
- ``PS_2D_cov``: Covariance matrix of FE% between pixels

Fractional error is defined as::

    FE% = |true - predicted| / |true| × 100

Interpretation: A 5% FE on log10(PS) corresponds to ~12% error on linear PS,
because a 5% error in the exponent multiplies the result by 10^0.05 ≈ 1.12.

To get absolute error in log10 units::

    abs_err_dex = PS_err / 100.0 * PS.value

See ``MHEmulatorProperties`` for detailed documentation of all error statistics.

Note: The RawEmulatorOutput classes store raw numpy arrays without units.
The EmulatorOutput classes (returned by `emulator.predict()`) automatically
attach units to all quantities.
"""

from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np
from scipy.special import expit

from .properties import emulator_properties

if TYPE_CHECKING:
    from .properties import (
        DefaultEmulatorProperties,
        MHEmulatorProperties,
        RadioBackgroundEmulatorProperties,
    )

# Astropy units - required dependency
import astropy.units as u


# ──────────────────────────────────────────────────────────────────────────────
# Unit definitions
# ──────────────────────────────────────────────────────────────────────────────


def _get_units():
    """Return a dict of units for each output field.

    Uses dex(base_unit) for log quantities, which represents log10(base_unit).
    Log quantities can be converted to linear via quantity.physical.
    """
    return {
        # Linear quantities
        "Tb": u.mK,
        "xHI": u.dimensionless_unscaled,
        "Ts": u.K,
        "Tr": u.K,
        "tau": u.dimensionless_unscaled,
        # Log quantities (dex = log10 of base unit)
        # PS is delta^2 (dimensionless power spectrum), units are mK^2
        "UVLFs": u.dex(u.Mpc**-3 * u.mag**-1),
        "PS": u.dex(u.mK**2),
        "PS_2D": u.dex(u.mK**2),
        "PS_2D_samples": u.dex(u.mK**2),
        "PS_2D_std": u.dex(u.mK**2),
        # Coordinate axes
        "redshifts": u.dimensionless_unscaled,
        "PS_redshifts": u.dimensionless_unscaled,
        "UVLF_redshifts": u.dimensionless_unscaled,
        "kperp": u.Mpc**-1,
        "kpar": u.Mpc**-1,
        "k": u.Mpc**-1,
        "PS_ks": u.Mpc**-1,
        "Muv": u.mag,
        "Nmodes": u.dimensionless_unscaled,
    }


def _get_log_quantities() -> set[str]:
    """Return the set of field names that are stored in log10 space.

    These quantities have units of dex(base_unit). The stored values are
    log10(physical_value), so use .physical to convert back to linear units.
    """
    return {"UVLFs", "PS", "PS_2D", "PS_2D_samples", "PS_2D_std"}


# Known data fields that should be wrapped with units
_UNIT_FIELDS = frozenset([
    "Tb", "xHI", "Ts", "Tr", "tau", "UVLFs", "PS", 
    "PS_2D", "PS_2D_samples", "PS_2D_std",
    "redshifts", "PS_redshifts", "UVLF_redshifts", 
    "kperp", "kpar", "k", "PS_ks", "Muv", "Nmodes",
])


@dataclass(frozen=True)
class EmulatorOutput:
    """Base class for emulator output with automatic unit handling.

    All output quantities are returned as astropy Quantities with units.
    Log quantities use dex(base_unit) units. To convert to linear:
        >>> output.PS           # log10(PS) in dex(mK^2)
        >>> output.PS.physical  # PS in mK^2

    See Also
    --------
    :func:`log_quantities` : Returns set of fields stored in log10 space.
    :func:`available_units` : Returns dict of all field units.
    """

    def __getattribute__(self, name: str):
        # Get the actual value first using object's getattribute to avoid recursion
        value = object.__getattribute__(self, name)
        
        # For methods, properties, private attrs, or non-unit fields, return as-is
        if name.startswith("_") or callable(value) or name == "properties":
            return value
        
        # Check if this is a known field that should have units
        if name not in _UNIT_FIELDS:
            return value
        
        # Don't wrap None values
        if value is None:
            return None
        
        # Wrap with units
        try:
            units = _get_units()
            if name in units:
                return value * units[name]
        except KeyError:
            pass
        
        return value

    def keys(self) -> Generator[str]:
        """Yield the keys of the main data products."""
        for k in dc.fields(self):
            yield k.name

    def items(self) -> Generator[tuple[str, np.ndarray]]:
        """Yield the keys and raw values of the main data products, like a dict."""
        for k in self.keys():
            yield k, object.__getattribute__(self, k)

    def __getitem__(self, key: str):
        """Allow access to attributes as items (returns Quantity)."""
        return getattr(self, key)

    @property
    def redshifts(self) -> np.ndarray:
        """The redshifts of all quantities except the PS."""
        return self.properties.zs

    @property
    def PS_redshifts(self) -> np.ndarray:
        """The redshifts for the PS."""
        return self.properties.PS_zs

    @property
    def PS_ks(self) -> np.ndarray:
        """The ks [MPC^{-1}] for the PS."""
        return self.properties.PS_ks

    @property
    def k(self) -> np.ndarray:
        """The ks [MPC^{-1}] for the PS."""
        return self.properties.PS_ks

    def write(
        self,
        fname: str | Path,
        theta: np.ndarray | dict | None = None,
        store: list[str] | None = None,
        clobber: bool = False,
    ):
        """Write this instance's data to a file.

        This saves the output as a numpy .npz file. The output is saved as a dictionary
        with the keys being the names of the attributes of this class and the values
        being the corresponding values of those attributes. If theta is not None, then
        the inputs are also saved under the key "inputs".

        Parameters
        ----------
        fname : str or Path
            The filename to write to.
        theta : np.ndarray or dict or None, optional
            The input parameters associated with this output data to write to the file.
            If None, the inputs are not written.
        store : list of str or None, optional
            The names of the attributes to write to the file. If None, all attributes
            are written.
        clobber : bool, optional
            Whether to overwrite the file if it already exists.
        """
        if store is None:
            store = list(self.__dict__.keys())

        pth = Path(fname)
        if pth.exists() and not clobber:
            raise ValueError(f"File {pth} exists and clobber=False.")

        # Always write raw values (not Quantities)
        out = {k: object.__getattribute__(self, k) for k in store}
        if theta is not None:
            out["inputs"] = theta

        np.savez(fname, out)

    # ──────────────────────────────────────────────────────────────────────────
    # Unit information methods
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def available_units(cls) -> dict:
        """Return a dict mapping field names to their units.

        Log quantities use dex(base_unit) to represent log10(base_unit).

        Returns
        -------
        dict
            Mapping of field name -> astropy unit.
        """
        return _get_units()

    @classmethod
    def log_quantities(cls) -> set[str]:
        """Return the set of field names stored in log10 space.

        These quantities have units of dex(base_unit). To convert to
        linear units, use the `.physical` attribute:
            >>> output.PS           # dex(mK^2)
            >>> output.PS.physical  # mK^2

        Returns
        -------
        set of str
            Field names that are in log10 space: {"UVLFs", "PS", "PS_samples"}.
        """
        return _get_log_quantities()

    def is_log(self, field: str) -> bool:
        """Check if a field is stored in log10 space.

        Parameters
        ----------
        field : str
            The field name to check.

        Returns
        -------
        bool
            True if the field is in log10 space.
        """
        return field in _get_log_quantities()

    def unit(self, field: str):
        """Get the unit for a specific field.

        Parameters
        ----------
        field : str
            The field name.

        Returns
        -------
        astropy.units.Unit
            The unit for this field.

        Raises
        ------
        KeyError
            If the field has no defined unit.
        """
        units = _get_units()
        if field not in units:
            raise KeyError(f"No unit defined for field '{field}'")
        return units[field]


@dataclass(frozen=True)
class DefaultEmulatorOutput(EmulatorOutput):
    """A simple class that makes it easier to access the corrected emulator output."""

    Tb: np.ndarray
    xHI: np.ndarray
    Ts: np.ndarray
    PS: np.ndarray
    tau: np.ndarray
    UVLFs: np.ndarray

    properties = emulator_properties(emulator="acg")

    @property
    def Muv(self) -> np.ndarray:
        """The Muv-values of the UVLFs."""
        # Crop the M_UV to -20 to -10 range
        m = np.logical_and(
            self.properties.UVLFs_MUVs <= -10, self.properties.UVLFs_MUVs >= -20
        )
        return self.properties.UVLFs_MUVs[m]

    @property
    def UVLF_redshifts(self) -> np.ndarray:
        """The redshifts of the UVLFs."""
        return self.properties.uv_lf_zs

    def squeeze(self):
        """Return a new EmulatorOutput with all dimensions of length 1 removed."""
        return DefaultEmulatorOutput(**{k: np.squeeze(v) for k, v in self.items()})


@dataclass(frozen=True)
class RadioEmulatorOutput(EmulatorOutput):
    """A simple class that makes it easier to access the corrected emulator output."""

    Tb: np.ndarray
    xHI: np.ndarray
    Tr: np.ndarray
    PS: np.ndarray
    tau: np.ndarray

    properties = emulator_properties(emulator="radio")

    def squeeze(self):
        """Return a new EmulatorOutput with all dimensions of length 1 removed."""
        return RadioEmulatorOutput(**{k: np.squeeze(v) for k, v in self.items()})


@dataclass(frozen=True)
class RawEmulatorOutput:
    """A super data-class that makes it easier to access the raw emulator output."""

    output: np.ndarray

    @property
    def nz(self) -> int:
        """Number of redshifts in the output."""
        return np.array(self.properties.zs).shape[0]

    @property
    def nparams(self) -> int:
        """Number of sets of parameters in the output."""
        return self.output.shape[0]

    @property
    def PS_nz(self) -> int:
        """Number of redshifts in the output."""
        return np.array(self.properties.PS_zs).shape[0]

    @property
    def PS_nk(self) -> int:
        """Number of redshifts in the output."""
        return np.array(self.properties.PS_ks).shape[0]


@dataclass(frozen=True)
class DefaultRawEmulatorOutput(RawEmulatorOutput):
    """A simple sub data-class that makes it easier to access the raw emulator output.

    Parameters
    ----------
    output : np.ndarray
        The raw output array from the emulator.
    """

    output: np.ndarray
    properties = emulator_properties(emulator="acg")

    @property
    def Tb(self) -> np.ndarray:
        """Mean 21cm brightness temperature in mK as a function of redshift."""
        return self.output[:, : self.nz]

    @property
    def xHI(self) -> np.ndarray:
        """Neutral fraction as a function of redshift."""
        return self.output[:, self.nz : 2 * self.nz]

    @property
    def Ts(self) -> np.ndarray:
        """Mean spin temperature in K as a function of redshift."""
        return self.output[:, 2 * self.nz : 3 * self.nz]

    @property
    def reshift_where_Ts_becomes_defined(self) -> np.ndarray:
        """The redshift at which the spin temperature becomes well-defined."""
        return self.output[:, 3 * self.nz]

    @property
    def PS(self) -> np.ndarray:
        """The power spectrum in mK^2 as a function of redshift and k."""
        return self.output[:, 3 * self.nz + 1 : 3 * self.nz + 1 + 60 * 12].reshape(
            (-1, 60, 12)
        )

    @property
    def tau(self) -> np.ndarray:
        """The optical depth of reionization."""
        return self.output[:, self.nz * 3 + 1 + 60 * 12]

    @property
    def UVLFs(self) -> np.ndarray:
        """The UV luminosity functions as a function of z and Muv."""
        full_UVLFs = self.output[:, self.nz * 3 + 1 + 60 * 12 + 1 :].reshape(
            (-1, len(self.properties.uv_lf_zs), len(self.properties.UVLFs_MUVs))
        )
        # Crop M_UVs to allowed range of [-20,-10]
        m = np.logical_and(
            self.properties.UVLFs_MUVs <= -10, self.properties.UVLFs_MUVs >= -20
        )
        return full_UVLFs[..., m]

    def renormalize(self, name: str):
        """Renormalize a normalized quantity.

        This ajudsts the quantity (as it exists in this class) back to its native
        range by adding the emulator data mean and multiplying by the emulator data
        standard deviation.
        """
        if name not in self.properties.normalized_quantities:
            raise ValueError(
                f"Cannot renormalize {name}. It is not a normalized quantity."
            )
        return getattr(self.properties, f"{name}_mean") + getattr(
            self.properties, f"{name}_std"
        ) * getattr(self, name)

    def get_renormalized(self) -> EmulatorOutput:
        """Get the output with normalized quantities re-normalized.

        Returns
        -------
        EmulatorOutput
            The emulator output with normalized quantities re-normalized back to
            physical units. Nothing is in log except UV LFs.
        """
        # Restore dimensions
        # Renormalize stuff that needs renormalization
        renorm = {k: self.renormalize(k) for k in self.properties.normalized_quantities}

        other = {
            k.name: getattr(self, k.name)
            for k in dc.fields(DefaultEmulatorOutput)
            if k.name not in renorm
        }

        out = {**renorm, **other}

        # Set the xHI < z(Ts undefined) to 0
        # For Ts, set it to NaN
        for i in range(self.nparams):
            zbin = np.argmin(
                abs(self.properties.zs - self.reshift_where_Ts_becomes_defined[i])
            )
            if out["xHI"][i, zbin] < 1e-1:
                out["xHI"][i, :zbin] = 0.0
            out["Ts"][i, :zbin] = np.nan

        # Undo log10 on some quantities
        out["PS"] = 10 ** out["PS"]
        out["Ts"] = 10 ** out["Ts"]
        out["tau"] = 10 ** out["tau"]

        return DefaultEmulatorOutput(**out).squeeze()


class RadioRawEmulatorOutput(RawEmulatorOutput):
    """A simple sub data-class that makes it easier to access the raw emulator output.

    Parameters
    ----------
    output : np.ndarray
        The raw output array from the emulator.
    """

    output: np.ndarray

    properties = emulator_properties(emulator="radio")

    @property
    def Tb(self) -> np.ndarray:
        """Mean 21cm brightness temperature in mK as a function of redshift."""
        return self.output[:, : self.nz]

    @property
    def Tr(self) -> np.ndarray:
        """Radio temperature in K as a function of redshift."""
        return self.output[:, self.nz : 2 * self.nz]

    @property
    def xHI(self) -> np.ndarray:
        """Neutral fraction as a function of redshift."""
        return self.output[:, 2 * self.nz : 3 * self.nz]

    @property
    def PS(self) -> np.ndarray:
        r""":math:`\Delta^{2}_{21} [\rm{mK}^2]` as a function of redshift and k."""
        return self.output[:, 3 * self.nz : -1].reshape(
            (self.output.shape[0], self.PS_nz, self.PS_nk)
        )

    @property
    def tau(self) -> np.ndarray:
        """The optical depth of reionization."""
        return self.output[:, -1]

    def get_renormalized(self) -> EmulatorOutput:
        """Get the output with normalized quantities re-normalized.

        Returns
        -------
        EmulatorOutput
            The emulator output with normalized quantities re-normalized back to
            physical units.
        """
        # Restore dimensions
        # Renormalize stuff that needs renormalization
        out = {}

        out["Tb"] = (
            -(
                10
                ** ((self.Tb * self.properties.logTb_std) + self.properties.logTb_mean)
            )
            + self.properties.Tb_scale
        )
        out["Tr"] = 10 ** (
            (self.Tr * self.properties.logTr_std) + self.properties.logTr_mean
        )
        out["PS"] = 10 ** (
            (self.PS * self.properties.logPS_std) + self.properties.logPS_mean
        )

        # zs axis was flipped.
        # I'll flip the global summaries instead to keep the zs in increasing order.
        out["xHI"] = self.xHI[:, ::-1]
        out["Tb"] = out["Tb"][:, ::-1]
        out["Tr"] = out["Tr"][:, ::-1]

        out["tau"] = 10 ** (self.tau)

        other = {
            k.name: getattr(self, k.name)
            for k in dc.fields(RadioEmulatorOutput)
            if k.name not in out.keys()
        }

        return RadioEmulatorOutput(**{**out, **other}).squeeze()


@dataclass(frozen=True)
class MHEmulatorOutput(EmulatorOutput):
    """A simple class that makes it easier to access v3 emulator output.
    
    Attributes
    ----------
    Tb : np.ndarray
        Brightness temperature in mK, shape (batch, 32 redshifts).
    xHI : np.ndarray
        Neutral hydrogen fraction, shape (batch, 32 redshifts).
    Ts : np.ndarray
        Spin temperature in K, shape (batch, 32 redshifts).
    tau : np.ndarray
        Optical depth to reionization, shape (batch,).
    UVLFs : np.ndarray
        UV luminosity functions log10(phi / Mpc^-3 mag^-1),
        shape (batch, n_Muv, n_redshifts).
    PS : np.ndarray
        1D power spectrum log10(PS / mK^2) from LSTM model,
        shape (batch, 32 redshifts, 32 k).
    PS_2D : np.ndarray | None
        2D power spectrum log10(PS / mK^2) median over realizations,
        shape (batch, n_redshifts, 32 kperp, 64 kpar). Only when emulate_2d_ps=True.
    PS_2D_samples : np.ndarray | None
        2D power spectrum samples log10(PS / mK^2),
        shape (batch, n_redshifts, n_samples, 32 kperp, 64 kpar).
        Only when emulate_2d_ps=True.
    PS_2D_std : np.ndarray | None
        Std of 2D power spectrum over realizations,
        shape (batch, n_redshifts, 32 kperp, 64 kpar). Only when emulate_2d_ps=True.
    _ps_redshifts : np.ndarray | None
        Redshifts for the 2D PS (if different from default).
    """

    Tb: np.ndarray
    xHI: np.ndarray
    Ts: np.ndarray
    tau: np.ndarray
    UVLFs: np.ndarray
    PS: np.ndarray
    PS_2D: np.ndarray | None
    PS_2D_samples: np.ndarray | None
    PS_2D_std: np.ndarray | None
    _ps_redshifts: np.ndarray | None

    properties = emulator_properties(emulator="mcg")

    @property
    def PS_redshifts(self) -> np.ndarray | None:
        """Redshifts for the 2D PS (only when emulate_2d_ps=True)."""
        return self._ps_redshifts

    @property
    def PS_1D_k(self) -> np.ndarray:
        """Wavenumbers for 1D PS in h/Mpc."""
        return self.properties.PS_1D_k

    @property
    def PS_1D_redshifts(self) -> np.ndarray:
        """Redshifts for 1D PS."""
        return self.properties.PS_1D_redshifts

    @property
    def kperp(self) -> np.ndarray:
        """Perpendicular wavenumbers for 2D PS in h/Mpc."""
        return self.properties.kperp

    @property
    def kpar(self) -> np.ndarray:
        """Parallel wavenumbers for 2D PS in h/Mpc."""
        return self.properties.kpar

    @property
    def Nmodes(self) -> np.ndarray:
        """Number of modes for 2D PS, shape (32 kperp, 64 kpar)."""
        return self.properties.Nmodes

    @property
    def PS_err(self) -> np.ndarray:
        """Median fractional error (%) on 1D PS **log10** values.
        
        This is the median FE% computed on log10(PS) across the test set at each
        (z, k) pixel. The error is on the **log10** value, not the linear PS.
        
        Interpretation
        --------------
        If PS_err[i, j] = 5%, this means the typical error on log10(PS) at that
        pixel is 5% of the true log10(PS) value. To convert to linear error:
        
        - A 5% error on log10(PS) ≈ 5% uncertainty in the exponent
        - This corresponds to ~12% error in linear PS (since 10^0.05 ≈ 1.12)
        
        Computing Absolute Error
        ------------------------
        To get the absolute error in log10 units at each pixel::
        
            abs_err_log10 = PS_err / 100.0 * PS.value  # in dex
        
        Shape: (32 redshifts, 32 k).
        
        Returns
        -------
        np.ndarray
            Median FE% array with shape (32, 32).
        
        See Also
        --------
        PS_2D_err : Equivalent for 2D power spectrum.
        MHEmulatorProperties : Full documentation of error conventions.
        """
        return self.properties.PS_1D_med_err
    
    @property
    def PS_2D_err(self) -> np.ndarray | None:
        """Median fractional error (%) on 2D PS **log10** values.
        
        This is the median FE% computed on log10(PS) across the test set at each
        (kperp, kpar) pixel. The error is on the **log10** value.
        
        Note: This property returns the ODE sampler error (default, more accurate).
        For EM sampler errors, use ``properties.PS_med_err_em``.
        
        Interpretation
        --------------
        See ``PS_err`` for detailed interpretation. A 5% FE on log10(PS)
        corresponds to approximately:
        
        - 5% uncertainty in the log10 exponent
        - ~12% uncertainty in linear PS
        
        Computing Absolute Error
        ------------------------
        To get the absolute error in log10 units at each pixel::
        
            abs_err_log10 = PS_2D_err / 100.0 * PS_2D.value  # in dex
        
        Shape: (32 kperp, 64 kpar).
        
        Returns
        -------
        np.ndarray | None
            Median FE% array shape (32, 64), or None if emulate_2d_ps=False.
        
        See Also
        --------
        PS_err : Equivalent for 1D power spectrum.
        PS_2D_var : Variance of the error distribution.
        PS_2D_cov : Full covariance matrix of errors.
        """
        if self.PS_2D is None:
            return None
        return self.properties.PS_med_err
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2D PS Error Distribution Statistics
    # ═══════════════════════════════════════════════════════════════════════════
    # These statistics characterize the emulator error distribution for 2D PS.
    # All are computed on the fractional error (FE%) of log10(PS).
    #
    # IMPORTANT: These statistics describe errors on LOG10 values, not linear PS.
    # A 5% FE on log10(PS) ≈ 12% error on linear PS (since 10^0.05 ≈ 1.12).
    
    @property
    def PS_2D_var(self) -> np.ndarray | None:
        """Variance of 2D PS emulator error across test set.
        
        This is the mean variance of the fractional error (FE%) on log10(PS) across
        the test set, computed at each (kperp, kpar) pixel. Uses the ODE sampler
        by default (more accurate than EM).
        
        Shape: (32 kperp, 64 kpar)
        Units: FE%^2 (variance of percentage error on log10(PS))
        
        Returns
        -------
        np.ndarray | None
            Variance array, or None if emulate_2d_ps=False or not available.
        """
        if self.PS_2D is None:
            return None
        return self.properties.PS_var
    
    @property
    def PS_2D_cov(self) -> np.ndarray | None:
        """Covariance matrix of 2D PS emulator error across test set.
        
        This is the mean covariance of the fractional error (FE%) on log10(PS)
        between all pairs of (kperp, kpar) pixels. Uses ODE sampler by default.
        
        The matrix is flattened: pixels are ordered by raveling (32, 64) -> (2048,).
        To reshape to 4D: cov.reshape(32, 64, 32, 64) gives cov[i, j, k, l] =
        covariance between pixels (i, j) and (k, l).
        
        Shape: (2048, 2048) = (32*64, 32*64)
        Units: FE%^2 (covariance of percentage error on log10(PS))
        
        Returns
        -------
        np.ndarray | None
            Covariance matrix, or None if emulate_2d_ps=False or not available.
        """
        if self.PS_2D is None:
            return None
        return self.properties.PS_cov
    
    def PS_2D_cov_4d(self) -> np.ndarray | None:
        """Covariance matrix reshaped to (32, 64, 32, 64).
        
        Convenience method that reshapes the flat (2048, 2048) covariance matrix
        to 4D for easier indexing: cov_4d[i, j, k, l] gives the covariance
        between pixel (kperp_i, kpar_j) and pixel (kperp_k, kpar_l).
        
        Returns
        -------
        np.ndarray | None
            4D covariance array, or None if not available.
        """
        cov = self.PS_2D_cov
        if cov is None:
            return None
        return cov.reshape(32, 64, 32, 64)
    
    @property
    def PS_2D_corr_diag_frac(self) -> float | None:
        """Fraction of total variance on the covariance matrix diagonal.
        
        This measures how much of the total error variance is captured by the
        diagonal (i.e., pixel-wise independent errors). A value close to 1 means
        errors are nearly uncorrelated between pixels; low values indicate
        significant correlations.
        
        Returns
        -------
        float | None
            Diagonal fraction (0 to 1), or None if not available.
        """
        if self.PS_2D is None:
            return None
        return self.properties.diag_frac
    
    @property
    def PS_2D_mean_abs_corr(self) -> float | None:
        """Mean absolute off-diagonal correlation.
        
        This is the mean |r_ij| over all pairs of pixels i != j, where r_ij
        is the Pearson correlation coefficient. Measures the typical strength
        of correlations between pixel errors.
        
        Returns
        -------
        float | None
            Mean absolute correlation (0 to 1), or None if not available.
        """
        if self.PS_2D is None:
            return None
        return self.properties.mean_abs_corr

    @property
    def Muv(self) -> np.ndarray:
        m = np.logical_and(
            self.properties.UVLFs_MUVs <= -10, self.properties.UVLFs_MUVs >= -20
        )
        return self.properties.UVLFs_MUVs[m]

    @property
    def UVLF_redshifts(self) -> np.ndarray:
        return self.properties.uv_lf_zs

    @property
    def redshifts(self) -> np.ndarray:
        return self.properties.redshifts[::-1]

    def squeeze(self):
        return MHEmulatorOutput(**{k: np.squeeze(v) for k, v in self.items()})


@dataclass(frozen=True)
class MHRawEmulatorOutput(RawEmulatorOutput):
    """A data class that wraps raw v3 emulator outputs."""

    output: tuple
    properties = emulator_properties(emulator="mcg")

    @property
    def Tb(self) -> np.ndarray:
        out = self.output[1]
        return out.cpu().detach().numpy() if hasattr(out, "cpu") else out

    @property
    def xHI(self) -> np.ndarray:
        out = self.output[0]
        return out.cpu().detach().numpy() if hasattr(out, "cpu") else out

    @property
    def Ts(self) -> np.ndarray:
        out = self.output[2]
        return out.cpu().detach().numpy() if hasattr(out, "cpu") else out

    @property
    def tau(self) -> np.ndarray:
        out = self.output[4]
        return out.cpu().detach().numpy() if hasattr(out, "cpu") else out

    @property
    def UVLFs(self) -> np.ndarray:
        full_UVLFs = self.output[3]
        m = np.logical_and(
            self.properties.UVLFs_MUVs <= -10, self.properties.UVLFs_MUVs >= -20
        )
        if hasattr(full_UVLFs, "cpu"):
            return full_UVLFs.cpu().detach().numpy()[:, m, :]
        return full_UVLFs[:, m, :]

    @property
    def PS(self) -> np.ndarray:
        """1D PS from LSTM model (normalized, needs denormalization)."""
        out = self.output[5]
        if out is None:
            return None
        return out.cpu().detach().numpy() if hasattr(out, "cpu") else out

    @property
    def PS_2D_samples(self) -> np.ndarray | None:
        """2D PS samples from score model (linear, needs log10)."""
        return self.output[6] if len(self.output) > 6 else None

    @property
    def PS_redshifts(self) -> np.ndarray | None:
        """Redshifts for 2D PS."""
        return self.output[7] if len(self.output) > 7 else None

    @property
    def _ps_redshifts(self) -> np.ndarray | None:
        """Alias for compatibility with MHEmulatorOutput field name."""
        return self.output[7] if len(self.output) > 7 else None
    
    @property
    def PS_2D(self) -> np.ndarray | None:
        """Placeholder - computed in get_renormalized."""
        return None
    
    @property
    def PS_2D_std(self) -> np.ndarray | None:
        """Placeholder - computed in get_renormalized."""
        return None

    def renormalize(self, name: str):
        if name not in self.properties.normalized_quantities:
            raise ValueError(
                f"Cannot renormalize {name}. It is not a normalized quantity."
            )
        return getattr(self.properties, f"{name}_mean") + getattr(
            self.properties, f"{name}_std"
        ) * getattr(self, name)

    def get_renormalized(self) -> EmulatorOutput:
        renorm = {k: self.renormalize(k) for k in self.properties.normalized_quantities}

        other = {
            k.name: getattr(self, k.name)
            for k in dc.fields(MHEmulatorOutput)
            if k.name not in renorm
        }

        out = {**renorm, **other}
        # Ts has shape (B, N_z, 2) where ch0=value, ch1=validity logit
        # Apply sigmoid to logit; where sigmoid > 0.5, Ts is valid; else NaN
        ts_raw = out["Ts"]
        if ts_raw.ndim > 2:
            ts_val = ts_raw[..., 0]
            ts_logit = ts_raw[..., 1]
            # Use scipy's numerically stable sigmoid to avoid overflow
            validity = expit(ts_logit)
            ts_val = np.where(validity > 0.5, ts_val, np.nan)
        else:
            ts_val = ts_raw
        out["Ts"] = 10 ** ts_val.squeeze()[..., ::-1]
        out["xHI"] = out["xHI"].squeeze()[..., ::-1]
        out["Tb"] = out["Tb"].squeeze()[..., ::-1]
        out["tau"] = 10 ** out["tau"]
        out["UVLFs"] = np.swapaxes(out["UVLFs"], 2, 1)
        
        # 1D PS from LSTM: denormalize to get log10(PS)
        # Formula: log10(PS) = PS_norm * PS_1D_scale + PS_1D_bias
        ps_1d_norm = out.get("PS")
        if ps_1d_norm is not None:
            out["PS"] = (
                ps_1d_norm.squeeze() * self.properties.PS_1D_scale 
                + self.properties.PS_1D_bias
            )
        
        # 2D PS samples from score model: convert linear -> log10
        ps_2d_samples_lin = out.get("PS_2D_samples")
        if ps_2d_samples_lin is not None:
            out["PS_2D_samples"] = np.log10(ps_2d_samples_lin)
            # Compute median and std over realizations (axis=2)
            out["PS_2D"] = np.median(out["PS_2D_samples"], axis=2)
            out["PS_2D_std"] = np.std(out["PS_2D_samples"], axis=2)
        else:
            out["PS_2D"] = None
            out["PS_2D_samples"] = None
            out["PS_2D_std"] = None

        return MHEmulatorOutput(
            Tb=out["Tb"],
            xHI=out["xHI"],
            Ts=out["Ts"],
            tau=out["tau"],
            UVLFs=out["UVLFs"],
            PS=out["PS"],
            PS_2D=out["PS_2D"],
            PS_2D_samples=out["PS_2D_samples"],
            PS_2D_std=out["PS_2D_std"],
            _ps_redshifts=out.get("_ps_redshifts"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Emulator Error Classes
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EmulatorErrors:
    """Base class for emulator error statistics.
    
    This is the abstract base class for all emulator-specific error classes.
    Each emulator version (ACG/v1, Radio/v2, MH/v3) has its own error class
    that inherits from this base.
    
    Error Terminology
    -----------------
    - **Fractional Error (FE%)**: The percentage error relative to the true value,
      computed as ``100 * |predicted - true| / |true|``. This is what the ACG
      and Radio emulators store directly.
    - **Absolute Error**: The error in physical units, computed as 
      ``FE% / 100 * |output_value|``. The MH emulator computes these from FE%.
    
    Error Aggregation
    -----------------
    Errors are aggregated from a held-out test set of ~300-1000 simulations:
    
    1. For each test sample, compute the pointwise error at each (z, k) or (z, M) bin
    2. Aggregate across test samples using the **median** (robust to outliers)
    3. Store the resulting error array with shape matching the output quantity
    
    The median is chosen over the mean because some parameter combinations
    can produce pathological errors that would skew mean statistics.
    
    Dict-like Interface
    -------------------
    All error classes support dict-like access for compatibility::
    
        errors = emu.predict(params)[2]
        errors["PS_err"]        # Dict-style access
        errors.PS_err           # Attribute access
        "PS_err" in errors      # Key checking
        list(errors.keys())     # List available errors
        for key, val in errors.items(): ...  # Iteration
    
    See Also
    --------
    ACGEmulatorErrors : Errors for ACG/Default (v1) emulator.
    RadioEmulatorErrors : Errors for Radio (v2) emulator.
    MHEmulatorErrors : Errors for MH/MCG (v3) emulator.
    """
    
    def available_errors(self) -> dict[str, str]:
        """Return dict of available error fields and their descriptions."""
        return {}
    
    def summary(self) -> str:
        """Return a human-readable summary of error statistics."""
        lines = ["Emulator Error Statistics", "=" * 40]
        for name, desc in self.available_errors().items():
            val = getattr(self, name, None)
            if val is None:
                lines.append(f"{name}: N/A ({desc})")
            elif hasattr(val, 'shape'):
                med = np.nanmedian(val.value if hasattr(val, 'value') else val)
                lines.append(f"{name}: median={med:.2f} ({desc})")
            else:
                lines.append(f"{name}: {val} ({desc})")
        return "\n".join(lines)


@dataclass(frozen=True)
class MHEmulatorErrors(EmulatorErrors):
    """Error statistics for the MH (v3) emulator with proper astropy units.
    
    This class provides **absolute errors** computed from the test set's fractional
    errors (FE%) applied to the emulator output values. Unlike the ACG and Radio
    emulators which store raw FE%, the MH emulator computes output-dependent
    absolute errors in physical units.
    
    Error Computation
    -----------------
    Absolute errors are computed as::
    
        absolute_error = FE% / 100 * |output_value|
    
    where FE% is the **median** fractional error from a held-out test set of
    ~1000 simulations. The median is used (rather than mean) for robustness to
    outliers from pathological parameter combinations.
    
    Log vs Linear Errors
    --------------------
    **IMPORTANT**: Different quantities have errors computed in different spaces:
    
    - **Log quantities** (PS, UVLFs): Errors are in **dex** (log10 units).
      A ``PS_err`` of 0.05 dex means log10(PS) is off by ~0.05, corresponding
      to a multiplicative factor of 10^0.05 ≈ 1.12 (12%) in linear PS.
    
    - **Linear quantities** (Tb, xHI, Ts, tau): Errors are in physical units.
      A ``Tb_err`` of 2 mK means the brightness temperature is off by ~2 mK.
    
    2D Power Spectrum Errors
    ------------------------
    The MH emulator uses a score-based diffusion model for 2D PS which has
    additional error statistics accessible via the ``properties`` attribute:
    
    - **Variance**: Per-bin variance from test set residuals
    - **Covariance**: Full covariance matrix between (kperp, kpar) bins
    - **Correlation**: Normalized correlation matrix
    
    These can be accessed via helper methods::
    
        errors.get_ps_variance()           # Shape (32, 64) variance array
        errors.get_ps_covariance()         # Shape (2048, 2048) covariance matrix
        errors.ps_diagonal_fraction        # Fraction of variance on diagonal
        errors.ps_mean_abs_correlation     # Mean |off-diagonal correlation|
    
    Two sampling methods are available: 'em' (Euler-Maruyama) and 'ode' (ODE solver),
    with different error characteristics accessible via the ``method`` parameter.
    
    Attributes
    ----------
    PS_err : Quantity
        Absolute error on 1D PS in dex(mK²). Shape (n_z, n_k).
        Interpretation: log10(PS_true) ≈ log10(PS_pred) ± PS_err
    Tb_err : Quantity
        Absolute error on brightness temperature Tb in mK. Shape (n_z,).
        Interpretation: Tb_true ≈ Tb_pred ± Tb_err
    xHI_err : Quantity
        Absolute error on neutral fraction xHI (dimensionless). Shape (n_z,).
        Range typically 0-1, error is additive.
    Ts_err : Quantity
        Absolute error on spin temperature Ts in K. Shape (n_z,).
    tau_err : Quantity
        Absolute error on optical depth tau (dimensionless). Scalar.
    UVLFs_err : Quantity
        Absolute error on linear UV luminosity function φ in Mpc⁻³ mag⁻¹.
        Shape (n_z, n_mag). For plotting linear LF with errorbars.
    UVLFs_logerr : Quantity
        Absolute error on log10(φ) in dex(Mpc⁻³ mag⁻¹). Shape (n_z, n_mag).
        Preferred for log-scale LF plots.
    
    Examples
    --------
    Basic usage with prediction:
    
    >>> emu = Emulator(emulator="mcg")
    >>> theta, output, errors = emu.predict(params)
    >>> print(errors.PS_err.unit)  # dex(mK2)
    >>> print(errors.summary())    # Human-readable summary
    
    Accessing 2D PS error statistics:
    
    >>> errors.get_ps_variance()            # Per-bin variance
    >>> errors.get_ps_covariance()          # Full covariance matrix
    >>> print(f"Diagonal fraction: {errors.ps_diagonal_fraction:.2%}")
    >>> print(f"Mean |correlation|: {errors.ps_mean_abs_correlation:.3f}")
    
    See Also
    --------
    MHEmulatorProperties : Raw error statistics from test set.
    MHEmulatorOutput : The output dataclass these errors correspond to.
    """
    
    # Required fields
    PS_err: u.Quantity
    Tb_err: u.Quantity
    xHI_err: u.Quantity
    Ts_err: u.Quantity
    tau_err: u.Quantity
    UVLFs_err: u.Quantity
    UVLFs_logerr: u.Quantity
    
    # Internal reference to emulator properties for advanced access
    _properties: object = dc.field(default=None, repr=False)
    _ps_sampling_method: str = dc.field(default="em", repr=False)
    
    def available_errors(self) -> dict[str, str]:
        """Return dict of available error fields and their descriptions."""
        return {
            "PS_err": "Absolute error on log10(PS)",
            "Tb_err": "Absolute error on brightness temperature",
            "xHI_err": "Absolute error on neutral fraction",
            "Ts_err": "Absolute error on spin temperature",
            "tau_err": "Absolute error on optical depth",
            "UVLFs_err": "Absolute error on linear LF",
            "UVLFs_logerr": "Absolute error on log10(LF)",
        }
    
    @property
    def properties(self):
        """Access the underlying emulator properties for advanced error statistics."""
        return self._properties
    
    @classmethod
    def from_output(
        cls,
        output: MHEmulatorOutput,
        properties: "MHEmulatorProperties",
        ps_sampling_method: str = "em",
    ) -> "MHEmulatorErrors":
        """Construct error statistics from emulator output.
        
        Parameters
        ----------
        output : MHEmulatorOutput
            The emulator output to compute errors for.
        properties : MHEmulatorProperties
            The emulator properties containing FE% arrays.
        ps_sampling_method : str, optional
            Sampling method for 2D PS: 'em' (default) or 'ode'.
        
        Returns
        -------
        MHEmulatorErrors
            Error statistics with proper units attached.
        """
        # Helper to get raw values (strip Quantity if present)
        def _raw(x):
            if x is None:
                return None
            return x.value if hasattr(x, 'value') else x
        
        # Get raw output values
        emu_PS = _raw(output.PS)
        emu_Tb = _raw(output.Tb)
        emu_xHI = _raw(output.xHI)
        emu_Ts = _raw(output.Ts)
        emu_UVLFs = _raw(output.UVLFs)
        emu_tau = _raw(output.tau)
        
        # Select method-specific PS FE% if available
        if ps_sampling_method == "ode":
            ps_fe = properties.PS_med_err_ode
        else:
            ps_fe = properties.PS_med_err_em
        
        # Get magnitude mask for UVLFs (M_UV in [-20, -10])
        m = np.logical_and(
            properties.UVLFs_MUVs <= -10, 
            properties.UVLFs_MUVs >= -20
        )
        
        # Compute absolute errors from FE%
        # PS: handle shape mismatch between 1D PS output and 2D PS error
        if emu_PS is not None:
            try:
                # 1D PS error: use 1D PS-specific errors
                ps_err = properties.PS_1D_med_err / 100.0 * np.abs(emu_PS)
            except (ValueError, AttributeError):
                # Fallback to scalar median
                ps_err = np.nanmedian(ps_fe) / 100.0 * np.abs(emu_PS)
        else:
            ps_err = np.nan
        
        # Linear quantity errors
        tb_err = properties.Tb_med_err / 100.0 * np.abs(emu_Tb)
        xhi_err = properties.xHI_med_err / 100.0 * emu_xHI
        ts_err = properties.Ts_med_err / 100.0 * emu_Ts
        tau_err = properties.tau_med_err / 100.0 * emu_tau
        
        # UVLF errors
        # FE% arrays have shape (n_mag, n_z), need to swap to match output shape
        uvlf_log_fe = np.swapaxes(properties.UVLFs_med_logerr[m], 1, 0)
        uvlf_logerr = uvlf_log_fe / 100.0 * np.abs(emu_UVLFs)
        
        # Linear LF error
        if properties.UVLFs_lin_med_err is not None:
            uvlf_lin_fe = np.swapaxes(properties.UVLFs_lin_med_err[m], 1, 0)
            uvlf_linerr = uvlf_lin_fe / 100.0 * (10 ** emu_UVLFs)
        else:
            # Fallback from log error
            uvlf_linerr = uvlf_log_fe / 100.0 * (10 ** emu_UVLFs)
        
        return cls(
            PS_err=ps_err * u.dex(u.mK**2),
            Tb_err=tb_err * u.mK,
            xHI_err=xhi_err * u.dimensionless_unscaled,
            Ts_err=ts_err * u.K,
            tau_err=tau_err * u.dimensionless_unscaled,
            UVLFs_err=uvlf_linerr * (u.Mpc**-3 * u.mag**-1),
            UVLFs_logerr=uvlf_logerr * u.dex(u.Mpc**-3 * u.mag**-1),
            _properties=properties,
            _ps_sampling_method=ps_sampling_method,
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Advanced Error Statistics (accessed via properties)
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_ps_fractional_error(
        self, 
        method: str | None = None, 
        stat: str = "median"
    ) -> np.ndarray:
        """Get 2D PS fractional error (FE%) for specified method and statistic.
        
        Parameters
        ----------
        method : str, optional
            Sampling method: 'em' or 'ode'. Default uses the method from output.
        stat : str, optional
            Error statistic: 'median' (default), 'mean', or 'std'.
        
        Returns
        -------
        np.ndarray
            FE% array with shape (32 kperp, 64 kpar) for 2D PS.
        """
        if self._properties is None:
            raise ValueError("Properties not available")
        method = method or self._ps_sampling_method
        return self._properties.get_ps_error(method=method, stat=stat)
    
    def get_ps_variance(self, method: str | None = None) -> np.ndarray | None:
        """Get 2D PS error variance (FE%²) for specified sampling method.
        
        Parameters
        ----------
        method : str, optional
            Sampling method: 'em' or 'ode'. Default uses the method from output.
        
        Returns
        -------
        np.ndarray | None
            Variance array shape (32, 64), or None if not available.
        """
        if self._properties is None:
            return None
        method = method or self._ps_sampling_method
        return self._properties.get_ps_variance(method=method)
    
    def get_ps_covariance(self, method: str | None = None) -> np.ndarray | None:
        """Get 2D PS error covariance matrix for specified sampling method.
        
        Parameters
        ----------
        method : str, optional
            Sampling method: 'em' or 'ode'. Default uses the method from output.
        
        Returns
        -------
        np.ndarray | None
            Covariance matrix shape (2048, 2048), or None if not available.
        """
        if self._properties is None:
            return None
        method = method or self._ps_sampling_method
        return self._properties.get_ps_covariance(method=method)
    
    @property
    def ps_diagonal_fraction(self) -> float | None:
        """Fraction of covariance on diagonal (1 = uncorrelated errors)."""
        if self._properties is None:
            return None
        if self._ps_sampling_method == "ode":
            return self._properties.diag_frac_ode
        return self._properties.diag_frac_em
    
    @property
    def ps_mean_abs_correlation(self) -> float | None:
        """Mean absolute off-diagonal correlation (0 = uncorrelated)."""
        if self._properties is None:
            return None
        if self._ps_sampling_method == "ode":
            return self._properties.mean_abs_corr_ode
        return self._properties.mean_abs_corr_em
    
    def __contains__(self, key: str) -> bool:
        """Check if error field exists (for dict-like 'in' operator)."""
        return key in self.available_errors()
    
    def __getitem__(self, key: str) -> u.Quantity:
        """Allow dict-like access to error fields."""
        return getattr(self, key)
    
    def keys(self) -> list[str]:
        """Return list of error field names."""
        return list(self.available_errors().keys())
    
    def items(self):
        """Yield (name, value) pairs for all error fields."""
        for key in self.keys():
            yield key, getattr(self, key)


@dataclass(frozen=True)
class ACGEmulatorErrors(EmulatorErrors):
    """Error statistics for the ACG/Default (v1) emulator.
    
    This class provides **fractional errors (FE%)** from the test set for all
    outputs. Unlike the MH emulator which computes output-dependent absolute
    errors, the ACG emulator stores the raw FE% arrays directly.
    
    Error Interpretation
    --------------------
    The errors represent the **median fractional error** across a held-out test
    set of ~300 simulations. Each FE% value indicates the typical percentage
    error at that (redshift, k-mode) or (redshift, magnitude) bin::
    
        absolute_error = FE% / 100 * |emulator_output|
    
    For example, if ``PS_err[z_idx, k_idx] = 5%``, the emulator's power spectrum
    prediction at that bin is typically within 5% of the true value.
    
    Physics Context
    ---------------
    The ACG emulator (Atomic Cooling Galaxies) models the 21-cm signal from
    star-forming galaxies in atomic-cooling halos (M > 10^8 M_sun). It does
    **not** include:
    
    - Mini-halos (molecular cooling)
    - Exotic radio backgrounds
    - Variable cosmology (fixed ΣCDM)
    
    This emulator is described in Breitman et al. (2024).
    
    Attributes
    ----------
    PS_err : Quantity
        Power spectrum fractional error in percent. Shape (n_z, n_k).
        Units: percent. Typical values: 1-10%.
    Tb_err : Quantity
        Global brightness temperature fractional error. Shape (n_z,).
        Units: percent. Typical values: 1-5%.
    xHI_err : Quantity
        Neutral fraction fractional error. Shape (n_z,).
        Units: percent. Typical values: 1-3%.
    Ts_err : Quantity
        Spin temperature fractional error. Shape (n_z,).
        Units: percent. Typical values: 1-5%.
    tau_err : Quantity
        Optical depth fractional error. Scalar.
        Units: percent. Typical value: ~1%.
    UVLFs_err : Quantity
        UV luminosity function (linear) fractional error. Shape (n_z, n_mag).
        Units: percent.
    UVLFs_logerr : Quantity
        UV luminosity function (log10) fractional error. Shape (n_z, n_mag).
        Units: percent. Typically smaller than linear errors.
    
    Examples
    --------
    Basic usage:
    
    >>> emu = Emulator(emulator="acg")
    >>> theta, output, errors = emu.predict(params)
    >>> print(f"Median PS error: {np.median(errors.PS_err):.1f}")
    
    Computing absolute errors:
    
    >>> # Convert FE% to absolute error
    >>> abs_ps_err = errors.PS_err.value / 100 * np.abs(output.PS)
    
    Plotting with error bands:
    
    >>> import matplotlib.pyplot as plt
    >>> z_idx = 20  # Some redshift
    >>> plt.fill_between(
    ...     emu.properties.PS_ks,
    ...     output.PS[z_idx] * (1 - errors.PS_err[z_idx]/100),
    ...     output.PS[z_idx] * (1 + errors.PS_err[z_idx]/100),
    ...     alpha=0.3
    ... )
    
    See Also
    --------
    DefaultEmulatorOutput : The output dataclass these errors correspond to.
    DefaultEmulatorProperties : Emulator properties including error arrays.
    """
    
    PS_err: u.Quantity
    Tb_err: u.Quantity
    xHI_err: u.Quantity
    Ts_err: u.Quantity
    tau_err: u.Quantity
    UVLFs_err: u.Quantity
    UVLFs_logerr: u.Quantity
    
    _properties: object = dc.field(default=None, repr=False)
    
    def available_errors(self) -> dict[str, str]:
        """Return dict of available error fields and their descriptions."""
        return {
            "PS_err": "PS fractional error (FE%)",
            "Tb_err": "Tb fractional error (FE%)",
            "xHI_err": "xHI fractional error (FE%)",
            "Ts_err": "Ts fractional error (FE%)",
            "tau_err": "tau fractional error (FE%)",
            "UVLFs_err": "Linear UVLF fractional error (FE%)",
            "UVLFs_logerr": "Log UVLF fractional error (FE%)",
        }
    
    @property
    def properties(self):
        """Access the underlying emulator properties."""
        return self._properties
    
    @classmethod
    def from_properties(
        cls,
        properties: "DefaultEmulatorProperties",
    ) -> "ACGEmulatorErrors":
        """Construct error statistics from emulator properties.
        
        Parameters
        ----------
        properties : DefaultEmulatorProperties
            The emulator properties containing error arrays.
        
        Returns
        -------
        ACGEmulatorErrors
            Error statistics with dimensionless units (FE% values).
        """
        return cls(
            PS_err=properties.PS_err * u.percent,
            Tb_err=properties.Tb_err * u.percent,
            xHI_err=properties.xHI_err * u.percent,
            Ts_err=properties.Ts_err * u.percent,
            tau_err=properties.tau_err * u.percent,
            UVLFs_err=properties.UVLFs_err * u.percent,
            UVLFs_logerr=properties.UVLFs_logerr * u.percent,
            _properties=properties,
        )
    
    def __contains__(self, key: str) -> bool:
        """Check if error field exists."""
        return key in self.available_errors()
    
    def __getitem__(self, key: str) -> u.Quantity:
        """Allow dict-like access to error fields."""
        return getattr(self, key)
    
    def keys(self) -> list[str]:
        """Return list of error field names."""
        return list(self.available_errors().keys())
    
    def items(self):
        """Yield (name, value) pairs for all error fields."""
        for key in self.keys():
            yield key, getattr(self, key)


@dataclass(frozen=True)
class RadioEmulatorErrors(EmulatorErrors):
    """Error statistics for the Radio Background (v2) emulator.
    
    This class provides **fractional errors (FE%)** from the test set for all
    outputs of the radio background emulator. The radio emulator has a different
    output set than ACG/MH: it includes radio temperature (Tr) but does **not**
    include spin temperature (Ts) or UV luminosity functions (UVLFs).
    
    Error Interpretation
    --------------------
    The errors represent the **median fractional error** across a held-out test
    set. Each FE% value indicates the typical percentage error at that
    (redshift, k-mode) bin::
    
        absolute_error = FE% / 100 * |emulator_output|
    
    Physics Context
    ---------------
    The Radio emulator models the 21-cm signal including an **exotic radio
    background** component from high-z radio sources. Key differences from ACG:
    
    - Includes mini-halos (molecular cooling, M < 10^8 M_sun)
    - Models Lyman-Werner feedback
    - Outputs radio temperature Tr instead of spin temperature Ts
    - Does not output UVLFs (focus is on earlier epochs)
    
    This emulator is described in Reis et al. (2023).
    
    Attributes
    ----------
    PS_err : Quantity
        Power spectrum fractional error in percent. Shape (n_z, n_k).
        Units: percent.
    Tb_err : Quantity
        Global brightness temperature fractional error. Shape (n_z,).
        Units: percent.
    xHI_err : Quantity
        Neutral fraction fractional error. Shape (n_z,).
        Units: percent.
    Tr_err : Quantity
        Radio temperature fractional error. Shape (n_z,).
        Units: percent. This is the background radio temperature from
        high-z radio sources, not the CMB.
    tau_err : Quantity
        Optical depth fractional error. Scalar.
        Units: percent.
    
    Notes
    -----
    The radio emulator does **not** include:
    
    - ``Ts_err``: No spin temperature output (use Tr instead)
    - ``UVLFs_err``: No UV luminosity functions
    
    Examples
    --------
    Basic usage:
    
    >>> emu = Emulator(emulator="radio")
    >>> theta, output, errors = emu.predict(params)
    >>> print(f"Median radio temp error: {np.median(errors.Tr_err):.1f}%")
    
    Available error fields:
    
    >>> print(errors.keys())  # ['PS_err', 'Tb_err', 'xHI_err', 'Tr_err', 'tau_err']
    >>> print("UVLFs_err" in errors)  # False - not available for radio emulator
    
    See Also
    --------
    RadioEmulatorOutput : The output dataclass these errors correspond to.
    RadioBackgroundEmulatorProperties : Emulator properties including error arrays.
    """
    
    PS_err: u.Quantity
    Tb_err: u.Quantity
    xHI_err: u.Quantity
    Tr_err: u.Quantity
    tau_err: u.Quantity
    
    _properties: object = dc.field(default=None, repr=False)
    
    def available_errors(self) -> dict[str, str]:
        """Return dict of available error fields and their descriptions."""
        return {
            "PS_err": "PS fractional error (FE%)",
            "Tb_err": "Tb fractional error (FE%)",
            "xHI_err": "xHI fractional error (FE%)",
            "Tr_err": "Radio temperature Tr fractional error (FE%)",
            "tau_err": "tau fractional error (FE%)",
        }
    
    @property
    def properties(self):
        """Access the underlying emulator properties."""
        return self._properties
    
    @classmethod
    def from_properties(
        cls,
        properties: "RadioBackgroundEmulatorProperties",
    ) -> "RadioEmulatorErrors":
        """Construct error statistics from emulator properties.
        
        Parameters
        ----------
        properties : RadioBackgroundEmulatorProperties
            The emulator properties containing error arrays.
        
        Returns
        -------
        RadioEmulatorErrors
            Error statistics with dimensionless units (FE% values).
        """
        return cls(
            PS_err=properties.PS_err * u.percent,
            Tb_err=properties.Tb_err * u.percent,
            xHI_err=properties.xHI_err * u.percent,
            Tr_err=properties.Tr_err * u.percent,
            tau_err=properties.tau_err * u.percent,
            _properties=properties,
        )
    
    def __contains__(self, key: str) -> bool:
        """Check if error field exists."""
        return key in self.available_errors()
    
    def __getitem__(self, key: str) -> u.Quantity:
        """Allow dict-like access to error fields."""
        return getattr(self, key)
    
    def keys(self) -> list[str]:
        """Return list of error field names."""
        return list(self.available_errors().keys())
    
    def items(self):
        """Yield (name, value) pairs for all error fields."""
        for key in self.keys():
            yield key, getattr(self, key)
