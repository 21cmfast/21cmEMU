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
- PS : 21-cm power spectrum Δ² [mK²] in LINEAR units
- PS_2D : 2D power spectrum [mK²] in LINEAR units (MH emulator only)
- PS_2D_samples : 2D PS samples [mK²] in LINEAR units

Logarithmic quantities (returned in dex units):
- UVLFs : UV luminosity functions [dex(Mpc^-3 mag^-1)] = log10(φ)

To convert log quantities to linear, use `.physical`::

    output.UVLFs.physical  # returns Mpc^-3 mag^-1

Coordinate axes (returned with units):
- redshifts : Redshift values [dimensionless]
- kperp, kpar, k : Wavenumbers [Mpc^-1]
- Muv : UV magnitude [mag]

Error Statistics
----------------
Error statistics are available via properties like ``output.PS_err``.

**IMPORTANT**: PS is ALWAYS returned in LINEAR mK² units. Internally, PS values
are normalized and trained in log10 space, but they are converted to linear units
before being returned to the user. If you need log10(PS), use ``np.log10(output.PS)``.

Error types:
- ``PS_err``: Median fractional error (FE%) computed on log10(PS) values
- ``PS_2D_err``: Median FE% on 2D PS log10 values
- ``PS_2D_var``: Variance of FE% across test set
- ``PS_2D_cov``: Covariance matrix of FE% between pixels

Fractional error is defined as::

    FE% = |log10(true) - log10(predicted)| / |log10(true)| × 100

Interpretation: A 5% FE on log10(PS) corresponds to ~12% error on linear PS,
because a 5% error in the log10 exponent multiplies the result by 10^0.05 ≈ 1.12.

See ``MHEmulatorProperties`` for detailed documentation of all error statistics.

Note: The RawEmulatorOutput classes store raw numpy arrays without units.
The EmulatorOutput classes (returned by `emulator.predict()`) automatically
attach units to all quantities.
"""

from __future__ import annotations

import dataclasses as dc
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import expit

from .properties import emulator_properties

if TYPE_CHECKING:  # pragma: no cover
    from .properties import (
        DefaultEmulatorProperties,
        MHEmulatorProperties,
        RadioEmulatorProperties,
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
        # PS is delta^2 (dimensionless power spectrum), returned in LINEAR mK^2
        "PS": u.mK**2,
        "PS_2D": u.mK**2,
        "PS_2D_samples": u.mK**2,
        "PS_2D_std": u.mK**2,
        # Log quantities (dex = log10 of base unit)
        "UVLFs": u.dex(u.Mpc**-3 * u.mag**-1),
        # Coordinate axes
        "redshifts": u.dimensionless_unscaled,
        "PS_2D_redshifts": u.dimensionless_unscaled,
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

    Note: PS is NOT in this set - it's always returned in linear units (mK^2).
    """
    return {"UVLFs"}


# Known data fields that should be wrapped with units
_UNIT_FIELDS = frozenset(
    [
        "Tb",
        "xHI",
        "Ts",
        "Tr",
        "tau",
        "UVLFs",
        "PS",
        "PS_2D",
        "PS_2D_samples",
        "PS_2D_std",
        "redshifts",
        "PS_2D_redshifts",
        "UVLF_redshifts",
        "kperp",
        "kpar",
        "k",
        "PS_ks",
        "Muv",
        "Nmodes",
    ]
)


@dataclass(frozen=True)
class EmulatorOutput:
    """Base class for emulator output with automatic unit handling.

    All output quantities are returned as astropy Quantities with units.

    **Units Convention**:
    - PS quantities (PS, PS_2D): Always in LINEAR units [mK²]
    - UVLFs: In log10 space [dex(Mpc⁻³ mag⁻¹)], use .physical to convert
    - Other quantities: In physical units (Tb [mK], xHI [dimensionless], etc.)

    Example::

        output.PS              # mK² (linear units)
        output.UVLFs           # dex(Mpc⁻³ mag⁻¹) (log10 units)
        output.UVLFs.physical  # Mpc⁻³ mag⁻¹ (converts to linear)

    See Also
    --------
    :func:`log_quantities` : Returns set of fields stored in log10 space.
    :func:`available_units` : Returns dict of all field units.
    """

    def __post_init__(self):
        """Attach astropy units to data fields at construction time.

        Iterates over all dataclass fields and wraps numeric arrays with their
        corresponding astropy units.  Fields that are already Quantities (e.g.
        when constructing from ``squeeze()``) or that are ``None`` are left
        unchanged.  Using ``__post_init__`` instead of ``__getattribute__``
        avoids the infinite recursion that arises when tools like typeguard
        instrument attribute access.
        """
        units = _get_units()
        for f in dc.fields(self):
            if f.name not in _UNIT_FIELDS:
                continue  # pragma: no cover
            value = object.__getattribute__(self, f.name)
            if value is None or isinstance(value, u.Quantity):
                continue
            if f.name in units:  # pragma: no branch
                object.__setattr__(self, f.name, value * units[f.name])

    def keys(self) -> Generator[str]:
        """Yield the keys of the main data products."""
        for k in dc.fields(self):
            yield k.name

    def items(self) -> Generator[tuple[str, np.ndarray | None]]:
        """Yield the keys and raw values of the main data products, like a dict."""
        for k in self.keys():
            yield k, object.__getattribute__(self, k)

    def __getitem__(self, key: str):
        """Allow access to attributes as items (returns Quantity)."""
        return getattr(self, key)

    @property
    def redshifts(self) -> u.Quantity:
        """Redshifts for global summaries (Tb, xHI, Ts, tau).

        Returns
        -------
        Quantity[dimensionless]
            Redshift values at which global summaries are evaluated
        """
        return self.properties.zs * u.dimensionless_unscaled

    @property
    def PS_redshifts(self) -> u.Quantity:
        """Redshifts at which power spectrum is evaluated.

        Returns
        -------
        Quantity[dimensionless]
            Redshift values for PS output
        """
        return self.properties.PS_zs * u.dimensionless_unscaled

    @property
    def PS_ks(self) -> u.Quantity:
        """Wavenumbers for power spectrum.

        Returns
        -------
        Quantity[Mpc⁻¹]
            Wavenumbers in comoving Mpc⁻¹ units
        """
        return self.properties.PS_ks * u.Mpc**-1

    @property
    def k(self) -> u.Quantity:
        """Wavenumbers for power spectrum (alias for PS_ks).

        Returns
        -------
        Quantity[Mpc⁻¹]
            Wavenumbers in comoving Mpc⁻¹ units
        """
        return self.properties.PS_ks * u.Mpc**-1

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
        out = {}
        for k in store:
            val = object.__getattribute__(self, k)
            out[k] = val.value if isinstance(val, u.Quantity) else val
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
        linear units, use the `.physical` attribute::

            output.UVLFs           # dex(Mpc⁻³ mag⁻¹)
            output.UVLFs.physical  # Mpc⁻³ mag⁻¹

        **Note**: PS is NOT in this set - PS is always returned in LINEAR
        mK² units, not log10 units.

        Returns
        -------
        set of str
            Field names that are in log10 space: {"UVLFs"}.
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
    """Output from the Default/ACG (v1) emulator.

    All quantities are returned with astropy units attached.

    Attributes
    ----------
    Tb : Quantity[mK]
        Global brightness temperature as function of redshift. Shape: (n_z,)
    xHI : Quantity[dimensionless]
        Neutral hydrogen fraction as function of redshift. Shape: (n_z,)
    Ts : Quantity[K]
        Spin temperature as function of redshift. Shape: (n_z,)
    PS : Quantity[mK²]
        1D power spectrum Δ² in LINEAR units. Shape: (n_z, n_k)
    tau : Quantity[dimensionless]
        Optical depth to reionization. Scalar.
    UVLFs : Quantity[dex(Mpc⁻³ mag⁻¹)]
        UV luminosity functions in log10 space. Shape: (n_z_uvlf, n_mag)
        Use .physical to convert to linear units.
    """

    Tb: u.Quantity
    xHI: u.Quantity
    Ts: u.Quantity
    PS: u.Quantity
    tau: u.Quantity
    UVLFs: u.Quantity

    properties = emulator_properties(emulator="acg")

    @property
    def PS_1D_redshifts(self) -> u.Quantity:
        """Redshifts at which 1D PS is evaluated.

        Returns
        -------
        Quantity[dimensionless]
            Redshift values, shape (32,)
        """
        return self.properties.PS_zs * u.dimensionless_unscaled

    @property
    def PS_1D_k(self) -> u.Quantity:
        """Wavenumbers at which 1D PS is evaluated.

        Returns
        -------
        Quantity[Mpc⁻¹]
            k-values in comoving Mpc⁻¹ units
        """
        return self.properties.PS_ks * u.Mpc**-1

    @property
    def Muv(self) -> u.Quantity:
        """UV absolute magnitudes for UVLF sampling.

        Returns
        -------
        Quantity[mag]
            UV magnitudes in range [-20, -10], shape (n_mag,)
        """
        # Crop the M_UV to -20 to -10 range
        m = np.logical_and(
            self.properties.UVLFs_MUVs <= -10, self.properties.UVLFs_MUVs >= -20
        )
        return self.properties.UVLFs_MUVs[m] * u.mag

    @property
    def UVLF_redshifts(self) -> u.Quantity:
        """Redshifts at which UVLFs are evaluated.

        Returns
        -------
        Quantity[dimensionless]
            Redshift values, shape (n_z_uvlf,)
        """
        return self.properties.uv_lf_zs * u.dimensionless_unscaled

    def squeeze(self):
        """Return a new EmulatorOutput with all dimensions of length 1 removed."""
        return DefaultEmulatorOutput(**{k: np.squeeze(v) for k, v in self.items()})


@dataclass(frozen=True)
class RadioEmulatorOutput(EmulatorOutput):
    """Output from the Radio (v2) emulator.

    All quantities are returned with astropy units attached.

    Attributes
    ----------
    Tb : Quantity[mK]
        Global brightness temperature as function of redshift. Shape: (n_z,)
    xHI : Quantity[dimensionless]
        Neutral hydrogen fraction as function of redshift. Shape: (n_z,)
    Tr : Quantity[K]
        Radio background temperature as function of redshift. Shape: (n_z,)
    PS : Quantity[mK²]
        1D power spectrum Δ² in LINEAR units. Shape: (n_z, n_k)
    tau : Quantity[dimensionless]
        Optical depth to reionization. Scalar.
    """

    Tb: u.Quantity
    xHI: u.Quantity
    Tr: u.Quantity
    PS: u.Quantity
    tau: u.Quantity

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
        """Raw normalized 1D power spectrum (in log10 space, needs denormalization).

        This returns the raw emulator output before denormalization.
        After calling get_renormalized(), PS will be converted to LINEAR mK² units.
        """
        return self.output[:, 3 * self.nz + 1 : 3 * self.nz + 1 + 60 * 12].reshape(
            (-1, 60, 12)
        )

    @property
    def tau(self) -> np.ndarray:
        """The optical depth of reionization."""
        return self.output[:, self.nz * 3 + 1 + 60 * 12]

    @property
    def UVLFs(self) -> np.ndarray:
        """Raw normalized UV luminosity functions (in log10 space, needs denormalization).

        This returns the raw emulator output before denormalization.
        After calling get_renormalized(), UVLFs will be in log10 space with units
        [dex(Mpc⁻³ mag⁻¹)]. Use .physical on the final output to convert to linear.
        """
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
        DefaultEmulatorOutput
            The emulator output with all quantities in physical units.
            - PS: LINEAR mK² units
            - UVLFs: log10 space [dex(Mpc⁻³ mag⁻¹)]
            - All other quantities: physical units (Tb [mK], xHI [dimensionless], etc.)
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

        # Convert quantities from log10 to linear space
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
        r"""Raw normalized 1D power spectrum (in log10 space, needs denormalization).

        This returns the raw emulator output before denormalization.
        After calling get_renormalized(), PS will be converted to LINEAR mK² units.
        """
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
        RadioEmulatorOutput
            The emulator output with all quantities in physical units.
            - PS: LINEAR mK² units
            - All other quantities: physical units (Tb [mK], xHI [dimensionless], etc.)
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
        # PS: denormalize from log space and convert to linear units (mK^2)
        out["PS"] = 10 ** (
            (self.PS * self.properties.logPS_std) + self.properties.logPS_mean
        )

        # zs axis was flipped.
        # I'll flip the global summaries instead to keep the zs in increasing order.
        out["xHI"] = self.xHI[:, ::-1]
        out["Tb"] = out["Tb"][:, ::-1]
        out["Tr"] = out["Tr"][:, ::-1]

        # Convert tau from log10 to linear space
        # Note: PS stays in log10 space (dex units) - use .physical to convert
        out["tau"] = 10 ** (self.tau)

        other = {
            k.name: getattr(self, k.name)
            for k in dc.fields(RadioEmulatorOutput)
            if k.name not in out.keys()
        }

        return RadioEmulatorOutput(**{**out, **other}).squeeze()


@dataclass(frozen=True)
class MHEmulatorOutput(EmulatorOutput):
    """Output from the MH/MCG (v3) emulator.

    All quantities are returned with astropy units attached.

    **IMPORTANT**: All PS quantities are returned in LINEAR mK² units, NOT log10.
    Internally, the emulator trains on log10(PS) but converts to linear before
    returning. To get log10 values, use ``np.log10(output.PS)``.

    Attributes
    ----------
    Tb : Quantity[mK]
        Global brightness temperature as function of redshift.
        Shape: (batch, 32 redshifts)
    xHI : Quantity[dimensionless]
        Neutral hydrogen fraction as function of redshift.
        Shape: (batch, 32 redshifts)
    Ts : Quantity[K]
        Spin temperature as function of redshift.
        Shape: (batch, 32 redshifts)
    tau : Quantity[dimensionless]
        Optical depth to reionization.
        Shape: (batch,)
    UVLFs : Quantity[dex(Mpc⁻³ mag⁻¹)]
        UV luminosity functions in log10 space: log10(φ).
        Use .physical to convert to linear units.
        Shape: (batch, n_Muv, n_redshifts)
    PS : Quantity[mK²]
        1D power spectrum Δ² in LINEAR units from LSTM model.
        Shape: (batch, 32 redshifts, 32 k)
    PS_2D : Quantity[mK²] | None
        2D power spectrum Δ² in LINEAR units, median over realizations.
        Only available when emulate_2d_ps=True.
        Shape: (batch, n_redshifts, 32 kperp, 64 kpar)
    PS_2D_samples : Quantity[mK²] | None
        2D power spectrum samples in LINEAR units from diffusion model.
        Only available when emulate_2d_ps=True.
        Shape: (batch, n_redshifts, n_samples, 32 kperp, 64 kpar)
    PS_2D_std : Quantity[mK²] | None
        Standard deviation of 2D PS over realizations in LINEAR units.
        Only available when emulate_2d_ps=True.
        Shape: (batch, n_redshifts, 32 kperp, 64 kpar)
    PS_2D_redshifts : Quantity[dimensionless] | None
        Redshifts for the 2D PS (user-specified or default).
        Only available when emulate_2d_ps=True.
    """

    Tb: u.Quantity
    xHI: u.Quantity
    Ts: u.Quantity
    tau: u.Quantity
    UVLFs: u.Quantity
    PS: u.Quantity
    PS_2D: u.Quantity | None
    PS_2D_samples: u.Quantity | None
    PS_2D_std: u.Quantity | None
    PS_2D_redshifts: u.Quantity | None

    properties = emulator_properties(emulator="mcg")

    @property
    def PS_1D_k(self) -> u.Quantity:
        """Wavenumbers for 1D power spectrum.

        Returns
        -------
        Quantity[Mpc⁻¹]
            Wavenumbers in comoving Mpc⁻¹ units, shape (32,)
        """
        return self.properties.PS_1D_k * u.Mpc**-1

    @property
    def PS_1D_redshifts(self) -> u.Quantity:
        """Redshifts at which 1D PS is evaluated.

        Returns
        -------
        Quantity[dimensionless]
            Redshift values, shape (32,)
        """
        return self.properties.PS_1D_redshifts * u.dimensionless_unscaled

    @property
    def kperp(self) -> u.Quantity:
        """Perpendicular wavenumbers for 2D power spectrum.

        Returns
        -------
        Quantity[Mpc⁻¹]
            k_perp values in comoving Mpc⁻¹ units, shape (32,)
        """
        return self.properties.kperp * u.Mpc**-1

    @property
    def kpar(self) -> u.Quantity:
        """Parallel wavenumbers for 2D power spectrum.

        Returns
        -------
        Quantity[Mpc⁻¹]
            k_parallel values in comoving Mpc⁻¹ units, shape (64,)
        """
        return self.properties.kpar * u.Mpc**-1

    @property
    def Nmodes(self) -> u.Quantity:
        """Number of Fourier modes per 2D PS bin.

        Returns
        -------
        Quantity[dimensionless]
            Mode counts for each (k_perp, k_par) bin, shape (32, 64)
        """
        return self.properties.Nmodes * u.dimensionless_unscaled

    @property
    def PS_err(self) -> u.Quantity:
        """Median fractional error on 1D PS values.

        **IMPORTANT**: Although PS is RETURNED in LINEAR mK² units, this error
        statistic is computed on log10(PS) during training and validation.

        Returns
        -------
        Quantity[dimensionless]
            Median fractional error as percentage (%), shape (32 redshifts, 32 k).
            This is FE% computed as: 100 * |log10(true) - log10(pred)| / |log10(true)|

        Interpretation
        --------------
        A 5% error on log10(PS) means:
        - The log10 exponent is off by ~5%
        - Corresponds to ~12% multiplicative error in LINEAR PS (10^0.05 ≈ 1.12)

        To estimate the error in your linear PS values::

            relative_error_linear = 10**(PS_err/100) - 1  # fractional error
            absolute_error_linear = output.PS * relative_error_linear  # in mK²

        See Also
        --------
        PS_2D_err : Equivalent for 2D power spectrum
        MHEmulatorProperties : Full error documentation
        """
        return self.properties.PS_1D_med_err * u.dimensionless_unscaled

    @property
    def PS_2D_err(self) -> u.Quantity | None:
        """Median fractional error on 2D PS values.

        **IMPORTANT**: Although PS_2D is RETURNED in LINEAR mK² units, this error
        statistic is computed on log10(PS) during training and validation.

        Returns
        -------
        Quantity[dimensionless] | None
            Median fractional error as percentage (%), shape (32 kperp, 64 kpar).
            Returns None if emulate_2d_ps=False.

        Note
        ----
        Uses ODE sampler error statistics (default). For EM sampler errors,
        access ``properties.PS_med_err_em`` directly.

        Interpretation
        --------------
        Same as PS_err - see that property for details. A 5% error on log10(PS)
        corresponds to ~12% multiplicative error in the returned linear PS values.

        To estimate the error in your linear PS_2D values::

            relative_error_linear = 10**(PS_2D_err/100) - 1
            absolute_error_linear = output.PS_2D * relative_error_linear  # in mK²

        See Also
        --------
        PS_err : Equivalent for 1D power spectrum
        PS_2D_var : Variance of error distribution
        PS_2D_cov : Full covariance matrix
        """
        if self.PS_2D is None:
            return None
        return self.properties.PS_med_err * u.dimensionless_unscaled

    # ═══════════════════════════════════════════════════════════════════════════
    # 2D PS Error Distribution Statistics
    # ═══════════════════════════════════════════════════════════════════════════
    # These statistics characterize the emulator error distribution for 2D PS.
    # All are computed on the fractional error (FE%) of log10(PS).
    #
    # IMPORTANT: These statistics describe errors on LOG10 values, not linear PS.
    # A 5% FE on log10(PS) ≈ 12% error on linear PS (since 10^0.05 ≈ 1.12).

    @property
    def PS_2D_var(self) -> u.Quantity | None:
        """Variance of 2D PS emulator error across test set.

        Variance of the fractional error (FE%) computed on log10(PS) values
        during validation. Measures the spread of errors at each pixel.

        Returns
        -------
        Quantity[dimensionless] | None
            Error variance as (FE%)², shape (32 kperp, 64 kpar).
            Returns None if emulate_2d_ps=False.

        Note
        ----
        Uses ODE sampler statistics. Despite PS_2D being returned in LINEAR
        units, this variance is computed on log10(PS) errors.
        """
        if self.PS_2D is None:
            return None
        var = self.properties.PS_var
        if var is None:
            return None
        return var * u.dimensionless_unscaled

    @property
    def PS_2D_cov(self) -> u.Quantity | None:
        """Covariance matrix of 2D PS emulator error.

        Full covariance of fractional errors (FE%) computed on log10(PS)
        between all pairs of (k_perp, k_par) pixels from test set validation.

        Returns
        -------
        Quantity[dimensionless] | None
            Flattened covariance matrix of error as (FE%)², shape (2048, 2048)
            where 2048 = 32 × 64 (flattened k-space grid).
            Returns None if emulate_2d_ps=False.

        Note
        ----
        Pixels ordered by row-major raveling. Use ``PS_2D_cov_4d()`` to get
        reshaped (32, 64, 32, 64) version where cov[i,j,k,l] gives covariance
        between pixels (k_perp[i], k_par[j]) and (k_perp[k], k_par[l]).

        See Also
        --------
        PS_2D_cov_4d : Reshaped 4D version
        PS_2D_corr_diag_frac : Diagonal fraction metric
        """
        if self.PS_2D is None:
            return None
        cov = self.properties.PS_cov
        if cov is None:
            return None
        return cov * u.dimensionless_unscaled

    def PS_2D_cov_4d(self) -> u.Quantity | None:
        """Covariance matrix reshaped to 4D for easier indexing.

        Reshapes the flattened (2048, 2048) covariance matrix to
        (32, 64, 32, 64) where dimensions are (k_perp, k_par, k_perp', k_par').

        Returns
        -------
        Quantity[dimensionless] | None
            4D covariance array where cov_4d[i, j, k, l] gives the covariance
            between pixels (k_perp[i], k_par[j]) and (k_perp[k], k_par[l]).
            Returns None if not available.

        See Also
        --------
        PS_2D_cov : Flattened version
        """
        cov = self.PS_2D_cov
        if cov is None:
            return None
        return cov.reshape(32, 64, 32, 64)

    @property
    def PS_2D_corr_diag_frac(self) -> float | None:
        """Fraction of error variance that is uncorrelated (diagonal).

        Quantifies how much of the total error variance comes from independent
        pixel errors vs. correlated errors across k-space.

        Returns
        -------
        float | None
            Fraction in [0, 1]. Values near 1: mostly independent pixel errors.
            Values < 0.8: significant spatial correlations in errors.
            Returns None if not available.

        See Also
        --------
        PS_2D_mean_abs_corr : Mean correlation strength
        """
        if self.PS_2D is None:
            return None
        return self.properties.diag_frac

    @property
    def PS_2D_mean_abs_corr(self) -> float | None:
        """Mean absolute correlation between pixel errors.

        Average |r_ij| over all off-diagonal pairs where r_ij is the Pearson
        correlation coefficient. Indicates typical correlation strength.

        Returns
        -------
        float | None
            Mean |correlation| in [0, 1]. Values < 0.1: weakly correlated.
            Values > 0.3: strong spatial correlation structure.
            Returns None if not available.

        See Also
        --------
        PS_2D_corr_diag_frac : Overall decorrelation metric
        """
        if self.PS_2D is None:
            return None
        return self.properties.mean_abs_corr

    @property
    def Muv(self) -> u.Quantity:
        """UV absolute magnitudes for UVLF sampling.

        Returns
        -------
        Quantity[mag]
            UV magnitudes in range [-20, -10], shape (n_mag,)
        """
        m = np.logical_and(
            self.properties.UVLFs_MUVs <= -10, self.properties.UVLFs_MUVs >= -20
        )
        return self.properties.UVLFs_MUVs[m] * u.mag

    @property
    def UVLF_redshifts(self) -> u.Quantity:
        """Redshifts at which UVLFs are evaluated.

        Returns
        -------
        Quantity[dimensionless]
            Redshift values, shape (n_z_uvlf,)
        """
        return self.properties.uv_lf_zs * u.dimensionless_unscaled

    @property
    def redshifts(self) -> u.Quantity:
        """Redshifts for global summaries (Tb, xHI, Ts).

        Returns
        -------
        Quantity[dimensionless]
            Redshift values for main outputs, shape (32,)
        """
        return self.properties.redshifts * u.dimensionless_unscaled

    def squeeze(self):
        return MHEmulatorOutput(
            **{k: (np.squeeze(v) if v is not None else None) for k, v in self.items()}
        )


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
        out = self.output[5]
        return out.cpu().detach().numpy() if hasattr(out, "cpu") else out

    @property
    def UVLFs(self) -> np.ndarray:
        full_UVLFs = self.output[3]
        m = np.logical_and(
            self.properties.UVLFs_MUVs <= -10, self.properties.UVLFs_MUVs >= -20
        )
        if hasattr(full_UVLFs, "cpu"):
            return full_UVLFs.cpu().detach().numpy()[:, m, :]
        return full_UVLFs[:, m, :]  # pragma: no cover

    @property
    def PS(self) -> np.ndarray:
        """Raw normalized 1D PS from LSTM (in log10 space, needs denormalization).

        This returns the raw emulator output before denormalization.
        After calling get_renormalized(), PS will be converted to LINEAR mK² units.
        """
        out = self.output[4]
        if out is None:
            return None  # pragma: no cover
        return out.cpu().detach().numpy() if hasattr(out, "cpu") else out

    @property
    def PS_2D_samples(self) -> np.ndarray | None:
        """Raw 2D PS samples from score model (in LINEAR mK², already denormalized).

        The diffusion model directly outputs PS values in linear space.
        These samples are already in physical mK² units.
        """
        return self.output[6] if len(self.output) > 6 else None

    @property
    def PS_2D_redshifts(self) -> np.ndarray | None:
        """Redshifts for 2D PS."""
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
            raise ValueError(  # pragma: no cover
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
        else:  # pragma: no cover
            ts_val = ts_raw
        out["Ts"] = 10 ** ts_val.squeeze()
        out["xHI"] = out["xHI"].squeeze()
        out["Tb"] = out["Tb"].squeeze()

        # Convert tau from log10 to linear space
        out["tau"] = 10 ** out["tau"]
        # UVLFs stay in log10 space (dex units) - use .physical to convert
        out["UVLFs"] = np.swapaxes(out["UVLFs"], 2, 1)

        # 1D PS from LSTM: denormalize from log space and convert to linear units (mK^2)
        # Formula: PS = 10^(PS_norm * PS_1D_log_std + PS_1D_log_mean)
        ps_1d_norm = out.get("PS")
        if ps_1d_norm is not None:  # pragma: no branch
            out["PS"] = 10 ** (
                ps_1d_norm.squeeze() * self.properties.PS_1D_log_std
                + self.properties.PS_1D_log_mean
            )

        # 2D PS samples from score model: already in log10 space, convert to linear
        ps_2d_samples = out.get("PS_2D_samples")
        if ps_2d_samples is not None:
            out["PS_2D_samples"] = ps_2d_samples
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
            PS_2D_redshifts=self.PS_2D_redshifts,
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
        return {}  # pragma: no cover

    def summary(self) -> str:
        """Return a human-readable summary of error statistics."""
        lines = ["Emulator Error Statistics", "=" * 40]
        for name, desc in self.available_errors().items():
            val = getattr(self, name, None)
            if val is None:
                lines.append(f"{name}: N/A ({desc})")  # pragma: no cover
            elif hasattr(val, "shape"):
                med = np.nanmedian(val.value if hasattr(val, "value") else val)
                lines.append(f"{name}: median={med:.2f} ({desc})")
            else:
                lines.append(f"{name}: {val} ({desc})")  # pragma: no cover
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

    - **PS errors**: Although PS is RETURNED in LINEAR mK² units, errors are
      computed on log10(PS) values and then converted. ``PS_err`` is in dex
      (log10 units). A ``PS_err`` of 0.05 dex means log10(PS) is off by ~0.05,
      corresponding to a multiplicative factor of 10^0.05 ≈ 1.12 (12%) error
      in the returned linear PS values.

    - **UVLF errors**: UVLFs are returned in log10 space [dex(Mpc⁻³ mag⁻¹)].
      ``UVLFs_logerr`` is in dex units. ``UVLFs_err`` gives the error on
      linear φ values after conversion from log space.

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
    PS_err : Quantity[dex(mK²)]
        Absolute error on log10(PS) in dex units. Shape (n_z, n_k).
        Note: Although PS is RETURNED in linear mK² units, this error is
        computed on log10(PS) values. Interpretation:
        log10(PS_true) ≈ log10(PS_pred) ± PS_err
    Tb_err : Quantity[mK]
        Absolute error on brightness temperature in mK. Shape (n_z,).
        Interpretation: Tb_true ≈ Tb_pred ± Tb_err
    xHI_err : Quantity[dimensionless]
        Absolute error on neutral fraction (dimensionless). Shape (n_z,).
        Range typically 0-1, error is additive.
    Ts_err : Quantity[K]
        Absolute error on spin temperature in K. Shape (n_z,).
    tau_err : Quantity[dimensionless]
        Absolute error on optical depth (dimensionless). Scalar.
    UVLFs_err : Quantity[Mpc⁻³ mag⁻¹]
        Absolute error on linear UV luminosity function φ in physical units.
        Shape (n_z, n_mag). For plotting linear LF with errorbars.
    UVLFs_logerr : Quantity[dex(Mpc⁻³ mag⁻¹)]
        Absolute error on log10(φ) in dex units. Shape (n_z, n_mag).
        Preferred for log-scale LF plots since UVLFs are returned in log10.

    Examples
    --------
    Basic usage with prediction::

        emu = Emulator(emulator="mcg")
        theta, output, errors = emu.predict(params)
        print(errors.PS_err.unit)  # dex(mK2)
        print(errors.summary())    # Human-readable summary

    Accessing 2D PS error statistics::

        errors.get_ps_variance()            # Per-bin variance
        errors.get_ps_covariance()          # Full covariance matrix
        print(f"Diagonal fraction: {errors.ps_diagonal_fraction:.2%}")
        print(f"Mean |correlation|: {errors.ps_mean_abs_correlation:.3f}")

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
        properties: MHEmulatorProperties,
        ps_sampling_method: str = "em",
    ) -> MHEmulatorErrors:
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
                return None  # pragma: no cover
            return x.value if hasattr(x, "value") else x

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
        m = np.logical_and(properties.UVLFs_MUVs <= -10, properties.UVLFs_MUVs >= -20)

        # Compute absolute errors from FE%
        # PS: handle shape mismatch between 1D PS output and 2D PS error
        if emu_PS is not None:
            try:
                # 1D PS error: use 1D PS-specific errors
                ps_err = properties.PS_1D_med_err / 100.0 * np.abs(emu_PS)
            except (ValueError, AttributeError):  # pragma: no cover
                # Fallback to scalar median
                ps_err = np.nanmedian(ps_fe) / 100.0 * np.abs(emu_PS)
        else:  # pragma: no cover
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
        if properties.UVLFs_lin_med_err is None:
            # Fallback from log error
            uvlf_linerr = uvlf_log_fe / 100.0 * (10**emu_UVLFs)
        else:  # pragma: no cover
            uvlf_lin_fe = np.swapaxes(properties.UVLFs_lin_med_err[m], 1, 0)
            uvlf_linerr = uvlf_lin_fe / 100.0 * (10**emu_UVLFs)

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
        self, method: str | None = None, stat: str = "median"
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
    Basic usage::

        emu = Emulator(emulator="acg")
        theta, output, errors = emu.predict(params)
        print(f"Median PS error: {np.median(errors.PS_err):.1f}")

    Computing absolute errors::

        # Convert FE% to absolute error
        abs_ps_err = errors.PS_err.value / 100 * np.abs(output.PS)

    Plotting with error bands::

        import matplotlib.pyplot as plt
        z_idx = 20  # Some redshift
        plt.fill_between(
            emu.properties.PS_ks,
            output.PS[z_idx] * (1 - errors.PS_err[z_idx]/100),
            output.PS[z_idx] * (1 + errors.PS_err[z_idx]/100),
            alpha=0.3
        )

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
            "PS_err": "Absolute error on PS [mK²]",
            "Tb_err": "Absolute error on brightness temperature [mK]",
            "xHI_err": "Absolute error on neutral fraction [dimensionless]",
            "Ts_err": "Absolute error on spin temperature [K]",
            "tau_err": "Absolute error on optical depth [dimensionless]",
            "UVLFs_err": "Absolute error on linear LF [Mpc⁻³ mag⁻¹]",
            "UVLFs_logerr": "Absolute error on log10(LF) [dex(Mpc⁻³ mag⁻¹)]",
        }

    @property
    def properties(self):
        """Access the underlying emulator properties."""
        return self._properties

    @classmethod
    def from_output(
        cls,
        output: DefaultEmulatorOutput,
        properties: DefaultEmulatorProperties,
    ) -> ACGEmulatorErrors:
        """Construct error statistics broadcast to match the output batch shape.

        The stored error arrays are pre-computed absolute errors (median absolute
        difference over the test set, after restoring units and removing any log
        transform).  This method broadcasts each 1-D / 2-D property error to the
        shape of the corresponding output field so that shapes always match,
        regardless of how many parameter sets were passed to ``predict()``.

        Parameters
        ----------
        output : DefaultEmulatorOutput
            The emulator output whose shapes define the target broadcast shape.
        properties : DefaultEmulatorProperties
            The emulator properties containing pre-computed absolute error arrays.

        Returns
        -------
        ACGEmulatorErrors
            Error statistics with physical units and batch dimension matching
            the output.
        """

        def _raw(x):
            return x.value if hasattr(x, "value") else np.asarray(x)

        def _bc(err, ref):
            """Broadcast err to the shape of ref (stripped of units)."""
            return np.broadcast_to(np.asarray(err), _raw(ref).shape)

        # Crop UVLFs error arrays to the same M_UV range used in the output
        m = np.logical_and(
            properties.UVLFs_MUVs <= -10,
            properties.UVLFs_MUVs >= -20,
        )
        uvlfs_err = properties.UVLFs_err[..., m]
        uvlfs_logerr = properties.UVLFs_logerr[..., m]

        return cls(
            PS_err=_bc(properties.PS_err, output.PS) * u.mK**2,
            Tb_err=_bc(properties.Tb_err, output.Tb) * u.mK,
            xHI_err=_bc(properties.xHI_err, output.xHI) * u.dimensionless_unscaled,
            Ts_err=_bc(properties.Ts_err, output.Ts) * u.K,
            tau_err=_bc(properties.tau_err, output.tau) * u.dimensionless_unscaled,
            UVLFs_err=_bc(uvlfs_err, output.UVLFs) * (u.Mpc**-3 * u.mag**-1),
            UVLFs_logerr=_bc(uvlfs_logerr, output.UVLFs) * u.dex(u.Mpc**-3 * u.mag**-1),
            _properties=properties,
        )

    @classmethod
    def from_properties(
        cls,
        properties: DefaultEmulatorProperties,
    ) -> ACGEmulatorErrors:
        """Construct error statistics from emulator properties (no batch dim).

        Prefer ``from_output`` when the emulator output is available, as it
        broadcasts the error arrays to match the output batch shape.

        Parameters
        ----------
        properties : DefaultEmulatorProperties
            The emulator properties containing pre-computed absolute error arrays.

        Returns
        -------
        ACGEmulatorErrors
            Error statistics with correct physical units attached.
        """
        # Crop UVLFs error arrays to the same M_UV range used in the output
        m = np.logical_and(
            properties.UVLFs_MUVs <= -10,
            properties.UVLFs_MUVs >= -20,
        )
        return cls(
            PS_err=properties.PS_err * u.mK**2,
            Tb_err=properties.Tb_err * u.mK,
            xHI_err=properties.xHI_err * u.dimensionless_unscaled,
            Ts_err=properties.Ts_err * u.K,
            tau_err=properties.tau_err * u.dimensionless_unscaled,
            UVLFs_err=properties.UVLFs_err[..., m] * (u.Mpc**-3 * u.mag**-1),
            UVLFs_logerr=properties.UVLFs_logerr[..., m] * u.dex(u.Mpc**-3 * u.mag**-1),
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
    Basic usage::

        emu = Emulator(emulator="radio")
        theta, output, errors = emu.predict(params)
        print(f"Median radio temp error: {np.median(errors.Tr_err):.1f}%")

    Available error fields::

        print(errors.keys())  # ['PS_err', 'Tb_err', 'xHI_err', 'Tr_err', 'tau_err']
        print("UVLFs_err" in errors)  # False - not available for radio emulator

    See Also
    --------
    RadioEmulatorOutput : The output dataclass these errors correspond to.
    RadioEmulatorProperties : Emulator properties including error arrays.
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
        properties: RadioEmulatorProperties,
    ) -> RadioEmulatorErrors:
        """Construct error statistics from emulator properties.

        Parameters
        ----------
        properties : RadioEmulatorProperties
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
