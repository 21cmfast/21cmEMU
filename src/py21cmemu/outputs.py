"""Module whose functionality is to organise the emulator output."""

from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np

from .properties import emulator_properties


@dataclass(frozen=True)
class EmulatorOutput:
    """A simple class that makes it easier to access the corrected emulator output."""

    def keys(self) -> Generator[str, None, None]:
        """Yield the keys of the main data products."""
        for k in dc.fields(self):
            yield k.name

    def items(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Yield the keys and values of the main data products, like a dict."""
        for k in self.keys():
            yield k, getattr(self, k)

    def __getitem__(self, key: str) -> np.ndarray:
        """Allow access to attributes as items."""
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

        out = {k: getattr(self, k) for k in store}
        if theta is not None:
            out["inputs"] = theta

        np.savez(fname, out)


@dataclass(frozen=True)
class DefaultEmulatorOutput(EmulatorOutput):
    """A simple class that makes it easier to access the corrected emulator output."""

    Tb: np.ndarray
    xHI: np.ndarray
    Ts: np.ndarray
    PS: np.ndarray
    tau: np.ndarray
    UVLFs: np.ndarray

    properties = emulator_properties(emulator="default")

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

    properties = emulator_properties(emulator="radio_background")

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
    properties = emulator_properties(emulator="default")

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

    properties = emulator_properties(emulator="radio_background")

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
