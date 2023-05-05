"""Output class."""
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

    Tb: np.ndarray
    xHI: np.ndarray
    Ts: np.ndarray
    PS: np.ndarray
    tau: np.ndarray
    UVLFs: np.ndarray

    properties = emulator_properties

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
    def k(self) -> np.ndarray:
        """The k-values of the power spectra."""
        return self.properties.ks_cut

    @property
    def Muv(self) -> np.ndarray:
        """The Muv-values of the UVLFs."""
        return self.properties.UVLFs_MUVs

    @property
    def UVLF_redshifts(self) -> np.ndarray:
        """The redshifts of the UVLFs."""
        return self.properties.uv_lf_zs

    @property
    def ps_redshifts(self) -> np.ndarray:
        """The redshifts of the power spectra."""
        return self.properties.zs_cut

    @property
    def redshifts(self) -> np.ndarray:
        """The redshifts of all quantities except the PS."""
        return self.properties.zs

    def squeeze(self):
        """Return a new EmulatorOutput with all dimensions of length 1 removed."""
        return EmulatorOutput(**{k: np.squeeze(v) for k, v in self.items()})

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
class RawEmulatorOutput:
    """A simple data-class that makes it easier to access the raw emulator output.

    Parameters
    ----------
    output : np.ndarray
        The raw output array from the emulator.
    """

    output: np.ndarray

    properties = emulator_properties

    @property
    def nz(self) -> int:
        """Number of redshifts in the output."""
        return np.array(self.properties.zs).shape[0]

    @property
    def nparams(self) -> int:
        """Number of sets of parameters in the output."""
        return self.output.shape[0]

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
        return self.output[:, self.nz * 3 + 1 + 60 * 12 + 1 :].reshape(
            (-1, len(self.properties.uv_lf_zs), len(self.properties.UVLFs_MUVs))
        )

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
            physical units.
        """
        # Restore dimensions
        # Renormalize stuff that needs renormalization
        renorm = {k: self.renormalize(k) for k in self.properties.normalized_quantities}

        other = {
            k.name: getattr(self, k.name)
            for k in dc.fields(EmulatorOutput)
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
        out["UVLFs"] = 10 ** out["UVLFs"]

        return EmulatorOutput(**out).squeeze()
