"""Radio-background + scattering-dark-matter emulators with and without radio heating.

Two emulators share the architecture in :mod:`.model`:

- ``rh``: with soft-photon (radio) heating of the gas
- ``norh``: the same model without radio heating

Each has a ``<name>_constants.npz`` file here (normalisation constants, redshifts,
parameter limits, network configuration, validation errors and the 21cmFAST
settings of the training database). The trained weights ``<name>_weights.pt``
are distributed separately (see :func:`.model.weights_search_paths`).
"""

from .model import MLP, LSTMEmulator, SummaryEmulator, find_weights, load_model

__all__ = ["MLP", "LSTMEmulator", "SummaryEmulator", "find_weights", "load_model"]
