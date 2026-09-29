"""Networks of the radio-heating emulators (``rh`` and ``norh``).

Both emulators share one architecture: an independent head per summary,

* :class:`LSTMEmulator` for the redshift series xHI, Tb and Tr, and
* :class:`MLP` for the optical depth tau,

collected in :class:`SummaryEmulator`. The networks take the min-max
normalised parameters ``theta`` (B, 6) and return *normalised* summaries;
the (de)normalisation is handled by
:class:`~py21cmemu.properties.RadioHeatingEmulatorProperties`.

The architecture is identical to the one used for training, so the saved
``state_dict`` files load directly.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
}
_FINAL = {None: nn.Identity, "none": nn.Identity, "sigmoid": nn.Sigmoid}


def _mlp_layers(
    n_in: int, n_out: int, hidden: int, n_layers: int, activation: str, dropout: float
) -> list[nn.Module]:
    """Return ``n_layers`` Linear layers (i.e. ``n_layers - 1`` hidden layers)."""
    act = _ACTIVATIONS[activation.lower()]
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if n_layers == 1:
        return [nn.Linear(n_in, n_out)]
    layers: list[nn.Module] = [nn.Linear(n_in, hidden), act()]
    for _ in range(n_layers - 2):
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden), act()]
    layers.append(nn.Linear(hidden, n_out))
    return layers


class MLP(nn.Module):
    """Fully connected net: theta (B, n_in) -> (B, n_out), or (B,) if ``n_out == 1``."""

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        hidden: int = 256,
        n_layers: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
        final_act: str | None = None,
    ):
        super().__init__()
        self.n_out = n_out
        self.net = nn.Sequential(
            *_mlp_layers(n_in, n_out, hidden, n_layers, activation, dropout),
            _FINAL[final_act](),
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the network."""
        out = self.net(theta)
        return out.squeeze(-1) if self.n_out == 1 else out


class LSTMEmulator(nn.Module):
    """LSTM mapping theta (B, n_params) to a redshift series (B, n_z).

    At every redshift step the LSTM receives ``[embed(theta), z_norm]``, where
    ``embed`` is a small MLP (skipped if ``embed_layers == 0``) and ``z_norm``
    is the redshift rescaled to [0, 1]. Each hidden state is projected to one
    value by a linear layer.

    ``direction='high_to_low'`` runs the LSTM from high to low redshift (forward
    in cosmic time); ``'low_to_high'`` the reverse; ``'bidirectional'`` both.
    The output is always ordered like the (increasing) redshift array.
    """

    def __init__(
        self,
        n_params: int,
        n_z: int,
        hidden: int = 128,
        num_layers: int = 2,
        embed_dim: int = 64,
        embed_layers: int = 2,
        embed_hidden: int = 128,
        activation: str = "gelu",
        dropout: float = 0.0,
        direction: str = "high_to_low",
        final_act: str | None = None,
    ):
        super().__init__()
        if direction not in ("high_to_low", "low_to_high", "bidirectional"):
            raise ValueError(f"Unknown direction {direction!r}")
        self.n_z = n_z
        self.direction = direction

        if embed_layers > 0:
            self.embed = nn.Sequential(
                *_mlp_layers(
                    n_params, embed_dim, embed_hidden, embed_layers, activation, 0.0
                ),
                _ACTIVATIONS[activation.lower()](),
            )
            lstm_in = embed_dim + 1
        else:
            self.embed = nn.Identity()
            lstm_in = n_params + 1

        bi = direction == "bidirectional"
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bi,
        )
        self.out = nn.Linear(hidden * (2 if bi else 1), 1)
        self.final = _FINAL[final_act]()
        # Normalised redshift fed at each step (stored in the state_dict).
        self.register_buffer("z_norm", torch.linspace(0.0, 1.0, n_z))

    def set_redshifts(self, z) -> None:
        """Set the redshift coordinate fed to the LSTM (rescaled to [0, 1])."""
        z = torch.as_tensor(z, dtype=torch.float32)
        if z.numel() != self.n_z:
            raise ValueError(f"Expected {self.n_z} redshifts, got {z.numel()}")
        z_norm = (z - z.min()) / (z.max() - z.min())
        self.z_norm.copy_(z_norm.to(self.z_norm.device))

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the network."""
        b = theta.shape[0]
        emb = self.embed(theta)  # (B, E)
        seq = torch.cat(
            [
                emb.unsqueeze(1).expand(b, self.n_z, emb.shape[-1]),
                self.z_norm.view(1, -1, 1).expand(b, self.n_z, 1),
            ],
            dim=-1,
        )  # (B, n_z, E + 1)
        # Redshifts are stored increasing, so 'high_to_low' means flip first.
        flip = self.direction == "high_to_low"
        if flip:
            seq = seq.flip(1)
        h, _ = self.lstm(seq)  # (B, n_z, H)
        y = self.out(h).squeeze(-1)  # (B, n_z)
        if flip:
            y = y.flip(1)
        return self.final(y)


class SummaryEmulator(nn.Module):
    """One independent head per summary.

    Parameters
    ----------
    n_params, n_z
        Number of input parameters and of redshift bins.
    targets
        Summaries to emulate, e.g. ``("xHI", "Tb", "Tr", "tau")``. ``tau`` gets
        an :class:`MLP`; every other summary an :class:`LSTMEmulator`.
    lstm_kwargs, mlp_kwargs
        Passed to the heads. ``xHI`` ends with a sigmoid unless
        ``sigmoid_xhi=False``.
    """

    SCALAR_TARGETS = ("tau",)

    def __init__(
        self,
        n_params: int,
        n_z: int,
        targets=("xHI", "Tb", "Tr", "tau"),
        lstm_kwargs: dict | None = None,
        mlp_kwargs: dict | None = None,
        sigmoid_xhi: bool = True,
    ):
        super().__init__()
        self.config = {
            "n_params": n_params,
            "n_z": n_z,
            "targets": list(targets),
            "lstm_kwargs": dict(lstm_kwargs or {}),
            "mlp_kwargs": dict(mlp_kwargs or {}),
            "sigmoid_xhi": sigmoid_xhi,
        }
        self.targets = tuple(targets)
        heads = {}
        for t in self.targets:
            if t in self.SCALAR_TARGETS:
                heads[t] = MLP(n_params, 1, **self.config["mlp_kwargs"])
            else:
                kw = dict(self.config["lstm_kwargs"])
                if t == "xHI" and sigmoid_xhi:
                    kw["final_act"] = "sigmoid"
                heads[t] = LSTMEmulator(n_params, n_z, **kw)
        self.heads = nn.ModuleDict(heads)

    @classmethod
    def from_config(cls, config: dict) -> SummaryEmulator:
        """Build the network from its configuration dictionary."""
        return cls(**config)

    def set_redshifts(self, z) -> None:
        """Set the redshifts of all LSTM heads."""
        for h in self.heads.values():
            if isinstance(h, LSTMEmulator):
                h.set_redshifts(z)

    def forward(self, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate all heads; returns a dict of normalised summaries."""
        return {t: head(theta) for t, head in self.heads.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Weights
# ══════════════════════════════════════════════════════════════════════════════


def weights_search_paths(name: str) -> list[Path]:
    """Locations searched (in order) for the weights of emulator ``name``.

    1. next to this file: ``models/radio_heating/<name>_weights.pt``
    2. the py21cmEMU data directory: ``<data-path>/radio_heating/<name>_weights.pt``
       (``data-path`` is set in the py21cmEMU config, see ``py21cmemu.config``)
    """
    from ...config import CONFIG

    fname = f"{name}_weights.pt"
    return [HERE / fname, Path(CONFIG.data_path) / "radio_heating" / fname]


def find_weights(name: str, weights_path: str | Path | None = None) -> Path:
    """Return the path of the weights file, raising a helpful error if missing."""
    if weights_path is not None:
        path = Path(weights_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Weights file {path} does not exist.")
        return path
    candidates = weights_search_paths(name)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No weights found for the '{name}' emulator. The weights are not part of "
        "the repository; put the file you received in one of\n"
        + "\n".join(f"  {p}" for p in candidates)
        + f"\nor pass Emulator(emulator='{name}', weights_path=...)."
    )


def load_model(
    name: str,
    config: dict,
    redshifts,
    weights_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> SummaryEmulator:
    """Build the network of emulator ``name`` and load its weights (eval mode)."""
    model = SummaryEmulator.from_config(config)
    model.set_redshifts(redshifts)
    path = find_weights(name, weights_path)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
