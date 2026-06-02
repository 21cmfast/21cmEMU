"""
Neural-network modules for the 21cm MH emulator (v3).

This module defines the HybridEmulator architecture used for inference.
Architecture matches the production_ema training checkpoint.
"""

import torch
import torch.nn as nn


class _LSTMHead(nn.Module):
    """Single LSTM head producing a 1-D sequence.

    Processes an input sequence of length ``seq_len`` through a
    multi-layer LSTM and projects each hidden state to ``output_dim``
    values, yielding ``(B, seq_len, output_dim)``.

    When ``output_dim=2`` (neutral Ts mode) the two channels are:
      - channel 0: predicted value
      - channel 1: validity logit (apply sigmoid → P(defined))
    """

    def __init__(
        self, input_size, hidden_size, num_layers=2, output_dim=1, dropout=0.0
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.lin = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)  # (B, seq_len, hidden)
        output = self.lin(output)  # (B, seq_len, output_dim)
        return output


class LSTM2DHead(nn.Module):
    """Generic two-pass LSTM decoder for any 2-D grid output.

    Works for both PS (z × k) and UVLFs (z × M_UV).

    1. **Row-LSTM** – Input is ``(B, n_rows, input_dim + 1)`` where the
       ``+1`` is a normalised row coordinate.
    2. **Col-LSTM** – Input is ``(B*n_rows, n_cols, hidden_row + 1)``
       where the ``+1`` is a normalised column coordinate.
    """

    def __init__(
        self,
        input_dim,
        n_rows,
        n_cols,
        *,
        hidden_row=256,
        hidden_col=128,
        num_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.hidden_row = hidden_row

        # Register buffers (will be populated by load_state_dict or set_axes)
        self.register_buffer("_row_norm", torch.zeros(n_rows))
        self.register_buffer("_col_norm", torch.zeros(n_cols))

        # Row-LSTM: input = (features, row_coord)  → input_dim + 1
        self.row_lstm = nn.LSTM(
            input_size=input_dim + 1,
            hidden_size=hidden_row,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Col-LSTM: input = (row_context, col_coord) → hidden_row + 1
        self.col_lstm = nn.LSTM(
            input_size=hidden_row + 1,
            hidden_size=hidden_col,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.col_proj = nn.Linear(hidden_col, 1)

    def set_axes(self, row_values, col_values, log_col=False):
        """Register normalised row and column coordinate buffers."""
        dev = next(self.parameters()).device

        r = torch.as_tensor(row_values, dtype=torch.float32)
        r_norm = (r - r.min()) / (r.max() - r.min())
        self.register_buffer("_row_norm", r_norm.to(dev))

        c = torch.as_tensor(col_values, dtype=torch.float32)
        if log_col:
            c = torch.log10(c)
        c_norm = (c - c.min()) / (c.max() - c.min())
        self.register_buffer("_col_norm", c_norm.to(dev))

    def forward(self, x):
        """x: (B, input_dim) — raw theta."""
        B = x.size(0)

        # ── Row-LSTM input: (B, n_rows, input_dim + 1) ───────
        tile = x.unsqueeze(1).expand(B, self.n_rows, -1)
        row_col = self._row_norm.unsqueeze(0).expand(B, -1).unsqueeze(-1)
        row_seq = torch.cat([tile, row_col], dim=-1)

        row_out, _ = self.row_lstm(row_seq)  # (B, n_rows, hidden_row)

        # ── Col-LSTM input: (B*n_rows, n_cols, hidden_row + 1)
        row_ctx = row_out.unsqueeze(2).expand(
            B, self.n_rows, self.n_cols, self.hidden_row
        )
        row_ctx = row_ctx.reshape(B * self.n_rows, self.n_cols, self.hidden_row)

        col_coord = (
            self._col_norm.unsqueeze(0).expand(B * self.n_rows, -1).unsqueeze(-1)
        )
        col_input = torch.cat([row_ctx, col_coord], dim=-1)

        col_out, _ = self.col_lstm(col_input)  # (B*n_rows, n_cols, hidden_col)
        col_vals = self.col_proj(col_out).squeeze(-1)  # (B*n_rows, n_cols)
        return col_vals.reshape(B, self.n_rows, self.n_cols)


class FeedForwardHead(nn.Module):
    """Simple MLP for scalar output (tau_e)."""

    def __init__(self, d_model, hidden=256, n_layers=4):
        super().__init__()
        layers = [nn.Linear(d_model, hidden), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, d_model) — flat theta."""
        return self.net(x).squeeze(-1)


class MH_Emulator(nn.Module):
    """
    Hybrid emulator:  theta → (xHI, Tb, Ts_neutral, UVLFs, PS, tau).

    This is the production architecture matching the trained checkpoint.
    All heads receive raw params directly (no encoder).

    Parameters
    ----------
    params_dict : dict
        Must contain:
        - 'n_params': int (11 for the 21cm emulator)
        - 'N_z': int (93 redshift bins)
        - 'N_LF_z': int (7 UVLF redshift bins)
        - 'N_mag': int (45 magnitude bins)
        - 'N_PS_Z': int (32 PS redshift bins)
        - 'N_PS_K': int (32 PS k bins)
    """

    def __init__(self, params_dict):
        super().__init__()

        n_params = params_dict.get("n_params", 11)
        N_z = params_dict.get("N_z", 93)
        N_LF_z = params_dict.get("N_LF_z", 7)
        N_mag = params_dict.get("N_mag", 45)
        N_PS_Z = params_dict.get("N_PS_Z", 32)
        N_PS_K = params_dict.get("N_PS_K", 32)

        self._n_params = n_params
        self._n_z = N_z

        # Register buffer for redshift normalization (populated by load_state_dict)
        self.register_buffer("_z_norm", torch.zeros(N_z))

        # Input dim for 1D LSTM heads: n_params + 1 (z coordinate)
        lstm_1d_in = n_params + 1

        # 1-D LSTM heads
        self.xhi_head = _LSTMHead(lstm_1d_in, hidden_size=N_z, num_layers=2)
        self.tb_head = _LSTMHead(lstm_1d_in, hidden_size=N_z, num_layers=2)
        self.ts_head = _LSTMHead(
            lstm_1d_in, hidden_size=N_z, num_layers=2, output_dim=2
        )

        # 2-D LSTM heads
        self.uvlf_head = LSTM2DHead(
            n_params,
            n_rows=N_LF_z,
            n_cols=N_mag,
            hidden_row=256,
            hidden_col=128,
            num_layers=2,
        )

        self.ps_head = LSTM2DHead(
            n_params,
            n_rows=N_PS_Z,
            n_cols=N_PS_K,
            hidden_row=256,
            hidden_col=128,
            num_layers=2,
        )

        # tau head
        self.tau_head = FeedForwardHead(n_params, hidden=256, n_layers=4)

    def set_redshifts(self, z):
        """Register normalised redshift vector as a buffer."""
        z_t = torch.as_tensor(z, dtype=torch.float32)
        z_norm = (z_t - z_t.min()) / (z_t.max() - z_t.min())
        dev = next(self.parameters()).device
        self.register_buffer("_z_norm", z_norm.to(dev))

    def set_ps_axes(self, ps_redshifts, ps_k):
        """Register PS axes."""
        self.ps_head.set_axes(ps_redshifts, ps_k, log_col=True)

    def set_uvlf_axes(self, lf_redshifts, lf_mag):
        """Register UVLF axes."""
        self.uvlf_head.set_axes(lf_redshifts, lf_mag, log_col=False)

    def _build_direct_sequence(self, theta):
        """Tile flat params across redshift steps and append normalised z.

        Returns (B, N_z, n_params + 1).
        """
        B = theta.size(0)
        tile = theta.unsqueeze(1).expand(B, self._n_z, -1)
        z_col = self._z_norm.unsqueeze(0).expand(B, -1).unsqueeze(-1)
        return torch.cat([tile, z_col], dim=-1)

    def forward(self, theta):
        """
        Parameters
        ----------
        theta : (B, n_params)

        Returns
        -------
        xHI           : (B, N_z)
        Tb            : (B, N_z)
        Ts            : (B, N_z, 2)   – ch0=Ts_neutral value, ch1=validity logit
        UVLFs         : (B, N_mag, N_LF_z)
        PS            : (B, N_PS_Z, N_PS_K)
        tau           : (B,)
        """
        # Build LSTM input sequence for 1-D heads
        seq = self._build_direct_sequence(theta)  # (B, N_z, n_params+1)

        # xHI (with sigmoid activation)
        xhi = self.xhi_head(seq).squeeze(-1)
        xhi = torch.sigmoid(xhi)

        # Tb
        tb = self.tb_head(seq).squeeze(-1)

        # Ts (neutral only, 2-channel output)
        ts = self.ts_head(seq)  # (B, N_z, 2)

        # UVLFs
        uvlf = self.uvlf_head(theta)  # (B, N_LF_z, N_mag)
        uvlf = uvlf.transpose(1, 2)  # → (B, N_mag, N_LF_z)

        # tau
        tau = self.tau_head(theta)

        # PS
        ps = self.ps_head(theta)  # (B, N_PS_Z, N_PS_K)

        return (
            torch.flip(xhi, dims=(-1,)),
            torch.flip(tb, dims=(-1,)),
            torch.flip(ts, dims=(-2,)),
            uvlf,
            ps,
            tau,
        )
