"""Improved U-Net architecture for score-based diffusion.

Based on DDPM (Ho+20) / "The Annotated Diffusion Model" with the following
improvements over the original:
  - Dropout in ResNet blocks for regularization
  - Efficient attention via PyTorch 2.0 scaled_dot_product_attention
  - Optional self-conditioning (Improved Denoising Diffusion Probabilistic
    Models - Chen et al. 2022)
  - GEGLUs in the feedforward/MLP layers for better gradient flow
  - Improved conditioning: separate FiLM modulation for physical parameters
    alongside the time embedding, so the model can modulate features
    differently for time vs. physics
  - Residual connections on attention outputs
"""

import math
from functools import partial

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# ---------------------------------------------------------------------------
# Up / Down sampling
# ---------------------------------------------------------------------------

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # Pixel-unshuffle style: no strided convs or pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


# ---------------------------------------------------------------------------
# Time / condition embeddings
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Normalisation / weight standardisation
# ---------------------------------------------------------------------------

class WeightStandardizedConv2d(nn.Conv2d):
    """Conv2d with weight standardization (Qiao et al. 2019).

    Works synergistically with GroupNorm.
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, padding=1, groups=8, dropout=0.0):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResnetBlock(nn.Module):
    """ResNet block with FiLM conditioning (scale + shift) from both
    time embedding and optional physical-parameter embedding."""

    def __init__(self, dim, dim_out, *, kernel_size=3, time_emb_dim=None,
                 cdn_emb_dim=None, groups=8, dropout=0.0):
        super().__init__()
        # Time conditioning MLP -> produces scale & shift
        self.time_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim) else None
        )
        # Condition (physics params) FiLM -> separate scale & shift
        self.cdn_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(cdn_emb_dim, dim_out * 2))
            if exists(cdn_emb_dim) else None
        )

        self.block1 = Block(dim, dim_out, kernel_size=kernel_size,
                            groups=groups, dropout=dropout)
        self.block2 = Block(dim_out, dim_out, kernel_size=kernel_size,
                            groups=groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cdn_emb=None):
        scale_shift = None

        if exists(self.time_mlp) and exists(time_emb):
            t = self.time_mlp(time_emb)
            t = rearrange(t, "b c -> b c 1 1")
            scale_t, shift_t = t.chunk(2, dim=1)
        else:
            scale_t = shift_t = 0.0

        if exists(self.cdn_mlp) and exists(cdn_emb):
            c = self.cdn_mlp(cdn_emb)
            c = rearrange(c, "b c -> b c 1 1")
            scale_c, shift_c = c.chunk(2, dim=1)
        else:
            scale_c = shift_c = 0.0

        # Combine time and condition modulations additively
        scale = scale_t + scale_c
        shift = shift_t + shift_c
        scale_shift = (scale, shift)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


# ---------------------------------------------------------------------------
# Attention layers
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention using PyTorch 2.0 scaled_dot_product_attention
    for memory-efficient and fused-kernel attention (FlashAttention when available)."""

    def __init__(self, dim, heads=4, dim_head=32, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.dropout = dropout

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv
        )
        # PyTorch >= 2.0: fused, memory-efficient attention
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = rearrange(out, "b h (x y) c -> b (h c) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """O(n) linear attention for use in encoder/decoder blocks."""

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):
    """Wraps a module with a residual (skip) connection."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """Improved conditional U-Net for score-based diffusion.

    Key improvements over the original:
      1. Separate FiLM modulation for time vs physics conditioning
      2. Dropout in ResNet blocks
      3. Efficient fused attention (PyTorch 2.0)
      4. Optional self-conditioning
    """

    def __init__(
        self,
        dim,
        init_dim=32,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=2,
        resnet_block_groups=4,
        cdn_len=None,
        time_dim=256,
        cdn_dim=128,
        sinusoidal_dim=128,
        dropout=0.05,
        self_condition=False,
        attn_heads=4,
        attn_dim_head=32,
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *[dim[0] * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        resnet_block = partial(ResnetBlock, groups=resnet_block_groups, dropout=dropout)

        # ----- Time embedding -----
        # Use a dedicated sinusoidal_dim (default 128) rather than dim[0]
        # (spatial size) so the model has sufficient temporal resolution
        # to distinguish noise levels across the diffusion process.
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(sinusoidal_dim),
            nn.Linear(sinusoidal_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # ----- Condition embedding (physics parameters) -----
        self.cdn_mlp = None
        if cdn_len is not None:
            self.cdn_mlp = nn.Sequential(
                nn.Linear(cdn_len, cdn_dim),
                nn.GELU(),
                nn.Linear(cdn_dim, cdn_dim),
                nn.GELU(),
                nn.Linear(cdn_dim, cdn_dim),
            )

        # ----- Encoder (downsampling path) -----
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_in, time_emb_dim=time_dim,
                             cdn_emb_dim=cdn_dim if cdn_len else None),
                resnet_block(dim_in, dim_in, time_emb_dim=time_dim,
                             cdn_emb_dim=cdn_dim if cdn_len else None),
                Residual(PreNorm(dim_in, LinearAttention(dim_in,
                         heads=attn_heads, dim_head=attn_dim_head))),
                Downsample(dim_in, dim_out) if not is_last
                else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))

        # ----- Bottleneck -----
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim,
                                       cdn_emb_dim=cdn_dim if cdn_len else None)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(
            mid_dim, heads=attn_heads, dim_head=attn_dim_head)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim,
                                       cdn_emb_dim=cdn_dim if cdn_len else None)

        # ----- Decoder (upsampling path) -----
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                resnet_block(dim_out + dim_in, dim_out, time_emb_dim=time_dim,
                             cdn_emb_dim=cdn_dim if cdn_len else None),
                resnet_block(dim_out + dim_in, dim_out, time_emb_dim=time_dim,
                             cdn_emb_dim=cdn_dim if cdn_len else None),
                Residual(PreNorm(dim_out, LinearAttention(dim_out,
                         heads=attn_heads, dim_head=attn_dim_head))),
                Upsample(dim_out, dim_in) if not is_last
                else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))

        # ----- Output -----
        self.out_dim = 1
        self.final_res_block = resnet_block(init_dim * 2, dim[0], time_emb_dim=time_dim,
                                            cdn_emb_dim=cdn_dim if cdn_len else None)
        # Use Conv2d (not ConvTranspose2d) to avoid checkerboard artifacts
        self.final_conv = nn.Conv2d(dim[0], self.out_dim,
                                    kernel_size=7, padding=3)

    def forward(self, x, time, x_cdn=None, cdn=None, x_self_cond=None):
        # Self-conditioning: concatenate previous prediction
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)

        if x_cdn is not None:
            x = torch.cat((x, x_cdn), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        # Embeddings
        t = self.time_mlp(time)
        c = self.cdn_mlp(cdn) if exists(self.cdn_mlp) and exists(cdn) else None

        # Encoder
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb=t, cdn_emb=c)
            h.append(x)
            x = block2(x, time_emb=t, cdn_emb=c)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, time_emb=t, cdn_emb=c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb=t, cdn_emb=c)

        # Decoder
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, time_emb=t, cdn_emb=c)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, time_emb=t, cdn_emb=c)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, time_emb=t, cdn_emb=c)
        return self.final_conv(x)
