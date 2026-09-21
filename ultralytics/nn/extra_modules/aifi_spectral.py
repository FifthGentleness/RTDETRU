# Spectral-Enhanced AIFI (AIFI_SET)
#
# Based on "SET: Spectral Enhancement for Tiny Object Detection" (CVF Open Access).
# This module integrates a Dynamic Spectral Filter branch into the AIFI
# self-attention pipeline so that Query/Key are driven by spectrally
# purified features (high SNR small-object responses) while Value retains
# the original features to preserve large-object semantics.
#
# Key design decisions (vs initial version):
#   1. 3x3 ComplexConv2d in freq domain: provides spatial-local smoothing
#      across frequency bins, enabling band-aware filtering (1x1 conv only
#      mixes channels, leaving each freq point independent).
#   2. Sigmoid on real part (not magnitude): gives full [0, 1] dynamic range
#      so the mask can fully suppress noise (sigmoid(|z|) >= 0.5 always).
#   3. Positional embedding only on Q/K, NOT on V: standard Transformer
#      semantics — pos provides spatial addressing for attention weights;
#      adding it to V would inject fixed sinusoidal patterns into features.
#   4. Cached positional embedding: avoids redundant recomputation when
#      input resolution is fixed (common in detection pipelines).
#
# Architecture overview:
#   X (B, C, H, W)
#   ├── [频域解耦分支]
#   │   2D rFFT
#   │   ComplexConv2d(3x3) → σ(real) as gain mask ∈ [0, 1]
#   │   Hadamard product in freq domain
#   │   2D irFFT
#   │   Sigmoid gating: G = σ(Conv1x1(X_spectral))
#   └── X_purified = X ⊙ G
#         │
#         ├── Q = X_purified·W_Q + pos,  K = X_purified·W_K + pos
#         ├── V = X·W_V  (no pos, preserves raw semantics)
#         │
#         └── Attn_Out = Softmax(QK^T/√d)·V
#               │
#               └── Output = LayerNorm(X + Attn_Out + FFN)

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DynamicSpectralFilter', 'AIFI_SET']


class ComplexConv2d(nn.Module):
    """Complex-valued 2D convolution operating on real/imag pairs.

    Implements (W_r + jW_i) * (X_r + jX_i) by expanding into four real
    convolutions:
        real_out = conv_r(X_r) - conv_i(X_i)
        imag_out = conv_r(X_i) + conv_i(X_r)

    Using kernel_size=3 with padding=1 provides local smoothing across
    neighboring frequency bins, which is essential for band-aware
    spectral filtering (1x1 conv leaves each freq point independent).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True):
        super().__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, bias=bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, bias=bias)

    def forward(self, x_real, x_imag):
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return out_real, out_imag


class DynamicSpectralFilter(nn.Module):
    """Dynamic frequency-domain filter with adaptive 2D spatial band modulation.

    Pipeline:
      1. 2D rFFT  →  complex spectrum (real, imag)
      2. ComplexConv2d(3x3) on spectrum  →  spatial-smooth freq response
      3. Sigmoid on real part as gain mask  →  full [0, 1] dynamic range
      4. Hadamard product:  X_freq ⊙ mask
      5. 2D irFFT  →  spatial feature X_spectral
      6. Sigmoid gating:  G = σ(Conv1x1(X_spectral))
      7. Purified feature:  X_purified = X ⊙ G

    Key fix vs initial version:
      - 3x3 ComplexConv2d (not 1x1): neighboring frequency bins interact,
        enabling smooth band-pass behavior instead of point-wise filtering.
      - Sigmoid on real part (not magnitude): sigmoid(|z|) ∈ [0.5, 1]
        cannot suppress noise; sigmoid(real) ∈ [0, 1] allows full attenuation.
      - No static band_prior: the 3x3 conv bias already serves as a
        learnable per-channel frequency offset, and the spatial kernel
        provides band-aware modulation that a (1,C,1,1) scalar cannot.

    Args:
        channels: Number of input channels C.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.complex_conv = ComplexConv2d(channels, channels, kernel_size=3,
                                          padding=1, bias=True)

        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input feature map (B, C, H, W).

        Returns:
            x_purified: Gated purified feature (B, C, H, W).
        """
        B, C, H, W = x.shape

        x_freq = torch.fft.rfft2(x, norm='ortho')

        x_real = x_freq.real
        x_imag = x_freq.imag

        mask_real, mask_imag = self.complex_conv(x_real, x_imag)

        mask_gain = torch.sigmoid(mask_real)

        filtered_real = x_real * mask_gain
        filtered_imag = x_imag * mask_gain

        x_freq_filtered = torch.complex(filtered_real, filtered_imag)

        x_spectral = torch.fft.irfft2(x_freq_filtered, s=(H, W), norm='ortho')

        gate = torch.sigmoid(self.gate_conv(x_spectral))

        x_purified = x * gate
        return x_purified


class AIFI_SET(nn.Module):
    """Spectral-Enhanced AIFI for tiny object detection.

    Combines DynamicSpectralFilter with standard multi-head self-attention:
      - Q, K are computed from spectrally purified features + pos_embed
        (small-object salient regions dominate the attention matrix).
      - V is computed from the original features WITHOUT pos_embed
        (preserves large-object semantics and raw texture; adding pos to V
        would inject fixed sinusoidal patterns into aggregated features).
      - Residual connection + LayerNorm as in the original AIFI.

    Args:
        c1: Input/output channels.
        cm: FFN hidden dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        act: Activation for the FFN.
        normalize_before: Pre-norm if True; AIFI default is False (post-norm).
    """

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0,
                 act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before

        self.spectral_filter = DynamicSpectralFilter(c1)

        self.q_proj = nn.Linear(c1, c1)
        self.k_proj = nn.Linear(c1, c1)
        self.v_proj = nn.Linear(c1, c1)
        self.out_proj = nn.Linear(c1, c1)

        self.num_heads = num_heads
        self.head_dim = c1 // num_heads
        self.scale = self.head_dim ** -0.5

        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

        self.register_buffer('pos_embed_cache', None, persistent=False)
        self.cached_hw = None

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def get_pos_embed(self, H, W, device, dtype):
        """Get or compute cached 2D sine-cosine positional embedding.

        Avoids redundant tensor allocation when input resolution is fixed
        (the common case in detection pipelines where S5 is always e.g. 20x20).
        """
        if self.cached_hw == (H, W) and self.pos_embed_cache is not None:
            return self.pos_embed_cache

        grid_w = torch.arange(W, dtype=torch.float32, device=device)
        grid_h = torch.arange(H, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')

        assert self.c1 % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.c1 // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1.0 / (10000.0 ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        pos_embed = torch.cat([torch.sin(out_w), torch.cos(out_w),
                               torch.sin(out_h), torch.cos(out_h)],
                              dim=1)[None].to(dtype)

        self.pos_embed_cache = pos_embed
        self.cached_hw = (H, W)
        return pos_embed

    def _mhsa(self, q_src, k_src, v_src):
        """Multi-head self-attention with separate Q/K and V sources.

        Args:
            q_src: Features + pos for Query projection (B, N, C).
            k_src: Features + pos for Key projection (B, N, C).
            v_src: Pure features for Value projection (B, N, C), no pos.

        Returns:
            Attention output (B, N, C).
        """
        B, N, C = q_src.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(q_src).view(B, N, H, D).permute(0, 2, 1, 3)
        k = self.k_proj(k_src).view(B, N, H, D).permute(0, 2, 1, 3)
        v = self.v_proj(v_src).view(B, N, H, D).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)

    def forward(self, x):
        """Forward a feature map of shape (B, C, H, W).

        Steps:
          1. Dynamic spectral filtering  →  X_purified
          2. Flatten to (B, H*W, C)
          3. MHSA: Q,K = X_purified + pos; V = X (no pos)
          4. FFN with residual + LayerNorm (same as original AIFI)
        """
        B, C, H, W = x.shape

        x_purified = self.spectral_filter(x)

        pos_embed = self.get_pos_embed(H, W, x.device, x.dtype)

        src_orig = x.flatten(2).permute(0, 2, 1)
        src_purified = x_purified.flatten(2).permute(0, 2, 1)

        if self.normalize_before:
            src_purified_norm = self.norm1(src_purified)
            src_orig_norm = self.norm1(src_orig)

            src2 = self._mhsa(
                q_src=src_purified_norm + pos_embed,
                k_src=src_purified_norm + pos_embed,
                v_src=src_orig_norm,
            )
            src = src_orig + self.dropout1(src2)
            src_norm = self.norm2(src)
            src2 = self.fc2(self.dropout(self.act(self.fc1(src_norm))))
            src = src + self.dropout2(src2)
        else:
            src2 = self._mhsa(
                q_src=src_purified + pos_embed,
                k_src=src_purified + pos_embed,
                v_src=src_orig,
            )
            src = src_orig + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(B, C, H, W).contiguous()