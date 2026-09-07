# DSAWACGA_v5: WaveletModulatedDSA local + FocusedLinearAttention global
#              + SE CA + StarFFN
#
# Key design vs CSFH:
#   Local:  WaveletModulatedDSA (DSA + wavelet modulation)
#           vs CSFH's 2-branch DilatedConv
#   Global: FocusedLinearAttention (FLatten: linear attention + focusing)
#           vs FourierUnit (FFT spectral conv)
#   CA:     SE (same)
#   FFN:    StarFFN (Star Operation with multi-scale DW gate)
#           vs CSFH's Split4 [id, DW3, DW5, DW7]

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import C2f
from .dsawacga_v2 import LearnableHaarDWT, WaveletModulation

__all__ = ['DSA', 'WaveletModulatedDSA', 'FocusedLinearAttention',
           'TokenMixer_For_Global',
           'DSAWACGA_v5_Mixer', 'StarFFN',
           'DSAWACGAv5Block', 'C2f_DSAWACGA_v5']


# ============================================================
# DSA: 6-branch multi-scale DWConv with softmax weighting
# ============================================================

class DSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.dwconv1 = nn.Conv2d(dim, dim, 1, groups=dim, bias=False)
        self.dwconv3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.dwconv5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.dwconv7 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.dwconv9 = nn.Conv2d(dim, dim, 9, padding=4, groups=dim, bias=False)
        self.dwconv11 = nn.Conv2d(dim, dim, 11, padding=5, groups=dim, bias=False)

        self.weight_conv = nn.Conv2d(dim * 6, 6, 1, bias=False)
        self.channel_mix = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        f0 = self.dwconv1(x)
        f1 = self.dwconv3(x)
        f2 = self.dwconv5(x)
        f3 = self.dwconv7(x)
        f4 = self.dwconv9(x)
        f5 = self.dwconv11(x)

        spatial_cat = torch.cat([f0, f1, f2, f3, f4, f5], dim=1)
        spatial_weights = F.softmax(self.weight_conv(spatial_cat), dim=1)

        out = (spatial_weights[:, 0:1] * f0 +
               spatial_weights[:, 1:2] * f1 +
               spatial_weights[:, 2:3] * f2 +
               spatial_weights[:, 3:4] * f3 +
               spatial_weights[:, 4:5] * f4 +
               spatial_weights[:, 5:6] * f5)
        out = self.channel_mix(out)

        return out


# ============================================================
# WaveletModulatedDSA: DSA + WaveletModulation
# ============================================================

class WaveletModulatedDSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dsa = DSA(dim)
        self.wavelet_mod = WaveletModulation(dim, bias=True)

    def forward(self, x):
        fdsa = self.dsa(x)
        wavelet_attn = self.wavelet_mod(x)
        return wavelet_attn * fdsa + fdsa


# ============================================================
# FocusedLinearAttention: FLatten Transformer global attention
#   (FLatten: Vision Transformer using Focused Linear Attention, ICCV 2023)
#
#   Core: φ(Q)(φ(K)ᵀV) / φ(Q)(φ(K)ᵀ1)  — O(Nd²) vs O(N²d)
#   Focusing: φ(x) = ReLU(x)^f sharpens attention distribution
#   DWC on V: incorporates local structural info
# ============================================================

class FocusedLinearAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.focusing_factor = focusing_factor
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(in_channels, out_channels * 3, 1)
        self.dwc = nn.Conv2d(out_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, groups=out_channels)
        self.proj = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        v = v + self.dwc(v)

        q = q.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        q = F.relu(q) ** self.focusing_factor + 1e-6
        k = F.relu(k) ** self.focusing_factor + 1e-6

        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        qkv = torch.einsum('bhnd,bhde->bhne', q, kv)

        k_sum = k.sum(dim=2)
        normalizer = torch.einsum('bhnd,bhd->bhn', q, k_sum).unsqueeze(-1)
        output = qkv / (normalizer + 1e-6)

        output = output.permute(0, 1, 3, 2).reshape(B, self.out_channels, H, W)
        output = self.proj(output)
        return output


# ============================================================
# TokenMixer_For_Global: FocusedLinearAttention + identity residual
# ============================================================

class TokenMixer_For_Global(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.FFC = FocusedLinearAttention(dim, dim)

    def forward(self, x):
        return x + self.FFC(x)


# ============================================================
# DSAWACGA_v5_Mixer: WaveletModulatedDSA + FocusedLinearAttention + SE
# ============================================================

class DSAWACGA_v5_Mixer(nn.Module):
    """DSAWACGA v5 Mixer: WaveletModulatedDSA + FocusedLinearAttention + SE.

    vs SFHF_Mixer (CSFH):
      - Local:  WaveletModulatedDSA vs DilatedConv(d=1,2)
      - Global: FocusedLinearAttention (FLatten) vs FourierUnit
      - CA:     SE (same)
      - Overall: 2/4 components different from CSFH

    Args:
        dim: Number of input/output channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.conv_init = nn.Conv2d(dim, dim * 2, 1)

        self.local_mixer = WaveletModulatedDSA(dim)
        self.global_mixer = TokenMixer_For_Global(dim)

        self.gelu = nn.GELU()

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.ca_conv = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        x = self.conv_init(x)
        x_local, x_global = x.chunk(2, dim=1)
        x_local = self.local_mixer(x_local)
        x_global = self.global_mixer(x_global)
        x = torch.cat([x_local, x_global], dim=1)
        x = self.gelu(x)
        x = x * self.ca(x)
        x = self.ca_conv(x)
        return x


# ============================================================
# StarFFN: Star Operation FFN with multi-scale spatial gate
# ============================================================

class StarFFN(nn.Module):
    """Star Operation FFN with multi-scale spatial gate.

    Architecture:
      fc1(dim → 2dim) → split(x1, x2)
        x1 → split(x1a, x1b) → [DW3(x1a), DW5(x1b)] → cat → x1'
        x2 unchanged
      GELU(x1') * x2 → fc2(dim → dim) + residual

    x1 path: multi-scale DWConv provides spatial-aware gate signal
    x2 path: keeps original info as gated content
    Star: spatial-aware gate × content = position-adaptive channel transform

    vs CSFH SFHF_FFN (Split4[id,DW3,DW5,DW7]+Cat+GELU):
      - Nonlinearity: polynomial kernel (Star) vs piecewise-linear (GELU)
      - Channel interaction: multiplicative (*) vs additive (Cat)
      - x2 branch: untouched (implicit residual) vs DW7 processed
      - Multi-scale: on gate path only vs on all branches

    Args:
        dim: Number of input/output channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.dw3 = nn.Conv2d(dim // 2, dim // 2, 3, padding=1,
                              groups=dim // 2, bias=False)
        self.dw5 = nn.Conv2d(dim // 2, dim // 2, 5, padding=2,
                              groups=dim // 2, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h = self.fc1(x)
        x1, x2 = h.chunk(2, dim=1)
        x1a, x1b = x1.chunk(2, dim=1)
        x1 = torch.cat([self.dw3(x1a), self.dw5(x1b)], dim=1)
        star = self.act(x1) * x2
        return x + self.fc2(star)


# ============================================================
# DSAWACGAv5Block: Transformer-like two-stage block
# ============================================================

class DSAWACGAv5Block(nn.Module):
    """DSAWACGA v5 Block: Mixer + FFN with LayerScale.

    Stage 1 (Attention):  x' = DSAWACGA_v5_Mixer(Norm1(x)) * β + x
    Stage 2 (FFN):        x'' = StarFFN(Norm2(x')) * γ + x'

    Args:
        dim: Number of input/output channels.
        norm_layer: Normalization layer (default BatchNorm2d).
    """

    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim

        self.norm1 = norm_layer(dim)
        self.mixer = DSAWACGA_v5_Mixer(dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

        self.norm2 = norm_layer(dim)
        self.ffn = StarFFN(dim)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = self.mixer(self.norm1(x)) * self.beta + x
        x = self.ffn(self.norm2(x)) * self.gamma + x
        return x


# ============================================================
# C2f_DSAWACGA_v5: C2f container with DSAWACGAv5Block
# ============================================================

class C2f_DSAWACGA_v5(C2f):
    """C2f with DSAWACGAv5Block replacing Bottleneck.

    cv1(c1→2c, 1×1) → split(2 chunks) → [DSAWACGAv5Block(c)×n] → cat → cv2(→c2, 1×1)

    DSAWACGAv5Block:
      Stage 1: DSAWACGA_v5_Mixer (WaveletModulatedDSA + FocusedLinearAttention + SE)
      Stage 2: StarFFN (Star Operation with multi-scale DW gate)

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of DSAWACGAv5Block repeats.
        shortcut: Whether to use shortcut connection (unused).
        g: Groups (unused, kept for C2f interface).
        e: Expansion ratio for hidden channels (default 0.5).
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DSAWACGAv5Block(self.c) for _ in range(n))