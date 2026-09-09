# DSAWACGA_v7: WaveletModulatedDSA local + Frequency_Convolution global
#              + SE CA + StarFFN
#
# Key design vs v5:
#   Local:  WaveletModulatedDSA (same as v5)
#   Global: Frequency_Convolution (FourierSR: block-diagonal spectral conv
#           + complex activation + softshrink sparsification)
#           replaces v5's FocusedLinearAttention (spatial linear attention)
#   CA:     SE (same as v5)
#   FFN:    StarFFN (same as v5)
#
# vs CSFH:
#   Local:  WaveletModulatedDSA vs DilatedConv(d=1,2)
#   Global: Frequency_Convolution vs FourierUnit
#           Both use FFT, but different parameterization:
#           - FourierUnit: BN + FPE + grouped conv + softmax on 2×dim
#           - Frequency_Convolution: block-diagonal complex linear on dim
#             + complex ReLU + softshrink (explicit spectral filtering)
#   CA:     SE (same)
#   FFN:    StarFFN vs Split4[id,DW3,DW5,DW7]
#
# Frequency_Convolution advantages over FourierUnit:
#   1. Block-diagonal: parameter efficient, preserves complex structure
#   2. Complex activation: physically meaningful (separate real/imag)
#   3. Softshrink: explicit spectral sparsification (removes noise freq)
#   4. Residual: stable training

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import C2f
from .dsawacga_v2 import LearnableHaarDWT, WaveletModulation

__all__ = ['DSA', 'WaveletModulatedDSA', 'Frequency_Convolution',
           'TokenMixer_For_Global',
           'DSAWACGA_v7_Mixer', 'StarFFN',
           'DSAWACGAv7Block', 'C2f_DSAWACGA_v7']


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
# Frequency_Convolution: Adaptive Fourier Neural Operator (AFNO)
#   Adapted from FourierSR for object detection
#
# Core idea:
#   1. FFT: transform to frequency domain
#   2. Block-diagonal complex linear: parameter-efficient spectral conv
#      weight [num_blocks, block_size, block_size, 2] (real+imag)
#      einsum('bkihw,kio->bkohw', x, weight) — complex matmul per block
#   3. Complex activation: separate real/imag ReLU with learnable weights
#      o_real = ReLU(w1_r * x.real - w1_i * x.imag + b_r)
#      o_imag = ReLU(w2_r * x.imag + w2_i * x.real + b_i)
#   4. Softshrink: explicit spectral sparsification
#      removes small frequency components (noise filtering)
#   5. IFFT: transform back to spatial domain
#   6. Residual: output + input
#
# vs CSFH FourierUnit:
#   - Both operate in Fourier domain
#   - FourierUnit: BN + FPE + grouped conv on 2×dim (real+imag concat)
#   - Frequency_Convolution: block-diagonal complex linear on dim
#     + complex activation + softshrink
#   - Frequency_Convolution is more parameter-efficient
#   - Frequency_Convolution has explicit spectral filtering (softshrink)
#   - FourierUnit has dynamic weighting (softmax)
#
# Args:
#   channels: number of input/output channels
#   num_blocks: number of blocks for block-diagonal structure
#     (higher = less params, less expressiveness per block)
#   sparsity_threshold: lambda for softshrink
#     (higher = more aggressive spectral sparsification)
# ============================================================

class Frequency_Convolution(nn.Module):
    def __init__(self, channels, num_blocks=8, sparsity_threshold=0.01):
        super().__init__()
        assert channels % num_blocks == 0, \
            f"channels {channels} must be divisible by num_blocks {num_blocks}"

        self.channels = channels
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = channels // num_blocks
        self.scale = 0.02

        self.w = nn.Parameter(
            self.scale * torch.randn(self.num_blocks, self.block_size,
                                     self.block_size, 2))
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        self.b = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])

        weight = torch.view_as_complex(self.w.contiguous())
        x = torch.einsum('bkihw,kio->bkohw', x, weight)

        o1_real = F.relu(
            torch.mul(x.real, self.w1[0].unsqueeze(dim=0)) -
            torch.mul(x.imag, self.w1[1].unsqueeze(dim=0)) +
            self.b[0, :, :, None, None]
        )

        o1_imag = F.relu(
            torch.mul(x.imag, self.w2[0].unsqueeze(dim=0)) +
            torch.mul(x.real, self.w2[1].unsqueeze(dim=0)) +
            self.b[1, :, :, None, None]
        )

        x = torch.stack([o1_real, o1_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, x.shape[3], x.shape[4])

        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)

        return x + bias


# ============================================================
# TokenMixer_For_Global: Frequency_Convolution (spectral global)
#
# vs v5's TokenMixer_For_Global (FocusedLinearAttention):
#   - v5: spatial linear attention — no frequency domain
#   - v7: spectral convolution — explicit frequency domain modeling
#   - v7 recovers the frequency domain processing that v5 lacked
#     (which was the key advantage of CSFH over v5)
# ============================================================

class TokenMixer_For_Global(nn.Module):
    def __init__(self, dim, num_blocks=8, sparsity_threshold=0.01):
        super().__init__()
        self.FFC = Frequency_Convolution(
            dim, num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold)

    def forward(self, x):
        return self.FFC(x)


# ============================================================
# DSAWACGA_v7_Mixer: WaveletModulatedDSA + Frequency_Convolution + SE
#
#   conv_init(dim→2dim) → chunk(2)
#     Local:  WaveletModulatedDSA(x_local)  ← DWT-guided multi-scale local
#     Global: Frequency_Convolution(x_global) ← block-diagonal spectral conv
#   Cat → GELU → SE_CA → ca_conv
# ============================================================

class DSAWACGA_v7_Mixer(nn.Module):
    def __init__(self, dim, num_blocks=8, sparsity_threshold=0.01):
        super().__init__()
        self.dim = dim

        self.conv_init = nn.Conv2d(dim, dim * 2, 1)

        self.local_mixer = WaveletModulatedDSA(dim)
        self.global_mixer = TokenMixer_For_Global(
            dim, num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold)

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
# DSAWACGAv7Block: Transformer-like two-stage block
# ============================================================

class DSAWACGAv7Block(nn.Module):
    def __init__(self, dim, num_blocks=8, sparsity_threshold=0.01,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim

        self.norm1 = norm_layer(dim)
        self.mixer = DSAWACGA_v7_Mixer(
            dim, num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

        self.norm2 = norm_layer(dim)
        self.ffn = StarFFN(dim)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = self.mixer(self.norm1(x)) * self.beta + x
        x = self.ffn(self.norm2(x)) * self.gamma + x
        return x


# ============================================================
# C2f_DSAWACGA_v7: C2f container with DSAWACGAv7Block
# ============================================================

class C2f_DSAWACGA_v7(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DSAWACGAv7Block(self.c) for _ in range(n))