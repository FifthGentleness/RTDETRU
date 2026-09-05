# DSAWACGA_v4: CSFH-framework + WaveletModulatedDSA local + FourierUnit global.
#
# Architecture:
#   C2f_DSAWACGA_v4 = C2f container with DSAWACGAv4Block replacing Bottleneck:
#     cv1(c1→2c, 1×1) → split(2 chunks) → [DSAWACGAv4Block(c)×n] → cat → cv2(→c2, 1×1)
#
# DSAWACGAv4Block mirrors SFHF_Block's two-stage Transformer-like structure:
#   Stage 1 (Attention):  x' = DSAWACGA_v4_Mixer(Norm1(x)) * β + x
#   Stage 2 (FFN):        x'' = FFN(Norm2(x')) * γ + x'
#
# DSAWACGA_v4_Mixer:
#   conv_init(dim→2dim) → split(2 chunks, each dim)
#     ├─ chunk1 → WaveletModulatedDSA(dim)   ← local: DSA + WaveletModulation
#     │            fdsa = DSA(x)
#     │            wavelet_attn = WaveletModulation(x)  ∈ [0,1]
#     │            output = wavelet_attn ⊙ fdsa + fdsa  ≡ (1+A) ⊙ fdsa
#     └─ chunk2 → TokenMixer_For_Global(dim)  ← global: FourierUnit (from CSFH)
#   → cat → GELU → SE(2dim) → ca_conv(2dim→dim)
#
# Key design vs DSAWACGA_v3:
#   - Local branch: WaveletModulatedDSA replaces standalone DSA
#     - DSA produces multi-scale spatial features (content provider)
#     - WaveletModulation generates spatial-channel attention map (modulator)
#     - Modulation fusion: (1+A) ⊙ fdsa, tighter interaction than parallel cat
#   - No internal split in local branch: DSA operates on full dim channels
#     (v2 split dim into dim/2 for DSA + dim/2 for Conv3×3, wasting capacity)
#   - Conv3×3 branch removed: redundant with DSA's dwconv3 and FFN's channel mixing
#   - Global branch: FourierUnit (true global, H×W receptive field)
#     replaces WaveletGlobal (DWT level=1, only 2×2 receptive field)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..modules.block import C2f
from .dsawacga_v2 import LearnableHaarDWT, WaveletModulation

__all__ = ['DSA', 'WaveletModulatedDSA', 'DSAWACGA_v4_FourierUnit',
           'TokenMixer_For_Global', 'DSAWACGA_v4_Mixer',
           'DSAWACGA_v4_FFN', 'DSAWACGAv4Block', 'C2f_DSAWACGA_v4']


# ============================================================
# DSA: 6-branch multi-scale DWConv with softmax weighting
# ============================================================

class DSA(nn.Module):
    """Dynamic Scale Attention: 6-branch multi-scale DWConv with softmax weighting.

    Operates on full dim channels without internal split.

    Args:
        dim: Number of input/output channels.
    """

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
# WaveletModulatedDSA: v2 modulation paradigm, full dim channels
# ============================================================

class WaveletModulatedDSA(nn.Module):
    """Wavelet-modulated DSA with residual, operating on full dim channels.

    DSAWACGA_v2 modulation paradigm without internal split:
      - DSA produces multi-scale spatial features fdsa [B, dim, H, W]
      - WaveletModulation generates attention map A ∈ [0,1] [B, dim, H, W]
      - Fusion: output = A ⊙ fdsa + fdsa  ≡  (1 + A) ⊙ fdsa

    No Conv3×3 branch (redundant with DSA's dwconv3 and FFN).
    No internal split (DSA uses full dim channels for maximum capacity).

    Args:
        dim: Number of input/output channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.dsa = DSA(dim)
        self.wavelet_mod = WaveletModulation(dim, bias=True)

    def forward(self, x):
        fdsa = self.dsa(x)
        wavelet_attn = self.wavelet_mod(x)
        return wavelet_attn * fdsa + fdsa


# ============================================================
# FourierUnit: from CSFH, for true global modeling
# ============================================================

class DSAWACGA_v4_FourierUnit(nn.Module):
    """Fourier spectral convolution unit with dynamic grouped weighting.

    Copied from CSFH's SFHF_FourierUnit for true global modeling (H×W receptive field).

    Data flow:
      rfft2 → cat(real,imag) → BN → FPE + residual → weight(softmax)
      → fdc(grouped) → einsum dynamic weighting → GELU → cat(real,imag) → irfft2

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        groups: Number of expert groups for dynamic weighting.
    """

    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.fdc = nn.Conv2d(in_channels=in_channels * 2,
                              out_channels=out_channels * 2 * self.groups,
                              kernel_size=1, stride=1, padding=0,
                              groups=self.groups, bias=True)
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2,
                      out_channels=self.groups,
                      kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )
        self.fpe = nn.Conv2d(in_channels * 2, in_channels * 2,
                              kernel_size=3, padding=1, stride=1,
                              groups=in_channels * 2, bias=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted
        dy_weight = self.weight(ffted)
        ffted = self.fdc(ffted).view(batch, self.groups, 2 * c, h, -1)
        ffted = torch.einsum('ijkml,ijml->ikml', ffted, dy_weight)
        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        return output


# ============================================================
# TokenMixer_For_Global: channel expansion + FourierUnit + residual
# ============================================================

class TokenMixer_For_Global(nn.Module):
    """Global token mixer with channel expansion + FourierUnit + residual.

    Mirrors CSFH's TokenMixer_For_Gloal structure:
      conv_init(dim→2dim)+GELU → FourierUnit(2dim) → +residual → conv_fina(2dim→dim)+GELU

    Args:
        dim: Number of input/output channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.FFC = DSAWACGA_v4_FourierUnit(dim * 2, dim * 2)
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        x = self.conv_fina(x + x0)
        return x


# ============================================================
# DSAWACGA_v4_Mixer: WaveletModulatedDSA (local) + FourierUnit (global)
# ============================================================

class DSAWACGA_v4_Mixer(nn.Module):
    """DSAWACGA v4 Mixer: split design with WaveletModulatedDSA + FourierUnit.

    Mirrors SFHF_Mixer's split design:
      SFHF_Mixer:          conv_init→split→[Local(DilatedConv)+Global(FourierUnit)]→cat→GELU→SE→ca_conv
      DSAWACGA_v4_Mixer:   conv_init→split→[WaveletModulatedDSA+Global(FourierUnit)]→cat→GELU→SE→ca_conv

    Branch design:
      - chunk1 → WaveletModulatedDSA(dim): local multi-scale perception modulated
        by wavelet-based spatial-channel attention. DSA operates on full dim channels.
      - chunk2 → TokenMixer_For_Global(dim): global frequency modeling via FourierUnit.

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
        x = self.ca(x) * x
        x = self.ca_conv(x)
        return x


# ============================================================
# DSAWACGA_v4_FFN: multi-scale DWConv FFN (same as CSFH's SFHF_FFN)
# ============================================================

class DSAWACGA_v4_FFN(nn.Module):
    """Feed-Forward Network with multi-scale DWConv.

    Same as CSFH's SFHF_FFN:
      Conv1x1(dim→2dim) → Split(4 chunks, each dim/2)
        → [identity, DWConv3, DWConv5, DWConv7] → Cat → GELU → Conv1x1(2dim→dim)

    Args:
        dim: Number of input/output channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.dim_sp = dim // 2

        self.conv_init = nn.Conv2d(dim, dim * 2, 1)

        self.conv1_1 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3,
                                  padding=1, groups=self.dim_sp)
        self.conv1_2 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5,
                                  padding=2, groups=self.dim_sp)
        self.conv1_3 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7,
                                  padding=3, groups=self.dim_sp)

        self.gelu = nn.GELU()
        self.conv_fina = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)
        return x


# ============================================================
# DSAWACGAv4Block: Transformer-like two-stage block
# ============================================================

class DSAWACGAv4Block(nn.Module):
    """DSAWACGA v4 Block: Mixer + FFN with LayerScale.

    Mirrors SFHF_Block's two-stage structure:
      Stage 1 (Attention):  x' = DSAWACGA_v4_Mixer(Norm1(x)) * β + x
      Stage 2 (FFN):        x'' = FFN(Norm2(x')) * γ + x'

    Mixer: WaveletModulatedDSA (local) + FourierUnit (global)
    FFN:   multi-scale DWConv (identity + DW3/5/7)

    Args:
        dim: Number of input/output channels.
        norm_layer: Normalization layer (default BatchNorm2d).
    """

    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim

        self.norm1 = norm_layer(dim)
        self.mixer = DSAWACGA_v4_Mixer(dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

        self.norm2 = norm_layer(dim)
        self.ffn = DSAWACGA_v4_FFN(dim)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = self.mixer(self.norm1(x)) * self.beta + x
        x = self.ffn(self.norm2(x)) * self.gamma + x
        return x


# ============================================================
# C2f_DSAWACGA_v4: C2f container with DSAWACGAv4Block
# ============================================================

class C2f_DSAWACGA_v4(C2f):
    """C2f with DSAWACGAv4Block replacing Bottleneck.

    CSFH-style C2f container using DSAWACGAv4Block as the core block:
      cv1(c1→2c, 1×1) → split(2 chunks) → [DSAWACGAv4Block(c)×n] → cat → cv2(→c2, 1×1)

    DSAWACGAv4Block is a full Transformer-like block:
      Stage 1 (Attention):  x' = DSAWACGA_v4_Mixer(Norm1(x)) * β + x
        Mixer: conv_init→split→[WaveletModulatedDSA+FourierUnit]→cat→GELU→SE→ca_conv
      Stage 2 (FFN):        x'' = FFN(Norm2(x')) * γ + x'
        FFN: conv_init→split(4)→[identity,DW3,DW5,DW7]→cat→GELU→conv_fina

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of DSAWACGAv4Block repeats.
        shortcut: Whether to use shortcut connection in Bottleneck (unused here).
        g: Groups (unused, kept for C2f interface).
        e: Expansion ratio for hidden channels (default 0.5).
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DSAWACGAv4Block(self.c) for _ in range(n))