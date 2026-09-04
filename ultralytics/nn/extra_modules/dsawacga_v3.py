# DSAWACGA_v3: CSFH-framework + DSAWACGA_Mixer core block.
#
# Adopts CSFH (CSFPR-RTDETR) backbone framework with DSAWACGA_Mixer
# replacing SFHF_Mixer as the attention mechanism.
#
# Backbone architecture:
#   Conv(3→64,s=2) → Conv(64→128,s=2) → C2f_DSAWACGA(128)
#   → Conv(128→256,s=2) → C2f_DSAWACGA(256)
#   → Conv(256→384,s=2) → C2f_DSAWACGA(384)
#   → Conv(384→384,s=2) → C2f_DSAWACGA(384)×3
#
# C2f_DSAWACGA = C2f container with DSAWACGAv2Block replacing Bottleneck:
#   cv1(c1→2c, 1×1) → split(2 chunks) → [DSAWACGAv2Block(c)×n] → cat → cv2(→c2, 1×1)
#
# DSAWACGAv2Block mirrors SFHF_Block's two-stage Transformer-like structure:
#   Stage 1 (Attention):  x' = DSAWACGA_Mixer(Norm1(x)) * β + x
#   Stage 2 (FFN):        x'' = FFN(Norm2(x')) * γ + x'
#
# DSAWACGA_Mixer mirrors SFHF_Mixer's split design:
#   conv_init(dim→2dim) → split(2 chunks, each dim)
#     ├─ chunk1 → DSA(dim)              ← local multi-scale (6-branch DWConv + softmax)
#     └─ chunk2 → WaveletGlobal(dim)    ← global modeling (DWT + dynamic grouped weighting)
#   → cat → GELU → SE(2dim) → ca_conv(2dim→dim)
#
# DSA: standalone 6-branch multi-scale DWConv with spatial-adaptive softmax weighting.
#   Operates on full dim channels (no internal split), aligned with
#   SFHF_Mixer's TokenMixer_For_Local which also operates on full dim channels.
#
# WaveletGlobal mirrors SFHF_FourierUnit's dynamic grouped weighting:
#   DWT → subband processing → BN → FPE → dynamic weight (softmax) → grouped conv → einsum → GELU → IDWT
#   Replaces FFT with DWT: preserves spatial position, directional subbands (LH/HL/HH)
#
# Key advantage over CSFH (SFHF_Block):
#   - Wavelet (DWT) replaces Fourier (FFT): local time-frequency analysis
#     preserves spatial position, better for small objects
#   - Both branches operate on full dim channels (no internal split waste):
#     chunk1: DSA (6-branch multi-scale local perception)
#     chunk2: WaveletGlobal (global frequency modeling with directional subbands)
#   - SE channel attention after cat: coarse channel-level calibration
#   - CSP gradient flow from C2f container

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import C2f
from .dsawacga_v2 import LearnableHaarDWT

__all__ = ['DSA', 'WaveletGlobal', 'TokenMixer_For_Global', 'DSAWACGA_Mixer', 'DSAWACGAv2Block', 'DSAWACGA_FFN', 'C2f_DSAWACGA']


class DSA(nn.Module):
    """Dynamic Scale Attention: 6-branch multi-scale DWConv with softmax weighting.

    Standalone local token mixer, extracted from DSAWACGAv2.
    Operates on full dim channels without internal split.

    Compared to SFHF_Mixer's TokenMixer_For_Local:
      - TokenMixer_For_Local: 2-branch DilatedConv(d=1,2) on dim channels
      - DSA: 6-branch DWConv(1,3,5,7,9,11) + softmax on dim channels
      - DSA has richer multi-scale coverage (6 scales vs 2)
      - DSA uses dynamic softmax weighting (input-adaptive) vs fixed dilation

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


class WaveletGlobal(nn.Module):
    """Wavelet-based global token mixer with dynamic grouped weighting.

    Mirrors SFHF_FourierUnit's dynamic mixture-of-experts mechanism:
      SFHF_FourierUnit:  FFT → BN → FPE → weight(softmax) → fdc(grouped) → einsum → GELU → IFFT
      WaveletGlobal:     DWT → subband_proc → BN → FPE → weight(softmax) → fdc(grouped) → einsum → GELU → IDWT

    Key differences from SFHF_FourierUnit:
      - DWT/IDWT replaces FFT/IFFT: local time-frequency analysis vs global frequency
      - DWT subbands (LL/LH/HL/HH) provide directional frequency decomposition:
        LL = low-frequency approximation (global structure)
        LH = horizontal edges, HL = vertical edges, HH = diagonal textures
      - Spatial position is preserved (DWT is spatially localized)
      - IDWT reconstructs from processed subbands (inverse wavelet transform)

    Args:
        dim: Number of input/output channels.
        groups: Number of expert groups for dynamic weighting (default 4).
    """

    def __init__(self, dim, groups=4):
        super().__init__()
        self.dim = dim
        self.groups = groups

        self.dwt = LearnableHaarDWT(level=1)

        self.ya_proj = nn.Conv2d(dim, dim // 4, kernel_size=1)
        self.yh_conv = nn.Conv2d(dim, dim // 4, kernel_size=(1, 3), padding=(0, 1), groups=dim // 4)
        self.yv_conv = nn.Conv2d(dim, dim // 4, kernel_size=(3, 1), padding=(1, 0), groups=dim // 4)
        self.yd_act = nn.Tanh()
        self.yd_proj = nn.Conv2d(dim, dim // 4, kernel_size=1)

        self.ll_conv = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, groups=dim // 4)
        self.horizontal_conv, self.vertical_conv, self.diagonal_conv = self._create_wave_conv(dim // 4)

        self.subband_fusion = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        self.bn = nn.BatchNorm2d(dim)
        self.fpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)

        self.weight = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=groups, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )
        self.fdc = nn.Conv2d(in_channels=dim, out_channels=dim * groups,
                              kernel_size=1, stride=1, padding=0, groups=groups, bias=True)

    def _create_conv_layer(self, kernel, dim):
        conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        conv.weight.data = kernel.repeat(dim, 1, 1, 1)
        return conv

    def _create_wave_conv(self, dim):
        horizontal_kernel = torch.tensor([[1, 1, 1],
                                          [0, 0, 0],
                                          [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        vertical_kernel = torch.tensor([[1, 0, -1],
                                        [1, 0, -1],
                                        [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        diagonal_kernel = torch.tensor([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        horizontal_conv = self._create_conv_layer(horizontal_kernel, dim)
        vertical_conv = self._create_conv_layer(vertical_kernel, dim)
        diagonal_conv = self._create_conv_layer(diagonal_kernel, dim)
        return horizontal_conv, vertical_conv, diagonal_conv

    def _dwt_process(self, x):
        ya, (yh, yv, yd) = self.dwt(x)

        ya_proc = self.ya_proj(ya)
        ya_proc = self.ll_conv(ya_proc)

        yh_proc = self.yh_conv(yh)
        yh_proc = self.horizontal_conv(yh_proc)

        yv_proc = self.yv_conv(yv)
        yv_proc = self.vertical_conv(yv_proc)

        yd_proc = self.yd_act(yd)
        yd_proc = self.yd_proj(yd_proc)
        yd_proc = self.diagonal_conv(yd_proc)

        subbands_proc = torch.cat([ya_proc, yh_proc, yv_proc, yd_proc], dim=1)
        return self.subband_fusion(subbands_proc)

    def _idwt_reconstruct(self, processed, x_original):
        return F.interpolate(
            processed,
            size=x_original.shape[2:],
            mode='bilinear',
            align_corners=False
        )

    def forward(self, x):
        batch, c, h, w = x.size()

        wavelet_repr = self._dwt_process(x)

        wavelet_repr = self.bn(wavelet_repr)
        wavelet_repr = self.fpe(wavelet_repr) + wavelet_repr

        dy_weight = self.weight(wavelet_repr)

        conv_out = self.fdc(wavelet_repr).view(batch, self.groups, c, h // 2, -1)
        result = torch.einsum('ijkml,ijml->ikml', conv_out, dy_weight)
        result = F.gelu(result)

        result = self._idwt_reconstruct(result, x)

        return result


class TokenMixer_For_Global(nn.Module):
    """Global token mixer with channel expansion + WaveletGlobal + residual.

    Mirrors CSFH's TokenMixer_For_Gloal structure:
      CSFH:    conv_init(dim→2dim)+GELU → FourierUnit(2dim) → +residual → conv_fina(2dim→dim)+GELU
      Ours:    conv_init(dim→2dim)+GELU → WaveletGlobal(2dim) → +residual → conv_fina(2dim→dim)+GELU

    Channel expansion serves as capacity expansion:
      - CSFH: dim→2dim for capacity, then 2dim→4dim for real/imag (total 4dim width)
      - Ours: dim→2dim for capacity, DWT is real-valued (total 2dim width)
      - Remaining 2× gap is FFT's real/imag requirement, not capacity

    The residual connection preserves the expanded features before WaveletGlobal,
    and conv_fina projects back to dim channels.

    Args:
        dim: Number of input/output channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.wavelet_global = WaveletGlobal(dim * 2)
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.wavelet_global(x)
        x = self.conv_fina(x + x0)
        return x


class DSAWACGA_Mixer(nn.Module):
    """DSAWACGA Mixer: split design with DSA (local) + TokenMixer_For_Global (global).

    Mirrors SFHF_Mixer's split design:
      SFHF_Mixer:      conv_init(dim→2dim) → split → [Local(dim) + Global(dim)] → cat → GELU → SE → ca_conv
      DSAWACGA_Mixer:  conv_init(dim→2dim) → split → [DSA(dim) + Global(dim)] → cat → GELU → SE → ca_conv

    Branch design:
      - chunk1 → DSA(dim): local multi-scale perception via 6-branch DWConv
        with spatial-adaptive softmax weighting (input-adaptive)
      - chunk2 → TokenMixer_For_Global(dim): global frequency modeling via
        channel expansion(dim→2dim) + WaveletGlobal(2dim) + residual + compression(2dim→dim)
        Mirrors CSFH's TokenMixer_For_Gloal structure for sufficient capacity.

    Both branches input/output dim channels (half of 2dim after split).

    Args:
        dim: Number of input/output channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.conv_init = nn.Conv2d(dim, dim * 2, 1)

        self.dsa = DSA(dim)

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
        x_local = self.dsa(x_local)
        x_global = self.global_mixer(x_global)
        x = torch.cat([x_local, x_global], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)
        return x


class DSAWACGAv2Block(nn.Module):
    """DSAWACGAv2 Block: full Transformer-like block with DSAWACGA_Mixer + FFN.

    Mirrors SFHF_Block's two-stage structure (mixer + ffn) while using
    DSAWACGA_Mixer (DSA local + WaveletGlobal global) as the
    attention mechanism:

      Stage 1 (Attention):  x' = DSAWACGA_Mixer(Norm1(x)) * β + x
      Stage 2 (FFN):        x'' = FFN(Norm2(x')) * γ + x'

    Compared to SFHF_Block:
      - DSAWACGA_Mixer replaces SFHF_Mixer:
        - SFHF_Mixer: conv_init→split→[Local(DilatedConv)+Global(FourierUnit)]→cat→GELU→SE→ca_conv
        - DSAWACGA_Mixer: conv_init→split→[DSA+WaveletGlobal]→cat→GELU→SE→ca_conv
        - Channel flow (dim→2dim→split→dim×2→cat→2dim→dim) is identical
        - Local: DSA (6-branch DWConv+softmax) replaces DilatedConv(d=1,2)
        - Global: WaveletGlobal (DWT+dynamic grouping) replaces FourierUnit (FFT+dynamic grouping)
      - DSAWACGA_FFN replaces SFHF_FFN (multi-scale DWConv 3/5/7)
        with multi-scale DWConv 3/5/7/9 (4 branches)

    Args:
        dim: Number of input/output channels.
        norm_layer: Normalization layer (default BatchNorm2d).
    """

    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim

        self.norm1 = norm_layer(dim)
        self.mixer = DSAWACGA_Mixer(dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

        self.norm2 = norm_layer(dim)
        self.ffn = DSAWACGA_FFN(dim)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = self.mixer(self.norm1(x)) * self.beta + x
        x = self.ffn(self.norm2(x)) * self.gamma + x
        return x


class DSAWACGA_FFN(nn.Module):
    """Feed-Forward Network for DSAWACGAv2Block.

    Mirrors SFHF_FFN's structure exactly:
      Conv1x1(dim→2dim) → Split(4 chunks, each dim/2)
        → [identity, DWConv3, DWConv5, DWConv7] → Cat → GELU → Conv1x1(2dim→dim)

    The first chunk is identity (no processing), preserving original features
    after conv_init. This matches CSFH's SFHF_FFN design where x[0] is
    not processed by any convolution, acting as a skip connection within the FFN.

    Args:
        dim: Number of input/output channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.dim_sp = dim * 2 // 4

        self.conv_init = nn.Conv2d(dim, dim * 2, 1)

        self.dw1 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3,
                              padding=1, groups=self.dim_sp)
        self.dw2 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5,
                              padding=2, groups=self.dim_sp)
        self.dw3 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7,
                              padding=3, groups=self.dim_sp)

        self.gelu = nn.GELU()
        self.conv_fina = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.dw1(x[1])
        x[2] = self.dw2(x[2])
        x[3] = self.dw3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)
        return x


class C2f_DSAWACGA(C2f):
    """C2f with DSAWACGAv2Block replacing Bottleneck.

    CSFH-style C2f container using DSAWACGAv2Block as the core block:
      cv1(c1→2c, 1×1) → split(2 chunks) → [DSAWACGAv2Block(c)×n] → cat → cv2(→c2, 1×1)

    DSAWACGAv2Block is a full Transformer-like block:
      Stage 1 (Attention):  x' = DSAWACGA_Mixer(Norm1(x)) * β + x
      Stage 2 (FFN):        x'' = FFN(Norm2(x')) * γ + x'

    This mirrors CSFH_Block's structure (SFHF_Mixer + SFHF_FFN) while
    using DSAWACGA_Mixer (DSA local + WaveletGlobal global)
    as the attention mechanism.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of DSAWACGAv2Block repeats.
        shortcut: Whether to use shortcut connection in Bottleneck (unused here).
        g: Groups (unused, kept for C2f interface).
        e: Expansion ratio for hidden channels (default 0.5).
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DSAWACGAv2Block(self.c) for _ in range(n))