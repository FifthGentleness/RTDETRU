# CWLSBlock / WLSformer: Wavelet–Linear–Star Transformer Block
#
# Naming convention (aligned with CSFHBlock/SFHformer):
#   CWLSBlock  : C(ontainer) + W + L + S  — C2f container
#   WLSformer  : W + L + S                — inner Transformer unit
#     W = Wavelet-modulated local aggregation  (Stage 1, local branch)
#     L = Linear attention for global context   (Stage 1, global branch)
#     S = Star-operation FFN                    (Stage 2)
#
# Sub-modules:
#   MSDA  : Multi-Scale Depthwise Aggregator       (6-branch DWConv + softmax)
#   WFAM  : Wavelet Frequency Attention Modulator  (Haar DWT subband + sigmoid)
#   WMLA  : Wavelet-Modulated Local Aggregator     (MSDA + WFAM, (1+A)⊙F)
#   FLA   : Focused Linear Attention               (FLatten, O(Nd²) global)
#   GTM   : Global Token Mixer                     (FLA + identity residual)
#   DBM   : Dual-Branch Mixer                      (WMLA + GTM + SE)
#   MSGFFN: Multi-Scale Gated Star FFN             (Star Operation + multi-scale DW gate)

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import C2f

__all__ = ['MSDA', 'WFAM', 'WMLA', 'FLA', 'GTM',
           'DBM', 'MSGFFN', 'WLSformer', 'CWLSBlock']


# ============================================================
# Haar DWT primitives (self-contained, no pywt dependency)
# ============================================================

_HAAR_DEC_LO = torch.tensor([0.7071067811865476, 0.7071067811865476], dtype=torch.float32)
_HAAR_DEC_HI = torch.tensor([-0.7071067811865476, 0.7071067811865476], dtype=torch.float32)
_HAAR_FILT_LEN = 2


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    return a_flat.unsqueeze(-1) * b_flat.unsqueeze(0)


def _get_pad(data_len: int, filt_len: int):
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2
    if data_len % 2 != 0:
        padr += 1
    return padr, padl


def _fwt_pad2(data: torch.Tensor, mode: str = "replicate") -> torch.Tensor:
    padb, padt = _get_pad(data.shape[-2], _HAAR_FILT_LEN)
    padr, padl = _get_pad(data.shape[-1], _HAAR_FILT_LEN)
    return F.pad(data, [padl, padr, padt, padb], mode=mode)


class _LearnableHaarDWT(nn.Module):
    def __init__(self, level=1, mode="replicate"):
        super().__init__()
        self.level = level
        self.mode = mode
        dec_lo = _HAAR_DEC_LO.clone()
        dec_hi = _HAAR_DEC_HI.clone()
        dec_lo_flipped = dec_lo.flip(-1).unsqueeze(0)
        dec_hi_flipped = dec_hi.flip(-1).unsqueeze(0)
        self.dec_lo = nn.Parameter(dec_lo_flipped, requires_grad=True)
        self.dec_hi = nn.Parameter(dec_hi_flipped, requires_grad=True)

    def _build_kernel(self, c):
        lo = self.dec_lo.squeeze(0)
        hi = self.dec_hi.squeeze(0)
        ll = _outer(lo, lo)
        lh = _outer(hi, lo)
        hl = _outer(lo, hi)
        hh = _outer(hi, hi)
        filt = torch.stack([ll, lh, hl, hh], 0)
        dwt_kernel = filt.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)
        return dwt_kernel

    def forward(self, x):
        b, c, h, w = x.shape
        dwt_kernel = self._build_kernel(c)

        l_component = x
        wavelet_component = []
        for _ in range(self.level):
            l_component = _fwt_pad2(l_component, mode=self.mode)
            h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
            res = h_component.reshape(b, c, 4, h_component.shape[-2], h_component.shape[-1])
            l_component = res[:, :, 0, :, :]
            lh_component = res[:, :, 1, :, :]
            hl_component = res[:, :, 2, :, :]
            hh_component = res[:, :, 3, :, :]
            wavelet_component.append((lh_component, hl_component, hh_component))
        wavelet_component.append(l_component)
        return wavelet_component[::-1]


# ============================================================
# MSDA: Multi-Scale Depthwise Aggregator
#   6-branch multi-scale DWConv (k=1,3,5,7,9,11) with softmax
#   spatial-adaptive weighting + 1x1 channel mixing
# ============================================================

class MSDA(nn.Module):
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
# WFAM: Wavelet Frequency Attention Modulator
#   Haar DWT → directional subband processing → Sigmoid → [0,1]
#   Output: spatial-channel attention map A ∈ [0,1]
# ============================================================

class WFAM(nn.Module):
    """Wavelet Frequency Attention Modulator.

    Generates a spatial-channel attention map [B, dim, H, W] from DWT
    subbands. No QKV attention -- purely wavelet-driven modulation weights.

    Output is Sigmoid-normalized to [0,1], suitable for:
        output = wavelet_attn * f_msda + f_msda  (SME-DETR DPF-style)
    """

    def __init__(self, dim, bias=True):
        super().__init__()

        self.dwt = _LearnableHaarDWT(level=1)

        self.ya_proj = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=bias)
        self.yh_conv = nn.Conv2d(dim, dim // 4, kernel_size=(1, 3), padding=(0, 1), groups=dim // 4, bias=bias)
        self.yv_conv = nn.Conv2d(dim, dim // 4, kernel_size=(3, 1), padding=(1, 0), groups=dim // 4, bias=bias)
        self.yd_act = nn.Tanh()
        self.yd_proj = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=bias)

        self.subband_fusion = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias),
            nn.Sigmoid()
        )

        self.ll_conv = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, stride=1, padding=1, groups=dim // 4, bias=bias)
        self.horizontal_conv, self.vertical_conv, self.diagonal_conv = self._create_wave_conv(dim // 4)

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

    def forward(self, x):
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

        wavelet_attention_map = self.subband_fusion(subbands_proc)
        wavelet_attention_map = F.interpolate(
            wavelet_attention_map,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return wavelet_attention_map


# ============================================================
# WMLA: Wavelet-Modulated Local Aggregator
#   MSDA (multi-scale spatial aggregation) modulated by WFAM
#   (wavelet frequency attention): output = (1 + A) ⊙ MSDA(x)
# ============================================================

class WMLA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.msda = MSDA(dim)
        self.wfam = WFAM(dim, bias=True)

    def forward(self, x):
        f_msda = self.msda(x)
        wavelet_attn = self.wfam(x)
        return wavelet_attn * f_msda + f_msda


# ============================================================
# FLA: Focused Linear Attention
#   (FLatten: Vision Transformer using Focused Linear Attention, ICCV 2023)
#
#   Core: φ(Q)(φ(K)ᵀV) / φ(Q)(φ(K)ᵀ1)  — O(Nd²) vs O(N²d)
#   Focusing: φ(x) = ReLU(x)^f sharpens attention distribution
#   DWC on V: incorporates local structural info
# ============================================================

class FLA(nn.Module):
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
# GTM: Global Token Mixer
#   FLA + identity residual: x + FLA(x)
# ============================================================

class GTM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fla = FLA(dim, dim)

    def forward(self, x):
        return x + self.fla(x)


# ============================================================
# DBM: Dual-Branch Mixer
#   conv_init(dim→2dim) → split → [WMLA(local) + GTM(global)]
#   → cat → GELU → SE → ca_conv
# ============================================================

class DBM(nn.Module):
    """Dual-Branch Mixer: WMLA (local) + GTM (global) + SE.

    Args:
        dim: Number of input/output channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.conv_init = nn.Conv2d(dim, dim * 2, 1)

        self.local_mixer = WMLA(dim)
        self.global_mixer = GTM(dim)

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
# MSGFFN: Multi-Scale Gated Star FFN
#   Star Operation with multi-scale spatial gate
#   GELU(x1') * x2, where x1' = [DW3(x1a), DW5(x1b)]
# ============================================================

class MSGFFN(nn.Module):
    """Multi-Scale Gated Star FFN.

    Architecture:
      fc1(dim → 2dim) → split(x1, x2)
        x1 → split(x1a, x1b) → [DW3(x1a), DW5(x1b)] → cat → x1'
        x2 unchanged
      GELU(x1') * x2 → fc2(dim → dim) + residual

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
# WLSformer: Wavelet–Linear–Star Transformer unit
#   Stage 1 (Mixer):  x' = DBM(Norm1(x)) * β + x
#   Stage 2 (FFN):    x'' = MSGFFN(Norm2(x')) * γ + x'
# ============================================================

class WLSformer(nn.Module):
    """WLSformer: W + L + S Transformer unit with LayerScale.

    W = Wavelet-modulated local aggregation (WMLA)
    L = Linear attention for global context   (FLA)
    S = Star-operation FFN                    (MSGFFN)

    Args:
        dim: Number of input/output channels.
        norm_layer: Normalization layer (default BatchNorm2d).
    """

    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim

        self.norm1 = norm_layer(dim)
        self.mixer = DBM(dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

        self.norm2 = norm_layer(dim)
        self.ffn = MSGFFN(dim)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = self.mixer(self.norm1(x)) * self.beta + x
        x = self.ffn(self.norm2(x)) * self.gamma + x
        return x


# ============================================================
# CWLSBlock: C2f container with WLSformer
#   cv1(c1→2c, 1×1) → split(2 chunks) → [WLSformer(c)×n] → cat → cv2(→c2, 1×1)
# ============================================================

class CWLSBlock(C2f):
    """C2f container housing WLSformer units.

    C = Container (C2f-style CSP)
    W = Wavelet-modulated local aggregation
    L = Linear attention for global context
    S = Star-operation FFN

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of WLSformer repeats.
        shortcut: Whether to use shortcut connection (unused).
        g: Groups (unused, kept for C2f interface).
        e: Expansion ratio for hidden channels (default 0.5).
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(WLSformer(self.c) for _ in range(n))