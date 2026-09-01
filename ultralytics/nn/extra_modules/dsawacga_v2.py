# DSAWACGA_v2: Wavelet-modulated DSA (SME-DETR DPF-style modulation).
#
# Key design (aligned with SME-DETR DPF paradigm):
#   - WaveletModulation generates spatial-channel attention map A [B,cr,H,W]
#     from DWT subbands (no QKV attention, pure wavelet-based).
#   - DSA produces multi-scale spatial features fdsa [B,cr,H,W].
#   - Fusion: output = A ⊙ fdsa + fdsa  (modulation + residual).
#     Equivalent to (1 + A) ⊙ fdsa, where A ∈ [0,1] via Sigmoid.
#     Residual ensures DSA information is preserved; modulation enhances
#     spatial-channel selective regions identified by wavelet analysis.
#
# Changes from DSAWACGA (dsawacga.py):
#   - Removed: cross_gate, QKV attention from WACGA
#   - WACGA → WaveletModulation: pure wavelet attention map generator
#   - Fusion: cat→fusion replaced by wavelet_attn ⊙ fdsa + fdsa
#   - Advantage over SME-DETR DCA: spatial-channel modulation (not channel-only)
#     is more fine-grained and better for small objects.
#
# DSA redesign (6-branch multi-scale DWConv, no Scharr, no dilated conv):
#   - DW1x1, DW3x3, DW5x5, DW7x7, DW9x9, DW11x11 (6 branches)
#   - Removed: Scharr edge branch (fixed kernel, competes with learnable
#     branches under softmax and tends to be suppressed over training)
#   - Removed: dilated DWConv (replaced by larger standard kernels DW9x9/DW11x11
#     for cleaner multi-scale coverage without gridding artifacts)
#   - Spatial-adaptive softmax weighting preserved (6-way)

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import BasicBlock
from ..modules.conv import Conv

__all__ = ['DSAWACGAv2', 'DSAWACGAv2BasicBlock', 'BlocksDSAWACGAv2']


# ============================================================
# Self-contained Haar DWT (no pywt dependency)
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


class LearnableHaarDWT(nn.Module):
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
# WaveletModulation: Pure wavelet-based spatial-channel attention map
# ============================================================

class WaveletModulation(nn.Module):
    """Generates a spatial-channel attention map [B, dim, H, W] from DWT
    subbands. No QKV attention -- purely wavelet-driven modulation weights.

    Output is Sigmoid-normalized to [0,1], suitable for:
        output = wavelet_attn * fdsa + fdsa  (SME-DETR DPF-style)
    """

    def __init__(self, dim, bias=True):
        super().__init__()

        self.dwt = LearnableHaarDWT(level=1)

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
# DSAWACGAv2: Wavelet-modulated DSA (SME-DETR DPF-style)
# ============================================================

class DSAWACGAv2(nn.Module):
    """DSAWACGAv2: Wavelet-modulated DSA with residual.

    Half-channel split design:
    - First half  -> DSA (6-branch multi-scale DWConv) modulated by
                     WaveletModulation: output = wavelet_attn ⊙ fdsa + fdsa
    - Second half -> Conv3x3 for local feature extraction
    - dim = ch_out of the block (64/128/256/512), cr = dim // 2.

    Modulation paradigm (aligned with SME-DETR DPF):
        wavelet_attn = WaveletModulation(xs)   # [B, cr, H, W], Sigmoid [0,1]
        fdsa = DSA(xs)                         # [B, cr, H, W]
        dsawacga_out = wavelet_attn * fdsa + fdsa  # (1 + A) ⊙ fdsa
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.half_dim = half_dim

        cr = half_dim
        self.cr = cr

        # --- DSA: 6-branch multi-scale DWConv ---
        self.dwconv1 = nn.Conv2d(cr, cr, 1, groups=cr, bias=False)
        self.dwconv3 = nn.Conv2d(cr, cr, 3, padding=1, groups=cr, bias=False)
        self.dwconv5 = nn.Conv2d(cr, cr, 5, padding=2, groups=cr, bias=False)
        self.dwconv7 = nn.Conv2d(cr, cr, 7, padding=3, groups=cr, bias=False)
        self.dwconv9 = nn.Conv2d(cr, cr, 9, padding=4, groups=cr, bias=False)
        self.dwconv11 = nn.Conv2d(cr, cr, 11, padding=5, groups=cr, bias=False)

        # --- DSA: Channel mixing ---
        self.channel_mix = nn.Conv2d(cr, cr, 1, bias=False)

        # --- DSA: Spatial-adaptive weights (6-way) ---
        self.weight_conv = nn.Conv2d(cr * 6, 6, 1, bias=False)

        # --- WaveletModulation: generates spatial-channel attention map ---
        self.wavelet_mod = WaveletModulation(cr, bias=True)

        # --- Conv path: Conv3x3+BN+ReLU ---
        self.conv_path = Conv(half_dim, half_dim, 3, 1, act=nn.ReLU())

    def forward(self, x):
        x_dsawacga, x_conv = x.chunk(2, dim=1)

        # === DSA path (first half) ===
        xs = x_dsawacga

        f0 = self.dwconv1(xs)
        f1 = self.dwconv3(xs)
        f2 = self.dwconv5(xs)
        f3 = self.dwconv7(xs)
        f4 = self.dwconv9(xs)
        f5 = self.dwconv11(xs)

        spatial_cat = torch.cat([f0, f1, f2, f3, f4, f5], dim=1)
        spatial_weights = F.softmax(self.weight_conv(spatial_cat), dim=1)

        fdsa = (spatial_weights[:, 0:1] * f0 +
                spatial_weights[:, 1:2] * f1 +
                spatial_weights[:, 2:3] * f2 +
                spatial_weights[:, 3:4] * f3 +
                spatial_weights[:, 4:5] * f4 +
                spatial_weights[:, 5:6] * f5)
        fdsa = self.channel_mix(fdsa)

        # === Wavelet modulation (SME-DETR DPF-style) ===
        wavelet_attn = self.wavelet_mod(xs)
        dsawacga_out = wavelet_attn * fdsa + fdsa

        # === Conv path (second half) ===
        conv_out = self.conv_path(x_conv)

        # === Concat two paths ===
        out = torch.cat([dsawacga_out, conv_out], dim=1)

        return out


# ============================================================
# BasicBlock with DSAWACGAv2 replacing branch2a
# ============================================================

class DSAWACGAv2BasicBlock(BasicBlock):
    """BasicBlock with DSAWACGAv2 replacing branch2a.

    DSAWACGAv2 replaces branch2a; branch2b (3x3) and the residual shortcut
    (and final act) are unchanged.
    """
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        del self.branch2a
        self.dsawacga = DSAWACGAv2(ch_out)

    def forward(self, x):
        out = self.dsawacga(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.act(out)
        return out


# ============================================================
# Stage container: vanilla BasicBlocks + last-block DSAWACGAv2
# ============================================================

class BlocksDSAWACGAv2(nn.Module):
    """Stage container: all blocks vanilla BasicBlock except the LAST one,
    which is DSAWACGAv2BasicBlock (DSAWACGAv2 replaces branch2a).

    Stride/shortcut logic identical to ultralytics Blocks.
    """

    def __init__(self, ch_in, ch_out, block, count, stage_num, act='relu', variant='d'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            if i < count - 1:
                self.blocks.append(
                    block(
                        ch_in,
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1,
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                    )
                )
            else:
                # Last block: DSAWACGAv2 replaces branch2a
                self.blocks.append(
                    DSAWACGAv2BasicBlock(
                        ch_in,
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1,
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                    )
                )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out