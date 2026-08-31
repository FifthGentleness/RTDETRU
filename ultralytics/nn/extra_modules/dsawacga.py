# DSAWACGA ported from rtdetr_pytorch/src/nn/backbone/presnet_dsawacga.py
# (config: rtdetr_r18vd_6x_visdrone_dsawacga.yml, backbone PResNet_DSAWACGA).
#
# Porting rules (structure kept 1:1 with rtdetr_pytorch, no channel changes):
#   - DSAWACGA module math is identical (ScharrEdge / LearnableHaarDWT / WACGA /
#     cross_gate / half-channel chunk design).
#   - DSAWACGABasicBlock replicates _insert_dsawacga semantics: DSAWACGA replaces
#     branch2a of the LAST BasicBlock of each stage; branch2b (3x3) and the
#     residual shortcut are untouched.
#   - BlocksDSAWACGA is the stage container: all blocks are vanilla BasicBlock
#     except the last one, which is DSAWACGABasicBlock. Channel layout follows
#     PResNet-18vd: 64/128/256/512, so DSAWACGA dim = ch_out, cr = ch_out // 2.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import BasicBlock
from ..modules.conv import Conv

__all__ = ['DSAWACGA', 'DSAWACGABasicBlock', 'BlocksDSAWACGA']


class ScharrEdge(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        scharr_x = torch.tensor(
            [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32
        ).reshape(1, 1, 3, 3) / 16.0
        scharr_y = torch.tensor(
            [[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32
        ).reshape(1, 1, 3, 3) / 16.0

        self.register_buffer('kernel_x', scharr_x.repeat(channels, 1, 1, 1))
        self.register_buffer('kernel_y', scharr_y.repeat(channels, 1, 1, 1))

    def forward(self, x):
        gx = F.conv2d(x, self.kernel_x, padding=1, groups=self.channels)
        gy = F.conv2d(x, self.kernel_y, padding=1, groups=self.channels)
        magnitude = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return magnitude


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
# WACGA: Wavelet-Aware Convolutional Gating Attention
# ============================================================

class WACGA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

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
        b, c, h, w = x.shape

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

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = out.reshape(b, c, h, w)
        out = self.project_out(out)

        out = out * wavelet_attention_map
        return out


# ============================================================
# DSAWACGA: DSA + WACGA with cross-interaction
# ============================================================

class DSAWACGA(nn.Module):
    """DSAWACGA: Half-channel split design, DSA + WACGA.

    Identical to rtdetr_pytorch presnet_dsawacga.DSAWACGA:
    - Input channels are split into two halves along channel dim:
      * First half  -> DSA + WACGA innovative branch (parallel, cross-gated)
      * Second half -> Conv3x3 for local feature extraction
    - dim = ch_out of the block (64/128/256/512), cr = dim // 2.
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.half_dim = half_dim

        cr = half_dim
        self.cr = cr

        # --- DSA: Multi-scale DWConv ---
        self.dwconv3 = nn.Conv2d(cr, cr, 3, padding=1, groups=cr, bias=False)
        self.dwconv5 = nn.Conv2d(cr, cr, 5, padding=2, groups=cr, bias=False)
        self.dwconv7 = nn.Conv2d(cr, cr, 7, padding=3, groups=cr, bias=False)
        self.dwconv_d4 = nn.Conv2d(cr, cr, 3, padding=4, dilation=4, groups=cr, bias=False)

        # --- DSA: Scharr edge detection branch ---
        self.scharr = ScharrEdge(cr)

        # --- DSA: Channel mixing ---
        self.channel_mix = nn.Conv2d(cr, cr, 1, bias=False)

        # --- DSA: Spatial-adaptive weights (5-way) ---
        self.weight_conv = nn.Conv2d(cr * 5, 5, 1, bias=False)

        # --- WACGA: Wavelet-Aware Convolutional Gating Attention ---
        self.wacga = WACGA(cr, num_heads=num_heads, bias=True)

        # --- Cross-interaction ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cross_gate = nn.Sequential(
            nn.Conv2d(cr * 2, cr * 2, 1, bias=False),
            nn.Sigmoid(),
        )

        # --- DSAWACGA internal fusion: Concat -> Conv1x1+BN+SiLU ---
        self.dsawacga_fusion = Conv(cr * 2, half_dim, 1, 1)

        # --- Conv path: Conv3x3+BN+ReLU ---
        self.conv_path = Conv(half_dim, half_dim, 3, 1, act=nn.ReLU())

    def forward(self, x):
        x_dsawacga, x_conv = x.chunk(2, dim=1)

        # === DSA path (first half) ===
        xs = x_dsawacga

        f1 = self.dwconv3(xs)
        f2 = self.dwconv5(xs)
        f3 = self.dwconv7(xs)
        f4 = self.dwconv_d4(xs)
        f_scharr = self.scharr(xs)
        f_scharr = f_scharr - f_scharr.mean(dim=[2, 3], keepdim=True)

        spatial_cat = torch.cat([f1, f2, f3, f4, f_scharr], dim=1)
        spatial_weights = F.softmax(self.weight_conv(spatial_cat), dim=1)

        fdsa = (spatial_weights[:, 0:1] * f1 +
                spatial_weights[:, 1:2] * f2 +
                spatial_weights[:, 2:3] * f3 +
                spatial_weights[:, 3:4] * f4 +
                spatial_weights[:, 4:5] * f_scharr)
        fdsa = self.channel_mix(fdsa)

        # === WACGA path (first half, same input as DSA) ===
        fwacga = self.wacga(xs)

        # === Cross-interaction ===
        dsa_signal = self.gap(fdsa)
        wacga_signal = self.gap(fwacga)
        cross_input = torch.cat([dsa_signal, wacga_signal], dim=1)
        cross_weight = self.cross_gate(cross_input)

        # === DSAWACGA internal fusion ===
        fused = torch.cat([fdsa, fwacga], dim=1)
        fused = fused * cross_weight
        dsawacga_out = self.dsawacga_fusion(fused)

        # === Conv path (second half) ===
        conv_out = self.conv_path(x_conv)

        # === Concat two paths ===
        out = torch.cat([dsawacga_out, conv_out], dim=1)

        return out


# ============================================================
# BasicBlock with DSAWACGA replacing branch2a
# ============================================================

class DSAWACGABasicBlock(BasicBlock):
    """Replicates rtdetr_pytorch DSAWACGABasicBlock (presnet_dsawacga.py):
    DSAWACGA replaces branch2a; branch2b (3x3) and the residual shortcut
    (and final act) are unchanged.
    """
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        del self.branch2a
        self.dsawacga = DSAWACGA(ch_out)

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
# Stage container: vanilla BasicBlocks + last-block DSAWACGA
# ============================================================

class BlocksDSAWACGA(nn.Module):
    """Stage container replicating rtdetr_pytorch PResNet_DSAWACGA._insert_dsawacga.

    All blocks are the vanilla `block` (BasicBlock) except the LAST block of
    the stage, which is DSAWACGABasicBlock (DSAWACGA replaces branch2a).
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
                # Last block: DSAWACGA replaces branch2a (rtdetr_pytorch uses
                # stride=1, shortcut=True for the last block of each stage).
                self.blocks.append(
                    DSAWACGABasicBlock(
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
