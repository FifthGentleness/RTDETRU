# DSAWACGA_v6: Full subband reconstruction with direction-aware high-freq processing
#
# Key design:
#   Local branch:  x_local + DSA(x_local)           ← multi-scale local + residual
#   Global branch: DWT → process all 4 subbands → IDWT → SE → FLA
#     LL → Conv1×1            ← global structure
#     LH → DWConv(1,3)        ← horizontal edge enhancement
#     HL → DWConv(3,1)        ← vertical edge enhancement
#     HH → Conv1×1 + Tanh     ← diagonal texture (lightweight)
#     → IDWT(LL',LH',HL',HH') → SE → FLA
#
#   vs old v6 (LL only): IDWT preserves high-freq edge info critical for small objects
#   vs CSFH (FFT): DWT provides directional decomposition (LH/HL/HH)
#     while FFT mixes all directions into real/imag parts

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import C2f
from .dsawacga_v2 import LearnableHaarDWT, _HAAR_DEC_LO, _HAAR_DEC_HI, _outer
from .dsawacga_v5 import FocusedLinearAttention

__all__ = ['DSA', 'LearnableHaarIDWT', 'LearnableHaarIDWT_Full',
           'GlobalBranch',
           'DSAWACGA_v6_Mixer', 'DSAWACGA_v6_FFN',
           'DSAWACGAv6Block', 'C2f_DSAWACGA_v6']


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
# LearnableHaarIDWT: Inverse DWT from LL only (high-freq = 0)
#
# Uses ConvTranspose2d as the adjoint of the stride-2 Conv2d in DWT.
# For orthogonal Haar wavelet this gives exact inverse.
# Setting LH=HL=HH=0 and applying IDWT yields the proper
# low-pass filtered reconstruction — more principled than
# bilinear interpolation which ignores wavelet structure.
# ============================================================

class LearnableHaarIDWT(nn.Module):
    def __init__(self):
        super().__init__()
        rec_lo = _HAAR_DEC_LO.clone()
        self.rec_lo = nn.Parameter(rec_lo.unsqueeze(0), requires_grad=True)

    def _build_ll_kernel(self, c):
        lo = self.rec_lo.squeeze(0)
        ll = _outer(lo, lo)
        ll_kernel = ll.unsqueeze(1).repeat(c, 1, 1, 1)
        return ll_kernel

    def forward(self, ll, target_size):
        b, c, h, w = ll.shape
        ll_kernel = self._build_ll_kernel(c)
        x = F.conv_transpose2d(ll, ll_kernel, stride=2,
                               groups=c, output_size=target_size)
        return x


# ============================================================
# LearnableHaarIDWT_Full: Inverse DWT from all 4 subbands
#
# Reconstruction filters for orthogonal Haar:
#   rec_lo = dec_lo.flip() = [1/√2, 1/√2]   (symmetric, same)
#   rec_hi = dec_hi.flip() = [1/√2, -1/√2]  (time-reversed analysis high-freq)
#
# 4 reconstruction kernels (outer products):
#   LL: rec_lo ⊗ rec_lo   LH: rec_hi ⊗ rec_lo
#   HL: rec_lo ⊗ rec_hi   HH: rec_hi ⊗ rec_hi
#
# ConvTranspose2d with stride=2 reconstructs to full resolution.
# ============================================================

class LearnableHaarIDWT_Full(nn.Module):
    def __init__(self):
        super().__init__()
        rec_lo = _HAAR_DEC_LO.clone()
        rec_hi = _HAAR_DEC_HI.clone().flip(-1)
        self.rec_lo = nn.Parameter(rec_lo.unsqueeze(0), requires_grad=True)
        self.rec_hi = nn.Parameter(rec_hi.unsqueeze(0), requires_grad=True)

    def _build_rec_kernel(self, c):
        lo = self.rec_lo.squeeze(0)
        hi = self.rec_hi.squeeze(0)
        ll = _outer(lo, lo)
        lh = _outer(hi, lo)
        hl = _outer(lo, hi)
        hh = _outer(hi, hi)
        filt = torch.stack([ll, lh, hl, hh], 0)
        rec_kernel = filt.repeat(c, 1, 1)
        rec_kernel = rec_kernel.unsqueeze(dim=1)
        return rec_kernel

    def forward(self, ll, lh, hl, hh, target_size=None):
        b, c, h_half, w_half = ll.shape
        subbands = torch.stack([ll, lh, hl, hh], dim=2).reshape(
            b, 4 * c, h_half, w_half)
        rec_kernel = self._build_rec_kernel(c)
        x = F.conv_transpose2d(subbands, rec_kernel, stride=2, groups=c)
        if target_size is not None and x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear',
                              align_corners=False)
        return x


# ============================================================
# GlobalBranch: Full subband reconstruction with direction-aware
# high-frequency processing
#
# Flow:
#   DWT(x) → LL, LH, HL, HH
#   LL → Conv1×1            ← global structure
#   LH → DWConv(1,3)        ← horizontal edge enhancement
#   HL → DWConv(3,1)        ← vertical edge enhancement
#   HH → Conv1×1 + Tanh     ← diagonal texture (lightweight)
#   → IDWT(LL',LH',HL',HH') → x_recon
#   → x_recon × SE(x_recon) → x_se
#   → x_se + FLA(x_se)
#
# Direction-aware rationale:
#   LH captures horizontal edges → DWConv(1,3) enhances along
#     the horizontal direction where edges exist
#   HL captures vertical edges → DWConv(3,1) enhances along
#     the vertical direction where edges exist
#   HH captures diagonal texture → Conv1×1 is isotropic and
#     lightweight; Tanh bounds output to prevent noise explosion
#
# SE after IDWT (not before):
#   SE operates on full-resolution reconstructed signal where
#   both structure (from LL) and edges (from LH/HL/HH) are
#   present → can properly judge "which channels matter" for
#   the combined signal
# ============================================================

class GlobalBranch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwt = LearnableHaarDWT(level=1)
        self.idwt = LearnableHaarIDWT_Full()

        self.ll_conv = nn.Conv2d(dim, dim, 1)

        self.lh_conv = nn.Conv2d(dim, dim, kernel_size=(1, 3),
                                 padding=(0, 1), groups=dim, bias=False)
        self.hl_conv = nn.Conv2d(dim, dim, kernel_size=(3, 1),
                                 padding=(1, 0), groups=dim, bias=False)
        self.hh_conv = nn.Conv2d(dim, dim, 1)
        self.hh_act = nn.Tanh()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.fla = FocusedLinearAttention(dim, dim)

    def forward(self, x):
        target_size = x.shape[2:]
        ll, (lh, hl, hh) = self.dwt(x)

        ll_proc = self.ll_conv(ll)
        lh_proc = self.lh_conv(lh)
        hl_proc = self.hl_conv(hl)
        hh_proc = self.hh_act(self.hh_conv(hh))

        x_recon = self.idwt(ll_proc, lh_proc, hl_proc, hh_proc,
                            target_size=target_size)

        x_se = x_recon * self.se(x_recon)

        return x_se + self.fla(x_se)


# ============================================================
# DSAWACGA_v6_Mixer: Local(DSA) + Global(LL→SE→FLA)
#
#   conv_init(dim→2dim) → chunk(2)
#     Local:  x_local + DSA(x_local)           ← residual + multi-scale local
#     Global: LL → SE → FLA                    ← serial L2→L1→L3
#   Cat → GELU → SE_CA → ca_conv
# ============================================================

class DSAWACGA_v6_Mixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.conv_init = nn.Conv2d(dim, dim * 2, 1)

        self.local_mixer = DSA(dim)

        self.global_mixer = GlobalBranch(dim)

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

        x_local = x_local + self.local_mixer(x_local)
        x_global = self.global_mixer(x_global)

        x = torch.cat([x_local, x_global], dim=1)
        x = self.gelu(x)
        x = x * self.ca(x)
        x = self.ca_conv(x)
        return x


# ============================================================
# DSAWACGA_v6_FFN: multi-scale DWConv FFN with identity residual
# ============================================================

class DSAWACGA_v6_FFN(nn.Module):
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
        h = self.conv_init(x)
        h = list(torch.split(h, self.dim_sp, dim=1))
        h[1] = self.dw1(h[1])
        h[2] = self.dw2(h[2])
        h[3] = self.dw3(h[3])
        h = torch.cat(h, dim=1)
        h = self.gelu(h)
        h = self.conv_fina(h)
        return x + h


# ============================================================
# DSAWACGAv6Block: Transformer-like two-stage block
# ============================================================

class DSAWACGAv6Block(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim

        self.norm1 = norm_layer(dim)
        self.mixer = DSAWACGA_v6_Mixer(dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

        self.norm2 = norm_layer(dim)
        self.ffn = DSAWACGA_v6_FFN(dim)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = self.mixer(self.norm1(x)) * self.beta + x
        x = self.ffn(self.norm2(x)) * self.gamma + x
        return x


# ============================================================
# C2f_DSAWACGA_v6: C2f container with DSAWACGAv6Block
# ============================================================

class C2f_DSAWACGA_v6(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DSAWACGAv6Block(self.c) for _ in range(n))