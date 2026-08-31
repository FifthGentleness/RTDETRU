# HybridEncoderP2SPDOKM ported from rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder_p2_spd_okm.py
# (config: rtdetr_r18vd_6x_visdrone_p2_spd_okm.yml, encoder HybridEncoderP2SPDOKM)
#
# Porting rules (structure kept 1:1 with rtdetr_pytorch, no channel changes):
#   - P2/P3/P4 are consumed raw (input_proj[0..2] = Identity, only P5 gets
#     Conv1x1(bias=False)+BN proj to 256), handled by the yaml layers.
#   - AIFI (1 layer, dff=1024, 8 heads, dropout=0) is handled by the yaml AIFI module.
#   - FPN/PAN blocks use ultralytics RepC3: mathematically equivalent to the
#     rtdetr_pytorch CSPRepLayer (50/50 CSP split + additive fusion;
#     RepVggBlock == RepConv with bn=False) and supports model.fuse().
#   - CCFFP2 is the multi-input module replicating the base-version CCFF part:
#       SPDConv(P2 64->128) -> concat[p2_spd, y4_up, P3] = 512
#       -> cv1(512->512) -> split[128, 384] -> BottleNect(128) on innovation half
#       -> cat -> cv2(512->512) -> RepC3(512->256) -> f3
#   - BottleNect is the VERBATIM port of the OKNet bottleneck (strictly aligned
#     with OKNet original source): FCA -> SCA -> FGM global branch +
#     4 depthwise large-kernel convs, residual x + ..., ReLU, out_conv.
#     NOTE the differences vs the v6 CCFFBlock:
#       * fft/ifft norm='backward' (OKNet original), not 'ortho'
#       * NO BatchNorm before the final ReLU
#       * NO DCFM / FreqScale / small_fuse local branches
#
# Hyperparams from the base yml: large_kernel=31, split_ratio=0.25,
# expansion=0.5, depth_mult=1, act='silu', hidden_dim=256
# (CCFFP2 defaults below match the yml).

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import RepC3, get_activation
from ..modules.conv import Conv
from .hybrid_encoder_p2_spd_okm_fs_v6 import _SPDConv

__all__ = ['CCFFP2']


# ============================================================
# FGM: verbatim from rtdetr_pytorch hybrid_encoder_p2_spd_okm.py
# (OKNet original, fft/ifft norm='backward')
# ============================================================

class FGM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x2_fft = torch.fft.fft2(x2, norm='backward')
        out = x1 * x2_fft
        out = torch.fft.ifft2(out, s=fft_size, dim=(-2, -1), norm='backward')
        out = torch.abs(out)
        return out * self.alpha + x * self.beta


# ============================================================
# BottleNect: verbatim from rtdetr_pytorch hybrid_encoder_p2_spd_okm.py
# (OKNet bottleneck: FCA -> SCA -> FGM global branch + 4 dw large-kernel
#  convs, residual fusion, ReLU, out_conv; no BatchNorm inside)
# ============================================================

class BottleNect(nn.Module):
    def __init__(self, dim, large_kernel=31):
        super().__init__()
        pad = large_kernel // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)

        self.dw_1k = nn.Conv2d(dim, dim, kernel_size=(1, large_kernel), padding=(0, pad), stride=1, groups=dim)
        self.dw_k1 = nn.Conv2d(dim, dim, kernel_size=(large_kernel, 1), padding=(pad, 0), stride=1, groups=dim)
        self.dw_kk = nn.Conv2d(dim, dim, kernel_size=large_kernel, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

        # sca
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # fca
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)

        # fca
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca)

        # sca
        x_att_sca = self.conv(self.pool(x_fca))
        x_sca = x_att_sca * x_fca
        x_sca = self.fgm(x_sca)

        out = x + self.dw_1k(out) + self.dw_k1(out) + self.dw_kk(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)


# ============================================================
# CCFFP2: yaml-facing module for the base-version CCFF P2 fusion
# ============================================================

class CCFFP2(nn.Module):
    """Replicates the CCFF part of rtdetr_pytorch HybridEncoderP2SPDOKM.

    Inputs (from yaml): [p2 (raw backbone P2, 64ch), p3 (raw backbone P3, 128ch), y4 (256ch)]
    Output: f3 (hidden_dim=256ch, P3 stride)

    Data flow (identical to rtdetr_pytorch hybrid_encoder_p2_spd_okm.py):
        p2_spd = SPDConv(p2)                      64 -> 128
        y4_up  = upsample(y4, x2, nearest)        256
        ccff_input = cat([p2_spd, y4_up, p3])    128 + 256 + 128 = 512
        mixed = cv1(512 -> 512)
        split -> [innovation 128, identity 384]
        innovation_out = BottleNect(128, large_kernel=31)
        fused = cat -> 512 -> cv2(512 -> 512)
        f3 = RepC3(512 -> 256, n=3, e=0.5)
    """

    def __init__(self, ch_p2, ch_p3, ch_y4, hidden_dim=256,
                 large_kernel=31, split_ratio=0.25,
                 act='silu', expansion=0.5, depth_mult=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.ccff_spd_conv = _SPDConv(ch_p2, ch_p3, act=act)
        ccff_concat_ch = ch_p3 * 2 + hidden_dim
        self.split_channels = int(ccff_concat_ch * split_ratio)
        self.remaining_channels = ccff_concat_ch - self.split_channels

        self.ccff_cv1 = Conv(ccff_concat_ch, ccff_concat_ch, 1, 1, act=get_activation(act))
        self.ccff_innovation = BottleNect(self.split_channels, large_kernel=large_kernel)
        self.ccff_cv2 = Conv(ccff_concat_ch, ccff_concat_ch, 1, 1, act=get_activation(act))
        # CSPRepLayer(512->256, n=3, expansion=0.5) == RepC3(512->256, n=3, e=0.5)
        assert act == 'silu'
        self.ccff_fuse_block = RepC3(ccff_concat_ch, hidden_dim,
                                      n=round(3 * depth_mult), e=expansion)

    def forward(self, x):
        p2, p3, y4 = x

        p2_spd = self.ccff_spd_conv(p2)
        y4_up = F.interpolate(y4, scale_factor=2., mode='nearest')
        ccff_input = torch.concat([p2_spd, y4_up, p3], dim=1)

        mixed = self.ccff_cv1(ccff_input)
        ok_branch, identity = torch.split(mixed, [self.split_channels, self.remaining_channels], dim=1)
        innovation_out = self.ccff_innovation(ok_branch)
        fused = torch.cat([innovation_out, identity], dim=1)
        fused = self.ccff_cv2(fused)
        f3 = self.ccff_fuse_block(fused)

        return f3
