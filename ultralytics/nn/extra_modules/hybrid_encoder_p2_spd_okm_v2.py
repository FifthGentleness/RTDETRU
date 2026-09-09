# HybridEncoderP2SPDOKMv2: adapted for DSAWACGAv5/v7 backbone channels
#
# Key change vs original P2SPDOKM (ResNet18 backbone):
#   - SPDConv output is DECOUPLED from ch_p3 via new parameter spd_out_ch
#   - This allows P2→128, P3→256 (raw) while SPDConv still outputs 128
#   - concat = spd_out_ch + ch_p3 + hidden_dim = 128 + 256 + 256 = 640
#   - split_ratio=0.25 → innovation=160, identity=480
#   - BottleNect(160) as the innovation block
#
# Design rationale:
#   - P2=128ch: DSAWACGA backbone outputs 128ch at P2, no projection needed
#   - P3=256ch: kept raw to preserve full intermediate-scale information
#   - P4=256ch: projected from 384ch via Conv1x1 in yaml
#   - P5=256ch: projected from 384ch via Conv1x1 in yaml
#   - SPDConv(128→128): SPD 128*4=512 → Conv(512→128,3x3) → 128ch
#   - FPN/PAN: all 256ch, Concat=512, same as original RT-DETR

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import RepC3, get_activation
from ..modules.conv import Conv
from .hybrid_encoder_p2_spd_okm_fs_v6 import _SPDConv

__all__ = ['CCFFP2V2']


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

        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)

        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca)

        x_att_sca = self.conv(self.pool(x_fca))
        x_sca = x_att_sca * x_fca
        x_sca = self.fgm(x_sca)

        out = x + self.dw_1k(out) + self.dw_k1(out) + self.dw_kk(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)


class CCFFP2V2(nn.Module):
    """CCFFP2v2: adapted for DSAWACGA backbone channels.

    Key change vs CCFFP2: SPDConv output channel is decoupled from ch_p3
    via the spd_out_ch parameter.

    Example with DSAWACGAv5 backbone (P2=128, P3=256, Y4=256):
        SPDConv(128→128) → 128ch
        concat = 128 + 256 + 256 = 640
        split(ratio=0.25) → [innovation 160, identity 480]
        BottleNect(160, large_kernel=31)
        RepC3(640→256) → f3: 256ch

    Args:
        ch_p2: P2 input channels (from backbone, e.g. 128)
        ch_p3: P3 input channels (from backbone, e.g. 256)
        ch_y4: Y4 input channels (from FPN lateral, e.g. 256)
        hidden_dim: output channels (default 256)
        spd_out_ch: SPDConv output channels (default None → auto=ch_p3)
            When DSAWACGA backbone has P2=128, P3=256, set spd_out_ch=128
            to keep SPDConv output at 128 instead of 256
        large_kernel: large kernel size for BottleNect (default 31)
        split_ratio: innovation/identity split ratio (default 0.25)
        act: activation function name (default 'silu')
        expansion: RepC3 expansion ratio (default 0.5)
        depth_mult: RepC3 depth multiplier (default 1)
    """

    def __init__(self, ch_p2, ch_p3, ch_y4, hidden_dim=256,
                 spd_out_ch=None,
                 large_kernel=31, split_ratio=0.25,
                 act='silu', expansion=0.5, depth_mult=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        if spd_out_ch is None:
            spd_out_ch = ch_p3

        self.ccff_spd_conv = _SPDConv(ch_p2, spd_out_ch, act=act)
        ccff_concat_ch = spd_out_ch + ch_p3 + hidden_dim
        self.split_channels = int(ccff_concat_ch * split_ratio)
        self.remaining_channels = ccff_concat_ch - self.split_channels

        self.ccff_cv1 = Conv(ccff_concat_ch, ccff_concat_ch, 1, 1, act=get_activation(act))
        self.ccff_innovation = BottleNect(self.split_channels, large_kernel=large_kernel)
        self.ccff_cv2 = Conv(ccff_concat_ch, ccff_concat_ch, 1, 1, act=get_activation(act))
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