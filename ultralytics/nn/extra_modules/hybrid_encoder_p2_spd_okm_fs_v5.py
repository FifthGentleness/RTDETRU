# HybridEncoderP2SPDOKMFSV5 ported from rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder_p2_spd_okm_fs_v5.py
# (config: rtdetr_r18vd_6x_visdrone_p2_spd_okm_fs_v5.yml, encoder HybridEncoderP2SPDOKMFSV5)
#
# v5 vs v6 (the ONLY structural difference):
#   PDCBlock keeps the internal residual at stride=1:
#       v5: y = conv2(ReLU(BN(PDCConv(x)))) + x   (with residual)
#       v6: y = conv2(ReLU(BN(PDCConv(x))))        (no residual, pure differential)
#   Everything else (SPDConv / OKNetLargeKernel / FreqScale / SCA / FGM /
#   CCFFBlock data flow / hyperparams / channel layout) is identical to v6.
#
# To guarantee 1:1 fidelity while avoiding ~300 duplicated lines, this file
# reuses the v6 ported components verbatim and only redefines the parts whose
# behavior differs (PDCBlock -> DCFM -> CCFFBlock -> CCFFP2V5 container).

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import ConvNormLayer
from .hybrid_encoder_p2_spd_okm_fs_v6 import (
    CSPRepLayer,
    OKNetLargeKernel,
    FreqScale,
    SCA,
    _FGM,
    PDCConv,
    _SPDConv,
)

__all__ = ['CCFFP2V5']


# ============================================================
# PDCBlock (v5): WITH internal residual at stride=1
# ============================================================

class PDCBlockV5(nn.Module):
    """v5: keeps the internal residual.

    v5: y = conv2(ReLU(BN(PDCConv(x)))) + x   (with residual)
    (v6 removed this residual; see hybrid_encoder_p2_spd_okm_fs_v6.py)
    """

    def __init__(self, pdc_type, inplane, ouplane, stride=1, theta=0.875):
        super().__init__()
        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)

        self.conv1 = nn.Sequential(
            PDCConv(inplane, inplane, kernel_size=3, padding=1, groups=inplane, pdc_type=pdc_type, theta=theta),
            nn.BatchNorm2d(inplane),
        )
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        if self.stride > 1:
            identity = self.pool(identity)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            identity = self.shortcut(identity)
        y = y + identity
        return y


class DCFMV5(nn.Module):
    def __init__(self, channels, act='relu', theta=0.875):
        super().__init__()
        self.pdc_cv = PDCBlockV5(pdc_type='cv', inplane=channels, ouplane=channels, theta=theta)
        self.pdc_cd = PDCBlockV5(pdc_type='cd', inplane=channels, ouplane=channels, theta=theta)

        self.attention_fc = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 2, kernel_size=1)
        )

    def forward(self, x):
        diff_cv = self.pdc_cv(x)
        diff_cd = self.pdc_cd(x)

        diff_stack = torch.cat([diff_cv, diff_cd], dim=1)
        attention_weights = self.attention_fc(diff_stack)
        attention_weights = F.softmax(attention_weights, dim=1)

        diff_cv_weighted = diff_cv * attention_weights[:, 0:1, :, :]
        diff_cd_weighted = diff_cd * attention_weights[:, 1:2, :, :]

        fused_features = diff_cv_weighted + diff_cd_weighted
        return fused_features


# ============================================================
# CCFFBlock (v5): local(DCFM v5) + large-kernel(OKM) + global(FreqScale->SCA->FGM)
# ============================================================

class CCFFBlockV5(nn.Module):
    """v5 CCFFBlock: identical data flow to v6, but DCFM uses PDCBlock WITH residual."""

    def __init__(self, channels, large_kernel=31,
                 fs_group=16, fs_num_filters=4, fs_base_size=14,
                 dcfm_theta=0.875,
                 fs_reweight_ratio=0.25, fs_init_scale=1e-5):
        super().__init__()
        sc = channels

        self.dcfm = DCFMV5(sc, theta=dcfm_theta)
        self.small_fuse = ConvNormLayer(sc * 2, sc, 1, 1, act='silu')

        self.large_kernel_conv = OKNetLargeKernel(sc, large_kernel=large_kernel)

        fs_group_sc = max(1, fs_group)
        self.freq_scale = FreqScale(sc, group=fs_group_sc, num_filters=fs_num_filters,
                                    base_size=fs_base_size,
                                    reweight_ratio=fs_reweight_ratio, init_scale=fs_init_scale)
        self.sca = SCA(sc)
        self.fgm = _FGM(sc)
        self.global_out = ConvNormLayer(sc, sc, 1, 1, act='silu')

        self.fuse_out = ConvNormLayer(sc, sc, 1, 1, act='silu')

    def forward(self, x):
        small_orig = x
        small_dcfm = self.dcfm(x)
        small_cat = torch.cat([small_orig, small_dcfm], dim=1)
        small_out = self.small_fuse(small_cat)

        large_out = self.large_kernel_conv(x)

        f_freq = self.freq_scale(x)
        f_sca = self.sca(f_freq)
        f_cross = self.fgm(f_sca)
        global_out = self.global_out(f_cross)

        out = small_out + large_out + global_out
        out = self.fuse_out(out)

        return out


# ============================================================
# CCFFP2V5: yaml-facing module for the v5 CCFF P2 fusion
# ============================================================

class CCFFP2V5(nn.Module):
    """Replicates the CCFF part of rtdetr_pytorch HybridEncoderP2SPDOKMFSV5.

    Inputs (from yaml): [p2 (raw backbone P2, 64ch), p3 (raw backbone P3, 128ch), y4 (256ch)]
    Output: f3 (hidden_dim=256ch, P3 stride)

    Data flow (identical to v5):
        p2_spd = SPDConv(p2)                      64 -> 128
        y4_up  = upsample(y4, x2, nearest)        256
        ccff_input = cat([p2_spd, y4_up, p3])    128 + 256 + 128 = 512
        mixed = cv1(512 -> 512)
        split -> [innovation 128, identity 384]
        innovation_out = CCFFBlockV5(128)         (PDCBlock WITH residual)
        fused = cat -> 512 -> cv2(512 -> 512)
        f3 = CSPRepLayer(512 -> 256)
    """

    def __init__(self, ch_p2, ch_p3, ch_y4, hidden_dim=256,
                 large_kernel=31, fs_group=16, fs_num_filters=4, fs_base_size=14,
                 split_ratio=0.25, dcfm_theta=0.875,
                 fs_reweight_ratio=0.25, fs_init_scale=1e-5,
                 act='silu', expansion=0.5, depth_mult=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.ccff_spd_conv = _SPDConv(ch_p2, ch_p3, act=act)
        ccff_concat_ch = ch_p3 * 2 + hidden_dim
        self.split_channels = int(ccff_concat_ch * split_ratio)
        self.remaining_channels = ccff_concat_ch - self.split_channels

        self.ccff_cv1 = ConvNormLayer(ccff_concat_ch, ccff_concat_ch, 1, 1, act=act)
        self.ccff_innovation = CCFFBlockV5(self.split_channels,
                                            large_kernel=large_kernel,
                                            fs_group=fs_group,
                                            fs_num_filters=fs_num_filters,
                                            fs_base_size=fs_base_size,
                                            dcfm_theta=dcfm_theta,
                                            fs_reweight_ratio=fs_reweight_ratio,
                                            fs_init_scale=fs_init_scale)
        self.ccff_cv2 = ConvNormLayer(ccff_concat_ch, ccff_concat_ch, 1, 1, act=act)
        self.ccff_fuse_block = CSPRepLayer(ccff_concat_ch, hidden_dim,
                                           round(3 * depth_mult), act=act, expansion=expansion)

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
