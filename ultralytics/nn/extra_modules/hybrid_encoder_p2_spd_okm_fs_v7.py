# HybridEncoderP2SPDOKMFSV7: identity-residual-preserved version of V6
#
# V6 regression root cause: CCFFBlock lost the identity residual connection
# that BottleNect (base) relies on. In BottleNect, out = x + large_kernel + global,
# so x has a direct path to the output and gradients flow unimpeded.
# In V6 CCFFBlock, out = fuse_out(small_fuse([x, DCFM(x)]) + OKNetLK(x) + global(x)),
# so x must pass through two random 1x1 convs, destroying information flow.
#
# V7 fixes this by restoring the identity residual while keeping DCFM + FreqScale:
#
#   base: out = x + large_kernel + FCA->SCA->FGM
#         (identity residual OK, but no local enhancement)
#
#   V6:   out = 1x1( 1x1([x, DCFM(x)]) + OKNetLK(x) + FreqScale->SCA->FGM(x) )
#         (identity residual LOST, information flow degraded)
#
#   V7:   out = x + DCFM(x) + large_kernel(in_conv(x)) + FCA->SCA->FGM->FreqScale
#         (identity residual RESTORED + local enhancement from DCFM + FreqScale refinement)
#
# Key design decisions:
#   1. Identity residual: out = x + local + large + global
#      -> x has direct path, gradient = I + ..., information preserved
#   2. DCFM as direct incremental enhancement:
#      -> DCFM output is pure differential, added directly (no cat+1x1 mixing)
#   3. Large kernel shares in_conv with global branch (like base BottleNect):
#      -> out = in_conv(x) is shared input for both large kernel and FCA
#   4. Global branch serial: FCA->SCA->FGM (effective from start, kaiming init)
#      + FreqScale as second-stage refinement on FGM output (warmup gradually)
#      -> At training start: global ≈ FCA->SCA->FGM + 0 (same as base)
#      -> After warmup: global ≈ FCA->SCA->FGM + FreqScale refinement
#   5. FreqScale init_scale=1e-2 (vs V6's 1e-5):
#      -> Faster warmup, FreqScale contributes meaningfully earlier
#   6. FGM norm='ortho' (vs base's 'backward'):
#      -> More numerically stable FFT normalization
#
# CCFFP2V7 supports spd_out_ch (from V2) for DSAWACGAv5 backbone compatibility.
#
# Information flow quality comparison (training初期 t≈0):
#   base: out ≈ x + small_noise + meaningful_channel_modulation  -> ~90%+ preserved
#   V6:   out ≈ random_1x1(random_mixed_x + noise + 0)          -> ~30-40% preserved
#   V7:   out ≈ x + diff_enhance + large_kernel + FCA->SCA->FGM -> ~90%+ preserved

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import RepC3, get_activation
from ..modules.conv import Conv
from .hybrid_encoder_p2_spd_okm_fs_v6 import (
    _SPDConv, DCFM, FreqScale, SCA, _FGM,
)

__all__ = ['CCFFP2V7']


# ============================================================
# CCFFBlockV7: identity-residual-preserved innovation block
# ============================================================

class CCFFBlockV7(nn.Module):
    """V7 innovation block: DCFM + FreqScale with identity residual preserved.

    Data flow:
        local_enhance = DCFM(x)                                       # pure differential, direct add
        out = in_conv(x)                                              # shared 1x1+GELU
        large_out = dw_1k(out) + dw_k1(out) + dw_kk(out) + dw_11(out) # large kernel
        x_fca = FCA(out) -> x_sca = SCA(x_fca) -> x_fgm = FGM(x_sca) # serial global
        x_freq = FreqScale(x_fgm)                                     # second-stage refinement
        global_out = x_fgm + x_freq                                   # FGM effective from start, FS warmup
        out = x + local_enhance + large_out + global_out              # IDENTITY RESIDUAL
        out = ReLU(out) -> out_conv(out)

    Key vs V6 CCFFBlock:
        - x has identity path to output (not through 1x1 mixing)
        - DCFM added directly as enhancement (not cat+1x1)
        - Large kernel shares in_conv with global (like base BottleNect)
        - Global branch serial: FCA->SCA->FGM effective from start
        - FreqScale as refinement on top of FGM (not replacement of FCA)
        - FreqScale init_scale=1e-2 (vs V6's 1e-5, faster warmup)
    """

    def __init__(self, channels, large_kernel=31,
                 fs_group=16, fs_num_filters=4, fs_base_size=14,
                 dcfm_theta=0.875,
                 fs_reweight_ratio=0.25, fs_init_scale=1e-2):
        super().__init__()
        sc = channels

        # ===== Local branch: DCFM =====
        self.dcfm = DCFM(sc, theta=dcfm_theta)

        # ===== Shared in_conv for large kernel and global =====
        self.in_conv = nn.Sequential(
            nn.Conv2d(sc, sc, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )

        # ===== Large kernel convs (same as base BottleNect) =====
        pad = large_kernel // 2
        self.dw_1k = nn.Conv2d(sc, sc, kernel_size=(1, large_kernel),
                                padding=(0, pad), stride=1, groups=sc)
        self.dw_k1 = nn.Conv2d(sc, sc, kernel_size=(large_kernel, 1),
                                padding=(pad, 0), stride=1, groups=sc)
        self.dw_kk = nn.Conv2d(sc, sc, kernel_size=large_kernel,
                                padding=pad, stride=1, groups=sc)
        self.dw_11 = nn.Conv2d(sc, sc, kernel_size=1,
                                padding=0, stride=1, groups=sc)

        # ===== Global branch: FCA -> SCA -> FGM -> FreqScale =====
        # FCA: frequency channel attention (kaiming init, effective from start)
        self.fac_conv = nn.Conv2d(sc, sc, kernel_size=1, padding=0,
                                  stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))

        # SCA: spatial channel attention
        self.sca = SCA(sc)

        # FGM: frequency global modulation (norm='ortho')
        self.fgm = _FGM(sc)

        # FreqScale: second-stage refinement on FGM output
        # init_scale=1e-2 (vs V6's 1e-5) for faster warmup
        fs_group_sc = max(1, fs_group)
        self.freq_scale = FreqScale(sc, group=fs_group_sc,
                                    num_filters=fs_num_filters,
                                    base_size=fs_base_size,
                                    reweight_ratio=fs_reweight_ratio,
                                    init_scale=fs_init_scale)

        # ===== Output =====
        self.act = nn.ReLU()
        self.out_conv = nn.Conv2d(sc, sc, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        # ===== Local enhancement: DCFM as direct incremental =====
        local_enhance = self.dcfm(x)

        # ===== Shared in_conv =====
        out = self.in_conv(x)

        # ===== Large kernel branch =====
        large_out = self.dw_1k(out) + self.dw_k1(out) + self.dw_kk(out) + self.dw_11(out)

        # ===== Global branch: FCA -> SCA -> FGM (serial, effective from start) =====
        # FCA: frequency channel attention
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='ortho')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='ortho')
        x_fca = torch.abs(x_fca)

        # SCA: spatial channel attention
        x_sca = self.sca(x_fca)

        # FGM: frequency global modulation
        x_fgm = self.fgm(x_sca)

        # FreqScale: second-stage refinement (warmup gradually)
        x_freq = self.freq_scale(x_fgm)

        # Global = FGM (effective from start) + FreqScale (warmup refinement)
        global_out = x_fgm + x_freq

        # ===== Identity residual fusion =====
        out = x + local_enhance + large_out + global_out
        out = self.act(out)
        return self.out_conv(out)


# ============================================================
# CCFFP2V7: yaml-facing module for the v7 CCFF P2 fusion
# ============================================================

class CCFFP2V7(nn.Module):
    """CCFFP2V7: V7 CCFF P2 fusion with identity-residual-preserved CCFFBlockV7.

    Supports spd_out_ch (from V2) for DSAWACGAv5 backbone compatibility.

    Example with DSAWACGAv5 backbone (P2=128, P3=256, Y4=256):
        SPDConv(128->128) -> 128ch
        concat = 128 + 256 + 256 = 640
        split(ratio=0.25) -> [innovation 160, identity 480]
        CCFFBlockV7(160, large_kernel=31)
        RepC3(640->256) -> f3: 256ch

    Example with R18 backbone (P2=64, P3=128, Y4=256):
        SPDConv(64->128) -> 128ch
        concat = 128 + 128 + 256 = 512
        split(ratio=0.25) -> [innovation 128, identity 384]
        CCFFBlockV7(128, large_kernel=31)
        RepC3(512->256) -> f3: 256ch

    Args:
        ch_p2: P2 input channels (from backbone, e.g. 128 for DSAWACGAv5)
        ch_p3: P3 input channels (from backbone, e.g. 256 for DSAWACGAv5)
        ch_y4: Y4 input channels (from FPN lateral, e.g. 256)
        hidden_dim: output channels (default 256)
        spd_out_ch: SPDConv output channels (default None -> auto=ch_p3)
            When DSAWACGA backbone has P2=128, P3=256, set spd_out_ch=128
        large_kernel: large kernel size for CCFFBlockV7 (default 31)
        split_ratio: innovation/identity split ratio (default 0.25)
        fs_group: FreqScale group count (default 16)
        fs_num_filters: FreqScale filter count (default 4)
        fs_base_size: FreqScale base size (default 14)
        dcfm_theta: DCFM theta parameter (default 0.875)
        fs_reweight_ratio: FreqScale reweight ratio (default 0.25)
        fs_init_scale: FreqScale init scale (default 1e-2, vs V6's 1e-5)
        act: activation function name (default 'silu')
        expansion: RepC3 expansion ratio (default 0.5)
        depth_mult: RepC3 depth multiplier (default 1)
    """

    def __init__(self, ch_p2, ch_p3, ch_y4, hidden_dim=256,
                 spd_out_ch=None,
                 large_kernel=31, split_ratio=0.25,
                 fs_group=16, fs_num_filters=4, fs_base_size=14,
                 dcfm_theta=0.875,
                 fs_reweight_ratio=0.25, fs_init_scale=1e-2,
                 act='silu', expansion=0.5, depth_mult=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        if spd_out_ch is None:
            spd_out_ch = ch_p3

        self.ccff_spd_conv = _SPDConv(ch_p2, spd_out_ch, act=act)
        ccff_concat_ch = spd_out_ch + ch_p3 + hidden_dim
        self.split_channels = int(ccff_concat_ch * split_ratio)
        self.remaining_channels = ccff_concat_ch - self.split_channels

        self.ccff_cv1 = Conv(ccff_concat_ch, ccff_concat_ch, 1, 1,
                             act=get_activation(act))
        self.ccff_innovation = CCFFBlockV7(
            self.split_channels,
            large_kernel=large_kernel,
            fs_group=fs_group,
            fs_num_filters=fs_num_filters,
            fs_base_size=fs_base_size,
            dcfm_theta=dcfm_theta,
            fs_reweight_ratio=fs_reweight_ratio,
            fs_init_scale=fs_init_scale
        )
        self.ccff_cv2 = Conv(ccff_concat_ch, ccff_concat_ch, 1, 1,
                             act=get_activation(act))
        assert act == 'silu'
        self.ccff_fuse_block = RepC3(ccff_concat_ch, hidden_dim,
                                      n=round(3 * depth_mult), e=expansion)

    def forward(self, x):
        p2, p3, y4 = x

        p2_spd = self.ccff_spd_conv(p2)
        y4_up = F.interpolate(y4, scale_factor=2., mode='nearest')
        ccff_input = torch.concat([p2_spd, y4_up, p3], dim=1)

        mixed = self.ccff_cv1(ccff_input)
        ok_branch, identity = torch.split(
            mixed, [self.split_channels, self.remaining_channels], dim=1)
        innovation_out = self.ccff_innovation(ok_branch)
        fused = torch.cat([innovation_out, identity], dim=1)
        fused = self.ccff_cv2(fused)
        f3 = self.ccff_fuse_block(fused)

        return f3