# DSADOC_v11: Based on DSADOC_v10 with cross_gate mechanism removed.
#
# Changes from v10:
#   - Removed: cross_gate (Conv1x1 + Sigmoid global gating)
#   - Removed: GAP on fdsa/fdcc for gate signal generation
#   - Reason: Global gating is spatial-blind (harmful for small objects),
#     redundant with dsadoc_fusion (Conv1x1+BN+SiLU already does channel
#     selection), and Sigmoid(0,1) is purely inhibitory.
#   - Effect: fdsa and fdcc are concatenated directly into dsadoc_fusion
#     without prior channel-wise suppression. dsadoc_fusion alone handles
#     DSA-DCC cross-channel interaction and compression (2cr -> cr).
#   - Parameter savings per stage (cr=ch_out//2):
#       cross_gate Conv1x1: 2cr * 2cr = 2*cr^2 params removed
#       Stage2(64): 2048, Stage3(128): 8192, Stage4(256): 32768, Stage5(512): 131072
#       Total: ~174K params removed

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import BasicBlock
from ..modules.conv import Conv

__all__ = ['DSADOC_v11', 'DSADOCv11BasicBlock', 'BlocksDSADOCv11']


class HaarDWT(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, [0, W % 2, 0, H % 2], mode='reflect')
            _, _, H, W = x.shape

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x10 + x01 + x11) * 0.5
        lh = (x00 - x10 + x01 - x11) * 0.5
        hl = (x00 + x10 - x01 - x11) * 0.5
        hh = (x00 - x10 - x01 + x11) * 0.5

        return ll, lh, hl, hh


class HaarIDWT(nn.Module):
    def forward(self, ll, lh, hl, hh):
        x00 = ll + lh + hl + hh
        x01 = ll + lh - hl - hh
        x10 = ll - lh + hl - hh
        x11 = ll - lh - hl + hh

        B, C, H2, W2 = ll.shape
        out = torch.empty(B, C, H2 * 2, W2 * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out


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


class DSADOC_v11(nn.Module):
    """DSADOC_v11: v10 with cross_gate removed.

    Half-channel split design, replaces branch2a:
    - First half  -> DSA (multi-scale spatial) + DCC (wavelet calibration + FiLM)
    - Second half -> Conv3x3 for local feature extraction
    - DSA and DCC are concatenated directly into dsadoc_fusion (Conv1x1+BN+SiLU)
      without global gating, letting the fusion layer handle cross-channel interaction.
    """

    def __init__(self, dim):
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

        # --- DCC: Global pooling ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # --- DCC: DWT / IDWT ---
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        # --- DCC: Omnidirectional spatial feature for LL (all 4 subbands) ---
        self.spatial_mix_ll = nn.Sequential(
            nn.Conv2d(cr * 4, cr, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        # --- DCC: High-freq direction-specific spatial features ---
        self.spatial_mix_lh = nn.Sequential(
            nn.Conv2d(cr * 3, cr, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.spatial_mix_hl = nn.Sequential(
            nn.Conv2d(cr * 3, cr, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.spatial_mix_hh = nn.Sequential(
            nn.Conv2d(cr * 3, cr, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        # --- DCC: Global descriptor scalar fusion (alpha, beta, gamma) ---
        self.fusion_conv = nn.Conv2d(cr * 3, 3, 1, bias=False)

        # --- DCC: Wavelet-domain subband calibration ---
        self.cal_ll = nn.Conv2d(cr, cr, 1, bias=False)
        self.cal_lh = nn.Conv2d(cr, cr, 1, bias=False)
        self.cal_hl = nn.Conv2d(cr, cr, 1, bias=False)
        self.cal_hh = nn.Conv2d(cr, cr, 1, bias=False)

        # --- DCC: Normalization for IDWT output ---
        self.cal_norm = nn.GroupNorm(num_groups=1, num_channels=cr, eps=1e-5)

        # --- DCC: FiLM parameter generation with learnable scale ---
        self.film_gen = nn.Conv2d(cr, cr * 2, 1, bias=True)
        self.film_scale = nn.Parameter(torch.zeros(1))

        # --- DSADOC internal fusion: Concat -> Conv1x1+BN+SiLU ---
        # No cross_gate: fdsa and fdcc are concatenated directly
        self.dsadoc_fusion = Conv(cr * 2, half_dim, 1, 1)

        # --- Conv path: aligned with branch2a (Conv+BN+ReLU) ---
        self.conv_path = Conv(half_dim, half_dim, 3, 1, act=nn.ReLU())

    def forward(self, x):
        x_dsadoc, x_conv = x.chunk(2, dim=1)

        # === DSADOC path (first half) ===
        xs = x_dsadoc

        # DSA branch
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

        # DCC branch
        favg = self.gap(xs)
        fmax = self.gmp(xs)

        ll, lh, hl, hh = self.dwt(xs)

        dwt_ll = self.gap(ll)
        global_cat = torch.cat([favg, fmax, dwt_ll], dim=1)
        weights = F.softmax(self.fusion_conv(global_cat), dim=1)
        alpha = weights[:, 0:1]
        beta = weights[:, 1:2]
        gamma = weights[:, 2:3]
        global_feat = alpha * favg + beta * fmax + gamma * dwt_ll

        all_cat = torch.cat([ll, lh, hl, hh], dim=1)
        spatial_ll = self.spatial_mix_ll(all_cat)

        high_cat = torch.cat([lh, lh, hh], dim=1)
        spatial_lh = self.spatial_mix_lh(high_cat)
        spatial_hl = self.spatial_mix_hl(high_cat)
        spatial_hh = self.spatial_mix_hh(high_cat)

        cal_ll = self.cal_ll(ll) + global_feat + spatial_ll
        cal_lh = self.cal_lh(lh) + global_feat + spatial_lh
        cal_hl = self.cal_hl(hl) + global_feat + spatial_hl
        cal_hh = self.cal_hh(hh) + global_feat + spatial_hh

        cal_map = self.idwt(cal_ll, cal_lh, cal_hl, cal_hh)

        if cal_map.shape[2] != xs.shape[2] or cal_map.shape[3] != xs.shape[3]:
            cal_map = cal_map[:, :, :xs.shape[2], :xs.shape[3]]

        cal_map = self.cal_norm(cal_map)

        film_params = self.film_gen(cal_map)
        gamma_raw, beta_raw = film_params.chunk(2, dim=1)

        scale = self.film_scale
        film_gamma = 1.0 + scale * torch.tanh(gamma_raw)
        film_beta = scale * torch.tanh(beta_raw)

        fdcc = xs * film_gamma + film_beta

        # DSADOC internal fusion (no cross_gate)
        fused = torch.cat([fdsa, fdcc], dim=1)
        dsadoc_out = self.dsadoc_fusion(fused)

        # === Conv path (second half) ===
        conv_out = self.conv_path(x_conv)

        # === Concat two paths, branch2b will fuse channels ===
        out = torch.cat([dsadoc_out, conv_out], dim=1)

        return out


class DSADOCv11BasicBlock(BasicBlock):
    """BasicBlock with DSADOC_v11 replacing branch2a."""
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        del self.branch2a
        self.dsadcc = DSADOC_v11(ch_out)

    def forward(self, x):
        out = self.dsadcc(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.act(out)
        return out


class BlocksDSADOCv11(nn.Module):
    """Stage container: vanilla BasicBlocks + last-block DSADOC_v11."""

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
                self.blocks.append(
                    DSADOCv11BasicBlock(
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