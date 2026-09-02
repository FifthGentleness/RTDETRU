# DSADOC_v11: DSA upgraded to DSAWACGAv2-style 6-branch, DOC (DCC) preserved.
#
# Changes from original DSADOC_v11 (v10-based):
#   - DSA: 5-branch (3,5,7,dilated3,Scharr) → 6-branch (1,3,5,7,9,11)
#     Removed Scharr (fixed kernel, competes with learnable branches).
#     Removed dilated DWConv (replaced by larger standard kernels DW9x9/DW11x11
#     for cleaner multi-scale coverage without gridding artifacts).
#   - DOC (DCC) preserved: HaarDWT → subband calibration → HaarIDWT → FiLM
#   - Fusion preserved: cat([fdsa, fdcc]) → Conv1x1+BN+SiLU
#   - DSA spatial-adaptive weights: 5-way → 6-way

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


class DSADOC_v11(nn.Module):
    """DSADOC_v11: DSA upgraded to 6-branch, DOC (DCC) preserved.

    Half-channel split design, replaces branch2a:
    - First half  -> DSA (6-branch multi-scale DWConv) + DCC (wavelet calibration + FiLM)
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
        self.dsadoc_fusion = Conv(cr * 2, half_dim, 1, 1)

        # --- Conv path: aligned with branch2a (Conv+BN+ReLU) ---
        self.conv_path = Conv(half_dim, half_dim, 3, 1, act=nn.ReLU())

    def forward(self, x):
        x_dsadoc, x_conv = x.chunk(2, dim=1)

        # === DSADOC path (first half) ===
        xs = x_dsadoc

        # DSA branch (6-branch)
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

        high_cat = torch.cat([lh, hl, hh], dim=1)
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

        # DSADOC internal fusion
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