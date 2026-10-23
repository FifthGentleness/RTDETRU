# -*- coding: utf-8 -*-
# =====================================================================================
# DLFF: Difference-Large-kernel-Frequency Fusion Module
# -------------------------------------------------------------------------------------
# Paper-facing re-implementation of the P2SPDOKMFSV7 (CCFFP2V7) innovation module with
# a clean, self-consistent naming scheme. Structure is 1:1 identical to
# hybrid_encoder_p2_spd_okm_fs_v7.py (CCFFP2V7 / CCFFBlockV7), only names changed.
#
# Naming scheme (paper  <->  code  <->  legacy engineering name):
#   DLFF  Difference-Large-kernel-Frequency Fusion        CCFFP2V7          yaml-facing module
#   DLFB  DLF Fusion Block                                CCFFBlockV7       3-branch + identity block
#   S2D   Space-to-Depth Downsampling                     _SPDConv          lossless small-object downsample
#   DCB   Difference Convolution Branch                   DCFM              local detail branch
#         - VC   Vanilla Convolution path                 PDCBlock('cv')    standard conv path
#         - CDC  Central Difference Convolution path      PDCBlock('cd')    gradient/detail path
#         - AWF  Adaptive Weight Fusion                   attention_fc      softmax gate over VC/CDC
#   LKB   Large-Kernel Branch                             dw_1k/k1/kk/11    long-range context branch
#   SPT   Shared Point-wise Transform (1x1 Conv + GELU)   in_conv           shared by LKB & FGB
#   FGB   Frequency-Guided Branch                         FCA->SCA->FGM     global frequency branch
#         - FCA  Frequency Channel Attention              fca               channel gate in FFT domain
#         - SCA  Spatial-Channel Attention                sca               pooled channel attention
#         - FGM  Frequency Global Modulation (2-stage)    fgm               paper-facing 3rd module
#                   stage-M (Modulation)                                  spectrum interaction + gated residual
#                   stage-R (Refinement, warmup)                          content-routed complex re-weighting
#   IR    Identity Residual                               identity path     out = x + DCB + LKB + FGB
#
# DLFB data flow:
#   local  = DCB(x)                      pure differential enhancement, direct add
#   s      = SPT(x)                      shared 1x1 + GELU
#   large  = LKB(s)                      1xK, Kx1, KxK, 1x1 depthwise, summed
#   fgb    = FGM(SCA(FCA(s))) + FSR(...) FGM effective from start, FSR warmup refinement
#   out    = ReLU(x + local + large + fgb) -> 1x1 Conv     (identity residual preserved)
# =====================================================================================

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import RepC3, get_activation
from ..modules.conv import Conv

__all__ = ['DLFF', 'DLFB', 'S2D', 'DCB', 'LKB', 'FGB', 'FCA', 'SCA', 'FGM', 'FSR']


# ============================================================
# S2D: Space-to-Depth Downsampling (was _SPDConv)
# ============================================================

class S2D(nn.Module):
    """Space-to-Depth (SPD) downsampling: no information loss for small objects.

    Rearranges H/2 x W/2 x 4C then applies a 3x3 conv to restore channel dim.
    """

    def __init__(self, in_channels, out_channels, act='silu'):
        super().__init__()
        self.conv = Conv(in_channels * 4, out_channels, 3, 1, act=get_activation(act))

    def forward(self, x):
        x1 = x[:, :, 0::2, 0::2]
        x2 = x[:, :, 1::2, 0::2]
        x3 = x[:, :, 0::2, 1::2]
        x4 = x[:, :, 1::2, 1::2]
        x_spd = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x_spd)


# ============================================================
# Difference convolution primitives for DCB
# ============================================================

class DiffConv(nn.Module):
    """Unified difference convolution: 'vc' (vanilla) or 'cdc' (central difference)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, mode='vc', theta=0.875):
        super().__init__()
        self.mode = mode
        self.theta = theta
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        if self.mode == 'vc':
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        elif self.mode == 'cdc':
            # central difference: subtract the channel-mean (theta-scaled) response
            weights_c = self.weight.sum(dim=[2, 3], keepdim=True) * self.theta
            yc = F.conv2d(x, weights_c, stride=self.stride, padding=0, groups=self.groups)
            y = F.conv2d(x, self.weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
            return y - yc
        else:
            raise ValueError(f'Unknown diff conv mode: {self.mode}')


class DiffBlock(nn.Module):
    """DiffConv + BN -> ReLU -> 1x1 conv; stride=1 keeps pure differential output.

    stride>1 keeps a shortcut only for dimension alignment (maxpool + 1x1).
    """

    def __init__(self, mode, in_channels, out_channels, stride=1, theta=0.875):
        super().__init__()
        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        self.conv1 = nn.Sequential(
            DiffConv(in_channels, in_channels, kernel_size=3, padding=1,
                     groups=in_channels, mode=mode, theta=theta),
            nn.BatchNorm2d(in_channels),
        )
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                     padding=0, bias=False)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.pool(x)
            y = y + self.shortcut(x)
        return y


# ============================================================
# DCB: Difference Convolution Branch (was DCFM)
# ============================================================

class DCB(nn.Module):
    """Local detail enhancement branch: VC path + CDC path + Adaptive Weight Fusion.

    forward(x) = AWF(softmax) * [ VC(x), CDC(x) ]
    The CDC path is a pure differential feature (no identity term at stride=1),
    so the branch output is a pure additive enhancement for the IR sum.
    """

    def __init__(self, channels, act='relu', theta=0.875):
        super().__init__()
        self.vc = DiffBlock(mode='vc', in_channels=channels, out_channels=channels, theta=theta)
        self.cdc = DiffBlock(mode='cdc', in_channels=channels, out_channels=channels, theta=theta)

        # AWF: adaptive weight fusion (softmax gate over the two paths)
        self.awf = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 2, kernel_size=1)
        )

    def forward(self, x):
        y_vc = self.vc(x)
        y_cdc = self.cdc(x)

        gate = self.awf(torch.cat([y_vc, y_cdc], dim=1))
        gate = F.softmax(gate, dim=1)

        return y_vc * gate[:, 0:1, :, :] + y_cdc * gate[:, 1:2, :, :]


# ============================================================
# LKB: Large-Kernel Branch (was the inline dw_1k/k1/kk/11 convs)
# ============================================================

class LKB(nn.Module):
    """Long-range context branch: parallel depthwise large-kernel convs.

    forward(s) = dw_{1xK}(s) + dw_{Kx1}(s) + dw_{KxK}(s) + dw_{1x1}(s)
    Input s is expected to be the SPT (shared point-wise transform) output.
    """

    def __init__(self, dim, large_kernel=31):
        super().__init__()
        pad = large_kernel // 2
        self.dw_1k = nn.Conv2d(dim, dim, kernel_size=(1, large_kernel),
                                     padding=(0, pad), stride=1, groups=dim)
        self.dw_k1 = nn.Conv2d(dim, dim, kernel_size=(large_kernel, 1),
                                     padding=(pad, 0), stride=1, groups=dim)
        self.dw_kk = nn.Conv2d(dim, dim, kernel_size=large_kernel,
                                     padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1,
                                     padding=0, stride=1, groups=dim)

    def forward(self, s):
        return self.dw_1k(s) + self.dw_k1(s) + self.dw_kk(s) + self.dw_11(s)


# ============================================================
# FGB sub-modules: FCA / SCA / FGM (FSR folded in as FGM stage-R, kept for ablation)
# ============================================================

class FCA(nn.Module):
    """Frequency Channel Attention: pooled channel gate applied in the FFT domain."""

    def __init__(self, dim):
        super().__init__()
        self.gate_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0,
                                         stride=1, groups=1, bias=True)
        self.gate_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        gate = self.gate_conv(self.gate_pool(x))
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft = gate * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='ortho')
        return torch.abs(x_fca)


class SCA(nn.Module):
    """Spatial-Channel Attention: GAP -> 1x1 conv -> channel-wise recalibration."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0,
                                    stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.conv(self.pool(x)) * x


class FGM(nn.Module):
    """Frequency Global Modulation: two-stage (paper-facing single module).

    Stage-M (Modulation): FFT-domain multiplicative interaction of two 1x1 convs,
        with a per-channel gated residual (alpha=0 / beta=1 at init -> identity,
        effective from the very start of training).

        Stage-R (Refinement): content-routed learnable complex re-weighting
        (the former FSR / FreqScale). Complex weights are initialized with
        std=fs_init_scale (near zero) so this stage warms up gradually.

        forward(x) = Stage-M(x) + Stage-R(Stage-M(x))

    Set use_refine=False to ablate Stage-R.
    """

    def __init__(self, dim, fs_group=16, fs_num_filters=4, fs_base_size=14,
                 fs_reweight_ratio=0.25, fs_init_scale=1e-2, use_refine=True):
        super().__init__()
        self.use_refine = use_refine

        # ===== Stage-M: spectrum interaction + gated residual =====
        self.conv_1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.conv_2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

        # ===== Stage-R: content-routed spectrum refinement (warmup) =====
        if self.use_refine:
            self.refine = FSR(dim, group=fs_group, num_filters=fs_num_filters,
                              base_size=fs_base_size,
                              reweight_ratio=fs_reweight_ratio,
                              init_scale=fs_init_scale)

    def forward(self, x):
        # Stage-M
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x2_fft = torch.fft.fft2(x2, norm='ortho')
        out = x1 * x2_fft
        out = torch.fft.ifft2(out, dim=(2, 3), norm='ortho')
        out = torch.abs(out)
        x_mod = out * self.alpha + x * self.beta

        # Stage-R (warmup refinement)
        if self.use_refine:
            x_mod = x_mod + self.refine(x_mod)
        return x_mod



class _StarReLU(nn.Module):
    """StarReLU helper for FSR routing."""

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                        requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                       requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class FSR(nn.Module):
    """Frequency-Scale Refinement: content-routed learnable complex re-weighting.

    init_scale is small (1e-2 in DLFB) so FSR contributes as a warmup refinement
    on top of FGM without disturbing early training.
    """

    def __init__(self, dim, group=8, num_filters=4, base_size=8,
                 reweight_ratio=0.25, init_scale=1e-5):
        super().__init__()
        assert dim % group == 0, f'FSR: dim({dim}) must be divisible by group({group})'
        self.dim = dim
        self.group = group
        self.num_filters = num_filters
        self.base_size = base_size
        self.filter_size = base_size // 2 + 1

        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        reweight_hidden = max(1, int(reweight_ratio * dim))
        self.reweight = nn.Sequential(
            nn.Linear(dim, reweight_hidden, bias=False),
            _StarReLU(),
            nn.Linear(reweight_hidden, group * num_filters, bias=False)
        )

        self.complex_weights = nn.Parameter(
            torch.empty(num_filters, dim // group, base_size, self.filter_size,
                        dtype=torch.float32)
        )
        nn.init.trunc_normal_(self.complex_weights, std=init_scale)

    def forward(self, x):
        B, C, H, W = x.shape

        x_in = self.in_conv(x)

        x_rfft = torch.fft.rfft2(x_in.to(torch.float32), dim=(2, 3), norm='ortho')
        _, _, RH, RW = x_rfft.shape

        x_perm = x_in.permute(0, 2, 3, 1)
        routing = self.reweight(x_perm.mean(dim=(1, 2)))
        routing = routing.view(B, self.group, self.num_filters).tanh_()

        weight = self.complex_weights
        if not weight.shape[2:4] == x_rfft.shape[2:4]:
            weight = F.interpolate(weight, size=x_rfft.shape[2:4], mode='bicubic',
                                   align_corners=True)
        weight = torch.einsum('bgf,fchw->bgchw', routing, weight)
        weight = weight.reshape(B, C, RH, RW)

        x_rfft = torch.view_as_complex(
            torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1))
        out = torch.fft.irfft2(x_rfft, s=(H, W), dim=(2, 3), norm='ortho')

        return out


# ============================================================
# FGB: Frequency-Guided Branch (serial FCA -> SCA -> FGM)
# ============================================================

class FGB(nn.Module):
    """Global frequency branch: FCA -> SCA -> FGM (three paper-facing modules).

    global = FGM(SCA(FCA(s)))
    FGM stage-M is effective from training start; its stage-R refines with warmup.
    """

    def __init__(self, dim, fs_group=16, fs_num_filters=4, fs_base_size=14,
                 fs_reweight_ratio=0.25, fs_init_scale=1e-2, use_refine=True):
        super().__init__()
        self.fca = FCA(dim)
        self.sca = SCA(dim)
        self.fgm = FGM(dim, fs_group=fs_group, fs_num_filters=fs_num_filters,
                       fs_base_size=fs_base_size,
                       fs_reweight_ratio=fs_reweight_ratio,
                       fs_init_scale=fs_init_scale, use_refine=use_refine)

    def forward(self, s):
        return self.fgm(self.sca(self.fca(s)))



# ============================================================
# DLFB: DLF Fusion Block (was CCFFBlockV7)
# ============================================================

class DLFB(nn.Module):
    """DLF Fusion Block: DCB (local) + LKB (context) + FGB (global) + IR (identity).

    Data flow:
        local  = DCB(x)                       pure differential, direct add
        s      = SPT(x)                       shared 1x1 Conv + GELU
        large  = LKB(s)                       large-kernel context
        global = FGB(s)                       FCA -> SCA -> FGM (2-stage)
        out    = ReLU(x + local + large + global) -> 1x1 Conv

    The identity residual (IR) keeps a direct gradient path from input to output
    (out = x + ...), which is the key fix of this design over the V6 variant.
    """

    def __init__(self, channels, large_kernel=31,
                 fs_group=16, fs_num_filters=4, fs_base_size=14,
                 dcfm_theta=0.875,
                 fs_reweight_ratio=0.25, fs_init_scale=1e-2):
        super().__init__()
        sc = channels

        # ===== Local branch: DCB =====
        self.dcb = DCB(sc, theta=dcfm_theta)

        # ===== SPT: shared point-wise transform for LKB & FGB =====
        self.spt = nn.Sequential(
            nn.Conv2d(sc, sc, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )

        # ===== Context branch: LKB =====
        self.lkb = LKB(sc, large_kernel=large_kernel)

        # ===== Global branch: FGB =====
        self.fgb = FGB(sc, fs_group=fs_group, fs_num_filters=fs_num_filters,
                       fs_base_size=fs_base_size,
                       fs_reweight_ratio=fs_reweight_ratio,
                       fs_init_scale=fs_init_scale)

        # ===== Output: IR fusion -> ReLU -> 1x1 conv =====
        self.act = nn.ReLU()
        self.out_conv = nn.Conv2d(sc, sc, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        # local differential enhancement
        local = self.dcb(x)

        # shared transform
        s = self.spt(x)

        # large-kernel context
        large = self.lkb(s)

        # frequency-guided global
        global_out = self.fgb(s)

        # identity residual fusion
        out = x + local + large + global_out
        out = self.act(out)
        return self.out_conv(out)


# ============================================================
# DLFF: yaml-facing Difference-Large-kernel-Frequency Fusion module (was CCFFP2V7)
# ============================================================

class DLFF(nn.Module):
    """DLFF: multi-scale fusion of P2 / P3 / Y4 into the enhanced f3 level.

    Supports spd_out_ch for backbones whose P2 channel count differs (e.g. DSAWACGAv5).

    Example with R18 backbone (P2=64, P3=128, Y4=256):
        S2D(64 -> 128)
        concat = 128 + 256 + 128 = 512
        split(ratio=0.25) -> [DLFB 128, IR 384]
        DLFB(128, large_kernel=31)
        RepC3(512 -> 256) -> f3: 256ch

    Args:
        ch_p2: P2 input channels (backbone)
        ch_p3: P3 input channels (backbone)
        ch_y4: Y4 input channels (FPN lateral)
        hidden_dim: output channels (default 256)
        spd_out_ch: S2D output channels (default None -> auto=ch_p3)
        large_kernel: LKB large kernel size (default 31)
        split_ratio: DLFB / IR channel split ratio (default 0.25)
        fs_group / fs_num_filters / fs_base_size / fs_reweight_ratio / fs_init_scale: FSR params
        dcfm_theta: DCB CDC theta (default 0.875)
        act: activation name (default 'silu')
        expansion: RepC3 expansion (default 0.5)
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

        self.s2d = S2D(ch_p2, spd_out_ch, act=act)
        dlff_concat_ch = spd_out_ch + ch_p3 + hidden_dim
        self.split_channels = int(dlff_concat_ch * split_ratio)
        self.remaining_channels = dlff_concat_ch - self.split_channels

        self.cv1 = Conv(dlff_concat_ch, dlff_concat_ch, 1, 1,
                        act=get_activation(act))
        self.dlfb = DLFB(
            self.split_channels,
            large_kernel=large_kernel,
            fs_group=fs_group,
            fs_num_filters=fs_num_filters,
            fs_base_size=fs_base_size,
            dcfm_theta=dcfm_theta,
            fs_reweight_ratio=fs_reweight_ratio,
            fs_init_scale=fs_init_scale
        )
        self.cv2 = Conv(dlff_concat_ch, dlff_concat_ch, 1, 1,
                        act=get_activation(act))
        assert act == 'silu'
        self.fuse = RepC3(dlff_concat_ch, hidden_dim,
                          n=round(3 * depth_mult), e=expansion)

    def forward(self, x):
        p2, p3, y4 = x

        p2_s2d = self.s2d(p2)
        y4_up = F.interpolate(y4, scale_factor=2., mode='nearest')
        dlff_input = torch.concat([p2_s2d, y4_up, p3], dim=1)

        mixed = self.cv1(dlff_input)
        dlfb_branch, identity = torch.split(
            mixed, [self.split_channels, self.remaining_channels], dim=1)
        dlfb_out = self.dlfb(dlfb_branch)
        fused = torch.cat([dlfb_out, identity], dim=1)
        fused = self.cv2(fused)
        f3 = self.fuse(fused)

        return f3


