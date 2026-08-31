# HybridEncoderP2SPDOKMFSV6 ported from rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder_p2_spd_okm_fs_v6.py
# (config: rtdetr_r18vd_6x_visdrone_p2_spd_okm_fs_v6.yml, encoder HybridEncoderP2SPDOKMFSV6)
#
# Porting rules (structure kept 1:1 with rtdetr_pytorch, no channel changes):
#   - P2 is consumed raw (input_proj[0..2] = Identity, only P5 gets 1x1+BN proj to 256).
#   - AIFI (1 layer, dff=1024, 8 heads) is handled by the yaml AIFI module (equivalent port).
#   - FPN/PAN blocks use CSPRepLayer/RepVggBlock ported VERBATIM from
#     rtdetr_pytorch hybrid_encoder.py (RTDETRU's RepC3 is NOT equivalent:
#     additive fusion + RepConv, vs concat + RepVggBlock).
#   - CCFFP2V6 is the multi-input module replicating the v6 CCFF part:
#       SPDConv(P2 64->128) -> concat[p2_spd, y4_up, P3] = 512
#       -> cv1(512->512) -> split[128, 384] -> CCFFBlock(128) on innovation half
#       -> cat -> cv2(512->512) -> CSPRepLayer(512->256) -> f3
#   - v6 signature change kept: PDCBlock has NO internal residual at stride=1
#     (DCFM outputs pure differential features).
#   - CCFFBlock channels: sc = 128 (int(512 * 0.25)), FreqScale dim=128 group=16.
#
# NOTE on names: SPDConv / FGM / StarReLU already exist in RTDETRU
# extra_modules/block.py, so they are NOT exported (kept module-internal here).
# Only CSPRepLayer (used in yaml fpn/pan) and CCFFP2V6 (yaml CCFF module) are exported.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import get_activation
from ..modules.conv import Conv

__all__ = ['CSPRepLayer', 'CCFFP2V6']


# ============================================================
# SPDConv (internal): space-to-depth + 3x3 Conv
# ============================================================

class _SPDConv(nn.Module):
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
# PDC (partial difference convolution) for DCFM
# ============================================================

class PDCConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, pdc_type='cv', theta=0.875):
        super().__init__()
        self.pdc_type = pdc_type
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
        if self.pdc_type == 'cv':
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        elif self.pdc_type == 'cd':
            weights_c = self.weight.sum(dim=[2, 3], keepdim=True) * self.theta
            yc = F.conv2d(x, weights_c, stride=self.stride, padding=0, groups=self.groups)
            y = F.conv2d(x, self.weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
            return y - yc
        else:
            raise ValueError(f'Unknown pdc_type: {self.pdc_type}')


class PDCBlock(nn.Module):
    """v6: no internal residual at stride=1, output is a pure differential feature.

    v5: y = conv2(ReLU(BN(PDCConv(x)))) + x   (with residual)
    v6: y = conv2(ReLU(BN(PDCConv(x))))       (no residual, pure differential)
    stride>1 still keeps the shortcut for dimension alignment.
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
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.pool(x)
            y = y + self.shortcut(x)
        return y


class DCFM(nn.Module):
    def __init__(self, channels, act='relu', theta=0.875):
        super().__init__()
        self.pdc_cv = PDCBlock(pdc_type='cv', inplane=channels, ouplane=channels, theta=theta)
        self.pdc_cd = PDCBlock(pdc_type='cd', inplane=channels, ouplane=channels, theta=theta)

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
# OKNet large-kernel conv
# ============================================================

class OKNetLargeKernel(nn.Module):
    def __init__(self, dim, large_kernel=31):
        super().__init__()
        pad = large_kernel // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.dw_1k = nn.Conv2d(dim, dim, kernel_size=(1, large_kernel), padding=(0, pad), stride=1, groups=dim)
        self.dw_k1 = nn.Conv2d(dim, dim, kernel_size=(large_kernel, 1), padding=(pad, 0), stride=1, groups=dim)
        self.dw_kk = nn.Conv2d(dim, dim, kernel_size=large_kernel, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.in_conv(x)
        out = self.dw_1k(out) + self.dw_k1(out) + self.dw_kk(out) + self.dw_11(out)
        out = self.norm(out)
        out = self.act(out)
        return out


# ============================================================
# FreqScale (internal StarReLU helper)
# ============================================================

class _StarReLU(nn.Module):
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


class FreqScale(nn.Module):
    def __init__(self, dim, group=8, num_filters=4, base_size=8,
                 reweight_ratio=0.25, init_scale=1e-5):
        super().__init__()
        assert dim % group == 0, f'FreqScale: dim({dim}) must be divisible by group({group})'
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
            weight = F.interpolate(weight, size=x_rfft.shape[2:4], mode='bicubic', align_corners=True)
        weight = torch.einsum('bgf,fchw->bgchw', routing, weight)
        weight = weight.reshape(B, C, RH, RW)

        x_rfft = torch.view_as_complex(torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1))
        out = torch.fft.irfft2(x_rfft, s=(H, W), dim=(2, 3), norm='ortho')

        return out


# ============================================================
# SCA / FGM (internal)
# ============================================================

class SCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x_att = self.conv(self.pool(x))
        return x_att * x


class _FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='ortho')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(2, 3), norm='ortho')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta


# ============================================================
# CCFFBlock: local(DCFM) + large-kernel(OKM) + global(FreqScale->SCA->FGM)
# ============================================================

class CCFFBlock(nn.Module):
    """v6: PDCBlock without internal residual, DCFM outputs pure differential features.

    Local branch data flow:
        x -> DCFM(x) = a*conv_cv(x) + (1-a)*conv_cd(x)   (pure differential, no x)
        cat([x, DCFM(x)]) -> small_fuse                     (x appears exactly once)
    """

    def __init__(self, channels, large_kernel=31,
                 fs_group=16, fs_num_filters=4, fs_base_size=14,
                 dcfm_theta=0.875,
                 fs_reweight_ratio=0.25, fs_init_scale=1e-5):
        super().__init__()
        sc = channels

        self.dcfm = DCFM(sc, theta=dcfm_theta)
        self.small_fuse = Conv(sc * 2, sc, 1, 1)

        self.large_kernel_conv = OKNetLargeKernel(sc, large_kernel=large_kernel)

        fs_group_sc = max(1, fs_group)
        self.freq_scale = FreqScale(sc, group=fs_group_sc, num_filters=fs_num_filters,
                                    base_size=fs_base_size,
                                    reweight_ratio=fs_reweight_ratio, init_scale=fs_init_scale)
        self.sca = SCA(sc)
        self.fgm = _FGM(sc)
        self.global_out = Conv(sc, sc, 1, 1)

        self.fuse_out = Conv(sc, sc, 1, 1)

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
# RepVggBlock / CSPRepLayer: ported VERBATIM from
# rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py
# ============================================================

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = Conv(ch_in, ch_out, 3, 1, act=False)
        self.conv2 = Conv(ch_in, ch_out, 1, 1, act=False)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: Conv):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1, act=get_activation(act))
        self.conv2 = Conv(in_channels, hidden_channels, 1, 1, act=get_activation(act))
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = Conv(hidden_channels, out_channels, 1, 1, act=get_activation(act))
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# ============================================================
# CCFFP2V6: yaml-facing module for the v6 CCFF P2 fusion
# ============================================================

class CCFFP2V6(nn.Module):
    """Replicates the CCFF part of rtdetr_pytorch HybridEncoderP2SPDOKMFSV6.

    Inputs (from yaml): [p2 (raw backbone P2, 64ch), p3 (raw backbone P3, 128ch), y4 (256ch)]
    Output: f3 (hidden_dim=256ch, P3 stride)

    Data flow (identical to v6):
        p2_spd = SPDConv(p2)                      64 -> 128
        y4_up  = upsample(y4, x2, nearest)        256
        ccff_input = cat([p2_spd, y4_up, p3])    128 + 256 + 128 = 512
        mixed = cv1(512 -> 512)
        split -> [innovation 128, identity 384]
        innovation_out = CCFFBlock(128)
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

        self.ccff_cv1 = Conv(ccff_concat_ch, ccff_concat_ch, 1, 1, act=get_activation(act))
        self.ccff_innovation = CCFFBlock(self.split_channels,
                                         large_kernel=large_kernel,
                                         fs_group=fs_group,
                                         fs_num_filters=fs_num_filters,
                                         fs_base_size=fs_base_size,
                                         dcfm_theta=dcfm_theta,
                                         fs_reweight_ratio=fs_reweight_ratio,
                                         fs_init_scale=fs_init_scale)
        self.ccff_cv2 = Conv(ccff_concat_ch, ccff_concat_ch, 1, 1, act=get_activation(act))
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
