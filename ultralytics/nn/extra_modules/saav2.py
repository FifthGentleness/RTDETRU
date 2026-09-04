# SAAv2 Backbone: CSP-style Scale-Aware Attentional Backbone v2.
#
# Based on SAA from SME-DETR, redesigned with CSFH backbone framework.
# Key changes from original SAAv2:
#   - Adopted CSFH (CSFPR-RTDETR) backbone framework:
#     Conv(3→64,s=2) → Conv(64→128,s=2) → C2f_DPF(128)
#     → Conv(128→256,s=2) → C2f_DPF(256)
#     → Conv(256→384,s=2) → C2f_DPF(384)
#     → Conv(384→384,s=2) → C2f_DPF(384)×1
#   - C2f_DPF: C2f container with DPF replacing SFHF_Block
#     cv1(c1→2c, 1×1) → split(2 chunks) → [DPF(c)×n] → cat → cv2(→c2, 1×1)
#   - DPF block unchanged (DSA + DCA fusion)
#   - Per-stage DCA kernel size K = 11 + 2*dpf_n, deeper stages use larger K
#
# Legacy SAAStagev2 / SAABackbonev2 kept for backward compatibility.

import torch
import torch.nn as nn

from ..modules.block import C2f
from ..modules.conv import Conv
from .saa import DPF

__all__ = ['DPFBlock', 'C2f_DPF', 'DownSample', 'SAAStagev2', 'SAABackbonev2']


class DPFBlock(nn.Module):
    """DPF Block: full Transformer-like block with DPF attention + FFN.

    Mirrors SFHF_Block's two-stage structure (mixer + ffn) while using
    SAAv2's DPF (DSA + DCA) as the attention mechanism:

      Stage 1 (Attention):  x' = DPF(Norm1(x)) * β + x
      Stage 2 (FFN):        x'' = FFN(Norm2(x')) * γ + x'

    Compared to SFHF_Block:
      - DPF replaces SFHF_Mixer (local dilated conv + global Fourier + SE)
        with DSA (dynamic multi-scale) + DCA (directional channel attention)
      - DPF_FFN replaces SFHF_FFN (multi-scale DWConv 3/5/7)
        with multi-scale DWConv 3/5/7/9 (4 branches)

    Args:
        dim: Number of input/output channels.
        dpf_n: DCA kernel size parameter (K = 11 + 2*dpf_n).
        scales: DSA multi-scale kernel sizes.
        norm_layer: Normalization layer (default BatchNorm2d).
    """

    def __init__(self, dim, dpf_n=3, scales=(3, 5, 7, 9, 11),
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim

        self.norm1 = norm_layer(dim)
        self.dpf = DPF(dim, n=dpf_n, scales=scales)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

        self.norm2 = norm_layer(dim)
        self.ffn = DPF_FFN(dim)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = self.dpf(self.norm1(x)) * self.beta + x
        x = self.ffn(self.norm2(x)) * self.gamma + x
        return x


class DPF_FFN(nn.Module):
    """Feed-Forward Network for DPFBlock.

    Multi-scale depthwise convolutions with GELU activation,
    mirroring SFHF_FFN's structure but with 4 DWConv branches (3/5/7/9):

      Conv1x1(dim→2dim) → Split(4 chunks, each dim/2) → DWConv(3/5/7/9) → Cat → GELU → Conv1x1(2dim→dim)

    Args:
        dim: Number of input/output channels.
    """

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
        self.dw4 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=9,
                              padding=4, groups=self.dim_sp)

        self.gelu = nn.GELU()
        self.conv_fina = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        x = self.conv_init(x)
        x1, x2, x3, x4 = torch.split(x, self.dim_sp, dim=1)
        x = torch.cat([self.dw1(x1), self.dw2(x2),
                        self.dw3(x3), self.dw4(x4)], dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)
        return x


class C2f_DPF(C2f):
    """C2f with DPFBlock replacing Bottleneck.

    CSFH-style C2f container using DPFBlock as the core block:
      cv1(c1→2c, 1×1) → split(2 chunks) → [DPFBlock(c)×n] → cat → cv2(→c2, 1×1)

    DPFBlock is a full Transformer-like block:
      Stage 1 (Attention):  x' = DPF(Norm1(x)) * β + x
      Stage 2 (FFN):        x'' = FFN(Norm2(x')) * γ + x'

    This mirrors CSFH_Block's structure (SFHF_Mixer + SFHF_FFN) while
    using SAAv2's DPF (DSA + DCA) as the attention mechanism.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of DPFBlock repeats.
        dpf_n: DCA kernel size parameter (K = 11 + 2*dpf_n).
        shortcut: Whether to use shortcut connection in Bottleneck (unused here).
        g: Groups (unused, kept for C2f interface).
        e: Expansion ratio for hidden channels (default 0.5).
        scales: DSA multi-scale kernel sizes.
    """

    def __init__(self, c1, c2, n=1, dpf_n=3, shortcut=False, g=1, e=0.5,
                 scales=(3, 5, 7, 9, 11)):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DPFBlock(self.c, dpf_n=dpf_n, scales=scales) for _ in range(n))


class DownSample(nn.Module):
    """Dedicated spatial downsampling module.

    Conv k=3 s=2 with same channels, reducing spatial resolution by 2x
    while preserving channel dimension.

    Args:
        channels: Number of input/output channels (unchanged).
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = Conv(channels, channels, 3, 2)

    def forward(self, x):
        return self.conv(x)


class SAAStagev2(nn.Module):
    """CSP-style Stage for SAAv2 Backbone.

    Pipeline:
      F_{l-1} -> DownSample(s=2) -> Conv -> Split -> [Identity, DPF] -> Concat -> Fuse -> F_l

    - DownSample: Dedicated spatial downsampling (Conv k=3 s=2, ch_in -> ch_in)
    - Conv: Dimension transition convolution (ch_in -> 2*ch_out)
    - Channel Split: Split into X_{l-1}^{(1)} (Identity) and X_{l-1}^{(2)} (DPF)
    - Identity: X_{l-1}^{(1)} direct pass-through
    - DPF: X_{l-1}^{(2)} through Dual-Path Fusion block
    - Concat + Fuse: Merge branches and reduce channels (2*ch_out -> ch_out)

    Args:
        ch_in: Input channel count.
        ch_out: Output channel count.
        n: DCA kernel size parameter (K = 11 + 2*n).
        scales: DSA multi-scale kernel sizes.
    """

    def __init__(self, ch_in, ch_out, n=1, scales=(3, 5, 7, 9, 11)):
        super().__init__()
        self.downsample = DownSample(ch_in)
        self.conv = Conv(ch_in, ch_out * 2, 3, 1)
        self.dpf = DPF(ch_out, n=n, scales=scales)
        self.fuse = Conv(ch_out * 2, ch_out, 1, 1)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv(x)
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.dpf(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse(x)
        return x


class SAABackbonev2(nn.Module):
    """SAAv2 Backbone: Stem + 4 CSP-style Stages.

    Outputs 4 feature maps at different resolutions for FPN/MDAP:
      F1 (S2, /4), F2 (S3, /8), F3 (S4, /16), F4 (S5, /32)

    Args:
        stem_channels: Stem output channel count.
        channels: Tuple of 4 stage output channel counts.
        n_list: Tuple of 4 DCA n parameters for each stage.
        scales: DSA multi-scale kernel sizes.
    """

    def __init__(self, stem_channels=32, channels=(64, 128, 256, 512),
                 n_list=(0, 1, 2, 3), scales=(3, 5, 7, 9, 11)):
        super().__init__()
        self.stem = Conv(3, stem_channels, 3, 2)

        self.stages = nn.ModuleList()
        ch_in = stem_channels
        for ch_out, n_val in zip(channels, n_list):
            self.stages.append(SAAStagev2(ch_in, ch_out, n=n_val, scales=scales))
            ch_in = ch_out

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


def count_params_and_flops(model, input_size=(1, 3, 640, 640)):
    """Count parameters and FLOPs for SAAv2 backbone.

    Args:
        model: The SAAv2 backbone model.
        input_size: Input tensor size (batch, channels, height, width).

    Returns:
        dict: Dictionary with params, params_m, flops_g keys.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {
        'total_params': total_params,
        'total_params_m': total_params / 1e6,
        'trainable_params': trainable_params,
        'trainable_params_m': trainable_params / 1e6,
    }

    try:
        from thop import profile
        x = torch.randn(*input_size)
        flops, params = profile(model, inputs=(x,), verbose=False)
        result['flops'] = flops
        result['flops_g'] = flops / 1e9
        result['thop_params'] = params
        result['thop_params_m'] = params / 1e6
    except ImportError:
        result['flops'] = None
        result['flops_g'] = None

    return result


def analytical_params(stem_channels=32, channels=(64, 128, 256, 512),
                      n_list=(0, 1, 2, 3), scales=(3, 5, 7, 9, 11)):
    """Analytically compute parameter count for SAAv2 backbone.

    Returns per-component and total parameter counts.
    """
    n_scales = len(scales)

    def conv_params(c1, c2, k):
        return c1 * c2 * k * k + 2 * c2

    def dsa_params(c, n_scales):
        p = conv_params(c, c, 3)
        p += n_scales * c * 9
        p += n_scales
        p += conv_params(c, c, 1)
        return p

    def dca_params(c, n):
        k = 11 + 2 * n
        p = c * k
        p += c * k
        p += c * 9
        p += c * c
        return p

    def dpf_params(c, n, n_scales):
        return dsa_params(c, n_scales) + dca_params(c, n) + conv_params(c, c, 1)

    def downsample_params(c):
        return conv_params(c, c, 3)

    def stage_params(ch_in, ch_out, n, n_scales):
        p_down = downsample_params(ch_in)
        p_conv = conv_params(ch_in, ch_out * 2, 3)
        p_dpf = dpf_params(ch_out, n, n_scales)
        p_fuse = conv_params(ch_out * 2, ch_out, 1)
        return {
            'downsample': p_down,
            'conv': p_conv,
            'dpf': p_dpf,
            'fuse': p_fuse,
            'total': p_down + p_conv + p_dpf + p_fuse,
        }

    info = {}

    p_stem = conv_params(3, stem_channels, 3)
    info['stem'] = p_stem

    ch_in = stem_channels
    total = p_stem
    for i, (ch_out, n_val) in enumerate(zip(channels, n_list)):
        s = stage_params(ch_in, ch_out, n_val, n_scales)
        info[f'stage{i + 1}'] = s
        total += s['total']
        ch_in = ch_out

    info['total'] = total
    info['total_m'] = total / 1e6
    return info


if __name__ == '__main__':
    print('=' * 70)
    print('SAAv2 Backbone — Architecture Verification')
    print('=' * 70)

    model = SAABackbonev2()
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    print(f'\nInput:  {x.shape}')
    for i, o in enumerate(outputs):
        print(f'F{i + 1} (S{i + 2}): {o.shape}')

    print('\n' + '=' * 70)
    print('SAAv2 Backbone — Parameter Count (Analytical)')
    print('=' * 70)
    info = analytical_params()
    print(f'  Stem:       {info["stem"]:>10,}')
    for i in range(1, 5):
        s = info[f'stage{i}']
        print(f'  Stage {i}:    {s["total"]:>10,}  '
              f'(Down={s["downsample"]:,}  Conv={s["conv"]:,}  '
              f'DPF={s["dpf"]:,}  Fuse={s["fuse"]:,})')
    print(f'  {"─" * 50}')
    print(f'  Total:      {info["total"]:>10,}  ({info["total_m"]:.2f}M)')

    print('\n' + '=' * 70)
    print('SAAv2 Backbone — Parameter Count (torch.numel)')
    print('=' * 70)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Total params: {total_params:,}  ({total_params / 1e6:.2f}M)')

    print('\n' + '=' * 70)
    print('SAAv2 Backbone — FLOPs (thop)')
    print('=' * 70)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(x,), verbose=False)
        print(f'  FLOPs:  {flops:,}  ({flops / 1e9:.2f}G)')
        print(f'  Params: {params:,}  ({params / 1e6:.2f}M)')
    except ImportError:
        print('  Install thop for FLOPs: pip install thop')

    print('\n' + '=' * 70)
    print('SAAStagev2 — Per-Stage Detail')
    print('=' * 70)
    ch_in = 32
    channels = (64, 128, 256, 512)
    n_list = (0, 1, 2, 3)
    for i, (ch_out, n_val) in enumerate(zip(channels, n_list)):
        stage = SAAStagev2(ch_in, ch_out, n=n_val)
        p = sum(p.numel() for p in stage.parameters())
        k = 11 + 2 * n_val
        print(f'  Stage {i + 1}: ch_in={ch_in}, ch_out={ch_out}, n={n_val} (K={k}), '
              f'params={p:,} ({p / 1e6:.2f}M)')
        ch_in = ch_out