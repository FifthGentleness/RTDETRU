# RAAv2 Backbone: CSP-style Scale-Aware Attentional Backbone v2.
#
# Based on RAA (SAA) from SME-DETR, redesigned with CSPNet/YOLO pipeline.
# Key changes from RAA (SAA):
#   - Abandoned ResNet18-style stem (3 ConvNormLayers + MaxPool)
#   - Adopted YOLO-style single Conv stem
#   - Each stage uses CSP-style pipeline:
#     DownSample -> Conv -> Channel Split -> [Identity, DPF] -> Concat -> Fuse
#   - DPF block unchanged (DSA + DCA fusion)
#
# Architecture:
#   Stem (Conv k=3 s=2, 3->stem_ch) at P1/2
#   Stage 1: stem_ch->64   at P2/4  -> F1 (S2)  n=0, K=11
#   Stage 2: 64->128       at P3/8  -> F2 (S3)  n=1, K=13
#   Stage 3: 128->256      at P4/16 -> F3 (S4)  n=2, K=15
#   Stage 4: 256->512      at P5/32 -> F4 (S5)  n=3, K=17
#
# Per-stage pipeline (CSP-style):
#   F_{l-1} -> DownSample(s=2) -> Conv -> Split -> [Identity, DPF] -> Concat -> Fuse -> F_l
#   - DownSample: Dedicated spatial downsampling (Conv k=3 s=2, channels unchanged)
#   - Conv: Dimension transition convolution (ch_in -> 2*ch_out)
#   - Channel Split: X_{l-1}^{(1)} (Identity) + X_{l-1}^{(2)} (DPF)
#   - Concat + Fuse: Merge branches (2*ch_out -> ch_out via Conv 1x1)

import torch
import torch.nn as nn

from ..modules.conv import Conv
from .saa import DPF

__all__ = ['DownSample', 'RAAStagev2', 'RAABackbonev2']


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


class RAAStagev2(nn.Module):
    """CSP-style Stage for RAAv2 Backbone.

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


class RAABackbonev2(nn.Module):
    """RAAv2 Backbone: Stem + 4 CSP-style Stages.

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
            self.stages.append(RAAStagev2(ch_in, ch_out, n=n_val, scales=scales))
            ch_in = ch_out

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


def count_params_and_flops(model, input_size=(1, 3, 640, 640)):
    """Count parameters and FLOPs for RAAv2 backbone.

    Args:
        model: The RAAv2 backbone model.
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
    """Analytically compute parameter count for RAAv2 backbone.

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
    print('RAAv2 Backbone — Architecture Verification')
    print('=' * 70)

    model = RAABackbonev2()
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    print(f'\nInput:  {x.shape}')
    for i, o in enumerate(outputs):
        print(f'F{i + 1} (S{i + 2}): {o.shape}')

    print('\n' + '=' * 70)
    print('RAAv2 Backbone — Parameter Count (Analytical)')
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
    print('RAAv2 Backbone — Parameter Count (torch.numel)')
    print('=' * 70)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Total params: {total_params:,}  ({total_params / 1e6:.2f}M)')

    print('\n' + '=' * 70)
    print('RAAv2 Backbone — FLOPs (thop)')
    print('=' * 70)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(x,), verbose=False)
        print(f'  FLOPs:  {flops:,}  ({flops / 1e9:.2f}G)')
        print(f'  Params: {params:,}  ({params / 1e6:.2f}M)')
    except ImportError:
        print('  Install thop for FLOPs: pip install thop')

    print('\n' + '=' * 70)
    print('RAAStagev2 — Per-Stage Detail')
    print('=' * 70)
    ch_in = 32
    channels = (64, 128, 256, 512)
    n_list = (0, 1, 2, 3)
    for i, (ch_out, n_val) in enumerate(zip(channels, n_list)):
        stage = RAAStagev2(ch_in, ch_out, n=n_val)
        p = sum(p.numel() for p in stage.parameters())
        k = 11 + 2 * n_val
        print(f'  Stage {i + 1}: ch_in={ch_in}, ch_out={ch_out}, n={n_val} (K={k}), '
              f'params={p:,} ({p / 1e6:.2f}M)')
        ch_in = ch_out