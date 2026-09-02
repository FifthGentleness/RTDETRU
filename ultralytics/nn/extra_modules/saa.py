# SAA Backbone: Scale-Aware Attentional Backbone from SME-DETR.
#
# Reference: SME-DETR (IEEE TGRS 2025)
#
# Architecture:
#   - DSA (Dynamic Scale-Aware): 5-branch multi-scale DWConv (3,5,7,9,11) with
#     learnable scalar weights normalized by softmax, plus pointwise conv.
#     Equation (1): xout = sum(wi * xi), then PointwiseConv.
#   - DCA (Directional Channel Attention): AGAP -> 1D horizontal/vertical
#     convs -> DWConv + PointwiseConv -> Sigmoid channel attention.
#     Equation (2): AGAP(X)_c = (1/HW) sum X_c(i,j)
#     Equation (3): h_attn = Conv_h(AGAP(X)), v_attn = Conv_v(AGAP(X))
#     Equation (4): A = Sigmoid(PWConv(DWConv(h_attn + v_attn)))
#   - DPF (Dual-Path Fusion): output = Conv1x1(A * DSA(x) + DSA(x))
#     = Conv1x1((1+A) * fdsa)
#     A is channel attention [B,C,1,1], DSA output is [B,C,H,W].
#     A broadcasts spatially to modulate all spatial positions per-channel.
#     Final 1x1 conv for channel mixing after DPF fusion.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import BasicBlock
from ..modules.conv import Conv

__all__ = ['SAA', 'SAABasicBlock', 'BlocksSAA']


class DSA(nn.Module):
    """Dynamic Scale-Aware Module (SME-DETR Equation 1).

    Multi-scale depthwise separable convolutions with learnable scalar
    weights normalized by softmax, followed by pointwise channel mixing.

    Args:
        channels: Number of input/output channels.
        scales: List of kernel sizes for multi-scale DWConv branches.
    """

    def __init__(self, channels, scales=(3, 5, 7, 9, 11)):
        super().__init__()
        self.channels = channels
        self.scales = scales
        n = len(scales)

        self.dw_convs = nn.ModuleList()
        for k in scales:
            self.dw_convs.append(
                nn.Conv2d(channels, channels, kernel_size=k,
                          padding=k // 2, groups=channels, bias=False)
            )

        self.weights = nn.Parameter(torch.zeros(n))

        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        feats = [conv(x) for conv in self.dw_convs]

        w = F.softmax(self.weights, dim=0)

        xout = sum(w[i] * feats[i] for i in range(len(self.scales)))

        xout = self.pw_conv(xout)
        return xout


class DCA(nn.Module):
    """Directional Channel Attention Module (SME-DETR Equations 2-4).

    Generates channel attention map A ∈ [0,1] with shape [B,C,1,1] via:
      1. AGAP: Global average pooling to extract global context
      2. Directional 1D convs: horizontal (1xK) and vertical (Kx1)
      3. Fusion: DWConv + PointwiseConv + Sigmoid

    Args:
        channels: Number of input/output channels.
        n: Kernel size parameter. K = 11 + 2*n for directional convs.
    """

    def __init__(self, channels, n=1):
        super().__init__()
        self.channels = channels
        k = 11 + 2 * n

        self.agap = nn.AdaptiveAvgPool2d(1)

        self.conv_h = nn.Conv2d(
            channels, channels, kernel_size=(1, k),
            padding=(0, k // 2), groups=channels, bias=False
        )
        self.conv_v = nn.Conv2d(
            channels, channels, kernel_size=(k, 1),
            padding=(k // 2, 0), groups=channels, bias=False
        )

        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=1, groups=channels, bias=False
        )
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        gap = self.agap(x)

        h_attn = self.conv_h(gap)
        v_attn = self.conv_v(gap)

        fused = h_attn + v_attn
        fused = self.dw_conv(fused)
        A = torch.sigmoid(self.pw_conv(fused))

        return A


class DPF(nn.Module):
    """Dual-Path Fusion Block (SME-DETR).

    Combines DSA and DCA:
        fdsa = DSA(x)
        A = DCA(x)    # channel attention [B,C,1,1], Sigmoid [0,1]
        output = Conv1x1(A * fdsa + fdsa) = Conv1x1((1 + A) * fdsa)

    The residual connection ( + fdsa) ensures DSA information is preserved.
    A ≈ 0 → output ≈ Conv1x1(fdsa) (DCA deactivates, keep original DSA)
    A ≈ 1 → output ≈ Conv1x1(2 * fdsa) (DCA enhances, amplify DSA features)
    Final 1x1 conv for channel mixing after fusion.

    Args:
        channels: Number of input/output channels.
        n: DCA kernel size parameter.
        scales: DSA multi-scale kernel sizes.
    """

    def __init__(self, channels, n=1, scales=(3, 5, 7, 9, 11)):
        super().__init__()
        self.dsa = DSA(channels, scales=scales)
        self.dca = DCA(channels, n=n)
        self.fusion_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        fdsa = self.dsa(x)
        A = self.dca(x)
        return self.fusion_conv(A * fdsa + fdsa)


class SAA(nn.Module):
    """Scale-Aware Attentional module replacing branch2a.

    Implements the SAA Backbone's DPF block from SME-DETR.
    Half-channel split design (consistent with DSAWACGAv2/DSADOC pattern):
    - First half  -> DPF (DSA + DCA)
    - Second half -> Conv3x3 for local feature extraction

    Args:
        dim: Number of input/output channels (= ch_out of the block).
        n: DCA kernel size parameter.
        scales: DSA multi-scale kernel sizes.
    """

    def __init__(self, dim, n=1, scales=(3, 5, 7, 9, 11)):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.half_dim = half_dim

        self.dpf = DPF(half_dim, n=n, scales=scales)

        self.conv_path = Conv(half_dim, half_dim, 3, 1, act=nn.ReLU())

    def forward(self, x):
        x_saa, x_conv = x.chunk(2, dim=1)

        saa_out = self.dpf(x_saa)

        conv_out = self.conv_path(x_conv)

        out = torch.cat([saa_out, conv_out], dim=1)

        return out


class SAABasicBlock(BasicBlock):
    """BasicBlock with SAA (DPF) replacing branch2a.

    Args:
        n: DCA kernel size parameter (K = 11 + 2*n).
            Deeper stages should use larger n for longer channel dependency.
    """
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d', n=1):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        del self.branch2a
        self.saa = SAA(ch_out, n=n)

    def forward(self, x):
        out = self.saa(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.act(out)
        return out


class BlocksSAA(nn.Module):
    """Stage container: vanilla BasicBlocks + last-block SAA (DPF).

    Args:
        n: DCA kernel size parameter for the SAA block (K = 11 + 2*n).
            Per-stage design: deeper stages use larger n.
            Stage 2 → n=0 (K=11), Stage 3 → n=1 (K=13),
            Stage 4 → n=2 (K=15), Stage 5 → n=3 (K=17).
    """

    def __init__(self, ch_in, ch_out, block, count, stage_num, act='relu', variant='d', n=1):
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
                    SAABasicBlock(
                        ch_in,
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1,
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        n=n,
                    )
                )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out