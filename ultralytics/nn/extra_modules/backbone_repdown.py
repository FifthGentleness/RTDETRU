"""YOLO-ULM's RepDown (CVPR 2026), adapted to the R18 P2-to-P3 transition."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_weights

from ..modules.block import BasicBlock, ConvNormLayer, get_activation

__all__ = ("BasicBlock_RepDown",)


class RepDown(nn.Module):
    """Project channels before parallel 7x7 context and 3x3 detail downsampling."""

    def __init__(self, in_channels, out_channels, act):
        super().__init__()
        self.project = ConvNormLayer(in_channels, out_channels, 1, 1)
        self.context = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 7, 2, 3, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.detail = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 2, 1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = get_activation(act)

    def forward(self, x):
        x = self.project(x)
        if hasattr(self, "reparam"):
            return self.act(self.reparam(x))
        return self.act(self.context(x) + self.detail(x))

    @torch.no_grad()
    def switch_to_deploy(self):
        """Fold both branches using their BatchNorm running statistics."""
        if hasattr(self, "reparam"):
            return

        context, context_bn = self.context
        detail, detail_bn = self.detail
        context_weight, context_bias = fuse_conv_bn_weights(
            context.weight, context.bias, context_bn.running_mean, context_bn.running_var,
            context_bn.eps, context_bn.weight, context_bn.bias,
        )
        detail_weight, detail_bias = fuse_conv_bn_weights(
            detail.weight, detail.bias, detail_bn.running_mean, detail_bn.running_var,
            detail_bn.eps, detail_bn.weight, detail_bn.bias,
        )

        channels = context.out_channels
        self.reparam = nn.Conv2d(
            channels, channels, 7, 2, 3, groups=channels,
            device=context.weight.device, dtype=context.weight.dtype,
        ).requires_grad_(False)
        self.reparam.weight.copy_(context_weight + F.pad(detail_weight, (2, 2, 2, 2)))
        self.reparam.bias.copy_(context_bias + detail_bias)
        del self.context, self.detail


class BasicBlock_RepDown(BasicBlock):
    """Use RepDown for the first convolution of a stride-2 residual block."""

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        super().__init__(ch_in, ch_out, stride, shortcut, act=act, variant=variant)
        if stride == 2:
            self.branch2a = RepDown(ch_in, ch_out, act)
