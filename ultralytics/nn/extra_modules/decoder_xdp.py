# X007 D-DP: 解码器逐层DropPath (解码器, [C]训练期正则, 推理零开销).
# 机制: 每层输出对(层输出-层输入)乘随机深度掩码, 逐层线性递增
# (ndl=3 -> p=[0, 0.05, 0.1]); 治val/cls晚期过拟合(诊断5)。eval恒等。

import torch
import torch.nn as nn

from ultralytics.nn.modules.transformer import (DeformableTransformerDecoder,
                                                DeformableTransformerDecoderLayer)
from ultralytics.nn.modules.head import RTDETRDecoder
from .decoder_innov_common import DropPath

__all__ = ('RTDETRDecoderXDP', 'RTDETRDecoderXLogN')


class XDPayer(DeformableTransformerDecoderLayer):
    def __init__(self, *a, dp=0.0, **kw):
        super().__init__(*a, **kw)
        self.dp = DropPath(dp)

    def forward(self, embed, *a, **kw):
        out = super().forward(embed, *a, **kw)
        return embed + self.dp(out - embed)


class RTDETRDecoderXDP(RTDETRDecoder):
    """RTDETRDecoder + 逐层递增DropPath(默认 0~0.1)。"""

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4,
                 nh=8, ndl=6, d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(),
                 nd=100, label_noise_ratio=0.5, box_noise_scale=1.0,
                 learnt_init_query=False, dp_max=0.1):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx,
                         dropout, act, nd, label_noise_ratio, box_noise_scale,
                         learnt_init_query)
        layers = nn.ModuleList()
        for i in range(ndl):
            p = dp_max * i / max(ndl - 1, 1)
            layers.append(XDPayer(hd, nh, d_ffn, dropout, act, self.nl, ndp, dp=p))
        self.decoder = DeformableTransformerDecoder(hd, layers[0], ndl, eval_idx)
        self.decoder.layers = layers


# --- X010 D-LogN: 逐类logit仿射归一化 (IJCV 2024, [A]已核实) ---

class LogNWrap(nn.Module):
    """score head外包裹: logits*scale+shift (逐类; ones/zeros=恒等)。"""

    def __init__(self, head):
        super().__init__()
        self.head = head
        nc = head.out_features if hasattr(head, 'out_features') else head[-1].out_features
        self.scale = nn.Parameter(torch.ones(nc))
        self.shift = nn.Parameter(torch.zeros(nc))

    def forward(self, x):
        return self.head(x) * self.scale + self.shift


class RTDETRDecoderXLogN(RTDETRDecoder):
    """X010: LogN逐类logit归一化(Logit Normalization for Long-Tailed
    Object Detection, IJCV 2024); 恒等初始化 => step-0等价基线。参数+2*nc。"""

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4,
                 nh=8, ndl=6, d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(),
                 nd=100, label_noise_ratio=0.5, box_noise_scale=1.0,
                 learnt_init_query=False):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx,
                         dropout, act, nd, label_noise_ratio, box_noise_scale,
                         learnt_init_query)
        for i, head in enumerate(self.dec_score_head):
            self.dec_score_head[i] = LogNWrap(head)
