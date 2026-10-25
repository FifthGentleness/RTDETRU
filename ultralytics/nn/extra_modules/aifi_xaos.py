# X006 A-AOS: 注意力残差逐通道LayerScale (AIFI, [C]位置6收尾件).
# 机制: 注意力输出乘逐通道可学习门(ones初始化=恒等起步);"少混合"的
# 可微调旋钮——若学到<1即呼应"少混合者赢"规律。参数+512。

import torch
import torch.nn as nn

from .aifi_xbase import AIFIXBase

__all__ = ['AIFI_XAOS']


class ScaledMHA(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.inner = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.out_scale = nn.Parameter(torch.ones(dim))    # ones=恒等

    def forward(self, qk, tokens):
        o, _ = self.inner(qk.transpose(0, 1), qk.transpose(0, 1),
                          tokens.transpose(0, 1), need_weights=False)
        return o.transpose(0, 1) * self.out_scale


class AIFI_XAOS(AIFIXBase):
    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1, self.normalize_before = c1, normalize_before
        self.attn = ScaledMHA(c1, num_heads, dropout)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    def forward(self, x):
        return self._forward_core(x)

    def extra_repr(self):
        s = self.attn.out_scale.detach()
        return f'out_scale=[{s.min():.3f},{s.max():.3f}]'
