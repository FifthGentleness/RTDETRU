# X013 A-ZOP: 注意力静默起步 (AIFI, [C]).
# 机制: out_proj零初始化 => 注意力支路第0步输出恰为0(AIFI退化为纯FFN层),
# 注意力权重随训练从零生长("注意力闸门"); 与零门控的区别: 门控是标量/通道
# 乘子, 本方案直接改投影初始化(优化景观不同)。+0参数。

import torch
import torch.nn as nn

from .aifi_xbase import AIFIXBase

__all__ = ['AIFI_XZOP']


class SilentMHA(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.inner = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        nn.init.zeros_(self.inner.out_proj.weight)
        nn.init.zeros_(self.inner.out_proj.bias)

    def forward(self, qk, tokens):
        o, _ = self.inner(qk.transpose(0, 1), qk.transpose(0, 1),
                          tokens.transpose(0, 1), need_weights=False)
        return o.transpose(0, 1)


class AIFI_XZOP(AIFIXBase):
    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1, self.normalize_before = c1, normalize_before
        self.attn = SilentMHA(c1, num_heads, dropout)
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
