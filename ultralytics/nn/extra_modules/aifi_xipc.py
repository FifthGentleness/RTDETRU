# X011 A-IPC: AIFI输入预条件化 (AIFI, [C]; 概念谱系: Early Convolutions Help
# Transformers See Better, Xiao et al. 2022 - 原文2022违反时效窗, 降级[C]).
# 机制: S5特征图在进注意力前经零门控3x3深度卷积补一层局部结构(token
# 获得邻域上下文后再交互); 区别于ASConvFFN(FFN隐层内)。+231参数。

import torch
import torch.nn as nn

from .aifi_xbase import AIFIXBase

__all__ = ['AIFI_XIPC']


class _MHA2Arg(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.inner = nn.MultiheadAttention(dim, num_heads, dropout=dropout)

    def forward(self, qk, tokens):
        o, _ = self.inner(qk.transpose(0, 1), qk.transpose(0, 1),
                          tokens.transpose(0, 1), need_weights=False)
        return o.transpose(0, 1)


class AIFI_XIPC(AIFIXBase):
    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1, self.normalize_before = c1, normalize_before
        self.pre = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)
        self.pre_gate = nn.Parameter(torch.zeros(1))     # 单守门
        self.attn = _MHA2Arg(c1, num_heads, dropout)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    def forward(self, x):
        return self._forward_core(x + self.pre_gate * self.pre(x))

    def extra_repr(self):
        return f'|pre_gate|={self.pre_gate.item():.4f}'
