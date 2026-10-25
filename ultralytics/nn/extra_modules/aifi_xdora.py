# X032 A-DoRA: FFN输出权重幅向解耦 (AIFI, [A]DoRA ICML 2024 Oral, PMLR v235).
# 机制: fc2权重 W 按列分解为幅值m与方向W/||W||_col, 幅值独立可学习
# (m初始化为当前列范数 => step-0等价基线); 训练中网络获得权重几何的
# 额外自由度(适配器方法向全量训练的原则迁移)。参数+256(输出列数)。

import torch
import torch.nn as nn

from .aifi_xbase import AIFIXBase
from .aifi_xipc import _MHA2Arg

__all__ = ['AIFI_XDoRA']


class DoRALinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        with torch.no_grad():                       # m = 逐输出神经元行范数(恒等起步)
            self.mag = nn.Parameter(
                self.linear.weight.norm(dim=1).clone())          # [out]

    def forward(self, x):
        d = self.linear.weight / self.linear.weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return torch.nn.functional.linear(x, d * self.mag.view(-1, 1),
                                          self.linear.bias)


class AIFI_XDoRA(AIFIXBase):
    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1, self.normalize_before = c1, normalize_before
        self.attn = _MHA2Arg(c1, num_heads, dropout)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = DoRALinear(nn.Linear(cm, c1))    # 仅输出层幅向解耦
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    def forward(self, x):
        return self._forward_core(x)
