# X031 A-MoE2: 逐token软路由双专家FFN (AIFI, [B]锚点: Synergizing Denoising
# Tasks with Sparse Mixture-of-Experts, ECCV 2024 —— 检测语境含去噪框架,
# 借用其"块内MoE-FFN"概念, 标注待核实+DETR邻接风险).
# 机制: 原FFN(E1)保留 + 窄专家E2(256宽); 逐token sigmoid路由 r(x),
# out = (1-g*r)*E1(x) + g*r*E2(x), g零初始化 => step-0等价基线。
# 区别于SAG-FFN(单路径标量增益)与DHCFFN(无条件低秩路径): 本方案是
# 逐token条件计算的双专家混合。参数+E2(+131K)+router。

import torch
import torch.nn as nn

from .aifi_xbase import AIFIXBase
from .aifi_xipc import _MHA2Arg

__all__ = ['AIFI_XMoE2']


class AIFI_XMoE2(AIFIXBase):
    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False, expert2_dim=256):
        super().__init__()
        self.c1, self.normalize_before = c1, normalize_before
        self.attn = _MHA2Arg(c1, num_heads, dropout)
        self.fc1 = nn.Linear(c1, cm)                 # 专家E1(原FFN)
        self.fc2 = nn.Linear(cm, c1)
        self.e2_1 = nn.Linear(c1, expert2_dim)       # 窄专家E2
        self.e2_2 = nn.Linear(expert2_dim, c1)
        self.router = nn.Linear(c1, 1)
        nn.init.zeros_(self.router.weight); nn.init.zeros_(self.router.bias)
        self.g = nn.Parameter(torch.zeros(1))        # 单守门
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    def _ffn(self, src):
        e1 = self.fc2(self.dropout(self.act(self.fc1(src))))
        r = torch.sigmoid(self.router(src)) * self.g
        e2 = self.e2_2(self.act(self.e2_1(src)))
        return (1.0 - r) * e1 + r * e2

    def forward(self, x):
        return self._forward_core(x)

    def extra_repr(self):
        return f'|g|={self.g.item():.4f} router|w|={self.router.weight.abs().mean().item():.4f}'
