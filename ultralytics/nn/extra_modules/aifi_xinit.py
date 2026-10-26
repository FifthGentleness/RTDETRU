# X057/X058 AIFI初始化族 (批次6, [C]).
# X057 A-PScale: 可学习PE缩放(标量p, init=1 => step-0等价基线)
# X058 A-QKV05: qkv投影权重半幅初始化(注意力logit幅度起点减半, 软启动)

import torch
import torch.nn as nn

from .aifi_xbase import AIFIXBase
from .aifi_xipc import _MHA2Arg
from .aifi_block_sparse import AIFI_BlockSparse

__all__ = ('AIFI_XPScale', 'AIFI_XQKV05')


class AIFI_XPScale(AIFIXBase):
    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1, self.normalize_before = c1, normalize_before
        self.attn = _MHA2Arg(c1, num_heads, dropout)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act
        self.pos_scale = nn.Parameter(torch.ones(1))     # init=1恒等

    def forward(self, x):
        b, c, h, w = x.shape
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype) * self.pos_scale
        tokens = x.flatten(2).permute(0, 2, 1)
        qk = tokens + pos
        src = tokens + self.dropout1(self.attn(qk, tokens))
        src = self.norm1(src)
        src = src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src)))))
        return self.norm2(src).permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        return f'pos_scale={self.pos_scale.item():.4f}'


class AIFI_XQKV05(AIFIXBase):
    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False, qkv_gain=0.5):
        super().__init__()
        self.c1, self.normalize_before = c1, normalize_before
        self.attn = _MHA2Arg(c1, num_heads, dropout)
        with torch.no_grad():                            # in_proj半幅初始化
            self.attn.inner.in_proj_weight.mul_(qkv_gain)
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
