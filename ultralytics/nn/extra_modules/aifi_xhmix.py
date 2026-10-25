# X005 A-HMix: 头混合注意力 (AIFI, [B]锚点待核实: 多头聚合重加权方向).
# 机制: 8头输出经可学习8x8混合矩阵重组合(h头冗余/同质化时可学出互补组合),
# 恒等+零初始化非对角 => step-0等于普通多头注意力。参数+64。

import torch
import torch.nn as nn

from .aifi_xbase import AIFIXBase

__all__ = ['AIFI_XHMix']


class HeadMixAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.h = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.mix = nn.Parameter(torch.eye(num_heads))    # 恒等初始化(活性)
        self.gate = nn.Parameter(torch.zeros(1))         # 单守门: mix偏差零门控
        self.drop = nn.Dropout(dropout)

    def forward(self, qk, tokens):
        b, n, c = qk.shape
        h, d = self.h, c // self.h
        q, k, v = self.qkv(qk).chunk(3, dim=-1)
        q = q.view(b, n, h, d).transpose(1, 2)
        k = k.view(b, n, h, d).transpose(1, 2)
        v = v.view(b, n, h, d).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) * (d ** -0.5)
        out = att.softmax(-1) @ v                          # [B,h,N,d]
        m = torch.eye(self.h, device=out.device) + self.gate * self.mix
        out = torch.einsum('ij,bjnd->bind', m, out)        # 头间重组合
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.drop(self.proj(out))


class AIFI_XHMix(AIFIXBase):
    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1, self.normalize_before = c1, normalize_before
        self.attn = HeadMixAttention(c1, num_heads, dropout)
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
        return f'|gate|={self.attn.gate.item():.4f}'
