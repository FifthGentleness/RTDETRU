# X012 A-LPE: 低秩可学习残差位置编码 (AIFI, [C]).
# 机制: sincos固定PE之外追加因子化可学习残差 PE = A@B (400×r @ r×C),
# gamma零门控 => step-0等价基线; 非20x20尺寸时按token数线性插值。+~11K。
# 区别于CoPE-BS(语义门控累积位置, 挂BlockSparse): 本方案是直接PE残差。

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifi_xbase import AIFIXBase

__all__ = ['AIFI_XLPE']


class AIFI_XLPE(AIFIXBase):
    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False, max_tokens=400, rank=16):
        super().__init__()
        self.c1, self.normalize_before = c1, normalize_before
        self.pe_a = nn.Parameter(torch.randn(max_tokens, rank) * 0.02)
        self.pe_b = nn.Parameter(torch.randn(rank, c1) * 0.02)   # 常规init(活性)
        self.pe_gamma = nn.Parameter(torch.zeros(c1))            # 单守门
        self.attn = nn.MultiheadAttention(c1, num_heads, dropout=dropout)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    def _lpe(self, n):
        pe = self.pe_a @ self.pe_b                               # [T, C]
        if n != pe.shape[0]:
            pe = F.interpolate(pe.t().unsqueeze(0), size=n,
                               mode='linear', align_corners=False).squeeze(0).t()
        return pe[None]                                          # [1, n, C]

    def _forward_lpe(self, x):
        b, c, h, w = x.shape
        pos_sincos = None
        from .aifi_block_sparse import AIFI_BlockSparse
        pos_sincos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        tokens = x.flatten(2).permute(0, 2, 1)
        qk = tokens + pos_sincos + self.pe_gamma * self._lpe(tokens.shape[1]).to(tokens.dtype)
        if self.normalize_before:
            q = self.norm1(tokens) + pos_sincos
            src = tokens + self.dropout1(
                self.attn(q.transpose(0, 1), q.transpose(0, 1),
                          tokens.transpose(0, 1), need_weights=False)[0].transpose(0, 1))
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = tokens + self.dropout1(
                self.attn(qk.transpose(0, 1), qk.transpose(0, 1),
                          tokens.transpose(0, 1), need_weights=False)[0].transpose(0, 1))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)
        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def forward(self, x):
        return self._forward_lpe(x)

    def extra_repr(self):
        return f'|pe_gamma|={self.pe_gamma.abs().mean().item():.4f}'
