# AIFI_CoPEBS: Content-Conditioned Position Encoding for Block-Sparse AIFI.
#
# 方案一（主位置2 Token集合-PE构造 + 配套位置4 零初始化门控）
# 灵感来源: Contextual Position Encoding: Learning to Count What's Important
#           (CoPE), ICLR 2025, Meta FAIR。
#   - 位置2: 固定2D sincos之外, 追加内容门控累积的"语义位置": K个可学习语义锚,
#     每token的门控响应沿栅格序累积归一化 => p_i ∈ R^K (数语义对象而非数格子)
#   - 位置4: W_p 零初始化 + 逐通道门 gamma => step-0 与 AIFI_BlockSparse 逐位相等
#   - 位置1: BlockSparse 主干原样叠加, 路由结构零改动 —— 只改变路由相似度的
#     坐标系, 不加任何通路(区别于已证伪的寄存器/超图/记忆全局桥家族)
#
# 设计依据(仓库教训): 同角色弱目标(同类车流)因空间遥远难以互相路由是"孤立块
# 割裂"的坐标系根源; 语义位置让 top-4 相似块度量摆脱几何纠缠。参数增量 ~8.7K。
# 部署注: cumsum(N=400, K=16) 可等价替换为常量下三角矩阵matmul(TRT原生)。

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifi_block_sparse import AIFI_BlockSparse, BlockSparseAttention

__all__ = ['AIFI_CoPEBS']


class CoPE2D(nn.Module):
    """内容条件化语义位置: p = norm(cumsum(gate(x))), 追加项零初始化注入。

    Args:
        dim: Token channel dim.
        K: 语义锚数量.
    """

    def __init__(self, dim, K=16):
        super().__init__()
        self.anchor = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.w_p = nn.Linear(K, dim)
        # w_p 常规初始化(必须非零): 若与 gamma=0 同时零初始化会互锁梯度,
        # 分支永远打不开(见 aifi_dhc_ffn.py 同型bug注释); step-0 等价性由
        # gamma=0 单独保证(0*x=0 精确成立)
        nn.init.zeros_(self.w_p.bias)
        self.gamma = nn.Parameter(torch.zeros(dim))       # 配套位置4: 零门控

    def forward(self, t):                                  # t: [B, N, dim]
        g = torch.sigmoid(t @ self.anchor.t())             # [B, N, K] 内容门控
        p = g.cumsum(dim=1) / g.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return t + self.gamma * self.w_p(p)


class AIFI_CoPEBS(nn.Module):
    """AIFI with block-sparse attention + content-conditioned position encoding.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads of the block-sparse trunk.
        block_size: Number of tokens per sparse block.
        topk: Number of key blocks attended by each query block.
        K: Number of semantic anchors of CoPE.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=1024, num_heads=8, block_size=4, topk=4,
                 K=16, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = BlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size, topk=topk,
            dropout=dropout)
        self.cope = CoPE2D(c1, K=K)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_CoPEBS expected {self.c1} channels, got {c}')
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        # --- 内容条件化语义位置（位置2 PE构造, 位置4 零门控）---
        src = self.cope(src)

        # --- BlockSparse 主干（位置1, 不改动）---
        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src)))
        else:
            src = src + self.dropout1(self.attn(src))

        # --- FFN（与 AIFI_BlockSparse 一致）---
        if self.normalize_before:
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        g = self.cope.gamma.detach().abs().mean().item()
        return (f'c1={self.c1}, cm={self.fc1.out_features}, '
                f'K={self.cope.anchor.shape[0]}, |gamma|_mean={g:.4f}')
