# AIFI_CBKFFN: Cross-Block Boundary-KAN FFN for Block-Sparse AIFI.
#
# 方案三（主位置5 FFN + 配套位置4 零初始化门控）
# 灵感来源: KAN or MLP: A Fairer Comparison, ICLR 2025.
#   - 位置5: 跨块边界桥接组(BBG) + 分组切比雪夫KAN分支（混合架构, 原MLP保留）
#   - 位置4: KAN分支输出的零初始化门控
#   - 位置1: BlockSparse 主干原样叠加, 注意力结构不改动

import torch
import torch.nn as nn

from .aifi_block_sparse import AIFI_BlockSparse, BlockSparseAttention

__all__ = ['AIFI_CBKFFN']


class CrossBlockBoundaryKANFFN(nn.Module):
    """混合FFN: 原MLP分支 + (边界桥接 -> 低维切比雪夫KAN) 新分支.

    Args:
        dim: Token channel dim.
        cm: Hidden dim of the original MLP branch (unchanged).
        block_size: Sparse block side used to locate border tokens.
        cheb_dim: Low dim of the Chebyshev-KAN branch.
        cheb_groups: Number of channel groups inside the KAN branch.
        cheb_order: Chebyshev polynomial order (fixed to 3).
        dropout: Dropout of the original MLP branch.
    """

    def __init__(self, dim, cm=1024, block_size=4, cheb_dim=64,
                 cheb_groups=4, cheb_order=3, dropout=0.0):
        super().__init__()
        if cheb_order != 3:
            raise ValueError('CBK-FFN currently supports cheb_order=3')
        if cheb_dim % cheb_groups != 0:
            raise ValueError('cheb_dim must be divisible by cheb_groups')

        self.dim = dim
        self.block_size = int(block_size)
        self.K = int(cheb_order)
        self.G = int(cheb_groups)
        self.dpg = cheb_dim // self.G

        # 原MLP分支（不动）
        self.fc1 = nn.Linear(dim, cm)
        self.fc2 = nn.Linear(cm, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        # 位置5: 边界桥接组（固定索引 gather/index_add, 无动态控制流）
        self.bridge = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 2 * dim))

        # 位置5: 分组切比雪夫KAN（可学习基系数, 小方差初始化）
        self.down = nn.Linear(dim, cheb_dim)
        self.up = nn.Linear(cheb_dim, dim)
        self.cheb = nn.Parameter(
            torch.randn(self.G, self.dpg, self.K + 1) * 0.01)

        # 位置4: 零初始化门控
        self.g_f = nn.Parameter(torch.zeros(dim))

        self._border_cache = {}

    def _border_pairs(self, h, w, device):
        """返回(水平+垂直)相邻块面对面边界token索引对; 不可整除时返回None。"""
        key = (h, w, str(device))
        if key not in self._border_cache:
            bs = self.block_size
            if h % bs or w % bs:
                self._border_cache[key] = None
            else:
                gh, gw = h // bs, w // bs
                left, right = [], []
                for i in range(gh):          # 水平邻接对
                    for j in range(gw - 1):
                        rows = torch.arange(i * bs, (i + 1) * bs,
                                            device=device)
                        left.append(rows * w + (j + 1) * bs - 1)
                        right.append(rows * w + (j + 1) * bs)
                for i in range(gh - 1):      # 垂直邻接对
                    for j in range(gw):
                        cols = torch.arange(j * bs, (j + 1) * bs,
                                            device=device)
                        left.append(((i + 1) * bs - 1) * w + cols)
                        right.append(((i + 1) * bs) * w + cols)
                if left:
                    self._border_cache[key] = (torch.cat(left), torch.cat(right))
                else:
                    self._border_cache[key] = None
        return self._border_cache[key]

    def _bridge(self, x, h, w):
        pairs = self._border_pairs(h, w, x.device)
        if pairs is None:
            return x
        left, right = pairs
        z = self.bridge(torch.cat([x[:, left, :], x[:, right, :]], dim=-1))
        da, db = z.chunk(2, dim=-1)
        add = x.new_zeros(x.shape)
        add.index_add_(1, left, da)
        add.index_add_(1, right, db)
        return x + add

    def _cheb(self, u):
        b, n, d = u.shape
        u = u.view(b, n, self.G, self.dpg)
        ts = [torch.ones_like(u), u,
              2 * u * u - 1,
              4 * u * u * u - 3 * u]          # T0..T3 (K=3)
        out = 0
        for k in range(self.K + 1):
            out = out + self.cheb[..., k] * ts[k]
        return out.reshape(b, n, d)

    def forward(self, x, h, w):
        ffn_out = self.fc2(self.dropout(self.act(self.fc1(x))))
        cheb_out = self.up(self._cheb(self.down(self._bridge(x, h, w))))
        return ffn_out + self.g_f * cheb_out


class AIFI_CBKFFN(nn.Module):
    """AIFI with block-sparse attention + cross-block boundary-KAN FFN.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads of the block-sparse trunk.
        block_size: Number of tokens per sparse block side.
        topk: Number of key blocks attended by each query block.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
        cheb_dim: Low dim of the Chebyshev-KAN branch.
        cheb_groups: Number of channel groups inside the KAN branch.
    """

    def __init__(self, c1, cm=1024, num_heads=8, block_size=4, topk=4,
                 dropout=0.0, act=nn.GELU(), normalize_before=False,
                 cheb_dim=64, cheb_groups=4):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before

        self.attn = BlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size,
            topk=topk, dropout=dropout)

        self.ffn = CrossBlockBoundaryKANFFN(
            dim=c1, cm=cm, block_size=block_size, cheb_dim=cheb_dim,
            cheb_groups=cheb_groups, dropout=dropout)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_CBKFFN expected {self.c1} channels, got {c}')

        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        # --- BlockSparse 主干（位置1, 不改动）---
        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src)))
        else:
            src = src + self.dropout1(self.attn(src))

        # --- 混合FFN（位置5 BBG+KAN, 位置4 零初始化门控）---
        if self.normalize_before:
            src = src + self.dropout2(self.ffn(self.norm2(src), h, w))
        else:
            src = self.norm1(src)
            src = src + self.dropout2(self.ffn(src, h, w))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()
