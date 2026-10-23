# AIFI_ASB: Anchor State-space Bridge for Block-Sparse AIFI.
#
# 方案二（主位置2 Token集合 + 配套位置4 零初始化门控）
# 灵感来源: MambaVision: A Hybrid Mamba-Transformer Vision Backbone, CVPR 2025.
#   - 位置2: 确定性块级锚点token构造（块摘要池化 + 可学习锚点位置编码）
#   - 位置2: 锚点序列上的双向选择性SSM扫描（锚点token自身机制, 不触碰patch-token注意力）
#   - 位置4: 锚点上下文注回的零初始化门控
#   - 位置1: BlockSparse 主干原样叠加, 注意力结构不改动

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifi_block_sparse import AIFI_BlockSparse, BlockSparseAttention

__all__ = ['AIFI_ASB']


class BidirSelectiveScan(nn.Module):
    """轻量双向选择性状态空间扫描（序列长度=块数, 默认25, 开销可忽略）。"""

    def __init__(self, dim):
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(dim))
        self.dt_proj = nn.Linear(dim, dim)
        self.bc_proj = nn.Linear(dim, dim * 2)
        self.d_proj = nn.Parameter(torch.ones(dim))

    def _scan(self, x):
        b, l, d = x.shape
        dt = F.softplus(self.dt_proj(x))               # [B, L, D]
        bmat, cmat = self.bc_proj(x).chunk(2, dim=-1)  # [B, L, D] x2
        A = -torch.exp(self.A_log)                     # [D], 负值保证稳定衰减
        y = x.new_zeros(b, l, d)
        h = x.new_zeros(b, d)
        for t in range(l):
            decay = torch.exp(A[None, :] * dt[:, t])   # [B, D]
            h = decay * h + dt[:, t] * bmat[:, t] * x[:, t]
            y[:, t] = cmat[:, t] * h + self.d_proj * x[:, t]
        return y

    def forward(self, x):
        return self._scan(x) + self._scan(torch.flip(x, dims=[1])).flip(1)


class AIFI_ASB(nn.Module):
    """AIFI with block-sparse attention + anchor state-space bridge.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads of the block-sparse trunk.
        block_size: Number of tokens per sparse block side.
        topk: Number of key blocks attended by each query block.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
        ssm_dim: Low dim of the anchor scan path (position 2).
        anchor_grid: Expected anchor grid side (assumes 640 input -> 5x5).
    """

    def __init__(self, c1, cm=1024, num_heads=8, block_size=4, topk=4,
                 dropout=0.0, act=nn.GELU(), normalize_before=False,
                 ssm_dim=64, anchor_grid=5):
        super().__init__()
        self.c1 = c1
        self.block_size = int(block_size)
        self.normalize_before = normalize_before

        self.attn = BlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size,
            topk=topk, dropout=dropout)

        # ---- 位置2: 锚点token构造（可学习锚点PE, 网格变化时双三次插值）----
        g = int(anchor_grid)
        self.anchor_pe = nn.Parameter(torch.randn(1, c1, g, g) * 0.02)
        self.anchor_down = nn.Linear(c1, ssm_dim)
        self.anchor_ssm = BidirSelectiveScan(ssm_dim)
        self.anchor_up = nn.Linear(ssm_dim, c1)

        # ---- 位置4: 零初始化门控 ----
        self.g = nn.Parameter(torch.zeros(c1))

        # ---- FFN / norm / dropout（与 AIFI_BlockSparse 一致）----
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

        self._meta_cache = {}

    # ------------------------------------------------------------------
    def _block_meta(self, h, w, device):
        key = (h, w, str(device))
        if key not in self._meta_cache:
            bs = self.block_size
            gh, gw = math.ceil(h / bs), math.ceil(w / bs)
            row = torch.arange(h, device=device) // bs
            col = torch.arange(w, device=device) // bs
            block_id = (row[:, None] * gw + col[None, :]).reshape(-1)
            rv = (torch.arange(gh * bs, device=device) < h).float() \
                .view(gh, bs).sum(1)
            cv = (torch.arange(gw * bs, device=device) < w).float() \
                .view(gw, bs).sum(1)
            counts = (rv[:, None] * cv[None, :]).reshape(-1).clamp(min=1.0)
            self._meta_cache[key] = (gh, gw, block_id, counts)
        return self._meta_cache[key]

    def _block_pool(self, x, h, w):
        b, _, c = x.shape
        gh, gw, block_id, counts = self._block_meta(h, w, x.device)
        s = x.new_zeros(b, gh * gw, c)
        s.index_add_(1, block_id, x)
        return s / counts.view(1, -1, 1), gh, gw

    def _block_expand(self, ctx, h, w):
        _, _, block_id, _ = self._block_meta(h, w, ctx.device)
        return ctx[:, block_id, :]

    # ------------------------------------------------------------------
    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_ASB expected {self.c1} channels, got {c}')

        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        # --- 锚点token构造 + 双向SSM扫描（位置2）---
        anchors, gh, gw = self._block_pool(src, h, w)  # [B, nb, C]
        pe = self.anchor_pe
        if pe.shape[-2:] != (gh, gw):
            pe = F.interpolate(pe, size=(gh, gw), mode='bicubic',
                               align_corners=False)
        anchors = anchors + pe.flatten(2).transpose(1, 2).expand(b, -1, -1)
        e = self.anchor_down(anchors)                  # [B, nb, d]
        ctx = self.anchor_up(self.anchor_ssm(e))       # [B, nb, C]

        # --- 零初始化门控注入（位置4）---
        src = src + self.g * self._block_expand(ctx, h, w)

        # --- BlockSparse 主干（位置1, 不改动）---
        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src)))
        else:
            src = src + self.dropout1(self.attn(src))

        # --- FFN / norm 拓扑（与 AIFI_BlockSparse 一致）---
        if self.normalize_before:
            src = src + self.dropout2(self.ffn(self.norm2(src)))
        else:
            src = self.norm1(src)
            src = src + self.dropout2(self.ffn(src))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def ffn(self, src):
        return self.fc2(self.dropout(self.act(self.fc1(src))))
