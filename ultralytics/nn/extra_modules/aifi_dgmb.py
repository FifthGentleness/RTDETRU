# AIFI_DGMB: Delta-Gated Memory Bridge for Block-Sparse AIFI.
#
# 方案一（主位置2 Token集合 + 配套位置4 零初始化门控）
# 灵感来源: Gated Delta Networks (Gated DeltaNet), ICLR 2025.
#   - 位置2: M个可学习记忆寄存器, 块级读(cross-attn)/写(delta rule)通道构造
#   - 位置4: 读侧/写回侧零初始化门控
#   - 位置1: BlockSparse 主干原样叠加, 注意力结构不改动

import math

import torch
import torch.nn as nn

from .aifi_block_sparse import AIFI_BlockSparse, BlockSparseAttention

__all__ = ['AIFI_DGMB']


class AIFI_DGMB(nn.Module):
    """AIFI with block-sparse attention + delta-gated memory bridge.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads of the block-sparse trunk.
        block_size: Number of tokens per sparse block side.
        topk: Number of key blocks attended by each query block.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
        mem_slots: Number of learnable memory registers M (position 2).
    """

    def __init__(self, c1, cm=1024, num_heads=8, block_size=4, topk=4,
                 dropout=0.0, act=nn.GELU(), normalize_before=False,
                 mem_slots=16):
        super().__init__()
        self.c1 = c1
        self.block_size = int(block_size)
        self.normalize_before = normalize_before

        self.attn = BlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size,
            topk=topk, dropout=dropout)

        # ---- 位置2: 记忆寄存器 + 读/写通道 ----
        self.memory = nn.Parameter(torch.randn(mem_slots, c1) * 0.02)
        self.s0 = nn.Parameter(torch.zeros(c1, c1))
        self.wq_r = nn.Linear(c1, c1)   # 读查询
        self.wo_r = nn.Linear(c1, c1)
        self.wk_w = nn.Linear(c1, c1)   # 写 key
        self.wv_w = nn.Linear(c1, c1)   # 写 value
        self.wk_q = nn.Linear(c1, c1)   # 终态读出查询
        self.wo_o = nn.Linear(c1, c1)
        self.w_beta = nn.Linear(c1 * 2, 1)

        # ---- 位置4: 零初始化门控 ----
        self.g_r = nn.Parameter(torch.zeros(c1))
        self.g_o = nn.Parameter(torch.zeros(c1))

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
    # block 网格工具
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
        _, _, block_id, counts = self._block_meta(h, w, x.device)
        s = x.new_zeros(b, block_id.max().item() + 1, c)
        s.index_add_(1, block_id, x)
        return s / counts.view(1, -1, 1)

    def _block_expand(self, ctx, h, w):
        _, _, block_id, _ = self._block_meta(h, w, ctx.device)
        return ctx[:, block_id, :]

    # ------------------------------------------------------------------
    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_DGMB expected {self.c1} channels, got {c}')

        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        s = self._block_pool(src, h, w)                       # [B, nb, C]

        # --- 记忆读取（位置2）+ 零初始化注入（位置4）---
        logits = self.wq_r(s) @ self.memory.t() * (self.c1 ** -0.5)
        r = logits.softmax(-1) @ self.memory                  # [B, nb, C]
        src = src + self.g_r * self._block_expand(self.wo_r(r), h, w)

        # --- BlockSparse 主干（位置1, 不改动）---
        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src)))
        else:
            src = src + self.dropout1(self.attn(src))

        # --- Gated Delta Rule 记忆写入（按块索引伪时序, nb 步, 开销可忽略）---
        u = self._block_pool(src, h, w)
        k_all, v_all = self.wk_w(u), self.wv_w(u)
        # DeltaNet 稳定性: key 做 L2 归一化, 保证 beta*|k|^2 有界
        k_all = k_all / (k_all.norm(dim=-1, keepdim=True) + 1e-6)
        beta = 2.0 * torch.sigmoid(self.w_beta(torch.cat([s, u], dim=-1)))
        S = self.s0.unsqueeze(0).expand(b, -1, -1)
        for i in range(u.shape[1]):
            k, v, beta_i = k_all[:, i], v_all[:, i], beta[:, i]
            kS = torch.einsum('bi,bij->bj', k, S)
            S = S - beta_i[:, None] * torch.einsum('bi,bj->bij', k, kS) \
                  + beta_i[:, None] * torch.einsum('bi,bj->bij', k, v)

        # --- 终态记忆读出 + 零初始化门控融合（位置4）---
        k2 = self.wk_q(u)
        r2 = torch.einsum('bni,bij->bnj', k2, S)
        src = src + self.g_o * self._block_expand(self.wo_o(r2), h, w)

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
