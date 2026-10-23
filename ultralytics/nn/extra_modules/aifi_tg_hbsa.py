# TG-HBSA: Topology-Guided Hypergraph Block-Sparse AIFI (AIFI_TGHBSA).
#
# Design:
#   Branch A (existing): BlockSparseAttention (V1) performs intra-block token
#     interaction with top-k block routing (spatial locality prior).
#   Branch B (new):      HypergraphBlockBridge models higher-order inter-block
#     relations. The S5 map is partitioned into true 2D P x P spatial blocks;
#     each block is a vertex, and E learnable hyperedges dynamically connect
#     arbitrary subsets of blocks via an attention-generated incidence matrix
#     with a learnable distance-decay spatial prior. A gated residual injects
#     hypergraph messages only where needed (anti-oversmoothing).
#
# Reference:
#   Hyper-YOLO: When Visual Object Detection Meets Hypergraph Computation,
#   NeurIPS 2024 (dynamic hyperedge incidence over visual tokens).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifi_block_sparse import BlockSparseAttention

__all__ = ['AIFI_TGHBSA']


class HypergraphBlockBridge(nn.Module):
    """Dynamic hypergraph convolution over 2D block vertices of the S5 map.

    Vertices: P x P spatial blocks. Hyperedges: E learnable seeds whose
    attention logits against block descriptors produce a soft incidence
    matrix S [B, E, M]. Normalized hypergraph convolution exchanges messages
    across arbitrary (possibly distant) block subsets.
    """

    def __init__(self, dim, block_size=4, num_hyperedges=8, d=128):
        super().__init__()
        self.dim = dim
        self.block_size = int(block_size)
        self.num_hyperedges = int(num_hyperedges)
        self.d = d

        self.norm = nn.LayerNorm(dim)
        # mean+max pooled block descriptor -> node feature (max keeps weak
        # single-target responses that mean pooling would dilute).
        self.node_proj = nn.Linear(2 * dim, dim)
        self.edge_seed = nn.Parameter(torch.randn(num_hyperedges, d) * 0.02)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(dim, d, bias=False)
        self.v_proj = nn.Linear(dim, dim)
        # Per-hyperedge spatial anchor + distance-decay strength (softplus
        # keeps the extent >= 0). Each hyperedge prefers blocks near its
        # learnable anchor; attention can override this prior.
        self.alpha = nn.Parameter(torch.full((num_hyperedges,), -2.0))
        self.anchor = nn.Parameter(torch.rand(num_hyperedges, 2))
        self.out_norm = nn.LayerNorm(dim)

    def _dist_prior(self, mh, mw, device, dtype):
        """Anchor distance-decay prior [E, M] (per hyperedge, per block)."""
        ys = torch.arange(mh, device=device, dtype=dtype)
        xs = torch.arange(mw, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        pos = torch.stack([gx.flatten() / max(mw - 1, 1),
                           gy.flatten() / max(mh - 1, 1)], dim=1)  # [M, 2]
        dist = torch.cdist(pos[None], self.anchor.to(dtype)[None])[0]  # [M, E]
        dist = dist / (2.0 ** 0.5)
        return -F.softplus(self.alpha).to(dtype)[:, None] * dist.T

    def forward(self, x, h, w):
        """x: [B, N, C] tokens in raster order; h, w: spatial dims."""
        b, n, c = x.shape
        bs = self.block_size
        mh, mw = math.ceil(h / bs), math.ceil(w / bs)
        ph, pw = mh * bs - h, mw * bs - w

        xn = self.norm(x).transpose(1, 2).view(b, c, h, w)
        if ph or pw:
            xn = F.pad(xn, (0, pw, 0, ph))
        # 2D block partition: [B, C, mh*bs, mw*bs] -> [B, M, bs*bs, C]
        blocks = xn.view(b, c, mh, bs, mw, bs)
        blocks = blocks.permute(0, 2, 4, 3, 5, 1).reshape(
            b, mh * mw, bs * bs, c)
        m = mh * mw

        u = torch.cat([blocks.mean(dim=2), blocks.amax(dim=2)], dim=-1)
        u = self.node_proj(u)                       # [B, M, C]

        q = self.q_proj(self.edge_seed)             # [E, d]
        k = self.k_proj(u)                          # [B, M, d]
        logits = torch.einsum('ed,bmd->bem', q, k) / self.d ** 0.5
        logits = logits + self._dist_prior(mh, mw, x.device, x.dtype)
        s = logits.softmax(dim=-1)                  # [B, E, M]

        de = s.sum(dim=-1, keepdim=True) + 1e-6     # [B, E, 1] edge degree
        dv = s.sum(dim=1).unsqueeze(-1) + 1e-6      # [B, M, 1] vertex degree
        z_node = (s.transpose(1, 2) / dv) @ (s / de) @ self.v_proj(u)  # [B, M, C]

        # Broadcast block messages back to tokens (2D unpartition + crop).
        z = z_node.view(b, mh, mw, 1, 1, c).expand(b, mh, mw, bs, bs, c)
        z = z.permute(0, 1, 3, 2, 4, 5).reshape(b, mh * bs, mw * bs, c)
        z = z[:, :h, :w].reshape(b, n, c)
        return self.out_norm(z)


class AIFI_TGHBSA(nn.Module):
    """AIFI with Block-Sparse Attention + Topology-Guided Hypergraph Bridge.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        block_size: Tokens per sparse block (also the hypergraph vertex size).
        topk: Key blocks attended by each query block (Block-Sparse routing).
        num_hyperedges: Number of hyperedges E in the bridge.
        d: Hidden dimension of hyperedge queries/keys.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 num_hyperedges=8, d=128, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before

        # Branch A: existing Block-Sparse global attention.
        self.attn = BlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size, topk=topk,
            dropout=dropout)

        # Branch B: hypergraph topology bridge.
        self.bridge = HypergraphBlockBridge(
            dim=c1, block_size=block_size, num_hyperedges=num_hyperedges,
            d=d)

        # Gated fusion: bias=-2 starts with g~0.12 so the bridge injects
        # gently and the block-sparse path dominates early training.
        self.gate = nn.Linear(2 * c1, c1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256,
                                           temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sincos embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([torch.sin(out_w), torch.cos(out_w),
                          torch.sin(out_h), torch.cos(out_h)], dim=1)[None]

    def _fuse(self, src, h, w, x_bs):
        z = self.bridge(src, h, w)
        g = torch.sigmoid(self.gate(torch.cat([x_bs, z], dim=-1)))
        return x_bs + g * z

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(
                f'AIFI_TGHBSA expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        if self.normalize_before:
            src_norm = self.norm1(src)
            x_bs = src + self.dropout1(self.attn(src_norm))
            src = self._fuse(src_norm, h, w, x_bs)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            x_bs = src + self.dropout1(self.attn(src))
            src = self._fuse(src, h, w, x_bs)
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()



