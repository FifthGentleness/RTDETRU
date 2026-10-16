# Block-sparse global attention V13 for RT-DETR AIFI.
#
# V13 = V1 with 2x2 SPATIAL blocks instead of 1D row blocks - a clean
# single-variable geometry ablation.
#
# Why this experiment: V2 changed THREE variables at once (2D blocks 4x4
# [density x4] + per-head routing + projected-Q/K routing) and failed; its
# failure does not implicate block GEOMETRY alone. V13 isolates geometry:
#
#   V1 : 1D row blocks, 4 tokens each  -> P5: 100 blocks (1x4 strips)
#   V13: 2D 2x2 blocks, 4 tokens each  -> P5: 100 blocks (spatial squares)
#
# Token count per block (4), block count (100), density (topk=4 + self
# = 5%), routing (mean-pooled raw x, zero parameters), shared mask, and
# the whole qkv/FFN/norm stack are IDENTICAL to V1. Only the block
# partitioning changes.
#
# Implemented with the compact gather attention (V11-style) so it also
# runs efficiently at P4 (40x40 -> 400 blocks of 2x2).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_BlockSparseV13']


class BlockSparseAttentionV13(nn.Module):
    """V1 attention on 2x2 spatial blocks (compact gather implementation)."""

    def __init__(self, dim, num_heads=8, block_size=2, topk=4, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        if block_size < 1:
            raise ValueError('block_size must be >= 1')
        if topk < 1:
            raise ValueError('topk must be >= 1')

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = int(block_size)   # spatial side; block = s x s tokens
        self.topk = int(topk)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self._cache_hw = None
        self._cache = None

    def _build_mappings(self, h, w, s, device):
        """Token -> 2D block-major order mapping.

        Returns:
            gather_idx: (npad,) original token index per reordered position
                (-1 for padding slots), block-major layout.
            nb: number of blocks, npad: reordered token count.
        """
        bh = math.ceil(h / s)
        bw = math.ceil(w / s)
        nb = bh * bw
        blksz = s * s
        npad = nb * blksz

        r = torch.arange(h, device=device).unsqueeze(1).expand(h, w).flatten()
        c = torch.arange(w, device=device).unsqueeze(0).expand(h, w).flatten()
        bid = (r // s) * bw + (c // s)
        pos = (r % s) * s + (c % s)

        block_order = torch.full((nb, blksz), -1, dtype=torch.long, device=device)
        block_order[bid, pos] = torch.arange(h * w, device=device)
        return block_order.flatten(), nb, npad

    def forward(self, x, h=None, w=None):
        b, n, c = x.shape
        if h is None:
            h = int(math.sqrt(n))
        if w is None:
            w = n // h
        assert h * w == n, f'spatial {h}x{w} != tokens {n}'

        H, d = self.num_heads, self.head_dim
        s = self.block_size
        blksz = s * s

        if self._cache_hw != (h, w) or self._cache is None \
                or self._cache[0].device != x.device:
            gather_idx, nb, npad = self._build_mappings(h, w, s, x.device)
            self._cache_hw = (h, w)
            self._cache = (gather_idx, nb, npad)
        gather_idx, nb, npad = self._cache

        # Reorder tokens into block-major layout (2x2 spatial blocks).
        valid = gather_idx >= 0                                     # (npad,)
        gather_safe = gather_idx.clamp(min=0)
        gather_b = gather_safe.unsqueeze(0).expand(b, -1)
        x_r = x.gather(1, gather_b.unsqueeze(-1).expand(-1, -1, c))
        x_r = x_r * valid.unsqueeze(0).unsqueeze(-1).to(x.dtype)

        qkv = self.qkv(x_r).chunk(3, dim=-1)
        q, k, v = [t.view(b, npad, H, d).permute(0, 2, 1, 3) for t in qkv]

        # --- Routing: mean-pooled raw (reordered) input, zero parameters ---
        x_blk = x_r.view(b, nb, blksz, c)
        q_blk = x_blk.mean(dim=2)
        block_logits = torch.matmul(q_blk, q_blk.transpose(-2, -1)) * self.scale

        block_valid = x_blk.abs().sum(dim=(2, 3)) > 0
        block_logits = block_logits.masked_fill(~block_valid[:, None, :], -1e9)
        # Guarantee self-coverage (V1 adds the identity block via eye).
        eye = torch.eye(nb, dtype=torch.bool, device=x.device)
        block_logits = block_logits.masked_fill(eye.unsqueeze(0), 1e9)

        k1 = min(self.topk + 1, nb)
        sel = block_logits.topk(k=k1, dim=-1).indices               # (b, nb, k1)

        # --- Compact attention over the selected blocks ---
        key_tok = sel.unsqueeze(-1) * blksz + torch.arange(
            blksz, device=x.device).view(1, 1, 1, blksz)            # (b, nb, k1, blksz)
        key_tok = key_tok.view(b, nb, k1 * blksz)
        kt = key_tok.unsqueeze(1).expand(-1, H, -1, -1)
        kt = kt.reshape(b, H, nb * k1 * blksz, 1).expand(-1, -1, -1, d)
        k_g = k.gather(2, kt).view(b, H, nb, k1 * blksz, d)
        v_g = v.gather(2, kt).view(b, H, nb, k1 * blksz, d)

        q_r = q.view(b, H, nb, blksz, d)
        logits = torch.matmul(q_r, k_g.transpose(-2, -1)) * self.scale
        logits = logits.view(b, H, nb, blksz, k1 * blksz)

        key_invalid = (key_tok >= n) | (key_tok < 0)                # -1 slots
        if key_invalid.any():
            logits = logits.masked_fill(key_invalid[:, None, :, None, :], -1e9)

        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v_g)
        out = out.reshape(b, H, npad, d).permute(0, 2, 1, 3).reshape(b, npad, c)

        # Scatter back to the original spatial order.
        out_orig = torch.zeros_like(out)
        valid_exp = valid.unsqueeze(0).unsqueeze(-1).expand(b, npad, c)
        out_orig.scatter_(1, gather_b.unsqueeze(-1).expand(-1, -1, c),
                          out * valid_exp.to(out.dtype))
        out_orig = out_orig[:, :n, :]
        return self.proj_drop(self.proj(out_orig))


class AIFI_BlockSparseV13(nn.Module):
    """AIFI with V13 block-sparse attention on 2x2 spatial blocks.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        block_size: Spatial side of a block (block = block_size^2 tokens).
        topk: Number of key blocks attended by each query block
            (excluding the guaranteed self block).
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=2, topk=4,
                 dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = BlockSparseAttentionV13(
            dim=c1, num_heads=num_heads, block_size=block_size,
            topk=topk, dropout=dropout)
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

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(
                f'AIFI_BlockSparseV13 expected {self.c1} channels, got {c}')
        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos
        if self.normalize_before:
            src_norm = self.norm1(src)
            src = src + self.dropout1(self.attn(src_norm, h=h, w=w))
            src_norm = self.norm2(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src_norm)))))
        else:
            src = src + self.dropout1(self.attn(src, h=h, w=w))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)
        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()
