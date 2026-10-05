# Block-sparse global attention V2 for RT-DETR AIFI.
#
# Focus: maximize precision by improving ROUTING QUALITY, not density.
#
# Key changes vs V1 (aifi_block_sparse.py):
#   1. 2D spatial block partitioning (block_size=4 -> 4x4 spatial patches)
#      instead of flat 1D rows. Blocks have spatial meaning.
#   2. Per-head independent top-k routing: each attention head selects its own
#      top-k key blocks. With 8 heads, effective global coverage ~= 8x topk
#      unique blocks (assuming heads learn diverse routing patterns).
#   3. Routing based on projected Q/K (not raw input features). Routing now
#      operates in the same representation space as attention.
#   4. Per-head learnable routing temperature.
#   5. Padded positions excluded from block descriptors.
#   6. NO local window: preserves V1's pure top-k sparse advantage.
#
# Density analysis for P5 (20x20 tokens, block_size=4):
#   V1: 100 1D blocks x topk=4 (all heads share) = 5/100 = 5% density
#   V2: 25 2D blocks x topk=4 per head (independent per head)
#       Per-head: 5/25 = 20% density (but with spatial meaning)
#       Effective across heads: up to 8x4+1 = 33 unique blocks / 25 = >100%
#       (i.e. full coverage if heads diversify, ~20% if they collapse)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_BlockSparseV2']


class BlockSparseAttentionV2(nn.Module):
    """Pure top-k block-sparse attention with per-head routing and 2D blocks.

    No local window. Each head independently selects its top-k key blocks
    based on Q/K similarity, preserving the sparse advantage while improving
    routing accuracy and head diversity.
    """

    def __init__(self, dim, num_heads=8, block_size=4, topk=4, dropout=0.0):
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
        self.block_size = int(block_size)
        self.topk = int(topk)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Per-head learnable routing temperature.
        self.route_scale = nn.Parameter(
            torch.ones(num_heads) * self.scale)

        # Cache for spatial mappings (rebuilt when spatial size changes).
        self._cache_hw = None
        self._cache = None

    def _build_mappings(self, h, w, bs, device):
        """Build vectorized 2D block mappings.

        Returns:
            gather_idx: (npad,) original token index for each reordered position.
            bh, bw, num_blocks, npad: block grid info.
        """
        bh = math.ceil(h / bs)
        bw = math.ceil(w / bs)
        num_blocks = bh * bw
        blksz = bs * bs
        npad = num_blocks * blksz

        # Token to block mapping (vectorized).
        r = torch.arange(h, device=device).unsqueeze(1).expand(h, w).flatten()
        c = torch.arange(w, device=device).unsqueeze(0).expand(h, w).flatten()

        br = r // bs
        bc = c // bs
        block_id = br * bw + bc
        pos_in_block = (r % bs) * bs + (c % bs)

        # Build block_order: (num_blocks, blksz) original token index.
        block_order = torch.full((num_blocks, blksz), -1,
                                 dtype=torch.long, device=device)
        orig_idx = torch.arange(h * w, device=device)
        block_order[block_id, pos_in_block] = orig_idx

        gather_idx = block_order.flatten()  # (npad,)

        return gather_idx, bh, bw, num_blocks, npad

    def forward(self, x, h=None, w=None):
        b, n, c = x.shape
        if h is None:
            h = int(math.sqrt(n))
        if w is None:
            w = n // h
        assert h * w == n, f'spatial {h}x{w} != tokens {n}'

        H, d = self.num_heads, self.head_dim
        bs = self.block_size
        blksz = bs * bs

        # Build/cache spatial mappings.
        cache_key = (h, w)
        if self._cache_hw != cache_key or self._cache[0].device != x.device:
            gather_idx, bh, bw, num_blocks, npad = \
                self._build_mappings(h, w, bs, x.device)
            self._cache_hw = cache_key
            self._cache = (gather_idx, bh, bw, num_blocks, npad)
        gather_idx, bh, bw, num_blocks, npad = self._cache

        # Expand for batch.
        gather_b = gather_idx.unsqueeze(0).expand(b, -1)  # (b, npad)
        valid_mask = gather_b >= 0  # (b, npad)
        gather_safe = gather_b.clamp(min=0)

        # Reorder tokens into block-major layout.
        x_blk = x.gather(1, gather_safe.unsqueeze(-1).expand(-1, -1, c))
        x_blk = x_blk * valid_mask.unsqueeze(-1).to(x.dtype)

        # QKV projections.
        qkv = self.qkv(x_blk).chunk(3, dim=-1)
        q, k, v = [t.view(b, npad, H, d).permute(0, 2, 1, 3) for t in qkv]
        # q, k, v: (b, H, npad, d)

        # Block descriptors from projected Q and K per head.
        # Reshape: (b, H, num_blocks, blksz, d)
        q_blk_full = q.view(b, H, num_blocks, blksz, d)
        k_blk_full = k.view(b, H, num_blocks, blksz, d)
        valid_blk = valid_mask.view(b, num_blocks, blksz).float()  # (b, nb, blksz)

        # Mean over valid tokens per block: (b, H, num_blocks, d)
        vsum = valid_blk.sum(2, keepdim=True).clamp(min=1).unsqueeze(1)  # (b,1,nb,1)
        q_blk = (q_blk_full * valid_blk.unsqueeze(1).unsqueeze(-1)).sum(3) / vsum
        k_blk = (k_blk_full * valid_blk.unsqueeze(1).unsqueeze(-1)).sum(3) / vsum

        # Per-head block logits: (b, H, num_blocks, num_blocks)
        scale = self.route_scale.view(1, H, 1, 1).to(x.dtype)
        block_logits = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * scale

        # Mask invalid key blocks.
        block_valid = valid_blk.sum(2) > 0  # (b, num_blocks)
        block_logits = block_logits.masked_fill(
            ~block_valid.unsqueeze(1).unsqueeze(1), -1e4)

        # Per-head top-k selection.
        topk = min(self.topk, num_blocks)
        topk_idx = block_logits.topk(k=topk, dim=-1).indices
        # topk_idx: (b, H, num_blocks, topk)

        # Build per-head block mask: (b, H, num_blocks, num_blocks)
        block_mask = torch.zeros(
            b, H, num_blocks, num_blocks, dtype=torch.bool, device=x.device)
        block_mask.scatter_(-1, topk_idx, True)

        # Add identity (self-block).
        eye = torch.eye(num_blocks, dtype=torch.bool, device=x.device)
        block_mask = block_mask | eye.unsqueeze(0).unsqueeze(0)

        # Mask invalid key blocks.
        block_mask = block_mask & block_valid.unsqueeze(1).unsqueeze(1)

        # Expand to token-level mask per head:
        # (b, H, nb, nb) -> (b, H, npad, npad)
        token_mask = block_mask.reshape(b * H, num_blocks, num_blocks)
        token_mask = token_mask.repeat_interleave(blksz, dim=1)
        token_mask = token_mask.repeat_interleave(blksz, dim=2)
        token_mask = token_mask.view(b, H, npad, npad)

        # Mask padded key positions.
        token_mask = token_mask & valid_mask.unsqueeze(1).unsqueeze(1)

        # Attention per head.
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (b,H,npad,npad)
        logits = logits.masked_fill(~token_mask, -1e4)
        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v)  # (b, H, npad, d)
        out = out.permute(0, 2, 1, 3).reshape(b, npad, c)

        # Scatter back to original order.
        out_original = torch.zeros_like(out)
        valid_exp = valid_mask.unsqueeze(-1).expand(-1, -1, c)
        out_original.scatter_(
            1, gather_safe.unsqueeze(-1).expand(-1, -1, c),
            out * valid_exp.to(out.dtype))
        out_original = out_original[:, :n, :]

        return self.proj_drop(self.proj(out_original))


class AIFI_BlockSparseV2(nn.Module):
    """AIFI with V2 block-sparse attention: pure top-k, per-head routing, 2D blocks.

    Key improvements over V1 (all preserve sparsity, no local window):
      - 2D spatial blocks (4x4 patches) instead of flat 1D rows
      - Per-head independent top-k routing (up to 8x global coverage)
      - Routing from projected Q/K (same representation space as attention)
      - Per-head learnable routing temperature
      - Padded tokens excluded from routing descriptors

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        block_size: Spatial block size (tokens per side of 2D block).
        topk: Number of globally attended key blocks per query block per head.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = BlockSparseAttentionV2(
            dim=c1,
            num_heads=num_heads,
            block_size=block_size,
            topk=topk,
            dropout=dropout,
        )
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
                f'AIFI_BlockSparseV2 expected {self.c1} channels, got {c}')

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
