# Block-sparse global attention V15 for RT-DETR AIFI.
#
# V15 = V13 (V1 on 2x2 spatial blocks) reimplemented on V1's DENSE MASKED
# attention path - no gather. The full npad x npad logit matrix is computed
# and non-selected entries masked with -1e4 before a full softmax, exactly
# like aifi_block_sparse.py (V1).
#
# Selection logic identical to V13 (identical to V1): mean-pooled raw input
# routing, zero parameters, fixed scale, topk + guaranteed self block,
# shared mask across heads, 5% density at P5.
#
# The 2x2 blockification uses pure view/permute/reshape (no gather):
#   (b, H, W, c) -> (b, bh, s, bw, s, c) -> (b, bh, bw, s, s, c)
#                -> (b, nb, s*s, c)                    [block-major layout]
# and the inverse permutation restores the original spatial order.
#
# Memory: full npad x npad logits - fine at P5 (400 tokens), NOT suitable
# for P4 (1600 tokens). Use aifi_block_sparse_v13.py for P4.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_BlockSparseV15']


class BlockSparseAttentionV15(nn.Module):
    """V1 attention on 2x2 spatial blocks, dense masked implementation."""

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

    def forward(self, x, h=None, w=None):
        b, n, c = x.shape
        if h is None:
            h = int(math.sqrt(n))
        if w is None:
            w = n // h
        assert h * w == n, f'spatial {h}x{w} != tokens {n}'

        H_, d = self.num_heads, self.head_dim
        s = self.block_size
        blksz = s * s

        # Spatial padding to a multiple of s (no-op at 20x20 / 640 input).
        ph = (s - h % s) % s
        pw = (s - w % s) % s
        H, W = h + ph, w + pw
        x2d = x.view(b, h, w, c)
        if ph or pw:
            x2d = F.pad(x2d, (0, 0, 0, pw, 0, ph))

        bh, bw = H // s, W // s
        nb = bh * bw
        npad = nb * blksz

        # --- Blockify WITHOUT gather: pure view/permute/reshape ---
        xb = x2d.view(b, bh, s, bw, s, c).permute(0, 1, 3, 2, 4, 5) \
                .reshape(b, nb, blksz, c)                          # block-major

        # --- Routing: identical to V1 (mean-pooled raw input, zero params) ---
        q_blk = xb.mean(dim=2)
        block_logits = torch.matmul(q_blk, q_blk.transpose(-2, -1)) * self.scale

        block_valid = xb.abs().sum(dim=(2, 3)) > 0
        block_logits = block_logits.masked_fill(~block_valid[:, None, :], -1e9)
        # Guarantee self-coverage (V1's eye).
        eye = torch.eye(nb, dtype=torch.bool, device=x.device)
        block_logits = block_logits.masked_fill(eye.unsqueeze(0), 1e9)

        k1 = min(self.topk + 1, nb)
        sel = block_logits.topk(k=k1, dim=-1).indices              # (b, nb, k1)
        block_mask = torch.zeros(b, nb, nb, dtype=torch.bool, device=x.device)
        block_mask.scatter_(-1, sel, True)
        block_mask = block_mask | eye.unsqueeze(0)

        # --- V1 dense masked attention path on the block-major sequence ---
        qkv = self.qkv(xb).chunk(3, dim=-1)
        q, k, v = [t.view(b, npad, H_, d).permute(0, 2, 1, 3) for t in qkv]

        token_mask = block_mask.repeat_interleave(blksz, dim=1)
        token_mask = token_mask.repeat_interleave(blksz, dim=2)

        # Mask tokens of spatially padded rows/cols (they must not act as keys).
        rr = torch.arange(H, device=x.device).view(H, 1).expand(H, W) \
                .reshape(nb, blksz)
        cc = torch.arange(W, device=x.device).view(1, W).expand(H, W) \
                .reshape(nb, blksz)
        tok_valid = (rr < h) & (cc < w)                            # (nb, blksz)
        token_mask = token_mask & tok_valid.view(1, 1, npad)

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(~token_mask.unsqueeze(1), -1e4)
        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v)                                # (b, H, npad, d)
        out = out.permute(0, 2, 1, 3).reshape(b, npad, c)

        # --- Inverse blockification (no gather) ---
        out = out.view(b, bh, bw, s, s, c).permute(0, 1, 3, 2, 4, 5) \
                 .reshape(b, H, W, c)[:, :h, :w].reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class AIFI_BlockSparseV15(nn.Module):
    """AIFI with V15 block-sparse attention on 2x2 spatial blocks
    (dense masked implementation).

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
        self.attn = BlockSparseAttentionV15(
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
                f'AIFI_BlockSparseV15 expected {self.c1} channels, got {c}')
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
