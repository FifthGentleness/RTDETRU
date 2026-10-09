# Block-sparse global attention V9 for RT-DETR AIFI.
#
# V9 = V6 + V7 (both individually validated against V1 as single variables):
#   - V6 component: second-order block descriptors concat([mean, std])
#     for routing (COBS-inspired, arXiv:2607.09052).
#   - V7 component: zero-init-gated coarse compressed branch (NSA-inspired,
#     arXiv:2502.11089) that guarantees global coverage on top of top-k
#     selection.
#
# Both components keep their individually validated properties:
#   - Routing: zero learnable parameters, fixed route_scale (2c)^-0.5.
#   - Compressed branch: reuses qkv weights, gated by a single zero-init
#     scalar (training starts exactly at V6 behavior).
#   - Density: selected branch ~5% token-level, compressed branch is
#     block-level and negligible.
#
# NOTE: the combination itself must be re-validated against V6 and V7
# individually - single-variable gains do not automatically add up.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_BlockSparseV9']


class BlockSparseAttentionV9(nn.Module):
    """Second-order routing (V6) + zero-init-gated compressed branch (V7)."""

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

        # Fixed routing scale for 2c-dim descriptors (zero parameters, V6).
        self.route_scale = (2 * dim) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Zero-init gate for the compressed branch (the ONLY new parameter).
        self.compress_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        bs = self.block_size
        num_blocks = math.ceil(n / bs)
        pad_len = num_blocks * bs - n

        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        npad = num_blocks * bs

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, npad, h, d).permute(0, 2, 1, 3) for t in qkv]

        # --- V6 component: second-order block routing ---
        x_blk = x.view(b, num_blocks, bs, c)
        q_blk = torch.cat([x_blk.mean(dim=2), x_blk.std(dim=2, unbiased=False)], dim=-1)
        k_blk = q_blk
        block_logits = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * self.route_scale
        topk = min(self.topk, num_blocks)
        topk_idx = block_logits.topk(k=topk, dim=-1).indices

        block_mask = torch.zeros(
            b, num_blocks, num_blocks, dtype=torch.bool, device=x.device)
        block_mask.scatter_(-1, topk_idx, True)
        eye = torch.eye(num_blocks, dtype=torch.bool, device=x.device)
        block_mask = block_mask | eye.unsqueeze(0)

        token_mask = block_mask.repeat_interleave(bs, dim=1)
        token_mask = token_mask.repeat_interleave(bs, dim=2)
        token_mask = token_mask[:, :npad, :npad]
        if pad_len:
            valid = torch.zeros(b, npad, dtype=torch.bool, device=x.device)
            valid[:, :n] = True
            token_mask = token_mask & valid[:, None, :]

        # --- Selected branch: identical to V1/V6 ---
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(~token_mask.unsqueeze(1), -1e4)
        attn = self.attn_drop(logits.softmax(dim=-1))
        out_sel = torch.matmul(attn, v)

        # --- V7 component: coarse compressed branch over ALL pooled blocks ---
        pooled = x_blk.mean(dim=2)                       # (b, nb, c)
        qkv_c = self.qkv(pooled).chunk(3, dim=-1)        # reuse qkv weights
        k_c = qkv_c[1].view(b, num_blocks, h, d).permute(0, 2, 1, 3)
        v_c = qkv_c[2].view(b, num_blocks, h, d).permute(0, 2, 1, 3)

        coarse_logits = torch.matmul(q, k_c.transpose(-2, -1)) * self.scale
        if pad_len:
            block_valid = x_blk.abs().sum(dim=(2, 3)) > 0     # (b, nb)
            coarse_logits = coarse_logits.masked_fill(
                ~block_valid[:, None, None, :], -1e4)
        attn_c = self.attn_drop(coarse_logits.softmax(dim=-1))
        out_cmp = torch.matmul(attn_c, v_c)               # (b, h, npad, d)

        # --- Merge: selected (fixed 1.0) + zero-init-gated compressed ---
        out = out_sel + self.compress_gate * out_cmp
        out = out.permute(0, 2, 1, 3).reshape(b, npad, c)
        if pad_len:
            out = out[:, :n, :]
        return self.proj_drop(self.proj(out))


class AIFI_BlockSparseV9(nn.Module):
    """AIFI with V9 block-sparse attention: V6 second-order routing + V7
    zero-init-gated compressed branch.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        block_size: Number of tokens per sparse block.
        topk: Number of key blocks attended by each query block.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = BlockSparseAttentionV9(
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
                f'AIFI_BlockSparseV9 expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        if self.normalize_before:
            src_norm = self.norm1(src)
            src = src + self.dropout1(self.attn(src_norm))
            src_norm = self.norm2(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src_norm)))))
        else:
            src = src + self.dropout1(self.attn(src))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()
