# Block-sparse global attention V4 for RT-DETR AIFI.
#
# Design: V1 framework + per-head routing via projected Q/K.
#
# Lessons learned from V1/V2/V3:
#   V1 (baseline, best): 1D row blocks, density ~5%, zero-parameter routing
#     using mean-pooled raw input x, shared routing mask across all heads.
#     Weakness: all 8 heads attend to the same 5 blocks (redundant).
#   V2 (failed): changed to 2D blocks (density 4x higher) + per-head routing.
#     Density increase destroyed sparsity regularization.
#   V3 (failed): added MLP router, routing bias, QK-norm, DWConv, gates on
#     top of V2. More complexity = worse.
#
# V4 keeps V1's proven framework and fixes only the identified weakness:
#   - 1D row blocks (block_size=4) -> identical to V1
#   - Density ~5% per head -> identical to V1
#   - Per-head routing from projected Q/K (zero extra parameters)
#   - Each head independently selects its own top-k key blocks
#   - Union coverage across heads can reach ~33/100 blocks (vs V1's 5/100)
#   - No MLP router, no learnable temperature, no QK-norm, no bias,
#     no DWConv, no gates, no separate QKV

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_BlockSparseV4']


class BlockSparseAttentionV4(nn.Module):
    """V1-style block-sparse attention with per-head routing.

    The ONLY difference from V1: routing logits are computed per-head from
    the projected Q/K tensors (not from raw input x shared across heads).
    Block structure, density, and all other aspects are identical to V1.
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

    def forward(self, x):
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        bs = self.block_size
        num_blocks = math.ceil(n / bs)
        pad_len = num_blocks * bs - n

        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        npad = num_blocks * bs

        # Fused QKV projection (same as V1).
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, npad, h, d).permute(0, 2, 1, 3) for t in qkv]
        # q, k, v: (b, H, npad, d)

        # === V4 KEY CHANGE: per-head routing from projected Q/K ===
        # Reshape per-head Q/K into blocks and mean-pool:
        #   (b, H, npad, d) -> (b, H, nb, bs, d) -> mean -> (b, H, nb, d)
        q_blk = q.view(b, h, num_blocks, bs, d).mean(dim=3)
        k_blk = k.view(b, h, num_blocks, bs, d).mean(dim=3)

        # Per-head routing logits: (b, H, nb, nb)
        block_logits = torch.einsum('bhid,bhjd->bhij', q_blk, k_blk) * self.scale

        topk = min(self.topk, num_blocks)
        # Per-head top-k indices: (b, H, nb, topk)
        topk_idx = block_logits.topk(k=topk, dim=-1).indices

        # Build per-head block mask: (b, H, nb, nb)
        block_mask = torch.zeros(
            b, h, num_blocks, num_blocks, dtype=torch.bool, device=x.device)
        block_mask.scatter_(-1, topk_idx, True)
        eye = torch.eye(num_blocks, dtype=torch.bool, device=x.device)
        block_mask = block_mask | eye.unsqueeze(0).unsqueeze(0)

        # Disable padded key blocks.
        if pad_len:
            valid_tokens = torch.zeros(npad, dtype=torch.bool, device=x.device)
            valid_tokens[:n] = True
            block_valid = valid_tokens.view(num_blocks, bs).any(dim=1)  # (nb,)
            block_mask = block_mask & block_valid.view(1, 1, 1, -1)

        # Expand per-head block mask to per-head token mask:
        # (b, H, nb, nb) -> (b, H, npad, npad)
        token_mask = block_mask.reshape(b * h, num_blocks, num_blocks)
        token_mask = token_mask.repeat_interleave(bs, dim=1)
        token_mask = token_mask.repeat_interleave(bs, dim=2)
        token_mask = token_mask.view(b, h, npad, npad)

        # Mask padded key positions at token level.
        if pad_len:
            valid = torch.zeros(b, npad, dtype=torch.bool, device=x.device)
            valid[:, :n] = True
            token_mask = token_mask & valid[:, None, None, :]

        # Standard attention computation (same as V1, but mask is per-head).
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (b, H, npad, npad)
        logits = logits.masked_fill(~token_mask, -1e4)
        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v)  # (b, H, npad, d)
        out = out.permute(0, 2, 1, 3).reshape(b, npad, c)
        if pad_len:
            out = out[:, :n, :]
        return self.proj_drop(self.proj(out))


class AIFI_BlockSparseV4(nn.Module):
    """AIFI with V4 block-sparse attention.

    Identical to AIFI_BlockSparse (V1) except attention uses per-head
    routing computed from projected Q/K. All other components (FFN, norms,
    position embedding, residual connections) are unchanged from V1.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        block_size: Number of tokens per 1D row block.
        topk: Number of key blocks attended by each query block per head.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = BlockSparseAttentionV4(
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
                f'AIFI_BlockSparseV4 expected {self.c1} channels, got {c}')

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
