# Block-sparse global attention V11 for RT-DETR AIFI (dual-scale scheme).
#
# V11 = "extend AIFI from P5 to P4" with a memory-compact top-k gather
# attention. The dense AIFI is limited to P5 because P4 (40x40 = 1600
# tokens) would cost 8 heads x 1600^2 attention pairs with a dense
# implementation. V11 keeps V1's block-sparse math but gathers only the
# selected key blocks, so the attention tensor is (npad x (topk+1)*bs)
# instead of (npad x npad) - ~50x less memory at P4, enabling AIFI there.
#
# Math vs V1 (aifi_block_sparse.py):
#   - routing: identical (mean-pooled raw x, zero parameters, fixed scale)
#   - mask:    V1 = topk blocks + eye(self). V11 forces the diagonal to
#     +inf before topk and selects topk+1 blocks, so the selected set is a
#     SUPERSET of V1's (it always contains self + topk other blocks; V1
#     contains self + topk blocks only when self was not in the raw topk).
#   - softmax: V1 masks non-selected keys with -1e4 inside the dense
#     matrix; V11 only computes the selected keys (exact softmax over the
#     selected subset; masked entries contribute ~0 weight anyway).
#   - invalid (padded) key blocks/tokens are masked with -1e9 (finite, so
#     padded query rows produce uniform garbage instead of NaN; those rows
#     are sliced away before returning and cannot contaminate gradients).
#
# Registration: AIFI_BlockSparseV11 in tasks.py; intended for the P4 layer
# of rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-AIFI-BlockSparseV11.yaml (P5 keeps
# the untouched V1 module for single-variable discipline).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_BlockSparseV11']


class BlockSparseAttentionV11(nn.Module):
    """V1-equivalent block-sparse attention with compact top-k gather.

    Args:
        dim: channels.
        num_heads: attention heads.
        block_size: tokens per block.
        topk: key blocks attended by each query block (excluding self).
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

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, npad, h, d).permute(0, 2, 1, 3) for t in qkv]

        # --- Routing: identical to V1 (zero parameters, raw input) ---
        x_blk = x.view(b, num_blocks, bs, c)
        q_blk = x_blk.mean(dim=2)
        k_blk = q_blk
        block_logits = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * self.scale

        # Exclude fully padded blocks from being selected as keys.
        block_valid = x_blk.abs().sum(dim=(2, 3)) > 0          # (b, nb)
        block_logits = block_logits.masked_fill(
            ~block_valid[:, None, :], -1e9)
        # Guarantee self-coverage (V1 adds the identity block via eye).
        eye = torch.eye(num_blocks, dtype=torch.bool, device=x.device)
        block_logits = block_logits.masked_fill(eye.unsqueeze(0), 1e9)

        k1 = min(self.topk + 1, num_blocks)
        sel = block_logits.topk(k=k1, dim=-1).indices          # (b, nb, k1)

        # --- Compact attention: gather only the selected key tokens ---
        key_tok = sel.unsqueeze(-1) * bs + torch.arange(
            bs, device=x.device).view(1, 1, 1, bs)             # (b, nb, k1, bs)
        key_tok = key_tok.view(b, num_blocks, k1 * bs)
        kt = key_tok.unsqueeze(1).expand(-1, h, -1, -1)        # (b, h, nb, k1*bs)
        kt = kt.reshape(b, h, num_blocks * k1 * bs, 1).expand(-1, -1, -1, d)
        k_g = k.gather(2, kt).view(b, h, num_blocks, k1 * bs, d)
        v_g = v.gather(2, kt).view(b, h, num_blocks, k1 * bs, d)

        q_r = q.view(b, h, num_blocks, bs, d)
        logits = torch.matmul(q_r, k_g.transpose(-2, -1)) * self.scale
        logits = logits.view(b, h, num_blocks, bs, k1 * bs)

        # Mask invalid key tokens (finite -1e9 avoids NaN on padded rows).
        key_invalid = (key_tok >= n)                           # (b, nb, k1*bs)
        if key_invalid.any():
            logits = logits.masked_fill(
                key_invalid[:, None, :, None, :], -1e9)

        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v_g)                          # (b, h, nb, bs, d)
        out = out.reshape(b, h, npad, d).permute(0, 2, 1, 3).reshape(b, npad, c)
        if pad_len:
            out = out[:, :n, :]
        return self.proj_drop(self.proj(out))


class AIFI_BlockSparseV11(nn.Module):
    """AIFI with V11 compact block-sparse attention (any input resolution).

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        block_size: Number of tokens per sparse block.
        topk: Number of key blocks attended by each query block
            (excluding the guaranteed self block).
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = BlockSparseAttentionV11(
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
                f'AIFI_BlockSparseV11 expected {self.c1} channels, got {c}')

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
