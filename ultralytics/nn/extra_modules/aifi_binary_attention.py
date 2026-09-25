# One-bit QK-attention version of RT-DETR AIFI.
#
# Reference:
#   BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers
#   CVPR 2026. https://github.com/EdwardChasel/BinaryAttention
#
# This module binarizes only Q and K with a sign function and a straight-through
# estimator. V and all output projections remain full precision, which is safer
# for small-object features than binarizing the whole attention path.
#
# FIX (pos-leak bug): previously `src = x + pos` was used as the residual
# stream, leaking the fixed positional embedding into the residual and FFN.
# Now pos is added to Q/K only (attention weights), matching the original
# RT-DETR AIFI semantics (q = k = src + pos, value = src).

import torch
import torch.nn as nn

__all__ = ['AIFI_BinaryAttention']


def sign_ste(x):
    """Sign function with a straight-through gradient estimator."""
    sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    return x + (sign - x).detach()


class BinaryQKAttention(nn.Module):
    """Multi-head attention with one-bit Q/K and full-precision V."""

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, qk_input, v_input):
        """Apply attention with separated Q/K and V inputs (DETR convention).

        pos is added to Q/K only (by the caller); the residual stream and V
        stay free of positional embedding, matching the original RT-DETR AIFI
        (q = k = src + pos, value = src).
        """
        b, n, c = qk_input.shape
        h, d = self.num_heads, self.head_dim
        q, k = self.qkv(qk_input).chunk(3, dim=-1)[:2]
        v = self.qkv(v_input).chunk(3, dim=-1)[2]
        q = q.view(b, n, h, d).permute(0, 2, 1, 3)
        k = k.view(b, n, h, d).permute(0, 2, 1, 3)
        v = v.view(b, n, h, d).permute(0, 2, 1, 3)

        q = sign_ste(q)
        k = sign_ste(k)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale + self.bias
        logits = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(logits, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class AIFI_BinaryAttention(nn.Module):
    """AIFI with one-bit Q/K attention.

    The interface is identical to the original AIFI: input and output are both
    ``(B, C, H, W)`` feature maps.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0,
                 act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = BinaryQKAttention(c1, num_heads, dropout)
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
                f'AIFI_BinaryAttention expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        # pos enters Q/K only (attention weights); the residual stream and V
        # stay free of positional embedding, matching the original AIFI
        # (TransformerEncoderLayer.forward_post: q = k = src + pos, value = src).
        src = x.flatten(2).permute(0, 2, 1)  # clean residual stream

        if self.normalize_before:
            src_norm = self.norm1(src)
            src = src + self.dropout1(self.attn(src_norm + pos, src_norm))
            src_norm = self.norm2(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src_norm)))))
        else:
            src = src + self.dropout1(self.attn(src + pos, src))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()
