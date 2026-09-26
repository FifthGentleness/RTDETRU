# QK-Norm version of RT-DETR AIFI.
#
# Mechanism source:
#   "Scaling Vision Transformers to 22 Billion Parameters"
#   Dehghani et al., Google, arXiv:2302.05442 (2023).
#   (QK-norm is a widely adopted stability mechanism in 2024-2025 models.)
#
# Core idea:
#   q_hat = (q / ||q||_2) * s_q,   k_hat = (k / ||k||_2) * s_k
#
#   L2-normalizing Q and K after projection bounds the attention logits to a
#   common magnitude, which prevents a few high-norm tokens (large objects /
#   dominant background semantics at P5) from saturating the softmax and
#   starving weak small-object tokens of gradient. The per-head learnable
#   scales (s_q, s_k) act as a learnable temperature and let each head recover
#   its preferred sharpness during training.
#
# Design decisions (lesson-driven):
#   1. Pure normalization change: NO rank compression, NO sparsity, NO
#      binarization, NO token compression. The attention is still an exact,
#      full-length, full-precision MHSA.
#   2. Warm start by construction: normalization cannot inject random noise
#      at t=0; logits start bounded (flat, high-entropy attention) and the
#      learnable scales sharpen them only if training demands it.
#   3. Positional embedding enters Q/K only; V and the residual stream stay
#      free of pos (original RT-DETR AIFI / DETR convention).
#   4. Orthogonal to every other AIFI variant: it can be stacked on top of
#      differential attention / neighborhood branches in later experiments.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['QKNormAttention', 'AIFI_QKNorm']


class QKNormAttention(nn.Module):
    """Multi-head self-attention with per-head L2-normalized Q/K and learnable scales.

    Args:
        dim: Token dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        scale_init: Initial value of the per-head Q/K scales (default 1.0;
            larger values start with sharper attention).
    """

    def __init__(self, dim, num_heads=8, dropout=0.0, scale_init=1.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        # Per-head learnable scales -> effective temperature s_q * s_k.
        self.q_scale = nn.Parameter(torch.full((num_heads,), float(scale_init)))
        self.k_scale = nn.Parameter(torch.full((num_heads,), float(scale_init)))

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, qk_input, v_input):
        """Attention with separated Q/K and V sources (DETR convention).

        Args:
            qk_input: (B, N, C), features + positional embedding.
            v_input: (B, N, C), clean features without pos.
        """
        b, n, c = qk_input.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
        v = self.v_proj(v_input)

        q = q.view(b, n, h, d).permute(0, 2, 1, 3)  # (B, h, N, d)
        k = k.view(b, n, h, d).permute(0, 2, 1, 3)
        v = v.view(b, n, h, d).permute(0, 2, 1, 3)

        # QK-Norm: L2-normalize then rescale per head.
        q = F.normalize(q, dim=-1) * self.q_scale.view(1, h, 1, 1)
        k = F.normalize(k, dim=-1) * self.k_scale.view(1, h, 1, 1)

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B, h, N, d)

        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class AIFI_QKNorm(nn.Module):
    """RT-DETR AIFI with QK-norm stabilized MHSA.

    Interface is identical to the original AIFI: input and output are both
    ``(B, C, H, W)`` feature maps.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True (AIFI default: False).
        scale_init: Initial per-head Q/K scale (attention temperature).
    """

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0,
                 act=nn.GELU(), normalize_before=False, scale_init=1.0):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = QKNormAttention(
            c1, num_heads=num_heads, dropout=dropout, scale_init=scale_init)
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
            'Embed dimension must be divisible by 4 for 2D sincos position embedding'
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
            raise ValueError(f'AIFI_QKNorm expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        # pos enters Q/K only; the residual stream and V stay free of pos.
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
