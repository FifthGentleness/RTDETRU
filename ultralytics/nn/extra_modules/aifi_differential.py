# Differential-attention version of RT-DETR AIFI.
#
# Reference:
#   Differential Transformer (ICLR 2025, arXiv:2410.05258)
#   Tianzhu Ye et al., Microsoft Research.
#   Official code: https://github.com/microsoft/unilm (diff-transformer)
#
# Core idea:
#   Attn(X) = ( softmax(Q1 K1^T / sqrt(d)) - lambda * softmax(Q2 K2^T / sqrt(d)) ) V
#
#   The two attention maps share the "common-mode" background response
#   (dominant background / large-object tokens at P5); subtracting the second
#   map cancels this common mode and leaves the differential (contrast) signal,
#   which relatively amplifies weak-but-distinct small-object responses.
#
# Design decisions (lesson-driven, see AgentV2/BA/SET/BlockSparse failures):
#   1. Both branches are FULL-PRECISION, FULL-LENGTH softmax attention.
#      No rank compression, no binarization, no hard sparsity, no fixed agent
#      tokens: exact QK content matching is preserved everywhere.
#   2. lambda is learnable, sigmoid-parameterized and initialized at 0.2
#      (single-layer warm start: weak subtraction at t=0, so the module starts
#      close to a standard MHSA and learns the subtraction depth).
#   3. Per-head lambda (more flexible than the official head-grouping; grouping
#      is an efficiency detail, not the mechanism).
#   4. Positional embedding enters Q/K only; V and the residual stream stay
#      free of pos (original RT-DETR AIFI / DETR convention).
#   5. V stays single-projection full precision (only Q/K are doubled).
#
# Compared with SET: a [0,1] spectral gate can only suppress; differential
# attention can both suppress (common mode) and amplify (differential mode).

import math

import torch
import torch.nn as nn

__all__ = ['DifferentialAttention', 'AIFI_DiffAttn']


class DifferentialAttention(nn.Module):
    """Full-precision dual-softmax attention with a learnable subtraction.

    out = (softmax(q1 k1^T) - lambda * softmax(q2 k2^T)) v

    Args:
        dim: Token dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        lambda_init: Initial value of the subtraction coefficient (default 0.2,
            warm start; the official paper schedules 0.2 for the first layer).
    """

    def __init__(self, dim, num_heads=8, dropout=0.0, lambda_init=0.2):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        if not (0.0 < lambda_init < 1.0):
            raise ValueError(f'lambda_init must be in (0, 1), got {lambda_init}')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q1/Q2/K1/K2 from the pos-carrying input; V from the clean input.
        self.qk_proj = nn.Linear(dim, dim * 4)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        # Per-head learnable lambda, sigmoid-parameterized to (0, 1).
        self.lambda_logits = nn.Parameter(
            torch.full((num_heads,), math.log(lambda_init / (1.0 - lambda_init))))

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qk_proj.weight)
        nn.init.zeros_(self.qk_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, qk_input, v_input):
        """Apply differential attention with separated Q/K and V sources.

        Args:
            qk_input: (B, N, C), features + positional embedding (drives Q1/Q2/K1/K2).
            v_input: (B, N, C), clean features without pos (drives V).

        Returns:
            (B, N, C) attention output.
        """
        b, n, c = qk_input.shape
        h, d = self.num_heads, self.head_dim
        lam = torch.sigmoid(self.lambda_logits)  # (h,)

        q1, q2, k1, k2 = self.qk_proj(qk_input).chunk(4, dim=-1)
        v = self.v_proj(v_input)

        # (B, N, C) -> (B, h, N, d)
        def heads(t):
            return t.view(b, n, h, d).permute(0, 2, 1, 3)

        q1, q2, k1, k2, v = heads(q1), heads(q2), heads(k1), heads(k2), heads(v)

        attn1 = torch.softmax(q1 @ k1.transpose(-2, -1) * self.scale, dim=-1)
        attn2 = torch.softmax(q2 @ k2.transpose(-2, -1) * self.scale, dim=-1)
        attn = attn1 - lam.view(1, h, 1, 1) * attn2
        attn = self.attn_drop(attn)
        out = attn @ v  # (B, h, N, d)

        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class AIFI_DiffAttn(nn.Module):
    """RT-DETR AIFI with differential attention.

    Interface is identical to the original AIFI: input and output are both
    ``(B, C, H, W)`` feature maps, so it is a drop-in replacement in any
    RT-DETR yaml that uses ``AIFI``.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True (AIFI default: False).
        lambda_init: Initial subtraction coefficient (warm start).
    """

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0,
                 act=nn.GELU(), normalize_before=False, lambda_init=0.2):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = DifferentialAttention(
            c1, num_heads=num_heads, dropout=dropout, lambda_init=lambda_init)
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
            raise ValueError(f'AIFI_DiffAttn expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        # pos enters Q/K only; the residual stream and V stay free of pos
        # (original AIFI semantics: q = k = src + pos, value = src).
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
