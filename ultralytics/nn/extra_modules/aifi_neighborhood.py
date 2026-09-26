# Gated local-global AIFI with a Neighborhood-Attention local branch.
#
# Reference (mechanism source):
#   Neighborhood Attention Transformer (NAT / DiNAT), CVPR 2023
#   Ali Hassani et al. https://github.com/ALI1000  (NAT repo + NATTEN kernels)
#
# Design decisions (lesson-driven):
#   1. The GLOBAL branch is the original full-precision MHSA, completely
#      untouched. No rank compression / sparsity / binarization of global
#      attention (the AgentV2/BA/BlockSparse failure mode).
#   2. The LOCAL branch is a parallel Neighborhood Attention: every token
#      attends to its k x k spatial neighborhood with EXACT, full-precision
#      per-pair softmax inside the window. This is the corrected version of
#      AgentV2's DWConv(V) idea: exact local attention instead of a randomly
#      initialized depthwise conv.
#   3. Zero-init gating: out = x + gamma_g * MHSA(x) + gamma_l * NA(x) with
#      gamma_g = 1.0 (original behavior preserved) and gamma_l = 0.0 (the new
#      local branch is silent at t=0 and warms up during training; the
#      validated warm-start pattern of this project).
#   4. Pure PyTorch implementation via F.unfold (no NATTEN / CUDA extension
#      needed; at P5 20x20x256 the overhead is negligible).
#   5. Positional embedding enters the global branch Q/K only; the local
#      branch is position-local by construction and needs no pos.
#
# Why small objects: VisDrone small objects occupy 0.5-2 tokens at P5 and
# appear in clusters (parking lots, crowds). Global softmax averages the
# neighborhood structure away; the gated local branch preserves per-pair
# exact content matching inside the window while the global branch keeps the
# full receptive field.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['NeighborhoodAttention', 'AIFI_Neighborhood']


class NeighborhoodAttention(nn.Module):
    """Sliding-window attention: each token attends to its k x k neighborhood.

    Full-precision softmax inside each window (no sampling, no compression).
    Implemented with F.unfold; works with dilation for a larger effective
    window at the same cost.

    Args:
        dim: Token dimension.
        num_heads: Number of attention heads.
        kernel_size: Window size (odd).
        dilation: Window dilation.
        dropout: Dropout probability.
    """

    def __init__(self, dim, num_heads=8, kernel_size=7, dilation=1, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        if kernel_size % 2 != 1:
            raise ValueError(f'kernel_size must be odd, got {kernel_size}')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.kv_proj = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in (self.q_proj, self.kv_proj, self.proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """Apply neighborhood attention on a (B, C, H, W) feature map."""
        b, c, h, w = x.shape
        nh, d = self.num_heads, self.head_dim
        k, dil = self.kernel_size, self.dilation
        pad = dil * (k // 2)
        n = h * w

        q = self.q_proj(x)                      # (B, C, H, W)
        kv = self.kv_proj(x)
        k_full = kv[:, :c]
        v_full = kv[:, c:]

        # (B, C, H, W) -> (B, C*k*k, L) -> (B, C, k*k, L) -> (B, L, k*k, C)
        k_win = F.unfold(k_full, kernel_size=k, dilation=dil, padding=pad, stride=1)
        v_win = F.unfold(v_full, kernel_size=k, dilation=dil, padding=pad, stride=1)
        k_win = k_win.view(b, c, k * k, n).permute(0, 3, 2, 1)
        v_win = v_win.view(b, c, k * k, n).permute(0, 3, 2, 1)

        # q: (B, L, C) -> (B, h, L, d)
        q_t = q.flatten(2).permute(0, 2, 1).view(b, n, nh, d).permute(0, 2, 1, 3)
        # k/v: (B, L, k*k, C) -> (B, h, L, k*k, d)
        k_t = k_win.view(b, n, k * k, nh, d).permute(0, 3, 1, 2, 4)
        v_t = v_win.view(b, n, k * k, nh, d).permute(0, 3, 1, 2, 4)

        # (B, h, L, k*k)
        attn = torch.einsum('bhld,bhlkd->bhlk', q_t, k_t) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum('bhlk,bhlkd->bhld', attn, v_t)  # (B, h, L, d)

        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        out = out.permute(0, 2, 1).view(b, c, h, w)
        return self.proj_drop(self.proj(out))


class AIFI_Neighborhood(nn.Module):
    """RT-DETR AIFI with an intact global MHSA plus a gated local NA branch.

    out = x + gamma_g * MHSA(x + pos, x) + gamma_l * NA(x)

    gamma_g is learnable, initialized to 1.0 (global branch behaves exactly
    like the original AIFI at t=0); gamma_l is learnable, initialized to 0.0
    (the local branch is silent at t=0 and warms up).

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads (shared by both branches).
        kernel_size: Local window size (odd, default 7).
        dilation: Local window dilation (default 1; 2 for DiNAT-style).
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True (AIFI default: False).
        local_gate_init: Initial value of gamma_l (default 0.0 = zero-init).
    """

    def __init__(self, c1, cm=2048, num_heads=8, kernel_size=7, dilation=1,
                 dropout=0.0, act=nn.GELU(), normalize_before=False,
                 local_gate_init=0.0):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before

        # Global branch: the original full-precision MHSA, untouched.
        self.global_attn = nn.MultiheadAttention(
            c1, num_heads, dropout=dropout, batch_first=True)
        # Local branch: exact sliding-window attention.
        self.local_attn = NeighborhoodAttention(
            c1, num_heads=num_heads, kernel_size=kernel_size,
            dilation=dilation, dropout=dropout)

        # Gates: gamma_g=1 (identity to original AIFI), gamma_l=0 (warm start).
        self.gamma_g = nn.Parameter(torch.tensor(1.0))
        self.gamma_l = nn.Parameter(torch.tensor(float(local_gate_init)))

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
            raise ValueError(f'AIFI_Neighborhood expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1)  # clean residual stream

        if self.normalize_before:
            src_norm = self.norm1(src)
            q = k = src_norm + pos
            global_out = self.global_attn(q, k, value=src_norm)[0]
            local_out = self.local_attn(x).flatten(2).permute(0, 2, 1)
            src = src + self.dropout1(
                self.gamma_g * global_out + self.gamma_l * local_out)
            src_norm = self.norm2(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src_norm)))))
        else:
            q = k = src + pos
            global_out = self.global_attn(q, k, value=src)[0]
            local_out = self.local_attn(x).flatten(2).permute(0, 2, 1)
            src = src + self.dropout1(
                self.gamma_g * global_out + self.gamma_l * local_out)
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()
