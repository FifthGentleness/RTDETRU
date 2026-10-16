# Block-sparse global attention V12 for RT-DETR AIFI.
#
# V12 = V1 + CDF adaptive block selection (from the original paper:
# "Block-sparse global attention for efficient multi-view geometry
# transformers", Wang et al. 2026, Sec 4.1).
#
# V1 gives every query block a FIXED budget of topk=4 key blocks. On aerial
# imagery this is misallocated: uniform sky/road blocks waste budget while
# object-dense blocks are starved. The paper's fix: per query block, sort
# key-block probabilities (softmax over block similarity), select blocks
# until the cumulative probability covers tau, with a minimum count
# k_min = ceil(B * (1 - rho)) and a hard cap to bound compute.
#
# Unchanged from V1 (winning properties preserved):
#   - routing from mean-pooled RAW input x (zero parameters, fixed scale)
#   - block geometry (1D rows), shared mask across heads
#   - same qkv/proj/FFN/norm stack
#
# NOTE: the routing softmax uses a temperature (route_temp, default 2.0).
# At T=1 the pooled-similarity distribution is so peaked (top-1 prob ~0.99)
# that the CDF threshold degenerates to the fixed k_min floor. T=2 yields
# adaptive counts of 5..12 per query block (mean ~5.7 at P5), which is the
# intended behavior. Calibrated empirically on AIFI-scale inputs.
#
# Paper choices NOT adopted (with reasons):
#   - routing from pooled PROJECTED Q/K: the paper is training-free (Q/K
#     already trained by dense attention); from-scratch this is the V4
#     feedback loop - falsified by our experiments.
#
# Selection semantics: adaptive set UNION {self} (V1 guarantees self via
# eye). Implemented with the compact gather attention (V11-style) so the
# key tensor is (npad x (cap+1)*bs) regardless of adaptive counts.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_BlockSparseV12']


class BlockSparseAttentionV12(nn.Module):
    """V1 attention with adaptive Top-K + CDF-threshold block selection."""

    def __init__(self, dim, num_heads=8, block_size=4, tau=0.9, rho=0.95,
                 cap=12, route_temp=2.0, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        if not 0.0 < tau <= 1.0:
            raise ValueError('tau must be in (0, 1]')
        if not 0.0 <= rho < 1.0:
            raise ValueError('rho must be in [0, 1)')
        if cap < 2:
            raise ValueError('cap must be >= 2')

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = int(block_size)
        self.tau = float(tau)
        self.rho = float(rho)
        self.cap = int(cap)
        self.route_temp = float(route_temp)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Selection statistics of the last forward (detached, for logging).
        self.last_selection = None

    def forward(self, x):
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        bs = self.block_size
        nb = math.ceil(n / bs)
        pad_len = nb * bs - n
        cap_eff = min(self.cap, nb)
        k_min = max(1, math.ceil(nb * (1 - self.rho) - 1e-9))

        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        npad = nb * bs

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, npad, h, d).permute(0, 2, 1, 3) for t in qkv]

        # --- Routing: identical to V1 (raw input, zero parameters) ---
        x_blk = x.view(b, nb, bs, c)
        q_blk = x_blk.mean(dim=2)
        block_logits = torch.matmul(q_blk, q_blk.transpose(-2, -1)) * self.scale

        block_valid = x_blk.abs().sum(dim=(2, 3)) > 0
        block_logits = block_logits.masked_fill(~block_valid[:, None, :], -1e9)

        # --- Adaptive Top-K + CDF selection ---
        order = block_logits.argsort(dim=-1, descending=True)      # (b, nb, nb)
        probs = (block_logits / self.route_temp).softmax(dim=-1)
        cum = probs.gather(-1, order).cumsum(-1)                   # (b, nb, nb)
        # blocks needed to reach tau; nb+1 if never reached (clamped later)
        count = (cum < self.tau).sum(-1) + 1                       # (b, nb)
        count = count.clamp(min=k_min, max=cap_eff)

        # rank of the self block in each row's sorted order
        self_rank = (order == torch.arange(nb, device=x.device)
                     .view(1, nb, 1)).int().argmax(-1)             # (b, nb)

        # --- Compact gather: top-cap ranked blocks + appended self column ---
        sel = order[:, :, :cap_eff]                                # (b, nb, cap)
        self_col = torch.arange(nb, device=x.device).view(1, nb, 1).expand(b, nb, 1)
        sel = torch.cat([sel, self_col], dim=-1)                   # (b, nb, cap+1)
        rank_mask = torch.arange(cap_eff, device=x.device).view(1, 1, cap_eff) \
            < count.unsqueeze(-1)                                  # (b, nb, cap)
        self_extra = self_rank >= count                            # (b, nb)
        block_sel = torch.cat(
            [rank_mask, self_extra.unsqueeze(-1)], dim=-1)         # (b, nb, cap+1)

        key_tok = sel.unsqueeze(-1) * bs + torch.arange(
            bs, device=x.device).view(1, 1, 1, bs)                 # (b, nb, (cap+1)*bs)
        key_tok = key_tok.view(b, nb, (cap_eff + 1) * bs)
        kt = key_tok.unsqueeze(1).expand(-1, h, -1, -1)
        kt = kt.reshape(b, h, nb * (cap_eff + 1) * bs, 1).expand(-1, -1, -1, d)
        k_g = k.gather(2, kt).view(b, h, nb, (cap_eff + 1) * bs, d)
        v_g = v.gather(2, kt).view(b, h, nb, (cap_eff + 1) * bs, d)

        q_r = q.view(b, h, nb, bs, d)
        logits = torch.matmul(q_r, k_g.transpose(-2, -1)) * self.scale
        logits = logits.view(b, h, nb, bs, (cap_eff + 1) * bs)

        tok_sel = block_sel.unsqueeze(-1).expand(-1, -1, -1, bs).reshape(
            b, nb, (cap_eff + 1) * bs)                             # (b, nb, keys)
        key_invalid = (key_tok >= n) | (~tok_sel)
        logits = logits.masked_fill(key_invalid[:, None, :, None, :], -1e9)

        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v_g)
        out = out.reshape(b, h, npad, d).permute(0, 2, 1, 3).reshape(b, npad, c)
        if pad_len:
            out = out[:, :n, :]

        self.last_selection = {
            'mean_blocks': count.float().mean().detach(),
            'min_blocks': int(count.min()),
            'max_blocks': int(count.max()),
            'density': (count.float().mean() / nb).detach(),
        }
        return self.proj_drop(self.proj(out))


class AIFI_BlockSparseV12(nn.Module):
    """AIFI with V12 adaptive block-sparse attention.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        block_size: Number of tokens per sparse block.
        tau: CDF probability coverage threshold.
        rho: sparsity ratio lower bound; selects >= ceil(B*(1-rho)) blocks.
        cap: maximum number of key blocks per query block.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, tau=0.9,
                 rho=0.95, cap=12, route_temp=2.0, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = BlockSparseAttentionV12(
            dim=c1, num_heads=num_heads, block_size=block_size,
            tau=tau, rho=rho, cap=cap, route_temp=route_temp, dropout=dropout)
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
                f'AIFI_BlockSparseV12 expected {self.c1} channels, got {c}')
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
