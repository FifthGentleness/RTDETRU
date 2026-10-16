# Block-sparse global attention V14 for RT-DETR AIFI.
#
# V14 = V12 (V1 + CDF adaptive selection) reimplemented on V1's DENSE
# MASKED attention path - no gather. The full npad x npad logit matrix is
# computed and non-selected entries are masked with -1e4 before a full
# softmax, exactly like aifi_block_sparse.py (V1).
#
# Selection logic is identical to V12:
#   probs = softmax(block_logits / route_temp)
#   per query block: pick ranked blocks until CDF >= tau,
#   at least k_min = ceil(B*(1-rho)) blocks, at most cap blocks,
#   self block always selected (V1's eye guarantee).
#
# Differences to V12 are IMPLEMENTATION-ONLY (verified element-equivalent):
#   - full masked softmax instead of subset softmax over gathered keys
#   - mask built by scattering rank masks through argsort order
#
# Memory: full npad x npad logits - fine at P5 (400 tokens, ~5MB/image),
# NOT suitable for P4 (1600 tokens). Use aifi_block_sparse_v12.py for P4.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_BlockSparseV14']


class BlockSparseAttentionV14(nn.Module):
    """CDF adaptive block-sparse attention on V1's dense masked path."""

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

        # --- Adaptive Top-K + CDF selection (identical to V12) ---
        blT = block_logits / self.route_temp
        order = blT.argsort(dim=-1, descending=True)               # (b, nb, nb)
        cum = blT.softmax(dim=-1).gather(-1, order).cumsum(-1)
        count = ((cum < self.tau).sum(-1) + 1).clamp(min=k_min, max=cap_eff)

        # rank-space mask -> block space via scatter through argsort order
        rank = torch.arange(nb, device=x.device).view(1, 1, nb)
        rank_sel = (rank < count.unsqueeze(-1))                    # (b, nb, nb)
        block_mask = torch.zeros(b, nb, nb, dtype=torch.bool, device=x.device)
        block_mask.scatter_(-1, order, rank_sel)
        # self block always selected (V1's eye guarantee)
        eye = torch.eye(nb, dtype=torch.bool, device=x.device)
        block_mask = block_mask | eye.unsqueeze(0)

        # --- V1 dense masked attention path ---
        token_mask = block_mask.repeat_interleave(bs, dim=1)
        token_mask = token_mask.repeat_interleave(bs, dim=2)
        token_mask = token_mask[:, :npad, :npad]
        if pad_len:
            valid = torch.zeros(b, npad, dtype=torch.bool, device=x.device)
            valid[:, :n] = True
            token_mask = token_mask & valid[:, None, :]

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(~token_mask.unsqueeze(1), -1e4)
        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, npad, c)
        if pad_len:
            out = out[:, :n, :]

        self.last_selection = {
            'mean_blocks': count.float().mean().detach(),
            'min_blocks': int(count.min()),
            'max_blocks': int(count.max()),
            'density': (count.float().mean() / nb).detach(),
        }
        return self.proj_drop(self.proj(out))


class AIFI_BlockSparseV14(nn.Module):
    """AIFI with V14 CDF adaptive block-sparse attention (dense path).

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        block_size: Number of tokens per sparse block.
        tau: CDF probability coverage threshold.
        rho: sparsity ratio lower bound; selects >= ceil(B*(1-rho)) blocks.
        cap: maximum number of key blocks per query block.
        route_temp: softmax temperature for the routing distribution
            (T=1 degenerates to the k_min floor; T=2 calibrated).
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
        self.attn = BlockSparseAttentionV14(
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
                f'AIFI_BlockSparseV14 expected {self.c1} channels, got {c}')
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
