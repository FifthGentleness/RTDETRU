# SS-AIFI: Selective-Scan 2D global modeling for AIFI (AIFI_SelectiveScan).
#
# Problem addressed: the global modeling available to the network is either
# softmax attention (AIFI) or the STATIC focusing of FLatten (DSAWACGAv5).
# Neither modulates information flow PER TOKEN according to content.
#
# Mechanism (VMamba: Visual State Space Model, NeurIPS 2024; LocalMamba,
# ECCV 2024): a 2D Selective Scan (SS2D) replaces softmax attention - tokens
# are flattened along 4 directions (row / reverse-row / column /
# reverse-column) and processed by a selective SSM whose decay and write
# gates (Delta / B / C) are INPUT-DEPENDENT, i.e. every token decides how
# much global context to absorb. Linear complexity O(N).
#
# Implementation note: this is a faithful-in-spirit PURE-PYTORCH version -
# real-valued diagonal SSM with a chunked parallel scan (chunk-stable two-
# level cumsum, no custom CUDA kernel required). It keeps VMamba's
# selectivity while remaining portable.
#
# Zero-overlap: no wavelet/frequency, no conv branches, no kernel attention.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_SelectiveScan']


class SelectiveScan2D(nn.Module):
    """4-direction 2D selective scan token mixer (pure PyTorch)."""

    def __init__(self, dim, d_state=16, dt_rank=16, chunk_len=32):
        super().__init__()
        self.dim = dim
        self.d_state = int(d_state)
        self.dt_rank = int(dt_rank)
        self.chunk_len = int(chunk_len)

        self.in_proj = nn.Linear(dim, dim * 2)             # x, z(gate)
        self.x_proj = nn.Linear(dim, dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(dt_rank, d_state)
        nn.init.constant_(self.dt_proj.bias, -4.0)         # small init deltas

        # Per-direction log-decay init A = 1..N (VMamba convention).
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .repeat(4, 1))                                 # (4, N)
        self.D = nn.Parameter(torch.zeros(4, dim))         # skip, 0-init

        self.out_proj = nn.Linear(dim, dim)

    def _scan(self, u, log_a):
        """Chunked parallel scan. u: (b, n, c, N); log_a: (b, n, 1, N)."""
        b, n, cc, n_state = u.shape
        h = u.new_zeros(b, cc, n_state)
        ys = []
        step = min(self.chunk_len, n)
        for s in range(0, n, step):
            uc = u[:, s:s + step]
            L = log_a[:, s:s + step].cumsum(dim=1)         # (b, lc, 1, N)
            S = (torch.exp(-L) * uc).cumsum(dim=1)         # (b, lc, c, N)
            hc = torch.exp(L) * (h.unsqueeze(1) + S)       # (b, lc, c, N)
            ys.append(hc)
            h = hc[:, -1]
        return torch.cat(ys, dim=1)                        # (b, n, c, N)

    def _forward_direction(self, seq, d_idx, B, C, dt_raw):
        b, n, c = seq.shape
        N = self.d_state
        dt = F.softplus(self.dt_proj(dt_raw))              # (b, n, N), > 0
        A = torch.exp(self.A_log[d_idx]).view(1, 1, N)
        log_a = torch.log(torch.exp(-dt * A)).clamp(-1.0, 0.0).unsqueeze(2)
        u = seq.unsqueeze(-1) * B.unsqueeze(2)             # (b, n, c, N)
        h = self._scan(u, log_a)                           # (b, n, c, N)
        y = (C.unsqueeze(2) * h).sum(-1) + self.D[d_idx] * seq  # (b, n, c)
        return y


    def forward_tokens(self, x_bchw):
        """x_bchw: (b, c, h, w). Returns tokens (b, n, c)."""
        b, c, hh, ww = x_bchw.shape
        xz = self.in_proj(x_bchw.flatten(2).permute(0, 2, 1))   # (b, n, 2c)
        xv, z = xz.chunk(2, dim=-1)
        dt_raw, B, C = self.x_proj(xv).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)

        grid = xv.transpose(1, 2).reshape(b, c, hh, ww)
        outs = 0.0
        seqs = []
        row = grid.flatten(2).transpose(1, 2)                  # (b, n, c)
        col = grid.permute(0, 3, 2, 1).reshape(b, hh * ww, c)
        seqs = [row, torch.flip(row, dims=[1]),
                col, torch.flip(col, dims=[1])]
        for d_idx, seq in enumerate(seqs):
            y = self._forward_direction(seq, d_idx, B, C, dt_raw)
            if d_idx in (1, 3):
                y = torch.flip(y, dims=[1])
            outs = outs + y
        outs = outs / 4.0

        out = outs * F.silu(z)
        out = self.out_proj(out)                               # (b, n, c)
        return out


class AIFI_SelectiveScan(nn.Module):
    """AIFI with a 4-direction selective-scan mixer (VMamba-style).

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        d_state: SSM state dimension per direction.
        dt_rank: Rank of the input-dependent step-size Delta.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, d_state=16, dt_rank=16, dropout=0.0,
                 act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = SelectiveScan2D(
            dim=c1, d_state=d_state, dt_rank=dt_rank)
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
                f'AIFI_SelectiveScan expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos              # (b, n, c)

        if self.normalize_before:
            src = src + self.dropout1(
                self.attn.forward_tokens(self.norm1(src).permute(0, 2, 1)
                                          .view(b, c, h, w)))
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = src + self.dropout1(
                self.attn.forward_tokens(src.permute(0, 2, 1).view(b, c, h, w)))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        return (f'c1={self.c1}, d_state={self.attn.d_state}, '
                f'dt_rank={self.attn.dt_rank}')
