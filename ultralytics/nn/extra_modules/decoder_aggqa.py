"""
AGGQA: Alternating Group-Global Query Attention (RT-DETR decoder innovation).

Inspired by:
  - VGGT (CVPR 2025 Best Paper): alternating frame-local / global attention.

Queries are dynamically assigned to spatial cells (grid x grid) using their
current reference points. Dense self-attention happens only inside each cell
(where duplicate/overlapped competitors live), then per-cell mean
representatives exchange global context via a small full attention and are
broadcast back with a zero-init gated residual. Groups are re-computed every
decoder layer, so the neighborhood structure follows box refinement.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.transformer import DeformableTransformerDecoderLayer

from .decoder_innov_common import (_InnovSelfAttnLayer, _RTDETRDecoderInnov,
                                   DeformableTransformerDecoderInnov)

__all__ = ('AGGQAAttention', 'DecoderLayerAGGQA', 'RTDETRDecoderAGGQA')


class AGGQAAttention(nn.Module):
    """Group-local dense attention + global group-representative attention.

    Args:
        d_model (int): query feature dim.
        n_heads (int): attention heads.
        grid (int): spatial grouping resolution.
        dropout (float): dropout prob.
    """

    def __init__(self, d_model=256, n_heads=8, grid=8, dropout=0.0):
        super().__init__()
        self.grid, self.n_heads = grid, n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.rep_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ref_xy, pos=None):
        """
        Args:
            x (Tensor): [B, N, C] queries (matching part only).
            ref_xy (Tensor): [B, N, 2] reference centers.
            pos (Tensor, optional): [B, N, C] positional embedding.

        Returns:
            Tensor: [B, N, C] updated queries (residuals + norms included).
        """
        B, N, C = x.shape
        H, dh = self.n_heads, C // self.n_heads
        cell = (ref_xy.clamp(0.0, 1.0) * self.grid).long().clamp(max=self.grid - 1)
        gid = cell[..., 0] * self.grid + cell[..., 1]                  # [B,N]
        same = gid[:, :, None] == gid[:, None, :]                      # [B,N,N]

        inp = x if pos is None else x + pos
        q = self.q_proj(inp).view(B, N, H, dh).transpose(1, 2)
        k = self.k_proj(inp).view(B, N, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        att = att.masked_fill(~same[:, None], torch.finfo(att.dtype).min)
        out = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, N, C)
        x = self.norm1(x + self.drop(self.out_proj(out)))

        G = self.grid * self.grid
        oh = F.one_hot(gid, G).to(x.dtype)                             # [B,N,G]
        cnt = oh.sum(1)                                                # [B,G]
        reps = (oh.transpose(1, 2) @ x) / cnt.clamp(min=1).unsqueeze(-1)
        kpm = cnt <= 0
        r = self.rep_attn(reps, reps, reps, key_padding_mask=kpm, need_weights=False)[0]
        r = torch.where(kpm.unsqueeze(-1), torch.zeros_like(r), r)
        x = self.norm2(x + torch.tanh(self.alpha) * (oh @ r))
        return x


class DecoderLayerAGGQA(_InnovSelfAttnLayer):
    """Self-attention replaced by Alternating Group-Global Query Attention."""

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(),
                 n_levels=4, n_points=4, group_grid=8):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.innov = AGGQAAttention(d_model, n_heads, group_grid, dropout)

    def _innov_attn(self, mt, ref_xy, pos, boxes):
        return self.innov(mt, ref_xy, pos)


class RTDETRDecoderAGGQA(_RTDETRDecoderInnov):
    """RT-DETR decoder with self-attention replaced by alternating group-global
    query attention (VGGT-style frame/global alternation).

    Extra yaml args (after the standard RTDETRDecoder args):
        group_grid (int, default 8): grouping grid resolution.
    """

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp, eval_idx,
                         group_grid=8, **_):
        layer = DecoderLayerAGGQA(hd, nh, d_ffn, dropout, act, self.nl, ndp, group_grid)
        return DeformableTransformerDecoderInnov(hd, layer, ndl, eval_idx)
