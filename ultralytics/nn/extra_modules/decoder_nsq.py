"""
NSQ: Native-Sparse Query Attention (RT-DETR decoder innovation).

Inspired by:
  - Native Sparse Attention (ACL 2025 Best Paper): natively trainable sparse
    attention with compression / selection / window branches + gated fusion.

Three branches:
  - select:   per-query attention over the top-n competitor queries ranked by
              predicted-box IoU (the pairs that actually need dedup);
  - compress: attention over grid-pooled regional summary tokens (global
              saturation context);
  - local:    windowed attention along the Morton scan order (spatial
              neighborhood).

Branches are fused with a learned per-query softmax gate.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.transformer import DeformableTransformerDecoderLayer

from .decoder_innov_common import (_InnovSelfAttnLayer, _morton_codes, _pairwise_iou,
                                   _RTDETRDecoderInnov, DeformableTransformerDecoderInnov)

__all__ = ('NSQAttention', 'DecoderLayerNSQ', 'RTDETRDecoderNSQ')


class NSQAttention(nn.Module):
    """Three-branch sparse query attention with learned per-query gating.

    Args:
        d_model (int): query feature dim.
        n_heads (int): attention heads.
        topn (int): competitors per query in the select branch.
        n_grid (int): pooling grid for the compress branch.
        window (int): window size along morton order for the local branch.
        dropout (float): dropout prob.
    """

    def __init__(self, d_model=256, n_heads=8, topn=16, n_grid=4, window=12, dropout=0.0):
        super().__init__()
        self.n_heads, self.topn, self.n_grid, self.window = n_heads, topn, n_grid, window
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, 3)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, t, B, N):
        return t.view(B, N, self.n_heads, -1).transpose(1, 2)          # [B,H,N,dh]

    def forward(self, x, ref_xy, pos=None, boxes=None):
        """
        Args:
            x (Tensor): [B, N, C] queries (matching part only).
            ref_xy (Tensor): [B, N, 2] reference centers.
            pos (Tensor, optional): [B, N, C] positional embedding.
            boxes (Tensor, optional): [B, N, 4] current predicted boxes
                (sigmoid, cxcywh) for competitor ranking.

        Returns:
            Tensor: [B, N, C] updated queries (residual + norm included).
        """
        B, N, C = x.shape
        H, dh = self.n_heads, C // self.n_heads
        inp = x if pos is None else x + pos
        q, k, v = self.q_proj(inp), self.k_proj(inp), self.v_proj(x)

        # ---- branch 1: competitor selection --------------------------
        if boxes is None:
            q1, k1, v1 = self._split(q, B, N), self._split(k, B, N), self._split(v, B, N)
            o1 = ((q1 @ k1.transpose(-1, -2)) / math.sqrt(dh)).softmax(-1) @ v1
            o1 = o1.transpose(1, 2).reshape(B, N, C)
        else:
            tn = min(self.topn, N)
            iou = _pairwise_iou(boxes)
            iou.diagonal(dim1=1, dim2=2).fill_(1.0)
            idx = iou.topk(tn, dim=-1).indices                          # [B,N,tn]
            eidx = idx[:, None, :, :, None].expand(-1, H, -1, -1, dh)   # [B,H,N,tn,dh]
            k5 = self._split(k, B, N).unsqueeze(3).expand(-1, -1, -1, tn, -1)
            v5 = self._split(v, B, N).unsqueeze(3).expand(-1, -1, -1, tn, -1)
            kk = torch.gather(k5, 2, eidx)                              # [B,H,N,tn,dh]
            vv = torch.gather(v5, 2, eidx)
            att = (self._split(q, B, N).unsqueeze(3) @ kk.transpose(-1, -2)) / math.sqrt(dh)
            o1 = (att.softmax(-1) @ vv).squeeze(3).transpose(1, 2).reshape(B, N, C)

        # ---- branch 2: compressed regional tokens --------------------
        G = self.n_grid * self.n_grid
        cell = (ref_xy.clamp(0.0, 1.0) * self.n_grid).long().clamp(max=self.n_grid - 1)
        gid = cell[..., 0] * self.n_grid + cell[..., 1]
        oh = F.one_hot(gid, G).to(x.dtype)
        cnt = oh.sum(1)
        tk = (oh.transpose(1, 2) @ k) / cnt.clamp(min=1).unsqueeze(-1)  # [B,G,C]
        tv = (oh.transpose(1, 2) @ v) / cnt.clamp(min=1).unsqueeze(-1)
        q2 = self._split(q, B, N)
        k2 = tk.view(B, G, H, dh).transpose(1, 2)
        v2 = tv.view(B, G, H, dh).transpose(1, 2)
        att2 = (q2 @ k2.transpose(-1, -2)) / math.sqrt(dh)
        att2 = att2.masked_fill((cnt <= 0)[:, None, None], torch.finfo(att2.dtype).min)
        o2 = (att2.softmax(-1) @ v2).transpose(1, 2).reshape(B, N, C)

        # ---- branch 3: morton-ordered local window -------------------
        w = min(self.window, N)
        srt = _morton_codes(ref_xy, 16).argsort(dim=1, stable=True)
        inv = srt.argsort(dim=1, stable=True)
        pad = (-N) % w

        def reorder(t):
            t = torch.gather(t, 1, srt.unsqueeze(-1).expand(-1, -1, C))
            if pad:
                t = F.pad(t, (0, 0, 0, pad))
            return t.view(B, -1, w, H, dh).transpose(2, 3)              # [B,nW,H,w,dh]

        qw, kw, vw = reorder(q), reorder(k), reorder(v)
        att3 = (qw @ kw.transpose(-1, -2)) / math.sqrt(dh)
        o3 = (att3.softmax(-1) @ vw).transpose(2, 3).reshape(B, -1, C)
        if pad:
            o3 = o3[:, :N]
        o3 = torch.gather(o3, 1, inv.unsqueeze(-1).expand(-1, -1, C))

        # ---- gated fusion --------------------------------------------
        g = self.gate(x).softmax(-1)
        out = g[..., 0:1] * o1 + g[..., 1:2] * o2 + g[..., 2:3] * o3
        return self.norm(x + self.drop(self.out_proj(out)))


class DecoderLayerNSQ(_InnovSelfAttnLayer):
    """Self-attention replaced by Native-Sparse Query Attention."""

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(),
                 n_levels=4, n_points=4, topn=16, nsq_grid=4, nsq_window=12):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.innov = NSQAttention(d_model, n_heads, topn, nsq_grid, nsq_window, dropout)

    def _innov_attn(self, mt, ref_xy, pos, boxes):
        return self.innov(mt, ref_xy, pos, boxes)


class RTDETRDecoderNSQ(_RTDETRDecoderInnov):
    """RT-DETR decoder with self-attention replaced by native-sparse query
    attention (competitor-select / region-compress / morton-local + gate).

    Extra yaml args (after the standard RTDETRDecoder args):
        topn (int, default 16): competitors per query.
        nsq_grid (int, default 4): compress-branch pooling grid.
        nsq_window (int, default 12): local-branch window along morton order.
    """

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp, eval_idx,
                         topn=16, nsq_grid=4, nsq_window=12, **_):
        layer = DecoderLayerNSQ(hd, nh, d_ffn, dropout, act, self.nl, ndp,
                                topn, nsq_grid, nsq_window)
        return DeformableTransformerDecoderInnov(hd, layer, ndl, eval_idx)
