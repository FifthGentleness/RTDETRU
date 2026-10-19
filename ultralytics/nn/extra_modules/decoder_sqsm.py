"""
SQSM: Spatially-Ordered Query State-space Modeling (RT-DETR decoder innovation).

Inspired by:
  - Vision Mamba (ICML 2024): bidirectional selective state-space modeling.
  - ZigMa (ECCV 2024): space-filling-curve scan order preserving 2D locality.

Queries are sorted by the Z-order (Morton) code of their reference-point
centers so that 2D-neighboring queries become 1D neighbors, then scanned by an
input-dependent bidirectional SSM. A neighborhood differential gate supplies
an explicit 'how do I differ from my spatial predecessors' signal which helps
duplicate suppression in dense aerial scenes.

Pure-PyTorch fallback (no mamba_ssm CUDA kernel required).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.transformer import DeformableTransformerDecoderLayer

from .decoder_innov_common import (DropPath, _chunked_selective_scan, _morton_codes,
                                   _RTDETRDecoderInnov, DeformableTransformerDecoderInnov)

__all__ = ('SQSMBlock', 'DecoderLayerSQSM', 'RTDETRDecoderSQSM')


class SQSMBlock(nn.Module):
    """Bidirectional selective state-space modeling over spatially-ordered queries.

    Args:
        d_model (int): query feature dim.
        d_state (int): SSM state dimension.
        grid (int): morton quantization grid.
        drop_path (float): stochastic depth prob.
    """

    def __init__(self, d_model=256, d_state=16, grid=16, drop_path=0.0):
        super().__init__()
        self.d_state, self.grid = d_state, grid
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.bc_proj = nn.Linear(d_model, 2 * d_state)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))
        self.gate = nn.Linear(2 * d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop_path = DropPath(drop_path)

    def _ssm(self, x):
        delta = F.softplus(self.dt_proj(x))
        Bm, Cm = self.bc_proj(x).chunk(2, -1)
        A = -torch.exp(self.A_log.float())
        y = _chunked_selective_scan(x, delta, A, Bm, Cm)
        return y + x * self.D.float()

    def forward(self, q, ref_xy):
        """
        Args:
            q (Tensor): [B, N, C] query features (matching part only).
            ref_xy (Tensor): [B, N, 2] reference point centers in [0, 1].

        Returns:
            Tensor: [B, N, C] updated queries (residual included).
        """
        B, N, C = q.shape
        idx = _morton_codes(ref_xy, self.grid).argsort(dim=1, stable=True)
        idx_inv = idx.argsort(dim=1, stable=True)
        x, z = self.in_proj(self.norm(q)).chunk(2, -1)
        xs = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, C)).float()
        y = (self._ssm(xs) + self._ssm(xs.flip(1)).flip(1)).to(q.dtype)
        y = y * F.silu(z)
        g = torch.sigmoid(self.gate(torch.cat([y, x], dim=-1)))
        y = y + g * (y - x)                                    # neighborhood diff gate
        y = torch.gather(y, 1, idx_inv.unsqueeze(-1).expand(-1, -1, C))
        return q + self.drop_path(self.out_proj(y))


class DecoderLayerSQSM(DeformableTransformerDecoderLayer):
    """Standard decoder layer + SQSM inserted after self-attention (matching
    queries only; dn queries keep the original masked path)."""

    INNOV_KWARGS = True

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(),
                 n_levels=4, n_points=4, d_state=16, sqsm_grid=16):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.sqsm = SQSMBlock(d_model, d_state, sqsm_grid)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None,
                attn_mask=None, query_pos=None, dn_num=0, dec_boxes=None):
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed = self.norm1(embed + self.dropout1(tgt))
        if dn_num > 0:
            mt = self.sqsm(embed[:, dn_num:], refer_bbox[:, dn_num:, :2])
            embed = torch.cat([embed[:, :dn_num], mt], dim=1)
        else:
            embed = self.sqsm(embed, refer_bbox[..., :2])
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2),
                              feats, shapes, padding_mask)
        embed = self.norm2(embed + self.dropout2(tgt))
        return self.forward_ffn(embed)


class RTDETRDecoderSQSM(_RTDETRDecoderInnov):
    """RT-DETR decoder + SQSM (spatially-ordered bidirectional SSM) after
    self-attention in every decoder layer.

    Extra yaml args (after the standard RTDETRDecoder args):
        d_state (int, default 16): SSM state dim.
        sqsm_grid (int, default 16): morton grid resolution.
    """

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp, eval_idx,
                         d_state=16, sqsm_grid=16, **_):
        layer = DecoderLayerSQSM(hd, nh, d_ffn, dropout, act, self.nl, ndp,
                                 d_state, sqsm_grid)
        return DeformableTransformerDecoderInnov(hd, layer, ndl, eval_idx)
