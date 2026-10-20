"""
LBCA: Latent-Bottleneck Cross-Attention with sparse residual correction
(RT-DETR decoder innovation).

Inspired by:
  - Multi-head Latent Attention (MLA, DeepSeek) and its systematic migration
    TransMLA (NeurIPS 2025): compress Keys/Values into a low-rank latent
    space so attention interacts with a compact latent set instead of the
    full token sequence.

Mechanism:
  - Content-adaptive soft compression: all multi-scale tokens are softly
    assigned to n_latent learned slots (slot-assignment softmax over tokens)
    and aggregated into latent tokens C = A @ V_down(feats).
  - Queries attend the latent tokens only (O(N * L) instead of O(N * T)),
    which fuses cross-scale context in the latent space.
  - Low-rank compression is biased against isolated weak evidence (small
    objects), so tokens with the largest compression residual
    ||v - A^T C|| bypass the bottleneck through a zero-init gated sparse
    attention path over the top-M residual tokens.

Cross-attention of matching queries is replaced; dn queries keep the original
MSDeformAttn path and self-attention is the baseline masked attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.transformer import DeformableTransformerDecoderLayer

from .decoder_innov_common import (_InnovCrossAttnLayer, _RTDETRDecoderInnov,
                                   DeformableTransformerDecoderInnov)

__all__ = ('LatentBottleneckCross', 'DecoderLayerLBCA', 'RTDETRDecoderLBCA')


class LatentBottleneckCross(nn.Module):
    """Low-rank latent cross-attention with a sparse residual bypass."""

    def __init__(self, d_model=256, n_latent=256, n_heads=8, top_sparse=192, dropout=0.):
        super().__init__()
        self.n_latent = n_latent
        self.top_sparse = top_sparse

        self.norm_x = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.v_down = nn.Linear(d_model, d_model)
        self.slot_logits = nn.Linear(d_model, n_latent)   # soft assignment over tokens
        self.attn_lat = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                              batch_first=True)
        self.attn_sparse = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                 batch_first=True)
        self.res_gate = nn.Parameter(torch.zeros(1))     # zero-init sparse bypass
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, ref_boxes, feats, shapes, padding_mask=None):
        """Cross-read for matching queries.

        Args:
            q (Tensor):    [B, N, C] query content + positional embedding.
            ref_boxes:     [B, N, 4] cxcywh reference boxes in [0, 1].
            feats (Tensor):[B, T, C] flattened multi-scale features.
            shapes (List): [(h, w), ...] per-level shapes.

        Returns:
            Tensor: [B, N, C] cross-attention output (pre-residual).
        """
        B, N, C = q.shape
        T = feats.shape[1]
        v = self.v_down(self.norm_x(feats))                       # [B, T, C]
        # content-adaptive soft compression into latent slots
        A = self.slot_logits(v).softmax(dim=1).transpose(1, 2)    # [B, L, T] (softmax over tokens)
        lat = A @ v                                               # [B, L, C]
        qn = self.norm_q(q)
        ln = self.norm_x(lat)
        out, _ = self.attn_lat(qn, ln, ln, need_weights=False)

        # sparse residual bypass: tokens the latent bottleneck cannot reconstruct
        M = min(self.top_sparse, T)
        res = v - A.transpose(1, 2) @ lat                         # [B, T, C]
        energy = res.pow(2).sum(-1)                               # [B, T]
        idx = energy.topk(M, dim=1).indices                       # [B, M]
        sparse = torch.gather(feats, 1, idx.unsqueeze(-1).expand(-1, -1, C))  # [B, M, C]
        sn = self.norm_x(sparse)
        out_sp, _ = self.attn_sparse(qn, sn, sn, need_weights=False)

        return self.out_proj(out + torch.tanh(self.res_gate) * out_sp)


class DecoderLayerLBCA(_InnovCrossAttnLayer):
    """Baseline decoder layer whose matching-query cross-attention is replaced
    by the LBCA latent-bottleneck read."""

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(),
                 n_levels=4, n_points=4, n_latent=256, top_sparse=192):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.lbca = LatentBottleneckCross(d_model, n_latent, top_sparse=top_sparse)

    def _innov_cross(self, q, embed, ref_boxes, feats, shapes, padding_mask=None):
        return self.lbca(q, ref_boxes, feats, shapes, padding_mask)


class RTDETRDecoderLBCA(_RTDETRDecoderInnov):
    """RT-DETR decoder + LBCA (latent-bottleneck cross-attention).

    Extra yaml args (after the standard RTDETRDecoder args):
        n_latent (int, default 256): latent slot count.
        top_sparse (int, default 192): residual bypass token count.
    """

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp, eval_idx,
                         n_latent=256, top_sparse=192, **_):
        layer = DecoderLayerLBCA(hd, nh, d_ffn, dropout, act, self.nl, ndp,
                                 n_latent, top_sparse)
        return DeformableTransformerDecoderInnov(hd, layer, ndl, eval_idx)
