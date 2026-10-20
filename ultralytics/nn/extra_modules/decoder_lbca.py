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
  - The latent compression and the sparse evidence selection are
    input-dependent only, so they are computed ONCE per image and shared by
    every decoder layer; each layer keeps its own attention readout heads so
    every stage of box refinement queries the shared evidence differently.

Cross-attention of matching queries is replaced; dn queries keep the original
MSDeformAttn path and self-attention is the baseline masked attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.transformer import DeformableTransformerDecoderLayer
from ultralytics.nn.modules.utils import inverse_sigmoid

from .decoder_innov_common import (_InnovCrossAttnLayer, _RTDETRDecoderInnov,
                                   DeformableTransformerDecoderInnov)

__all__ = ('LatentEvidenceBuilder', 'LBCAReadout', 'DecoderLayerLBCA',
           'DeformableTransformerDecoderLBCA', 'RTDETRDecoderLBCA')


class LatentEvidenceBuilder(nn.Module):
    """Shared per-image latent compression + sparse residual evidence selection."""

    def __init__(self, d_model=256, n_latent=256, top_sparse=192):
        super().__init__()
        self.n_latent = n_latent
        self.top_sparse = top_sparse
        self.norm_x = nn.LayerNorm(d_model)
        self.v_down = nn.Linear(d_model, d_model)
        self.slot_logits = nn.Linear(d_model, n_latent)   # soft assignment over tokens

    def build_states(self, feats, padding_mask=None):
        """Compress the multi-scale tokens once per image.

        Args:
            feats (Tensor): [B, T, C] flattened multi-scale features.

        Returns:
            dict with latent tokens 'lat' [B, L, C], sparse bypass tokens
            'sparse' [B, M, C] (raw features) and their normed version.
        """
        v = self.v_down(self.norm_x(feats))                       # [B, T, C]
        if padding_mask is not None:
            v = v.masked_fill(padding_mask[..., None], 0.0)
        # content-adaptive soft compression into latent slots
        A = self.slot_logits(v).softmax(dim=1).transpose(1, 2)    # [B, L, T] (over tokens)
        lat = A @ v                                               # [B, L, C]
        # sparse residual bypass: tokens the latent bottleneck cannot reconstruct
        M = min(self.top_sparse, feats.shape[1])
        res = v - A.transpose(1, 2) @ lat                         # [B, T, C]
        energy = res.pow(2).sum(-1)                               # [B, T]
        idx = energy.topk(M, dim=1).indices                       # [B, M]
        sparse = torch.gather(feats, 1, idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
        return dict(lat=lat, sparse=sparse)


class LBCAReadout(nn.Module):
    """Per-layer readout head over the shared latent evidence."""

    def __init__(self, d_model=256, n_heads=8, dropout=0.):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn_lat = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                              batch_first=True)
        self.attn_sparse = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                 batch_first=True)
        self.res_gate = nn.Parameter(torch.zeros(1))     # zero-init sparse bypass
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, states):
        """Cross-read for matching queries.

        Args:
            q (Tensor):    [B, N, C] query content + positional embedding.
            states (dict): shared evidence from LatentEvidenceBuilder.

        Returns:
            Tensor: [B, N, C] cross-attention output (pre-residual).
        """
        qn = self.norm_q(q)
        ln = self.norm_kv(states['lat'])
        out, _ = self.attn_lat(qn, ln, ln, need_weights=False)
        sn = self.norm_kv(states['sparse'])
        out_sp, _ = self.attn_sparse(qn, sn, sn, need_weights=False)
        return self.out_proj(out + torch.tanh(self.res_gate) * out_sp)


class DecoderLayerLBCA(_InnovCrossAttnLayer):
    """Baseline decoder layer whose matching-query cross-attention is replaced
    by a readout over the shared LBCA latent evidence."""

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(),
                 n_levels=4, n_points=4):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.readout = LBCAReadout(d_model, n_heads, dropout)

    def _innov_cross(self, q, embed, ref_boxes, feats, shapes, padding_mask=None,
                     shared_states=None):
        assert shared_states is not None, 'LBCA layer requires shared evidence states'
        return self.readout(q, shared_states)


class DeformableTransformerDecoderLBCA(DeformableTransformerDecoderInnov):
    """Decoder wrapper that compresses the features ONCE and feeds the shared
    latent evidence to every decoder layer."""

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1, builder=None):
        super().__init__(hidden_dim, decoder_layer, num_layers, eval_idx)
        self.builder = builder

    def forward(self, embed, refer_bbox, feats, shapes, bbox_head, score_head,
                pos_mlp, attn_mask=None, padding_mask=None, dn_num=0):
        states = self.builder.build_states(feats, padding_mask)
        output = embed
        dec_bboxes, dec_cls = [], []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            extra = dict(dn_num=dn_num, dec_boxes=refer_bbox, shared_states=states) \
                if getattr(layer, 'INNOV_KWARGS', False) else {}
            output = layer(output, refer_bbox, feats, shapes, padding_mask,
                           attn_mask, pos_mlp(refer_bbox), **extra)

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            if i == self.num_layers - 1:
                break
            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class RTDETRDecoderLBCA(_RTDETRDecoderInnov):
    """RT-DETR decoder + LBCA (latent-bottleneck cross-attention, shared states).

    Extra yaml args (after the standard RTDETRDecoder args):
        n_latent (int, default 256): latent slot count.
        top_sparse (int, default 192): residual bypass token count.
    """

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp, eval_idx,
                         n_latent=256, top_sparse=192, **_):
        layer = DecoderLayerLBCA(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        builder = LatentEvidenceBuilder(hd, n_latent, top_sparse)
        return DeformableTransformerDecoderLBCA(hd, layer, ndl, eval_idx, builder=builder)
