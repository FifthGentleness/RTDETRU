"""
Shared utilities / base classes for the 5 RT-DETR decoder innovations.

Each innovation lives in its own module:
    decoder_sqsm.py   - SQSM:  Spatially-Ordered Query State-space Modeling
    decoder_aggqa.py  - AGGQA: Alternating Group-Global Query Attention
    decoder_cldm.py   - CLDM:  Cross-Layer Detection Memory
    decoder_nsq.py    - NSQ:   Native-Sparse Query Attention
    decoder_qga.py    - QGA:   Query Gram Anchoring (training-only loss)

All designs keep full compatibility with denoising (DN) training: dn queries
are always processed by the original masked self-attention path, innovations
only act on the matching-query part.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.transformer import (DeformableTransformerDecoder,
                                                DeformableTransformerDecoderLayer)
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.modules.utils import inverse_sigmoid

__all__ = ('DeformableTransformerDecoderInnov',)


# ---------------------------------------------------------------------------
# common utils
# ---------------------------------------------------------------------------

def _morton_codes(centers, grid=16):
    """Compute Z-order (Morton) codes of 2D centers.

    Args:
        centers (Tensor): [B, N, 2], normalized coordinates in [0, 1].
        grid (int): quantization resolution per axis.

    Returns:
        Tensor: [B, N] int64 morton codes (spatially-close points tend to be
        close in code order).
    """
    q = (centers.clamp(0.0, 1.0) * grid).long().clamp(max=grid - 1)
    x, y = q[..., 0], q[..., 1]
    nbits = max(1, int(math.log2(grid)))
    code = torch.zeros_like(x)
    for i in range(nbits):
        code |= ((x >> i) & 1) << (2 * i + 1)
        code |= ((y >> i) & 1) << (2 * i)
    return code


def _pairwise_iou(boxes):
    """Pairwise IoU of cxcywh boxes in [0, 1].

    Args:
        boxes (Tensor): [B, N, 4] in cxcywh format.

    Returns:
        Tensor: [B, N, N] IoU matrix.
    """
    xy, wh = boxes[..., :2], boxes[..., 2:].clamp(min=1e-4)
    tl, br = xy - wh / 2, xy + wh / 2
    area = (wh[..., 0] * wh[..., 1])
    lt = torch.maximum(tl[:, :, None], tl[:, None, :])
    rb = torch.minimum(br[:, :, None], br[:, None, :])
    inter = (rb - lt).clamp(min=0)
    inter = inter[..., 0] * inter[..., 1]
    union = area[:, :, None] + area[:, None, :] - inter
    return inter / union.clamp(min=1e-6)


class DropPath(nn.Module):
    """Stochastic depth (per-sample)."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep)
        return x * mask / keep


def _chunked_selective_scan(x, delta, A, Bm, Cm, chunk=32):
    """Numerically-safe chunked parallel selective scan (pure PyTorch).

    Computes h_t = exp(dtA_t) * h_{t-1} + dBx_t ; y_t = C_t . h_t with the
    associative log-cumsum trick inside chunks of `chunk` steps and a
    sequential state carry across chunks. Per-step decay is clamped so that
    exp(-cum) stays within fp32 range.

    Args:
        x (Tensor):     [B, N, E] input sequence (float32).
        delta (Tensor): [B, N, E] positive step sizes.
        A (Tensor):     [E, S] negative state matrix (log-parameterized).
        Bm (Tensor):    [B, N, S] input-dependent B.
        Cm (Tensor):    [B, N, S] input-dependent C.

    Returns:
        Tensor: [B, N, E] scan output (float32).
    """
    Bsz, N, E = x.shape
    S = A.shape[-1]
    dtA = (delta.unsqueeze(-1) * A).clamp(min=-2.0)          # [B,N,E,S], <= 0
    dBx = delta.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)
    pad = (-N) % chunk
    if pad:
        dtA = F.pad(dtA, (0, 0, 0, 0, 0, pad))
        dBx = F.pad(dBx, (0, 0, 0, 0, 0, pad))
        Cm = F.pad(Cm, (0, 0, 0, pad))
    Np, nc = N + pad, (N + pad) // chunk
    dtA = dtA.view(Bsz, nc, chunk, E, S)
    dBx = dBx.view(Bsz, nc, chunk, E, S)
    Cm = Cm.view(Bsz, nc, chunk, S)
    cum = dtA.cumsum(2)                                      # <= 0
    exp_neg = torch.exp(-cum)                                # <= e^64, fp32 ok
    prefix = (dBx * exp_neg).cumsum(2)
    inner = prefix[:, :, -1]                                 # [B,nc,E,S]
    exp_pos = torch.exp(cum)                                 # <= 1
    exp_end = exp_pos[:, :, -1]
    h = x.new_zeros(Bsz, E, S)
    ys = []
    for c in range(nc):
        hseq = exp_pos[:, c] * (prefix[:, c] + h.unsqueeze(1))   # [B,chunk,E,S]
        ys.append((hseq * Cm[:, c].unsqueeze(2)).sum(-1))        # [B,chunk,E]
        h = exp_end[:, c] * (h + inner[:, c])
    y = torch.stack(ys, 1).reshape(Bsz, Np, E)
    return y[:, :N]


# ---------------------------------------------------------------------------
# decoder layer base for self-attention-replacing innovations
# ---------------------------------------------------------------------------

class _InnovSelfAttnLayer(DeformableTransformerDecoderLayer):
    """Base for layers replacing self-attention by an innovation attention on
    the matching part, while dn queries keep the original masked self-attn."""

    INNOV_KWARGS = True

    def _innov_attn(self, mt, ref_xy, pos, boxes):
        raise NotImplementedError

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None,
                attn_mask=None, query_pos=None, dn_num=0, dec_boxes=None):
        if dn_num > 0:
            dn, mt = embed[:, :dn_num], embed[:, dn_num:]
            pos_mt = query_pos[:, dn_num:]
            mask_dn = attn_mask[:dn_num, :dn_num] if attn_mask is not None else None
            q = k = self.with_pos_embed(dn, query_pos[:, :dn_num])
            tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), dn.transpose(0, 1),
                                 attn_mask=mask_dn)[0].transpose(0, 1)
            dn = self.norm1(dn + self.dropout1(tgt))
            boxes_mt = dec_boxes[:, dn_num:] if dec_boxes is not None else None
            embed = torch.cat([dn, self._innov_attn(mt, refer_bbox[:, dn_num:, :2],
                                                    pos_mt, boxes_mt)], dim=1)
        else:
            embed = self._innov_attn(embed, refer_bbox[..., :2], query_pos, dec_boxes)
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2),
                              feats, shapes, padding_mask)
        embed = self.norm2(embed + self.dropout2(tgt))
        return self.forward_ffn(embed)


# ---------------------------------------------------------------------------
# decoder wrapper (dn_num passing + CLDM read/write + QGA caching)
# ---------------------------------------------------------------------------

class DeformableTransformerDecoderInnov(DeformableTransformerDecoder):
    """DeformableTransformerDecoder with optional CLDM memory and QGA loss.

    Args:
        cldm: CLDMMemory instance (optional) - cross-layer memory module.
        qga: QGALoss instance (optional) - gram-anchoring regularizer (training only).
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1,
                 cldm=None, qga=None):
        super().__init__(hidden_dim, decoder_layer, num_layers, eval_idx)
        self.cldm = cldm
        self.qga = qga

    def forward(self, embed, refer_bbox, feats, shapes, bbox_head, score_head,
                pos_mlp, attn_mask=None, padding_mask=None, dn_num=0):
        output = embed
        dec_bboxes, dec_cls = [], []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        if self.qga is not None:
            self.qga._cache = []
            self.qga.route_qga_loss = None
        if self.cldm is not None:
            self.cldm.reset(refer_bbox.shape[0], refer_bbox.shape[1] - dn_num,
                            refer_bbox.device, output.dtype)
        for i, layer in enumerate(self.layers):
            extra = dict(dn_num=dn_num, dec_boxes=refer_bbox) \
                if getattr(layer, 'INNOV_KWARGS', False) else {}
            output = layer(output, refer_bbox, feats, shapes, padding_mask,
                           attn_mask, pos_mlp(refer_bbox), **extra)

            if self.cldm is not None:  # memory read (matching part only)
                if dn_num > 0:
                    mt = self.cldm.read(output[:, dn_num:], refer_bbox[:, dn_num:])
                    output = torch.cat([output[:, :dn_num], mt], dim=1)
                else:
                    output = self.cldm.read(output, refer_bbox)

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            need_scores = self.training or i == self.eval_idx or self.cldm is not None
            scores = score_head[i](output) if need_scores else None

            if self.training:
                dec_cls.append(scores)
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(scores)
                dec_bboxes.append(refined_bbox)

            if self.cldm is not None and scores is not None:
                s = scores[:, dn_num:].sigmoid().max(-1).values if dn_num > 0 \
                    else scores.sigmoid().max(-1).values
                self.cldm.update(output[:, dn_num:] if dn_num > 0 else output,
                                 refined_bbox[:, dn_num:] if dn_num > 0 else refined_bbox, s)
            if self.qga is not None and self.training:
                self.qga.cache(output[:, dn_num:] if dn_num > 0 else output,
                               refined_bbox[:, dn_num:] if dn_num > 0 else refined_bbox)

            if not self.training and i == self.eval_idx:
                break
            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        if self.qga is not None and self.training:
            self.qga.compute()
        return torch.stack(dec_bboxes), torch.stack(dec_cls)


# ---------------------------------------------------------------------------
# RTDETRDecoder head base (yaml-registered drop-in replacements)
# ---------------------------------------------------------------------------

def _innov_forward(self, x, batch=None):
    """RTDETRDecoder.forward with dn_num passed to the innovation decoder."""
    from ultralytics.models.utils.ops import get_cdn_group
    feats, shapes = self._get_encoder_input(x)
    dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
        batch, self.nc, self.num_queries, self.denoising_class_embed.weight,
        self.num_denoising, self.label_noise_ratio, self.box_noise_scale, self.training)
    embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes,
                                                                        dn_embed, dn_bbox)
    dn_num = dn_meta['dn_num_split'][0] if dn_meta is not None else 0
    dec_bboxes, dec_scores = self.decoder(embed, refer_bbox, feats, shapes,
                                          self.dec_bbox_head, self.dec_score_head,
                                          self.query_pos_head, attn_mask=attn_mask,
                                          dn_num=dn_num)
    out = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
    if self.training:
        return out
    y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
    return y if self.export else (y, out)


class _RTDETRDecoderInnov(RTDETRDecoder):
    """Shared base: rebuilds the decoder with innovation layers/modules and
    forwards dn_num into the decoder."""

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp, eval_idx):
        raise NotImplementedError

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4, nh=8,
                 ndl=6, d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(), nd=100,
                 label_noise_ratio=0.5, box_noise_scale=1.0, learnt_init_query=False,
                 **innov_kwargs):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx, dropout, act,
                         nd, label_noise_ratio, box_noise_scale, learnt_init_query)
        self.decoder = self._rebuild_decoder(hd, nh, ndl, d_ffn, dropout, act, ndp,
                                             eval_idx, **innov_kwargs)
        self.forward = _innov_forward.__get__(self, type(self))

# ---------------------------------------------------------------------------
# token utilities for cross-attention innovations
# ---------------------------------------------------------------------------

def _token_centers(shapes, device, dtype=torch.float32):
    """Flattened multi-scale token centers (x, y) in [0, 1].

    Encoder features are concatenated level by level, row-major within each
    level (see RTDETRDecoder._get_encoder_input), so token t of level (h, w)
    maps to grid position (t // w, t % w).

    Returns:
        Tensor: [T, 2] token centers, x/y normalized to [0, 1].
    """
    centers = []
    for h, w in shapes:
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        centers.append(torch.stack([gx.reshape(-1), gy.reshape(-1)], -1))
    return torch.cat(centers, 0)


# ---------------------------------------------------------------------------
# decoder layer base for cross-attention-replacing innovations
# ---------------------------------------------------------------------------

class _InnovCrossAttnLayer(DeformableTransformerDecoderLayer):
    """Base for layers that replace the cross-attention of matching queries
    with an innovation read. Self-attention stays the baseline masked attention
    and denoising queries keep the original MSDeformAttn cross-attention, so
    DN training semantics are unchanged."""

    INNOV_KWARGS = True

    def _innov_cross(self, q, embed, ref_boxes, feats, shapes, padding_mask=None):
        """Cross-attention replacement for the matching-query part.

        Args:
            q (Tensor):      [B, N, C] query content + positional embedding.
            embed (Tensor):  [B, N, C] query content (no positional embedding).
            ref_boxes:       [B, N, 4] cxcywh reference boxes in [0, 1].
            feats (Tensor):  [B, T, C] flattened multi-scale features.
            shapes (List):   [(h, w), ...] per-level shapes.
            padding_mask:    [B, T] or None.

        Returns:
            Tensor: [B, N, C] cross-attention output (pre-residual).
        """
        raise NotImplementedError

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None,
                attn_mask=None, query_pos=None, dn_num=0, dec_boxes=None,
                shared_states=None):
        # self-attention (baseline; dn queries masked)
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed = self.norm1(embed + self.dropout1(tgt))
        # cross-attention: dn queries keep MSDeformAttn, matching queries innovate
        if dn_num > 0:
            tgt_dn = self.cross_attn(self.with_pos_embed(embed[:, :dn_num], query_pos[:, :dn_num]),
                                     refer_bbox[:, :dn_num].unsqueeze(2), feats, shapes,
                                     padding_mask)
            tgt_mt = self._innov_cross(self.with_pos_embed(embed[:, dn_num:], query_pos[:, dn_num:]),
                                       embed[:, dn_num:], refer_bbox[:, dn_num:], feats, shapes,
                                       padding_mask, shared_states)
            tgt = torch.cat([tgt_dn, tgt_mt], dim=1)
        else:
            tgt = self._innov_cross(self.with_pos_embed(embed, query_pos), embed,
                                    refer_bbox, feats, shapes, padding_mask, shared_states)
        embed = self.norm2(embed + self.dropout2(tgt))
        return self.forward_ffn(embed)
