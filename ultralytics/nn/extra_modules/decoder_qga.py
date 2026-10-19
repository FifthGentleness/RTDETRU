"""
QGA: Query Gram Anchoring (RT-DETR decoder innovation, training-only).

Inspired by:
  - DINOv3 (2025) Gram Anchoring: keeping the pairwise feature-similarity
    structure from degrading over long training.

Label-free variant: per-layer cached (feature, box) pairs are used to penalize
high feature similarity between queries whose predicted boxes heavily overlap
(competitor/duplicate pairs), keeping the pairwise similarity structure
discriminative for duplicate suppression in dense scenes.

Zero inference cost - architecture is identical to the baseline; the loss is
exposed via `route_qga_loss` and routed into the total loss by
DetectionModel.loss (see tasks.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.transformer import DeformableTransformerDecoderLayer

from .decoder_innov_common import (_RTDETRDecoderInnov, _pairwise_iou,
                                   DeformableTransformerDecoderInnov)

__all__ = ('QGALoss', 'RTDETRDecoderQGA')


class QGALoss(nn.Module):
    """Query diversity Gram-anchoring loss (label-free variant).

    Args:
        margin (float): hinge margin on competitor-pair cosine similarity.
        iou_thr (float): box-IoU threshold defining competitor pairs.
    """

    def __init__(self, margin=0.3, iou_thr=0.5):
        super().__init__()
        self.margin, self.iou_thr = margin, iou_thr
        self.route_qga_loss = None
        self._cache = []

    def cache(self, feat, boxes):
        """Cache one decoder layer's matching-part features and boxes."""
        if self.training:
            self._cache.append((feat, boxes))

    def compute(self):
        """Compute the anchored Gram loss from cached layers and reset state."""
        losses = []
        for feat, boxes in self._cache:
            f = F.normalize(feat, dim=-1)
            G = f @ f.transpose(1, 2)
            iou = _pairwise_iou(boxes)
            n = feat.shape[1]
            eye = torch.eye(n, dtype=torch.bool, device=feat.device).unsqueeze(0)
            comp = (iou > self.iou_thr) & (~eye)
            if bool(comp.any()):
                losses.append(F.relu(G[comp] - self.margin).pow(2).mean())
        self._cache = []
        self.route_qga_loss = torch.stack(losses).mean() if losses else None
        return self.route_qga_loss


class RTDETRDecoderQGA(_RTDETRDecoderInnov):
    """RT-DETR decoder + query Gram-anchoring regularization (training-only,
    zero inference cost; architecture identical to baseline).

    Extra yaml args (after the standard RTDETRDecoder args):
        qga_margin (float, default 0.3): hinge margin.
        qga_iou_thr (float, default 0.5): competitor-pair IoU threshold.
    """

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp, eval_idx,
                         qga_margin=0.3, qga_iou_thr=0.5, **_):
        layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act,
                                                  self.nl, ndp)
        qga = QGALoss(qga_margin, qga_iou_thr)
        return DeformableTransformerDecoderInnov(hd, layer, ndl, eval_idx, qga=qga)
