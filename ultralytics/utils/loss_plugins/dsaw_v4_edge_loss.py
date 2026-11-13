"""DSAWv4 with a gated four-edge loss on the final regular decoder layer."""

import math

import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy

from .dsaw_v4_loss import RTDETRDetectionLossDSAWV4


class RTDETRDetectionLossDSAWV4Edge(RTDETRDetectionLossDSAWV4):
    """Add a size-normalized Smooth L1 term while retaining all v4 losses."""

    def __init__(self, *args, edge_gain=0.7, edge_beta=0.05, edge_size_max=32.0,
                 edge_short_min=8.0, edge_short_max=16.0, edge_iou_min=0.60,
                 edge_iou_max=0.85, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_gain = float(edge_gain)
        self.edge_beta = float(edge_beta)
        self.edge_size_max = float(edge_size_max)
        self.edge_short_min = float(edge_short_min)
        self.edge_short_max = float(edge_short_max)
        self.edge_iou_min = float(edge_iou_min)
        self.edge_iou_max = float(edge_iou_max)
        if not all(math.isfinite(v) for v in (self.edge_gain, self.edge_beta, self.edge_size_max,
                                               self.edge_short_min, self.edge_short_max,
                                               self.edge_iou_min, self.edge_iou_max)):
            raise ValueError("DSAWv4 edge parameters must be finite")
        if not (self.edge_gain >= 0 and self.edge_beta > 0
                and 0 < self.edge_short_min < self.edge_short_max <= self.edge_size_max
                and 0 <= self.edge_iou_min < self.edge_iou_max <= 1):
            raise ValueError("Invalid DSAWv4 edge gain, Smooth L1 beta or gate bounds")

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        if self.edge_gain == 0:
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)

        # The inherited forward calls _get_loss for the last regular layer first.
        self._edge_image_hw = batch['image_hw']
        try:
            losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        finally:
            self._edge_image_hw = None
        # The base creates zero DN placeholders during validation.
        losses.pop('loss_edge_dn', None)
        return losses

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        image_hw = getattr(self, '_edge_image_hw', None) if not postfix else None
        if image_hw is not None:
            # Consume the marker before any auxiliary or denoising loss is called.
            self._edge_image_hw = None
            if match_indices is None:
                match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                             masks=masks, gt_mask=gt_mask)

        losses = super()._get_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                   masks=masks, gt_mask=gt_mask, postfix=postfix, match_indices=match_indices)
        if image_hw is not None:
            idx, gt_idx = self._get_index(match_indices)
            losses['loss_edge'] = self._edge_loss(pred_bboxes[idx], gt_bboxes[gt_idx], image_hw)
        return losses

    def _edge_loss(self, pred, gt, image_hw):
        pred, gt = pred.float(), gt.detach().float()
        h, w = image_hw
        scale = pred.new_tensor((w, h))
        sides = gt[:, 2:] * scale
        short = sides.amin(dim=-1)
        size = sides.prod(dim=-1).clamp_min(0).sqrt()
        q = bbox_iou(pred.detach(), gt, xywh=True).squeeze(-1)
        selected = ((short >= self.edge_short_min) & (short < self.edge_short_max)
                    & (size < self.edge_size_max) & (q >= self.edge_iou_min) & (q < self.edge_iou_max))

        # Left/right errors use GT width; top/bottom use GT height.
        delta = ((xywh2xyxy(pred) - xywh2xyxy(gt)) * scale.repeat(2)
                 / sides.clamp_min(self.edge_short_min).repeat(1, 2))
        per_box = F.smooth_l1_loss(delta, delta.new_zeros(delta.shape),
                                   beta=self.edge_beta, reduction='none').mean(dim=-1)
        # Normalize by all regular final-layer matches, including out-of-gate pairs.
        return self.edge_gain * per_box.masked_fill(~selected, 0.0).sum() / max(len(gt), 1)
