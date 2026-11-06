"""Penalize excess overlap between matched neighbors (Point2RBox-v2, CVPR 2025)."""

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou


def gaussian_overlap(a, b):
    """Bhattacharyya coefficient for axis-aligned xywh boxes in pixel coordinates."""
    a_var = (0.5 * a[..., 2:].clamp_min(2.0)).square()
    b_var = (0.5 * b[..., 2:].clamp_min(2.0)).square()
    mean_var = 0.5 * (a_var + b_var)
    distance = (a[..., :2] - b[..., :2]).square() / (8.0 * mean_var)
    scale = 0.5 * (mean_var.log() - 0.5 * (a_var.log() + b_var.log()))
    return (-(distance + scale).sum(dim=-1).clamp_min(0.0)).exp()


class RTDETRDetectionLossNeighborOverlap(RTDETRDetectionLoss):
    """One extra geometry term on final-layer matching boxes; auxiliary and DN losses are unchanged."""

    innovation_schedule = True

    def __init__(self, *args, max_weight=0.05, margin=0.05, smooth_beta=0.05, start_epoch=0, full_epoch=1,
                 min_iou=0.50, small_px=32.0, max_distance=2.5, ambiguous_iou=0.80, **kwargs):
        super().__init__(*args, **kwargs)
        if (max_weight < 0 or not 0 <= margin < 1 or smooth_beta <= 0 or not 0 <= start_epoch < full_epoch
                or not 0 <= min_iou <= 1 or small_px <= 0 or max_distance <= 0 or not 0 < ambiguous_iou <= 1):
            raise ValueError('Invalid matched-neighbor overlap loss parameters')
        self.max_weight, self.weight = max_weight, max_weight
        self.margin, self.smooth_beta = margin, smooth_beta
        self.start_epoch, self.full_epoch = start_epoch, full_epoch
        self.min_iou, self.small_px = min_iou, small_px
        self.max_distance, self.ambiguous_iou = max_distance, ambiguous_iou
        self._neighbor_context = None

    def set_epoch(self, epoch, total_epochs=0):
        progress = (epoch + 1 - self.start_epoch) / (self.full_epoch - self.start_epoch)
        self.weight = self.max_weight * min(1.0, max(0.0, progress))

    @torch.no_grad()
    def _neighbor_indices(self, pred, gt, cls, predicted_cls):
        """Mutual nearest two of the eligible GT neighbors; count each pair once."""
        correct = (predicted_cls == cls) & (bbox_iou(pred, gt, xywh=True).flatten() >= self.min_iou)
        small = gt[:, 2:].prod(dim=-1) < self.small_px ** 2
        dist_xy = gt[:, None, :2] - gt[None, :, :2]
        scale = (0.5 * (gt[:, None, 2:] + gt[None, :, 2:])).clamp_min(4.0)
        dist_sq = (dist_xy / scale).square().sum(dim=-1)
        gt_iou = bbox_iou(gt[:, None, :], gt[None, :, :], xywh=True).squeeze(-1)
        row = torch.arange(len(gt), device=gt.device)[:, None]
        valid = (correct[:, None] & correct[None, :] & (cls[:, None] == cls[None, :])
                 & (small[:, None] | small[None, :]) & (gt_iou < self.ambiguous_iou)
                 & (dist_sq <= self.max_distance ** 2) & (row != row.T))
        distance, neighbor = dist_sq.masked_fill(~valid, torch.inf).topk(min(2, len(gt) - 1), largest=False)
        finite = torch.isfinite(distance)
        reciprocal = ((neighbor[neighbor] == row[..., None]) & finite[neighbor]).any(dim=-1)
        return neighbor, finite & reciprocal & (row < neighbor)

    def _matched_loss(self, boxes, scores, gt_bboxes, gt_cls, image_hw, match_indices):
        idx, gt_idx = self._get_index(match_indices)
        h, w = image_hw
        pixel_scale = boxes.new_tensor([w, h, w, h], dtype=torch.float32)
        pred = boxes[idx].float() * pixel_scale
        gt = gt_bboxes[gt_idx].detach().float() * pixel_scale
        cls = gt_cls[gt_idx]
        predicted_cls = scores.detach()[idx].argmax(dim=-1)
        sizes = [len(src) for src, _ in match_indices]
        loss = pred.new_zeros(())
        for p, g, c, pc in zip(pred.split(sizes), gt.split(sizes), cls.split(sizes), predicted_cls.split(sizes)):
            if len(p) < 2:
                continue
            neighbor, valid = self._neighbor_indices(p.detach(), g, c, pc)
            excess = (gaussian_overlap(p[:, None], p[neighbor])
                      - gaussian_overlap(g[:, None], g[neighbor]) - self.margin).clamp_min(0.0)
            cost = F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=self.smooth_beta, reduction='none')
            loss = loss + cost.masked_fill(~valid, 0.0).sum()
        return loss / max(len(pred), 1)

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        if self.weight == 0:
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        if 'image_hw' not in batch:
            raise ValueError('Neighbor overlap loss requires image_hw=(height, width) in targets')
        context = {'image_hw': batch['image_hw'], 'term': None}
        self._neighbor_context = context
        try:
            loss = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        finally:
            self._neighbor_context = None
        # Add the term after the base criterion constructs aux/DN keys.
        loss['loss_neighbor_overlap'] = context['term']
        return loss

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        context = self._neighbor_context
        # The base criterion calls _get_loss on the final matching layer
        # first, then the auxiliary layers, then the '_dn' branch.
        final = context is not None and context['term'] is None and not postfix
        if final and match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                         masks=masks, gt_mask=gt_mask)
        loss = super()._get_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                 masks=masks, gt_mask=gt_mask, postfix=postfix, match_indices=match_indices)
        if final:
            context['term'] = self.weight * self._matched_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls,
                                                              context['image_hw'], match_indices)
        return loss
