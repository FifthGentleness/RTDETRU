"""MGIoU specialized to horizontal boxes (AAAI 2026)."""

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss


class RTDETRDetectionLossMGIoUHBB(RTDETRDetectionLoss):
    """Average the two 1D GIoUs from the horizontal and vertical projections."""

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        l1_key, geometry_key = f'loss_bbox{postfix}', f'loss_giou{postfix}'
        num_boxes = len(gt_bboxes)
        if num_boxes == 0:
            return {key: torch.zeros((), device=pred_bboxes.device) for key in (l1_key, geometry_key)}

        l1 = F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum')
        # Compute tiny box geometry in FP32 under AMP, keeping native L1 precision.
        low_precision = (torch.float16, torch.bfloat16)
        pred = pred_bboxes.float() if pred_bboxes.dtype in low_precision else pred_bboxes
        target = gt_bboxes.float() if gt_bboxes.dtype in low_precision else gt_bboxes
        geometry = self._geometry_loss(pred, target)
        return {
            l1_key: self.loss_gain['bbox'] * l1 / num_boxes,
            geometry_key: self.loss_gain['giou'] * geometry.sum() / num_boxes,
        }

    def _geometry_loss(self, pred, target):
        pred_half, target_half = pred[..., 2:] * 0.5, target[..., 2:] * 0.5
        pred_lo, pred_hi = pred[..., :2] - pred_half, pred[..., :2] + pred_half
        target_lo, target_hi = target[..., :2] - target_half, target[..., :2] + target_half
        # Signed intersection is negative across a gap, preserving a gradient there.
        intersection = torch.minimum(pred_hi, target_hi) - torch.maximum(pred_lo, target_lo)
        enclosure = torch.maximum(pred_hi, target_hi) - torch.minimum(pred_lo, target_lo)
        projected_giou = intersection / enclosure.clamp_min(1e-7)
        return 0.5 * (1.0 - projected_giou.mean(dim=-1))
