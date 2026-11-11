"""MGIoU-HBB with each projected IoU term replaced by 1-D Inner-IoU."""

import torch

from .mgiou_hbb_loss import RTDETRDetectionLossMGIoUHBB


class RTDETRDetectionLossMGIoUHBBInnerIoU(RTDETRDetectionLossMGIoUHBB):
    """Replace each projected IoU by InnerIoU, retaining its gap penalty."""

    def __init__(self, *args, inner_ratio=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < inner_ratio <= 1.5:
            raise ValueError('inner_ratio must be in (0, 1.5]')
        self.inner_ratio = inner_ratio

    def _geometry_loss(self, pred, target):
        pred_half, target_half = pred[..., 2:] * 0.5, target[..., 2:] * 0.5
        pred_lo, pred_hi = pred[..., :2] - pred_half, pred[..., :2] + pred_half
        target_lo, target_hi = target[..., :2] - target_half, target[..., :2] + target_half

        # Preserve the original 1-D GIoU gap penalty on each axis.
        intersection = (torch.minimum(pred_hi, target_hi) - torch.maximum(pred_lo, target_lo)).clamp_min(0)
        union = pred[..., 2:] + target[..., 2:] - intersection
        enclosure = torch.maximum(pred_hi, target_hi) - torch.minimum(pred_lo, target_lo)
        gap_penalty = (enclosure - union).clamp_min(0) / enclosure.clamp_min(1e-7)

        # Replace only the projected IoU term with the IoU of scaled 1-D boxes.
        inner_half_pred = pred_half * self.inner_ratio
        inner_half_target = target_half * self.inner_ratio
        inner_intersection = (
            torch.minimum(pred[..., :2] + inner_half_pred, target[..., :2] + inner_half_target)
            - torch.maximum(pred[..., :2] - inner_half_pred, target[..., :2] - inner_half_target)
        ).clamp_min(0)
        inner_union = inner_half_pred * 2 + inner_half_target * 2 - inner_intersection
        inner_iou = inner_intersection / inner_union.clamp_min(1e-12)
        return (1.0 - inner_iou + gap_penalty).mean(dim=-1) * 0.5
