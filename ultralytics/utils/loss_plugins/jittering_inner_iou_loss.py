"""Jittering DIoU with each candidate's IoU term replaced by Inner-IoU."""

import torch

from ultralytics.utils.metrics import get_inner_iou

from .jittering_iou_loss import RTDETRDetectionLossJitteringIoU


class RTDETRDetectionLossJitteringInnerIoU(RTDETRDetectionLossJitteringIoU):
    """Replace each candidate DIoU overlap term with its InnerIoU counterpart."""

    def __init__(self, *args, inner_ratio=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < inner_ratio <= 1.5:
            raise ValueError('inner_ratio must be in (0, 1.5]')
        self.inner_ratio = inner_ratio

    def _geometry_loss(self, pred, target):
        if self.jitter_scales.device != target.device:
            self.jitter_scales = self.jitter_scales.to(device=target.device)
        pred, target = pred.unsqueeze(-2), target.unsqueeze(-2)
        pred_half = pred[..., 2:] * 0.5
        target_half = target[..., 2:] * self.jitter_scales * 0.5
        pred_lo, pred_hi = pred[..., :2] - pred_half, pred[..., :2] + pred_half
        target_lo, target_hi = target[..., :2] - target_half, target[..., :2] + target_half

        # Inner-IoU replaces IoU inside every jittered DIoU candidate.
        expanded_pred = torch.cat((pred[..., :2], pred[..., 2:] * self.inner_ratio), dim=-1)
        expanded_target = torch.cat((target[..., :2], target[..., 2:] * self.inner_ratio), dim=-1)
        expanded_pred = expanded_pred.expand_as(target)
        inner_iou = get_inner_iou(expanded_pred, expanded_target, eps=1e-12).squeeze(-1)
        losses = 1.0 - inner_iou

        if self.jitter_metric == 'diou':
            enclosure = torch.maximum(pred_hi, target_hi) - torch.minimum(pred_lo, target_lo)
            center_distance = (pred[..., :2] - target[..., :2]).square().sum(dim=-1)
            losses = losses + center_distance / enclosure.square().sum(dim=-1).clamp_min(1e-12)

        total = losses[..., 0] + losses.min(dim=-1).values
        return total * 0.5 if self.jitter_normalize else total
