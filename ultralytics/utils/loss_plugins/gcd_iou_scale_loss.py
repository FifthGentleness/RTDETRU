import math

import torch

from ultralytics.utils.metrics import bbox_iou

from .gcd_loss import RTDETRDetectionLossGCD


class RTDETRDetectionLossGCDIoUScale(RTDETRDetectionLossGCD):
    """Scale each matched box's GCD loss by its 2D IoU."""

    def __init__(self, *args, iou_low=0.55, iou_high=0.80, max_scale=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.iou_low, self.iou_high, self.max_scale = map(float, (iou_low, iou_high, max_scale))
        if not 0.0 <= self.iou_low < self.iou_high <= 1.0:
            raise ValueError("Expected 0 <= iou_low < iou_high <= 1.")
        if not math.isfinite(self.max_scale) or self.max_scale < 1.0:
            raise ValueError("Expected a finite max_scale >= 1.")

    def _geometry_loss(self, pred, target):
        with torch.no_grad():
            iou = bbox_iou(pred, target, xywh=True).squeeze(-1)
            ramp = ((iou - self.iou_low) / (self.iou_high - self.iou_low)).clamp(0.0, 1.0)
            scale = 1.0 + (self.max_scale - 1.0) * ramp
        return super()._geometry_loss(pred, target) * scale
