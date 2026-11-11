"""Inner-GIoU: replace GIoU's overlap term with Inner-IoU."""

import torch

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou, get_inner_iou


class RTDETRDetectionLossInnerGIoU(RTDETRDetectionLoss):
    """Use Inner-IoU for overlap while retaining the original GIoU enclosure penalty."""

    def __init__(self, *args, inner_ratio=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < inner_ratio <= 1.5:
            raise ValueError('inner_ratio must be in (0, 1.5]')
        self.inner_ratio = inner_ratio

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        losses = super()._get_loss_bbox(pred_bboxes, gt_bboxes, postfix)
        if len(gt_bboxes) == 0:
            return losses

        low_precision = (torch.float16, torch.bfloat16)
        pred = pred_bboxes.float() if pred_bboxes.dtype in low_precision else pred_bboxes
        target = gt_bboxes.float() if gt_bboxes.dtype in low_precision else gt_bboxes
        inner_iou = get_inner_iou(pred, target, eps=1e-12, ratio=self.inner_ratio).squeeze(-1)
        iou = bbox_iou(pred, target, xywh=True).squeeze(-1)
        giou = bbox_iou(pred, target, xywh=True, GIoU=True).squeeze(-1)
        # Inner-GIoU = GIoU + IoU - InnerIoU; preserve the GIoU penalty.
        inner_giou = giou + iou - inner_iou
        key = f'loss_giou{postfix}'
        losses[key] = self.loss_gain['giou'] * inner_giou.mean()
        return losses
