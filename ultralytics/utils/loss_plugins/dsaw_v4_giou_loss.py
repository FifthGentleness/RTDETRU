"""DSAW-v4 ablation: retain its decoupled weighting, use pure GIoU."""

import torch

from .dsaw_v4_loss import RTDETRDetectionLossDSAWV4
from ultralytics.utils.metrics import bbox_iou

__all__ = ('RTDETRDetectionLossDSAWV4GIoU',)


class RTDETRDetectionLossDSAWV4GIoU(RTDETRDetectionLossDSAWV4):
    """Inherit v4 classification, EMA and matching; replace only its box geometry.

    Unlike setting blend_center=0, this computes pure GIoU even at IoU=0.
    Inherited blend/inner parameters have no effect on this override.
    """

    def _get_loss_bbox_w(self, pred_bboxes, gt_bboxes, w_geo_l1,
                         w_geo_iou=None, postfix=''):
        name_bbox, name_giou = f'loss_bbox{postfix}', f'loss_giou{postfix}'
        if len(gt_bboxes) == 0:
            return {key: torch.tensor(0., device=pred_bboxes.device)
                    for key in (name_bbox, name_giou)}

        if w_geo_l1 is None:
            w_geo_l1 = torch.ones(len(gt_bboxes), device=pred_bboxes.device)
        if w_geo_iou is None:
            w_geo_iou = w_geo_l1

        # Keep v4's L1 reduction and independent geometry weights exactly.
        l1 = (pred_bboxes - gt_bboxes).abs().sum(-1)
        giou = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True).squeeze(-1)
        loss = {
            name_bbox: self.loss_gain['bbox'] * (l1 * w_geo_l1).sum() / len(gt_bboxes),
            name_giou: self.loss_gain['giou'] * (giou * w_geo_iou).sum() / len(gt_bboxes),
        }
        return {key: value.squeeze() for key, value in loss.items()}
