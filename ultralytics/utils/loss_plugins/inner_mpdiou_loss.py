# Ultralytics YOLO 🐛, AGPL-3.0 license
"""
Inner-MPDIoU Loss for RT-DETR — self-contained module.

Port of Inner-MPDIoU from UAV-DETR (SFS-DETR):
    MPDIoU + Inner-IoU → Inner-MPDIoU

    MPDIoU reference: Ma et al., "MPDIoU: A Better IoU Metric for Bounding
    Box Regression", arXiv 2307.07952.

    Inner-IoU reference: Zhang et al., "Inner-IoU: A More Efficient IoU
    Loss for Bounding Box Regression".

Design goals (minimal invasion of the original RT-DETR code):
    - Everything lives in THIS file. `ultralytics/models/utils/loss.py`,
      `ultralytics/utils/metrics.py` and all other core files stay untouched.
    - `RTDETRDetectionLossInnerMPDIoU` is a drop-in subclass of
      `RTDETRDetectionLoss`: only the GIoU term of the bbox regression loss
      is replaced by Inner-MPDIoU. L1 term, VFL classification loss, Hungarian
      matching cost, loss gains, auxiliary/denoising loss plumbing are all
      inherited unchanged.
    - This file is registered in the loss plugin registry
      (`ultralytics/utils/loss_plugins/__init__.py`) under the name
      'inner_mpdiou'. `RTDETRDetectionModel.init_criterion()` resolves
      `loss_name: inner_mpdiou` from the model yaml through that registry.

Core idea
---------
Inner-MPDIoU combines two orthogonal improvements:

1. MPDIoU (Minimum Point Distance IoU): Instead of using center-point distance
   like DIoU, MPDIoU directly penalizes the distances between the top-left
   and bottom-right corner points of the predicted and GT boxes:
       d1 = (x1_gt - x1_pred)^2 + (y1_gt - y1_pred)^2   (top-left corner)
       d2 = (x2_gt - x2_pred)^2 + (y2_gt - y2_pred)^2   (bottom-right corner)
       MPDIoU = IoU - d1/h_w - d2/h_w
   This provides more direct supervision on box corner alignment than
   center-based distances, and naturally captures both position and shape
   differences simultaneously.

2. Inner-IoU: Scale both the predicted and GT boxes by a ratio (center-fixed)
   before computing IoU. When ratio < 1, the "inner" boxes are smaller than
   the originals, making IoU more sensitive to boundary alignment → faster
   convergence for high-IoU predictions. When ratio > 1, the "outer" boxes
   are larger, providing softer gradients for low-IoU predictions.

Inner-MPDIoU formula:
    InnerMPDIoU = InnerIoU - d1/h_w - d2/h_w

    InnerIoU = inter_inner / union_inner
        where inter_inner and union_inner are computed from boxes scaled by ratio

    d1 = (x1_gt - x1_pred)^2 + (y1_gt - y1_pred)^2
    d2 = (x2_gt - x2_pred)^2 + (y2_gt - y2_pred)^2
    h_w = normalization factor (default: image_height * image_width)
"""

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('inner_mpdiou', 'InnerMPDIoULoss', 'RTDETRDetectionLossInnerMPDIoU')


def _get_inner_iou(box1, box2, xywh=True, eps=1e-7, ratio=0.7):
    """
    Compute IoU on ratio-scaled (center-fixed) boxes.

    Args:
        box1 (torch.Tensor): Predicted boxes, shape (..., 4).
        box2 (torch.Tensor): Ground-truth boxes, shape (..., 4).
        xywh (bool): If True, inputs are (cx, cy, w, h).
        eps (float): Small value to avoid division by zero.
        ratio (float): Scale factor for inner/outer boxes.
            ratio < 1: inner boxes (smaller) → more sensitive to boundary alignment.
            ratio > 1: outer boxes (larger) → softer gradient for low-IoU predictions.

    Returns:
        (torch.Tensor): Inner-IoU values, shape (..., 1).
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        x1, y1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        x2, y2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    b1_x1, b1_x2 = x1 - (w1 * ratio) / 2, x1 + (w1 * ratio) / 2
    b1_y1, b1_y2 = y1 - (h1 * ratio) / 2, y1 + (h1 * ratio) / 2
    b2_x1, b2_x2 = x2 - (w2 * ratio) / 2, x2 + (w2 * ratio) / 2
    b2_y1, b2_y2 = y2 - (h2 * ratio) / 2, y2 + (h2 * ratio) / 2

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
    return inter / union


def inner_mpdiou(box1, box2, xywh=True, mpdiou_hw=2, ratio=0.7, eps=1e-7):
    """
    Inner-MPDIoU between predicted and ground-truth boxes.

    Combines Inner-IoU (ratio-scaled boxes) with MPDIoU's minimum point
    distance penalty terms.

    Args:
        box1 (torch.Tensor): Predicted boxes, shape (..., 4).
        box2 (torch.Tensor): Ground-truth boxes, shape (..., 4).
        xywh (bool): If True, inputs are (cx, cy, w, h); if False, (x1, y1, x2, y2).
        mpdiou_hw (float): Normalization factor for the point distance terms.
            Typically set to image_height * image_width (e.g., 640*640=409600
            for 640px images). Default 2 (for normalized coordinates where
            the image spans [0,1]x[0,1], h*w=1A).
        ratio (float): Scale factor for Inner-IoU computation.
            ratio < 1: inner boxes → more sensitive to boundary alignment.
            ratio > 1: outer boxes → softer gradient for low-IoU predictions.
        eps (float): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): Inner-MPDIoU values, shape (..., 1).
            Use `1 - InnerMPDIoU` as loss.
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_, x1 + w1_
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_
        b2_x1, b2_x2 = x2 - w2_, x2 + w2_
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inner_iou = _get_inner_iou(box1, box2, xywh=xywh, ratio=ratio)

    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2

    return inner_iou - d1 / mpdiou_hw - d2 / mpdiou_hw


class InnerMPDIoULoss(torch.nn.Module):
    """Standalone module wrapper of the Inner-MPDIoU bbox regression loss."""

    def __init__(self, mpdiou_hw=2, ratio=0.7):
        """Initialize with MPDIoU normalization factor and Inner-IoU ratio."""
        super().__init__()
        self.mpdiou_hw = mpdiou_hw
        self.ratio = ratio

    def forward(self, pred_bboxes, gt_bboxes):
        """Return mean Inner-MPDIoU regression loss `1 - InnerMPDIoU` for matched box pairs."""
        return (1.0 - inner_mpdiou(pred_bboxes, gt_bboxes, xywh=True,
                                   mpdiou_hw=self.mpdiou_hw, ratio=self.ratio)).mean()


class RTDETRDetectionLossInnerMPDIoU(RTDETRDetectionLoss):
    """
    Drop-in replacement of `RTDETRDetectionLoss` using Inner-MPDIoU for the
    bbox regression term.

    Only `_get_loss_bbox` is overridden: the GIoU term is replaced by
    `1 - InnerMPDIoU`. Everything else (VFL classification loss, L1 term,
    Hungarian matching, auxiliary and denoising losses, loss gains) is
    inherited unchanged from `RTDETRDetectionLoss`.
    """

    def __init__(self, *args, mpdiou_hw=2, inner_ratio=0.7, **kwargs):
        """
        Args:
            mpdiou_hw (float): Normalization factor for MPDIoU point distance
                terms. For normalized coordinates [0,1], the image spans
                h*w area. Default 2 (UAV-DETR convention).
            inner_ratio (float): Scale factor for Inner-IoU computation.
                ratio < 1: inner boxes (smaller) → more sensitive to boundary
                    alignment, accelerates high-IoU convergence.
                ratio > 1: outer boxes (larger) → softer gradient for low-IoU
                    predictions, accelerates low-IoU convergence.
                UAV-DETR default: 0.7.
            *args, **kwargs: Forwarded to `RTDETRDetectionLoss` unchanged.
        """
        super().__init__(*args, **kwargs)
        self.mpdiou_hw = mpdiou_hw
        self.inner_ratio = inner_ratio

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        """L1 term unchanged + GIoU term replaced by Inner-MPDIoU."""
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=self.device)
            loss[name_giou] = torch.tensor(0., device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / len(gt_bboxes)

        loss[name_giou] = 1.0 - inner_mpdiou(pred_bboxes, gt_bboxes, xywh=True,
                                              mpdiou_hw=self.mpdiou_hw, ratio=self.inner_ratio)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}