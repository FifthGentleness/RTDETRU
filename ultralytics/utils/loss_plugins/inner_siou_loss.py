# Ultralytics YOLO 🐛, AGPL-3.0 license
"""
Inner-SIoU Loss for RT-DETR — self-contained module.

Port of Inner-SIoU from UAV-DETR:
    SIoU (Scylla IoU) + Inner-IoU → Inner-SIoU

    SIoU reference: Zhora et al., "SIoU Loss: More Powerful Learning
    for Bounding Box Regression", arXiv 2205.12740.

    Inner-IoU reference: Zhang et al., "Inner-IoU: A More Efficient IoU
    Loss for Bounding Box Regression".

Design goals (minimal invasion of the original RT-DETR code):
    - Everything lives in THIS file. `ultralytics/models/utils/loss.py`,
      `ultralytics/utils/metrics.py` and all other core files stay untouched.
    - `RTDETRDetectionLossInnerSIoU` is a drop-in subclass of
      `RTDETRDetectionLoss`: only the GIoU term of the bbox regression loss
      is replaced by Inner-SIoU. L1 term, VFL classification loss, Hungarian
      matching cost, loss gains, auxiliary/denoising loss plumbing are all
      inherited unchanged.
    - This file is registered in the loss plugin registry
      (`ultralytics/utils/loss_plugins/__init__.py`) under the name
      'inner_siou'. `RTDETRDetectionModel.init_criterion()` resolves
      `loss_name: inner_siou` from the model yaml through that registry.

Core idea
---------
Inner-SIoU combines two orthogonal improvements:

1. Inner-IoU: Scale both the predicted and GT boxes by a ratio (center-fixed)
   before computing IoU. When ratio < 1, the "inner" boxes are smaller than
   the originals, making IoU more sensitive to boundary alignment → faster
   convergence for high-IoU predictions. When ratio > 1, the "outer" boxes
   are larger, which can help low-IoU predictions converge faster by
   providing a softer gradient signal. UAV-DETR uses ratio=1.25 (>1).

2. SIoU (Scylla IoU): Replaces the isotropic center-distance term of GIoU/DIoU
   with an angle-aware distance cost that considers the relative angle between
   the center offset vector and the axes. This makes the loss penalize
   diagonal offsets differently from axis-aligned offsets, and adds a shape
   consistency term.

Inner-SIoU formula:
    InnerSIoU = InnerIoU - 0.5 * (DistanceCost + ShapeCost)

    InnerIoU = inter_inner / union_inner
        where inter_inner and union_inner are computed from boxes scaled by ratio

    Angle cost (SIoU):
        s_cw = (cx2 - cx1) / 2,  s_ch = (cy2 - cy1) / 2
        sigma = sqrt(s_cw^2 + s_ch^2)
        sin_alpha = min(|s_cw|, |s_ch|) / sigma  (after thresholding)
        angle_cost = cos(2 * arcsin(sin_alpha) - pi/2)

    Distance cost (SIoU):
        gamma = angle_cost - 2
        distance_cost = 2 - exp(gamma * rho_x^2) - exp(gamma * rho_y^2)
        where rho_x = s_cw / cw, rho_y = s_ch / ch

    Shape cost (SIoU):
        omega_w = |w1 - w2| / max(w1, w2)
        omega_h = |h1 - h2| / max(h1, h2)
        shape_cost = (1 - exp(-omega_w))^4 + (1 - exp(-omega_h))^4
"""

import math

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('inner_siou', 'InnerSIoULoss', 'RTDETRDetectionLossInnerSIoU')


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


def inner_siou(box1, box2, xywh=True, ratio=0.7, eps=1e-7):
    """
    Inner-SIoU between predicted and ground-truth boxes.

    Combines Inner-IoU (ratio-scaled boxes) with SIoU's angle-aware distance
    and shape cost terms.

    Args:
        box1 (torch.Tensor): Predicted boxes, shape (..., 4).
        box2 (torch.Tensor): Ground-truth boxes, shape (..., 4).
        xywh (bool): If True, inputs are (cx, cy, w, h); if False, (x1, y1, x2, y2).
        ratio (float): Scale factor for Inner-IoU computation.
            ratio < 1: inner boxes → more sensitive to boundary alignment.
            ratio > 1: outer boxes → softer gradient for low-IoU predictions.
            UAV-DETR default: 1.25 (outer boxes for faster low-IoU convergence).
        eps (float): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): Inner-SIoU values, shape (..., 1). Use `1 - InnerSIoU` as loss.
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

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
    sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = pow(2, 0.5) / 2
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

    rho_x = (s_cw / cw) ** 2
    rho_y = (s_ch / ch) ** 2
    gamma = angle_cost - 2
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

    omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)

    return inner_iou - 0.5 * (distance_cost + shape_cost) + eps


class InnerSIoULoss(torch.nn.Module):
    """Standalone module wrapper of the Inner-SIoU bbox regression loss."""

    def __init__(self, ratio=0.7):
        """Initialize with the Inner-IoU ratio."""
        super().__init__()
        self.ratio = ratio

    def forward(self, pred_bboxes, gt_bboxes):
        """Return mean Inner-SIoU regression loss `1 - InnerSIoU` for matched box pairs."""
        return (1.0 - inner_siou(pred_bboxes, gt_bboxes, xywh=True, ratio=self.ratio)).mean()


class RTDETRDetectionLossInnerSIoU(RTDETRDetectionLoss):
    """
    Drop-in replacement of `RTDETRDetectionLoss` using Inner-SIoU for the bbox
    regression term.

    Only `_get_loss_bbox` is overridden: the GIoU term is replaced by
    `1 - InnerSIoU`. Everything else (VFL classification loss, L1 term,
    Hungarian matching, auxiliary and denoising losses, loss gains) is
    inherited unchanged from `RTDETRDetectionLoss`.
    """

    def __init__(self, *args, inner_ratio=0.7, **kwargs):
        """
        Args:
            inner_ratio (float): Scale factor for Inner-IoU computation.
                ratio < 1: inner boxes (smaller) → more sensitive to boundary
                    alignment, accelerates high-IoU convergence.
                ratio > 1: outer boxes (larger) → softer gradient for low-IoU
                    predictions, accelerates low-IoU convergence.
                UAV-DETR default: 1.25.
            *args, **kwargs: Forwarded to `RTDETRDetectionLoss` unchanged.
        """
        super().__init__(*args, **kwargs)
        self.inner_ratio = inner_ratio

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        """L1 term unchanged + GIoU term replaced by Inner-SIoU."""
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=self.device)
            loss[name_giou] = torch.tensor(0., device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / len(gt_bboxes)

        loss[name_giou] = 1.0 - inner_siou(pred_bboxes, gt_bboxes, xywh=True, ratio=self.inner_ratio)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}