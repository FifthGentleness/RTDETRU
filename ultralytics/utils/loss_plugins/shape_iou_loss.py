# Ultralytics YOLO 🐛, AGPL-3.0 license
"""
Shape-IoU Loss for RT-DETR — self-contained module.

Port of Shape-IoU from DCIDETR:
    Zhang et al., "Shape-IoU: Shape-Aware IoU for Bounding Box Regression",
    https://github.com/malaghoureuse/DCIDETR

Design goals (minimal invasion of the original RT-DETR code):
    - Everything lives in THIS file. `ultralytics/models/utils/loss.py`,
      `ultralytics/utils/metrics.py` and all other core files stay untouched.
    - `RTDETRDetectionLossShapeIoU` is a drop-in subclass of
      `RTDETRDetectionLoss`: only the GIoU term of the bbox regression loss
      is replaced by Shape-IoU. L1 term, VFL classification loss, Hungarian
      matching cost, loss gains, auxiliary/denoising loss plumbing are all
      inherited unchanged.
    - This file is registered in the loss plugin registry
      (`ultralytics/utils/loss_plugins/__init__.py`) under the name
      'shape_iou'. `RTDETRDetectionModel.init_criterion()` resolves
      `loss_name: shape_iou` from the model yaml through that registry.

Core idea
---------
Shape-IoU introduces anisotropic weights (ww, hh) derived from the GT box
aspect ratio, so that the distance and shape penalty terms are scaled
according to the target's shape:

    ww = 2 * w_gt^scale / (w_gt^scale + h_gt^scale)
    hh = 2 * h_gt^scale / (w_gt^scale + h_gt^scale)

Shape-Distance:  d = (hh * dx^2 + ww * dy^2) / c^2
Shape-Shape:     Ω = (1 - exp(-ω_w))^4 + (1 - exp(-ω_h))^4
    where ω_w = hh * |w1-w2| / max(w1,w2),  ω_h = ww * |h1-h2| / max(h1,h2)

ShapeIoU = IoU - Shape-Distance - 0.5 * Shape-Shape

For a horizontal long target (w >> h): ww is large, hh is small →
y-direction center offset and height difference get amplified.
For a vertical long target (h >> w): hh is large, ww is small →
x-direction center offset and width difference get amplified.
"""

import math

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('shape_iou', 'ShapeIoULoss', 'RTDETRDetectionLossShapeIoU')


def shape_iou(box1, box2, xywh=True, scale=0.0, eps=1e-7):
    """
    Shape-IoU between predicted and ground-truth boxes.

    Args:
        box1 (torch.Tensor): Predicted boxes, shape (..., 4).
        box2 (torch.Tensor): Ground-truth boxes, shape (..., 4).
        xywh (bool): If True, inputs are (cx, cy, w, h); if False, (x1, y1, x2, y2).
        scale (float): Exponent for the shape-weight computation. Controls how
            strongly the GT aspect ratio influences the anisotropic weights.
            0.0 gives equal weights (ww=hh=1), larger values amplify the
            aspect-ratio effect. Default 0.0, typical 0.1.
        eps (float): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): Shape-IoU values, shape (..., 1). Use `1 - ShapeIoU` as loss.
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

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
    hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
    center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    center_distance = hh * center_distance_x + ww * center_distance_y
    distance = center_distance / c2

    omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)

    return iou - distance - 0.5 * shape_cost


class ShapeIoULoss(torch.nn.Module):
    """Standalone module wrapper of the Shape-IoU bbox regression loss."""

    def __init__(self, scale=0.0):
        """Initialize with the shape scale exponent."""
        super().__init__()
        self.scale = scale

    def forward(self, pred_bboxes, gt_bboxes):
        """Return mean Shape-IoU regression loss `1 - ShapeIoU` for matched box pairs."""
        return (1.0 - shape_iou(pred_bboxes, gt_bboxes, xywh=True, scale=self.scale)).mean()


class RTDETRDetectionLossShapeIoU(RTDETRDetectionLoss):
    """
    Drop-in replacement of `RTDETRDetectionLoss` using Shape-IoU for the bbox
    regression term.

    Only `_get_loss_bbox` is overridden: the GIoU term is replaced by
    `1 - ShapeIoU`. Everything else (VFL classification loss, L1 term,
    Hungarian matching, auxiliary and denoising losses, loss gains) is
    inherited unchanged from `RTDETRDetectionLoss`.
    """

    def __init__(self, *args, shape_scale=0.0, **kwargs):
        """
        Args:
            shape_scale (float): Exponent for the shape-weight computation.
                0.0 gives equal weights (ww=hh=1). Typical values: 0.1.
                Larger values amplify the aspect-ratio effect of GT boxes.
            *args, **kwargs: Forwarded to `RTDETRDetectionLoss` unchanged.
        """
        super().__init__(*args, **kwargs)
        self.shape_scale = shape_scale

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        """L1 term unchanged + GIoU term replaced by Shape-IoU."""
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=self.device)
            loss[name_giou] = torch.tensor(0., device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / len(gt_bboxes)

        loss[name_giou] = 1.0 - shape_iou(pred_bboxes, gt_bboxes, xywh=True, scale=self.shape_scale)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}