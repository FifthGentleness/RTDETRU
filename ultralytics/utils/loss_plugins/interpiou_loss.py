# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Pure InterpIoU Loss for RT-DETR -- self-contained module (NO Inner-IoU term).

Regression loss (the GIoU term of the bbox loss is replaced; L1, VFL,
Hungarian matching, auxiliary / denoising plumbing are inherited unchanged):

    Reference: Liu & Watanabe, "InterpIoU: Robust bounding box regression loss
    within an interpolation-based IoU framework", Neurocomputing 666 (2026)
    132230. Official implementation: https://github.com/NuayHL/interpiou

Core idea
---------
Replace the hand-crafted geometric penalty terms (distance / aspect / shape)
with an interpolation-based IoU term. An interpolated box is built between the
prediction B and the ground-truth B_gt, and the final value is

    InterpIoU = IoU(B, B_gt) + IoU(B_i, B_gt) - 1

where B_i = (1 - alpha) * B + alpha * B_gt (affine in xyxy == affine in xywh).

Two variants (official defaults):

    InterpIoU   : fixed alpha = interp_coe (paper: 0.98)
    D-InterpIoU : per-sample dynamic alpha = clamp(1 - IoU, lv, hv)
                  (lv = 0.6, hv = 0.99)

    L = 1 - InterpIoU     (>= 0; the IoU_interp term keeps gradients alive
                           even for low-overlap / non-overlapping pairs)

Usage (model yaml):
    loss_name: interpiou
    loss_params:
        dynamic_interp: true   # false -> static InterpIoU (alpha = interp_coe)
        interp_coe: 0.98       # fixed coefficient (static variant)
        lv: 0.6                # D-InterpIoU lower bound
        hv: 0.99               # D-InterpIoU upper bound
"""

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('interpiou', 'InterpIoULoss', 'RTDETRDetectionLossInterpIoU')


def _iou_xyxy(b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, eps=1e-7):
    """Plain IoU from corner coordinates. Returns shape (..., 1)."""
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    return inter / union


def interpiou(box1, box2, interp_coe=0.98, dynamic_interp=False,
              lv=0.6, hv=0.99, xywh=True, eps=1e-7):
    """
    Pure InterpIoU / D-InterpIoU between predicted and ground-truth boxes.

    Args:
        box1 (torch.Tensor): Predicted boxes, shape (..., 4).
        box2 (torch.Tensor): Ground-truth boxes, shape (..., 4).
        interp_coe (float): Fixed interpolation coefficient in [0, 1] for the
            static variant. 0 keeps box1, 1 equals box2. Default 0.98.
        dynamic_interp (bool): If True, use D-InterpIoU with per-sample
            alpha = clamp(1 - IoU, lv, hv). Default False.
        lv (float): Lower bound of the dynamic coefficient. Default 0.6.
        hv (float): Upper bound of the dynamic coefficient. Default 0.99.
        xywh (bool): True if inputs are (cx, cy, w, h), False for (x1, y1, x2, y2).
        eps (float): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): InterpIoU values, shape (..., 1). Use `1 - interpiou` as loss.
    """
    if xywh:  # transform from (cx, cy, w, h) to corners
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # corners
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    # Standard IoU between prediction and target
    iou = _iou_xyxy(b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, eps=eps)

    # Interpolation coefficient: fixed for InterpIoU, dynamic for D-InterpIoU.
    # Detached so the dynamic alpha itself does not back-propagate.
    alpha = torch.clamp(1 - iou.detach(), min=lv, max=hv) if dynamic_interp else interp_coe

    # Interpolated box between box1 and box2 (affine in corners == affine in xywh)
    bi_x1 = (1 - alpha) * b1_x1 + alpha * b2_x1
    bi_y1 = (1 - alpha) * b1_y1 + alpha * b2_y1
    bi_x2 = (1 - alpha) * b1_x2 + alpha * b2_x2
    bi_y2 = (1 - alpha) * b1_y2 + alpha * b2_y2

    # IoU between the interpolated box and the ground-truth box
    iou_i = _iou_xyxy(bi_x1, bi_y1, bi_x2, bi_y2, b2_x1, b2_y1, b2_x2, b2_y2, eps=eps)

    return iou + iou_i - 1


class InterpIoULoss(torch.nn.Module):
    """Standalone wrapper: mean of `1 - InterpIoU` over matched box pairs."""

    def __init__(self, dynamic_interp=True, interp_coe=0.98, lv=0.6, hv=0.99):
        super().__init__()
        self.dynamic_interp = dynamic_interp
        self.interp_coe = interp_coe
        self.lv = lv
        self.hv = hv

    def forward(self, pred_bboxes, gt_bboxes):
        return (1.0 - interpiou(pred_bboxes, gt_bboxes,
                                interp_coe=self.interp_coe,
                                dynamic_interp=self.dynamic_interp,
                                lv=self.lv, hv=self.hv, xywh=True)).mean()


class RTDETRDetectionLossInterpIoU(RTDETRDetectionLoss):
    """
    Drop-in replacement of `RTDETRDetectionLoss`: only `_get_loss_bbox` is
    overridden -- the GIoU term becomes pure InterpIoU / D-InterpIoU
    (no Inner-IoU auxiliary term). Everything else is inherited unchanged.
    """

    def __init__(self, *args, dynamic_interp=True, interp_coe=0.98, lv=0.6, hv=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_interp = dynamic_interp
        self.interp_coe = interp_coe
        self.lv = lv
        self.hv = hv

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=self.device)
            loss[name_giou] = torch.tensor(0., device=self.device)
            return loss

        # L1 term -- inherited behaviour, unchanged.
        loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / len(gt_bboxes)

        # GIoU term -> 1 - InterpIoU (pure, no Inner-IoU)
        loss[name_giou] = 1.0 - interpiou(pred_bboxes, gt_bboxes,
                                          interp_coe=self.interp_coe,
                                          dynamic_interp=self.dynamic_interp,
                                          lv=self.lv, hv=self.hv, xywh=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}
