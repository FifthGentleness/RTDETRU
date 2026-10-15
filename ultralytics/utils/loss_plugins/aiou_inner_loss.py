# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
AIoU + Inner-IoU Loss for RT-DETR - self-contained module.

Third innovation point for rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7:
    AIoU (Adaptive IoU with Edge Alignment Penalty) + Inner-IoU -> A-Inner-IoU

    AIoU reference: "AIoU: An Adaptive Bounding Box Loss with an Edge
    Alignment Penalty" (ultralytics_aiou integration).

    Inner-IoU reference: Zhang et al., "Inner-IoU: A More Efficient IoU
    Loss for Bounding Box Regression".

Design goals (same minimal-invasion philosophy as the other loss plugins):
    - Everything lives in THIS file. Core RT-DETR files stay untouched.
    - `RTDETRDetectionLossAInnerIoU` is a drop-in subclass of
      `RTDETRDetectionLoss`: only the GIoU term of the bbox regression loss
      is replaced by A-Inner-IoU. L1 term, VFL classification loss,
      Hungarian matching, auxiliary/denoising loss plumbing and loss gains
      are all inherited unchanged.
    - Registered in the loss plugin registry
      (`ultralytics/utils/loss_plugins/__init__.py`) under 'aiou_inner'.

Core idea
---------
1. Inner-IoU: Scale both predicted and GT boxes by a ratio (center-fixed)
   before computing IoU.
   ratio < 1 -> inner (smaller) boxes: more sensitive to boundary
   alignment, faster high-IoU convergence.
   ratio > 1 -> outer (larger) boxes: softer gradients for low-IoU
   predictions, faster early-stage convergence. UAV-DETR default: 1.25.

2. AIoU keeps three orthogonal terms, now evaluated around Inner-IoU:
   a) Center-distance penalty rho2/c2 (DIoU-style).
   b) Aspect-ratio consistency v with adaptive weight alpha
      (CIoU-style, computed from Inner-IoU).
   c) Edge-alignment penalty e: mean relative Euclidean distance between
      the four corresponding edge midpoints of predicted and GT boxes,
      constrained to [0, 1] by atan; adaptive weight gamma.
   d) Adaptive weighting from the batch IoU distribution: median/quartile
      spread defines scale_factor; tanh splits lambda_alpha (high-IoU ->
      aspect-ratio penalty) vs lambda_gamma (low-IoU -> edge penalty).

A-Inner-IoU formula:
    AInnerIoU = InnerIoU - (rho2 / c2
                            + lambda_alpha * alpha * v
                            + lambda_gamma * gamma * e)
    loss = 1 - clamp(AInnerIoU, -0.3, 1.0)
"""

import math

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('aiou_inner', 'AInnerIoULoss', 'RTDETRDetectionLossAInnerIoU')


def _get_inner_iou(box1, box2, eps=1e-7, ratio=1.25):
    """
    Compute Inner-IoU on center-fixed ratio-scaled boxes (xywh input).

    ratio < 1: inner boxes (smaller) -> boundary-alignment sensitive.
    ratio > 1: outer boxes (larger)  -> softer gradients, low-IoU friendly.
    """
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1, h1 = w1.clamp(min=eps), h1.clamp(min=eps)
    w2, h2 = w2.clamp(min=eps), h2.clamp(min=eps)

    b1_x1, b1_x2 = x1 - (w1 * ratio) / 2, x1 + (w1 * ratio) / 2
    b1_y1, b1_y2 = y1 - (h1 * ratio) / 2, y1 + (h1 * ratio) / 2
    b2_x1, b2_x2 = x2 - (w2 * ratio) / 2, x2 + (w2 * ratio) / 2
    b2_y1, b2_y2 = y2 - (h2 * ratio) / 2, y2 + (h2 * ratio) / 2

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(min=0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(min=0)
    union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
    return (inter / union).clamp(min=0.0, max=1.0)  # guard against fp rounding above 1.0


def aiou_inner(box1, box2, ratio=1.25, eps=1e-7):
    """
    A-Inner-IoU between predicted and ground-truth boxes.

    AIoU with its plain-IoU term replaced by Inner-IoU.

    Args:
        box1 (torch.Tensor): Predicted boxes (cx, cy, w, h), shape (..., 4).
        box2 (torch.Tensor): Ground-truth boxes (cx, cy, w, h), shape (..., 4).
        ratio (float): Inner-IoU scale factor (<1 inner, >1 outer;
            UAV-DETR default 1.25).
        eps (float): Numerical stability.

    Returns:
        (torch.Tensor): A-Inner-IoU values, shape (..., 1).
            Use `1 - clamp(aiou_inner, -0.3, 1.0)` as the regression loss.
    """
    # 1. Decode boxes (edges/penalties are computed on the ORIGINAL boxes,
    #    exactly as in the original AIoU implementation)
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_

    w2c = w2.clamp(min=eps)
    h2c = h2.clamp(min=eps)

    # 2. Inner-IoU (replaces the plain IoU term of AIoU)
    iou = _get_inner_iou(box1, box2, eps=eps, ratio=ratio)

    # 3. Center-distance penalty (DIoU-style)
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4

    # 4. Aspect-ratio consistency (CIoU-style v + adaptive alpha)
    v = (4 / math.pi**2) * ((w2c / h2c).atan() - (w1 / h1.clamp(min=eps)).atan()).pow(2)

    # 5. Edge-alignment penalty: relative Euclidean distance between the
    #    four corresponding edge midpoints
    dist_left = (b2_x1 - b1_x1)**2 + ((b2_y1 + b2_y2) / 2 - (b1_y1 + b1_y2) / 2)**2
    dist_right = (b2_x2 - b1_x2)**2 + ((b2_y1 + b2_y2) / 2 - (b1_y1 + b1_y2) / 2)**2
    dist_top = ((b2_x1 + b2_x2) / 2 - (b1_x1 + b1_x2) / 2)**2 + (b2_y1 - b1_y1)**2
    dist_bottom = ((b2_x1 + b2_x2) / 2 - (b1_x1 + b1_x2) / 2)**2 + (b2_y2 - b1_y2)**2

    e = (dist_left / w2c.pow(2) + dist_right / w2c.pow(2) +
         dist_top / h2c.pow(2) + dist_bottom / h2c.pow(2)) / 4
    # atan constrains e in [0, 1]
    e = (2 / math.pi) * e.atan()

    # 6. Adaptive weights (no gradient, same as original AIoU)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
        gamma = e / (e - iou + (1 + eps))

        # Adaptive weighting from the batch IoU distribution
        iou_mean = iou.mean()
        # std() of a single element is NaN -> guard (e.g. one matched pair)
        iou_std = iou.std() if iou.numel() > 1 else torch.zeros_like(iou_mean)
        q1_iou = iou_mean - 0.6745 * iou_std   # approx 25th percentile
        q3_iou = iou_mean + 0.6745 * iou_std   # approx 75th percentile
        spread = q3_iou - q1_iou
        scale_factor = (1 - iou_mean) * (10 + spread)  # range in [0, 10]

        # tanh smoothing: lambda_gamma larger at low IoU (edge penalty),
        # lambda_alpha larger at high IoU (aspect-ratio penalty)
        lambda_gamma = torch.tanh(scale_factor * (0.5 - iou)).clamp(min=0)
        lambda_alpha = 1 - lambda_gamma

    # 7. Final A-Inner-IoU
    return iou - (rho2 / c2 + lambda_alpha * alpha * v + lambda_gamma * gamma * e)


class AInnerIoULoss(torch.nn.Module):
    """Standalone module wrapper of the AIoU + Inner-IoU regression loss."""

    def __init__(self, ratio=1.25):
        """Initialize with the Inner-IoU ratio."""
        super().__init__()
        self.ratio = ratio

    def forward(self, pred_bboxes, gt_bboxes):
        """Return mean A-Inner-IoU regression loss `1 - clamp(AInnerIoU)`."""
        enhanced = aiou_inner(pred_bboxes, gt_bboxes, ratio=self.ratio)
        return (1.0 - enhanced.clamp(-0.3, 1.0)).mean()


class RTDETRDetectionLossAInnerIoU(RTDETRDetectionLoss):
    """
    Drop-in replacement of `RTDETRDetectionLoss` using AIoU + Inner-IoU for
    the bbox regression term.

    Only `_get_loss_bbox` is overridden: the GIoU term is replaced by
    `1 - clamp(AInnerIoU)`. Everything else (VFL classification loss, L1
    term, Hungarian matching, auxiliary and denoising losses, loss gains)
    is inherited unchanged from `RTDETRDetectionLoss`.
    """

    def __init__(self, *args, inner_ratio=1.25, **kwargs):
        """
        Args:
            inner_ratio (float): Inner-IoU scale factor.
                ratio < 1: inner boxes -> boundary-alignment sensitive,
                    accelerates high-IoU convergence.
                ratio > 1: outer boxes -> softer gradients for low-IoU
                    predictions, accelerates early-stage convergence.
                UAV-DETR default: 1.25.
            *args, **kwargs: Forwarded to `RTDETRDetectionLoss` unchanged.
        """
        super().__init__(*args, **kwargs)
        self.inner_ratio = inner_ratio

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        """L1 term unchanged + GIoU term replaced by AIoU + Inner-IoU."""
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=self.device)
            loss[name_giou] = torch.tensor(0., device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / len(gt_bboxes)

        enhanced = aiou_inner(pred_bboxes, gt_bboxes, ratio=self.inner_ratio)
        loss[name_giou] = 1.0 - enhanced.clamp(-0.3, 1.0)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}
