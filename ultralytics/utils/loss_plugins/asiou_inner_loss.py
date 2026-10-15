# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
ASIoU + Inner-IoU Loss for RT-DETR - self-contained module.

Third innovation point for rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7:
    ASIoU (Adaptive Synergistic IoU) + Inner-IoU -> AS-Inner-IoU

    ASIoU reference: Pan et al., "ASIoU: An Adaptive Synergistic
    Intersection over Union Loss for Bounding Box Regression",
    Pattern Recognition Letters, 2026.
        https://doi.org/10.1016/j.patrec.2026.07.010

    Inner-IoU reference: Zhang et al., "Inner-IoU: A More Efficient IoU
    Loss for Bounding Box Regression".

Design goals (same minimal-invasion philosophy as the other loss plugins):
    - Everything lives in THIS file. Core RT-DETR files stay untouched.
    - `RTDETRDetectionLossASInnerIoU` is a drop-in subclass of
      `RTDETRDetectionLoss`: only the GIoU term of the bbox regression loss
      is replaced by AS-Inner-IoU. L1 term, VFL classification loss,
      Hungarian matching, auxiliary/denoising loss plumbing and loss gains
      are all inherited unchanged.
    - Registered in the loss plugin registry
      (`ultralytics/utils/loss_plugins/__init__.py`) under 'asiou_inner'.

Core idea
---------
1. Inner-IoU: Scale both predicted and GT boxes by a ratio (center-fixed)
   before computing IoU.
   ratio < 1 -> inner (smaller) boxes: more sensitive to boundary
   alignment, faster high-IoU convergence.
   ratio > 1 -> outer (larger) boxes: softer gradients for low-IoU
   predictions, faster early-stage convergence. UAV-DETR default: 1.25.

2. ASIoU keeps three orthogonal terms, but now evaluates them around the
   Inner-IoU:
   a) Scale-adaptive distance penalty: center-distance cost weighted by
      axis coefficients derived from the GT aspect ratio and GT area
      (adaptive_scale = scale * (1 + log(area/256))).
   b) Shape-aware penalty: aspect-ratio-weighted width/height mismatch
      costs (omega_w, omega_h -> (1 - exp(-omega))^4).
   c) Quality-driven modulation: for high-IoU predictions
      (balance_factor -> 0) the extra penalties fade out so the loss
      converges on pure IoU; for low-IoU predictions (balance_factor
      -> gamma) the scale-adaptive distance + shape penalties dominate.

AS-Inner-IoU formula:
    InnerIoU = inter_inner / union_inner          (ratio-scaled boxes)
    ASInnerIoU = InnerIoU - balance_factor * (distance_cost + shape_cost)
    loss = 1 - clamp(ASInnerIoU, -0.3, 1.0)
"""

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('asiou_inner', 'ASInnerIoULoss', 'RTDETRDetectionLossASInnerIoU')


def _decode_xywh(box, eps=1e-7):
    """Decode (cx, cy, w, h) chunks to corners + sizes, clamped positive."""
    x, y, w, h = box.chunk(4, -1)
    w = w.clamp(min=eps)
    h = h.clamp(min=eps)
    return x, y, w, h, x - w / 2, x + w / 2, y - h / 2, y + h / 2


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


def asiou_inner(box1, box2, ratio=1.25, scale=0.3, gamma=0.7, eps=1e-7):
    """
    AS-Inner-IoU between predicted and ground-truth boxes.

    ASIoU with its plain-IoU term replaced by Inner-IoU.

    Args:
        box1 (torch.Tensor): Predicted boxes (cx, cy, w, h), shape (..., 4).
        box2 (torch.Tensor): Ground-truth boxes (cx, cy, w, h), shape (..., 4).
        ratio (float): Inner-IoU scale factor (<1 inner, >1 outer;
            UAV-DETR default 1.25).
        scale (float): ASIoU scale-adaptive coefficient (paper default 0.3).
        gamma (float): ASIoU quality-modulation coefficient (paper default 0.7).
        eps (float): Numerical stability.

    Returns:
        (torch.Tensor): AS-Inner-IoU values, shape (..., 1).
            Use `1 - clamp(asiou_inner, -0.3, 1.0)` as the regression loss.
    """
    # 1. Decode boxes
    x1, y1, w1, h1, b1_x1, b1_x2, b1_y1, b1_y2 = _decode_xywh(box1, eps)
    x2, y2, w2, h2, b2_x1, b2_x2, b2_y1, b2_y2 = _decode_xywh(box2, eps)

    # 2. Inner-IoU (replaces the plain IoU term of ASIoU)
    iou = _get_inner_iou(box1, box2, eps=eps, ratio=ratio)

    # 3. Scale-adaptive axis weights (from GT geometry, no gradient)
    with torch.no_grad():
        area = (w2 * h2).clamp(min=32, max=4096)
        adaptive_scale = scale * (1 + torch.log(area / 256.0))

        w_pow = torch.pow(w2, adaptive_scale)
        h_pow = torch.pow(h2, adaptive_scale)
        ww = 2 * w_pow / (w_pow + h_pow + eps)
        hh = 2 * h_pow / (w_pow + h_pow + eps)

    # 4. Scale-adaptive distance cost
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    dx = (x2 - x1) * torch.sqrt(ww)   # axis weights exactly as in the original ASIoU code
    dy = (y2 - y1) * torch.sqrt(hh)
    distance_cost = (dx.pow(2) + dy.pow(2)) / c2

    # 5. Shape-aware penalty
    omega_w = hh * torch.abs(w1 - w2) / torch.maximum(w1, w2)
    omega_h = ww * torch.abs(h1 - h2) / torch.maximum(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-omega_w), 4) + \
                 torch.pow(1 - torch.exp(-omega_h), 4)

    # 6. Quality-driven modulation (high-IoU -> penalties fade out)
    with torch.no_grad():
        iou_ratio = iou.detach().clamp(0.1, 0.9)
        balance_factor = gamma * (1 - torch.exp(-5 * (1 - iou_ratio)))

    # 7. Final AS-Inner-IoU
    return iou - balance_factor * (distance_cost + shape_cost)


class ASInnerIoULoss(torch.nn.Module):
    """Standalone module wrapper of the ASIoU + Inner-IoU regression loss."""

    def __init__(self, ratio=1.25, scale=0.3, gamma=0.7):
        """Initialize with Inner-IoU ratio and ASIoU hyperparameters."""
        super().__init__()
        self.ratio = ratio
        self.scale = scale
        self.gamma = gamma

    def forward(self, pred_bboxes, gt_bboxes):
        """Return mean AS-Inner-IoU regression loss `1 - clamp(ASInnerIoU)`."""
        enhanced = asiou_inner(
            pred_bboxes, gt_bboxes,
            ratio=self.ratio, scale=self.scale, gamma=self.gamma)
        return (1.0 - enhanced.clamp(-0.3, 1.0)).mean()


class RTDETRDetectionLossASInnerIoU(RTDETRDetectionLoss):
    """
    Drop-in replacement of `RTDETRDetectionLoss` using ASIoU + Inner-IoU for
    the bbox regression term.

    Only `_get_loss_bbox` is overridden: the GIoU term is replaced by
    `1 - clamp(ASInnerIoU)`. Everything else (VFL classification loss, L1
    term, Hungarian matching, auxiliary and denoising losses, loss gains)
    is inherited unchanged from `RTDETRDetectionLoss`.
    """

    def __init__(self, *args, inner_ratio=1.25, scale=0.3, gamma=0.7, **kwargs):
        """
        Args:
            inner_ratio (float): Inner-IoU scale factor.
                ratio < 1: inner boxes -> boundary-alignment sensitive,
                    accelerates high-IoU convergence.
                ratio > 1: outer boxes -> softer gradients for low-IoU
                    predictions, accelerates early-stage convergence.
                UAV-DETR default: 1.25.
            scale (float): ASIoU scale-adaptive coefficient (default 0.3).
            gamma (float): ASIoU quality-modulation coefficient (default 0.7).
            *args, **kwargs: Forwarded to `RTDETRDetectionLoss` unchanged.
        """
        super().__init__(*args, **kwargs)
        self.inner_ratio = inner_ratio
        self.scale = scale
        self.gamma = gamma

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        """L1 term unchanged + GIoU term replaced by ASIoU + Inner-IoU."""
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=self.device)
            loss[name_giou] = torch.tensor(0., device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / len(gt_bboxes)

        enhanced = asiou_inner(
            pred_bboxes, gt_bboxes,
            ratio=self.inner_ratio, scale=self.scale, gamma=self.gamma)
        loss[name_giou] = 1.0 - enhanced.clamp(-0.3, 1.0)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}
