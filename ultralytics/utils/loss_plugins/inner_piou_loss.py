# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Inner-PIoU Loss for RT-DETR — self-contained module.
(Powerful-IoU + Inner-IoU auxiliary box, faithful port of the official PIoU)

Regression loss (the GIoU term of the bbox loss is replaced; L1, VFL,
Hungarian matching, auxiliary / denoising plumbing are inherited unchanged):

    Reference: Zhang & Zhang et al., "Powerful-IoU: More straightforward and
    faster bounding box regression loss with a nonmonotonic focusing
    mechanism", Neural Networks (2024). Official code:
    othercode/Powerful-IoU/YOLOv8/piou.py
    Auxiliary term: Zhang et al., "Inner-IoU: More Effective Intersection
    over Union Loss with Auxiliary Bounding Box", arXiv 2311.02877.

    Edge-deviation term (official formulation):
        dw1 = |b1_left  - b2_left |,  dw2 = |b1_right - b2_right|
        dh1 = |b1_top   - b2_top  |,  dh2 = |b1_bottom- b2_bottom|
        P   = ((dw1 + dw2) / |w_gt| + (dh1 + dh2) / |h_gt|) / 4

    PIoU v1 ('inner_piou'):
        L_main = (1 - IoU) + (1 - exp(-P^2))
    PIoU v2 ('inner_piou2', non-monotonic focusing):
        x = exp(-P) * Lambda
        L_main = 3 * x * exp(-x^2) * L_v1

    Inner-PIoU combo (this file):
        L = L_main + IoU - IoU_inner(ratio)

        ratio < 1 : shrunk inner boxes  -> sharper boundary supervision
        ratio = 1 : reduces to plain PIoU
        ratio > 1 : enlarged outer boxes -> softer gradients for low-IoU pairs

Usage (model yaml):
    loss_name: inner_piou          # or inner_piou2
    loss_params:
        inner_ratio: 0.8           # ablation grid: {0.7, 0.8, 1.0, 1.25}
        piou_lambda: 1.3           # v2 only, official default
"""

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('piou_loss',
           'InnerPIoULoss',
           'RTDETRDetectionLossInnerPIoU',
           'RTDETRDetectionLossInnerPIoU2')


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #
def _edges_xywh(box, xywh=True, eps=1e-7):
    """Return corner coordinates (x1, y1, x2, y2) from xywh or xyxy boxes."""
    if xywh:
        (x, y, w, h) = box.chunk(4, -1)
        w_, h_ = w / 2, h / 2
        return x - w_, y - h_, x + w_, y + h_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box.chunk(4, -1)
        return b1_x1, b1_y1, b1_x2, b1_y2


def _piou_core(box1, box2, xywh=True, eps=1e-7):
    """
    Plain-IoU and edge-deviation term P from the official PIoU code.

    Returns:
        (iou, P): both shape (..., 1).
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = _edges_xywh(box1, xywh=xywh, eps=eps)
    b2_x1, b2_y1, b2_x2, b2_y2 = _edges_xywh(box2, xywh=xywh, eps=eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter + eps
    iou = inter / union

    # official PIoU edge-deviation term
    dw1 = torch.abs(b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1))   # |left1 - left2|
    dw2 = torch.abs(b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1))   # |right1 - right2|
    dh1 = torch.abs(b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1))   # |top1 - top2|
    dh2 = torch.abs(b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1))   # |bottom1 - bottom2|
    w2 = torch.abs(b2_x2 - b2_x1) + eps
    h2 = torch.abs(b2_y2 - b2_y1) + eps
    P = ((dw1 + dw2) / w2 + (dh1 + dh2) / h2) / 4
    return iou, P


def _inner_iou(box1, box2, ratio, xywh=True, eps=1e-7):
    """Inner-IoU on center-fixed ratio-scaled auxiliary boxes. ratio: float or tensor (..., 1)."""
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        x1, y1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        x2, y2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    r = ratio
    b1_x1, b1_x2 = x1 - (w1 * r) / 2, x1 + (w1 * r) / 2
    b1_y1, b1_y2 = y1 - (h1 * r) / 2, y1 + (h1 * r) / 2
    b2_x1, b2_x2 = x2 - (w2 * r) / 2, x2 + (w2 * r) / 2
    b2_y1, b2_y2 = y2 - (h2 * r) / 2, y2 + (h2 * r) / 2

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 * r * r + w2 * h2 * r * r - inter + eps
    return inter / union


# --------------------------------------------------------------------------- #
# Unified Inner-PIoU loss
# --------------------------------------------------------------------------- #
def piou_loss(box1, box2, variant='piou', inner_ratio=0.8, piou_lambda=1.3, xywh=True, eps=1e-7):
    """
    Per-sample Inner-PIoU regression loss. Shape (..., 1).

    Args:
        variant (str): 'piou' (Powerful-IoU v1) | 'piou2' (v2, non-monotonic focusing).
        inner_ratio (float): Inner-IoU auxiliary-box scale factor.
        piou_lambda (float): Non-monotonic focusing coefficient (v2 only, official 1.3).
    """
    iou, P = _piou_core(box1, box2, xywh=xywh, eps=eps)

    # official PIoU main loss
    l_v1 = 1.0 - iou - torch.exp(-P ** 2) + 1.0        # (1 - IoU) + (1 - exp(-P^2))
    if variant == 'piou':
        loss_main = l_v1
    elif variant == 'piou2':
        x = torch.exp(-P) * piou_lambda
        loss_main = 3.0 * x * torch.exp(-x ** 2) * l_v1
    else:
        raise ValueError(f"Unknown PIoU variant: '{variant}'")

    # auxiliary term: Inner-IoU residual
    iou_inner = _inner_iou(box1, box2, inner_ratio, xywh=xywh, eps=eps)
    return loss_main + iou - iou_inner


class InnerPIoULoss(torch.nn.Module):
    """Standalone wrapper (mean reduction)."""
    def __init__(self, variant='piou', inner_ratio=0.8, piou_lambda=1.3):
        super().__init__()
        self.variant = variant
        self.inner_ratio = inner_ratio
        self.piou_lambda = piou_lambda

    def forward(self, pred_bboxes, gt_bboxes):
        return piou_loss(pred_bboxes, gt_bboxes, self.variant, self.inner_ratio, self.piou_lambda).mean()


# --------------------------------------------------------------------------- #
# RT-DETR loss plugins (only `_get_loss_bbox` is overridden)
# --------------------------------------------------------------------------- #
class _RTDETRDetectionLossInnerPIoUBase(RTDETRDetectionLoss):
    """Shared plumbing: L1 term unchanged + GIoU term replaced by Inner-PIoU."""

    _variant = 'piou'

    def __init__(self, *args, inner_ratio=0.8, piou_lambda=1.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_ratio = inner_ratio
        self.piou_lambda = piou_lambda

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=self.device)
            loss[name_giou] = torch.tensor(0., device=self.device)
            return loss

        # L1 term — inherited behaviour, unchanged.
        loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / len(gt_bboxes)

        # GIoU term -> Inner-PIoU
        loss[name_giou] = piou_loss(pred_bboxes, gt_bboxes,
                                    variant=self._variant,
                                    inner_ratio=self.inner_ratio,
                                    piou_lambda=self.piou_lambda,
                                    xywh=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}


class RTDETRDetectionLossInnerPIoU(_RTDETRDetectionLossInnerPIoUBase):
    """Inner-PIoU v1: straightforward Powerful-IoU + Inner-IoU auxiliary box."""
    _variant = 'piou'


class RTDETRDetectionLossInnerPIoU2(_RTDETRDetectionLossInnerPIoUBase):
    """Inner-PIoU v2: non-monotonic focusing Powerful-IoU + Inner-IoU auxiliary box."""
    _variant = 'piou2'
