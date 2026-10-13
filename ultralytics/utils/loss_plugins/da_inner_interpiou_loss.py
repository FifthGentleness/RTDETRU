# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
DA-InterpIoU Loss for RT-DETR — self-contained module.
(Dynamic-Adaptive Inner auxiliary box + Dynamic InterpIoU main term)

Regression loss (the GIoU term of the bbox loss is replaced; L1, VFL,
Hungarian matching, auxiliary / denoising plumbing are inherited unchanged):

    Reference (main term): Liu & Watanabe, "InterpIoU: Robust bounding box
    regression loss within an interpolation-based IoU framework",
    Neurocomputing 666 (2026) 132230 / arXiv:2507.12420.
    Reference (auxiliary mechanism): Zhang et al., "Inner-IoU: More Effective
    Intersection over Union Loss with Auxiliary Bounding Box", arXiv 2311.02877
    — improved here by replacing the FIXED ratio with a per-sample schedule
    driven by the current IoU (detached, so the schedule itself does not
    back-propagate while the auxiliary IoU still does):

        r(IoU) = r_min + (1 - r_min) * IoU^gamma
        L = L_main + IoU - IoU_inner(r(IoU))

    Semantics (defaults r_min=0.75, gamma=1.0):
        low  IoU -> r close to r_min (strong guidance, shrunk aux boxes)
        high IoU -> r close to 1     (pure fine boundary refinement)
        one parameter set covers the whole training, no stage switching.

Usage (model yaml):
    loss_name: da_inner_interpiou
    loss_params:
        da_r_min: 0.75       # ablation grid: {0.70, 0.75, 0.80}
        da_gamma: 1.0        # ablation grid: {1.0, 2.0}
        dynamic_interp: true # false -> static alpha = interp_alpha (paper: 0.99)
"""

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('da_inner_interpiou_loss', 'DAInnerInterpIoULoss', 'RTDETRDetectionLossDAInnerInterpIoU')


def _iou_xywh(box1, box2, xywh=True, eps=1e-7):
    """Plain IoU between two sets of boxes. Returns shape (..., 1)."""
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    return inter / union


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


def da_inner_interpiou_loss(box1, box2, da_r_min=0.75, da_gamma=1.0,
                            dynamic_interp=True, interp_alpha=0.99, xywh=True, eps=1e-7):
    """Per-sample DA-InterpIoU regression loss. Shape (..., 1)."""
    iou = _iou_xywh(box1, box2, xywh=xywh, eps=eps)

    # main term: Dynamic InterpIoU
    alpha = (1.0 - iou).detach() if dynamic_interp else interp_alpha
    interp_box = alpha * box1 + (1.0 - alpha) * box2
    iou_interp = _iou_xywh(interp_box, box2, xywh=xywh, eps=eps)
    loss_main = (1.0 - iou) + (1.0 - iou_interp)

    # auxiliary term: Dynamic-Adaptive Inner-IoU
    # (schedule detached: r follows the current IoU but does not back-propagate)
    r = da_r_min + (1.0 - da_r_min) * iou.detach().pow(da_gamma)
    iou_inner = _inner_iou(box1, box2, r, xywh=xywh, eps=eps)
    return loss_main + iou - iou_inner


class DAInnerInterpIoULoss(torch.nn.Module):
    """Standalone wrapper (mean reduction)."""
    def __init__(self, da_r_min=0.75, da_gamma=1.0, dynamic_interp=True, interp_alpha=0.99):
        super().__init__()
        self.da_r_min = da_r_min
        self.da_gamma = da_gamma
        self.dynamic_interp = dynamic_interp
        self.interp_alpha = interp_alpha

    def forward(self, pred_bboxes, gt_bboxes):
        return da_inner_interpiou_loss(pred_bboxes, gt_bboxes, self.da_r_min, self.da_gamma,
                                       self.dynamic_interp, self.interp_alpha).mean()


class RTDETRDetectionLossDAInnerInterpIoU(RTDETRDetectionLoss):
    """
    Drop-in replacement of `RTDETRDetectionLoss`: only `_get_loss_bbox` is
    overridden — the GIoU term becomes DA-InterpIoU.
    """

    def __init__(self, *args, da_r_min=0.75, da_gamma=1.0, dynamic_interp=True, interp_alpha=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.da_r_min = da_r_min
        self.da_gamma = da_gamma
        self.dynamic_interp = dynamic_interp
        self.interp_alpha = interp_alpha

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

        # GIoU term -> DA-InterpIoU
        loss[name_giou] = da_inner_interpiou_loss(pred_bboxes, gt_bboxes,
                                                  da_r_min=self.da_r_min,
                                                  da_gamma=self.da_gamma,
                                                  dynamic_interp=self.dynamic_interp,
                                                  interp_alpha=self.interp_alpha,
                                                  xywh=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}
