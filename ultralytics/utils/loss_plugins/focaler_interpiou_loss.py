# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Focaler-InterpIoU Loss for RT-DETR — self-contained module.

Regression loss (the GIoU term of the bbox loss is replaced; L1, VFL,
Hungarian matching, auxiliary / denoising plumbing are inherited unchanged):

    Reference (main term): Liu & Watanabe, "InterpIoU: Robust bounding box
    regression loss within an interpolation-based IoU framework",
    Neurocomputing 666 (2026) 132230 / arXiv:2507.12420.
    Reference (auxiliary term): Zhang et al., "Focaler-IoU: More Focused
    Intersection over Union Loss", arXiv 2401.10525; applied on RT-DETR
    aerial small targets by CAEM-DETR (IEEE TGRS 2025).

    Dynamic InterpIoU main term:
        alpha_i = 1 - IoU_i                (per-sample, detached)
        B_a     = alpha * B + (1 - alpha) * B_gt
        L_main  = (1 - IoU(B, B_gt)) + (1 - IoU(B_a, B_gt))

    Focaler-InterpIoU (this file):
        IoU_f = clamp((IoU - d) / (u - d), 0, 1)
        L = L_main + IoU - IoU_f

    Interval semantics (small-object defaults d=0.35, u=0.80):
        IoU < d   : auxiliary gradient saturated (D-InterpIoU interpolated
                    term takes over the hardest, non-overlapping samples)
        d..u      : difficulty interval, auxiliary gradient linearly focused
        IoU > u   : auxiliary term ~ 0, does not disturb fine convergence

Usage (model yaml):
    loss_name: focaler_interpiou
    loss_params:
        focaler_d: 0.35      # ablation grid: {0.30, 0.35, 0.40}
        focaler_u: 0.80      # ablation grid: {0.75, 0.80, 0.85}
        dynamic_interp: true # false -> static alpha = interp_alpha (paper: 0.99)
"""

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('focaler_interpiou_loss', 'FocalerInterpIoULoss', 'RTDETRDetectionLossFocalerInterpIoU')


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


def focaler_interpiou_loss(box1, box2, focaler_d=0.35, focaler_u=0.80,
                           dynamic_interp=True, interp_alpha=0.99, xywh=True, eps=1e-7):
    """Per-sample Focaler-InterpIoU regression loss. Shape (..., 1)."""
    iou = _iou_xywh(box1, box2, xywh=xywh, eps=eps)

    # main term: Dynamic InterpIoU
    alpha = (1.0 - iou).detach() if dynamic_interp else interp_alpha
    interp_box = alpha * box1 + (1.0 - alpha) * box2
    iou_interp = _iou_xywh(interp_box, box2, xywh=xywh, eps=eps)
    loss_main = (1.0 - iou) + (1.0 - iou_interp)

    # auxiliary term: Focaler interval mapping
    iou_focaler = ((iou - focaler_d) / (focaler_u - focaler_d)).clamp(0.0, 1.0)
    return loss_main + iou - iou_focaler


class FocalerInterpIoULoss(torch.nn.Module):
    """Standalone wrapper (mean reduction)."""
    def __init__(self, focaler_d=0.35, focaler_u=0.80, dynamic_interp=True, interp_alpha=0.99):
        super().__init__()
        self.focaler_d = focaler_d
        self.focaler_u = focaler_u
        self.dynamic_interp = dynamic_interp
        self.interp_alpha = interp_alpha

    def forward(self, pred_bboxes, gt_bboxes):
        return focaler_interpiou_loss(pred_bboxes, gt_bboxes, self.focaler_d, self.focaler_u,
                                      self.dynamic_interp, self.interp_alpha).mean()


class RTDETRDetectionLossFocalerInterpIoU(RTDETRDetectionLoss):
    """
    Drop-in replacement of `RTDETRDetectionLoss`: only `_get_loss_bbox` is
    overridden — the GIoU term becomes Focaler-InterpIoU.
    """

    def __init__(self, *args, focaler_d=0.35, focaler_u=0.80, dynamic_interp=True, interp_alpha=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.focaler_d = focaler_d
        self.focaler_u = focaler_u
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

        # GIoU term -> Focaler-InterpIoU
        loss[name_giou] = focaler_interpiou_loss(pred_bboxes, gt_bboxes,
                                                 focaler_d=self.focaler_d,
                                                 focaler_u=self.focaler_u,
                                                 dynamic_interp=self.dynamic_interp,
                                                 interp_alpha=self.interp_alpha,
                                                 xywh=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}
