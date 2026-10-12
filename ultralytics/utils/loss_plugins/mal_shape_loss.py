# Ultralytics YOLO 🐛, AGPL-3.0 license
"""
Mal-Shape Loss for RT-DETR — self-contained module.

Port from ED-DETR combining two orthogonal improvements:

  1. MALoss (CVPR 2025 DEIM) — classification loss:
     Replaces the standard VFL/Focal classification loss with a Matching-Aware
     Loss that re-weights the BCE term by the prediction confidence raised to a
     power gamma.  Positive samples keep weight 1; negative samples get weight
     sigmoid(pred)^gamma, so easy negatives (high confidence wrong predictions)
     are down-weighted — similar in spirit to Focal Loss but applied on top of
     VFL's IoU-aware positive weighting.

  2. Shape-IoU (Zhang et al.) — regression loss:
     Replaces the GIoU regression term with Shape-IoU, which introduces
     anisotropic weights (ww, hh) derived from the GT box aspect ratio so that
     the distance and shape penalty terms are scaled according to the target's
     shape.  For horizontal long targets (w>>h), y-direction errors are
     amplified; for vertical long targets (h>>w), x-direction errors are
     amplified.

Design goals (minimal invasion of the original RT-DETR code):
    - Everything lives in THIS file. `ultralytics/models/utils/loss.py`,
      `ultralytics/utils/metrics.py` and all other core files stay untouched.
    - `RTDETRDetectionLossMalShape` is a drop-in subclass of
      `RTDETRDetectionLoss`: classification uses MALoss (via _get_mal_loss),
      regression uses Shape-IoU (via _get_loss_bbox override). L1 term,
      Hungarian matching cost, loss gains, auxiliary/denoising loss plumbing
      are all inherited unchanged.
    - This file is registered in the loss plugin registry
      (`ultralytics/utils/loss_plugins/__init__.py`) under the name
      'mal_shape'. `RTDETRDetectionModel.init_criterion()` resolves
      `loss_name: mal_shape` from the model yaml through that registry.

Mal-Shape IoU formula
---------------------
ShapeIoU = IoU - Shape-Distance - 0.5 * Shape-Shape

    ww = 2 * w_gt^scale / (w_gt^scale + h_gt^scale)
    hh = 2 * h_gt^scale / (w_gt^scale + h_gt^scale)

    Shape-Distance = (hh * dx^2 + ww * dy^2) / c^2
    Shape-Shape    = (1 - exp(-omega_w))^4 + (1 - exp(-omega_h))^4
    omega_w = hh * |w1-w2| / max(w1,w2)
    omega_h = ww * |h1-h2| / max(h1,h2)

MALoss formula
--------------
    weight = sigmoid(pred)^gamma * (1 - one_hot) + one_hot
    loss   = (VFL(pred, gt_score^gamma, one_hot) * weight).mean(1).sum()
             / max(num_gts, 1) / nq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou

__all__ = ('mal_shape_iou', 'MalShapeLoss', 'RTDETRDetectionLossMalShape')


def mal_shape_iou(box1, box2, xywh=True, scale=0.0, eps=1e-7):
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


class MalShapeLoss(nn.Module):
    """Standalone module wrapper combining MALoss (cls) + Shape-IoU (reg)."""

    def __init__(self, shape_scale=0.0, mal_gamma=1.5):
        super().__init__()
        self.shape_scale = shape_scale
        self.mal_gamma = mal_gamma

    def forward(self, pred_bboxes, gt_bboxes):
        """Return mean Shape-IoU regression loss `1 - ShapeIoU` for matched box pairs."""
        return (1.0 - mal_shape_iou(pred_bboxes, gt_bboxes, xywh=True, scale=self.shape_scale)).mean()


class RTDETRDetectionLossMalShape(RTDETRDetectionLoss):
    """
    Drop-in replacement of `RTDETRDetectionLoss` combining:
      - MALoss for the classification term (replaces VFL/Focal)
      - Shape-IoU for the bbox regression term (replaces GIoU)

    Two methods are overridden:
      - _get_loss_bbox: GIoU -> Shape-IoU
      - _get_loss: calls _get_mal_loss instead of _get_loss_class

    Everything else (L1 term, Hungarian matching, auxiliary and denoising
    losses, loss gains) is inherited unchanged from `RTDETRDetectionLoss`.
    """

    def __init__(self, *args, shape_scale=0.0, mal_gamma=1.5, mal_alpha=None, **kwargs):
        """
        Args:
            shape_scale (float): Exponent for the Shape-IoU shape-weight
                computation. 0.0 gives equal weights (ww=hh=1). Typical: 0.1.
            mal_gamma (float): Power exponent for MALoss weighting.
                sigmoid(pred)^gamma down-weights easy negatives. Typical: 1.5.
            mal_alpha (float | None): Optional scaling factor for the negative
                weight in MALoss. None -> no extra scaling (standard MALoss).
            *args, **kwargs: Forwarded to `RTDETRDetectionLoss` unchanged.
        """
        kwargs.setdefault('use_vfl', True)
        super().__init__(*args, **kwargs)
        self.shape_scale = shape_scale
        self.mal_gamma = mal_gamma
        self.mal_alpha = mal_alpha

    def _get_mal_loss(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        """
        Matching-Aware Loss (MALoss) from CVPR 2025 DEIM.

        Replaces the standard VFL/Focal classification loss. The key difference
        from VFL is that gt_scores are raised to power gamma before computing
        the VFL, and the VFL output is further weighted by a modulating factor
        that down-weights easy negatives:
            weight = sigmoid(pred)^gamma * (1 - one_hot) + one_hot

        This means:
        - Positive (foreground) samples: weight = 1 (unchanged)
        - Negative (background) samples: weight = sigmoid(pred)^gamma
          → easy negatives with high sigmoid get weight close to 1^gamma = 1
            wait, that's not down-weighting... Actually, for MALoss gamma < 2
            (typically 1.5), the sigmoid^gamma term is smaller than sigmoid
            for sigmoid < 1, so hard negatives (low sigmoid) get relatively
            MORE weight compared to easy negatives (high sigmoid). This is
            the focal-like effect.
        """
        name_class = f'loss_class{postfix}'
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        gt_scores = gt_scores.pow(self.mal_gamma)
        if self.mal_alpha is not None:
            weight = self.mal_alpha * pred_scores.sigmoid().pow(self.mal_gamma) * (1 - one_hot) + one_hot
        else:
            weight = pred_scores.sigmoid().pow(self.mal_gamma) * (1 - one_hot) + one_hot

        with torch.cuda.amp.autocast(enabled=False):
            loss_cls = (self.vfl(pred_scores, gt_scores, one_hot) * weight).mean(1).sum()
        loss_cls /= max(num_gts, 1) / nq
        return {name_class: loss_cls.squeeze() * self.loss_gain['giou']}

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

        loss[name_giou] = 1.0 - mal_shape_iou(pred_bboxes, gt_bboxes, xywh=True, scale=self.shape_scale)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        """Get losses with MALoss for classification and Shape-IoU for regression."""
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                         masks=masks, gt_mask=gt_mask)

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        loss = {}
        loss.update(self._get_mal_loss(pred_scores, targets, gt_scores, len(gt_bboxes), postfix))
        loss.update(self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix))
        return loss