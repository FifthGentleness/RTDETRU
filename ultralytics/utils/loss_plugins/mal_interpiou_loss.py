# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Mal-InterpIoU Loss for RT-DETR — self-contained module.
(Matchability-Aware classification + Dynamic-InterpIoU regression, coupled)

Combines two orthogonal loss channels with an optional coupling mechanism:

  1. MALoss (CVPR 2025 DEIM) — classification channel:
     Replaces VFL. Positives keep weight 1; negatives are weighted by
     sigmoid(pred)^gamma; soft targets are the localization quality^gamma.

  2. Dynamic InterpIoU (Liu & Watanabe, Neurocomputing 2026) — regression
     channel, replaces the GIoU term:
         alpha_i = 1 - IoU_i   (per-sample, detached)
         B_a     = alpha * B + (1 - alpha) * B_gt
         L_main  = (1 - IoU(B, B_gt)) + (1 - IoU(B_a, B_gt))

  3. Novel coupling — interpolation-refined matchability target:
     The soft classification target of MAL is the IoU of the matched box.
     When `couple_target=True`, the target becomes a blend of the plain IoU
     and the InterpIoU-refined quality q = IoU(B_a, B_gt) (with the same
     dynamic alpha), i.e. the classification channel is supervised by the
     SAME refined localization quality that the regression channel optimizes:

         q     = IoU(B_a, B_gt),  alpha = 1 - IoU   (both detached)
         score = couple_beta * IoU + (1 - couple_beta) * q

     q >= IoU and is monotone: low-IoU matches are rewarded by the quality of
     their interpolation path toward the GT, giving the matcher a smoother,
     small-object-friendly target. `couple_target=False` reduces to plain
     (DEIM-style) MAL + D-InterpIoU.

Design goals (same as other plugins):
    - Everything lives in THIS file; core ultralytics files stay untouched.
    - L1 term, Hungarian matching cost, loss gains, auxiliary/denoising
      plumbing are inherited unchanged.

Usage (model yaml):
    loss_name: mal_interpiou
    loss_params:
        mal_gamma: 2.0        # DEIM default; ablation {1.5, 2.0}
        mal_alpha: null       # optional negative-weight scale (None = off)
        couple_target: true   # interpolation-refined matchability target
        couple_beta: 0.5      # 0 = pure refined q, 1 = plain IoU (DEIM-style)
        dynamic_interp: true  # D-InterpIoU main term
"""

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou

__all__ = ('mal_interpiou_loss', 'RTDETRDetectionLossMalInterpIoU')


# --------------------------------------------------------------------------- #
# IoU helpers
# --------------------------------------------------------------------------- #
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


def _refined_quality(box1, box2, dynamic_interp=True, interp_alpha=0.99, xywh=True, eps=1e-7):
    """Interpolation-refined localization quality IoU(B_a, B_gt), detached inputs."""
    iou = _iou_xywh(box1, box2, xywh=xywh, eps=eps)
    alpha = (1.0 - iou).detach() if dynamic_interp else interp_alpha
    interp_box = alpha * box1 + (1.0 - alpha) * box2
    return _iou_xywh(interp_box, box2, xywh=xywh, eps=eps)


def mal_interpiou_loss(box1, box2, dynamic_interp=True, interp_alpha=0.99, xywh=True, eps=1e-7):
    """Per-sample Dynamic-InterpIoU regression loss (main term only). Shape (..., 1)."""
    iou = _iou_xywh(box1, box2, xywh=xywh, eps=eps)
    alpha = (1.0 - iou).detach() if dynamic_interp else interp_alpha
    interp_box = alpha * box1 + (1.0 - alpha) * box2
    iou_interp = _iou_xywh(interp_box, box2, xywh=xywh, eps=eps)
    return (1.0 - iou) + (1.0 - iou_interp)


# --------------------------------------------------------------------------- #
# RT-DETR loss plugin
# --------------------------------------------------------------------------- #
class RTDETRDetectionLossMalInterpIoU(RTDETRDetectionLoss):
    """
    Drop-in replacement of `RTDETRDetectionLoss`:
      - classification: MALoss (DEIM, CVPR 2025)
      - regression GIoU term: Dynamic InterpIoU
      - optional coupling: MAL soft targets = interpolation-refined quality
    NOTE: unlike the pure-regression InterpIoU plugins, this one also changes
    the classification loss (that is what MAL is).
    """

    def __init__(self, *args,
                 mal_gamma=2.0,
                 mal_alpha=None,
                 couple_target=True,
                 couple_beta=0.5,
                 dynamic_interp=True,
                 interp_alpha=0.99,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mal_gamma = mal_gamma
        self.mal_alpha = mal_alpha
        self.couple_target = couple_target
        self.couple_beta = couple_beta
        self.dynamic_interp = dynamic_interp
        self.interp_alpha = interp_alpha

    def _get_mal_loss(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        """MALoss (DEIM, CVPR 2025): VFL with soft targets^gamma and
        sigmoid(pred)^gamma negative weighting."""
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
        # DEIM-style: classification gain follows the giou gain (see mal_shape plugin)
        return {name_class: loss_cls.squeeze() * self.loss_gain['giou']}

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        """L1 term unchanged + GIoU term replaced by Dynamic InterpIoU."""
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=self.device)
            loss[name_giou] = torch.tensor(0., device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / len(gt_bboxes)

        loss[name_giou] = mal_interpiou_loss(pred_bboxes, gt_bboxes,
                                             dynamic_interp=self.dynamic_interp,
                                             interp_alpha=self.interp_alpha,
                                             xywh=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        """Get losses with MAL classification and D-InterpIoU regression."""
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                         masks=masks, gt_mask=gt_mask)

        idx, gt_idx = self._get_index(match_indices)
        matched_pred, matched_gt = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(matched_gt):
            plain_iou = bbox_iou(matched_pred.detach(), matched_gt, xywh=True).squeeze(-1)
            if self.couple_target:
                q = _refined_quality(matched_pred.detach(), matched_gt,
                                     dynamic_interp=self.dynamic_interp,
                                     interp_alpha=self.interp_alpha).squeeze(-1)
                gt_scores[idx] = self.couple_beta * plain_iou + (1.0 - self.couple_beta) * q
            else:
                gt_scores[idx] = plain_iou

        loss = {}
        loss.update(self._get_mal_loss(pred_scores, targets, gt_scores, len(matched_gt), postfix))
        loss.update(self._get_loss_bbox(matched_pred, matched_gt, postfix))
        return loss

