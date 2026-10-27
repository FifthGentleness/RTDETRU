# DSAW-v2: Dynamic Scale-Aware supervision Weighting, version 2.
#
# Clean loss-only ablation. This plugin intentionally does NOT change the
# network architecture. It can be paired with either the standard AIFI or any
# AIFI replacement; here the provided YAML uses the standard AIFI so that
# DSAW-v2 remains a separate loss-only innovation point.
#
# Key changes vs. DSAW:
# 1. Scale factor is sharpened:
#        lam_v2 = clip(1 - sqrt(w * h) / tau, 0, 1) ** lam_pow
#    With lam_pow > 1, supervision focuses more on truly tiny objects.
#
# 2. Geometry suppression is also annealed by the dynamic factor:
#        w_cls = 1 + beta * lam_v2 * eta
#        w_geo = 1 - alpha * lam_v2 * eta
#    This restores geometric supervision as matching quality improves.
#
# Fallbacks:
# - lam_pow=1 and dynamic_geo=False reproduces original DSAW.
# - beta=0 and alpha=0 reproduce the baseline RT-DETR VFL loss.

import torch

from .dsaw_loss import RTDETRDetectionLossDSAW
from ultralytics.utils.metrics import bbox_iou

__all__ = ('RTDETRDetectionLossDSAWV2',)


class RTDETRDetectionLossDSAWV2(RTDETRDetectionLossDSAW):
    """Dynamic Scale-Aware supervision Weighting v2.

    Args:
        beta (float): Positive-sample classification boost strength.
        alpha (float): Geometry softening strength.
        tau (float): Normalized scale threshold; larger boxes have lam=0.
        eta_momentum (float): EMA momentum for matched-IoU quality.
        q_ema_init (float): Initial matched-IoU EMA value.
        dynamic_eta (bool): If true, eta=1-q_ema decays during training.
        lam_pow (float): Power applied to the scale factor. 2.0 focuses
            supervision on truly tiny objects.
        dynamic_geo (bool): If true, geometry softening is also annealed by
            eta. If false, this becomes a sharper-scale DSAW variant.
    """

    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, beta=0.5, alpha=0.25, tau=0.03125,
                 eta_momentum=0.999, q_ema_init=0.5, dynamic_eta=True,
                 lam_pow=2.0, dynamic_geo=True, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, beta=beta,
                         alpha=alpha, tau=tau, eta_momentum=eta_momentum,
                         q_ema_init=q_ema_init, dynamic_eta=dynamic_eta, **kw)
        self.lam_pow = float(lam_pow)
        self.dynamic_geo = bool(dynamic_geo)

    def _focused_lam(self, s):
        """Compute the sharpened, GT-side scale factor."""
        lam = (1.0 - s / self.tau).clamp(0.0, 1.0) if self.tau > 0 \
            else torch.zeros_like(s)
        if self.lam_pow != 1.0:
            lam = lam.pow(self.lam_pow)
        return lam

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        """Same matcher/target flow as DSAW, with focused scale and dynamic geometry."""
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes,
                                         gt_cls, gt_groups,
                                         masks=masks, gt_mask=gt_mask)

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device,
                             dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        w_cls_map = torch.ones([bs, nq], device=pred_scores.device)
        w_geo = None
        if len(gt_bboxes):
            iou = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)
            gt_scores[idx] = iou

            # GT-side scale factor. No gradient path through the size prior.
            s = (gt_bboxes[:, 2] * gt_bboxes[:, 3]).clamp(min=0).sqrt()
            lam = self._focused_lam(s)

            # Shared dynamic training-state factor.
            m = self.eta_momentum
            self.q_ema = m * self.q_ema + (1 - m) * float(iou.mean().detach())
            eta = self.eta

            # Positive-class boost and geometry softening both anneal.
            w_cls_map[idx] = 1.0 + self.beta * lam * eta
            geo_eta = eta if self.dynamic_geo else 1.0
            w_geo = 1.0 - self.alpha * lam * geo_eta

        loss = {}
        loss.update(self._get_loss_class_w(pred_scores, targets, gt_scores,
                                           len(gt_bboxes), w_cls_map, postfix))
        loss.update(self._get_loss_bbox_w(pred_bboxes, gt_bboxes, w_geo, postfix))
        return loss