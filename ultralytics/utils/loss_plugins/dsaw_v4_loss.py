# DSAW-v4: Segmented geometry + decoupled weights.
#
# Design goal: keep DSAW-v3's recall/mAP50 strength (strong gradients for
# marginal low-IoU boxes) while fixing its precision / mAP50-95 weaknesses
# (high-IoU gradient vacuum, inner-box "grow" bias, geometry softening).
#
# Inherits DSAW-v2 machinery unchanged:
#   lam  = clip(1 - sqrt(w*h)/tau, 0, 1) ** lam_pow      (GT-side, no grad)
#   eta  = 1 - q_ema                                     (dynamic quality gap)
#   w_cls = 1 + beta * lam * eta                         (cls positives boost)
#
# Geometry branch changes vs. v3:
#   1. Segmented blend (matched-IoU gated, detached):
#        mu  = sigmoid((IoU - blend_center) / blend_temp)
#        L_geo = mu * (1 - GIoU) + (1 - mu) * InnerInterpIoU
#      Low-IoU pairs  -> InnerInterpIoU (strong pull over the 0.5 threshold)
#      High-IoU pairs -> GIoU (clean refinement, no grow bias)
#   2. Decoupled softening:
#        L1  branch: x (1 - alpha_l1 * lam * geo_eta)   (keep the brake)
#        IoU branch: x (1 - alpha_iou * lam * geo_eta), default alpha_iou=0
#      Rationale: L1 has a constant gradient and needs the small-box brake;
#      IoU losses are already scale-adaptive (~1/2w) and softening them
#      twice suppresses high-IoU refinement.
#
# Fallbacks (degeneracy checks):
#   blend_center = 1.5, blend_temp = 0.05 -> mu ~ 0  -> pure v3 geometry
#   blend_center = 0.0                    -> mu ~ 1  -> pure v2 geometry (GIoU)
#   beta = 0, alpha_l1 = 0, alpha_iou = 0, blend_center = 0 -> bit-exact baseline

import torch

from .dsaw_v2_loss import RTDETRDetectionLossDSAWV2
from .inner_interpiou_loss import inner_interpiou_loss
from ultralytics.utils.metrics import bbox_iou

__all__ = ('RTDETRDetectionLossDSAWV4',)


class RTDETRDetectionLossDSAWV4(RTDETRDetectionLossDSAWV2):
    """DSAW-v4: DSAW-v2 weighting + IoU-gated GIoU/Inner-InterpIoU blend.

    Args:
        beta (float): DSAW positive-sample classification boost strength.
        alpha (float): Legacy single geometry softening strength; used as the
            default for alpha_l1 when alpha_l1 is None.
        alpha_l1 (float | None): Softening for the L1 branch. None -> alpha.
        alpha_iou (float): Softening for the IoU branch. Default 0 (off).
        tau (float): Normalized scale threshold (larger boxes get lam=0).
        eta_momentum (float): EMA momentum for matched-IoU quality.
        q_ema_init (float): Initial matched-IoU EMA value.
        dynamic_eta (bool): If true, eta = 1 - q_ema decays during training.
        lam_pow (float): Power applied to the scale factor.
        dynamic_geo (bool): If true, both softening factors anneal with eta.
        blend_center (float): Matched-IoU switch point of the GIoU/Inner blend.
            0.45 -> pairs below ~0.45 IoU use Inner-InterpIoU, above use GIoU.
        blend_temp (float): Sigmoid temperature of the blend (smoothness).
        inner_ratio (float): Inner-InterpIoU auxiliary-box ratio.
        dynamic_interp (bool): Per-sample dynamic interpolation alpha.
        interp_alpha (float): Static interpolation alpha when dynamic_interp
            is false.
    """

    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, beta=0.5, alpha=0.25, tau=0.03125,
                 eta_momentum=0.999, q_ema_init=0.5, dynamic_eta=True,
                 lam_pow=2.0, dynamic_geo=True, alpha_l1=None, alpha_iou=0.0,
                 blend_center=0.45, blend_temp=0.05, inner_ratio=0.8,
                 dynamic_interp=True, interp_alpha=0.99, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, beta=beta,
                         alpha=alpha, tau=tau, eta_momentum=eta_momentum,
                         q_ema_init=q_ema_init, dynamic_eta=dynamic_eta,
                         lam_pow=lam_pow, dynamic_geo=dynamic_geo, **kw)
        self.alpha_l1 = float(alpha) if alpha_l1 is None else float(alpha_l1)
        self.alpha_iou = float(alpha_iou)
        self.blend_center = float(blend_center)
        self.blend_temp = max(float(blend_temp), 1e-4)
        self.inner_ratio = float(inner_ratio)
        self.dynamic_interp = bool(dynamic_interp)
        self.interp_alpha = float(interp_alpha)

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        """Matcher/target flow identical to DSAW-v2, but emits two geometry
        weights: one for the L1 branch, one for the IoU branch."""
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
        w_geo_l1 = None
        w_geo_iou = None
        if len(gt_bboxes):
            iou = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)
            gt_scores[idx] = iou

            # GT-side scale factor (no gradient path through the size prior).
            s = (gt_bboxes[:, 2] * gt_bboxes[:, 3]).clamp(min=0).sqrt()
            lam = self._focused_lam(s)

            # Shared dynamic training-state factor.
            m = self.eta_momentum
            self.q_ema = m * self.q_ema + (1 - m) * float(iou.mean().detach())
            eta = self.eta

            # Positive-class boost (unchanged from v2/v3).
            w_cls_map[idx] = 1.0 + self.beta * lam * eta

            # Decoupled geometry weights.
            geo_eta = eta if self.dynamic_geo else 1.0
            w_geo_l1 = 1.0 - self.alpha_l1 * lam * geo_eta
            w_geo_iou = 1.0 - self.alpha_iou * lam * geo_eta

        loss = {}
        loss.update(self._get_loss_class_w(pred_scores, targets, gt_scores,
                                           len(gt_bboxes), w_cls_map, postfix))
        loss.update(self._get_loss_bbox_w(pred_bboxes, gt_bboxes,
                                          w_geo_l1, w_geo_iou, postfix))
        return loss

    def _get_loss_bbox_w(self, pred_bboxes, gt_bboxes, w_geo_l1,
                         w_geo_iou=None, postfix=''):
        """IoU-gated blend of GIoU and Inner-InterpIoU + weighted L1."""
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'
        loss = {}

        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=pred_bboxes.device)
            loss[name_giou] = torch.tensor(0., device=pred_bboxes.device)
            return loss

        if w_geo_l1 is None:
            w_geo_l1 = torch.ones(len(gt_bboxes), device=pred_bboxes.device)
        if w_geo_iou is None:
            w_geo_iou = w_geo_l1

        # L1 branch: keep the v2 softening (constant gradient needs a brake).
        l1 = (pred_bboxes - gt_bboxes).abs().sum(-1)
        loss[name_bbox] = self.loss_gain['bbox'] * (l1 * w_geo_l1).sum() / len(gt_bboxes)

        # IoU branch: segmented blend, gated by detached matched IoU.
        iou_plain = bbox_iou(pred_bboxes.detach(), gt_bboxes,
                             xywh=True).squeeze(-1)
        mu = torch.sigmoid((iou_plain - self.blend_center) / self.blend_temp)

        giou = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True,
                              GIoU=True).squeeze(-1)
        inner = inner_interpiou_loss(
            pred_bboxes,
            gt_bboxes,
            inner_ratio=self.inner_ratio,
            dynamic_interp=self.dynamic_interp,
            interp_alpha=self.interp_alpha,
            xywh=True,
        ).squeeze(-1)

        geo = mu * giou + (1.0 - mu) * inner
        loss[name_giou] = self.loss_gain['giou'] * (geo * w_geo_iou).sum() / len(gt_bboxes)

        return {k: v.squeeze() for k, v in loss.items()}
