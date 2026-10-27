# InnerInterpIoU-DSAW3: DSAW-v2 dynamic scale-aware weighting + Inner-InterpIoU.
#
# This is a clean loss-only combination:
#   - No AIFI / encoder architecture change.
#   - Classification branch: DSAW-v2 positive-sample weighting.
#   - L1 branch: DSAW-v2 geometry weighting.
#   - Geometry branch: Inner-InterpIoU replaces GIoU, then receives the same
#     DSAW-v2 geometry weight.
#
# DSAW-v2:
#   lam  = clip(1 - sqrt(w * h) / tau, 0, 1) ** lam_pow
#   eta  = 1 - q_ema
#   w_cls = 1 + beta * lam * eta
#   w_geo = 1 - alpha * lam * eta
#
# Inner-InterpIoU:
#   L = (1 - IoU(B, Bgt)) + (1 - IoU(B_a, Bgt))
#       + IoU(B, Bgt) - IoU(B_r, Bgt_r)
#
# The combined geometry term is:
#   L_geo = w_geo * InnerInterpIoU(B, Bgt)

import torch

from .dsaw_v2_loss import RTDETRDetectionLossDSAWV2
from .inner_interpiou_loss import inner_interpiou_loss

__all__ = ('RTDETRDetectionLossInnerInterpIoUDSAW3',)


class RTDETRDetectionLossInnerInterpIoUDSAW3(RTDETRDetectionLossDSAWV2):
    """DSAW-v2 weighting with Inner-InterpIoU as the geometry metric.

    Args:
        beta (float): DSAW-v2 positive-sample classification boost strength.
        alpha (float): DSAW-v2 geometry softening strength.
        tau (float): DSAW-v2 normalized scale threshold.
        eta_momentum (float): EMA momentum for matched-IoU quality.
        q_ema_init (float): Initial matched-IoU EMA value.
        dynamic_eta (bool): If true, eta=1-q_ema decays during training.
        lam_pow (float): Power applied to the DSAW-v2 scale factor.
        dynamic_geo (bool): If true, DSAW-v2 geometry softening is annealed.
        inner_ratio (float): Inner-InterpIoU auxiliary-box ratio. 0.8 is a
            good starting point for small objects.
        dynamic_interp (bool): Use per-sample dynamic interpolation alpha.
        interp_alpha (float): Static interpolation alpha when dynamic_interp
            is false.
    """

    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, beta=0.5, alpha=0.25, tau=0.03125,
                 eta_momentum=0.999, q_ema_init=0.5, dynamic_eta=True,
                 lam_pow=2.0, dynamic_geo=True, inner_ratio=0.8,
                 dynamic_interp=True, interp_alpha=0.99, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, beta=beta,
                         alpha=alpha, tau=tau, eta_momentum=eta_momentum,
                         q_ema_init=q_ema_init, dynamic_eta=dynamic_eta,
                         lam_pow=lam_pow, dynamic_geo=dynamic_geo, **kw)
        self.inner_ratio = float(inner_ratio)
        self.dynamic_interp = bool(dynamic_interp)
        self.interp_alpha = float(interp_alpha)

    def _get_loss_bbox_w(self, pred_bboxes, gt_bboxes, w_geo, postfix=''):
        """Weighted L1 + weighted Inner-InterpIoU bbox loss."""
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'
        loss = {}

        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=pred_bboxes.device)
            loss[name_giou] = torch.tensor(0., device=pred_bboxes.device)
            return loss

        if w_geo is None:
            w_geo = torch.ones(len(gt_bboxes), device=pred_bboxes.device)

        # Keep the DSAW-v2 L1 branch unchanged.
        l1 = (pred_bboxes - gt_bboxes).abs().sum(-1)
        loss[name_bbox] = self.loss_gain['bbox'] * (l1 * w_geo).sum() / len(gt_bboxes)

        # Replace GIoU with Inner-InterpIoU, then apply DSAW-v2's dynamic
        # geometry weight. The interpolation alpha inside
        # inner_interpiou_loss is detached; no extra scale factor is added here.
        geo = inner_interpiou_loss(
            pred_bboxes,
            gt_bboxes,
            inner_ratio=self.inner_ratio,
            dynamic_interp=self.dynamic_interp,
            interp_alpha=self.interp_alpha,
            xywh=True,
        ).squeeze(-1)
        loss[name_giou] = self.loss_gain['giou'] * (geo * w_geo).sum() / len(gt_bboxes)

        return {k: v.squeeze() for k, v in loss.items()}