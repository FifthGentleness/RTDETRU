# DSAW-v5: matching-end scale-aware intervention.
#
# Innovation point: the ASSIGNMENT side (Hungarian matching cost). All previous
# DSAW versions (v1-v4) reweight supervision AFTER matching; v5 changes WHO a
# tiny GT gets assigned to, which is where the tiny-object bottleneck actually
# sits: weights cannot rescue a wrong assignment.
#
# Motivation: a tiny GT has near-zero IoU with every query unless the overlap
# is nearly perfect, so cost_giou is almost constant across queries and the
# assignment of tiny objects is decided by negligible cost differences that
# flip between queries across iterations (pure gradient noise). NWD (normalized
# Wasserstein similarity) varies smoothly with center/size offsets even for
# non-overlapping tiny boxes, giving the matcher a discriminative,
# flip-resistant signal exactly where IoU is blind.
#
# Mechanism (matcher cost only, loss inherited from DSAW-v4 unchanged):
#   lam_j  = clip(1 - sqrt(w_j * h_j) / tau, 0, 1) ** lam_pow   (GT-side, focused,
#            same formula/scale gate as the DSAW loss lam)
#   cost_giou = 1 - ((1 - w_nwd * lam_j) * GIoU + (w_nwd * lam_j) * NWD)
#   Large GTs (lam=0) keep the bit-exact baseline cost.
#
# Assignment-supervision consistency: the matcher lam reuses the loss lam
# (tau, lam_pow), so "who gets assigned" and "how the pair is supervised" read
# from the same scale gate (same principle as SR-MAL component B).
#
# Fallbacks (degeneracy checks):
#   nwd_weight = 0 or tau = 0 -> lam masked to 0 -> bit-exact baseline matcher
#   (then v5 == DSAW-v4 exactly).

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.utils.metrics import bbox_iou, wasserstein_loss

from .dsaw_v4_loss import RTDETRDetectionLossDSAWV4

__all__ = ('RTDETRDetectionLossDSAWV5', 'DSAWV5HungarianMatcher')


def focused_gt_lambda(gt_bboxes, tau, lam_pow):
    """Per-GT NWD blend weight lam in [0, 1]: DSAW's focused scale factor.

    Same formula as the loss-side lam (no gradient, GT-side constant), so the
    matcher and the loss share one scale gate.
    """
    if tau is None or tau <= 0:
        return torch.zeros_like(gt_bboxes[..., 0])
    s = (gt_bboxes[..., 2] * gt_bboxes[..., 3]).clamp(min=0).sqrt()
    lam = (1.0 - s / float(tau)).clamp(0.0, 1.0)
    if lam_pow != 1.0:
        lam = lam.pow(float(lam_pow))
    return lam


class DSAWV5HungarianMatcher(HungarianMatcher):
    """HungarianMatcher whose GIoU cost is blended with NWD for small GTs.

    cost_giou = 1 - ((1 - w_nwd * lam_j) * GIoU + (w_nwd * lam_j) * NWD)

    Args:
        tau (float): Normalized scale threshold; GTs larger than tau get
            lam=0 and the exact baseline cost.
        lam_pow (float): Power applied to the scale factor (matches the loss).
        nwd_const (float): Wasserstein constant for normalized coords
            (0.02 = 12.8px / 640px; repo-wide SR-MAL precedent).
        nwd_weight (float): Max NWD blend strength at lam=1. 1.0 = full blend
            for tiny boxes (SR-MAL style); 0.5 = conservative half blend.
    """

    def __init__(self, cost_gain=None, use_fl=True, tau=0.03125, lam_pow=2.0,
                 nwd_const=0.02, nwd_weight=1.0, **kw):
        super().__init__(cost_gain=cost_gain, use_fl=use_fl, **kw)
        self.tau = float(tau)
        self.lam_pow = float(lam_pow)
        self.nwd_const = float(nwd_const)
        self.nwd_weight = float(nwd_weight)

    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                masks=None, gt_mask=None):
        bs, nq, nc = pred_scores.shape
        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long),
                     torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        # ---- identical to base matcher: classification cost ----
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        pred_bboxes = pred_bboxes.detach().view(-1, 4)

        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * (pred_scores ** self.gamma) * \
                (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * \
                (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        # ---- identical to base matcher: L1 cost ----
        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)

        # ---- v5 intervention: scale-gated GIoU/NWD blend ----
        iou_sim = bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0),
                           xywh=True, GIoU=True).squeeze(-1)                 # [nq, ng]
        nwd_sim = wasserstein_loss(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0),
                                   xywh=True, constant=self.nwd_const).squeeze(-1)
        lam = focused_gt_lambda(gt_bboxes, self.tau, self.lam_pow).unsqueeze(0)  # [1, ng]
        w = self.nwd_weight * lam
        cost_giou = 1.0 - ((1.0 - w) * iou_sim + w * nwd_sim)

        C = self.cost_gain['class'] * cost_class + \
            self.cost_gain['bbox'] * cost_bbox + \
            self.cost_gain['giou'] * cost_giou

        # ---- identical to base matcher: per-image assignment ----
        C[C.isnan() | C.isinf()] = 0.0
        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(C.split(gt_groups, -1))]
        gt_off = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        return [(torch.tensor(i, dtype=torch.long),
                 torch.tensor(j, dtype=torch.long) + gt_off[k])
                for k, (i, j) in enumerate(indices)]


class RTDETRDetectionLossDSAWV5(RTDETRDetectionLossDSAWV4):
    """DSAW-v5: DSAW-v4 loss + scale-aware matching cost.

    The loss branch is inherited unchanged from v4; the only change is that
    self.matcher is a DSAWV5HungarianMatcher whose GIoU cost is NWD-blended
    for small GTs, using the SAME scale gate (tau, lam_pow) as the loss.

    Args (new vs v4):
        nwd_const (float): Wasserstein constant for normalized coordinates.
        nwd_weight (float): Max NWD blend strength (1.0 = SR-MAL style full
            blend for tiny boxes).
    """

    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, beta=0.5, alpha=0.25, tau=0.03125,
                 eta_momentum=0.999, q_ema_init=0.5, dynamic_eta=True,
                 lam_pow=2.0, dynamic_geo=True, alpha_l1=None, alpha_iou=0.0,
                 blend_center=0.45, blend_temp=0.05, inner_ratio=0.8,
                 dynamic_interp=True, interp_alpha=0.99,
                 nwd_const=0.02, nwd_weight=1.0, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, beta=beta,
                         alpha=alpha, tau=tau, eta_momentum=eta_momentum,
                         q_ema_init=q_ema_init, dynamic_eta=dynamic_eta,
                         lam_pow=lam_pow, dynamic_geo=dynamic_geo,
                         alpha_l1=alpha_l1, alpha_iou=alpha_iou,
                         blend_center=blend_center, blend_temp=blend_temp,
                         inner_ratio=inner_ratio, dynamic_interp=dynamic_interp,
                         interp_alpha=interp_alpha, **kw)
        # Replace the base matcher (created inside DETRLoss.__init__) with the
        # scale-aware one. cost_gain matches the DETRLoss default exactly.
        self.matcher = DSAWV5HungarianMatcher(
            cost_gain={'class': 2, 'bbox': 5, 'giou': 2},
            use_fl=True,
            tau=self.tau,
            lam_pow=self.lam_pow,
            nwd_const=float(nwd_const),
            nwd_weight=float(nwd_weight),
        )
