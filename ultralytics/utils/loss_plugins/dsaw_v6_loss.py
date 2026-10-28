# DSAW-v6: Curricula + no-overlap-gated NWD matcher (v4 loss unchanged).
#
# Redesign after the DSAW-v5 negative result (v5 < v4 at every epoch >= 5,
# gap widening to -0.005 mAP50-95 by ep15). Each v5 failure mode gets one
# targeted fix; the loss branch is inherited from v4 bit-exact.
#
# v5 failure -> v6 fix:
#   (1) NWD perturbs mid/late training (shape-refinement phase)   ->
#       Curricula: the blend weight is scaled by eta (quality gap), so the
#       intervention is strong early and decays to zero as training
#       converges (same "dynamic" philosophy as the loss side).
#   (2) NWD dilutes shape supervision wherever GIoU already works ->
#       No-overlap gating: NWD is blended ONLY for (query, gt) pairs with
#       plain IoU < overlap_thresh. Overlapping pairs keep the exact
#       baseline GIoU cost. NWD's unique value (non-overlapping tiny boxes,
#       where GIoU gradients degenerate) is preserved; its blind spot
#       (ignores aspect/shape) is fenced off.
#   (3) lam^2 gate covers almost nothing (16px -> w=4%) while VisDrone's
#       mass sits at 10-30px                                      ->
#       Matcher gate is decoupled from the loss gate: linear lam
#       (matcher_lam_pow=1.0) with its own matcher_tau.
#   (4) v5 reused the loss lam for the matcher (signal entanglement) ->
#       matcher_tau / matcher_lam_pow are independent hyperparameters.
#   (5) The "GIoU cost ties" premise was never verified             ->
#       built-in diagnostic counters (w coverage / gate hit rate) for
#       cheap premise checking before committing GPU time.
#
# Formula (matcher only; loss = DSAW-v4 unchanged):
#   lam_j  = clip(1 - sqrt(w_j*h_j)/matcher_tau, 0, 1) ** matcher_lam_pow
#            (GT-side constant, no gradient path)
#   eta_t  = 1 - q_ema                     (EMA of matched IoU; the matcher
#                                           runs before the loss refreshes
#                                           q_ema, so it uses the previous
#                                           iteration's eta. The EMA time
#                                           constant is ~1000 iters; the
#                                           one-step delay is negligible.)
#   gate_ij= 1[ IoU_raw(pred_i, gt_j) < overlap_thresh ]   (detached)
#   w_ij   = nwd_weight * lam_j * eta_t * gate_ij
#   cost_giou = 1 - ((1 - w_ij) * GIoU + w_ij * NWD)
#
# Fallbacks (degeneracy checks):
#   nwd_weight = 0 or matcher_tau = 0  -> w = 0 -> bit-exact v4
#   (v4 loss + baseline HungarianMatcher).
#
# yaml loss_params (new vs v4):
#   nwd_const: 0.02        # Wasserstein constant (normalized coords)
#   nwd_weight: 0.5        # max blend strength at lam=1, eta=1, gate=1
#   overlap_thresh: 0.1    # pairs with IoU >= this keep pure GIoU cost
#   matcher_tau: 0.03125   # matcher-side scale threshold (independent)
#   matcher_lam_pow: 1.0   # matcher-side focus power (linear by default)

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.utils.metrics import bbox_iou, wasserstein_loss

from .dsaw_v4_loss import RTDETRDetectionLossDSAWV4
from .dsaw_v5_loss import focused_gt_lambda

__all__ = ('RTDETRDetectionLossDSAWV6', 'DSAWV6HungarianMatcher')


class DSAWV6HungarianMatcher(HungarianMatcher):
    """HungarianMatcher with curricula, no-overlap-gated GIoU/NWD blend.

    cost_giou = 1 - ((1 - w) * GIoU + w * NWD)
    w         = nwd_weight * lam_j * eta * 1[IoU_raw < overlap_thresh]

    Diagnostic counters (read after forward for cheap premise checks):
        last_w_mean        mean blend weight over the full cost matrix
        last_gate_frac     fraction of (query, gt) pairs with gate active
        last_w_active_mean mean w over gate-active pairs only
    """

    def __init__(self, cost_gain=None, use_fl=True, tau=0.03125, lam_pow=1.0,
                 nwd_const=0.02, nwd_weight=0.5, overlap_thresh=0.1, **kw):
        super().__init__(cost_gain=cost_gain, use_fl=use_fl, **kw)
        self.tau = float(tau)
        self.lam_pow = float(lam_pow)
        self.nwd_const = float(nwd_const)
        self.nwd_weight = float(nwd_weight)
        self.overlap_thresh = float(overlap_thresh)
        # Set by the loss after each _get_loss call (one-iteration delay).
        self.eta = 0.5
        # Diagnostics.
        self.last_w_mean = 0.0
        self.last_gate_frac = 0.0
        self.last_w_active_mean = 0.0

    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                masks=None, gt_mask=None):
        """Identical to the base matcher except the GIoU cost term."""
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

        # ---- v6 intervention: curricula + no-overlap-gated GIoU/NWD blend ----
        iou_raw = bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0),
                           xywh=True).squeeze(-1)                       # [nq, ng]
        giou_sim = bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0),
                            xywh=True, GIoU=True).squeeze(-1)           # [nq, ng]
        nwd_sim = wasserstein_loss(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0),
                                   xywh=True, constant=self.nwd_const).squeeze(-1)
        lam = focused_gt_lambda(gt_bboxes, self.tau, self.lam_pow).unsqueeze(0)  # [1, ng]
        gate = (iou_raw.detach() < self.overlap_thresh).float()         # [nq, ng]
        w = self.nwd_weight * lam * float(self.eta) * gate
        cost_giou = 1.0 - ((1.0 - w) * giou_sim + w * nwd_sim)

        # Diagnostics (cheap premise checks, no logging I/O here).
        with torch.no_grad():
            if w.numel():
                self.last_w_mean = float(w.mean())
                self.last_gate_frac = float(gate.mean())
                active = w[w > 0]
                self.last_w_active_mean = float(active.mean()) if active.numel() else 0.0

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


class RTDETRDetectionLossDSAWV6(RTDETRDetectionLossDSAWV4):
    """DSAW-v6: DSAW-v4 loss + redesigned scale-aware matching cost.

    Loss branch is inherited bit-exact from v4. The only change is the
    matcher: curricula (w scaled by eta), no-overlap gating, and a
    decoupled (independent) scale gate.

    Args (new vs v4):
        nwd_const (float): Wasserstein constant for normalized coordinates.
        nwd_weight (float): Max NWD blend strength at lam=1, eta=1, gate=1.
            0 -> bit-exact v4.
        overlap_thresh (float): Pairs with plain IoU >= this keep pure GIoU.
        matcher_tau (float | None): Matcher-side scale threshold. None ->
            reuse the loss tau.
        matcher_lam_pow (float): Matcher-side focus power. 1.0 = linear
            (broad coverage); decoupled from the loss-side lam_pow.
    """

    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, beta=0.5, alpha=0.25, tau=0.03125,
                 eta_momentum=0.999, q_ema_init=0.5, dynamic_eta=True,
                 lam_pow=2.0, dynamic_geo=True, alpha_l1=None, alpha_iou=0.0,
                 blend_center=0.45, blend_temp=0.05, inner_ratio=0.8,
                 dynamic_interp=True, interp_alpha=0.99,
                 nwd_const=0.02, nwd_weight=0.5, overlap_thresh=0.1,
                 matcher_tau=None, matcher_lam_pow=1.0, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, beta=beta,
                         alpha=alpha, tau=tau, eta_momentum=eta_momentum,
                         q_ema_init=q_ema_init, dynamic_eta=dynamic_eta,
                         lam_pow=lam_pow, dynamic_geo=dynamic_geo,
                         alpha_l1=alpha_l1, alpha_iou=alpha_iou,
                         blend_center=blend_center, blend_temp=blend_temp,
                         inner_ratio=inner_ratio, dynamic_interp=dynamic_interp,
                         interp_alpha=interp_alpha, **kw)
        # Replace the base matcher (created inside DETRLoss.__init__).
        # cost_gain matches the DETRLoss default exactly.
        self.matcher = DSAWV6HungarianMatcher(
            cost_gain={'class': 2, 'bbox': 5, 'giou': 2},
            use_fl=True,
            tau=self.tau if matcher_tau is None else float(matcher_tau),
            lam_pow=float(matcher_lam_pow),
            nwd_const=float(nwd_const),
            nwd_weight=float(nwd_weight),
            overlap_thresh=float(overlap_thresh),
        )
        # Initialize the matcher eta from the loss EMA init (one-step delay
        # afterwards: the matcher runs before the loss refreshes q_ema).
        self.matcher.eta = max(0.0, min(1.0, 1.0 - self.q_ema))

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        """v4 loss, then sync the curriculum factor to the matcher."""
        out = super()._get_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls,
                                gt_groups, masks=masks, gt_mask=gt_mask,
                                postfix=postfix, match_indices=match_indices)
        self.matcher.eta = self.eta
        return out
