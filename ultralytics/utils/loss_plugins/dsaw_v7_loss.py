# DSAW-v7: v6 matcher + quality-gated warmup (fixes the "noise-phase
# intervention" flaw identified in the v5/v6 post-mortem).
#
# Loss branch is inherited bit-exact from v4 (via v6). The ONLY change vs v6
# is one extra multiplicative factor on the matcher blend weight.
#
# == The v6 flaw being fixed ==
#
# v6's two gates stack into a peak of intervention exactly in the noise
# phase (first epochs), where predictions are still random:
#   - eta curriculum: strongest early (eta ~= 0.5 at start);
#   - no-overlap gate: early on nearly ALL (query, gt) pairs have IoU < 0.1,
#     so the gate is wide open.
# Result: the NWD blend perturbs a cost matrix that is pure noise -> the
# ep5 gap observed in the v6 probe run (0.0959 vs v4 0.1060).
#
# == The v7 fix: a quality-gated warmup ==
#
# The matcher tracks its own EMA of the mean IoU of ASSIGNED pairs
# (q_warm_ema, init 0.0, same momentum as the loss-side quality EMA).
# This is a direct, self-contained measure of "are the predictions
# meaningful yet" - no epoch plumbing into the criterion is needed, and it
# adapts to any dataset automatically.
#
#   warm_ij = clip(q_warm_ema / q_warmup, 0, 1) ** warmup_pow
#   w_ij    = nwd_weight * lam_j * eta_t * gate_ij * warm_ij
#
# Window shape over training (all three factors are detached statistics):
#   start  : q_warm_ema = 0        -> warm = 0  -> bit-exact v4 cost
#   ramp-in: mean matched IoU rises -> warm -> 1 as it crosses q_warmup
#   peak   : mid-training (assignments meaningful, churn still frequent)
#   ramp-out: eta -> 0 (v6 curriculum unchanged)
#
# The ramp-in is driven by QUALITY, not by wall-clock epochs: if the model
# learns fast, the intervention engages earlier; if the dataset is harder,
# it stays off longer. This is the same "dynamic, training-state aware"
# philosophy as the loss-side eta.
#
# Fallbacks (degeneracy checks):
#   nwd_weight = 0                    -> bit-exact v4 (w = 0)
#   q_warmup = 0 or warmup_pow = 0    -> warm = 1 -> bit-exact v6
#   q_warmup = 0 AND nwd_weight = 0   -> bit-exact v4
#
# New hyperparameters (vs v6):
#   q_warmup        quality threshold where the warmup ramp reaches full
#                   strength (mean assigned-IoU units). Default 0.2.
#   warmup_pow      sharpness of the ramp (>1 suppresses the noise phase
#                   harder). Default 1.0 (linear).
#   warmup_momentum EMA momentum of q_warm_ema. Default 0.999 (same time
#                   constant as the loss-side eta EMA, ~1000 iters).
#
# Diagnostic counters (read after forward):
#   inherited from v6: last_w_mean / last_gate_frac / last_w_active_mean
#   new: last_warmup_gate (mean warm factor over the cost matrix),
#        last_q_warm_ema  (current warmup EMA value)

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.utils.metrics import bbox_iou, wasserstein_loss

from .dsaw_v6_loss import RTDETRDetectionLossDSAWV6, DSAWV6HungarianMatcher
from .dsaw_v5_loss import focused_gt_lambda

__all__ = ('RTDETRDetectionLossDSAWV7', 'DSAWV7HungarianMatcher')


class DSAWV7HungarianMatcher(DSAWV6HungarianMatcher):
    """HungarianMatcher = v6 (curricula + no-overlap-gated NWD blend) with a
    quality-gated warmup on the blend weight.

    cost_giou = 1 - ((1 - w) * GIoU + w * NWD)
    w         = nwd_weight * lam_j * eta * 1[IoU_raw < overlap_thresh] * warm
    warm      = clip(q_warm_ema / q_warmup, 0, 1) ** warmup_pow

    q_warm_ema is an EMA of the mean plain IoU of the ASSIGNED pairs,
    updated inside forward (the matcher knows the assignment right after
    linear_sum_assignment, so the statistic matches the loss-side matched
    IoU exactly and needs no external sync).
    """

    def __init__(self, cost_gain=None, use_fl=True, tau=0.03125, lam_pow=1.0,
                 nwd_const=0.02, nwd_weight=0.5, overlap_thresh=0.1,
                 q_warmup=0.2, warmup_pow=1.0, warmup_momentum=0.999, **kw):
        super().__init__(cost_gain=cost_gain, use_fl=use_fl, tau=tau,
                         lam_pow=lam_pow, nwd_const=nwd_const,
                         nwd_weight=nwd_weight,
                         overlap_thresh=overlap_thresh, **kw)
        self.q_warmup = float(q_warmup)
        self.warmup_pow = float(warmup_pow)
        self.warmup_momentum = float(warmup_momentum)
        # Matcher-side quality EMA. Init 0 => warm=0 => the very first calls
        # are bit-exact v4 regardless of nwd_weight (intervention engages
        # only after assignments become meaningful).
        self.q_warm_ema = 0.0
        # Diagnostics.
        self.last_warmup_gate = 0.0
        self.last_q_warm_ema = 0.0

    @torch.no_grad()
    def _update_warm_ema(self, iou_raw, indices):
        """Update the assigned-pair mean-IoU EMA from the fresh assignment."""
        ious = []
        for (row, col) in indices:
            row = torch.as_tensor(row, device=iou_raw.device)
            col = torch.as_tensor(col, device=iou_raw.device)
            if row.numel():
                ious.append(iou_raw[row, col].detach())
        if not ious:
            return
        mean_iou = float(torch.cat(ious).mean())
        m = self.warmup_momentum
        self.q_warm_ema = m * self.q_warm_ema + (1.0 - m) * mean_iou

    def _warmup_gate(self):
        """Scalar warmup factor from the quality EMA."""
        if self.q_warmup <= 0.0 or self.warmup_pow <= 0.0:
            return 1.0  # disabled -> bit-exact v6 behavior
        warm = min(max(self.q_warm_ema / self.q_warmup, 0.0), 1.0)
        return warm ** self.warmup_pow

    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                masks=None, gt_mask=None):
        """Identical to v6 except the extra multiplicative warmup factor on w
        and the assigned-IoU EMA update."""
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

        # ---- v6 intervention + v7 warmup: gated GIoU/NWD blend ----
        iou_raw = bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0),
                           xywh=True).squeeze(-1)                       # [nq, ng]
        giou_sim = bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0),
                            xywh=True, GIoU=True).squeeze(-1)           # [nq, ng]
        nwd_sim = wasserstein_loss(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0),
                                   xywh=True, constant=self.nwd_const).squeeze(-1)
        lam = focused_gt_lambda(gt_bboxes, self.tau, self.lam_pow).unsqueeze(0)  # [1, ng]
        gate = (iou_raw.detach() < self.overlap_thresh).float()         # [nq, ng]

        warm = self._warmup_gate()
        w = self.nwd_weight * lam * float(self.eta) * gate * warm
        cost_giou = 1.0 - ((1.0 - w) * giou_sim + w * nwd_sim)

        C = self.cost_gain['class'] * cost_class + \
            self.cost_gain['bbox'] * cost_bbox + \
            self.cost_gain['giou'] * cost_giou

        # ---- identical to base matcher: per-image assignment ----
        C[C.isnan() | C.isinf()] = 0.0
        C = C.view(bs, nq, -1).cpu()
        raw_indices = [linear_sum_assignment(c[i])
                       for i, c in enumerate(C.split(gt_groups, -1))]
        gt_off = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        indices = [(torch.tensor(i, dtype=torch.long),
                    torch.tensor(j, dtype=torch.long) + gt_off[k])
                   for k, (i, j) in enumerate(raw_indices)]

        # ---- v7: quality EMA update + diagnostics (after assignment) ----
        with torch.no_grad():
            self._update_warm_ema(iou_raw, raw_indices)
            self.last_q_warm_ema = float(self.q_warm_ema)
            if w.numel():
                self.last_w_mean = float(w.mean())
                self.last_gate_frac = float(gate.mean())
                self.last_warmup_gate = float(warm)
                active = w[w > 0]
                self.last_w_active_mean = float(active.mean()) if active.numel() else 0.0

        return indices


class RTDETRDetectionLossDSAWV7(RTDETRDetectionLossDSAWV6):
    """DSAW-v7: v6 loss/matcher + quality-gated warmup on the matcher blend.

    Loss branch stays bit-exact v4 (inherited through v6). The only change
    is the matcher, which now suppresses the NWD intervention during the
    noise phase (first iterations, mean assigned IoU ~ 0) and ramps it in
    as assignments become meaningful.

    Args (new vs v6):
        q_warmup (float): Quality threshold (mean assigned-IoU) where the
            warmup ramp reaches full strength. 0 disables the warmup
            (bit-exact v6).
        warmup_pow (float): Ramp sharpness. 0 disables (bit-exact v6);
            >1 suppresses the noise phase harder.
        warmup_momentum (float): EMA momentum for the matcher-side
            assigned-IoU tracker. 0.999 matches the loss-side eta EMA time
            constant (~1000 iterations).
    """

    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, beta=0.5, alpha=0.25, tau=0.03125,
                 eta_momentum=0.999, q_ema_init=0.5, dynamic_eta=True,
                 lam_pow=2.0, dynamic_geo=True, alpha_l1=None, alpha_iou=0.0,
                 blend_center=0.45, blend_temp=0.05, inner_ratio=0.8,
                 dynamic_interp=True, interp_alpha=0.99,
                 nwd_const=0.02, nwd_weight=0.5, overlap_thresh=0.1,
                 matcher_tau=None, matcher_lam_pow=1.0,
                 q_warmup=0.2, warmup_pow=1.0, warmup_momentum=0.999, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, beta=beta,
                         alpha=alpha, tau=tau, eta_momentum=eta_momentum,
                         q_ema_init=q_ema_init, dynamic_eta=dynamic_eta,
                         lam_pow=lam_pow, dynamic_geo=dynamic_geo,
                         alpha_l1=alpha_l1, alpha_iou=alpha_iou,
                         blend_center=blend_center, blend_temp=blend_temp,
                         inner_ratio=inner_ratio, dynamic_interp=dynamic_interp,
                         interp_alpha=interp_alpha,
                         nwd_const=nwd_const, nwd_weight=nwd_weight,
                         overlap_thresh=overlap_thresh,
                         matcher_tau=matcher_tau,
                         matcher_lam_pow=matcher_lam_pow, **kw)
        # Replace the v6 matcher with the v7 matcher (same cost_gain as the
        # DETRLoss default, same convention as v6).
        self.matcher = DSAWV7HungarianMatcher(
            cost_gain={'class': 2, 'bbox': 5, 'giou': 2},
            use_fl=True,
            tau=self.tau if matcher_tau is None else float(matcher_tau),
            lam_pow=float(matcher_lam_pow),
            nwd_const=float(nwd_const),
            nwd_weight=float(nwd_weight),
            overlap_thresh=float(overlap_thresh),
            q_warmup=float(q_warmup),
            warmup_pow=float(warmup_pow),
            warmup_momentum=float(warmup_momentum),
        )
        # Same eta-sync convention as v6 (one-iteration delay afterwards).
        self.matcher.eta = max(0.0, min(1.0, 1.0 - self.q_ema))



