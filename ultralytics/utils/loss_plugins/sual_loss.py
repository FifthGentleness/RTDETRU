# Ultralytics YOLO 🐛, AGPL-3.0 license
"""
SUAL: Scale-Uncertainty-Aware Localization loss for RT-DETR (self-contained plugin).

Two sources, fused at the localization branch of matched queries:

[1] Scale branch - SLS (CVPR 2024)
    Liu et al., "Infrared Small Target Detection with Scale and Location
    Sensitivity", CVPR 2024. Official code: github.com/BIT-RuiLiu/MSHNet.
    Faithful port of both components to box form:
      * Scale-sensitive weight on the IoU term (Eq. 3 of the paper):
            alpha = (min(A_p, A_gt) + dis) / (max(A_p, A_gt) + dis)
            dis   = ((A_p - A_gt) / 2) ** 2
        with A = box area. The larger the predicted/GT scale gap, the smaller
        alpha is, hence the larger the loss under fixed IoU. The gradient also
        flows through A_p, so the branch actively shrinks the scale gap.
      * Location-sensitive penalty (LLoss in the official code): a polar
        center-point penalty (angular error + radial-length ratio) computed
        from predicted/GT box centers. More discriminative than L1/L2, which
        assign equal loss values to many distinct location errors.

[2] Uncertainty branch - UGS-inspired (ICCV 2025)
    Sun et al., "Uncertainty-Aware Gradient Stabilization for Small Object
    Detection", ICCV 2025. Core idea adapted WITHOUT any architecture change
    (parameter-free, zero inference cost):
      * SQCL (soft-quantized classification-style localization): continuous
        coordinates are mapped to differentiable fractional bin indices over
        non-uniform bin centers; a kernel-softmax turns each coordinate into a
        distribution and a cross-entropy between predicted/GT distributions
        replaces part of the continuous regression. The resulting gradients
        are bounded and saturate far from the target, mitigating the sharp
        loss curvature of small objects (the instability UGS identifies).
      * UM surrogate: inter-decoder-layer variance minimization of matched
        predicted boxes (prediction-variance suppression, scale weighted).
      * UR surrogate: uncertainty-driven reweighting of the IoU term
        (rho * normalized uncertainty). The original UR is a perturbation
        based refinement module; that requires architecture support and is
        intentionally NOT replicated here.

Total (per matched pair i):
    L_SUAL = loss_gain['bbox'] * L1                        # baseline L1 (cx,cy,w,h)
           + lambda_L * L_loc_polar                        # SLS polar location penalty
           + loss_gain['giou'] * sum_i w_u_i*(1-alpha_i*IoU_i)  # scale x uncertainty IoU
           + lambda_c * L_SQCL                             # UGS-style classification loc
           + mu * L_UM                                     # variance minimization
    with w_u_i = 1 + rho * u_hat_i, u_hat in [0, 1].

    Note: L1 is preserved from the baseline so that all four box coordinates
    (cx, cy, w, h) receive strong direct gradients throughout training.
    The polar location penalty is added ON TOP of L1 (not replacing it),
    providing extra directional discriminability for center localization.

Registered as loss_name: 'sual' in ultralytics/utils/loss_plugins.
"""

import math

import torch
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou

from ultralytics.models.utils.loss import DETRLoss, RTDETRDetectionLoss

__all__ = ('RTDETRDetectionLossSUAL',)


class RTDETRDetectionLossSUAL(RTDETRDetectionLoss):
    """RT-DETR loss with Scale-Uncertainty-Aware Localization (L_SUAL).

    Only the localization branch of matched queries is changed; the
    classification loss, Hungarian matching cost, auxiliary/denoising plumbing
    are inherited unchanged. Training-only: no parameters, no inference cost.
    """

    def __init__(self,
                 nc=80,
                 loss_gain=None,
                 aux_loss=True,
                 use_fl=True,
                 use_vfl=True,
                 use_sl=False,
                 use_emasl=False,
                 use_svfl=False,
                 use_emasvfl=False,
                 use_mal=False,
                 # ---- [1] SLS scale branch ----
                 sls_iou='giou',       # 'giou' | 'iou' | 'ciou'
                 sls_loc_weight=1.0,   # lambda_L: polar location penalty weight
                 # ---- [2] UGS-inspired uncertainty branch ----
                 sqcl_bins=32,         # K: number of quantization bins per coordinate
                 sqcl_sigma=1.0,       # kernel width in bin units
                 sqcl_nonuniform=1.0,  # gamma: >1 gives denser bins near 0 (UGS non-uniform idea)
                 sqcl_weight=1.0,      # lambda_c: classification-style localization weight
                 um_weight=0.5,        # mu: variance minimization weight
                 um_area_ref=0.01,     # scale anchor for UM weighting (~80px @640)
                 um_gamma=0.5,         # UM scale-weight exponent
                 um_w_max=4.0,         # UM scale-weight clamp
                 sual_rho=1.0):        # rho: uncertainty reweighting strength on the IoU term
        super().__init__(nc=nc,
                         loss_gain=loss_gain,
                         aux_loss=aux_loss,
                         use_fl=use_fl,
                         use_vfl=use_vfl,
                         use_sl=use_sl,
                         use_emasl=use_emasl,
                         use_svfl=use_svfl,
                         use_emasvfl=use_emasvfl,
                         use_mal=use_mal)
        self.sls_iou = str(sls_iou)
        self.sls_loc_weight = float(sls_loc_weight)
        self.sqcl_bins = int(sqcl_bins)
        self.sqcl_sigma = float(sqcl_sigma)
        self.sqcl_nonuniform = float(sqcl_nonuniform)
        self.sqcl_weight = float(sqcl_weight)
        self.um_weight = float(um_weight)
        self.um_area_ref = float(um_area_ref)
        self.um_gamma = float(um_gamma)
        self.um_w_max = float(um_w_max)
        self.sual_rho = float(sual_rho)
        # per-matched-pair uncertainty (set in forward, consumed by _get_loss_bbox)
        self._sual_u = None

    # ------------------------------------------------------------------
    # [1] SLS scale branch + polar location penalty (box form)
    # ------------------------------------------------------------------
    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        """L_SUAL localization terms for matched pairs ([M, 4] xywh normalized).

        loss_bbox = gain_bbox * L1  +  sls_loc_weight * polar_loc
        loss_giou = gain_giou * (1 - alpha * IoU) * (1 + rho * u_hat)

        L1 is preserved from the baseline so that all four coordinates
        (cx, cy, w, h) receive strong direct gradients. The polar location
        penalty is added on top of L1, not replacing it.
        """
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'
        if len(gt_bboxes) == 0:
            zero = torch.tensor(0., device=self.device)
            return {name_bbox: zero, name_giou: zero}
        n = len(gt_bboxes)

        # ---- baseline L1 regression (cx, cy, w, h) ----
        l1 = F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / n

        # ---- scale-sensitive weight (SLS Eq.3, box-area form) ----
        a_p = (pred_bboxes[:, 2] * pred_bboxes[:, 3]).clamp_min(1e-8)
        a_g = (gt_bboxes[:, 2] * gt_bboxes[:, 3]).clamp_min(1e-8)
        dis = ((a_p - a_g) * 0.5) ** 2
        alpha = (torch.minimum(a_p, a_g) + dis + 1e-8) / \
                (torch.maximum(a_p, a_g) + dis + 1e-8)                       # [M]

        iou = bbox_iou(pred_bboxes, gt_bboxes, xywh=True,
                       GIoU=(self.sls_iou == 'giou'),
                       CIoU=(self.sls_iou == 'ciou')).squeeze(-1)            # [M]
        iou_loss = 1.0 - alpha * iou

        # ---- uncertainty-driven reweighting (UGS/UR surrogate) ----
        if self._sual_u is not None and len(self._sual_u) == n:
            iou_loss = iou_loss * (1.0 + self.sual_rho * self._sual_u)

        loss_giou = self.loss_gain['giou'] * iou_loss.sum() / n

        # ---- location-sensitive penalty (SLS LLoss, polar center form) ----
        cx_p, cy_p = pred_bboxes[:, 0], pred_bboxes[:, 1]
        cx_g, cy_g = gt_bboxes[:, 0], gt_bboxes[:, 1]
        angle_loss = (4.0 / math.pi ** 2) * (torch.arctan(cy_p / (cx_p + 1e-8)) -
                                             torch.arctan(cy_g / (cx_g + 1e-8))) ** 2
        r_p = torch.sqrt(cx_p * cx_p + cy_p * cy_p + 1e-8)
        r_g = torch.sqrt(cx_g * cx_g + cy_g * cy_g + 1e-8)
        length_ratio = torch.minimum(r_p, r_g) / (torch.maximum(r_p, r_g) + 1e-8)
        loc = ((1.0 - length_ratio + angle_loss).sum()) / n

        return {
            name_bbox: self.loss_gain['bbox'] * l1 + self.sls_loc_weight * loc,
            name_giou: loss_giou,
        }

    # ------------------------------------------------------------------
    # [2] UGS-inspired uncertainty branch
    # ------------------------------------------------------------------
    def _frac_index(self, v):
        """Differentiable fractional bin index under (non-)uniform bin centers.

        Bin centers are c_k = (k/(K-1))**gamma; the exact inverse of this
        family is idx(v) = v**(1/gamma) * (K-1), so the mapping stays smooth
        and differentiable for any gamma (gamma=1 -> uniform bins).
        """
        g = max(self.sqcl_nonuniform, 1e-6)
        return v.clamp(1e-6, 1.0).pow(1.0 / g) * (self.sqcl_bins - 1)

    def _sqcl_loss(self, pred_assigned, gt_assigned):
        """Soft-quantized classification-style localization (UGS core idea).

        Both predicted and GT coordinates are converted into kernel
        distributions over K bins; the CE between them yields bounded,
        saturating gradients instead of the unbounded gradients of continuous
        regression on small objects.
        """
        centers = torch.arange(self.sqcl_bins, device=pred_assigned.device,
                               dtype=pred_assigned.dtype)                     # [K]
        idx_p = self._frac_index(pred_assigned)                               # [M, 4]
        idx_g = self._frac_index(gt_assigned)
        logits_p = -((centers - idx_p.unsqueeze(-1)) ** 2) / (2 * self.sqcl_sigma ** 2)
        logits_g = -((centers - idx_g.unsqueeze(-1)) ** 2) / (2 * self.sqcl_sigma ** 2)
        q_p = F.softmax(logits_p, dim=-1)
        q_g = F.softmax(logits_g, dim=-1)
        ce = -(q_g * torch.log(q_p.clamp_min(1e-8))).sum(-1)                  # [M, 4]
        return ce.sum(-1).mean()

    def _estimate_uncertainty(self, pred_bboxes, idx):
        """Inter-layer std of matched predicted boxes -> normalized [0, 1]."""
        if pred_bboxes.shape[0] < 2 or len(idx[0]) == 0:
            return None
        per_layer = torch.stack([pred_bboxes[l][idx] for l in range(pred_bboxes.shape[0])])
        std = per_layer.std(0, unbiased=False).mean(-1)                       # [M]
        return std / (std.max() + 1e-8)

    def _um_loss(self, pred_bboxes, idx, gt_bboxes, gt_idx):
        """UM surrogate: minimize inter-layer prediction variance of matched
        queries, scale-weighted so tiny objects are stabilized the most."""
        per_layer = torch.stack([pred_bboxes[l][idx] for l in range(pred_bboxes.shape[0])])
        var = per_layer.var(0, unbiased=False).mean(-1)                       # [M]
        if self.um_gamma > 0:
            a_g = (gt_bboxes[gt_idx][:, 2] * gt_bboxes[gt_idx][:, 3]).clamp_min(1e-8)
            w = ((self.um_area_ref / a_g) ** self.um_gamma).clamp(1.0, self.um_w_max)
        else:
            w = 1.0
        return (var * w).sum() / max(len(gt_idx), 1)

    # ------------------------------------------------------------------
    # forward: share match indices, keep cls/dn plumbing of the parent
    # ------------------------------------------------------------------
    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        pred_bboxes, pred_scores = preds
        self.device = pred_bboxes.device
        gt_cls, gt_bboxes, gt_groups = batch['cls'], batch['bboxes'], batch['gt_groups']

        match_indices = self.matcher(pred_bboxes[-1], pred_scores[-1], gt_bboxes,
                                     gt_cls, gt_groups, masks=None, gt_mask=None)
        idx, gt_idx = self._get_index(match_indices)

        # uncertainty consumed inside _get_loss_bbox (matched order aligned)
        self._sual_u = self._estimate_uncertainty(pred_bboxes, idx)

        total_loss = DETRLoss.forward(self, pred_bboxes, pred_scores, batch,
                                      match_indices=match_indices)

        if len(gt_idx) > 0 and sum(gt_groups) > 0:
            total_loss['loss_sual_ce'] = self.sqcl_weight * \
                self._sqcl_loss(pred_bboxes[-1][idx], gt_bboxes[gt_idx])
            total_loss['loss_sual_um'] = self.um_weight * \
                self._um_loss(pred_bboxes, idx, gt_bboxes, gt_idx)
        else:
            zero = torch.tensor(0., device=self.device)
            total_loss['loss_sual_ce'] = zero
            total_loss['loss_sual_um'] = zero
        self._sual_u = None

        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta['dn_pos_idx'], dn_meta['dn_num_group']
            assert len(batch['gt_groups']) == len(dn_pos_idx)
            dn_match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group,
                                                         batch['gt_groups'])
            total_loss.update(DETRLoss.forward(self, dn_bboxes, dn_scores, batch,
                                               postfix='_dn',
                                               match_indices=dn_match_indices))
        else:
            total_loss.update({f'{k}_dn': torch.tensor(0., device=self.device)
                               for k in total_loss.keys()})
        return total_loss