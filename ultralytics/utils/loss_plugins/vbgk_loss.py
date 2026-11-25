# Ultralytics YOLO 🚀, AGPL-3.0 license
"""VBGK: Variation-Bounded, GCA-Margin & Kendall-Calibrated Loss for RT-DETR.

Self-contained loss plugin for ``rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7`` that keeps
the backbone, AIFI, FPN/PAN, and decoder architectures untouched while directly
addressing the four failure modes diagnosed across prior FSV7 loss ablations:

1. Late-stage classification overfitting under constant LR (val/cls_loss rebound):
   Variation-Bounded Cross-Entropy (VCE) inspired by
   Wang et al., "Variation-Bounded Loss for Noise-Tolerant Learning", AAAI 2026.
   Bounds the variation ratio v(L) = (1 + a) / a < inf via scheduled a(t),
   preventing unbounded gradient memorization on noisy/ambiguous matches.
2. Long-tail class imbalance without head-class FP inflation:
   Generalized Class-Aware (GCA) confidence margin inspired by
   Cortes, Mohri & Zhong, "Improved Balanced Classification with Theoretically
   Grounded Loss Functions", NeurIPS 2025.
   Applies class-prior-derived logit margins (p_bar / p_c)^0.25 and bounded
   class scaling on positive targets.
3. Cross-image same-class quality-score rank inversion (21.6%-31.5%):
   Differentiable Kendall's tau pairwise ranking loss inspired by
   Guan et al., "Enhancing Logits Distillation with Plug&Play Kendall's tau
   Ranking Loss", ICML 2025.
   Applies zero-sum antisymmetric tanh-Kendall ranking across all same-class
   matched positives and hard false positives in the batch, aligning global
   score ordering with IoU without shifting mean confidence upward.
4. High-IoU localization preservation + batch=4 loss-variance stabilization:
   Retains exact native GIoU at IoU >= low_iou_thresh (avoiding high-IoU
   surrogate mismatch), adds a strictly low-IoU center-enclosure pull that
   vanishes identically at IoU >= low_iou_thresh, and applies EMA loss-variance
   calibration inspired by Cao et al., "Towards Calibrated Gradient-Based
   Multi-Task Learning" (VarGrad, CVPR Findings).

Degeneracy guarantee:
   Setting vbl_a_start=0, vbl_a_max=0, gca_margin=0, gca_pow=0, rank_gain=0,
   low_iou_center_gain=0, l1_tiny_soften=0 recovers native RTDETRDetectionLoss.
"""

import math

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou

__all__ = ('RTDETRDetectionLossVBGK',)


class RTDETRDetectionLossVBGK(RTDETRDetectionLoss):
    """Variation-Bounded, GCA-Margin, and Kendall-Calibrated detection loss."""

    innovation_schedule = True

    def __init__(
        self,
        *args,
        # AAAI 2026 VBL (Variation-Bounded Cross-Entropy)
        vbl_a_start=0.0,
        vbl_a_max=0.05,
        vbl_start_epoch=40,
        vbl_full_epoch=180,
        # NeurIPS 2025 GCA (Generalized Class-Aware confidence margin)
        gca_margin=0.35,
        gca_pow=0.20,
        gca_start_epoch=5,
        gca_full_epoch=25,
        class_counts=None,
        # ICML 2025 Differentiable Kendall's tau cross-image ranking
        rank_gain=0.40,
        rank_kappa=2.5,
        rank_delta_iou=0.08,
        rank_margin_scale=0.30,
        rank_pos_min_iou=0.50,
        rank_fp_max_iou=0.35,
        rank_fp_min_score=0.15,
        rank_max_fps_per_class=32,
        rank_start_epoch=8,
        rank_full_epoch=30,
        # Geometry & CVPR Findings VarGrad variance calibration
        low_iou_center_gain=0.45,
        low_iou_thresh=0.30,
        l1_tiny_soften=0.20,
        tiny_tau=0.03125,
        vargrad_strength=0.35,
        vargrad_momentum=0.98,
        **kwargs,
    ):
        kwargs.setdefault('use_vfl', True)
        super().__init__(*args, **kwargs)

        if self.vfl is None or self.fl is None or any((self.sl, self.emasl, self.svfl, self.emasvfl, self.mal)):
            raise ValueError('VBGK requires the native VFL + FocalLoss path (use_vfl=True)')

        scalars = (
            vbl_a_start, vbl_a_max, gca_margin, gca_pow,
            rank_gain, rank_kappa, rank_delta_iou, rank_margin_scale,
            rank_pos_min_iou, rank_fp_max_iou, rank_fp_min_score,
            low_iou_center_gain, low_iou_thresh, l1_tiny_soften, tiny_tau,
            vargrad_strength, vargrad_momentum,
        )
        if not all(math.isfinite(float(x)) for x in scalars):
            raise ValueError('All VBGK scalar hyperparameters must be finite')
        if not (0.0 <= vbl_a_start <= vbl_a_max < 1.0):
            raise ValueError('Require 0 <= vbl_a_start <= vbl_a_max < 1')
        if not (0 <= vbl_start_epoch < vbl_full_epoch):
            raise ValueError('Require 0 <= vbl_start_epoch < vbl_full_epoch')
        if not (0 <= gca_start_epoch < gca_full_epoch):
            raise ValueError('Require 0 <= gca_start_epoch < gca_full_epoch')
        if not (0 <= rank_start_epoch < rank_full_epoch):
            raise ValueError('Require 0 <= rank_start_epoch < rank_full_epoch')
        if gca_margin < 0.0 or gca_pow < 0.0:
            raise ValueError('gca_margin and gca_pow must be non-negative')
        if rank_gain < 0.0 or rank_kappa <= 0.0 or rank_delta_iou < 0.0 or rank_margin_scale < 0.0:
            raise ValueError('Invalid Kendall ranking hyperparameters')
        if not (0.0 <= rank_fp_max_iou <= rank_pos_min_iou <= 1.0):
            raise ValueError('Require 0 <= rank_fp_max_iou <= rank_pos_min_iou <= 1')
        if low_iou_center_gain < 0.0 or not (0.0 < low_iou_thresh <= 1.0):
            raise ValueError('Invalid low_iou_center_gain or low_iou_thresh')
        if not (0.0 <= l1_tiny_soften <= 0.5) or tiny_tau < 0.0:
            raise ValueError('Invalid l1_tiny_soften or tiny_tau')
        if vargrad_strength < 0.0 or not (0.0 <= vargrad_momentum < 1.0):
            raise ValueError('Invalid VarGrad calibration parameters')

        self.vbl_a_start = float(vbl_a_start)
        self.vbl_a_max = float(vbl_a_max)
        self.vbl_start_epoch = int(vbl_start_epoch)
        self.vbl_full_epoch = int(vbl_full_epoch)

        self.gca_margin = float(gca_margin)
        self.gca_pow = float(gca_pow)
        self.gca_start_epoch = int(gca_start_epoch)
        self.gca_full_epoch = int(gca_full_epoch)

        self.rank_gain = float(rank_gain)
        self.rank_kappa = float(rank_kappa)
        self.rank_delta_iou = float(rank_delta_iou)
        self.rank_margin_scale = float(rank_margin_scale)
        self.rank_pos_min_iou = float(rank_pos_min_iou)
        self.rank_fp_max_iou = float(rank_fp_max_iou)
        self.rank_fp_min_score = float(rank_fp_min_score)
        self.rank_max_fps_per_class = int(rank_max_fps_per_class)
        self.rank_start_epoch = int(rank_start_epoch)
        self.rank_full_epoch = int(rank_full_epoch)

        self.low_iou_center_gain = float(low_iou_center_gain)
        self.low_iou_thresh = float(low_iou_thresh)
        self.l1_tiny_soften = float(l1_tiny_soften)
        self.tiny_tau = float(tiny_tau)
        self.vargrad_strength = float(vargrad_strength)
        self.vargrad_momentum = float(vargrad_momentum)

        # Class-prior vectors for GCA (NeurIPS 2025).
        if class_counts is not None:
            counts = torch.as_tensor(class_counts, dtype=torch.float32)
            if counts.numel() != self.nc or (counts <= 0).any():
                raise ValueError(f'class_counts must have {self.nc} positive entries')
            priors = counts / counts.sum()
            mean_prior = 1.0 / self.nc
            ratio = mean_prior / priors
            # GCA margin scales softly with (p_bar / p_c)^0.25 - 1; clamped for stability.
            margin_vec = (ratio.pow(0.25) - 1.0).clamp(-0.8, 1.5)
            weight_vec = ratio.pow(self.gca_pow).clamp(0.85, 1.75) if self.gca_pow > 0 else torch.ones(self.nc)
        else:
            margin_vec = torch.zeros(self.nc, dtype=torch.float32)
            weight_vec = torch.ones(self.nc, dtype=torch.float32)
        self.register_buffer('gca_margin_vec', margin_vec, persistent=False)
        self.register_buffer('gca_weight_vec', weight_vec, persistent=False)

        # Dynamic epoch progress (initialized to 0; standalone calls without set_epoch stay at start).
        self._vbl_ramp = 0.0
        self._gca_ramp = 0.0
        self._rank_ramp = 0.0
        self._vbgk_context = None

        # VarGrad running loss statistics (plain floats to avoid ModelEMA buffer mismatch).
        self._vg_rank_mean = 0.0
        self._vg_rank_var = 0.0
        self._vg_rank_init = False
        self.last_vbgk_stats = {}

    @staticmethod
    def _ramp(epoch, start, full):
        return min(1.0, max(0.0, (float(epoch) + 1.0 - float(start)) / float(full - start)))

    def set_epoch(self, epoch, total_epochs=0):
        """Update epoch-scheduled ramps using the actual trainer epoch."""
        self._vbl_ramp = self._ramp(epoch, self.vbl_start_epoch, self.vbl_full_epoch)
        self._gca_ramp = self._ramp(epoch, self.gca_start_epoch, self.gca_full_epoch)
        self._rank_ramp = self._ramp(epoch, self.rank_start_epoch, self.rank_full_epoch)

    @property
    def current_vbl_a(self):
        return self.vbl_a_start + (self.vbl_a_max - self.vbl_a_start) * self._vbl_ramp

    def _vargrad_scale(self, loss_val):
        """CVPR Findings VarGrad-inspired EMA loss-variance calibration factor."""
        if self.vargrad_strength <= 0.0:
            return 1.0
        val = float(loss_val)
        if not math.isfinite(val):
            return 1.0
        m = self.vargrad_momentum
        if not self._vg_rank_init:
            self._vg_rank_mean = val
            self._vg_rank_var = 0.0
            self._vg_rank_init = True
            return 1.0
        delta = val - self._vg_rank_mean
        self._vg_rank_mean = m * self._vg_rank_mean + (1.0 - m) * val
        self._vg_rank_var = m * self._vg_rank_var + (1.0 - m) * (delta * delta)
        cv = math.sqrt(max(self._vg_rank_var, 0.0)) / (abs(self._vg_rank_mean) + 1e-6)
        return min(1.25, max(0.50, 1.0 / (1.0 + self.vargrad_strength * cv)))

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """Compute base losses + final-layer cross-image Kendall ranking loss."""
        self.last_vbgk_stats = {}
        boxes, _ = preds
        if self.training and torch.is_grad_enabled():
            context = {'final_ptr': boxes[-1].data_ptr(), 'seen': False, 'rank_loss': None}
            self._vbgk_context = context
            try:
                losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
            finally:
                self._vbgk_context = None
            if context['rank_loss'] is not None:
                losses['loss_kendall_rank'] = context['rank_loss']
            return losses
        return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)

    def _get_loss(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        masks=None,
        gt_mask=None,
        postfix='',
        match_indices=None,
    ):
        context = self._vbgk_context if not postfix else None
        is_final = (
            context is not None
            and not context['seen']
            and pred_bboxes.data_ptr() == context['final_ptr']
        )
        if is_final:
            context['seen'] = True
            if match_indices is None:
                match_indices = self.matcher(
                    pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                    masks=masks, gt_mask=gt_mask,
                )

        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                masks=masks, gt_mask=gt_mask,
            )

        idx, gt_idx = self._get_index(match_indices)
        matched_preds = pred_bboxes[idx]
        matched_gts = gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros((bs, nq), device=pred_scores.device)
        if len(matched_gts):
            gt_scores[idx] = bbox_iou(matched_preds.detach(), matched_gts, xywh=True).squeeze(-1)

        loss = {}
        loss.update(self._get_loss_class(pred_scores, targets, gt_scores, len(matched_gts), postfix))
        loss.update(self._get_loss_bbox(matched_preds, matched_gts, postfix))

        if is_final:
            rank_loss = self._kendall_rank_loss(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, match_indices
            )
            context['rank_loss'] = rank_loss

        return loss

    @staticmethod
    def _vb_bce_with_logits(logits, targets, a):
        """AAAI 2026 Variation-Bounded Cross-Entropy (exact BCE when a == 0).

        Uses log-space logaddexp so there is zero exp overflow for any logit value:
            l+(z; a) = -log((sigmoid(z) + a) / (1 + a))
            l-(z; a) = -log((1 - sigmoid(z) + a) / (1 + a))
        """
        if a <= 0.0:
            return F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        log_a = math.log(a)
        log_1pa = math.log(1.0 + a)
        log_p = -F.softplus(-logits)
        log_1mp = -F.softplus(logits)
        loss_pos = log_1pa - torch.logaddexp(log_p, logits.new_full((), log_a))
        loss_neg = log_1pa - torch.logaddexp(log_1mp, logits.new_full((), log_a))
        return targets * loss_pos + (1.0 - targets) * loss_neg

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        """Classification VFL augmented with GCA confidence margins (NeurIPS 2025)
        and Variation-Bounded BCE (AAAI 2026).
        """
        name_class = f'loss_class{postfix}'
        use_custom = self.training and not postfix
        a = self.current_vbl_a if use_custom else 0.0
        gca_r = self._gca_ramp if use_custom else 0.0

        # Exact native fallback if custom classification scheduling is inactive.
        if a <= 0.0 and (gca_r <= 0.0 or (self.gca_margin == 0.0 and self.gca_pow == 0.0)):
            return super()._get_loss_class(pred_scores, targets, gt_scores, num_gts, postfix)

        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        label = one_hot.float()

        if num_gts == 0:
            # Native FocalLoss fallback when batch has zero GTs.
            with torch.cuda.amp.autocast(enabled=False):
                logits_f = pred_scores.float()
                bce = self._vb_bce_with_logits(logits_f, label, a)
                p = logits_f.sigmoid()
                alpha_factor = label * self.fl.alpha + (1.0 - label) * (1.0 - self.fl.alpha)
                modulating = (torch.abs(label - p)) ** self.fl.gamma
                loss_cls = (bce * alpha_factor * modulating).mean(1).sum()
            return {name_class: loss_cls.squeeze() * self.loss_gain['class']}

        if self.gca_margin_vec.device != pred_scores.device:
            self.gca_margin_vec = self.gca_margin_vec.to(pred_scores.device)
            self.gca_weight_vec = self.gca_weight_vec.to(pred_scores.device)

        gt_scores_map = gt_scores.view(bs, nq, 1) * label
        with torch.cuda.amp.autocast(enabled=False):
            logits_f = pred_scores.float()
            # Apply GCA class-dependent confidence margin ONLY to positive target logits.
            if gca_r > 0.0 and self.gca_margin > 0.0:
                margin = (gca_r * self.gca_margin) * self.gca_margin_vec.view(1, 1, self.nc)
                logits_eff = logits_f - label * margin
            else:
                logits_eff = logits_f

            # Apply bounded GCA class weight ONLY to positive targets.
            if gca_r > 0.0 and self.gca_pow > 0.0:
                cls_w = (1.0 + gca_r * (self.gca_weight_vec - 1.0)).view(1, 1, self.nc)
            else:
                cls_w = 1.0

            p_raw = logits_f.sigmoid()
            weight = 0.75 * p_raw.pow(2.0) * (1.0 - label) + gt_scores_map * cls_w * label
            bce = self._vb_bce_with_logits(logits_eff, gt_scores_map, a)
            loss_cls = (bce * weight.detach()).mean(1).sum()

        loss_cls = loss_cls / (max(num_gts, 1) / nq)
        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        """Exact GIoU at IoU >= low_iou_thresh + strictly low-IoU normalized
        center-enclosure pull + mild tiny-box L1 damping.
        """
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'
        loss = {}
        num_boxes = len(gt_bboxes)
        if num_boxes == 0:
            loss[name_bbox] = torch.tensor(0.0, device=pred_bboxes.device)
            loss[name_giou] = torch.tensor(0.0, device=pred_bboxes.device)
            return loss

        with torch.cuda.amp.autocast(enabled=False):
            pred_f = pred_bboxes.float()
            gt_f = gt_bboxes.detach().float()

            # 1. Mild tiny-box L1 damping (constant L1 grad is relatively noisy for <16px boxes).
            l1 = (pred_f - gt_f).abs().sum(-1)
            if self.l1_tiny_soften > 0.0 and self.tiny_tau > 0.0 and not postfix:
                with torch.no_grad():
                    s = (gt_f[:, 2] * gt_f[:, 3]).clamp_min(0.0).sqrt()
                    lam = (1.0 - s / self.tiny_tau).clamp(0.0, 1.0).square()
                    w_l1 = 1.0 - self.l1_tiny_soften * lam
                l1_term = (l1 * w_l1).sum() / num_boxes
            else:
                l1_term = l1.sum() / num_boxes
            loss[name_bbox] = self.loss_gain['bbox'] * l1_term

            # 2. Pure native GIoU everywhere (never suppressed at IoU=0, never altered at high IoU).
            giou_loss = 1.0 - bbox_iou(pred_f, gt_f, xywh=True, GIoU=True).squeeze(-1)

            # 3. Strictly low-IoU center-enclosure alignment assist (identically 0 for IoU >= low_iou_thresh).
            if self.low_iou_center_gain > 0.0 and not postfix:
                with torch.no_grad():
                    iou_plain = bbox_iou(pred_f.detach(), gt_f, xywh=True).squeeze(-1)
                    gate = (1.0 - (iou_plain / self.low_iou_thresh).clamp(0.0, 1.0)).square()
                if gate.any():
                    p_half = pred_f[:, 2:] * 0.5
                    g_half = gt_f[:, 2:] * 0.5
                    enc_lo = torch.minimum(pred_f[:, :2] - p_half, gt_f[:, :2] - g_half)
                    enc_hi = torch.maximum(pred_f[:, :2] + p_half, gt_f[:, :2] + g_half)
                    c2 = (enc_hi - enc_lo).square().sum(dim=-1).clamp_min(1e-7)
                    rho2 = (pred_f[:, :2] - gt_f[:, :2]).square().sum(dim=-1)
                    giou_loss = giou_loss + self.low_iou_center_gain * gate * (rho2 / c2)

            loss[name_giou] = self.loss_gain['giou'] * giou_loss.sum() / num_boxes

        return {k: v.squeeze() for k, v in loss.items()}

    def _kendall_rank_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, match_indices):
        """ICML 2025 Differentiable Kendall's tau cross-image quality-logit ranking
        over same-class matched positives and hard false positives, calibrated by
        CVPR Findings VarGrad loss-variance dynamics.
        """
        zero = pred_scores.sum() * 0.0
        eff_gain = self.rank_gain * self._rank_ramp
        if eff_gain <= 0.0 or not len(gt_bboxes):
            return zero

        with torch.cuda.amp.autocast(enabled=False):
            boxes_f = pred_bboxes.detach().float()
            scores_f = pred_scores.float()
            gt_f = gt_bboxes.detach().float()
            gt_cls_l = gt_cls.long().view(-1)

            # Gather cross-image matched positives across the batch.
            idx, gt_idx = self._get_index(match_indices)
            if not len(gt_idx):
                return zero
            pos_cls = gt_cls_l[gt_idx]
            pos_logits = scores_f[idx[0], idx[1], pos_cls]
            pos_ious = bbox_iou(boxes_f[idx], gt_f[gt_idx], xywh=True).squeeze(-1).clamp(0.0, 1.0)
            if self.tiny_tau > 0.0:
                pos_s = (gt_f[gt_idx, 2] * gt_f[gt_idx, 3]).clamp_min(0.0).sqrt()
                pos_small = (1.0 - pos_s / self.tiny_tau).clamp(0.0, 1.0)
            else:
                pos_small = torch.zeros_like(pos_ious)

            # Gather high-scoring hard false positives across the batch (unmatched or low same-class IoU).
            fp_logits_list, fp_cls_list, fp_ious_list = [], [], []
            bs = len(gt_groups)
            offset = 0
            for b in range(bs):
                cnt = int(gt_groups[b])
                src_b, _ = match_indices[b]
                unmatched = torch.ones(scores_f.shape[1], dtype=torch.bool, device=scores_f.device)
                if len(src_b):
                    unmatched[src_b.to(scores_f.device)] = False

                probs_b = scores_f[b].detach().sigmoid()
                top_prob, top_cls = probs_b.max(dim=-1)
                cand_mask = unmatched & (top_prob >= self.rank_fp_min_score)
                if not cand_mask.any():
                    offset += cnt
                    continue

                cand_q_idx = cand_mask.nonzero(as_tuple=False).squeeze(-1)
                cand_cls = top_cls[cand_q_idx]
                if cnt > 0:
                    g_b = gt_f[offset:offset + cnt]
                    gc_b = gt_cls_l[offset:offset + cnt]
                    overlaps = bbox_iou(boxes_f[b, cand_q_idx][:, None], g_b[None], xywh=True).squeeze(-1)
                    same_cls = cand_cls[:, None] == gc_b[None, :]
                    max_same_iou = overlaps.masked_fill(~same_cls, 0.0).amax(dim=-1)
                else:
                    max_same_iou = torch.zeros(len(cand_q_idx), device=scores_f.device)

                hard_fp = max_same_iou <= self.rank_fp_max_iou
                if hard_fp.any():
                    sel_q = cand_q_idx[hard_fp]
                    sel_cls = cand_cls[hard_fp]
                    fp_logits_list.append(scores_f[b, sel_q, sel_cls])
                    fp_cls_list.append(sel_cls)
                    fp_ious_list.append(max_same_iou[hard_fp])
                offset += cnt

            if fp_logits_list:
                fp_logits = torch.cat(fp_logits_list)
                fp_cls = torch.cat(fp_cls_list)
                fp_ious = torch.cat(fp_ious_list)
            else:
                fp_logits = pos_logits.new_empty((0,))
                fp_cls = pos_cls.new_empty((0,))
                fp_ious = pos_ious.new_empty((0,))

            total_weighted_loss = zero
            total_weight = 0.0

            for c in pos_cls.unique().tolist():
                pm = pos_cls == c
                p_log = pos_logits[pm]
                p_q = pos_ious[pm]
                p_sm = pos_small[pm]

                # Positive anchors that are well-localized enough to outrank lower-quality peers.
                hi_mask = p_q >= self.rank_pos_min_iou
                if not hi_mask.any():
                    continue
                hi_log = p_log[hi_mask]
                hi_q = p_q[hi_mask]
                hi_sm = p_sm[hi_mask]

                # Competitors in the same class across the batch: all matched positives + capped hard FPs.
                fm = fp_cls == c
                if fm.any():
                    f_log = fp_logits[fm]
                    f_q = fp_ious[fm]
                    if len(f_log) > self.rank_max_fps_per_class:
                        _, topk_idx = f_log.detach().topk(self.rank_max_fps_per_class)
                        f_log = f_log[topk_idx]
                        f_q = f_q[topk_idx]
                    comp_log = torch.cat((p_log, f_log))
                    comp_q = torch.cat((p_q, f_q))
                else:
                    comp_log = p_log
                    comp_q = p_q

                q_diff = hi_q[:, None] - comp_q[None, :]
                valid = q_diff > self.rank_delta_iou
                if not valid.any():
                    continue

                margin = self.rank_margin_scale * q_diff
                logit_diff = hi_log[:, None] - comp_log[None, :] - margin
                pair_loss = 1.0 - torch.tanh(self.rank_kappa * logit_diff)
                pair_weight = q_diff * (1.0 + 0.5 * hi_sm[:, None])

                valid_w = pair_weight[valid]
                total_weighted_loss = total_weighted_loss + (pair_loss[valid] * valid_w).sum()
                total_weight = total_weight + float(valid_w.sum().item())

            if total_weight <= 0.0:
                return zero

            raw_rank = total_weighted_loss / total_weight
            vg_scale = self._vargrad_scale(raw_rank.detach())
            self.last_vbgk_stats = {
                'vbl_a': self.current_vbl_a,
                'gca_ramp': self._gca_ramp,
                'rank_ramp': self._rank_ramp,
                'vargrad_scale': vg_scale,
                'raw_rank': float(raw_rank.detach()),
            }
            return eff_gain * vg_scale * raw_rank
