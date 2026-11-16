"""AP-threshold ranking and box correction on final regular queries."""

import math

import torch
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou

from .dsaw_v4_iou_only_loss import RTDETRDetectionLossDSAWV4IoUOnly
from .dsaw_v4_loss import RTDETRDetectionLossDSAWV4


class _TCRLossMixin:
    """Use the base criterion's matches for extra supervision on final regular queries."""

    innovation_schedule = True
    thresholds = tuple(t / 100 for t in range(50, 100, 5))
    threshold_guard = 0.015
    rank_margin = 0.20
    rank_temperature = 0.50
    score_floor = 0.20
    cross_band = 0.06
    cross_target_pad = 0.01

    def __init__(self, *args, rank_gain=0.25, cross_gain=0.50, start_epoch=15, full_epoch=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_gain, self.cross_gain = float(rank_gain), float(cross_gain)
        if not all(math.isfinite(g) and g >= 0.0 for g in (self.rank_gain, self.cross_gain)):
            raise ValueError('TCR rank_gain and cross_gain must be finite and non-negative')
        if not (int(start_epoch) == start_epoch and int(full_epoch) == full_epoch and 0 <= start_epoch < full_epoch):
            raise ValueError('TCR requires integer epochs with 0 <= start_epoch < full_epoch')
        self.start_epoch, self.full_epoch = int(start_epoch), int(full_epoch)
        self._ramp = 0.0
        self._tcr_context = None
        self.last_tcr_stats = {}

    def set_epoch(self, epoch, total_epochs=0):
        progress = (epoch + 1 - self.start_epoch) / (self.full_epoch - self.start_epoch)
        self._ramp = min(1.0, max(0.0, progress))

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        self.last_tcr_stats = {}
        if not (self.training and torch.is_grad_enabled() and self._ramp and (self.rank_gain or self.cross_gain)):
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)

        # The parent visits final regular queries first, then auxiliary layers and DN.
        context = {}
        self._tcr_context = context
        try:
            losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        finally:
            self._tcr_context = None
        # Append after the parent assembles DN keys to avoid creating TCR DN terms.
        losses.update(context)
        return losses

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        context = self._tcr_context if not postfix else None
        if context is not None:
            self._tcr_context = None
            if match_indices is None:
                match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                             masks=masks, gt_mask=gt_mask)

        losses = super()._get_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                   masks=masks, gt_mask=gt_mask, postfix=postfix, match_indices=match_indices)
        if context is not None:
            rank, cross = self._tcr_terms(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, match_indices)
            context['loss_tcr_rank'] = self._ramp * self.rank_gain * rank
            context['loss_tcr_cross'] = self._ramp * self.cross_gain * cross
        return losses

    @staticmethod
    def _query_overlaps(boxes, predicted_cls, gt_boxes, gt_cls, gt_groups, match_indices):
        """Per-query overlap summaries; all returned GT indices are batch-global."""
        best_q = boxes.new_zeros(boxes.shape[:2])
        matched_q = torch.zeros_like(best_q)
        same_cls_q = torch.zeros_like(best_q)
        alt_cls_q = torch.zeros_like(best_q)
        owner = torch.full_like(predicted_cls, -1)
        exclusive = torch.zeros_like(predicted_cls, dtype=torch.bool)
        offset = 0
        for b, (count, (src, dst)) in enumerate(zip(gt_groups, match_indices)):
            if not count:
                continue
            q = bbox_iou(boxes[b, :, None], gt_boxes[None, offset:offset + count], xywh=True).squeeze(-1)
            local_cls = gt_cls[offset:offset + count]
            top_q, top = q.topk(min(2, count), dim=-1)
            local_owner = top[:, 0]
            best_q[b] = top_q[:, 0]
            owner[b] = local_owner + offset
            runner_up = top_q[:, 1] if count > 1 else 0.0
            exclusive[b] = (best_q[b] - runner_up >= 0.10) & (predicted_cls[b] == local_cls[local_owner])

            same_cls = predicted_cls[b, :, None] == local_cls[None, :]
            same_cls_q[b] = q.masked_fill(~same_cls, 0.0).amax(dim=-1)
            alternate = torch.arange(count, device=boxes.device)[None, :] != local_owner[:, None]
            alt_cls_q[b] = q.masked_fill(~(same_cls & alternate), 0.0).amax(dim=-1)
            matched_q[b, src] = q[src, dst - offset]
            offset += count
        return tuple(x.flatten() for x in (best_q, matched_q, same_cls_q, alt_cls_q, owner, exclusive))

    @staticmethod
    def _top_pairs(pair_mask, winner_mask, candidate_mask, candidate_logits):
        """Take the highest logit per winner/threshold; cap reuse at two per candidate/threshold."""
        # Winners are already ordered by ascending logit, so a stable grouping by
        # negative query puts the largest ranking violations first in each group.
        valid = pair_mask[None] & winner_mask[:, :, None] & candidate_mask[:, None, :]
        values, chosen = candidate_logits.masked_fill(~valid, -torch.inf).max(dim=-1)
        sentinel = len(candidate_logits)
        key = chosen.masked_fill(~torch.isfinite(values), sentinel)
        order = key.argsort(dim=-1, stable=True)
        grouped = key.gather(1, order)
        keep = grouped != sentinel
        keep[:, 2:] &= grouped[:, 2:] != grouped[:, :-2]
        selected = torch.empty_like(keep).scatter_(1, order, keep)
        return chosen, selected

    @torch.no_grad()
    def _mine(self, boxes, scores, gt_boxes, gt_cls, gt_groups, match_indices):
        score, pred_cls = scores.detach().float().max(dim=-1)
        logits = score.flatten()
        offset = boxes.shape[1]
        device_matches = [(src.to(boxes.device), dst.to(boxes.device)) for src, dst in match_indices]
        (batch_index, query), gt = self._get_index(device_matches)
        rows = batch_index * offset + query
        order = logits[rows].argsort(stable=True)
        rows, gt = rows[order], gt[order]
        pred_cls_flat = pred_cls.flatten()
        probability = logits.sigmoid()

        best, quality, same_q, other_q, owner, exclusive = self._query_overlaps(
            boxes.detach().float(), pred_cls, gt_boxes.detach().float(), gt_cls, gt_groups, device_matches)
        correct = pred_cls_flat[rows] == gt_cls[gt]
        matched_quality = quality[rows]
        thresholds = logits.new_tensor(self.thresholds)
        threshold = thresholds[:, None]
        winner = correct & (matched_quality >= threshold + self.threshold_guard)
        margin = logits[rows] - self.rank_margin
        unmatched = torch.ones_like(pred_cls_flat, dtype=torch.bool)
        unmatched[rows] = False

        rank_pairs = []
        if self.rank_gain:
            # D: an unmatched query with an unambiguous owner among all GT, and
            # no alternative same-class GT that it could explain at the threshold.
            ambiguous_alt = other_q >= threshold - self.threshold_guard
            duplicate = ((gt[:, None] == owner) & (best <= matched_quality[:, None] - 0.03)
                         & (logits > margin[:, None]))
            duplicate_ok = unmatched & exclusive & (best >= 0.50) & ~ambiguous_alt
            rank_pairs.append((*self._top_pairs(duplicate, winner, duplicate_ok, logits), False))

            # F: clear background only; it can compete with positives in other images.
            background = ((gt_cls[gt, None] == pred_cls_flat) & (logits > margin[:, None]))
            background_ok = unmatched & (best <= 0.10) & (probability >= self.score_floor)
            rank_pairs.append((*self._top_pairs(background, winner, background_ok[None], logits), False))

        # M: these are valid matches at lower thresholds. Limit negative mining
        # to the matching rows and never backpropagate rank through their logits.
        matched_logits = logits[rows]
        matched_cls = gt_cls[gt]
        matched_pairs = ((matched_cls[:, None] == matched_cls) & (matched_logits > margin[:, None]))
        matched_ok = correct & (matched_quality >= 0.40) & (probability[rows] >= self.score_floor)
        if self.rank_gain:
            calibrated = probability[rows] < (matched_quality + 0.03).clamp_max(0.98)
            rank_ok = matched_ok & (same_q[rows] <= threshold - self.threshold_guard)
            selected, keep = self._top_pairs(matched_pairs, winner & calibrated, rank_ok, matched_logits)
            rank_pairs.append((rows[selected], keep, True))

        cross_target = torch.zeros_like(matched_quality)
        if self.cross_gain:
            repair_threshold = (threshold >= 0.60) & (threshold <= 0.85)
            repair_ok = (matched_ok & (same_q[rows] < threshold) & (matched_quality >= threshold - self.cross_band)
                         & repair_threshold)
            selected, keep = self._top_pairs(matched_pairs, winner, repair_ok, matched_logits)
            # Collapse repeated requests for the same box to its nearest eligible threshold.
            num_matches = len(rows)
            taken = selected.new_zeros((len(thresholds), num_matches + 1))
            taken.scatter_add_(1, selected.masked_fill(~keep, num_matches), keep.long())
            active = taken[:, :num_matches] > 0
            nearest = torch.where(active, threshold, torch.inf).amin(dim=0)
            cross_target = torch.where(active.any(dim=0), nearest + self.cross_target_pad, 0.0)

        return rows, gt, pred_cls_flat, rank_pairs, cross_target

    def _tcr_terms(self, boxes, scores, gt_boxes, gt_cls, gt_groups, match_indices):
        zero = boxes.new_zeros((), dtype=torch.float32)
        num_matches = sum(len(src) for src, _ in match_indices)
        if not num_matches:
            return zero, zero

        rows, gt, classes, rank_pairs, cross_target = self._mine(
            boxes, scores, gt_boxes, gt_cls, gt_groups, match_indices)
        rank, cross = zero, zero
        stats = {'matches': num_matches}
        if self.rank_gain:
            logits = scores.float().flatten(0, 1).gather(1, classes[:, None]).squeeze(1)
            winner = logits[rows]
            for name, (candidates, keep, stop_negative_grad) in zip(('duplicate', 'background', 'matched'), rank_pairs):
                competitor = logits[candidates]
                if stop_negative_grad:
                    competitor = competitor.detach()
                gap = self.rank_margin - winner + competitor
                cost = self.rank_temperature * F.softplus(gap / self.rank_temperature)
                rank = rank + cost.masked_fill(~keep, 0.0).sum()
                stats[f'{name}_pairs'] = keep.sum()
            rank = rank / (3 * len(self.thresholds) * num_matches)

        if self.cross_gain:
            live_q = bbox_iou(boxes.float().flatten(0, 1)[rows], gt_boxes.detach().float()[gt], xywh=True).squeeze(-1)
            cross = (cross_target - live_q).clamp_min(0.0).sum() / num_matches
            stats['cross_boxes'] = (cross_target > 0).sum()
        self.last_tcr_stats = {**stats, 'rank': rank.detach(), 'cross': cross.detach()}
        return rank, cross


class RTDETRDetectionLossDSAWV8TCR(_TCRLossMixin, RTDETRDetectionLossDSAWV4):
    """Original TCR on complete DSAWv4, retained for existing experiments."""


class RTDETRDetectionLossDSAWV4IoUOnlyTCR(_TCRLossMixin, RTDETRDetectionLossDSAWV4IoUOnly):
    """TCR on native VFL, plain L1, and the DSAWv4 IoU blend."""
