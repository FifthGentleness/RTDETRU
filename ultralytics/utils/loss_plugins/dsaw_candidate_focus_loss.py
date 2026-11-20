"""Small-object candidate and localization supervision for FSV7.

The auxiliary assignment score follows PaQ-DETR (CVPR 2026). The bounded
edge gradients are motivated by UGS (ICCV 2025). The final-box overlap
window adapts the FoCIoU mapping from YOLO-ULM (CVPR 2026).
"""

import math

import torch
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy

from .dsaw_v4_loss import RTDETRDetectionLossDSAWV4


class RTDETRDetectionLossDSAWCandidateFocus(RTDETRDetectionLossDSAWV4):
    """DSAW-v4 with early candidate regression, final IoU focus and score ordering."""

    innovation_schedule = True

    def __init__(
        self,
        *args,
        candidate_gain=0.65,
        focus_gain=0.16,
        rank_gain=0.30,
        geometry_epochs=(8, 28),
        rank_epochs=(24, 48),
        candidate_fade_epochs=(160, 220),
        focus_iou=(0.55, 0.90),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if any(not math.isfinite(gain) or gain < 0 for gain in (candidate_gain, focus_gain, rank_gain)):
            raise ValueError("Loss gains must be finite and non-negative.")
        for start, end in (geometry_epochs, rank_epochs, candidate_fade_epochs):
            if not (int(start) == start and int(end) == end and 0 <= start < end):
                raise ValueError("Schedule boundaries must be increasing non-negative integer epochs.")
        if candidate_fade_epochs[0] < geometry_epochs[1]:
            raise ValueError("Candidate fade must start at or after the geometry ramp ends.")
        if not 0 <= focus_iou[0] < focus_iou[1] <= 1:
            raise ValueError("focus_iou must define an increasing subinterval of [0, 1].")

        self.candidate_gain = float(candidate_gain)
        self.focus_gain = float(focus_gain)
        self.rank_gain = float(rank_gain)
        self.geometry_epochs = tuple(geometry_epochs)
        self.rank_epochs = tuple(rank_epochs)
        self.candidate_fade_epochs = tuple(candidate_fade_epochs)
        self.focus_iou = tuple(focus_iou)
        self._candidate_weight = self._focus_weight = self._rank_weight = 0.0
        self._regular_matches = None

    def set_epoch(self, epoch, total_epochs=0):
        def ramp(boundaries):
            start, end = boundaries
            return min(1.0, max(0.0, (epoch + 1 - start) / (end - start)))

        geometry = ramp(self.geometry_epochs)
        self._candidate_weight = self.candidate_gain * geometry * (1.0 - ramp(self.candidate_fade_epochs))
        self._focus_weight = self.focus_gain * geometry
        self._rank_weight = self.rank_gain * ramp(self.rank_epochs)

    def _get_loss(
        self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
        masks=None, gt_mask=None, postfix='', match_indices=None,
    ):
        if self._regular_matches is not None and not postfix:
            if match_indices is None:
                match_indices = self.matcher(
                    pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
                )
            self._regular_matches.append(match_indices)
        return super()._get_loss(
            pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
            masks=masks, gt_mask=gt_mask, postfix=postfix, match_indices=match_indices,
        )

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        enabled = self._candidate_weight or self._focus_weight or self._rank_weight
        if not (enabled and self.training and torch.is_grad_enabled() and len(batch['bboxes'])):
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)

        # The inherited criterion visits the final layer first, then encoder and
        # intermediate decoder layers. Reuse its assignments without another solve.
        stage_matches = []
        self._regular_matches = stage_matches
        try:
            losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        finally:
            self._regular_matches = None

        boxes, scores = preds
        gt_boxes = batch['bboxes'].detach().float()
        gt_cls = batch['cls']
        gt_groups = batch['gt_groups']
        image_h, image_w = batch.get('image_hw', (640, 640))
        pixel_scale = gt_boxes.new_tensor([image_w, image_h])
        short_side = (gt_boxes[:, 2:] * pixel_scale).amin(dim=1)
        # Full weight at 8–16 px with a taper for the noisiest tiny boxes.
        size_weight = (short_side / 8).clamp(0, 1) * ((24 - short_side) / 8).clamp(0, 1)

        final_matches = [(src.to(boxes.device), dst.to(boxes.device)) for src, dst in stage_matches[0]]
        indices, target = self._get_index(final_matches)
        final_boxes = boxes[-1][indices].float()
        final_scores = scores[-1][indices].float()
        quality = self._iou(final_boxes, gt_boxes[target])
        correct = final_scores.detach().argmax(dim=1) == gt_cls[target]
        normalizer = max(len(target), 1)

        if self._focus_weight:
            low, high = self.focus_iou
            in_window = (quality.detach() > low) & (quality.detach() < high)
            shaped = ((quality - low) / (high - low)).clamp(0, 1).square()
            cost = (1 - shaped) * size_weight[target] * (correct & in_window)
            losses['loss_candidate_focus_iou'] = self._focus_weight * cost.sum() / normalizer

        if self._rank_weight:
            logits = final_scores.gather(1, gt_cls[target, None]).squeeze(1)
            rank = self._score_order_loss(logits, quality, correct, size_weight[target], gt_cls[target])
            losses['loss_candidate_focus_rank'] = self._rank_weight * rank.sum() / normalizer

        if self._candidate_weight and len(stage_matches) > 1:
            final_quality_by_gt = torch.zeros_like(size_weight)
            final_quality_by_gt[target] = quality.detach()
            # Only the selected encoder proposals and the first intermediate decoder
            # output can get an extra regression target.
            candidate_losses = [
                self._candidate_loss(
                    boxes[i], scores[i], gt_boxes, gt_cls, gt_groups, matches,
                    final_matches, final_quality_by_gt, size_weight, pixel_scale,
                )
                for i, matches in enumerate(stage_matches[1:3])
            ]
            losses['loss_candidate_focus_early'] = self._candidate_weight * torch.stack(candidate_losses).mean()
        return losses

    @staticmethod
    def _iou(boxes, targets):
        return bbox_iou(boxes, targets, xywh=True, eps=1e-9).squeeze(-1)

    @staticmethod
    def _score_order_loss(logits, quality, correct, size_weight, classes):
        """Raise only an inverted high-quality score, with its IoU as a ceiling."""
        if not len(logits):
            return logits
        with torch.no_grad():
            probability = logits.sigmoid()
            thresholds = quality.new_tensor(range(50, 96, 5)) / 100
            passed = (quality[:, None] >= thresholds).sum(dim=1)
            competitor = correct[None, :] & (passed[None, :] >= 1)
            competitor = competitor & (passed[:, None] - passed[None, :] >= 2)
            competitor = competitor & (classes[:, None] == classes[None, :])
            lower_score = probability[None, :].expand_as(competitor).masked_fill(~competitor, 0).amax(dim=1)
            target = (lower_score + 0.03).minimum(quality)
            active = correct & (passed >= 5) & (lower_score > probability) & (target > probability)
            target = torch.where(active, target, probability)

        cost = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        entropy = torch.special.entr(target) + torch.special.entr(1 - target)
        return (cost - entropy).clamp_min(0) * size_weight * active

    def _candidate_loss(
        self, boxes, scores, gt_boxes, gt_cls, gt_groups, matches,
        final_matches, final_quality, size_weight, pixel_scale,
    ):
        cost = boxes.new_zeros((), dtype=torch.float32)
        offset = 0
        for image, (count, match, final_match) in enumerate(zip(gt_groups, matches, final_matches)):
            gt_slice = slice(offset, offset + count)
            if count:
                queries, target, quality = self._mine_candidates(
                    boxes[image].detach().float(), scores[image].detach().float(),
                    gt_boxes[gt_slice], gt_cls[gt_slice], match, final_match, offset,
                    final_quality[gt_slice], size_weight[gt_slice],
                )
                gt = gt_boxes[target]
                # Normalize the edge gradients by the corresponding GT axis.
                # The 8 px lower bound also caps the gradient for tiny boxes.
                min_extent = 8.0 / pixel_scale
                divisor = gt[:, 2:].maximum(min_extent).repeat(1, 2)
                residual = (xywh2xyxy(boxes[image, queries].float()) - xywh2xyxy(gt)) / divisor
                edge = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=0.06, reduction='none')
                cost = cost + (edge.mean(dim=-1) * quality * size_weight[target]).sum()
            offset += count
        return cost / max(sum(len(src) for src, _ in matches), 1)

    @torch.no_grad()
    def _mine_candidates(
        self, boxes, scores, gt_boxes, gt_cls, match, final_match, offset, final_quality, size_weight,
    ):
        overlap = self._iou(boxes[:, None], gt_boxes[None])
        values, owners = overlap.topk(min(len(gt_boxes), 2), dim=1)
        quality, owner = values[:, 0], owners[:, 0]
        second_quality = values[:, 1] if len(gt_boxes) > 1 else 0.0

        src, dst = (index.to(boxes.device) for index in match)
        dst = dst - offset
        primary_quality = torch.zeros_like(final_quality)
        primary_quality[dst] = overlap[src, dst]
        free = torch.ones_like(quality, dtype=torch.bool)
        free[src] = False

        final_src, final_dst = final_match
        final_owner = torch.full_like(owner, -1)
        final_owner[final_src] = final_dst - offset
        compatible = (final_owner == -1) | (final_owner == owner)
        eligible = (
            free & compatible & (size_weight[owner] > 0)
            & (final_quality[owner] < 0.75) & (primary_quality[owner] < 0.70)
            & (quality >= 0.45) & (quality - second_quality >= 0.05)
            & (quality >= primary_quality[owner] - 0.05)
        )

        query = torch.arange(len(boxes), device=boxes.device)
        matchability = quality - 0.4 * scores[query, gt_cls[owner]].sigmoid()
        matchability = matchability.masked_fill(~eligible, -torch.inf)
        # Each query has one unambiguous owner. Pick only one query per owner.
        choices = torch.full_like(overlap, -torch.inf)
        choices[query, owner] = matchability
        best, queries = choices.max(dim=0)
        keep = best.isfinite()
        queries = queries[keep]
        targets = torch.arange(len(gt_boxes), device=boxes.device)[keep] + offset
        return queries, targets, quality[queries]
