"""SQA small-box regression with final-decoder false-positive ranking."""

import math

import torch
import torch.nn.functional as F

from ultralytics.utils.loss_plugins.sqa_loss import RTDETRDetectionLossSQA
from ultralytics.utils.metrics import bbox_iou

__all__ = ('RTDETRDetectionLossSQAv2',)


class RTDETRDetectionLossSQAv2(RTDETRDetectionLossSQA):
    """Penalize a false-positive logit above a same-image, same-class true match.

    Mine at most one reference per false positive. The reference is the
    lowest-scoring eligible match of its class. Only the false-positive
    logit receives ranking gradients, and the native assignment is reused.
    """

    def __init__(self, *args, cls_gain=0., rank_gain=.1, rank_pos_iou=.5,
                 rank_neg_iou=.4, **kwargs):
        if cls_gain != 0:
            raise ValueError('SQAv2 requires cls_gain=0 to disable SQA score boosting')
        if not math.isfinite(rank_gain) or rank_gain < 0:
            raise ValueError('SQAv2 rank_gain must be finite and nonnegative')
        if not 0 <= rank_neg_iou < rank_pos_iou <= 1:
            raise ValueError('SQAv2 needs 0 <= rank_neg_iou < rank_pos_iou <= 1')
        super().__init__(*args, cls_gain=cls_gain, **kwargs)
        self.rank_gain = float(rank_gain)
        self.rank_pos_iou = float(rank_pos_iou)
        self.rank_neg_iou = float(rank_neg_iou)
        self._sqa_rank_context = None

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        if not (self.training and torch.is_grad_enabled() and self._ramp and self.rank_gain):
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)

        context = {'final_ptr': preds[0][-1].data_ptr()}
        self._sqa_rank_context = context
        try:
            losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        finally:
            self._sqa_rank_context = None
        # Add the ranking key after the parent assembles regular and DN losses.
        losses['loss_sqa_rank'] = context['loss_sqa_rank']
        return losses

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        context = self._sqa_rank_context
        final = (context is not None and not postfix
                 and pred_bboxes.data_ptr() == context['final_ptr'])
        if final:
            self._sqa_rank_context = None
            if match_indices is None:
                match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                             masks=masks, gt_mask=gt_mask)

        losses = super()._get_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                   masks=masks, gt_mask=gt_mask, postfix=postfix,
                                   match_indices=match_indices)
        if final:
            rank, count = self._ranking_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls,
                                             gt_groups, match_indices)
            context['loss_sqa_rank'] = self._ramp * self.rank_gain * rank
            self.last_sqa_stats.setdefault('final', {}).update(rank_pairs=count, rank_loss=rank.detach())
        return losses

    @torch.no_grad()
    def _rank_references(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, match_indices):
        boxes = pred_bboxes.detach().float()
        gt_boxes = gt_bboxes.detach().float()
        gt_cls = gt_cls.reshape(-1)
        score, pred_cls = pred_scores.detach().float().max(-1)
        # +inf is used for queries without an eligible reference, yielding zero hinge loss.
        references = torch.full_like(score, torch.inf)
        offset = 0
        for image, (num_gt, (src, dst)) in enumerate(zip(gt_groups, match_indices)):
            gt_slice = slice(offset, offset + num_gt)
            local_dst = dst.to(boxes.device) - offset
            offset += num_gt
            if not src.numel():
                continue

            src = src.to(boxes.device)
            classes = pred_cls[image]
            local_classes = gt_cls[gt_slice]
            overlaps = bbox_iou(boxes[image, :, None], gt_boxes[None, gt_slice],
                                xywh=True).squeeze(-1)
            same_class = classes[:, None] == local_classes[None, :]
            max_same_iou = overlaps.masked_fill(~same_class, 0.).amax(dim=1)
            valid_ref = same_class[src, local_dst] & (overlaps[src, local_dst] >= self.rank_pos_iou)
            same_ref_class = classes[:, None] == local_classes[local_dst][None, :]
            ref_logits = score[image, src].masked_fill(~(same_ref_class & valid_ref), torch.inf)
            references[image] = ref_logits.amin(dim=1).masked_fill(max_same_iou > self.rank_neg_iou, torch.inf)
        return pred_cls, references

    def _ranking_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, match_indices):
        classes, references = self._rank_references(pred_bboxes, pred_scores, gt_bboxes, gt_cls,
                                                   gt_groups, match_indices)
        logits = pred_scores.gather(-1, classes.unsqueeze(-1)).squeeze(-1).float()
        gaps = F.relu(logits - references)
        count = gaps.detach().gt(0).sum()
        return gaps.sum() / count.clamp_min(1), count
