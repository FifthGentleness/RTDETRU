"""Scale-stable Quality Alignment (SQA) for the original RT-DETR criterion.

Use loss_name: sqa in the model YAML and train through the normal train.py.
Only current Hungarian matches, student predictions and GT are used. SQA has
no learned parameters or running quality state. Epochs are zero-based in
set_epoch(): the first start_epoch epochs use the original criterion exactly.
"""

import math

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou

__all__ = ('RTDETRDetectionLossSQA',)


class RTDETRDetectionLossSQA(RTDETRDetectionLoss):
    """Correct underconfident positives and blend small-box L1 with relative Huber.

    The encoder prediction must be stage 0 and the final regular decoder stage
    must be last, as assembled by RTDETRDetectionModel.loss(). Intermediate
    decoder and DN supervision retain the original criterion. Each selected
    stage reuses its original matching result, including use_uni_match.

    For matched class y, the classification addition is
        stopgrad(1[q >= quality_min] * max(q - sigmoid(z_y), 0)) * BCE(z_y, q).
    Thus only an underconfident true-class logit receives extra supervision;
    neither box coordinates nor the detached target/weight receive gradients.

    Regression blends the ORIGINAL L1 with scale-normalized Smooth L1 on
    center offsets and stabilized log side ratios. The blend uses GT size,
    stays in [0, reg_mix], and is zero for sqrt(area) >= size_max_px or
    short side >= short_max_px. GIoU is unchanged. Multiplication by
    relative_scale * side_floor_px / image_axis controls output gradients.
    All reductions use the original number of matches, including inactive GTs.
    """

    innovation_schedule = True

    def __init__(self, *args, cls_gain=.5, reg_mix=.35, enc_gain=.5, quality_min=.3,
                 size_max_px=32., short_max_px=16., side_floor_px=8., relative_scale=2.,
                 huber_beta=.1, start_epoch=10, full_epoch=40, **kwargs):
        super().__init__(*args, **kwargs)
        values = (cls_gain, reg_mix, enc_gain, quality_min, size_max_px, short_max_px,
                  side_floor_px, relative_scale, huber_beta)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('SQA parameters must be finite')
        if not (cls_gain >= 0 and 0 <= reg_mix <= 1 and 0 <= enc_gain <= 1 and 0 <= quality_min <= 1):
            raise ValueError('SQA needs cls_gain >= 0 and reg_mix/enc_gain/quality_min in [0, 1]')
        if not (0 < side_floor_px < short_max_px <= size_max_px and relative_scale > 0 and huber_beta > 0):
            raise ValueError('SQA needs 0 < side_floor_px < short_max_px <= size_max_px and positive Huber scales')
        if (not all(math.isfinite(v) and int(v) == v for v in (start_epoch, full_epoch))
                or not 0 <= start_epoch < full_epoch):
            raise ValueError('SQA needs integer epochs 0 <= start_epoch < full_epoch')
        if enc_gain and not self.aux_loss:
            raise ValueError('SQA enc_gain requires the original encoder auxiliary supervision')
        self.cls_gain, self.reg_mix, self.enc_gain = map(float, (cls_gain, reg_mix, enc_gain))
        self.quality_min = float(quality_min)
        self.size_max_px, self.short_max_px = float(size_max_px), float(short_max_px)
        self.side_floor_px, self.relative_scale = float(side_floor_px), float(relative_scale)
        self.huber_beta = float(huber_beta)
        self.start_epoch, self.full_epoch = int(start_epoch), int(full_epoch)
        self._ramp, self._sqa_context = 0., None
        self.last_sqa_stats = {}

    def set_epoch(self, epoch, total_epochs=0):
        """Use actual trainer epoch; never estimate progress from forward counts."""
        self._ramp = min(1., max(0., (epoch + 1 - self.start_epoch)
                                 / (self.full_epoch - self.start_epoch)))

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        self.last_sqa_stats = {}
        if not (self.training and torch.is_grad_enabled() and self._ramp and (self.cls_gain or self.reg_mix)):
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        boxes, _ = preds
        if self.enc_gain and len(boxes) < 2:
            raise ValueError('SQA needs encoder predictions at stage 0, before the decoder outputs')
        # Iteration/slicing in the original criterion creates different Python
        # tensor objects, so use the storage offset of each stage, not id(tensor).
        stages = {boxes[-1].data_ptr(): ('final', 1.)}
        if self.enc_gain:
            stages[boxes[0].data_ptr()] = ('enc', self.enc_gain)
        hw = batch.get('image_hw')
        if self.reg_mix and (hw is None or len(hw) != 2 or min(hw) <= 0):
            raise ValueError('SQA regression requires image_hw=(height, width)')
        self._sqa_context = (stages, hw)
        try:
            losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        finally:
            self._sqa_context = None
        if stages:
            raise RuntimeError('SQA did not encounter all original encoder/final matching stages')
        return losses

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        context = self._sqa_context if not postfix else None
        stage = context[0].pop(pred_bboxes.data_ptr(), None) if context is not None else None
        if stage and match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                         masks=masks, gt_mask=gt_mask)
        losses = super()._get_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                   masks=masks, gt_mask=gt_mask, postfix=postfix, match_indices=match_indices)
        if stage:
            name, gain = stage
            idx, gt_idx = self._get_index(match_indices)
            idx = tuple(i.to(pred_bboxes.device) for i in idx)
            gt_idx = gt_idx.to(gt_bboxes.device)
            pred, gt = pred_bboxes[idx].float(), gt_bboxes[gt_idx].detach().float()
            scores, cls = pred_scores[idx].float(), gt_cls[gt_idx].long()
            stats = dict(matches=len(gt))
            if len(gt):
                strength = self._ramp * gain
                if self.cls_gain:
                    gap, count = self._quality_gap(pred, scores, gt, cls)
                    losses['loss_class'] = losses['loss_class'] + (
                        strength * self.cls_gain * self.loss_gain['class'] * gap)
                    stats.update(cls_count=count, cls_gap_loss=gap.detach())
                if self.reg_mix:
                    losses['loss_bbox'], count = self._relative_regression(pred, gt, context[1], strength)
                    stats['reg_count'] = count
            self.last_sqa_stats[name] = stats
        return losses

    def _quality_gap(self, pred, scores, gt, cls):
        truth_logits = scores.gather(1, cls[:, None]).squeeze(1)
        with torch.no_grad():
            q = bbox_iou(pred.detach(), gt, xywh=True).squeeze(-1).clamp(0, 1)
            weight = (q - truth_logits.detach().sigmoid()).clamp_min(0) * (q >= self.quality_min)
        loss = (weight * F.binary_cross_entropy_with_logits(truth_logits, q, reduction='none')).sum()
        return loss / max(len(gt), 1), weight.gt(0).sum()

    def _relative_regression(self, pred, gt, hw, strength):
        h, w = hw
        axes = pred.new_tensor((w, h))
        gt_sides = gt[:, 2:] * axes
        with torch.no_grad():
            size = gt_sides.prod(-1).clamp_min(0).sqrt()
            # Full size weight at <=8px, tapering continuously to zero at 32px.
            size_weight = ((self.size_max_px - size)
                           / (self.size_max_px - self.side_floor_px)).clamp(0, 1)
            size_weight *= (gt_sides.amin(-1) > 0) & (gt_sides.amin(-1) < self.short_max_px)
            mix = strength * self.reg_mix * size_weight
        center_error = (pred[:, :2] - gt[:, :2]) * axes / gt_sides.clamp_min(self.side_floor_px)
        side_error = (torch.log1p(pred[:, 2:].clamp_min(0) * axes / self.side_floor_px)
                      - torch.log1p(gt_sides / self.side_floor_px))
        error = torch.cat((center_error, side_error), dim=-1)
        relative = F.smooth_l1_loss(error, torch.zeros_like(error), beta=self.huber_beta, reduction='none')
        kappa = (self.relative_scale * self.side_floor_px / axes).repeat(2)
        relative = (relative * kappa).sum(-1)
        l1 = (pred - gt).abs().sum(-1)
        blended = ((1 - mix) * l1 + mix * relative).sum() / max(len(gt), 1)
        return self.loss_gain['bbox'] * blended, mix.gt(0).sum()
