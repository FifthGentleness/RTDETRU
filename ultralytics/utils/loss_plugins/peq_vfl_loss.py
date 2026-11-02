"""Perturbation-expected quality targets; UGS (ICCV 2025) motivates uncertainty.
Only VFL quality targets change. Hungarian matching and L1/GIoU are unchanged.
"""
import torch
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou


class RTDETRDetectionLossPEQVFL(RTDETRDetectionLoss):
    def __init__(self, *args, beta=0.25, shift_px=0.5, max_shift_ratio=0.1, ramp_fraction=0.2, **kwargs):
        if not 0 <= beta <= 1 or shift_px < 0 or not 0 <= max_shift_ratio <= 1 or ramp_fraction <= 0:
            raise ValueError('Invalid PEQ-VFL parameters')
        super().__init__(*args, **kwargs)
        if self.vfl is None or self.fl is None:
            raise ValueError('PEQ-VFL requires use_fl=True and use_vfl=True')
        self.beta, self.shift_px, self.max_shift_ratio = beta, shift_px, max_shift_ratio
        self.ramp_fraction = ramp_fraction
        self.innovation_schedule = True
        self.progress = 1.0  # standalone criterion uses full strength
        self.image_hw = None

    def set_epoch(self, epoch, total_epochs):
        self.progress = min(epoch / max(total_epochs - 1, 1) / self.ramp_fraction, 1.0)

    def forward(self, preds, batch, **kwargs):
        self.image_hw = batch.get('image_hw')
        return super().forward(preds, batch, **kwargs)

    @torch.no_grad()
    def expected_quality(self, boxes, gt):
        if self.image_hw is None:
            raise ValueError('PEQ-VFL needs image_hw=(height,width) in targets')
        h, w = self.image_hw
        delta = torch.minimum(gt[:, 2:] * self.max_shift_ratio,
                              gt.new_tensor([self.shift_px / w, self.shift_px / h]))
        total = bbox_iou(boxes.detach(), gt, xywh=True).flatten()
        for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
            shifted = gt.clone()
            shifted[:, axis] += sign * delta[:, axis]
            # Keep GT dimensions unchanged; clip center to keep the full box in the image.
            shifted[:, axis] = shifted[:, axis].clamp(gt[:, axis+2] / 2, 1 - gt[:, axis+2] / 2)
            total = total + bbox_iou(boxes.detach(), shifted, xywh=True).flatten()
        return total / 5

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        beta = self.beta * self.progress
        if beta == 0 or self.shift_px == 0 or self.max_shift_ratio == 0:
            return super()._get_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                     masks, gt_mask, postfix, match_indices)
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                         masks=masks, gt_mask=gt_mask)
        idx, gt_idx = self._get_index(match_indices)
        assigned, gt = pred_bboxes[idx], gt_bboxes[gt_idx]
        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]
        quality = torch.zeros((bs, nq), device=pred_scores.device)
        if len(gt):
            q = bbox_iou(assigned.detach(), gt, xywh=True).flatten()
            quality[idx] = (1-beta) * q + beta * self.expected_quality(assigned, gt)
        loss = self._get_loss_class(pred_scores, targets, quality, len(gt), postfix)
        loss.update(self._get_loss_bbox(assigned, gt, postfix))
        return loss
