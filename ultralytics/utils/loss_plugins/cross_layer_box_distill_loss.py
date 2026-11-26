# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Late bidirectional box distillation between independently matched RT-DETR decoder layers."""

import math

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy


class RTDETRDetectionLossCrossLayerBoxDistill(RTDETRDetectionLoss):
    """Add an edge loss where the same query owns the same GT and one layer has better IoU.

    The final decoder layer teaches the intermediate decoder layers, and the best
    eligible intermediate layer can correct the final layer. Each target edge is
    detached and projected into the interval from the student edge to the GT.
    Native matching and losses supervise every layer throughout training.
    """

    innovation_schedule = True

    def __init__(
        self,
        *args,
        start_fraction=0.20,
        full_fraction=0.45,
        final_gain=1.0,
        intermediate_gain=0.85,
        student_min_iou=0.45,
        teacher_iou=(0.60, 0.78),
        iou_gap=(0.025, 0.12),
        side_floor_px=8.0,
        huber_beta=0.15,
        **kwargs,
    ):
        kwargs.setdefault('use_vfl', True)
        super().__init__(*args, **kwargs)
        if not 0 <= start_fraction < full_fraction <= 1:
            raise ValueError('Require 0 <= start_fraction < full_fraction <= 1')
        if len(teacher_iou) != 2 or not 0 <= student_min_iou < teacher_iou[0] < teacher_iou[1] <= 1:
            raise ValueError('Require 0 <= student_min_iou < teacher_iou[0] < teacher_iou[1] <= 1')
        if len(iou_gap) != 2 or not 0 <= iou_gap[0] < iou_gap[1] <= 1:
            raise ValueError('Require 0 <= iou_gap[0] < iou_gap[1] <= 1')
        if not all(math.isfinite(x) and x >= 0 for x in (final_gain, intermediate_gain)):
            raise ValueError('Distillation gains must be finite and non-negative')
        if not all(math.isfinite(x) and x > 0 for x in (side_floor_px, huber_beta)):
            raise ValueError('side_floor_px and huber_beta must be finite and positive')
        if (final_gain or intermediate_gain) and (self.use_uni_match or not self.aux_loss):
            raise ValueError('Box distillation requires independent matching and auxiliary decoder losses')

        self.start_fraction, self.full_fraction = start_fraction, full_fraction
        self.final_gain, self.intermediate_gain = final_gain, intermediate_gain
        self.student_min_iou, self.teacher_iou, self.iou_gap = student_min_iou, tuple(teacher_iou), tuple(iou_gap)
        self.side_floor_px, self.huber_beta = side_floor_px, huber_beta
        self._strength = 0.0
        self._decoder_matches = None

    def set_epoch(self, epoch, total_epochs):
        """Use the trainer's zero-based epoch and configured training horizon."""
        progress = (epoch + 1) / max(total_epochs, 1)
        ramp = (progress - self.start_fraction) / (self.full_fraction - self.start_fraction)
        ramp = min(1.0, max(0.0, ramp))
        self._strength = ramp * ramp * (3 - 2 * ramp)

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        # The first regular output is the encoder's selected proposals.
        decoder_boxes = preds[0][1:]
        active = (self.training and torch.is_grad_enabled() and self._strength
                  and (self.final_gain or self.intermediate_gain) and len(decoder_boxes) > 1
                  and batch['bboxes'].numel())
        if not active:
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)

        layer_keys = [boxes.data_ptr() for boxes in decoder_boxes]
        matches = dict.fromkeys(layer_keys)
        self._decoder_matches = matches
        try:
            losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        finally:
            self._decoder_matches = None
        image_hw = batch.get('image_hw', (640, 640))
        losses['loss_cross_layer_box_distill'] = self._strength * self._cross_layer_loss(
            decoder_boxes, batch['bboxes'], [matches[key] for key in layer_keys], image_hw
        )
        return losses

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None,
                  postfix='', match_indices=None):
        # Capture the assignment used by the native objective without repeating Hungarian matching.
        matches = self._decoder_matches
        if matches is not None and not postfix:
            key = pred_bboxes.data_ptr()
            if key in matches:
                if match_indices is None:
                    match_indices = self.matcher(
                        pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
                    )
                matches[key] = match_indices
        return super()._get_loss(
            pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks, gt_mask, postfix, match_indices
        )

    def _cross_layer_loss(self, decoder_boxes, gt_boxes, matches, image_hw):
        # Each row represents a final one-to-one match. Other layers must own that very same GT.
        (batch_idx, query_idx), gt_idx = self._get_index(matches[-1])
        if gt_idx.numel() == 0:
            return decoder_boxes.sum() * 0.0
        with torch.no_grad():
            agreement = []
            for layer_matches in matches[:-1]:
                owner = torch.full(decoder_boxes.shape[1:3], -1, dtype=torch.long, device=decoder_boxes.device)
                src, dst = self._get_index(layer_matches)
                owner[src] = dst.to(owner.device)
                agreement.append(owner[batch_idx, query_idx] == gt_idx.to(owner.device))
            agreement = torch.stack(agreement)

        with torch.autocast(device_type=decoder_boxes.device.type, enabled=False):
            height, width = image_hw
            scale = torch.tensor((width, height, width, height), dtype=torch.float32, device=decoder_boxes.device)
            boxes = decoder_boxes[:, batch_idx, query_idx].float() * scale
            gt = gt_boxes[gt_idx].detach().float() * scale
            edges = xywh2xyxy(boxes)
            gt_edges = xywh2xyxy(gt)
            normalizer = gt[:, 2:].clamp_min(self.side_floor_px).repeat(1, 2)
            with torch.no_grad():
                quality = bbox_iou(boxes.detach(), gt, xywh=True).squeeze(-1)
                best_quality, best_layer = quality[:-1].masked_fill(~agreement, -1.0).max(dim=0)
                best_edges = edges[best_layer, torch.arange(len(gt), device=gt.device)].detach()

            loss = edges.new_zeros(())
            if self.intermediate_gain:
                to_intermediate = self._edge_loss(
                    edges[:-1], edges[-1], gt_edges, normalizer, quality[:-1], quality[-1], agreement
                )
                loss = loss + self.intermediate_gain * to_intermediate
            if self.final_gain:
                to_final = self._edge_loss(
                    edges[-1], best_edges, gt_edges, normalizer, quality[-1], best_quality, best_quality >= 0
                )
                loss = loss + self.final_gain * to_final
        return loss

    def _edge_loss(self, student, teacher, gt, normalizer, student_iou, teacher_iou, agreement):
        with torch.no_grad():
            source = student.detach()
            teacher = teacher.detach()
            closer = (teacher - gt).abs() < (source - gt).abs()
            projected = teacher.clamp(min=source.minimum(gt), max=source.maximum(gt))
            target = torch.where(closer, projected, source)
            quality = ((teacher_iou - self.teacher_iou[0]) / (self.teacher_iou[1] - self.teacher_iou[0])).clamp(0, 1)
            advantage = (teacher_iou - student_iou - self.iou_gap[0]) / (self.iou_gap[1] - self.iou_gap[0])
            weight = agreement * (student_iou >= self.student_min_iou) * quality * advantage.clamp(0, 1)
        residual = (student - target) / normalizer
        error = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=self.huber_beta, reduction='none')
        return (error * weight.unsqueeze(-1)).mean()
