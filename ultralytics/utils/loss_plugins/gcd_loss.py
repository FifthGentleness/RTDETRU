"""Gaussian Combined Distance loss for horizontal boxes (IEEE GRSL 2025)."""

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss


class RTDETRDetectionLossGCD(RTDETRDetectionLoss):
    """Use both boxes' widths and heights to normalize center and size errors."""

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        l1_key, geometry_key = f'loss_bbox{postfix}', f'loss_giou{postfix}'
        num_boxes = len(gt_bboxes)
        if num_boxes == 0:
            return {key: torch.zeros((), device=pred_bboxes.device) for key in (l1_key, geometry_key)}

        l1 = F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum')
        # Compute tiny box geometry in FP32 under AMP, keeping native L1 precision.
        low_precision = (torch.float16, torch.bfloat16)
        pred = pred_bboxes.float() if pred_bboxes.dtype in low_precision else pred_bboxes
        target = gt_bboxes.float() if gt_bboxes.dtype in low_precision else gt_bboxes
        geometry = self._geometry_loss(pred, target)
        return {
            l1_key: self.loss_gain['bbox'] * l1 / num_boxes,
            geometry_key: self.loss_gain['giou'] * geometry.sum() / num_boxes,
        }

    def _geometry_loss(self, pred, target):
        center_delta = pred[..., :2] - target[..., :2]
        half_size_delta = (pred[..., 2:] - target[..., 2:]) * 0.5
        pred_wh, target_wh = pred[..., 2:].clamp_min(1e-7), target[..., 2:].clamp_min(1e-7)
        residuals = torch.cat(
            (center_delta / pred_wh, half_size_delta / pred_wh,
             center_delta / target_wh, half_size_delta / target_wh),
            dim=-1,
        )
        # vector_norm defines a finite zero subgradient for an exact match.
        distance = torch.linalg.vector_norm(residuals, dim=-1) * (0.5 ** 0.5)
        return -torch.expm1(-distance)
