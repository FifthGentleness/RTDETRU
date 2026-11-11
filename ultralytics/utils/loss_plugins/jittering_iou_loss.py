"""Point Teacher's GT size jitter, with IoU or DIoU as the base metric."""

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss


class RTDETRDetectionLossJitteringIoU(RTDETRDetectionLoss):
    """Combine the original GT loss with the best loss over nine GT sizes."""

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

    def __init__(self, *args, jitter_ratio=0.1, jitter_metric='diou', jitter_normalize=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= jitter_ratio < 1:
            raise ValueError('jitter_ratio must be in [0, 1)')
        if jitter_metric not in ('iou', 'diou'):
            raise ValueError("jitter_metric must be 'iou' or 'diou'")
        self.jitter_metric = jitter_metric
        self.jitter_normalize = jitter_normalize
        sizes = (1.0, 1.0 - jitter_ratio, 1.0 + jitter_ratio)
        self.register_buffer(
            'jitter_scales', torch.tensor([(w, h) for w in sizes for h in sizes]), persistent=False
        )

    def _geometry_loss(self, pred, target):
        # RT-DETR can initialize its criterion lazily, after moving the model to CUDA.
        if self.jitter_scales.device != target.device:
            self.jitter_scales = self.jitter_scales.to(device=target.device)
        pred, target = pred.unsqueeze(-2), target.unsqueeze(-2)
        pred_half = pred[..., 2:] * 0.5
        target_half = target[..., 2:] * self.jitter_scales * 0.5
        pred_lo, pred_hi = pred[..., :2] - pred_half, pred[..., :2] + pred_half
        target_lo, target_hi = target[..., :2] - target_half, target[..., :2] + target_half

        inter_wh = (torch.minimum(pred_hi, target_hi) - torch.maximum(pred_lo, target_lo)).clamp_min(0)
        inter = inter_wh.prod(dim=-1)
        union = (pred_hi - pred_lo).prod(dim=-1) + (target_hi - target_lo).prod(dim=-1) - inter
        losses = 1.0 - inter / union.clamp_min(1e-12)
        if self.jitter_metric == 'diou':
            enclosure = torch.maximum(pred_hi, target_hi) - torch.minimum(pred_lo, target_lo)
            center_distance = (pred[..., :2] - target[..., :2]).square().sum(dim=-1)
            losses = losses + center_distance / enclosure.square().sum(dim=-1).clamp_min(1e-12)

        total = losses[..., 0] + losses.min(dim=-1).values
        return total * 0.5 if self.jitter_normalize else total
