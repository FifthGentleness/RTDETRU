# Ultralytics YOLO, AGPL-3.0 license
# ThermalIoU + Inner-IoU loss for RT-DETR (self-contained module).
#
# ThermalIoU: Li et al., IEEE Signal Processing Letters, 2026.
#   DOI: 10.1109/LSP.2026.3699338
# Inner-IoU: Zhang et al., arXiv 2311.02877
#
# Regression loss (replaces the GIoU term in _get_loss_bbox):
#     L = (1 - ThermalIoU) + IoU_hard - IoU_inner(inner_ratio)

import math
import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('thermal_inner_iou_loss', 'RTDETRDetectionLossThermalInnerIoU')


def _xywh_to_corners(box):
    """Convert (cx, cy, w, h) -> x1, y1, x2, y2 corners."""
    cx, cy, w, h = box.chunk(4, -1)
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _hard_iou(box1, box2, eps=1e-7):
    """Standard hard IoU on xywh boxes."""
    x1a, y1a, x2a, y2a = _xywh_to_corners(box1)
    x1b, y1b, x2b, y2b = _xywh_to_corners(box2)
    inter = (x2a.minimum(x2b) - x1a.maximum(x1b)).clamp(min=0) * (
        y2a.minimum(y2b) - y1a.maximum(y1b)).clamp(min=0)
    union = (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b) - inter + eps
    return inter / union


def _inner_iou(box1, box2, ratio, eps=1e-7):
    """Inner-IoU on centre-fixed ratio-scaled auxiliary boxes."""
    x1a, y1a, wa, ha = box1.chunk(4, -1)
    x1b, y1b, wb, hb = box2.chunk(4, -1)
    a_x1 = x1a - wa * ratio / 2
    a_x2 = x1a + wa * ratio / 2
    a_y1 = y1a - ha * ratio / 2
    a_y2 = y1a + ha * ratio / 2
    b_x1 = x1b - wb * ratio / 2
    b_x2 = x1b + wb * ratio / 2
    b_y1 = y1b - hb * ratio / 2
    b_y2 = y1b + hb * ratio / 2
    inter = (a_x2.minimum(b_x2) - a_x1.maximum(b_x1)).clamp(min=0) * (
        a_y2.minimum(b_y2) - a_y1.maximum(b_y1)).clamp(min=0)
    union = wa * ha * ratio * ratio + wb * hb * ratio * ratio - inter + eps
    return inter / union


def _thermal_iou(pred_xywh, gt_xywh, sigma, grid_size=60, delta=1e-7, chunk=256):
    """
    ThermalIoU: soft Jaccard overlap of Gaussian-diffused box fields.

    Evaluates separable error-function diffusive fields on a local grid
    covering the union of both boxes + 3*sigma margin, then computes
    the element-wise min/max ratio (soft Jaccard).
    """
    device = pred_xywh.device
    N = pred_xywh.shape[0]
    if N == 0:
        return torch.zeros(0, device=device)

    if not torch.is_tensor(sigma):
        sigma = torch.full((N,), float(sigma), device=device, dtype=pred_xywh.dtype)
    sigma = sigma.clamp(min=1e-8)

    px1, py1, px2, py2 = _xywh_to_corners(pred_xywh)
    gx1, gy1, gx2, gy2 = _xywh_to_corners(gt_xywh)

    m = 3.0 * sigma.unsqueeze(1)  # (N, 1)
    ux1 = torch.minimum(px1, gx1) - m
    uy1 = torch.minimum(py1, gy1) - m
    ux2 = torch.maximum(px2, gx2) + m
    uy2 = torch.maximum(py2, gy2) + m

    t = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=pred_xywh.dtype)
    grid_x = ux1 + t.unsqueeze(0) * (ux2 - ux1)  # (N, G)
    grid_y = uy1 + t.unsqueeze(0) * (uy2 - uy1)  # (N, G)

    inv_s = 1.0 / (math.sqrt(2.0) * sigma).unsqueeze(1)  # (N, 1)

    qpx = 0.5 * (torch.erf((grid_x - px1) * inv_s) - torch.erf((grid_x - px2) * inv_s))
    qpy = 0.5 * (torch.erf((grid_y - py1) * inv_s) - torch.erf((grid_y - py2) * inv_s))
    qgx = 0.5 * (torch.erf((grid_x - gx1) * inv_s) - torch.erf((grid_x - gx2) * inv_s))
    qgy = 0.5 * (torch.erf((grid_y - gy1) * inv_s) - torch.erf((grid_y - gy2) * inv_s))

    result = torch.zeros(N, device=device, dtype=pred_xywh.dtype)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        u_p = qpx[start:end].unsqueeze(2) * qpy[start:end].unsqueeze(1)
        u_g = qgx[start:end].unsqueeze(2) * qgy[start:end].unsqueeze(1)
        inter = torch.minimum(u_p, u_g).sum(dim=(1, 2))
        union = torch.maximum(u_p, u_g).sum(dim=(1, 2)) + delta
        result[start:end] = inter / union

    return result


def thermal_inner_iou_loss(pred_bboxes, gt_bboxes, inner_ratio=0.8,
                           lambda_scale=0.3, sigma_min=0.01,
                           epoch_frac=0.0, E_anneal=0,
                           grid_size=60, delta=1e-7, eps=1e-7):
    """Per-sample ThermalIoU + Inner-IoU regression loss. Shape (N,)."""
    w_gt = gt_bboxes[..., 2:3].abs()
    h_gt = gt_bboxes[..., 3:4].abs()
    L = w_gt * h_gt / (2.0 * (w_gt + h_gt) + eps)

    if E_anneal > 0:
        A_e = max(0.0, 1.0 - epoch_frac / E_anneal)
    else:
        A_e = 1.0

    sigma = (lambda_scale * L * A_e + sigma_min).squeeze(-1)

    th_iou = _thermal_iou(pred_bboxes, gt_bboxes, sigma,
                          grid_size=grid_size, delta=delta)

    iou = _hard_iou(pred_bboxes, gt_bboxes, eps=eps).squeeze(-1)
    iou_inner = _inner_iou(pred_bboxes, gt_bboxes, inner_ratio, eps=eps).squeeze(-1)

    return (1.0 - th_iou) + iou - iou_inner


class RTDETRDetectionLossThermalInnerIoU(RTDETRDetectionLoss):
    """ThermalIoU + Inner-IoU loss plugin (replaces GIoU term only)."""

    def __init__(self, *args, inner_ratio=0.8, lambda_scale=0.3,
                 sigma_min=0.01, E_anneal=0, grid_size=60, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_ratio = inner_ratio
        self.lambda_scale = lambda_scale
        self.sigma_min = sigma_min
        self.E_anneal = E_anneal
        self.grid_size = grid_size
        self.current_epoch = 0
        self.total_epochs = 0

    def set_epoch(self, epoch, total_epochs=0):
        self.current_epoch = epoch
        if total_epochs > 0:
            self.total_epochs = total_epochs

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=""):
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=pred_bboxes.device)
            loss[name_giou] = torch.tensor(0., device=pred_bboxes.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(
            pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)

        epoch_frac = self.current_epoch / max(self.total_epochs, 1)
        giou = thermal_inner_iou_loss(
            pred_bboxes, gt_bboxes,
            inner_ratio=self.inner_ratio,
            lambda_scale=self.lambda_scale,
            sigma_min=self.sigma_min,
            epoch_frac=epoch_frac,
            E_anneal=self.E_anneal,
            grid_size=self.grid_size)

        loss[name_giou] = giou.sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}
