# Ultralytics YOLO 🐛, AGPL-3.0 license
"""
Scale-based Dynamic (SD) Loss for small object detection — self-contained module.

Faithful port of the SDB (bounding-box) version of SD Loss from:

    Yang et al., "Pinwheel-Shaped Convolution and Scale-Based Dynamic Loss for
    Infrared Small Target Detection", AAAI 2025, 39(9), pp. 9202-9210.
    arXiv: 2412.16986
    Official code: https://github.com/JN-Yang/PConv-SDloss-Data

Design goals (minimal invasion of the original RT-DETR code):
    - Everything lives in THIS file. `ultralytics/models/utils/loss.py`,
      `ultralytics/utils/metrics.py` and all other core files stay untouched.
    - `RTDETRDetectionLossSD` is a drop-in subclass of `RTDETRDetectionLoss`:
      only the GIoU term of the bbox regression loss is replaced by SDIoU.
      L1 term, VFL classification loss, Hungarian matching cost, loss gains,
      auxiliary/denoising loss plumbing are all inherited unchanged.
    - This file is registered in the loss plugin registry
      (`ultralytics/utils/loss_plugins/__init__.py`) under the name 'sd'.
      `RTDETRDetectionModel.init_criterion()` (in tasks.py) resolves
      `loss_name: sd` from the model yaml through that registry; every other
      config runs the original code path. Adding a new loss = one new file
      here + one registry line, tasks.py never changes again.

Core idea
---------
The regression loss trades off a location term (center distance, rho^2/c^2)
and a scale/shape term (aspect-ratio consistency, v*alpha) via a dynamic
coefficient computed from the ground-truth pixel area:

    beta = min(gt_area_px * delta / 81, delta)

- Tiny targets (area < 81 px^2, i.e. < 9x9 px):  beta < delta ->
  location term amplified, shape term damped. A 1-2 px center shift destroys
  a tiny box, so gradients focus on localization.
- Normal/large targets (area >= 81 px^2):        beta = delta ->
  SDIoU degenerates EXACTLY to standard CIoU, so large-object accuracy is
  completely preserved (verifiable property, see __main__ self-test below).

Note on image size: RT-DETR boxes are normalized, so GT pixel area needs the
input image size. The subclass defaults to 640x640 (this repo trains with
imgsz=640). If you train at another imgsz, either pass 'img_size' in the
batch/targets dict or set `RTDETRDetectionLossSD(sd_img_size=(H, W))`.
"""

import math

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('sd_bbox_iou', 'SDLoss', 'RTDETRDetectionLossSD')


def sd_bbox_iou(box1, box2, xywh=True, img_size=None, delta=0.5, eps=1e-7):
    """
    Scale-based Dynamic IoU (SDIoU) between predicted and ground-truth boxes.

    Args:
        box1 (torch.Tensor): Predicted boxes, shape (..., 4).
        box2 (torch.Tensor): Ground-truth boxes, shape (..., 4). The dynamic
            coefficient `beta` is computed from these GT boxes (official
            convention: box1=pred, box2=gt).
        xywh (bool): If True, inputs are (cx, cy, w, h) in normalized image
            coordinates (RT-DETR path); if False, (x1, y1, x2, y2) in pixels.
        img_size (tuple | list | torch.Tensor | float | None): (H, W) of the
            input image in pixels, used to convert normalized GT size to pixel
            area. None falls back to 640x640.
        delta (float): Maximum dynamic coefficient (official default 0.5).
        eps (float): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): SDIoU values, shape (..., 1). Use `1 - SDIoU` as loss.
    """
    if xywh:
        # Normalized xywh -> absolute pixel coordinates (RT-DETR path)
        if img_size is None:
            h_img, w_img = 640.0, 640.0
        elif isinstance(img_size, torch.Tensor):
            vals = img_size.flatten().tolist()
            h_img = float(vals[0])
            w_img = float(vals[1]) if len(vals) > 1 else float(vals[0])
        elif isinstance(img_size, (int, float)):
            h_img = w_img = float(img_size)
        else:
            img_size = tuple(img_size)
            h_img = float(img_size[0])
            w_img = float(img_size[1]) if len(img_size) > 1 else float(img_size[0])

        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        x1, y1, w1, h1 = x1 * w_img, y1 * h_img, w1 * w_img, h1 * h_img
        x2, y2, w2, h2 = x2 * w_img, y2 * h_img, w2 * w_img, h2 * h_img
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        # Already xyxy in absolute pixels (official YOLO-style path)
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection and union
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Smallest enclosing box and center distance
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

    # Aspect-ratio consistency term (same as CIoU)
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    # Scale-based dynamic coefficient from GT pixel area (official formula)
    beta = (w2 * h2 * delta) / 81
    beta = torch.where(beta > delta, torch.full_like(beta, delta), beta)

    # SDIoU: dynamic location/scale trade-off.
    # beta -> delta : exactly standard CIoU (normal/large targets)
    # beta -> 0     : location term amplified, shape term damped (tiny targets)
    return (delta - beta) + (1 - delta + beta) * (iou - v * alpha) - (1 + delta - beta) * (rho2 / c2)


class SDLoss(torch.nn.Module):
    """Standalone module wrapper of the SD bbox regression loss."""

    def __init__(self, delta=0.5, img_size=None):
        """Initialize with the max dynamic coefficient `delta` and optional (H, W)."""
        super().__init__()
        self.delta = delta
        self.img_size = img_size

    def forward(self, pred_bboxes, gt_bboxes):
        """Return mean SD regression loss `1 - SDIoU` for matched box pairs."""
        return (1.0 - sd_bbox_iou(pred_bboxes, gt_bboxes, xywh=True,
                                  img_size=self.img_size, delta=self.delta)).mean()


class RTDETRDetectionLossSD(RTDETRDetectionLoss):
    """
    Drop-in replacement of `RTDETRDetectionLoss` using the AAAI 2025 SD loss
    (SDB version) for the bbox regression term.

    Only `_get_loss_bbox` is overridden: the GIoU term is replaced by
    `1 - SDIoU`. Everything else (VFL classification loss, L1 term, Hungarian
    matching, auxiliary and denoising losses, loss gains) is inherited
    unchanged from `RTDETRDetectionLoss`.
    """

    def __init__(self, *args, sd_delta=0.5, sd_img_size=None, **kwargs):
        """
        Args:
            sd_delta (float): Max dynamic coefficient (official default 0.5).
            sd_img_size (tuple | None): (H, W) in pixels used to recover GT
                pixel area from normalized boxes. None -> 640x640, which is
                exact for this repo's default imgsz=640 training.
            *args, **kwargs: Forwarded to `RTDETRDetectionLoss` unchanged.
        """
        super().__init__(*args, **kwargs)
        self.sd_delta = sd_delta
        self.sd_img_size = sd_img_size

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """Optionally pick up an 'img_size' entry from the batch, then delegate."""
        if isinstance(batch, dict) and 'img_size' in batch:
            self.sd_img_size = batch['img_size']
        return super().forward(preds, batch, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta)

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        """L1 term unchanged + GIoU term replaced by the SD loss (SDB)."""
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=self.device)
            loss[name_giou] = torch.tensor(0., device=self.device)
            return loss

        # Same as the original: L1 over matched pairs, normalized by num_gts
        loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / len(gt_bboxes)

        # SD loss (SDB): replaces the original `1 - GIoU` term
        loss[name_giou] = 1.0 - sd_bbox_iou(pred_bboxes, gt_bboxes, xywh=True,
                                            img_size=self.sd_img_size, delta=self.sd_delta)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}


if __name__ == '__main__':
    # Self-test 1: degenerate property ? area >= 81 px^2 -> SDIoU == CIoU exactly
    from ultralytics.utils.metrics import bbox_iou

    torch.manual_seed(0)
    img = (640, 640)
    pred = torch.tensor([[0.52, 0.51, 0.140625, 0.109375]])  # ~90x70 px box

    gt_large = torch.tensor([[0.5, 0.5, 0.15625, 0.15625]])  # 100x100 px
    sd_l = sd_bbox_iou(pred, gt_large, img_size=img)
    ciou_l = bbox_iou(pred, gt_large, xywh=True, CIoU=True)
    print('large GT  SDIoU == CIoU :', torch.allclose(sd_l, ciou_l, atol=1e-5),
          float(sd_l), float(ciou_l))

    # Self-test 2: perfect prediction -> SDIoU == 1 regardless of scale
    for gt in (gt_large, torch.tensor([[0.5, 0.5, 0.009375, 0.009375]])):  # 100px / 6px
        sd_p = sd_bbox_iou(gt, gt, img_size=img)
        assert torch.allclose(sd_p, torch.ones_like(sd_p), atol=1e-6)
    print('perfect box -> SDIoU == 1 (small & large) : True')

    # Self-test 3: gradient behaviour for a tiny target (the actual design intent).
    # NOTE: (delta - beta) is a constant w.r.t. predictions, so absolute loss
    # magnitude is irrelevant; what matters is the gradient re-weighting:
    #   location term gradient  x (1 + delta - beta)  -> amplified for tiny GT
    #   scale/iou  term gradient  x (1 - delta + beta)  -> damped for tiny GT
    gt_small = torch.tensor([[0.5, 0.5, 0.009375, 0.009375]])  # 6x6 px, area 36 < 81
    beta = 36 * 0.5 / 81
    print(f'tiny GT beta = {beta:.4f} (< delta=0.5)')

    off = torch.tensor([[0.01, 0.008, 0.0, 0.0]], requires_grad=True)  # center offset
    loss_sd = (1 - sd_bbox_iou(gt_small + off, gt_small, img_size=img)).sum()
    loss_sd.backward()
    g_sd = off.grad.norm().item()

    off2 = torch.tensor([[0.01, 0.008, 0.0, 0.0]], requires_grad=True)
    loss_ciou = (1 - bbox_iou(gt_small + off2, gt_small, xywh=True, CIoU=True)).sum()
    loss_ciou.backward()
    g_ciou = off2.grad.norm().item()
    print(f'tiny GT center-offset gradient  SD/CIoU = {g_sd:.4f} / {g_ciou:.4f}  (expect > 1) :',
          g_sd > g_ciou)

    # scale-term gradient: perturb w/h only
    off3 = torch.tensor([[0.0, 0.0, 0.002, -0.002]], requires_grad=True)
    loss_sd2 = (1 - sd_bbox_iou(gt_small + off3, gt_small, img_size=img)).sum()
    loss_sd2.backward()
    g_sd2 = off3.grad.norm().item()

    off4 = torch.tensor([[0.0, 0.0, 0.002, -0.002]], requires_grad=True)
    loss_ciou2 = (1 - bbox_iou(gt_small + off4, gt_small, xywh=True, CIoU=True)).sum()
    loss_ciou2.backward()
    g_ciou2 = off4.grad.norm().item()
    print(f'tiny GT scale-term gradient     SD/CIoU = {g_sd2:.4f} / {g_ciou2:.4f}  (expect < 1) :',
          g_sd2 < g_ciou2)
