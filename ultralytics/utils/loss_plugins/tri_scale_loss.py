# Ultralytics AGPL-3.0 license
"""
Tri-Scale Loss: a loss-side synergy for small-object detection in RT-DETR.

Three orthogonal, loss-only components (zero inference cost):

1. SD  (regression side, inspired by AAAI 2025 "Scale-based Dynamic Loss"):
   Area-adaptive scheduling of the L1 location (x, y) vs scale (w, h) terms,
   plus an area-adaptive gain on the GIoU term. Small objects receive stronger
   position supervision and a stronger GIoU weight.

2. PGDE (supervision-density side, inspired by CVPR 2025 "Feature Information
   Driven Position Gaussian Distribution Estimation"):
   Parameter-free Gaussian distribution-map alignment. GT boxes are rasterized
   into multi-scale Gaussian mixture maps; matched predicted boxes are
   rasterized the same way; a soft-Dice loss aligns the two maps. This turns
   the sparse per-query supervision into dense spatial supervision without any
   extra parameters or inference cost.

3. SARD (supervision-quality side, inspired by CVPR 2026 "Structure-Aware
   Representation Distillation"):
   Structure-importance weighted self-distillation from the final decoder
   layer to the auxiliary layers. Per-GT importance combines a boundary
   salience proxy, a geometric complexity proxy and a local structure
   variation proxy measured on the PGDE Gaussian maps.

Selected through the model yaml:
    loss_name: tri_scale
    loss_params:
      use_sd: true
      use_pgde: true
      use_sard: true
      ...

This file is registered in the loss plugin registry
(`ultralytics/utils/loss_plugins/__init__.py`) under the name 'tri_scale'.
"""

import math

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import DETRLoss, RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou

__all__ = ('TriScaleDetectionLoss',)


class TriScaleDetectionLoss(RTDETRDetectionLoss):
    """
    RT-DETR detection loss with Tri-Scale small-object synergy:
    SD (area-adaptive regression) + PGDE (Gaussian map supervision)
    + SARD (importance-weighted cross-layer distillation).

    All three components are training-only: no parameters are added and
    inference behavior is identical to the baseline RT-DETR.
    """

    def __init__(self,
                 nc=80,
                 loss_gain=None,
                 aux_loss=True,
                 use_fl=True,
                 use_vfl=True,
                 use_sl=False,
                 use_emasl=False,
                 use_svfl=False,
                 use_emasvfl=False,
                 use_mal=False,
                 # ---- master switches (ablation) ----
                 use_sd=True,
                 use_pgde=True,
                 use_sard=True,
                 # ---- 1) SD: area-adaptive regression ----
                 sd_strength=1.0,
                 sd_a_min=1e-5,
                 sd_a_max=0.05,
                 sd_area_ref=0.01,
                 sd_giou_gamma=0.5,
                 sd_giou_max=4.0,
                 # ---- 2) PGDE: Gaussian distribution map alignment ----
                 pgde_grids=(64, 32, 16),
                 pgde_sigma_scale=0.25,
                 pgde_sigma_cells=2.5,
                 pgde_weight=2.0,
                 # ---- 3) SARD: importance-weighted distillation ----
                 distill_weight=1.0,
                 sard_T=2.0,
                 sard_ar_range=4.0,
                 sard_alpha=1.0,
                 sard_beta=1.0,
                 sard_gamma=1.0,
                 sard_imp_min=0.3,
                 aux_warmup_iters=8000):
        super().__init__(nc=nc,
                         loss_gain=loss_gain,
                         aux_loss=aux_loss,
                         use_fl=use_fl,
                         use_vfl=use_vfl,
                         use_sl=use_sl,
                         use_emasl=use_emasl,
                         use_svfl=use_svfl,
                         use_emasvfl=use_emasvfl,
                         use_mal=use_mal)
        self.use_sd = bool(use_sd)
        self.use_pgde = bool(use_pgde)
        self.use_sard = bool(use_sard)

        self.sd_strength = float(sd_strength)
        self.sd_a_min = float(sd_a_min)
        self.sd_a_max = float(sd_a_max)
        self.sd_area_ref = float(sd_area_ref)
        self.sd_giou_gamma = float(sd_giou_gamma)
        self.sd_giou_max = float(sd_giou_max)

        self.pgde_grids = tuple(int(g) for g in pgde_grids)
        self.pgde_sigma_scale = float(pgde_sigma_scale)
        self.pgde_sigma_cells = float(pgde_sigma_cells)
        self.pgde_gain = float(pgde_weight)

        self.distill_gain = float(distill_weight)
        self.sard_T = float(sard_T)
        self.sard_ar_range = float(sard_ar_range)
        self.sard_alpha = float(sard_alpha)
        self.sard_beta = float(sard_beta)
        self.sard_gamma = float(sard_gamma)
        self.sard_imp_min = float(sard_imp_min)
        self.aux_warmup_iters = max(int(aux_warmup_iters), 1)
        self._iters = 0
        self._aux_ramp = 1.0

    # ------------------------------------------------------------------
    # shared helpers
    # ------------------------------------------------------------------
    def _area_schedule(self, area):
        t = (torch.log(area.clamp_min(1e-8)) - math.log(self.sd_a_min)) / \
            (math.log(self.sd_a_max) - math.log(self.sd_a_min))
        return t.clamp(0.0, 1.0)

    def _gauss_map(self, boxes, grid):
        dev = boxes.device
        axis = torch.linspace(0.0, 1.0, grid, device=dev)
        ys, xs = torch.meshgrid(axis, axis, indexing='ij')
        cx = boxes[:, 0][:, None, None]
        cy = boxes[:, 1][:, None, None]
        sigma = (self.pgde_sigma_scale * torch.maximum(boxes[:, 2], boxes[:, 3]))
        sigma = sigma.clamp(self.pgde_sigma_cells / grid, 0.25)[:, None, None]
        d2 = ((xs[None] - cx) ** 2 + (ys[None] - cy) ** 2) / (2.0 * sigma ** 2 + 1e-12)
        return torch.exp(-d2).sum(0)

    @staticmethod
    def _soft_dice_loss(pred_map, gt_map):
        inter = (pred_map * gt_map).sum()
        denom = pred_map.sum() + gt_map.sum()
        return 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)

    # ------------------------------------------------------------------
    # 1) SD loss: area-adaptive L1 (x, y) / (w, h) scheduling + GIoU gain
    # ------------------------------------------------------------------
    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'
        if len(gt_bboxes) == 0:
            zero = torch.tensor(0., device=self.device)
            return {name_bbox: zero, name_giou: zero}
        if not self.use_sd:
            return super()._get_loss_bbox(pred_bboxes, gt_bboxes, postfix)

        n = len(gt_bboxes)
        diff = (pred_bboxes - gt_bboxes).abs()
        area = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        t = self._area_schedule(area)

        w_pos = 1.0 + self.sd_strength * (1.0 - t)
        w_scale = 1.0 + self.sd_strength * t
        l1 = ((w_pos.unsqueeze(-1) * diff[:, :2]).sum() +
              (w_scale.unsqueeze(-1) * diff[:, 2:]).sum()) / n

        giou = (1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)).squeeze(-1)
        g_w = ((self.sd_area_ref / area.clamp_min(1e-8)) ** self.sd_giou_gamma).clamp(1.0, self.sd_giou_max)

        return {
            name_bbox: self.loss_gain['bbox'] * l1,
            name_giou: self.loss_gain['giou'] * (giou * g_w).sum() / n,
        }

    # ------------------------------------------------------------------
    # 2) PGDE: multi-scale Gaussian map alignment (parameter-free)
    # ------------------------------------------------------------------
    def _get_loss_pgde(self, pred_bboxes, gt_bboxes, gt_groups, match_indices, postfix=''):
        name = f'loss_pgde{postfix}'
        if sum(gt_groups) == 0:
            return {name: torch.tensor(0., device=self.device)}
        idx, _ = self._get_index(match_indices)
        batch_idx, src_idx = idx
        if len(batch_idx) == 0:
            return {name: torch.tensor(0., device=self.device)}
        pred_matched = pred_bboxes[batch_idx, src_idx]

        losses = []
        off = 0
        for n_i in gt_groups:
            if n_i == 0:
                continue
            gt_i = gt_bboxes[off:off + n_i]
            pred_i = pred_matched[off:off + n_i]
            off += n_i
            for grid in self.pgde_grids:
                gt_map = self._gauss_map(gt_i, grid)
                pr_map = self._gauss_map(pred_i, grid)
                losses.append(self._soft_dice_loss(pr_map, gt_map))
        loss = torch.stack(losses).mean()
        return {name: self.pgde_gain * loss}

    # ------------------------------------------------------------------
    # 3) SARD: structure-importance weighted cross-layer distillation
    # ------------------------------------------------------------------
    def _ring_variation(self, gt_bboxes, gt_groups, grid):
        dev = gt_bboxes.device
        k = 8
        angles = torch.arange(k, device=dev, dtype=torch.float32) * (2 * math.pi / k)
        cos, sin = torch.cos(angles), torch.sin(angles)
        vals = []
        off = 0
        for n_i in gt_groups:
            if n_i == 0:
                continue
            boxes = gt_bboxes[off:off + n_i]
            off += n_i
            gt_map = self._gauss_map(boxes, grid)
            cx, cy = boxes[:, 0], boxes[:, 1]
            r = 0.75 * torch.maximum(boxes[:, 2], boxes[:, 3])
            px = (cx[:, None] + r[:, None] * cos[None]).clamp(0.0, 1.0 - 1e-6)
            py = (cy[:, None] + r[:, None] * sin[None]).clamp(0.0, 1.0 - 1e-6)
            ix = (px * grid).long().clamp(0, grid - 1)
            iy = (py * grid).long().clamp(0, grid - 1)
            v = gt_map[iy, ix]
            mean = v.mean(-1)
            std = v.std(-1) if n_i > 1 else torch.zeros_like(mean)
            vals.append((std / (mean + 1e-6)).clamp(0.0, 2.0) / 2.0)
        return torch.cat(vals) if vals else torch.zeros(0, device=dev)

    def _structure_importance(self, gt_bboxes, gt_groups):
        area = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        t = self._area_schedule(area)
        s1 = 1.0 - t

        ar = (gt_bboxes[:, 2] / gt_bboxes[:, 3].clamp_min(1e-8)).clamp_min(1e-4)
        s2 = (torch.log(ar).abs() / math.log(self.sard_ar_range)).clamp(0.0, 1.0)

        s3 = self._ring_variation(gt_bboxes, gt_groups, self.pgde_grids[0])

        imp_raw = self.sard_alpha * s1 + self.sard_beta * s2 + self.sard_gamma * s3
        imp_norm = imp_raw / (imp_raw.max() + 1e-6)
        return self.sard_imp_min + (1.0 - self.sard_imp_min) * imp_norm

    def _get_loss_distill(self, pred_bboxes, pred_scores, gt_bboxes, gt_groups, match_indices, postfix=''):
        name = f'loss_distill{postfix}'
        n_layers = pred_bboxes.shape[0]
        if n_layers < 2 or sum(gt_groups) == 0:
            return {name: torch.tensor(0., device=self.device)}
        idx, gt_idx = self._get_index(match_indices)
        if len(gt_idx) == 0:
            return {name: torch.tensor(0., device=self.device)}

        t_boxes = pred_bboxes[-1][idx].detach()
        t_probs = torch.sigmoid(pred_scores[-1][idx].detach() / self.sard_T)
        imp = self._structure_importance(gt_bboxes, gt_groups)[gt_idx]

        cls_part = pred_bboxes.new_tensor(0.0)
        box_part = pred_bboxes.new_tensor(0.0)
        for l in range(n_layers - 1):
            s_logits = pred_scores[l][idx]
            cls_part = cls_part + (F.binary_cross_entropy_with_logits(
                s_logits, t_probs, reduction='none').mean(-1) * imp).sum()
            s_boxes = pred_bboxes[l][idx]
            box_part = box_part + (F.smooth_l1_loss(
                s_boxes, t_boxes, reduction='none').mean(-1) * imp).sum()

        loss = (cls_part + box_part) / (max(len(gt_idx), 1) * (n_layers - 1))
        return {name: self.distill_gain * loss}

    # ------------------------------------------------------------------
    # forward: reuse DETRLoss with shared match indices, then add PGDE/SARD
    # ------------------------------------------------------------------
    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        pred_bboxes, pred_scores = preds
        self.device = pred_bboxes.device
        gt_cls, gt_bboxes, gt_groups = batch['cls'], batch['bboxes'], batch['gt_groups']

        match_indices = self.matcher(pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls,
                                     gt_groups, masks=None, gt_mask=None)

        if torch.is_grad_enabled():
            self._iters += 1
        self._aux_ramp = min(1.0, self._iters / self.aux_warmup_iters)

        total_loss = DETRLoss.forward(self, pred_bboxes, pred_scores, batch, match_indices=match_indices)

        if self.use_pgde:
            pgde = self._get_loss_pgde(pred_bboxes[-1], gt_bboxes, gt_groups, match_indices)
            total_loss.update({k: v * self._aux_ramp for k, v in pgde.items()})
        if self.use_sard and self.aux_loss:
            dist = self._get_loss_distill(pred_bboxes, pred_scores, gt_bboxes, gt_groups, match_indices)
            total_loss.update({k: v * self._aux_ramp for k, v in dist.items()})

        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta['dn_pos_idx'], dn_meta['dn_num_group']
            assert len(batch['gt_groups']) == len(dn_pos_idx)
            dn_match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch['gt_groups'])
            dn_loss = DETRLoss.forward(self, dn_bboxes, dn_scores, batch, postfix='_dn',
                                       match_indices=dn_match_indices)
            total_loss.update(dn_loss)
        else:
            total_loss.update({f'{k}_dn': torch.tensor(0., device=self.device) for k in total_loss.keys()})

        return total_loss