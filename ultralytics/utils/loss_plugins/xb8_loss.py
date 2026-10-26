# 批次8 损失机制 (X075/X078, [C]).
# X075 L-FCWCAS: 类频加权 + 分类晚期退火联合(单类承载双机制)
# X078 L-SizeQ: 小目标质量目标提升 q *= 1+beta*lam(s) capped

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.loss import FocalLoss

__all__ = ('RTDETRDetectionLossXFCWCAS', 'RTDETRDetectionLossXSizeQ')


class RTDETRDetectionLossXFCWCAS(RTDETRDetectionLoss):
    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, weights=None, t0_ep=150, t1_ep=300,
                 floor=0.5, ipe=1618, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)
        if weights is None:
            weights = [1.0] * nc
        self.register_buffer('cw', torch.tensor(weights, dtype=torch.float32))
        self.t0, self.t1, self.floor = int(t0_ep * ipe), int(t1_ep * ipe), float(floor)
        self.calls = 0
        self.fl = FocalLoss()

    def _factor(self):
        if self.calls <= self.t0:
            return 1.0
        if self.calls >= self.t1:
            return self.floor
        return 1.0 - (self.calls - self.t0) / max(self.t1 - self.t0, 1) * (1.0 - self.floor)

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        self.calls += 1
        out = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        f = self._factor()
        for k in list(out.keys()):
            if k.startswith('loss_class'):
                out[k] = out[k] * f
        return out

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        name_class = f'loss_class{postfix}'
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64,
                              device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        label = one_hot.float()
        gt_scores = gt_scores.view(bs, nq, 1) * label
        w = self.cw.view(1, 1, -1)
        weight = 0.75 * pred_scores.sigmoid().pow(2.0) * (1 - label) + gt_scores * w * label
        loss_cls = (F.binary_cross_entropy_with_logits(
            pred_scores.float(), gt_scores.float(), reduction='none') *
            weight).mean(1).sum()
        if num_gts == 0:
            loss_cls = self.fl(pred_scores, label)
        else:
            loss_cls /= max(num_gts, 1) / nq
        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}


class RTDETRDetectionLossXSizeQ(RTDETRDetectionLoss):
    """小目标质量目标提升: q'=min(q*(1+beta*lam(s)), 1), lam=clip(1-s/tau,0,1)。"""

    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, beta=0.3, tau=0.03125, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)
        self.beta = float(beta)
        self.tau = float(tau)
        self.fl = FocalLoss()

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        # 需在super前记录batch尺寸信息——直接在_get_loss_class不可得GT尺寸
        # 简化: 在_get_loss中把尺寸乘入gt_scores前先算好
        return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        from ultralytics.utils.metrics import bbox_iou
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes,
                                         gt_cls, gt_groups)
        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]
        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device,
                             dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]
        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            iou = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)
            s = (gt_bboxes[:, 2] * gt_bboxes[:, 3]).clamp(min=0).sqrt()
            lam = (1.0 - s / self.tau).clamp(0.0, 1.0)
            q = iou * (1.0 + self.beta * lam)
            gt_scores[idx] = q.clamp(max=1.0)
        loss = {}
        loss.update(self._get_loss_class(pred_scores, targets, gt_scores,
                                          len(gt_bboxes), postfix))
        loss.update(self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix))
        return loss

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        name_class = f'loss_class{postfix}'
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64,
                              device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        label = one_hot.float()
        gt_scores = gt_scores.view(bs, nq, 1) * label
        weight = 0.75 * pred_scores.sigmoid().pow(2.0) * (1 - label) + gt_scores * label
        loss_cls = (F.binary_cross_entropy_with_logits(
            pred_scores.float(), gt_scores.float(), reduction='none') *
            weight).mean(1).sum()
        if num_gts == 0:
            loss_cls = self.fl(pred_scores, label)
        else:
            loss_cls /= max(num_gts, 1) / nq
        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}
