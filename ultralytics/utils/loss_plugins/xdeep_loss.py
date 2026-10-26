# 批次7深水区损失机制 (X063/X068, [C]).
# X063 L-QClamp: 质量目标上界clamp(0.9)——防分类过自信(val/cls回升对策)
# X068 L-SmoothL1: L1->SmoothL1(beta=1.0)(回归形态微调)

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.loss import FocalLoss

__all__ = ('RTDETRDetectionLossXQClamp', 'RTDETRDetectionLossXSL1')


class RTDETRDetectionLossXQClamp(RTDETRDetectionLoss):
    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, q_max=0.9, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)
        self.q_max = float(q_max)
        self.fl = FocalLoss()

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        name_class = f'loss_class{postfix}'
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64,
                              device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        label = one_hot.float()
        q = gt_scores.view(bs, nq, 1).clamp(max=self.q_max) * label
        weight = 0.75 * pred_scores.sigmoid().pow(2.0) * (1 - label) + q * label
        loss_cls = (F.binary_cross_entropy_with_logits(
            pred_scores.float(), q.float(), reduction='none') *
            weight).mean(1).sum()
        if num_gts == 0:
            loss_cls = self.fl(pred_scores, label)
        else:
            loss_cls /= max(num_gts, 1) / nq
        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}


class RTDETRDetectionLossXSL1(RTDETRDetectionLoss):
    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, beta=1.0, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)
        self.beta = float(beta)

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        import torch as _t
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'
        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = _t.tensor(0., device=pred_bboxes.device)
            loss[name_giou] = _t.tensor(0., device=pred_bboxes.device)
            return loss
        sl1 = F.smooth_l1_loss(pred_bboxes, gt_bboxes, reduction='none',
                               beta=self.beta).sum(-1)
        loss[name_bbox] = self.loss_gain['bbox'] * sl1.sum() / len(gt_bboxes)
        from ultralytics.utils.metrics import bbox_iou
        giou = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = self.loss_gain['giou'] * giou.squeeze(-1).sum() / len(gt_bboxes)
        return {k: v.squeeze() for k, v in loss.items()}
