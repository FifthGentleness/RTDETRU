# X028/X029 损失变体 (批次3, [C]).
# X028 L-BBR6: L1 bbox增益 5->6
# X029 L-VFLA05: VFL负样本alpha 0.75->0.5

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.loss import FocalLoss

__all__ = ('RTDETRDetectionLossXBBR', 'RTDETRDetectionLossXVFLA')


class RTDETRDetectionLossXBBR(RTDETRDetectionLoss):
    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, bbox_gain=6.0, **kw):
        if loss_gain is None:
            loss_gain = {'class': 1, 'bbox': float(bbox_gain), 'giou': 2,
                         'no_object': 0.1, 'mask': 1, 'dice': 1}
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)


class RTDETRDetectionLossXVFLA(RTDETRDetectionLoss):
    """负样本注意力权重系数 alpha: 0.75(基线)->0.5(更重视难负样本)。"""

    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, neg_alpha=0.5, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)
        self.neg_alpha = float(neg_alpha)
        self.fl = FocalLoss()

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        name_class = f'loss_class{postfix}'
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64,
                              device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        label = one_hot.float()
        gt_scores = gt_scores.view(bs, nq, 1) * label
        weight = self.neg_alpha * pred_scores.sigmoid().pow(2.0) * (1 - label) + \
            gt_scores * label
        loss_cls = (F.binary_cross_entropy_with_logits(
            pred_scores.float(), gt_scores.float(), reduction='none') *
            weight).mean(1).sum()
        if num_gts == 0:
            loss_cls = self.fl(pred_scores, label)
        else:
            loss_cls /= max(num_gts, 1) / nq
        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}
