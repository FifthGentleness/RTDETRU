# X055/X056 损失深水区 (批次6, [C]).
# X055 L-NegG3: VFL负样本gamma 2->3(更强难负样本聚焦; 区别于X029的alpha)
# X056 L-LS01: 正样本质量目标label smoothing q -> q(1-e)+e/nc

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.loss import FocalLoss

__all__ = ('RTDETRDetectionLossXNegG', 'RTDETRDetectionLossXLS')


class RTDETRDetectionLossXNegG(RTDETRDetectionLoss):
    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, alpha=0.75, neg_gamma=3.0, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)
        self.alpha = float(alpha)
        self.neg_gamma = float(neg_gamma)
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
        weight = self.alpha * pred_scores.sigmoid().pow(self.neg_gamma) * (1 - label) \
            + gt_scores * label
        loss_cls = (F.binary_cross_entropy_with_logits(
            pred_scores.float(), gt_scores.float(), reduction='none') *
            weight).mean(1).sum()
        if num_gts == 0:
            loss_cls = self.fl(pred_scores, label)
        else:
            loss_cls /= max(num_gts, 1) / nq
        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}


class RTDETRDetectionLossXLS(RTDETRDetectionLoss):
    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, smooth_eps=0.1, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)
        self.eps = float(smooth_eps)

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        name_class = f'loss_class{postfix}'
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64,
                              device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        label = one_hot.float()
        q = gt_scores.view(bs, nq, 1) * label
        q = q * (1.0 - self.eps) + self.eps / self.nc * label   # 平滑正样本质量
        weight = 0.75 * pred_scores.sigmoid().pow(2.0) * (1 - label) + q * label
        loss_cls = (F.binary_cross_entropy_with_logits(
            pred_scores.float(), q.float(), reduction='none') *
            weight).mean(1).sum()
        if num_gts == 0:
            from ultralytics.utils.loss import FocalLoss
            loss_cls = FocalLoss()(pred_scores, label)
        else:
            loss_cls /= max(num_gts, 1) / nq
        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}
