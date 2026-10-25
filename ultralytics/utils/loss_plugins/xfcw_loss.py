# X004 L-FCW: 类频校准加权VFL (损失, [A]锚点已核实).
# 灵感: Fractal Calibration for Long-Tailed Object Detection (FRACAL),
#        CVPR 2025 (Alexandridis et al.; CVPR poster 33914 / OpenReview
#        cs8dm8MgOT) —— 长尾检测的逐类校准重加权思想。
# 机制: VFL正样本权重乘逐类校准因子 w_c (由loss_params传入, 建议
#        w_c = (f_max/f_c)^0.5, f_c=类频), 匹配代价class项同步加权。
#        w≡1时逐位等价基线。直击诊断6(bicycle/awning-tricycle最差)。

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.loss import FocalLoss

__all__ = ('RTDETRDetectionLossXFCW',)


class RTDETRDetectionLossXFCW(RTDETRDetectionLoss):
    """类频校准加权: loss_params['weights']=[nc个浮点], 默认全1=基线。"""

    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, weights=None, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)
        if weights is None:
            weights = [1.0] * nc
        assert len(weights) == nc, f'weights长度需为nc={nc}'
        self.register_buffer('cw', torch.tensor(weights, dtype=torch.float32))
        self.fl = FocalLoss()

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts,
                        postfix=''):
        name_class = f'loss_class{postfix}'
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64,
                              device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        label = one_hot.float()
        gt_scores = gt_scores.view(bs, nq, 1) * label
        w = self.cw.view(1, 1, -1)                        # 逐类校准因子
        weight = 0.75 * pred_scores.sigmoid().pow(2.0) * (1 - label) + \
            gt_scores * w * label                         # 正样本乘w_c
        loss_cls = (F.binary_cross_entropy_with_logits(
            pred_scores.float(), gt_scores.float(), reduction='none') *
            weight).mean(1).sum()
        if num_gts == 0:
            loss_cls = self.fl(pred_scores, label)
        else:
            loss_cls /= max(num_gts, 1) / nq
        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}
