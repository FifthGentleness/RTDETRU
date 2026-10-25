# X008 L-CAS: 分类损失晚期退火 (损失, [C]诊断驱动: val/cls从~150轮回升).
# 机制: cls损失权重随训练推进退火 1.0 -> floor(默认0.5), 以内部调用计数
#        近似epoch(T0=150轮起, T1=300轮止, 线性)。giou/l1不动。
#        T0之前逐位等价基线; 推理零开销。参数: t0_ep/t1_ep/floor/ipe
#        (ipe=每epoch迭代数, VisDrone 6471图/batch4 ≈1618)。

import torch

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('RTDETRDetectionLossXCAS',)


class RTDETRDetectionLossXCAS(RTDETRDetectionLoss):
    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, t0_ep=150, t1_ep=300, floor=0.5, ipe=1618,
                 **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)
        self.t0 = int(t0_ep * ipe)
        self.t1 = int(t1_ep * ipe)
        self.floor = float(floor)
        self.calls = 0

    def _factor(self):
        if self.calls <= self.t0:
            f = 1.0
        elif self.calls >= self.t1:
            f = self.floor
        else:
            r = (self.calls - self.t0) / max(self.t1 - self.t0, 1)
            f = 1.0 - r * (1.0 - self.floor)
        return f

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        self.calls += 1
        out = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        f = self._factor()
        for k in list(out.keys()):
            if k.startswith('loss_class'):
                out[k] = out[k] * f
        return out
