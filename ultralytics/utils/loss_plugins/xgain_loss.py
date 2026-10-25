# X037/X038: 损失增益网格探针 (批次4, [C]).
# X037 L-CLS15: class增益1->1.5 / X038 L-GIOU15: giou增益2->1.5
# 单一类参数化: loss_params任一增益可调, 默认值=基线。

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('RTDETRDetectionLossXGain',)


class RTDETRDetectionLossXGain(RTDETRDetectionLoss):
    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, cls_gain=1.0, bbox_gain=5.0, giou_gain=2.0, **kw):
        if loss_gain is None:
            loss_gain = {'class': float(cls_gain), 'bbox': float(bbox_gain),
                         
